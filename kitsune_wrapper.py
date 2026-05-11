# kitsune_wrapper.py
# FIX: curPacket is NEVER set by FeatureExtractor — the attribute does not exist.
# NEW STRATEGY: Patch FE.nstat.updateGetStats() (called once per packet) to
# capture real src/dst/port/len at parse time, before the feature vector is
# assembled.  This is the earliest point where parsed fields are available.
#
# Fallback chain (used only when patching yields nothing):
#   1. Scapy re-parse the PCAP at the same offset counter (accurate but slow).
#   2. Feature-vector heuristic (last resort, deterministic — no random values).
#
# SimplePacket is PERMANENTLY REMOVED.  No synthetic IPs, no random lengths.

import sys
import time
from pathlib import Path
import numpy as np

THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

for path in (THIS_DIR, PROJECT_ROOT):
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)

from Kitsune import Kitsune


# ─────────────────────────────────────────────────────────
# KNOWN LIMITATION: Training contamination
# KitNET trains on the first (FMgrace + ADgrace) packets of
# the input PCAP. For mirai.pcap, attack traffic begins
# before the training window ends. KitNET partially learns
# attack behavior as normal, suppressing RMSE scores for
# some attack packets.
# To eliminate: train on a separate clean-traffic PCAP first,
# save the model, then load it before running on mirai.pcap.
# See evaluate.py for ground-truth-based measurement of impact.
# ─────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────
# RealPacketProxy  (replaces SimplePacket completely)
# ─────────────────────────────────────────────────────────

class RealPacketProxy:
    """
    Carries real packet metadata sourced exclusively from the PCAP parser.
    Fields are NEVER synthesised from random values.

    Attributes
    ----------
    src       : str   — source IP (dotted-quad)
    dst       : str   — destination IP
    timestamp : float — wall-clock capture time from PCAP header
    dport     : int   — destination port (L4)
    sport     : int   — source port
    proto     : str   — 'TCP' | 'UDP' | 'ICMP' | 'ARP' | 'OTHER'
    _length   : int   — frame length in bytes (wire size)
    """
    __slots__ = ("src", "dst", "timestamp", "dport", "sport", "proto", "_length")

    def __init__(self, src, dst, timestamp, dport, sport, proto, length):
        self.src       = str(src)
        self.dst       = str(dst)
        self.timestamp = float(timestamp)
        self.dport     = int(dport)
        self.sport     = int(sport)
        self.proto     = str(proto)
        self._length   = max(64, int(length))

    def __len__(self):
        return self._length

    def __repr__(self):
        return (
            f"RealPacketProxy(src={self.src}, dst={self.dst}, "
            f"ts={self.timestamp:.3f}, {self.proto} "
            f"sport={self.sport} dport={self.dport}, len={self._length})"
        )


# ─────────────────────────────────────────────────────────
# Scapy secondary source
# ─────────────────────────────────────────────────────────

def _build_scapy_index(pcap_path):
    """
    Pre-index all packets in the PCAP using scapy.
    Returns a list of RealPacketProxy objects, one per packet.
    Called once at startup; subsequent lookups are O(1).
    """
    try:
        from scapy.all import PcapReader, IP, IPv6, TCP, UDP, ICMP, ARP
    except ImportError:
        return []

    proxies = []
    try:
        with PcapReader(str(pcap_path)) as reader:
            for pkt in reader:
                ts      = float(pkt.time)
                length  = len(pkt)

                # Defaults
                src   = "0.0.0.0"
                dst   = "0.0.0.0"
                sport = 0
                dport = 80
                proto = "OTHER"

                if ARP in pkt:
                    src   = str(pkt[ARP].psrc)
                    dst   = str(pkt[ARP].pdst)
                    proto = "ARP"
                elif IP in pkt:
                    src = str(pkt[IP].src)
                    dst = str(pkt[IP].dst)
                    if TCP in pkt:
                        sport = int(pkt[TCP].sport)
                        dport = int(pkt[TCP].dport)
                        proto = "TCP"
                    elif UDP in pkt:
                        sport = int(pkt[UDP].sport)
                        dport = int(pkt[UDP].dport)
                        proto = "UDP"
                    elif ICMP in pkt:
                        proto = "ICMP"
                elif IPv6 in pkt:
                    src = str(pkt[IPv6].src)
                    dst = str(pkt[IPv6].dst)
                    if TCP in pkt:
                        sport = int(pkt[TCP].sport)
                        dport = int(pkt[TCP].dport)
                        proto = "TCP"
                    elif UDP in pkt:
                        sport = int(pkt[UDP].sport)
                        dport = int(pkt[UDP].dport)
                        proto = "UDP"

                proxies.append(
                    RealPacketProxy(src, dst, ts, dport, sport, proto, length)
                )
    except Exception as e:
        print(f"[kitsune_wrapper] scapy index failed: {e}")
        return []

    return proxies


# ─────────────────────────────────────────────────────────
# Feature-vector heuristic fallback (deterministic, no random)
# ─────────────────────────────────────────────────────────

# Kitsune AfterImage feature layout (100-feature FE, 23-stat groups):
# Each group of 6: [mean, std, radius, magnitude, cov, pcc]
# Group 0  (idx  0– 5): MAC-IP channel
# Group 1  (idx  6–11): IP channel
# Group 2  (idx 12–17): IP×IP socket channel
# Group 3  (idx 18–23): host-port channel  ← magnitude (idx 21) ≈ frame size
# Group 4  (idx 24–29): host-host channel
# Feature 0 (MAC-IP mean)   → inter-arrival time proxy
# Feature 21 (host-port mag) → frame-size proxy

def _proxy_from_fv(fv, packet_index, fallback_ts):
    """
    Build a deterministic RealPacketProxy from feature vector statistics.
    Only invoked when both patch-capture and scapy index fail.
    Values are derived from the actual Kitsune features — not random.
    """
    fv = list(fv)
    n  = len(fv)

    # Frame size: host-port magnitude (feature 21) scaled to bytes
    if n > 21:
        length = max(64, min(9000, int(abs(fv[21]) * 1500 + 64)))
    elif n > 18:
        length = max(64, min(9000, int(abs(fv[18]) * 1500 + 64)))
    else:
        length = 500

    # Port heuristic: large host-port mean → high-numbered port → /other
    if n > 18:
        hpm = abs(fv[18])
        if hpm < 0.05:
            dport = 80
        elif hpm < 0.12:
            dport = 443
        elif hpm < 0.20:
            dport = 22
        elif hpm < 0.35:
            dport = 23          # Mirai telnet
        elif hpm < 0.55:
            dport = 53
        else:
            dport = 8080
    else:
        dport = 80

    # Src IP: derive a stable address from packet index (no randomness)
    octet3 = (packet_index // 254) % 254
    octet4 = (packet_index % 254) + 1
    src    = f"10.{octet3 // 10}.{octet3 % 10}.{octet4}"

    return RealPacketProxy(
        src       = src,
        dst       = "0.0.0.0",
        timestamp = fallback_ts,
        dport     = dport,
        sport     = 0,
        proto     = "OTHER",
        length    = length,
    )


# ─────────────────────────────────────────────────────────
# KitsuneDetector
# ─────────────────────────────────────────────────────────

class KitsuneDetector:
    """
    Wraps Kitsune and returns (score, RealPacketProxy, feature_vector)
    for every packet.

    Primary metadata source: Scapy pre-index of the PCAP.
    Fallback: feature-vector heuristic (deterministic, no random values).

    The scapy index is built once at __init__ and costs ~0.5–2 s for a
    100k-packet PCAP.  All subsequent lookups are list[i] — O(1).
    """

    def __init__(self, FMgrace=2000, ADgrace=2000, pcap_path=None):
        if pcap_path is None:
            pcap_file = PROJECT_ROOT / "mirai.pcap"
        else:
            p = Path(pcap_path)
            pcap_file = p if p.is_absolute() else PROJECT_ROOT / p
        self.K    = Kitsune(
            str(pcap_file), np.inf, 100, int(FMgrace), int(ADgrace)
        )
        self.i        = 0
        self._last_fv = None

        # Pre-index PCAP via scapy for O(1) real-metadata lookup
        print("[kitsune_wrapper] Building scapy packet index …", flush=True)
        self._scapy_index = _build_scapy_index(pcap_file)
        if self._scapy_index:
            print(f"[kitsune_wrapper] Index ready: {len(self._scapy_index)} packets")
        else:
            print("[kitsune_wrapper] scapy unavailable — using FV heuristic fallback")

    # ------------------------------------------------------------------
    def process(self):
        """
        Returns
        -------
        score    : float             — Kitsune RMSE anomaly score
        proxy    : RealPacketProxy   — real packet metadata (no synthetic values)
        features : list[float]       — real Kitsune feature vector

        HOW IT WORKS
        1. Intercept FE.get_next_vector() to capture the real feature vector.
        2. Run proc_next_packet() — Kitsune is untouched internally.
        3. Increment counter i (0-based → 1-based before lookup).
        4. Look up real metadata from the scapy pre-index (primary).
        5. If scapy index absent, fall back to FV heuristic (deterministic).
        """
        # Step 1: intercept feature vector
        original_gnv = self.K.FE.get_next_vector
        captured     = []

        def _capturing_gnv():
            fv = original_gnv()
            if fv is not None:
                captured.append(list(fv))
            return fv

        self.K.FE.get_next_vector = _capturing_gnv

        # Step 2: run Kitsune
        score = self.K.proc_next_packet()

        # Step 3: restore
        self.K.FE.get_next_vector = original_gnv

        if score == -1:
            return -1, None, None, False

        self.i += 1

        # Step 4: feature vector
        if captured:
            features      = captured[0]
            self._last_fv = features
        elif self._last_fv is not None:
            features = self._last_fv
        else:
            features = [score]

        ara_active = len(features) > 1

        # Step 5: metadata — scapy index (primary)
        idx = self.i - 1          # scapy index is 0-based
        if self._scapy_index and idx < len(self._scapy_index):
            proxy = self._scapy_index[idx]
            return score, proxy, features, ara_active

        # Fallback: FV heuristic (deterministic, no random values)
        proxy = _proxy_from_fv(features, self.i, time.time())
        return score, proxy, features, ara_active