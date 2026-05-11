# context_features.py
# FIX 1: PORT_MAP extended with Mirai-relevant ports (23/telnet, 53/dns,
#         8080/alt-http, 8280/Mirai C2, 10240/Mirai loader).
#         Previously only 80/443/22 were named — everything else collapsed
#         into "/other", hiding port-specific semantics in the Mirai dataset.
# FIX 2: proto field from RealPacketProxy is propagated into ctx so
#         classifiers can distinguish TCP/UDP/ICMP attacks.
# FIX 3: No changes to the stateful tracking logic (_ip_times, _ip_ports,
#         _ip_lengths) — those are correct.  The only sources of context
#         are packet.src, packet.dport, len(packet), and packet.timestamp.
#         All of these come from RealPacketProxy which is built from scapy
#         or the FV heuristic — never synthetic random values.

import math
from collections import defaultdict
import numpy as np
import config

_ip_times   = defaultdict(list)
_ip_ports   = defaultdict(list)
_ip_lengths = defaultdict(list)

_RATE_WINDOW = 30   # seconds

# Port→endpoint mapping is loaded from the active profile (profiles/mirai.yaml).
# SEE config.py → PROFILE_PORT_ENDPOINTS for the loaded data.
#
# _ENDPOINT_INDICES maps endpoint name → feature-vector numeric ID.
# These IDs are NOT profile-dependent: changing them shifts the feature
# distribution seen by KitNET and would require a full re-training run.
_ENDPOINT_INDICES = {
    "/web":        0,
    "/secure":     1,
    "/ssh":        2,
    "/other":      3,
    "/telnet":     5,
    "/dns":        6,
    "/mirai_c2":   7,
    "/mirai_load": 8,
}
DEFAULT = (4, "/other")

_PORT_MAP = None  # built lazily on first call; rebuilt if init_profile() re-runs


def _get_port_map():
    global _PORT_MAP
    if _PORT_MAP is None:
        _PORT_MAP = {
            port: (_ENDPOINT_INDICES.get(ep, 4), ep)
            for port, ep in config.PROFILE_PORT_ENDPOINTS.items()
        }
    return _PORT_MAP

# Endpoints that are intrinsically high-value (amplify risk)
SENSITIVE_ENDPOINTS = {"/ssh", "/secure", "/telnet", "/mirai_c2", "/mirai_load"}


def time_feature(timestamp):
    hour = (timestamp % 86400) / 3600
    return [
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
    ]


def endpoint(packet):
    port = getattr(packet, "dport", None)
    return _get_port_map().get(port, DEFAULT)


def request_rate(packet):
    """Packets-per-window from this source IP."""
    ip    = packet.src
    t     = packet.timestamp
    times = _ip_times.setdefault(ip, [])
    times[:] = [x for x in times if t - x < _RATE_WINDOW]
    rate  = len(times)
    times.append(t)
    return float(rate)


def inter_arrival_jitter(packet):
    """
    Std-dev of inter-arrival gaps in the rate window.
    High jitter + low freq  → stealth / scan behavior.
    Low  jitter + high freq → flood / bot behavior.
    """
    ip    = packet.src
    t     = packet.timestamp
    times = _ip_times.get(ip, [])
    recent = sorted(x for x in times if t - x < _RATE_WINDOW)
    if len(recent) < 2:
        return 0.0
    gaps    = [recent[k + 1] - recent[k] for k in range(len(recent) - 1)]
    mean    = sum(gaps) / len(gaps)
    var     = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return round(math.sqrt(var), 4)


def unique_endpoints_in_window(packet):
    """
    Count of distinct destination ports seen from this IP in the window.
    High count → port-scan / recon behavior.
    """
    ip   = packet.src
    t    = packet.timestamp
    port = getattr(packet, "dport", 80)

    ports = _ip_ports.setdefault(ip, [])
    ports[:] = [(ts, p) for ts, p in ports if t - ts < _RATE_WINDOW]
    ports.append((t, port))
    return float(len(set(p for _, p in ports)))


def length_variance(packet):
    """
    Std-dev of frame lengths from this IP in the window.
    High variance  → mixed-size evasion / exfil.
    Near-zero var  → uniform beaconing.
    """
    ip  = packet.src
    t   = packet.timestamp
    ln  = len(packet)

    lengths = _ip_lengths.setdefault(ip, [])
    lengths[:] = [(ts, l) for ts, l in lengths if t - ts < _RATE_WINDOW]
    lengths.append((t, ln))

    vals = [l for _, l in lengths]
    if len(vals) < 2:
        return 0.0
    mean = sum(vals) / len(vals)
    var  = sum((v - mean) ** 2 for v in vals) / len(vals)
    return round(math.sqrt(var), 2)


def extract_context(packet, profile=None):
    """
    Build a feature vector and a context dict from a RealPacketProxy.

    All values are derived from real packet fields — no synthetic data.

    Parameters
    ----------
    packet  : RealPacketProxy
    profile : dict | None — loaded profile (from config.load_profile()).
              If None, falls back to the globally-loaded profile via
              config.PROFILE_PORT_ENDPOINTS / _get_port_map().

    Returns
    -------
    features : list[float]   — numeric feature vector for ML
    ctx      : dict          — human-readable context for risk/classify/explain
    """
    t = time_feature(packet.timestamp)

    if profile is not None:
        _pm    = profile.get("port_map", {})
        _dep   = profile.get("default_endpoint", "/other")
        _local = {
            int(p): (_ENDPOINT_INDICES.get(v["endpoint"], 4), v["endpoint"])
            for p, v in _pm.items()
        }
        _dflt      = (_ENDPOINT_INDICES.get(_dep, 4), _dep)
        port       = getattr(packet, "dport", None)
        eid, ename = _local.get(port, _dflt)
    else:
        eid, ename = endpoint(packet)
    rate             = request_rate(packet)
    jitter           = inter_arrival_jitter(packet)
    unique_ep        = unique_endpoints_in_window(packet)
    len_var          = length_variance(packet)
    pkt_len          = len(packet)
    proto            = getattr(packet, "proto", "OTHER")

    # Numeric feature vector
    features = t + [
        float(eid),
        rate,
        jitter,
        unique_ep,
        len_var,
        float(pkt_len),
    ]

    # Context dict — all values sourced from real packet fields
    ctx = {
        "src_ip":       getattr(packet, "src",  "unknown"),
        "dst_ip":       getattr(packet, "dst",  "unknown"),
        "endpoint":     ename,
        "dport":        getattr(packet, "dport", 80),
        "sport":        getattr(packet, "sport", 0),
        "proto":        proto,
        "freq":         rate,
        "length":       pkt_len,
        "jitter":       jitter,
        "unique_ep":    unique_ep,
        "len_variance": len_var,
        # Convenience flags used by risk_engine and classifier
        "is_sensitive": ename in SENSITIVE_ENDPOINTS,
    }

    return features, ctx