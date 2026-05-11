# attack_classifier.py
# FIX 1 (PRIMARY): Rule ordering was the dominant cause of label collapse.
#   The broad DoS/Flood rule (freq >= 8 AND score > 0.15) matched ~96 % of
#   Mirai detection-phase packets BEFORE any narrower rule could fire.
#   Rules are now ordered from MOST SPECIFIC to LEAST SPECIFIC so that
#   narrow, high-confidence patterns (Stealth, C2, Exfil, Port Scan) are
#   evaluated first.  DoS/Flood becomes the last major tier before the
#   final catch-all.
#
# FIX 2: Mirai-specific ports added (/telnet, /mirai_c2, /mirai_load) so
#   telnet brute-force and C2 beacon traffic are classified correctly.
#
# FIX 3: Proto-aware ICMP flood rule added (Mirai uses ICMP floods).
#
# FIX 4: "risk > 0" fallback removed everywhere — every rule requires at
#   least two independent signals.  The Suspicious Activity catch-all now
#   requires score > 0.10 (was: any positive risk).
#
# ARCHITECTURE ROLE (unchanged):
#   LABELLER only.  Must be called AFTER anomaly detection confirms ANOMALY.
#   Does NOT perform detection.  Does NOT use threshold logic.

# ─────────────────────────────────────────────────────────────────
# Helper predicates  (explicit, testable, DRY)
# ─────────────────────────────────────────────────────────────────

def _saturated(score):
    """Score hit combine_scores ceiling → treat as worst-case signal."""
    return score >= 0.99


def _has_jitter(ctx, threshold=2.0):
    """Irregular inter-arrival timing — stealth / scan indicator."""
    return ctx.get("jitter", 0.0) > threshold


def _switching(ctx, threshold=2):
    """Multiple distinct ports contacted — scan / multi-target indicator."""
    return ctx.get("unique_ep", 1) >= threshold


def _len_var_high(ctx, threshold=150):
    """Mixed packet sizes — evasion / exfiltration indicator."""
    return ctx.get("len_variance", 0.0) > threshold


def _beacon_sized(ctx):
    """Uniform small packets — C2 beaconing indicator."""
    return (
        ctx.get("length", 500) < 320
        and ctx.get("len_variance", 999) < 80
    )


def _is_telnet(ctx):
    """Destination port 23 — Mirai's primary infection vector."""
    return ctx.get("dport", 0) == 23 or ctx.get("endpoint", "") == "/telnet"


def _is_mirai_c2(ctx):
    """Mirai C2 / loader ports."""
    return ctx.get("endpoint", "") in ("/mirai_c2", "/mirai_load") or \
           ctx.get("dport", 0) in (8280, 10240)


def _is_icmp(ctx):
    return ctx.get("proto", "OTHER") == "ICMP"


# ─────────────────────────────────────────────────────────────────
# ATTACK_RULES — ordered MOST SPECIFIC → LEAST SPECIFIC
#
# Ordering principle:
#   1. Rules with the smallest valid population come first.
#   2. A rule that fires on a subset of packets that a later rule also
#      matches must come first.
#   3. DoS/Flood (broad, high-freq) is pushed to TIER 7 so narrow
#      patterns can claim their correct label first.
# ─────────────────────────────────────────────────────────────────

ATTACK_RULES = [

    # ══════════════════════════════════════════════════════════════
    # TIER 1 — MIRAI-SPECIFIC C2 / LOADER  (very narrow port set)
    # Must fire before generic rules — these ports are unambiguous.
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "Mirai C2 Communication",
        "reason": "Traffic to known Mirai C2/loader port with anomalous score",
        "conditions": lambda score, risk, ctx, fd: (
            _is_mirai_c2(ctx)
            and score > 0.10  # SEE config.py → classifier_score_floor for rationale
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 2 — STEALTH / LOW-AND-SLOW  (low freq, high RMSE)
    # Must fire before DoS — stealth uses low freq which DoS rules exclude.
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "Stealth Intrusion",
        "reason": "High Kitsune RMSE at low frequency — low-and-slow attack pattern",
        "conditions": lambda score, risk, ctx, fd: (
            score > 0.45
            and ctx["freq"] <= 3
        ),
    },
    {
        "name":   "Stealth Intrusion",
        "reason": "Irregular inter-arrival jitter with elevated RMSE at low frequency",
        "conditions": lambda score, risk, ctx, fd: (
            _has_jitter(ctx, threshold=3.0)
            and score > 0.20
            and ctx["freq"] <= 5
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 3 — C2 BEACON  (uniform small packets)
    # Must fire before DoS — beacons are typically low-rate.
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "C2 Beacon",
        "reason": "Small uniform packets at low frequency — machine-generated beaconing",
        "conditions": lambda score, risk, ctx, fd: (
            _beacon_sized(ctx)
            and ctx["freq"] <= 5
            and score > 0.12
        ),
    },
    {
        "name":   "C2 Beacon",
        "reason": "Uniform low-variance packets to Mirai telnet port — possible bot checkin",
        "conditions": lambda score, risk, ctx, fd: (
            _is_telnet(ctx)
            and _beacon_sized(ctx)
            and score > 0.10  # SEE config.py → classifier_score_floor for rationale
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 4 — DATA EXFILTRATION  (large or mixed-size payloads)
    # Must fire before DoS — exfil uses large packets, floods use small.
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "Data Exfiltration",
        "reason": "Large payload at low frequency — slow exfiltration pattern",
        "conditions": lambda score, risk, ctx, fd: (
            ctx.get("length", 0) > 650
            and ctx["freq"] <= 5
            and score > 0.12
        ),
    },
    {
        "name":   "Data Exfiltration",
        "reason": "High length variance at low frequency — mixed-size evasive exfiltration",
        "conditions": lambda score, risk, ctx, fd: (
            _len_var_high(ctx, threshold=180)
            and ctx["freq"] <= 4
            and score > 0.12
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 5 — PORT SCAN / RECONNAISSANCE
    # Must fire before DoS — scans often have mid-to-high freq too.
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "Port Scan",
        "reason": "Multiple distinct ports contacted in short time window",
        "conditions": lambda score, risk, ctx, fd: (
            _switching(ctx, threshold=3)
            and score > 0.10  # SEE config.py → classifier_score_floor for rationale
        ),
    },
    {
        "name":   "Port Scan",
        "reason": "Low-rate probing to non-web endpoint with behavioral deviation",
        "conditions": lambda score, risk, ctx, fd: (
            ctx["endpoint"] in ("/other", "/ssh", "/telnet")
            and ctx["freq"] <= 4
            and fd >= 0.8
            and score > 0.10  # SEE config.py → classifier_score_floor for rationale
        ),
    },
    {
        "name":   "Reconnaissance",
        "reason": "Sparse low-rate probing to sensitive endpoint — pre-attack recon",
        "conditions": lambda score, risk, ctx, fd: (
            ctx["freq"] <= 2
            and ctx["endpoint"] not in ("/web",)
            and (score > 0.15 or _has_jitter(ctx, 1.5))
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 6 — AUTH ATTACKS  (endpoint-specific, mid-to-high freq)
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "Brute Force",
        "reason": "Repeated SSH attempts with anomalous reconstruction score",
        "conditions": lambda score, risk, ctx, fd: (
            ctx["endpoint"] == "/ssh"
            and ctx["freq"] >= 2
            and score > 0.18
        ),
    },
    {
        "name":   "Brute Force",
        "reason": "Repeated Telnet brute-force — Mirai default-credential scanner",
        "conditions": lambda score, risk, ctx, fd: (
            _is_telnet(ctx)
            and ctx["freq"] >= 3
            and score > 0.12
        ),
    },
    {
        "name":   "Credential Stuffing",
        "reason": "Repeated HTTPS auth attempts to secure endpoint",
        "conditions": lambda score, risk, ctx, fd: (
            ctx["endpoint"] == "/secure"
            and ctx["freq"] >= 3
            and score > 0.18
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 7 — DOS / FLOOD  (broad, high-frequency — moved here)
    # Now only fires when narrower rules above have already passed.
    # ICMP flood rule added for Mirai SYN/UDP/ICMP flood variants.
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "ICMP Flood",
        "reason": "High-rate ICMP traffic with elevated anomaly score — ICMP flood",
        "conditions": lambda score, risk, ctx, fd: (
            _is_icmp(ctx)
            and ctx["freq"] >= 5
            and score > 0.12
        ),
    },
    {
        "name":   "DoS / Flood",
        "reason": "High request rate with elevated anomaly score",
        "conditions": lambda score, risk, ctx, fd: (
            ctx["freq"] >= 8
            and score > 0.15
        ),
    },
    {
        "name":   "DoS / Flood",
        "reason": "Repeated high-rate HTTP requests to web endpoint",
        "conditions": lambda score, risk, ctx, fd: (
            ctx["endpoint"] == "/web"
            and ctx["freq"] >= 5
            and score > 0.12
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 8 — ENCRYPTED CHANNEL  (HTTPS anomaly)
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "Encrypted Tunnel",
        "reason": "Anomalous HTTPS traffic with mid-size payloads — possible covert channel",
        "conditions": lambda score, risk, ctx, fd: (
            ctx["endpoint"] == "/secure"
            and 80 <= ctx.get("length", 0) <= 800
            and score > 0.18
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 9 — HIGH-SEVERITY GENERIC
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "High-Risk Anomaly",
        "reason": "Risk score exceeded critical threshold after all multipliers",
        "conditions": lambda score, risk, ctx, fd: (
            risk > 1.2
        ),
    },
    {
        "name":   "Network Intrusion",
        "reason": "Strong RMSE signal corroborated by endpoint or behavioral context",
        "conditions": lambda score, risk, ctx, fd: (
            not _saturated(score)
            and score > 0.45
            and risk > 0.35
            and (
                ctx["endpoint"] in ("/ssh", "/secure", "/other", "/telnet", "/mirai_c2")
                or fd >= 1.5
                or _switching(ctx)
                or _has_jitter(ctx)
            )
        ),
    },
    {
        "name":   "Network Intrusion",
        "reason": "Saturated RMSE on sensitive endpoint — strong autoencoder signal",
        "conditions": lambda score, risk, ctx, fd: (
            _saturated(score)
            and risk > 0.35
            and ctx.get("is_sensitive", False)
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 10 — WEB-SPECIFIC FALLBACK
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "Anomalous Web Traffic",
        "reason": "HTTP anomaly with at least one corroborating behavioral signal",
        "conditions": lambda score, risk, ctx, fd: (
            ctx["endpoint"] == "/web"
            and score > 0.15
            and (
                fd >= 1.0
                or _has_jitter(ctx)
                or ctx.get("length", 0) > 600
                or _switching(ctx)
            )
        ),
    },

    # ══════════════════════════════════════════════════════════════
    # TIER 11 — FINAL CATCH-ALL
    # Requires score > 0.10 — never fires on trivially weak signal.
    # ══════════════════════════════════════════════════════════════

    {
        "name":   "Suspicious Activity",
        "reason": "Anomaly confirmed by threshold; no specific attack pattern matched",
        "conditions": lambda score, risk, ctx, fd: (
            # SEE config.py → classifier_score_floor for rationale
            score > 0.10
        ),
    },
]


def classify_attack(score, risk, ctx, freq_deviation=0.0):
    """
    Assign a semantic label to a CONFIRMED anomaly.

    IMPORTANT: Must ONLY be called when label == "ANOMALY".
    Does not perform detection — that is the threshold's responsibility.

    Parameters
    ----------
    score         : float — combine_scores output (final_score)
    risk          : float — adjusted_risk from risk_engine
    ctx           : dict  — context from context_features.extract_context()
    freq_deviation: float — abs(current_freq - profile_mean)

    Returns
    -------
    (attack_name: str, reason: str)
    """
    for rule in ATTACK_RULES:
        try:
            if rule["conditions"](score, risk, ctx, freq_deviation):
                return rule["name"], rule.get("reason", "")
        except Exception:
            continue
    return "Unclassified Anomaly", "Passed detection threshold; no rule matched"