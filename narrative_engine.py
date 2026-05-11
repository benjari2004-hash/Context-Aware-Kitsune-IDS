# narrative_engine.py
# FIX 1: _ATTACK_INTROS extended with Mirai-specific attack types:
#   "Mirai C2 Communication", "ICMP Flood", "Brute Force / Telnet".
# FIX 2: _ENDPOINT_LABELS extended with new ports from context_features.py
#   (/telnet, /dns, /mirai_c2, /mirai_load) so narratives are accurate.
# FIX 3: proto field added to observation sentence (S1) so the narrative
#   correctly states TCP/UDP/ICMP rather than always implying HTTP/web.
# FIX 4: generate() guard retained — returns "" for non-ANOMALY labels.
#   No output is produced for NORMAL or TRAIN packets.

from config import EXPLAINABILITY_CONFIG

_ATTACK_INTROS = {
    "Brute Force":              "repeated authentication attempts",
    "DoS / Flood":              "a high-rate flood of requests",
    "ICMP Flood":               "a high-rate ICMP flood",
    "Port Scan":                "low-frequency port probing across multiple endpoints",
    "C2 Beacon":                "small uniform packets consistent with C2 beaconing",
    "Mirai C2 Communication":   "traffic to a known Mirai C2 or loader port",
    "Data Exfiltration":        "large or mixed-size outbound payloads at low frequency",
    "Encrypted Tunnel":         "sustained anomalous encrypted traffic",
    "Reconnaissance":           "sparse low-rate probing to sensitive endpoints",
    "Credential Stuffing":      "repeated credential attempts over HTTPS",
    "Stealth Intrusion":        "low-frequency traffic with high autoencoder anomaly score",
    "Network Intrusion":        "a strong multi-signal intrusion pattern",
    "High-Risk Anomaly":        "a high-risk anomalous traffic burst",
    "Anomalous Web Traffic":    "abnormal HTTP traffic with behavioral deviation",
    "Suspicious Activity":      "multi-signal anomaly without a specific pattern match",
    "Unclassified Anomaly":     "an anomaly that exceeded the detection threshold",
}

_ENDPOINT_LABELS = {
    "/web":        "HTTP web endpoint (port 80)",
    "/secure":     "HTTPS encrypted endpoint (port 443)",
    "/ssh":        "SSH management endpoint (port 22)",
    "/telnet":     "Telnet endpoint (port 23) — Mirai infection vector",
    "/dns":        "DNS endpoint (port 53) — amplification / C2 channel",
    "/mirai_c2":   "Mirai C2 endpoint (port 8280)",
    "/mirai_load": "Mirai loader endpoint (port 10240)",
    "/other":      "unclassified endpoint (port 8080+)",
}

_ACTION_MAP = {
    "HIGH":   "⚠ IMMEDIATE ACTION: isolate host, review auth logs, block if confirmed.",
    "MEDIUM": "🔍 INVESTIGATE: review recent sessions and correlate with other alerts.",
    "LOW":    "📋 MONITOR: flag for follow-up; suppress if pattern does not persist.",
}


def generate(
    entity_id,
    timestamp,
    attack_type,
    ara_result,
    bfde_result,
    tcce_result,
    rsdr_result,
    ctx,
    attack_reason="",
):
    """
    Generate a human-readable SOC narrative for a confirmed anomaly.

    Returns empty string immediately for non-anomaly / training contexts.
    All fields sourced from real packet context — no synthetic values.

    Returns
    -------
    str — formatted narrative box, or "" if not applicable
    """
    # Guard: no narrative for normal or training traffic
    if not attack_type or attack_type in ("Normal", "Training", ""):
        return ""

    endpoint   = ctx.get("endpoint",    "/web")
    freq       = ctx.get("freq",        0.0)
    length     = ctx.get("length",      500)
    jitter     = ctx.get("jitter",      0.0)
    unique_ep  = ctx.get("unique_ep",   1)
    proto      = ctx.get("proto",       "OTHER")   # FIX: real proto from packet
    confidence = rsdr_result.get("confidence",    "LOW")
    sigs       = rsdr_result.get("signals_fired", 0)

    ep_label = _ENDPOINT_LABELS.get(endpoint, endpoint)
    intro    = _ATTACK_INTROS.get(attack_type, "anomalous network activity")

    # S1: Observation — includes protocol so narrative is accurate for
    #     ICMP floods, UDP amplification, TCP SYN floods, etc.
    s1 = (
        f"At t={timestamp:.1f}s, entity {entity_id} exhibited {intro} "
        f"on {ep_label} "
        f"[proto={proto}, freq={freq:.0f}, length={length}B, "
        f"jitter={jitter:.2f}, unique_ep={unique_ep:.0f}]. "
        f"Classification reason: {attack_reason}."
    )

    # S2: Behavioral baseline deviation
    bfde_devs = bfde_result.get("deviations", [])
    ep_desc   = bfde_result.get("endpoint_desc", "")
    if bfde_devs:
        top = bfde_devs[0]
        s2 = (
            f"Behavioral baseline: '{top['dim']}' is {top['z']:+.2f}σ from "
            f"{entity_id}'s norm (current={top['value']}, "
            f"baseline≈{top['baseline']}, severity={top['severity']}). "
            f"Endpoint: {ep_desc}."
        )
    else:
        s2 = f"Behavioral baseline: insufficient history. Endpoint: {ep_desc}."

    # S3: Temporal escalation chain
    chain    = tcce_result.get("chain", [])
    duration = tcce_result.get("duration_seconds", 0)
    if len(chain) >= 2:
        first = chain[0]
        last  = chain[-1]
        s3 = (
            f"Temporal escalation over {duration:.0f}s: "
            f"[{first['description']}] → [{last['description']}]."
        )
    elif chain:
        s3 = f"Single prior event: {chain[0]['description']}."
    else:
        s3 = "No significant temporal escalation before this alert."

    # S4: ARA model anatomy
    ara_top = ara_result.get("top_features", [])
    if ara_top and ara_result.get("valid", False):
        tf = ara_top[0]
        s4 = (
            f"Model anatomy: '{tf['name']}' drives {tf['contribution_pct']}% "
            f"of RMSE (z={tf['z_score']:+.2f}, "
            f"observed={tf['value']}, expected≈{tf['expected']})."
        )
    else:
        s4 = (
            "Model anatomy: ARA inactive — real Kitsune feature vector required. "
            "Connect FeatureExtractor.get_next_vector() output to enable."
        )

    # S5: Risk audit summary
    cf = rsdr_result.get("counterfactual", {})
    s5 = (
        f"Risk audit: {sigs}/5 signals aligned (confidence={confidence}). "
        f"Counterfactual — {cf.get('condition', 'N/A')}: "
        f"risk≈{cf.get('cf_risk', '?')} → {cf.get('cf_label', '?')}."
    )

    # S6: Recommended SOC action
    s6 = _ACTION_MAP.get(confidence, "Review alert manually.")

    # ── Box renderer ──────────────────────────────────────────────
    w   = 64
    bar = "─" * w

    def _wrap(sentence, width):
        words = sentence.split()
        lines = []
        line  = ""
        for word in words:
            if len(line) + len(word) + 1 > width:
                lines.append(line)
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            lines.append(line)
        return lines

    box_lines = [
        f"┌{bar}┐",
        f"│ NARRATIVE ALERT REPORT{' ' * (w - 23)}│",
        f"│ Attack: {attack_type:<28} Confidence: {confidence:<7}│",
        f"├{bar}┤",
        f"│ [OBSERVATION]{' ' * (w - 14)}│",
    ]

    for sentence in [s1, s2, s3, s4, s5]:
        for wrapped in _wrap(sentence, w - 2):
            box_lines.append(f"│ {wrapped:<{w-2}} │")
        box_lines.append(f"│{' ' * w}│")

    box_lines += [
        f"├{bar}┤",
        f"│ [RECOMMENDED ACTION]{' ' * (w - 21)}│",
        f"│ {s6:<{w-2}} │",
        f"└{bar}┘",
    ]

    return "\n".join(box_lines)