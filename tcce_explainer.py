# tcce_explainer.py
# Temporal Causal Chain Explanation (TCCE)
#
# FIXED: Uses real float timestamps (seconds) not packet indices.
# FIXED: Deviation signals derived from numeric z-scores only.
# RESEARCH NOTE: Causal chains map to MITRE ATT&CK multi-stage patterns.

from collections import defaultdict, deque
from config import EXPLAINABILITY_CONFIG
from utils import seconds_ago_str, deviation_label


# entity_id → deque of event dicts
_journals = defaultdict(lambda: deque())


def record_event(entity_id, timestamp, ctx, freq_z, length_z, score):
    """
    Record one event in the entity's rolling temporal journal.

    IMPORTANT: Call every packet (training + detection) BEFORE
    generating explanations, but AFTER computing deviations.

    Parameters
    ----------
    timestamp : float — real time in seconds (e.g. packet.timestamp)
    freq_z    : float — z-score of freq vs entity baseline
    length_z  : float — z-score of length vs entity baseline
    score     : float — Kitsune anomaly score
    """
    cfg    = EXPLAINABILITY_CONFIG
    window = cfg["tcce_window_seconds"]

    event = {
        "timestamp": float(timestamp),
        "endpoint":  ctx.get("endpoint", "/web"),
        "freq":      ctx.get("freq",     0.0),
        "length":    ctx.get("length",   500),
        # FIXED: store actual z-scores, not raw deviation values
        "freq_z":    round(float(freq_z),    2),
        "length_z":  round(float(length_z),  2),
        "score":     round(float(score),     4),
    }

    journal = _journals[entity_id]
    journal.append(event)

    # FIXED: prune by real time delta, not count
    cutoff = float(timestamp) - window
    while journal and journal[0]["timestamp"] < cutoff:
        journal.popleft()


def explain(entity_id, current_timestamp):
    """
    Build the causal chain leading up to the current alert.

    Returns
    -------
    dict: chain (list), duration_seconds, summary
    """
    cfg      = EXPLAINABILITY_CONFIG
    gate     = cfg["tcce_deviation_gate"]
    min_ev   = cfg["tcce_min_events"]
    max_show = cfg["narrative_max_chain_events"]

    journal = list(_journals[entity_id])

    # filter to events that had meaningful deviation OR elevated score
    significant = [
        e for e in journal
        if abs(e["freq_z"]) >= gate
        or abs(e["length_z"]) >= gate
        or e["score"] > 0.15
    ]

    if len(significant) < min_ev:
        return {
            "chain":            [],
            "duration_seconds": 0,
            "summary":          "Insufficient temporal context for causal chain.",
        }

    chain_events = significant[-max_show:]
    chain        = []

    for e in chain_events:
        ago   = seconds_ago_str(float(current_timestamp), e["timestamp"])
        flags = []
        if abs(e["freq_z"]) >= gate:
            flags.append(
                f"freq {deviation_label(e['freq_z'])} "
                f"(z={e['freq_z']:+.1f})"
            )
        if abs(e["length_z"]) >= gate:
            flags.append(
                f"length {deviation_label(e['length_z'])} "
                f"(z={e['length_z']:+.1f})"
            )
        if e["score"] > 0.3:
            flags.append(f"score={e['score']:.3f}")

        desc = (
            f"[{ago}] {e['endpoint']} | " + ", ".join(flags)
            if flags else
            f"[{ago}] {e['endpoint']} | mild deviation"
        )
        chain.append({
            "timestamp":   e["timestamp"],
            "ago":         ago,
            "endpoint":    e["endpoint"],
            "freq":        e["freq"],
            "length":      e["length"],
            "freq_z":      e["freq_z"],
            "length_z":    e["length_z"],
            "score":       e["score"],
            "flags":       flags,
            "description": desc,
        })

    duration = (
        float(current_timestamp) - chain_events[0]["timestamp"]
        if chain_events else 0.0
    )

    if chain:
        first   = chain[0]
        last    = chain[-1]
        f_flag  = first["flags"][0] if first["flags"] else "mild deviation"
        l_flag  = last["flags"][0]  if last["flags"]  else "current alert"
        summary = (
            f"Causal chain: {len(chain)} events over {duration:.1f}s — "
            f"began with [{f_flag}] → escalated to [{l_flag}]."
        )
    else:
        summary = "No significant causal chain detected."

    return {
        "chain":            chain,
        "duration_seconds": round(duration, 1),
        "summary":          summary,
    }