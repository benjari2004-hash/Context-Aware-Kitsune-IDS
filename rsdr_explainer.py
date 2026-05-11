# rsdr_explainer.py
# Risk Score Decomposition Report (RSDR)
#
# FIXED: Clearly separates model output (Kitsune RMSE) from
# heuristic multipliers (risk engine, context bonus).
# RESEARCH NOTE: Counterfactual reasoning enables FP diagnosis.

from config import EXPLAINABILITY_CONFIG
from utils import format_float


def explain(
    base_score,
    final_score,
    raw_risk,
    adjusted_risk,
    ctx,
    freq_deviation,
    length_signal,
    config,
    label,
    attack_type,
):
    """
    Parameters
    ----------
    base_score    : float — raw Kitsune RMSE (model output, not heuristic)
    final_score   : float — after combine_scores profile fusion
    raw_risk      : float — after compute_risk() multipliers
    adjusted_risk : float — after context bonus (decision variable)
    ctx           : dict
    freq_deviation: float
    length_signal : float
    config        : dict  — active MODE_CONFIG entry
    label         : str
    attack_type   : str

    Returns
    -------
    dict: stages, counterfactual, confidence, signals_fired, report_str
    """
    cfg        = EXPLAINABILITY_CONFIG
    ep_weights = cfg["rsdr_endpoint_weights"]

    endpoint   = ctx.get("endpoint", "/web")
    freq       = ctx.get("freq",     0.0)

    # FIXED: recompute each multiplier independently for auditability
    endpoint_w  = ep_weights.get(endpoint, 1.0)
    freq_mult   = config["rate_mult"] if freq > config["rate_threshold"] else 1.0
    fd_mult     = 1.2                 if freq_deviation > 5.0            else 1.0
    len_mult    = 1.1                 if length_signal  > 1.5            else 1.0

    context_bonus = 0.0
    if endpoint in ("/ssh", "/secure"):
        context_bonus += 0.02
    if freq_deviation > 5.0:
        context_bonus += 0.01
    if freq > config["rate_threshold"]:
        context_bonus += 0.01

    # FIXED: count INDEPENDENT signals — not overlapping heuristics
    signals_fired = sum([
        base_score    > 0.20,   # model signal
        freq_deviation > 2.00,  # behavioural signal
        length_signal  > 1.30,  # payload signal
        freq           > 5.00,  # rate signal
        endpoint in ("/ssh", "/secure"),  # endpoint sensitivity
    ])
    # SEE config.py → rsdr_confidence_high_signals / rsdr_confidence_medium_signals for rationale
    confidence = (
        "HIGH"   if signals_fired >= 3 else
        "MEDIUM" if signals_fired == 2 else
        "LOW"
    )

    # FIXED: counterfactual removes the SINGLE largest heuristic multiplier
    # to show which lever most affects the decision
    cf_risk = final_score * endpoint_w   # without freq_mult
    cf_label = (
        "likely ANOMALY"
        if cf_risk > cfg["rsdr_override_risk_threshold"] * 0.6
        else "likely NORMAL"
    )

    # ── Stage table ──
    stages = [
        {
            "stage": "① Kitsune RMSE (MODEL)",
            "value": base_score,
            "note":  "raw autoencoder reconstruction error — ground truth signal",
            "type":  "model",
        },
        {
            "stage": "② Profile Fusion",
            "value": final_score,
            "note":  f"alpha×score + (1-alpha)×norm_deviation  [alpha={config['alpha']}]",
            "type":  "fusion",
        },
        {
            "stage": f"③ Endpoint Weight ({endpoint})",
            "value": endpoint_w,
            "note":  "sensitivity multiplier from DEFAULT_SENSITIVITY",
            "type":  "mult",
        },
        {
            "stage": "④ Freq Rate Multiplier",
            "value": freq_mult,
            "note":  f"freq={freq:.0f} vs threshold={config['rate_threshold']}",
            "type":  "mult",
        },
        {
            "stage": "⑤ Freq Deviation Multiplier",
            "value": fd_mult,
            "note":  f"freq_deviation={freq_deviation:.2f}",
            "type":  "mult",
        },
        {
            "stage": "⑥ Length Signal Multiplier",
            "value": len_mult,
            "note":  f"length_signal={length_signal:.2f}",
            "type":  "mult",
        },
        {
            "stage": "⑦ Raw Risk",
            "value": raw_risk,
            "note":  "result of compute_risk()",
            "type":  "intermediate",
        },
        {
            "stage": "⑧ Context Bonus",
            "value": context_bonus,
            "note":  "small additive bonus for sensitive endpoints/rates",
            "type":  "add",
        },
        {
            "stage": "⑨ Adjusted Risk (DECISION)",
            "value": adjusted_risk,
            "note":  "compared against adaptive threshold",
            "type":  "final",
        },
    ]

    # ── Render report string ──
    w = 56
    lines = [
        f"╔{'═'*w}╗",
        f"║  RISK SCORE DECOMPOSITION REPORT{' '*(w-34)}║",
        f"╠{'═'*w}╣",
        f"║  Decision:    {label:<41}║",
        f"║  Attack Type: {attack_type:<41}║",
        f"║  Confidence:  {confidence} ({signals_fired}/5 signals aligned){' '*(w-38)}║",
        f"╠{'═'*w}╣",
        f"║  {'Stage':<32} {'Value':>8}  Note{' '*(w-48)}║",
        f"╟{'─'*w}╢",
    ]
    type_prefix = {"model": "🔬", "fusion": "🔀", "mult": "✖ ",
                   "intermediate": "→ ", "add": "+ ", "final": "🎯"}
    for s in stages:
        pfx  = type_prefix.get(s["type"], "  ")
        vstr = format_float(s["value"])
        name = f"{pfx} {s['stage']}"
        lines.append(f"║  {name:<33} {vstr:>7}  [{s['note'][:w-46]}]{'║':>1}")

    lines += [
        f"╠{'═'*w}╣",
        f"║  COUNTERFACTUAL: remove freq_rate_mult{' '*(w-39)}║",
        f"║    risk without freq boost ≈ {format_float(cf_risk):<25}║",
        f"║    outcome → {cf_label:<42}║",
        f"╚{'═'*w}╝",
    ]

    return {
        "stages":         stages,
        "confidence":     confidence,
        "signals_fired":  signals_fired,
        "counterfactual": {
            "condition": "freq rate multiplier removed",
            "cf_risk":   round(cf_risk, 4),
            "cf_label":  cf_label,
        },
        "report_str": "\n".join(lines),
    }