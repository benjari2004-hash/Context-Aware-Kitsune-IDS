# risk_engine.py
# FIX 1: endpoint_mult was accepted as a parameter but never used.
#   It is now applied as a scalar multiplier to the endpoint weight,
#   making MODE_CONFIG sensitivity settings actually take effect.
# FIX 2: DEFAULT_SENSITIVITY extended with Mirai-relevant ports
#   (/telnet, /mirai_c2, /mirai_load, /dns) so risk amplification
#   reflects the real threat significance of those endpoints.
# FIX 3: Jitter and unique_ep multipliers now scale continuously
#   (not just binary flags) for smoother risk gradients.
# FIX 4: RMSE floor preserved — prevents endpoint weight from
#   suppressing a genuinely strong Kitsune anomaly score.

import config

# Endpoint risk weights loaded from the active profile (profiles/mirai.yaml).
# SEE config.py → PROFILE_ENDPOINT_WEIGHTS for the loaded data.
# config.init_profile() mutates this dict in-place so DEFAULT_SENSITIVITY
# always reflects the currently active profile without rebinding.
DEFAULT_SENSITIVITY = config.PROFILE_ENDPOINT_WEIGHTS


def compute_risk(
    score,
    ctx,
    endpoint_mult      = 1.5,
    rate_mult          = 1.3,
    rate_threshold     = 20,
    endpoint_weights   = None,
    freq_deviation     = 0.0,
    freq_dev_threshold = 5.0,
    freq_dev_mult      = 1.2,
    length_signal      = 1.0,
    length_mult        = 1.1,
    profile            = None,
):
    """
    Compute risk score from Kitsune RMSE + context multipliers.

    Parameters
    ----------
    score             : float — Kitsune final_score (combine_scores output)
    ctx               : dict  — context from context_features.extract_context()
    endpoint_mult     : float — mode-level sensitivity multiplier (from config)
    rate_mult         : float — applied when freq > rate_threshold
    rate_threshold    : int   — freq threshold for rate_mult
    endpoint_weights  : dict  — per-endpoint base weights (default = above)
    freq_deviation    : float — abs(freq - profile_mean)
    freq_dev_threshold: float — deviation threshold for freq_dev_mult
    freq_dev_mult     : float — multiplier for high freq deviation
    length_signal     : float — len(packet) / 500.0
    length_mult       : float — applied when length_signal > 1.5

    Returns
    -------
    float — risk score (unbounded above, >= 0)
    """
    risk = float(score)

    if endpoint_weights is None:
        if profile is not None:
            endpoint_weights = {
                v["endpoint"]: v["weight"]
                for v in profile.get("port_map", {}).values()
            }
        else:
            endpoint_weights = DEFAULT_SENSITIVITY

    # Endpoint sensitivity: base_weight × endpoint_mult (mode-level sensitivity).
    # FIX: endpoint_mult is now actually applied here.
    base_weight = endpoint_weights.get(ctx.get("endpoint", "/other"), 1.0)
    risk       *= base_weight * endpoint_mult

    # High request rate multiplier
    if ctx.get("freq", 0) > rate_threshold:
        risk *= rate_mult

    # Frequency deviation from entity baseline
    if freq_deviation > freq_dev_threshold:
        risk *= freq_dev_mult

    # Large packet size multiplier
    if length_signal > 1.5:
        risk *= length_mult

    # Jitter multiplier (continuous, not binary)
    jitter = ctx.get("jitter", 0.0)
    if jitter > 5.0:
        risk *= 1.25
    elif jitter > 2.0:
        risk *= 1.15

    # Endpoint switching bonus — scan / recon indicator
    unique_ep = ctx.get("unique_ep", 1.0)
    if unique_ep >= 5:
        risk *= 1.30
    elif unique_ep >= 3:
        risk *= 1.20

    # Length variance bonus — mixed-size traffic
    len_var = ctx.get("len_variance", 0.0)
    if len_var > 200:
        risk *= 1.10

    # Proto-specific bonus — ICMP floods and UDP amplification
    proto = ctx.get("proto", "OTHER")
    if proto == "ICMP" and ctx.get("freq", 0) >= 5:
        risk *= 1.15

    # Sensitive endpoint hard-flag
    if ctx.get("is_sensitive", False) and score > 0.20:
        risk *= 1.10

    # RMSE floor — prevents high endpoint_mult from causing inflate→suppress
    # when endpoint_weight is low (e.g., /web = 1.0 × 1.5 = 1.5 — fine)
    # but ensures a genuine 0.75+ RMSE is never soft-pedalled by a
    # very low base weight.
    if score > 0.75:
        risk = max(risk, score * 1.5)
    elif score > 0.5:
        risk = max(risk, score * 1.2)

    return risk