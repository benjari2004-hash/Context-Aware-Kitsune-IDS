# bfde_explainer.py
# Behavioral Fingerprint Deviation Explanation (BFDE)
#
# FIXED: Categorical features (endpoint) are NO LONGER treated as
# numeric z-scores. They are handled with frequency-based comparison.
# Only truly numeric features (freq, length) use z-scores.
#
# RESEARCH NOTE: Per-entity baselines enable "entity-relative"
# explanations — more actionable than population-level anomaly scores.

from collections import defaultdict
from config import EXPLAINABILITY_CONFIG
from utils import zscore, deviation_label, direction_word, pct_change, format_float


# ── Numeric feature profiles: entity → dim → {count, mean, M2} ──
_num_profiles = defaultdict(lambda: defaultdict(
    lambda: {"count": 0, "mean": 0.0, "M2": 0.0}
))

# FIXED: Categorical profile — entity → endpoint → count
_cat_profiles = defaultdict(lambda: defaultdict(int))
_cat_totals   = defaultdict(int)


def _num_update(entity_id, dim, value):
    p = _num_profiles[entity_id][dim]
    p["count"] += 1
    delta       = value - p["mean"]
    p["mean"]  += delta / p["count"]
    p["M2"]    += delta * (value - p["mean"])


def _num_std(entity_id, dim):
    p = _num_profiles[entity_id][dim]
    if p["count"] < 2:
        return 1.0
    return max(1e-9, (p["M2"] / (p["count"] - 1)) ** 0.5)


def update(entity_id, ctx):
    """
    Update behavioral profile for entity_id.
    IMPORTANT: Call AFTER explanation is generated to avoid data leakage.

    Parameters
    ----------
    ctx : dict — must contain freq (float), length (int), endpoint (str)
    """
    # FIXED: numeric features only
    _num_update(entity_id, "freq",   float(ctx.get("freq",   0.0)))
    _num_update(entity_id, "length", float(ctx.get("length", 500.0)))

    # FIXED: categorical endpoint stored as frequency table
    ep = ctx.get("endpoint", "/web")
    _cat_profiles[entity_id][ep] += 1
    _cat_totals[entity_id]       += 1


def _endpoint_surprise(entity_id, current_ep):
    """
    FIXED: Returns a surprise score (0–1) for seeing this endpoint.
    Rare endpoints get high surprise; common ones get low surprise.
    """
    total = _cat_totals[entity_id]
    if total < 3:
        return 0.0, "insufficient history"
    count   = _cat_profiles[entity_id].get(current_ep, 0)
    freq_p  = count / total
    surprise = 1.0 - freq_p          # 0 = always seen, 1 = never seen
    desc     = (
        f"endpoint '{current_ep}' seen {count}/{total} times "
        f"({100*freq_p:.0f}% of history) — "
        f"{'rare' if surprise > 0.7 else 'common'} for this entity"
    )
    return round(surprise, 3), desc


def explain(entity_id, ctx):
    """
    Parameters
    ----------
    entity_id : str  — IP address
    ctx       : dict — current packet context

    Returns
    -------
    dict: deviations, endpoint_surprise, top_driver, summary
    """
    cfg   = EXPLAINABILITY_CONFIG
    top_k = cfg["bfde_top_k"]
    min_n = cfg["bfde_min_samples"]

    numeric_dims = {
        "freq":   float(ctx.get("freq",   0.0)),
        "length": float(ctx.get("length", 500.0)),
    }

    deviations = []
    for dim, current_val in numeric_dims.items():
        profile = _num_profiles[entity_id][dim]
        if profile["count"] < min_n:
            continue
        mean = profile["mean"]
        std  = _num_std(entity_id, dim)
        z    = zscore(current_val, mean, std)
        sev  = deviation_label(z)
        pct  = pct_change(current_val, mean)
        deviations.append({
            "dim":         dim,
            "value":       round(current_val, 3),
            "baseline":    round(mean, 3),
            "z":           round(z, 2),
            "severity":    sev,
            "description": (
                f"{dim}: {format_float(current_val)} "
                f"({'+' if pct >= 0 else ''}{pct:.0f}% "
                f"{direction_word(z)} baseline {format_float(mean)})"
            ),
        })

    deviations.sort(key=lambda d: abs(d["z"]), reverse=True)
    deviations = deviations[:top_k]

    # FIXED: endpoint handled separately as categorical surprise
    ep_surprise, ep_desc = _endpoint_surprise(
        entity_id, ctx.get("endpoint", "/web")
    )

    top_driver = deviations[0]["dim"] if deviations else (
        "endpoint" if ep_surprise > 0.7 else "unknown"
    )

    if deviations:
        d       = deviations[0]
        summary = (
            f"Entity {entity_id}: '{d['dim']}' is {d['z']:+.2f}σ from baseline "
            f"(severity={d['severity']}). {ep_desc}."
        )
    else:
        summary = (
            f"Entity {entity_id}: insufficient numeric history. {ep_desc}."
        )

    return {
        "deviations":        deviations,
        "endpoint_surprise": ep_surprise,
        "endpoint_desc":     ep_desc,
        "top_driver":        top_driver,
        "summary":           summary,
    }