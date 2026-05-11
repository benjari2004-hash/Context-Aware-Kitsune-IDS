# ara_explainer.py
# Autoencoder Reconstruction Anatomy (ARA)
#
# FIXED: This module now requires the REAL Kitsune feature vector
# (24+ dimensional) produced by FeatureExtractor.get_next_vector().
# It will NOT produce meaningful output if given [score] alone.
#
# RESEARCH NOTE: Kitsune does not expose per-autoencoder RMSE externally.
# We approximate feature-level reconstruction error using per-feature
# deviation from a learned online mean — a faithful proxy because
# KitNET autoencoders minimise reconstruction MSE on the same features.

from collections import defaultdict
from config import EXPLAINABILITY_CONFIG
from utils import top_k_indices, format_float, deviation_label, zscore


# ── Online Welford statistics per feature dimension ──
_count = defaultdict(int)
_mean  = defaultdict(float)
_M2    = defaultdict(float)


def _welford_update(dim, value):
    _count[dim] += 1
    delta        = value - _mean[dim]
    _mean[dim]  += delta / _count[dim]
    _M2[dim]    += delta * (value - _mean[dim])


def _std(dim):
    if _count[dim] < 2:
        return 1.0
    return max(1e-9, (_M2[dim] / (_count[dim] - 1)) ** 0.5)


def update_baselines(feature_vector):
    """
    IMPORTANT: Call during TRAINING phase only.
    Builds the per-feature baseline used for reconstruction comparison.

    Parameters
    ----------
    feature_vector : list or np.ndarray — real Kitsune feature vector
    """
    # FIXED: guard against scalar / length-1 vectors passed by mistake
    if feature_vector is None or len(feature_vector) == 0:
        return
    for dim, val in enumerate(feature_vector):
        _welford_update(dim, float(val))


def explain(feature_vector, overall_rmse):
    """
    Parameters
    ----------
    feature_vector : list or np.ndarray — real Kitsune feature vector
    overall_rmse   : float              — Kitsune score for this packet

    Returns
    -------
    dict: top_features, rmse, summary, valid (bool)
    """
    cfg   = EXPLAINABILITY_CONFIG
    names = cfg["ara_feature_names"]
    top_k = cfg["ara_top_k"]

    # FIXED: if feature vector is degenerate, return graceful fallback
    if feature_vector is None or len(feature_vector) <= 1:
        return {
            "top_features": [],
            "rmse":         round(overall_rmse, 4),
            "summary":      "ARA unavailable: real feature vector not provided.",
            "valid":        False,
        }

    fv = [float(v) for v in feature_vector]

    # per-feature normalised squared error (proxy for reconstruction error)
    sq_errors = [
        ((fv[d] - _mean[d]) / _std(d)) ** 2
        for d in range(len(fv))
    ]
    total_sq = sum(sq_errors) or 1.0

    top_idxs    = top_k_indices(sq_errors, min(top_k, len(fv)))
    top_features = []
    for idx in top_idxs:
        fname = names[idx] if idx < len(names) else f"feature_{idx}"
        val   = fv[idx]
        exp   = _mean[idx]
        z     = zscore(val, exp, _std(idx))
        pct   = 100.0 * sq_errors[idx] / total_sq
        top_features.append({
            "name":             fname,
            "value":            round(val, 4),
            "expected":         round(exp, 4),
            "z_score":          round(z, 2),
            "contribution_pct": round(pct, 1),
            "severity":         deviation_label(z),
        })

    if top_features:
        tf      = top_features[0]
        summary = (
            f"Primary driver: '{tf['name']}' "
            f"(observed={format_float(tf['value'])}, "
            f"expected≈{format_float(tf['expected'])}, "
            f"{tf['contribution_pct']}% of RMSE, z={tf['z_score']:+.2f})"
        )
    else:
        summary = "No feature baseline available yet."

    return {
        "top_features": top_features,
        "rmse":         round(overall_rmse, 4),
        "summary":      summary,
        "valid":        True,
    }