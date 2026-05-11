# config.py
# Central configuration for the Explainability Engine

import yaml
from pathlib import Path

_THIS_DIR            = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = str(_THIS_DIR / "profiles" / "mirai.yaml")
DEFAULT_PROFILE      = _THIS_DIR / "profiles" / "mirai.yaml"


def load_profile(profile_path):
    with open(profile_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Profile-derived lookup tables — populated in-place by init_profile().
# In-place mutation (.clear() + .update()) means all existing references
# in risk_engine.py, rsdr_explainer.py, etc. see updates without rebinding.
PROFILE_PORT_ENDPOINTS  = {}  # int(port) → endpoint_name
PROFILE_ENDPOINT_WEIGHTS = {}  # endpoint_name → float weight


def init_profile(path=None):
    """
    Load a traffic profile YAML and populate PROFILE_PORT_ENDPOINTS /
    PROFILE_ENDPOINT_WEIGHTS in-place. Call with the path from --profile,
    or omit to use DEFAULT_PROFILE_PATH (profiles/mirai.yaml).
    """
    _path = path or DEFAULT_PROFILE_PATH
    with open(_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    port_map = data.get("port_map", {})

    PROFILE_PORT_ENDPOINTS.clear()
    PROFILE_PORT_ENDPOINTS.update(
        {int(port): entry["endpoint"] for port, entry in port_map.items()}
    )

    PROFILE_ENDPOINT_WEIGHTS.clear()
    PROFILE_ENDPOINT_WEIGHTS.update(
        {entry["endpoint"]: entry["weight"] for entry in port_map.values()}
    )


EXPLAINABILITY_CONFIG = {

    # ── Behavioral Fingerprint (BFDE) ──
    "bfde_min_samples":   5,
    "bfde_zscore_high":   2.5,
    "bfde_zscore_medium": 1.5,
    "bfde_top_k":         3,

    # ── Temporal Causal Chain (TCCE) ──
    # FIXED: window is now in real seconds, not packet index
    "tcce_window_seconds": 300,
    "tcce_min_events":     2,
    "tcce_deviation_gate": 1.2,

    # ── ARA (Autoencoder Reconstruction Anatomy) ──
    "ara_top_k": 5,
    # RESEARCH NOTE: These names map to Kitsune's AfterImage stat groups.
    # Each group of 6 = [mean, std, radius, magnitude, cov, pcc]
    # Groups: MAC-IP | Host-Host | Jitter | Host-Port
    "ara_feature_names": [
        "mac_ip_mean",      "mac_ip_std",      "mac_ip_radius",
        "mac_ip_magnitude", "mac_ip_cov",      "mac_ip_pcc",
        "h2h_mean",         "h2h_std",         "h2h_radius",
        "h2h_magnitude",    "h2h_cov",         "h2h_pcc",
        "jitter_mean",      "jitter_std",       "jitter_radius",
        "jitter_magnitude", "jitter_cov",       "jitter_pcc",
        "h2p_mean",         "h2p_std",          "h2p_radius",
        "h2p_magnitude",    "h2p_cov",          "h2p_pcc",
    ],

    # ── Risk Decomposition (RSDR) ──
    # rsdr_endpoint_weights references PROFILE_ENDPOINT_WEIGHTS directly.
    # init_profile() mutates that dict in-place so rsdr_explainer always
    # sees the active profile's weights without rebinding.
    "rsdr_endpoint_weights": PROFILE_ENDPOINT_WEIGHTS,
    "rsdr_override_risk_threshold":  1.5,
    "rsdr_override_score_threshold": 0.75,

    # ── Narrative Engine ──
    "narrative_confidence_high":   0.75,
    "narrative_confidence_medium": 0.50,
    "narrative_max_chain_events":  5,

    # ── Detection Thresholds ──
    # Controls the minimum Kitsune score accepted by any classifier rule.
    # Range: 0.0–1.0. Increasing: rules fire less; more traffic falls to the
    # catch-all "Suspicious Activity". Decreasing: trivially weak RMSE values
    # can drive attack labels.
    # No derivation — needs ablation study.
    "classifier_score_floor": 0.10,

    # Controls the hard-override risk level (CLI --hard-override default).
    # Range: 0.5–5.0. Increasing: extreme-risk packets may slip to the adaptive
    # threshold path instead of being force-labelled ANOMALY. Decreasing: more
    # packets bypass the adaptive threshold entirely → higher FP rate.
    # No derivation — needs ablation study.
    "hard_override_risk": 1.5,

    # Controls the Kitsune-score weight in combine_scores() (profile_scoring.py).
    # Range: 0.0–1.0. Increasing: detection relies more on raw Kitsune RMSE.
    # Decreasing: profile deviation dominates, penalising behavioural changes more.
    # No derivation — needs ablation study.
    "profile_fusion_alpha": 0.7,

    # Controls the ceiling on AdaptiveThreshold in NORMAL mode (MODE_CONFIG).
    # Range: 1.0–10.0. Increasing: threshold can drift higher → more FNs.
    # Decreasing: tighter cap → adaptive threshold is more aggressive.
    # No derivation — needs ablation study.
    "adaptive_max_threshold_normal": 2.5,

    # Controls the request-rate count that activates the rate multiplier.
    # Measured in packets per _RATE_WINDOW (30 s). Range: 10–200.
    # Increasing: rate multiplier fires less → lower risk scores overall.
    # Decreasing: more traffic treated as high-rate → higher risk scores.
    # No derivation — needs ablation study.
    "rate_threshold_normal": 50,

    # Controls RSDR confidence tier boundaries (out of 5 independent signals).
    # HIGH if signals_fired >= rsdr_confidence_high_signals (default 3).
    # MEDIUM if signals_fired == rsdr_confidence_medium_signals (default 2).
    # Increasing high_signals: fewer packets reach HIGH confidence.
    # Decreasing medium_signals: more packets collapse to LOW confidence.
    # No derivation — needs ablation study.
    "rsdr_confidence_high_signals":   3,
    "rsdr_confidence_medium_signals": 2,
}

# Populate profile tables at import time using the default profile.
# All modules that reference PROFILE_PORT_ENDPOINTS / PROFILE_ENDPOINT_WEIGHTS
# see the populated data immediately. Call init_profile(path) to switch profiles.
init_profile()
