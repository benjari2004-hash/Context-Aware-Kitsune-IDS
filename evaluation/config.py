"""
evaluation/config.py
====================
Central configuration for the UNSW-NB15 evaluation pipeline.
All paths, hyperparameters, and seeds live here.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path

_EVAL_DIR  = Path(__file__).resolve().parent
_ROOT      = _EVAL_DIR.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

UNSW_CSV   = _ROOT / "UNSW_NB15_training-set" / "UNSW_NB15_training-set.csv"
RESULTS_DIR = _EVAL_DIR / "results"


@dataclass
class ExperimentConfig:
    # --- data ---
    unsw_csv:        Path = UNSW_CSV
    results_dir:     Path = RESULTS_DIR
    test_fraction:   float = 0.20    # stratified train/test split
    random_seed:     int   = 42

    # --- KitNET hyperparameters ---
    kitnet_max_ae:        int   = 10
    kitnet_fm_grace:      int   = 5_000   # feature-mapping phase (samples)
    kitnet_ad_grace:      int   = 15_000  # anomaly-detection training phase
    kitnet_learning_rate: float = 0.1
    kitnet_hidden_ratio:  float = 0.75

    # --- anomaly threshold (percentile of training RMSE scores) ---
    threshold_percentile: float = 99.0

    # --- decision layer pass-through ---
    decision_rate_threshold: int   = 50   # packets/30 s to trigger rate multiplier
    decision_hard_override:  float = 1.5  # risk level that forces ANOMALY

    # --- column names in UNSW-NB15 CSV ---
    label_col:      str = "label"         # binary: 0=Normal 1=Attack
    attack_cat_col: str = "attack_cat"    # string category
    id_col:         str = "id"            # row identifier (drop)
    cat_cols:       list = field(default_factory=lambda: ["proto", "service", "state"])

    def as_dict(self) -> dict:
        d = asdict(self)
        d["unsw_csv"]    = str(d["unsw_csv"])
        d["results_dir"] = str(d["results_dir"])
        return d


# Module-level default — callers can import this directly or construct their own.
DEFAULT_CONFIG = ExperimentConfig()
