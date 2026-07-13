"""
evaluation/baseline_runner.py
==============================
Compares three system configurations on UNSW-NB15:

  System A — KitNET alone
    Raw RMSE threshold on preprocessed UNSW-NB15 features.
    Threshold: threshold_percentile of training RMSE on normal flows.

  System B — KitNET + Decision Layer
    Uses the project's decision_engine.make_decision() on top of KitNET scores.
    The decision layer maps (score, risk, attack_type, freq) -> action.
    For UNSW-NB15 flows we set: risk = normalised_score, attack_type = "",
    freq = 1 (no temporal context in flow data).

  System C — KitNET + Decision Layer + Risk Engine
    Applies the project's risk_engine._compute_adjusted_risk() with a neutral
    context (endpoint="/other", freq=1) to compute adjusted_risk, then passes
    that through the decision layer.

Each system returns an EvalMetrics object with full confusion matrix and AUC.
Results are printed and saved to evaluation/results/comparison.csv.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from evaluation.config import ExperimentConfig, DEFAULT_CONFIG
from evaluation.unsw_dataset import UNSWDataset
from evaluation.unsw_preprocessor import FittedPreprocessor
from evaluation.kitsune_adapter import KitsuneUNSWAdapter
from evaluation.unsw_evaluator import EvalMetrics, evaluate


# ---------------------------------------------------------------------------
# System A: KitNET alone
# ---------------------------------------------------------------------------

def run_system_a(
    adapter: KitsuneUNSWAdapter,
    ds: UNSWDataset,
    prep: FittedPreprocessor,
) -> EvalMetrics:
    """KitNET RMSE threshold only."""
    scores, preds = adapter.predict_dataset(ds, prep, verbose=False)
    return evaluate(
        system_name = "A: KitNET only",
        y_true      = ds.y_test,
        y_pred      = preds,
        scores      = scores,
        threshold   = adapter.threshold,
    )


# ---------------------------------------------------------------------------
# System B: KitNET + Decision Layer
# ---------------------------------------------------------------------------

def run_system_b(
    adapter: KitsuneUNSWAdapter,
    ds: UNSWDataset,
    prep: FittedPreprocessor,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> EvalMetrics:
    """KitNET RMSE -> normalised risk -> decision_engine."""
    decision_engine = _load_decision_engine()
    if decision_engine is None:
        print("[baseline_runner] System B: decision_engine unavailable, skipping.")
        return _null_metrics("B: KitNET + Decision Layer", ds)

    scores, _ = adapter.predict_dataset(ds, prep, verbose=False)
    # Normalise scores to [0, max_risk_ceiling] for the decision layer.
    score_max = max(scores.max(), adapter.threshold * 2, 1e-9)
    preds_b = np.zeros(len(scores), dtype=int)

    for i, score in enumerate(scores):
        norm_risk = min(score / score_max * 3.0, 5.0)   # scale to risk 0-5
        result = decision_engine.make_decision(
            score       = float(score),
            risk        = norm_risk,
            attack_type = "",
            freq        = 1,
        )
        # decision_engine returns dict; "action" is "BLOCK" | "ALERT" | "ALLOW"
        action = result.get("action", "ALLOW") if isinstance(result, dict) else str(result)
        preds_b[i] = 1 if action in ("BLOCK", "ALERT") else 0

    return evaluate(
        system_name = "B: KitNET + Decision Layer",
        y_true      = ds.y_test,
        y_pred      = preds_b,
        scores      = scores,
        threshold   = adapter.threshold,
    )


# ---------------------------------------------------------------------------
# System C: KitNET + Decision Layer + Risk Engine
# ---------------------------------------------------------------------------

def run_system_c(
    adapter: KitsuneUNSWAdapter,
    ds: UNSWDataset,
    prep: FittedPreprocessor,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> EvalMetrics:
    """KitNET RMSE -> risk engine adjusted_risk -> decision_engine."""
    decision_engine = _load_decision_engine()
    risk_fn         = _load_risk_function()

    if decision_engine is None or risk_fn is None:
        print("[baseline_runner] System C: risk_engine unavailable, skipping.")
        return _null_metrics("C: KitNET + Decision Layer + Risk Engine", ds)

    scores, _ = adapter.predict_dataset(ds, prep, verbose=False)
    score_max = max(scores.max(), adapter.threshold * 2, 1e-9)
    preds_c = np.zeros(len(scores), dtype=int)

    neutral_ctx = {
        "endpoint": "/other",
        "freq":     1,
        "src":      "0.0.0.0",
    }

    for i, score in enumerate(scores):
        norm_risk = min(score / score_max * 3.0, 5.0)
        try:
            raw_risk, adjusted_risk = risk_fn(norm_risk, neutral_ctx)
        except Exception:
            adjusted_risk = norm_risk

        result = decision_engine.make_decision(
            score       = float(score),
            risk        = adjusted_risk,
            attack_type = "",
            freq        = 1,
        )
        action = result.get("action", "ALLOW") if isinstance(result, dict) else str(result)
        preds_c[i] = 1 if action in ("BLOCK", "ALERT") else 0

    return evaluate(
        system_name = "C: KitNET + Decision Layer + Risk Engine",
        y_true      = ds.y_test,
        y_pred      = preds_c,
        scores      = scores,
        threshold   = adapter.threshold,
    )


# ---------------------------------------------------------------------------
# Convenience: run all systems
# ---------------------------------------------------------------------------

def run_all_baselines(
    adapter: KitsuneUNSWAdapter,
    ds: UNSWDataset,
    prep: FittedPreprocessor,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> List[EvalMetrics]:
    print("\n[baseline_runner] Running System A ...")
    t0 = time.perf_counter()
    a  = run_system_a(adapter, ds, prep)
    print(f"  done ({time.perf_counter()-t0:.1f}s)")

    print("[baseline_runner] Running System B ...")
    t0 = time.perf_counter()
    b  = run_system_b(adapter, ds, prep, cfg)
    print(f"  done ({time.perf_counter()-t0:.1f}s)")

    print("[baseline_runner] Running System C ...")
    t0 = time.perf_counter()
    c  = run_system_c(adapter, ds, prep, cfg)
    print(f"  done ({time.perf_counter()-t0:.1f}s)")

    return [a, b, c]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_decision_engine():
    try:
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from decision_layer.decision_engine import make_decision

        class _DE:
            @staticmethod
            def make_decision(score, risk, attack_type, freq):
                return make_decision(score, risk, attack_type, freq)

        return _DE()
    except Exception as e:
        print(f"[baseline_runner] Cannot load decision_engine: {e}")
        return None


def _load_risk_function():
    try:
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        from risk_engine import _compute_adjusted_risk
        return _compute_adjusted_risk
    except Exception as e:
        print(f"[baseline_runner] Cannot load risk_engine: {e}")
        return None


def _null_metrics(name: str, ds: UNSWDataset) -> EvalMetrics:
    n = len(ds.y_test)
    return evaluate(
        system_name = name + " [UNAVAILABLE]",
        y_true      = ds.y_test,
        y_pred      = np.zeros(n, dtype=int),
        scores      = np.zeros(n, dtype=float),
        threshold   = 0.0,
    )
