"""
evaluation/experiment_logger.py
================================
Reproducibility and experiment tracking.

Saves:
  - experiment_meta.json  — seed, git hash, timestamp, config snapshot
  - metrics_comparison.csv  — one row per system with all metrics
  - training_rmse.csv  — KitNET RMSE scores from the training phase (normal flows)

All output goes to cfg.results_dir (default: evaluation/results/).
"""

from __future__ import annotations
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from evaluation.config import ExperimentConfig, DEFAULT_CONFIG
from evaluation.unsw_evaluator import EvalMetrics, save_metrics_csv


def save_experiment(
    cfg: ExperimentConfig,
    metrics_list: List[EvalMetrics],
    train_rmse: Optional[List[float]] = None,
    extra_meta: Optional[dict] = None,
) -> Path:
    """
    Persist all experiment outputs to cfg.results_dir.
    Returns the results directory path.
    """
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    meta = _build_meta(cfg, extra_meta or {})
    meta_path = results_dir / "experiment_meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[logger] Metadata saved: {meta_path}")

    metrics_path = results_dir / "metrics_comparison.csv"
    save_metrics_csv(metrics_list, metrics_path)
    print(f"[logger] Metrics saved:  {metrics_path}")

    if train_rmse:
        rmse_path = results_dir / "training_rmse.csv"
        _save_rmse(train_rmse, rmse_path)
        print(f"[logger] Training RMSE saved: {rmse_path}")

    return results_dir


def print_comparison_table(metrics_list: List[EvalMetrics]) -> None:
    """Side-by-side comparison of all systems."""
    if not metrics_list:
        return

    cols = ["system_name", "accuracy", "f1_anomaly", "recall_anomaly",
            "fpr", "fnr", "roc_auc", "tp", "fp", "tn", "fn"]
    widths = [40, 10, 12, 14, 10, 10, 10, 8, 8, 8, 8]

    header = "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print("\n" + "=" * len(header))
    print("BASELINE COMPARISON SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for m in metrics_list:
        values = [
            m.system_name,
            f"{m.accuracy:.4f}",
            f"{m.f1_anomaly:.4f}",
            f"{m.recall_anomaly:.4f}",
            f"{m.fpr:.4f}",
            f"{m.fnr:.4f}",
            f"{m.roc_auc:.4f}",
            str(m.tp),
            str(m.fp),
            str(m.tn),
            str(m.fn),
        ]
        row = "  ".join(f"{v:<{w}}" for v, w in zip(values, widths))
        print(row)

    print("=" * len(header))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_meta(cfg: ExperimentConfig, extra: dict) -> dict:
    return {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "git_hash":   _git_hash(),
        "config":     cfg.as_dict(),
        **extra,
    }


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _save_rmse(rmse: List[float], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "rmse"])
        for i, v in enumerate(rmse):
            writer.writerow([i, f"{v:.8f}"])
