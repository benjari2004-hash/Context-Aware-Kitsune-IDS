"""
evaluation/run_unsw_experiment.py
==================================
Main entry point for the UNSW-NB15 evaluation experiment.

Usage
-----
    python evaluation/run_unsw_experiment.py [options]

Options
-------
    --csv PATH          Path to UNSW-NB15 CSV (default: from config.py)
    --results DIR       Output directory (default: evaluation/results/)
    --seed INT          Random seed (default: 42)
    --test-frac FLOAT   Test split fraction (default: 0.20)
    --threshold-pct FLOAT  RMSE percentile for anomaly threshold (default: 99.0)
    --fm-grace INT      KitNET feature-mapping grace period (default: 5000)
    --ad-grace INT      KitNET anomaly-detection grace period (default: 15000)
    --system {A,B,C,all}  Which system(s) to run (default: all)
    --no-save           Skip saving results to disk

Example
-------
    D:\Anaconda3\envs\kitsune\python.exe evaluation/run_unsw_experiment.py
    D:\Anaconda3\envs\kitsune\python.exe evaluation/run_unsw_experiment.py --system A --threshold-pct 95
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

# Make the project root importable when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.config import ExperimentConfig, UNSW_CSV
from evaluation.unsw_dataset import load_unsw_nb15, print_dataset_summary
from evaluation.unsw_preprocessor import fit_preprocessor
from evaluation.kitsune_adapter import KitsuneUNSWAdapter
from evaluation.baseline_runner import run_all_baselines, run_system_a, run_system_b, run_system_c
from evaluation.unsw_evaluator import print_metrics, save_metrics_csv
from evaluation.experiment_logger import save_experiment, print_comparison_table


def main(argv=None):
    args = _parse_args(argv)

    cfg = ExperimentConfig(
        unsw_csv             = Path(args.csv),
        results_dir          = Path(args.results),
        random_seed          = args.seed,
        test_fraction        = args.test_frac,
        threshold_percentile = args.threshold_pct,
        kitnet_fm_grace      = args.fm_grace,
        kitnet_ad_grace      = args.ad_grace,
    )

    print("\n" + "=" * 60)
    print("  Context-Aware Kitsune IDS — UNSW-NB15 Evaluation")
    print("=" * 60)
    print(f"  Dataset: {cfg.unsw_csv}")
    print(f"  Seed:    {cfg.random_seed}   Test fraction: {cfg.test_fraction}")
    print(f"  AD grace: {cfg.kitnet_ad_grace:,}   Threshold pct: {cfg.threshold_percentile}")
    print("=" * 60)

    # ── Step 1: Load dataset ──────────────────────────────────────────────
    print("\n[1/4] Loading UNSW-NB15 dataset ...")
    t0 = time.perf_counter()
    ds = load_unsw_nb15(cfg)
    print(f"  Loaded in {time.perf_counter()-t0:.1f}s")
    print_dataset_summary(ds)

    # ── Step 2: Preprocess ────────────────────────────────────────────────
    print("\n[2/4] Fitting preprocessor on training data ...")
    t0 = time.perf_counter()
    prep = fit_preprocessor(ds, cfg)
    print(f"  Output dimension: {prep.output_dim} features  ({time.perf_counter()-t0:.1f}s)")

    # ── Step 3: Train KitNET on normal training flows ─────────────────────
    print("\n[3/4] Training KitNET on normal flows ...")
    t0 = time.perf_counter()
    adapter = KitsuneUNSWAdapter(cfg)
    adapter.fit(ds, prep, verbose=True)
    print(f"  Training complete in {time.perf_counter()-t0:.1f}s")

    # ── Step 4: Evaluate systems ───────────────────────────────────────────
    print("\n[4/4] Evaluating systems ...")
    system = args.system.lower()

    if system == "all":
        metrics_list = run_all_baselines(adapter, ds, prep, cfg)
    elif system == "a":
        metrics_list = [run_system_a(adapter, ds, prep)]
    elif system == "b":
        metrics_list = [run_system_b(adapter, ds, prep, cfg)]
    elif system == "c":
        metrics_list = [run_system_c(adapter, ds, prep, cfg)]
    else:
        sys.exit(f"Unknown system '{args.system}'. Use A, B, C, or all.")

    # ── Print results ─────────────────────────────────────────────────────
    print()
    for m in metrics_list:
        print_metrics(m)
        print()

    print_comparison_table(metrics_list)

    # ── Save results ──────────────────────────────────────────────────────
    if not args.no_save:
        results_dir = save_experiment(
            cfg          = cfg,
            metrics_list = metrics_list,
            train_rmse   = adapter._train_rmse,
            extra_meta   = {"system": args.system, "output_dim": prep.output_dim},
        )
        print(f"\n[done] Results saved to: {results_dir}")
    else:
        print("\n[done] (--no-save: results not persisted)")


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Evaluate Context-Aware Kitsune IDS on UNSW-NB15",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv",           default=str(UNSW_CSV), metavar="PATH",
                        help="Path to UNSW_NB15_training-set.csv")
    parser.add_argument("--results",       default="evaluation/results", metavar="DIR",
                        help="Output directory for metrics and logs")
    parser.add_argument("--seed",          default=42,   type=int)
    parser.add_argument("--test-frac",     default=0.20, type=float, dest="test_frac")
    parser.add_argument("--threshold-pct", default=99.0, type=float, dest="threshold_pct",
                        help="Percentile of training RMSE used as anomaly threshold")
    parser.add_argument("--fm-grace",      default=5000, type=int, dest="fm_grace",
                        help="KitNET feature-mapping grace period (samples)")
    parser.add_argument("--ad-grace",      default=15000, type=int, dest="ad_grace",
                        help="KitNET anomaly-detection grace period (samples)")
    parser.add_argument("--system",        default="all", choices=["A","B","C","all"],
                        help="Which system(s) to evaluate")
    parser.add_argument("--no-save",       action="store_true", dest="no_save",
                        help="Skip saving results to disk")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
