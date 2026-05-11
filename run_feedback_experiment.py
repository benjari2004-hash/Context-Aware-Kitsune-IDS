"""
Phase 8 — Explainability-Guided Learning Experiment
====================================================
Standalone script: loads existing results.csv + ground_truth.csv,
runs 10 rounds × 50 simulated analyst feedbacks, measures FP/FN/F1 per
round with learned feature weights, and saves three diagnostic plots.

Run from the my_ids/ directory:
    python run_feedback_experiment.py
"""
import os
import sys

# ── path setup so 'feedback' package is importable ──────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from feedback.evaluation import FeedbackEvaluator
from feedback.drift_detector import FeedbackDriftDetector
from feedback.feedback_simulator import FeedbackSimulator

RESULTS_PATH       = os.path.join(HERE, "results.csv")
GROUND_TRUTH_PATH  = os.path.join(HERE, "ground_truth.csv")
OUTPUT_DIR         = HERE
ROUNDS             = 10
FEEDBACKS_PER_ROUND = 50


def check_inputs():
    ok = True
    for p in (RESULTS_PATH, GROUND_TRUTH_PATH):
        if not os.path.exists(p):
            print(f"[ERROR] Missing file: {p}")
            ok = False
    return ok


def print_final_summary(metrics: list, plots: list):
    if not metrics:
        print("\n[SUMMARY] No metrics available.")
        return

    r0 = metrics[0]
    r_last = metrics[-1]

    fp0  = r0["fp_rate"]  * 100
    fp10 = r_last["fp_rate"] * 100
    fn0  = r0["fn_rate"]  * 100
    fn10 = r_last["fn_rate"] * 100
    fp_reduction = fp0 - fp10
    fn_stable    = abs(fn10 - fn0) < 2.0  # within 2 pp

    drift = r_last.get("drift", {})

    snap_last  = r_last.get("weight_snapshot", {})
    example_lines = []
    for ctx in sorted(snap_last):
        for feat, w in sorted(snap_last[ctx].items()):
            example_lines.append(
                f"  Context '{ctx}': {feat} weight changed from 1.0 to {w:.4f} "
                f"after {ROUNDS * FEEDBACKS_PER_ROUND} feedbacks"
            )

    print("\n" + "=" * 60)
    print("  PHASE 8 — FINAL SUMMARY")
    print("=" * 60)
    created = [
        "my_ids/feedback/__init__.py",
        "my_ids/feedback/feedback_types.py",
        "my_ids/feedback/feedback_store.py",
        "my_ids/feedback/feature_weight_learner.py",
        "my_ids/feedback/adaptive_threshold.py",
        "my_ids/feedback/drift_detector.py",
        "my_ids/feedback/feedback_simulator.py",
        "my_ids/feedback/evaluation.py",
        "my_ids/run_feedback_experiment.py",
    ]
    print("Files created:")
    for f in created:
        print(f"  {f}")

    sim_pass = "PASS" if metrics else "FAIL"
    print(f"\nFeedback simulation ran:    {sim_pass}")
    print(f"FP rate at round 0:         {fp0:.2f}%")
    print(f"FP rate at round {ROUNDS}:        {fp10:.2f}%")
    print(f"FP reduction:               {fp_reduction:.2f}%")
    print(f"FN rate stable:             {'YES' if fn_stable else 'NO'}  "
          f"({fn0:.2f}% → {fn10:.2f}%)")

    print("\nPlots generated:")
    for p in plots:
        exists = "OK" if os.path.exists(p) else "MISSING"
        print(f"  {os.path.basename(p)}  [{exists}]")

    if example_lines:
        print("\nFeature weight examples:")
        for line in example_lines[:6]:
            print(line)
    else:
        print("\n(No feature weight changes recorded — all feedback may be TP "
              "or ground truth is empty.)")

    drift_detected = drift.get("drift_detected", False)
    print(f"\nDrift detected:             {'YES' if drift_detected else 'NO'}")
    print(f"  current FP rate in window: {drift.get('current_fp_rate', 0):.4f}")
    print(f"  baseline FP rate:          {drift.get('baseline_fp_rate', 0):.4f}")
    print(f"  delta:                     {drift.get('delta', 0):+.4f}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("  Phase 8 — Explainability-Guided Learning Experiment")
    print("=" * 60)

    if not check_inputs():
        sys.exit(1)

    evaluator = FeedbackEvaluator(
        results_path=RESULTS_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        output_dir=OUTPUT_DIR,
        noise_rate=0.05,
        response_rate=0.30,
    )

    metrics = evaluator.run_full_evaluation(
        rounds=ROUNDS,
        feedbacks_per_round=FEEDBACKS_PER_ROUND,
    )

    evaluator.generate_report()

    expected_plots = [
        os.path.join(OUTPUT_DIR, "feedback_fp_reduction.png"),
        os.path.join(OUTPUT_DIR, "feedback_threshold_trajectory.png"),
        os.path.join(OUTPUT_DIR, "feedback_feature_weights.png"),
    ]

    print_final_summary(metrics, expected_plots)


if __name__ == "__main__":
    main()
