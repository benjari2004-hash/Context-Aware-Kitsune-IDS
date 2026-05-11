"""
Evaluates the impact of explainability-guided feedback learning.

Compares FP/FN/F1 metrics before and after N rounds of simulated analyst
feedback, then generates plots and a summary table.
"""
import os

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .feedback_simulator import FeedbackSimulator


class FeedbackEvaluator:
    """
    Evaluates the impact of feedback learning by running the simulator and
    measuring FP/FN/F1 per round.
    """

    def __init__(
        self,
        results_path="results.csv",
        ground_truth_path="ground_truth.csv",
        output_dir=".",
        noise_rate=0.05,
        response_rate=0.30,
    ):
        self.output_dir = output_dir
        self.simulator = FeedbackSimulator(
            results_path=results_path,
            ground_truth_path=ground_truth_path,
            noise_rate=noise_rate,
            response_rate=response_rate,
        )
        self._metrics: list = []

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run_full_evaluation(self, rounds=10, feedbacks_per_round=50) -> list:
        """
        1. Compute baseline metrics (round 0).
        2. Run FeedbackSimulator for `rounds` rounds.
        3. After each round, apply feature weight adjustments and recompute FP/FN/F1.
        4. Generate and save PNG plots.

        Returns list of per-round metric dicts.
        """
        print(f"[Evaluator] Running {rounds} feedback rounds × {feedbacks_per_round} feedbacks each …")
        self._metrics = self.simulator.run_simulation(
            rounds=rounds,
            feedbacks_per_round=feedbacks_per_round,
        )
        if not self._metrics:
            print("[Evaluator] No metrics — check that results.csv and ground_truth.csv exist.")
            return []

        self._plot_fp_reduction()
        self._plot_threshold_trajectory()
        self._plot_feature_weights()
        print(f"[Evaluator] Plots saved to: {os.path.abspath(self.output_dir)}")
        return self._metrics

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _savefig(self, name: str):
        path = os.path.join(self.output_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  Saved: {path}")

    def _plot_fp_reduction(self):
        if not HAS_MATPLOTLIB:
            return
        rounds = [m["round"] for m in self._metrics]
        fp_rates = [m["fp_rate"] * 100 for m in self._metrics]
        fn_rates = [m["fn_rate"] * 100 for m in self._metrics]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rounds, fp_rates, "r-o", label="FP rate (%)", linewidth=2)
        ax.plot(rounds, fn_rates, "b-s", label="FN rate (%)", linewidth=2)
        ax.set_xlabel("Feedback Round")
        ax.set_ylabel("Rate (%)")
        ax.set_title("FP / FN Rate Over Feedback Rounds")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._savefig("feedback_fp_reduction.png")

    def _plot_threshold_trajectory(self):
        if not HAS_MATPLOTLIB:
            return
        rounds = [m["round"] for m in self._metrics]
        thresholds = [m["threshold"] for m in self._metrics]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rounds, thresholds, "g-^", linewidth=2)
        ax.axhline(y=thresholds[0], color="grey", linestyle="--", alpha=0.5,
                   label=f"Baseline = {thresholds[0]:.3f}")
        ax.set_xlabel("Feedback Round")
        ax.set_ylabel("Risk Threshold")
        ax.set_title("Feedback-Adaptive Threshold Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._savefig("feedback_threshold_trajectory.png")

    def _plot_feature_weights(self):
        if not HAS_MATPLOTLIB:
            return

        # Collect all (context, feature) pairs seen across rounds
        all_series: dict = {}
        rounds = [m["round"] for m in self._metrics]

        for m in self._metrics:
            for ctx, feats in m.get("weight_snapshot", {}).items():
                for feat, w in feats.items():
                    key = f"{ctx} / {feat}"
                    all_series.setdefault(key, [])

        # Fill weight values per round (default 1.0 when not yet seen)
        for m in self._metrics:
            snap = m.get("weight_snapshot", {})
            seen = set()
            for ctx, feats in snap.items():
                for feat, w in feats.items():
                    key = f"{ctx} / {feat}"
                    all_series[key].append(w)
                    seen.add(key)
            for key in all_series:
                if key not in seen and len(all_series[key]) < len(rounds):
                    all_series[key].append(1.0)

        if not all_series:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        for key, weights in sorted(all_series.items()):
            if len(weights) == len(rounds):
                ax.plot(rounds, weights, marker="o", linewidth=1.5, label=key)

        ax.axhline(y=1.0, color="grey", linestyle="--", alpha=0.4, label="baseline (1.0)")
        ax.set_xlabel("Feedback Round")
        ax.set_ylabel("Feature Weight")
        ax.set_title("Feature Weight Evolution by Context")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        self._savefig("feedback_feature_weights.png")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        if not self._metrics:
            return "(no metrics — run run_full_evaluation first)"

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("  FEEDBACK LEARNING EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"{'Round':>5}  {'Threshold':>9}  {'FP Rate':>8}  {'FN Rate':>8}  {'F1':>6}")
        lines.append("-" * 60)

        for m in self._metrics:
            lines.append(
                f"{m['round']:>5}  "
                f"{m['threshold']:>9.3f}  "
                f"{m['fp_rate']*100:>7.2f}%  "
                f"{m['fn_rate']*100:>7.2f}%  "
                f"{m['f1']:>6.4f}"
            )

        lines.append("=" * 60)

        # Feature weight summary (last round)
        last = self._metrics[-1]
        snap = last.get("weight_snapshot", {})
        if snap:
            lines.append("\nFeature weight changes (initial → final):")
            first_snap = self._metrics[0].get("weight_snapshot", {})
            for ctx in sorted(snap):
                lines.append(f"  Context: {ctx}")
                for feat in sorted(snap[ctx]):
                    w_init = first_snap.get(ctx, {}).get(feat, 1.0)
                    w_final = snap[ctx][feat]
                    pct = (w_final - w_init) / w_init * 100
                    arrow = "↓" if pct < 0 else "↑"
                    lines.append(
                        f"    {feat:6s}: {w_init:.4f} → {w_final:.4f}  "
                        f"({arrow}{abs(pct):.1f}%)"
                    )

        # Drift summary
        drift = last.get("drift", {})
        lines.append(
            f"\nDrift detector: drift_detected={drift.get('drift_detected')}, "
            f"current_fp_rate={drift.get('current_fp_rate'):.4f}, "
            f"baseline_fp_rate={drift.get('baseline_fp_rate'):.4f}"
        )

        report = "\n".join(lines)
        print(report)
        return report
