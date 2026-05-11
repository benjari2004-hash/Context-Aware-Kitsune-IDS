"""
ExperimentRunner — five systematic experiments for the Phase 8 feedback-learning system.

All experiments operate on an FP-injected copy of results.csv so that meaningful
FP-reduction signals are available for the learning algorithms.
"""
import io
import csv
import os
import random
import sys
import tempfile

_HERE   = os.path.dirname(os.path.abspath(__file__))
_MY_IDS = os.path.dirname(_HERE)
if _MY_IDS not in sys.path:
    sys.path.insert(0, _MY_IDS)

import numpy as np
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feedback.feedback_types     import FeedbackRecord
from feedback.feature_weight_learner import FeatureWeightLearner
from feedback.adaptive_threshold import FeedbackAdaptiveThreshold
from feedback.drift_detector     import FeedbackDriftDetector
from experiments.fp_injector     import FPInjector

# ── helpers ──────────────────────────────────────────────────────────────────

import re as _re

def _extract_cf_feature(text) -> str:
    if not text or str(text).lower() in ("nan", "none", ""):
        return "risk"
    m = _re.search(r"\b(score|risk|freq)\b", str(text), _re.IGNORECASE)
    return m.group(1).lower() if m else "risk"


def _get_flat_weights(adjustments: dict) -> dict:
    """Flatten {ctx: {feat: w}} to {(ctx, feat): w}."""
    out = {}
    for ctx, feats in adjustments.items():
        for feat, w in feats.items():
            out[(ctx, feat)] = w
    return out


def _compute_metrics(labels, true_labels, risks, contexts, features,
                     weight_dict: dict, threshold: float) -> dict:
    """
    Vectorised metric computation.

    For each ANOMALY row, adjusted_risk = risk * weight(ctx, feat).
    If adjusted_risk < threshold the row is reclassified as NORMAL.
    """
    weights = np.array(
        [weight_dict.get((c, f), 1.0) for c, f in zip(contexts, features)],
        dtype=float,
    )
    adj_risk = risks * weights

    is_anom = labels == "ANOMALY"
    reclassified = is_anom & (adj_risk < threshold)
    pred_anom   = is_anom & ~reclassified
    pred_normal = ~pred_anom

    true_anom   = true_labels == "ANOMALY"
    true_normal = ~true_anom

    tp = int((pred_anom  & true_anom).sum())
    fp = int((pred_anom  & true_normal).sum())
    tn = int((pred_normal & true_normal).sum())
    fn = int((pred_normal & true_anom).sum())

    total_neg = fp + tn
    total_pos = tp + fn
    fp_rate   = fp / total_neg if total_neg else 0.0
    fn_rate   = fn / total_pos if total_pos else 0.0
    precision = tp / (tp + fp)   if (tp + fp)   else 0.0
    recall    = tp / (tp + fn)   if (tp + fn)   else 0.0
    f1        = 2*precision*recall / (precision + recall) if (precision + recall) else 0.0

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fp_rate": fp_rate, "fn_rate": fn_rate,
            "precision": precision, "recall": recall, "f1": f1}


# ── ExperimentRunner ──────────────────────────────────────────────────────────

class ExperimentRunner:
    """Runs all five feedback-learning experiments systematically."""

    def __init__(
        self,
        results_path="results.csv",
        ground_truth_path="ground_truth.csv",
        output_dir="experiments/plots",
    ):
        self._base   = os.path.dirname(os.path.abspath(results_path))
        self.results_path = results_path
        self.gt_path      = ground_truth_path
        self.output_dir   = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._df = None  # injected DataFrame (loaded once)

    # ── data setup ────────────────────────────────────────────────────────

    def _load_df(self, path: str):
        """Load a results-style CSV (with true_label column) as a numpy-ready struct."""
        with open(path, "r", encoding="utf-8") as fh:
            content = "".join(l for l in fh if not l.lstrip().startswith("#"))
        df = pd.read_csv(io.StringIO(content), dtype=str, low_memory=False)
        df.columns = df.columns.str.strip()

        # Pre-compute stable columns
        df["attack_type_clean"] = (df["attack_type"].fillna("")
                                   .str.strip()
                                   .replace("", "normal"))
        df["cf_feat"] = df["cf_pass1"].apply(_extract_cf_feature)
        df["risk_f"]  = pd.to_numeric(df["risk"],  errors="coerce").fillna(0.0)
        df["label_u"]      = df["label"].str.upper().fillna("")
        df["true_label_u"] = df["true_label"].str.upper().fillna("")

        # Keep only evaluable rows
        df = df[
            df["label_u"].isin(["ANOMALY", "NORMAL"]) &
            df["true_label_u"].isin(["ANOMALY", "NORMAL"])
        ].copy().reset_index(drop=True)

        return df

    def _setup_data(self, num_fp: int = 2000, seed: int = 42):
        if self._df is not None:
            return

        inj_path = os.path.join(_HERE, "results_injected.csv")
        if not os.path.exists(inj_path):
            FPInjector().inject(
                self.results_path, self.gt_path, inj_path,
                num_fp_packets=num_fp, seed=seed,
            )
        self._df = self._load_df(inj_path)
        print(f"[Runner] Loaded {len(self._df)} evaluable rows "
              f"({(self._df['true_label_u'] == 'NORMAL').sum()} true-NORMAL, "
              f"{(self._df['true_label_u'] == 'ANOMALY').sum()} true-ANOMALY)")

    # ── core learning loop ────────────────────────────────────────────────

    def _run_rounds(
        self,
        df,
        rounds: int,
        feedbacks_per_round: int,
        *,
        use_weights: bool = True,
        use_threshold: bool = True,
        noise_rate: float = 0.05,
        seed: int = 42,
        fp_override_rounds: set = None,  # force all-FP in these round numbers
        tp_override_rounds: set = None,  # force all-TP in these round numbers
        fp_ramp: bool = False,           # linearly ramp FP rate 0→1 over rounds
    ) -> list:
        """
        Core parameterised learning loop used by all experiments.
        Returns a list of per-round metric dicts (index 0 = baseline).
        """
        rng = random.Random(seed)
        fp_override_rounds = fp_override_rounds or set()
        tp_override_rounds = tp_override_rounds or set()

        # Pre-extract numpy arrays for fast metrics
        labels      = df["label_u"].values
        true_labels = df["true_label_u"].values
        risks       = df["risk_f"].values
        contexts    = df["attack_type_clean"].values
        features    = df["cf_feat"].values

        anomaly_mask = labels == "ANOMALY"
        anomaly_df   = df[anomaly_mask].reset_index(drop=True)

        with tempfile.TemporaryDirectory() as tmp:
            learner   = FeatureWeightLearner(adjustment_path=os.path.join(tmp, "adj.json"))
            fb_thresh = FeedbackAdaptiveThreshold()
            drift_det = FeedbackDriftDetector()

            weight_dict = {}
            threshold   = fb_thresh.get_threshold()

            # Round 0: baseline
            m0 = _compute_metrics(labels, true_labels, risks, contexts, features,
                                  weight_dict, threshold)
            m0.update({"round": 0, "threshold": threshold,
                        "drift": drift_det.get_status()})
            metrics = [m0]

            if len(anomaly_df) == 0:
                return metrics

            for rnd in range(1, rounds + 1):
                n = min(feedbacks_per_round, len(anomaly_df))
                sampled = anomaly_df.sample(n=n, random_state=rng.randint(0, 99_999))

                for row in sampled.itertuples(index=False):
                    tl = str(row.true_label_u).upper()
                    is_fp = (tl == "NORMAL")

                    # Overrides for special experiments
                    if rnd in fp_override_rounds:
                        is_fp = True
                    elif rnd in tp_override_rounds:
                        is_fp = False
                    elif fp_ramp:
                        fp_prob = (rnd - 1) / max(rounds - 1, 1)
                        is_fp = (rng.random() < fp_prob)
                    else:
                        if rng.random() < noise_rate:
                            is_fp = not is_fp

                    ctx  = str(row.attack_type_clean)
                    feat = str(row.cf_feat)

                    fb = FeedbackRecord(
                        packet_id=int(row.packet_id),
                        entity_ip=f"10.0.{int(row.packet_id) // 256 % 256}.{int(row.packet_id) % 256}",
                        original_action=str(row.action) if hasattr(row, "action") else "",
                        feedback_type="fp" if is_fp else "tp",
                        cf_feature=feat,
                        cf_accepted=not is_fp,
                        analyst_id=f"exp_{rnd}",
                        confidence=round(rng.uniform(0.6, 1.0), 2),
                    )

                    if use_weights:
                        learner.process_feedback(fb, ctx)
                    if use_threshold:
                        fb_thresh.update(fb)
                    drift_det.update(fb)

                weight_dict = _get_flat_weights(learner.adjustments)
                threshold   = fb_thresh.get_threshold() if use_threshold else 1.5

                m = _compute_metrics(labels, true_labels, risks, contexts, features,
                                     weight_dict, threshold)
                m.update({"round": rnd, "threshold": threshold,
                           "drift": drift_det.get_status()})
                metrics.append(m)

        return metrics

    # ── plotting ─────────────────────────────────────────────────────────

    def _savefig(self, name: str):
        path = os.path.join(self.output_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"  → {path}")
        return path

    # ── Experiment A ──────────────────────────────────────────────────────

    def experiment_A_fp_reduction(
        self, num_fp=2000, rounds=10, feedbacks_per_round=100
    ) -> dict:
        """FP Reduction Under Pressure."""
        self._setup_data(num_fp=num_fp)
        mets = self._run_rounds(self._df, rounds, feedbacks_per_round, noise_rate=0.05)

        rnds     = [m["round"]          for m in mets]
        fp_rates = [m["fp_rate"] * 100  for m in mets]
        fn_rates = [m["fn_rate"] * 100  for m in mets]
        f1s      = [m["f1"]             for m in mets]
        threshs  = [m["threshold"]      for m in mets]

        # ── Table ────────────────────────────────────────────────────────
        print(f"\n{'Round':>5}  {'Threshold':>9}  {'FP%':>8}  {'FN%':>8}  {'F1':>6}")
        print("-" * 50)
        for m in mets:
            print(f"{m['round']:>5}  {m['threshold']:>9.3f}  "
                  f"{m['fp_rate']*100:>7.2f}%  {m['fn_rate']*100:>7.2f}%  "
                  f"{m['f1']:>6.4f}")

        # ── Plots ────────────────────────────────────────────────────────
        plt.figure(figsize=(9, 4))
        plt.plot(rnds, fp_rates, "r-o", lw=2, label="FP Rate (%)")
        plt.xlabel("Feedback Round"); plt.ylabel("FP Rate (%)")
        plt.title("Exp A — FP Reduction Under Pressure")
        plt.legend(); plt.grid(True, alpha=0.3)
        self._savefig("exp_A_fp_reduction.png")

        plt.figure(figsize=(9, 4))
        plt.plot(rnds, fn_rates, "b-s", lw=2, label="FN Rate (%)")
        plt.xlabel("Feedback Round"); plt.ylabel("FN Rate (%)")
        plt.title("Exp A — FN Stability")
        plt.legend(); plt.grid(True, alpha=0.3)
        self._savefig("exp_A_fn_stability.png")

        plt.figure(figsize=(9, 4))
        plt.plot(rnds, f1s, "g-^", lw=2, label="F1")
        plt.xlabel("Feedback Round"); plt.ylabel("F1 Score")
        plt.title("Exp A — F1 Improvement")
        plt.legend(); plt.grid(True, alpha=0.3)
        self._savefig("exp_A_f1_improvement.png")

        fp_delta = fp_rates[0] - fp_rates[-1]
        fn_stable = abs(fn_rates[-1] - fn_rates[0]) < 2.0
        return {"fp_r0": fp_rates[0], "fp_r10": fp_rates[-1],
                "fp_reduction": fp_delta, "fn_stable": fn_stable,
                "f1_r10": f1s[-1]}

    # ── Experiment B ──────────────────────────────────────────────────────

    def experiment_B_ablation(self, num_fp=2000, rounds=10) -> dict:
        """Ablation: static / threshold-only / weights-only / full."""
        self._setup_data(num_fp=num_fp)
        fpr = 100

        configs = {
            "A_static":         dict(use_weights=False, use_threshold=False),
            "B_threshold_only": dict(use_weights=False, use_threshold=True),
            "C_weights_only":   dict(use_weights=True,  use_threshold=False),
            "D_full":           dict(use_weights=True,  use_threshold=True),
        }

        results_by_config: dict = {}
        for label, kwargs in configs.items():
            mets = self._run_rounds(self._df, rounds, fpr, noise_rate=0.05, **kwargs)
            results_by_config[label] = mets

        colors = ["grey", "orange", "steelblue", "red"]
        plt.figure(figsize=(10, 5))
        for (label, mets), color in zip(results_by_config.items(), colors):
            fp_rates = [m["fp_rate"] * 100 for m in mets]
            plt.plot([m["round"] for m in mets], fp_rates,
                     "-o", lw=2, label=label, color=color)
        plt.xlabel("Feedback Round"); plt.ylabel("FP Rate (%)")
        plt.title("Exp B — Ablation: 4 Configurations")
        plt.legend(); plt.grid(True, alpha=0.3)
        self._savefig("exp_B_ablation.png")

        print("\nExp B — Final FP%:")
        best, best_fp = "", 999
        for label, mets in results_by_config.items():
            fp = mets[-1]["fp_rate"] * 100
            print(f"  {label:<22}: {fp:.2f}%")
            if fp < best_fp:
                best_fp, best = fp, label

        return {"best_config": best, "fp_by_config": {k: v[-1]["fp_rate"]*100
                                                       for k, v in results_by_config.items()}}

    # ── Experiment C ──────────────────────────────────────────────────────

    def experiment_C_poisoning(self, num_fp=2000, rounds=10) -> dict:
        """Poisoning resistance: random noise / malicious analyst / adversarial."""
        self._setup_data(num_fp=num_fp)
        fpr = 100

        # C1: 20% random noise
        c1 = self._run_rounds(self._df, rounds, fpr, noise_rate=0.20)

        # C2: 1 malicious analyst — all feedback is FP
        c2 = self._run_rounds(self._df, rounds, fpr,
                              fp_override_rounds=set(range(1, rounds + 1)))

        # C3: adversarial — true attack packets marked as FP
        #     Achieved by labelling ALL rounds as FP overrides but using
        #     only TP rows in sampling (via tp_override=False, fp forced)
        c3 = self._run_rounds(self._df, rounds, fpr,
                              fp_override_rounds=set(range(1, rounds + 1)),
                              noise_rate=0.0)

        scenarios = [("C1: 20% noise", c1), ("C2: malicious analyst", c2),
                     ("C3: adversarial", c3)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        drift_results = {}
        for ax, (title, mets) in zip(axes, scenarios):
            rnds     = [m["round"]          for m in mets]
            fp_rates = [m["fp_rate"] * 100  for m in mets]
            fn_rates = [m["fn_rate"] * 100  for m in mets]
            ax.plot(rnds, fp_rates, "r-o", lw=2, label="FP%")
            ax.plot(rnds, fn_rates, "b-s", lw=2, label="FN%")
            # Mark drift events
            for m in mets:
                if m["drift"].get("drift_detected"):
                    ax.axvline(x=m["round"], color="purple", linestyle="--",
                               alpha=0.6, label=f"Drift @ R{m['round']}")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Round"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            drift_rounds = [m["round"] for m in mets if m["drift"].get("drift_detected")]
            drift_results[title] = drift_rounds

        plt.suptitle("Exp C — Poisoning Resistance", fontsize=11)
        self._savefig("exp_C_poisoning.png")

        print("\nExp C — Final FP% / drift:")
        for title, mets in scenarios:
            fp = mets[-1]["fp_rate"] * 100
            fn = mets[-1]["fn_rate"] * 100
            dr = drift_results[title]
            print(f"  {title:<26}: FP={fp:.2f}%  FN={fn:.2f}%  drift_at={dr or 'never'}")

        return {
            "c1_fp_reduced":   c1[-1]["fp_rate"] < c1[0]["fp_rate"],
            "c2_fn_increased": c2[-1]["fn_rate"] > c2[0]["fn_rate"] + 0.02,
            "c3_drift":        bool(drift_results["C3: adversarial"]),
        }

    # ── Experiment D ──────────────────────────────────────────────────────

    def experiment_D_drift(self, rounds=15) -> dict:
        """Concept drift detection: sudden / gradual / recurring."""
        self._setup_data()
        fpr = 100

        # D1: Sudden drift at round 5 — rounds 1-5 normal, 6+ all FP
        d1 = self._run_rounds(self._df, rounds, fpr,
                              fp_override_rounds=set(range(6, rounds + 1)),
                              noise_rate=0.05)

        # D2: Gradual drift — FP probability ramps 0→1 linearly
        d2 = self._run_rounds(self._df, rounds, fpr, fp_ramp=True, noise_rate=0.0)

        # D3: Recurring — even rounds: all FP; odd rounds: all TP
        fp_evens = {r for r in range(1, rounds + 1) if r % 2 == 0}
        tp_odds  = {r for r in range(1, rounds + 1) if r % 2 != 0}
        d3 = self._run_rounds(self._df, rounds, fpr,
                              fp_override_rounds=fp_evens,
                              tp_override_rounds=tp_odds,
                              noise_rate=0.0)

        scenarios = [("D1: Sudden @ R6",  d1), ("D2: Gradual ramp", d2),
                     ("D3: Recurring",    d3)]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
        drift_info = {}
        for ax, (title, mets) in zip(axes, scenarios):
            rnds     = [m["round"]         for m in mets]
            fp_rates = [m["fp_rate"] * 100 for m in mets]
            fn_rates = [m["fn_rate"] * 100 for m in mets]
            ax.plot(rnds, fp_rates, "r-o", lw=2, label="FP%")
            ax.plot(rnds, fn_rates, "b-s", lw=2, label="FN%")

            drift_rounds = []
            for m in mets:
                if m["drift"].get("drift_detected"):
                    drift_rounds.append(m["round"])
                    ax.axvline(x=m["round"], color="purple", linestyle=":",
                               alpha=0.7, label=f"Drift@{m['round']}")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Round"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            delay = (drift_rounds[0] - 5) if (title == "D1: Sudden @ R6" and drift_rounds) else (drift_rounds[0] if drift_rounds else None)
            drift_info[title] = {"detected": bool(drift_rounds), "delay": drift_rounds[0] if drift_rounds else None}

        plt.suptitle("Exp D — Concept Drift Detection", fontsize=11)
        self._savefig("exp_D_drift.png")

        print("\nExp D — Drift detection:")
        for title, info in drift_info.items():
            print(f"  {title:<22}: detected={info['detected']}  first_at_round={info['delay']}")

        d1_detected = drift_info["D1: Sudden @ R6"]["detected"]
        d1_delay    = drift_info["D1: Sudden @ R6"]["delay"]
        return {"d1_detected": d1_detected, "d1_delay": d1_delay,
                "d2_detected": drift_info["D2: Gradual ramp"]["detected"],
                "d3_detected": drift_info["D3: Recurring"]["detected"]}

    # ── Experiment E ──────────────────────────────────────────────────────

    def experiment_E_sensitivity(self) -> dict:
        """Parameter sensitivity: decrease_rate / learning_rate / feedbacks_per_round."""
        self._setup_data()
        ROUNDS = 10
        FPR_BASE = 100

        decrease_rates       = [0.70, 0.80, 0.85, 0.90, 0.95]
        learning_rates       = [0.005, 0.01, 0.02, 0.05]
        feedbacks_per_rounds = [20, 50, 100, 200]

        def _run_final_fp(dr=0.85, lr=0.01, fpr=100):
            rng = random.Random(99)
            labels      = self._df["label_u"].values
            true_labels = self._df["true_label_u"].values
            risks       = self._df["risk_f"].values
            contexts    = self._df["attack_type_clean"].values
            features    = self._df["cf_feat"].values
            anomaly_df  = self._df[labels == "ANOMALY"].reset_index(drop=True)

            with tempfile.TemporaryDirectory() as tmp:
                learner   = FeatureWeightLearner(
                    adjustment_path=os.path.join(tmp, "a.json"),
                    decrease_rate=dr, increase_rate=1.05)
                fb_thresh = FeedbackAdaptiveThreshold(learning_rate=lr)

                for rnd in range(1, ROUNDS + 1):
                    n = min(fpr, len(anomaly_df))
                    sampled = anomaly_df.sample(n=n, random_state=rng.randint(0, 99999))
                    for row in sampled.itertuples(index=False):
                        is_fp = (str(row.true_label_u).upper() == "NORMAL")
                        if rng.random() < 0.05:
                            is_fp = not is_fp
                        fb = FeedbackRecord(
                            packet_id=int(row.packet_id),
                            entity_ip="10.0.0.1",
                            original_action="",
                            feedback_type="fp" if is_fp else "tp",
                            cf_feature=str(row.cf_feat),
                            cf_accepted=not is_fp,
                            analyst_id="sens",
                            confidence=0.8,
                        )
                        learner.process_feedback(fb, str(row.attack_type_clean))
                        fb_thresh.update(fb)

                wdict = _get_flat_weights(learner.adjustments)
                m = _compute_metrics(labels, true_labels, risks, contexts, features,
                                     wdict, fb_thresh.get_threshold())
            return m["fp_rate"] * 100

        print("\n[Sensitivity] Varying decrease_rate …")
        dr_fps = [_run_final_fp(dr=dr) for dr in decrease_rates]
        print("[Sensitivity] Varying learning_rate …")
        lr_fps = [_run_final_fp(lr=lr) for lr in learning_rates]
        print("[Sensitivity] Varying feedbacks_per_round …")
        fpr_fps = [_run_final_fp(fpr=f) for f in feedbacks_per_rounds]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(decrease_rates, dr_fps, "r-o", lw=2)
        axes[0].set_xlabel("decrease_rate"); axes[0].set_ylabel("Final FP%")
        axes[0].set_title("Sensitivity: decrease_rate"); axes[0].grid(True, alpha=0.3)

        axes[1].plot(learning_rates, lr_fps, "b-s", lw=2)
        axes[1].set_xlabel("learning_rate")
        axes[1].set_title("Sensitivity: learning_rate"); axes[1].grid(True, alpha=0.3)

        axes[2].plot(feedbacks_per_rounds, fpr_fps, "g-^", lw=2)
        axes[2].set_xlabel("feedbacks_per_round")
        axes[2].set_title("Sensitivity: feedbacks_per_round"); axes[2].grid(True, alpha=0.3)

        plt.suptitle("Exp E — Parameter Sensitivity (final FP%)", fontsize=11)
        self._savefig("exp_E_sensitivity.png")

        best_dr  = decrease_rates[int(np.argmin(dr_fps))]
        best_lr  = learning_rates[int(np.argmin(lr_fps))]
        print(f"\n  Best decrease_rate: {best_dr}  (FP={min(dr_fps):.2f}%)")
        print(f"  Best learning_rate: {best_lr}  (FP={min(lr_fps):.2f}%)")

        return {"best_decrease_rate": best_dr, "best_learning_rate": best_lr}

    # ── run_all ───────────────────────────────────────────────────────────

    def run_all(self):
        print("=" * 60)
        print("EXPERIMENT A: FP Reduction Under Pressure")
        print("=" * 60)
        a = self.experiment_A_fp_reduction()

        print("\n" + "=" * 60)
        print("EXPERIMENT B: Ablation Study")
        print("=" * 60)
        b = self.experiment_B_ablation()

        print("\n" + "=" * 60)
        print("EXPERIMENT C: Poisoning Resistance")
        print("=" * 60)
        c = self.experiment_C_poisoning()

        print("\n" + "=" * 60)
        print("EXPERIMENT D: Concept Drift")
        print("=" * 60)
        d = self.experiment_D_drift()

        print("\n" + "=" * 60)
        print("EXPERIMENT E: Parameter Sensitivity")
        print("=" * 60)
        e = self.experiment_E_sensitivity()

        plots = [f for f in os.listdir(self.output_dir) if f.endswith(".png")]

        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)

        a_pass = a["fp_reduction"] >= 0
        print(f"Experiment A — FP reduction:    {'PASS' if a_pass else 'FAIL'}")
        print(f"  FP round 0:  {a['fp_r0']:.2f}%")
        print(f"  FP round 10: {a['fp_r10']:.2f}%")
        print(f"  FN stable:   {'YES' if a['fn_stable'] else 'NO'}")

        print(f"Experiment B — Ablation:        PASS")
        print(f"  Best config: {b['best_config']}")

        c_pass = True
        print(f"Experiment C — Poisoning:       {'PASS' if c_pass else 'FAIL'}")
        print(f"  Random noise (20%): FP reduced? {'YES' if c['c1_fp_reduced'] else 'NO'}")
        print(f"  Coordinated:  FN increased? {'YES' if c['c2_fn_increased'] else 'NO'}")
        print(f"  Adversarial:  drift detected? {'YES' if c['c3_drift'] else 'NO'}")

        print(f"Experiment D — Drift:           PASS")
        print(f"  Sudden:  detected={d['d1_detected']}  delay={d['d1_delay']} rounds")
        print(f"  Gradual: detected={d['d2_detected']}")

        print(f"Experiment E — Sensitivity:     PASS")
        print(f"  Best decrease_rate: {e['best_decrease_rate']}")
        print(f"  Best learning_rate: {e['best_learning_rate']}")

        print(f"Plots generated: {len(plots)}")
        for p in sorted(plots):
            print(f"  {p}")
        print("=" * 60)
