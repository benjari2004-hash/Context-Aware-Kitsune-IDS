"""
Simulates SOC-analyst feedback for experimental evaluation.

Uses results.csv (pipeline output) and ground_truth.csv to determine whether
each anomaly packet is a true or false positive, then generates realistic
FeedbackRecord objects with configurable noise and response rates.
"""
import csv
import os
import random
import re
import tempfile

from .feedback_types import FeedbackRecord
from .feedback_store import FeedbackStore
from .feature_weight_learner import FeatureWeightLearner
from .adaptive_threshold import FeedbackAdaptiveThreshold
from .drift_detector import FeedbackDriftDetector

_CF_FEATURES = ("score", "risk", "freq")
_ATTACK_IP_PREFIX = {
    "Mirai C2 Communication": "192.168.100",
    "DoS / Flood": "10.0.50",
    "C2 Beacon": "172.16.0",
}


def _synthesize_ip(packet_id: int, attack_type: str) -> str:
    prefix = _ATTACK_IP_PREFIX.get(attack_type, "192.168.1")
    return f"{prefix}.{packet_id % 200 + 1}"


def _extract_cf_feature(cf_text) -> str:
    """Parse the CF text to find which feature was flagged. Falls back to 'risk'."""
    if not cf_text or str(cf_text).lower() in ("nan", "none", ""):
        return "risk"
    match = re.search(r"\b(score|risk|freq)\b", str(cf_text), re.IGNORECASE)
    return match.group(1).lower() if match else "risk"


def _load_results(path: str) -> list:
    """Load results.csv rows as list of dicts, skipping comment and TRAIN rows."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        # Skip leading comment lines (start with '#')
        peeked = []
        for raw in fh:
            if not raw.lstrip().startswith("#"):
                peeked.append(raw)
                break
        remaining = list(fh)
        import io
        content = io.StringIO("".join(peeked + remaining))
        reader = csv.DictReader(content)
        for row in reader:
            if row.get("label", "").upper() == "TRAIN":
                continue
            rows.append(row)
    return rows


def _load_ground_truth(path: str) -> dict:
    """Return {packet_id (int): true_label (str)} from ground_truth.csv."""
    gt = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    pid = int(parts[0].strip())
                    label = parts[1].strip().upper()
                    gt[pid] = label
                except ValueError:
                    pass
    return gt


def _compute_metrics(rows: list, ground_truth: dict, learner: FeatureWeightLearner,
                     fb_thresh: FeedbackAdaptiveThreshold) -> dict:
    """
    Recompute detection decisions on all rows using current learned state.
    Returns TP/FP/TN/FN counts and derived rates.
    """
    tp = fp = tn = fn = 0
    threshold = fb_thresh.get_threshold()

    for row in rows:
        pid = int(row["packet_id"])
        true_label = ground_truth.get(pid)
        if true_label is None:
            continue

        predicted = row["label"].upper()

        if predicted == "ANOMALY":
            context = row.get("attack_type", "") or "normal"
            cf_feat = _extract_cf_feature(row.get("cf_pass1", ""))
            try:
                raw_risk = float(row["risk"])
            except (ValueError, KeyError):
                raw_risk = 0.0
            adj_risk = learner.get_adjusted_risk(raw_risk, context, [cf_feat])
            if adj_risk < threshold:
                predicted = "NORMAL"

        if true_label == "ANOMALY" and predicted == "ANOMALY":
            tp += 1
        elif true_label == "NORMAL" and predicted == "ANOMALY":
            fp += 1
        elif true_label == "NORMAL" and predicted == "NORMAL":
            tn += 1
        else:  # true ANOMALY, predicted NORMAL
            fn += 1

    total_neg = fp + tn
    total_pos = tp + fn
    fp_rate = fp / total_neg if total_neg else 0.0
    fn_rate = fn / total_pos if total_pos else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fp_rate": fp_rate, "fn_rate": fn_rate,
        "precision": precision, "recall": recall, "f1": f1,
    }


class FeedbackSimulator:
    """
    Simulates analyst feedback for experimental evaluation.

    Uses results.csv (pipeline output) and ground_truth.csv to determine
    whether each anomaly packet is a TP or FP, then generates FeedbackRecord
    objects with configurable noise and response rates.
    """

    def __init__(
        self,
        results_path="results.csv",
        ground_truth_path="ground_truth.csv",
        noise_rate=0.05,
        response_rate=0.30,
        seed=42,
    ):
        self.noise_rate = noise_rate
        self.response_rate = response_rate
        self._rng = random.Random(seed)

        self._results = []
        self._anomaly_rows = []
        self._ground_truth: dict = {}

        if os.path.exists(results_path):
            self._results = _load_results(results_path)
            self._anomaly_rows = [r for r in self._results
                                  if r.get("label", "").upper() == "ANOMALY"]

        if os.path.exists(ground_truth_path):
            self._ground_truth = _load_ground_truth(ground_truth_path)

    # ------------------------------------------------------------------
    # Single-round feedback generation
    # ------------------------------------------------------------------

    def generate_feedback(
        self,
        num_feedbacks=500,
        analyst_id="simulated_analyst_1",
    ) -> list:
        """
        Sample anomaly packets (honouring response_rate) and produce
        FeedbackRecord objects.

        For each sampled packet:
          1. Check ground truth to determine if it is a TP or FP.
          2. With probability (1-noise_rate): give the correct label.
             With probability noise_rate: give the wrong label.
          3. Extract CF feature from cf_pass1 (or pick randomly).
          4. If FP → cf_accepted=False (feature is normal).
             If TP → cf_accepted=True  (feature is genuinely anomalous).
        """
        if not self._anomaly_rows:
            return []

        pool = [r for r in self._anomaly_rows
                if int(r["packet_id"]) in self._ground_truth]
        if not pool:
            return []

        n = min(num_feedbacks, int(len(pool) * self.response_rate) + 1, len(pool))
        sampled = self._rng.sample(pool, n)

        records = []
        for row in sampled:
            pid = int(row["packet_id"])
            attack_type = row.get("attack_type", "") or ""
            true_label = self._ground_truth.get(pid, "ANOMALY")

            is_fp = (true_label == "NORMAL")
            if self._rng.random() < self.noise_rate:
                is_fp = not is_fp  # analyst makes a mistake

            feedback_type = "fp" if is_fp else "tp"
            cf_feat = _extract_cf_feature(row.get("cf_pass1", ""))
            cf_accepted = not is_fp  # FP → feature is normal; TP → feature is anomalous

            rec = FeedbackRecord(
                packet_id=pid,
                entity_ip=_synthesize_ip(pid, attack_type),
                original_action=row.get("action", "UNKNOWN"),
                feedback_type=feedback_type,
                cf_feature=cf_feat,
                cf_accepted=cf_accepted,
                analyst_id=analyst_id,
                confidence=round(self._rng.uniform(0.6, 1.0), 2),
                notes=f"auto-simulated | attack_type={attack_type or 'none'}",
            )
            records.append(rec)

        return records

    # ------------------------------------------------------------------
    # Multi-round simulation (self-contained learning loop)
    # ------------------------------------------------------------------

    def run_simulation(
        self,
        rounds=10,
        feedbacks_per_round=50,
    ) -> list:
        """
        Run N rounds of feedback, applying learning after each round.
        Returns a list of per-round metric dicts (round 0 = baseline).
        """
        if not self._results or not self._ground_truth:
            print("[FeedbackSimulator] Missing results or ground truth — skipping.")
            return []

        metrics_per_round = []
        drift_detector = FeedbackDriftDetector()

        with tempfile.TemporaryDirectory() as tmpdir:
            learner = FeatureWeightLearner(
                adjustment_path=os.path.join(tmpdir, "adj.json")
            )
            fb_thresh = FeedbackAdaptiveThreshold()
            store = FeedbackStore(path=os.path.join(tmpdir, "log.jsonl"))

            # Round 0 — baseline (no learning applied yet)
            baseline = _compute_metrics(self._results, self._ground_truth,
                                        learner, fb_thresh)
            baseline.update({
                "round": 0,
                "threshold": fb_thresh.get_threshold(),
                "weight_snapshot": learner.get_weights_snapshot(),
                "drift": drift_detector.get_status(),
            })
            metrics_per_round.append(baseline)

            for rnd in range(1, rounds + 1):
                feedbacks = self.generate_feedback(
                    num_feedbacks=feedbacks_per_round,
                    analyst_id=f"sim_analyst_{rnd}",
                )
                for fb in feedbacks:
                    context = fb.notes.split("attack_type=")[-1] or "normal"
                    learner.process_feedback(fb, context)
                    fb_thresh.update(fb)
                    drift_detector.update(fb)
                    store.record(fb)

                m = _compute_metrics(self._results, self._ground_truth,
                                     learner, fb_thresh)
                m.update({
                    "round": rnd,
                    "threshold": fb_thresh.get_threshold(),
                    "weight_snapshot": learner.get_weights_snapshot(),
                    "drift": drift_detector.get_status(),
                })
                metrics_per_round.append(m)

        return metrics_per_round
