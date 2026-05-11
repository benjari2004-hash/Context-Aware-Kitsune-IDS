from collections import deque

from .feedback_types import FeedbackRecord

# Minimum observations before a baseline is established.
_BASELINE_MIN = 50


class FeedbackDriftDetector:
    """
    Detects concept drift using the analyst FP rate as signal.

    If the FP rate in the recent window rises more than `threshold` above the
    established baseline, drift is declared.  This is novel: human judgment
    serves as the drift signal rather than data distribution alone.
    """

    def __init__(self, window=200, threshold=0.15):
        self.window = window
        self.threshold = threshold
        self.recent: deque = deque(maxlen=window)
        self.baseline_fp_rate = None
        self._total_seen = 0

    def update(self, feedback: FeedbackRecord) -> bool:
        """
        Incorporate one feedback record.
        Returns True when drift is detected, False otherwise.
        """
        self.recent.append(1 if feedback.feedback_type == "fp" else 0)
        self._total_seen += 1

        if self.baseline_fp_rate is None and len(self.recent) >= _BASELINE_MIN:
            self.baseline_fp_rate = self._current_fp_rate()

        return self._is_drifting()

    def _current_fp_rate(self):
        if not self.recent:
            return 0.0
        return sum(self.recent) / len(self.recent)

    def _is_drifting(self):
        if self.baseline_fp_rate is None:
            return False
        delta = self._current_fp_rate() - self.baseline_fp_rate
        return delta > self.threshold

    def get_status(self):
        current = self._current_fp_rate()
        baseline = self.baseline_fp_rate if self.baseline_fp_rate is not None else 0.0
        delta = current - baseline
        return {
            "drift_detected": self._is_drifting(),
            "current_fp_rate": round(current, 4),
            "baseline_fp_rate": round(baseline, 4),
            "delta": round(delta, 4),
            "observations": self._total_seen,
            "window_size": len(self.recent),
        }
