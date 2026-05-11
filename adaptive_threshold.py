import math
import statistics
from collections import deque


class AdaptiveThreshold:
    def __init__(self, window=100, mode="balanced", warmup=30, std_cap=None,
                 min_threshold=0.1, max_threshold=None):  # SEE config.py → adaptive_max_threshold_normal for rationale

        if mode == "high_security":
            k = 1.6
        elif mode == "balanced":
            k = 1.8
        elif mode == "low_noise":
            k = 2.5
        else:
            k = 1.8

        if window < 1:
            raise ValueError("window must be at least 1")
        if warmup < 0:
            raise ValueError("warmup must be at least 0")
        if std_cap is not None and std_cap < 0:
            raise ValueError("std_cap must be at least 0")
        if max_threshold is not None and min_threshold > max_threshold:
            raise ValueError("min_threshold cannot be greater than max_threshold")

        self.k            = float(k)
        self.warmup       = int(warmup)
        self.std_cap      = None if std_cap is None else float(std_cap)
        self.min_threshold = float(min_threshold)
        self.max_threshold = None if max_threshold is None else float(max_threshold)
        self.scores        = deque(maxlen=int(window))
        self.last_mean     = 0.0
        self.last_raw_std  = 0.0
        self.last_std      = 0.0
        self.last_threshold = self.min_threshold

        # margin: fraction of threshold added as dead-zone
        # a score must exceed threshold * (1 + margin) to be ANOMALY
        self.margin = 0.10   # 10 % dead-zone — suppresses borderline crossings

    def seed(self, values):
        """Seed with adjusted_risk samples collected during training."""
        for v in values:
            if math.isfinite(v):
                self.scores.append(float(v))

    def update(self, score):
        if math.isfinite(score):
            self.scores.append(float(score))

    def _bounded_scores(self):
        values = list(self.scores)
        if not values:
            return values
        median     = statistics.median(values)
        deviations = [abs(v - median) for v in values]
        mad        = statistics.median(deviations)
        robust_std = 1.4826 * mad
        spread     = max(robust_std, self.min_threshold, 1e-9)
        lower      = max(0.0, median - 6.0 * spread)
        upper      = median + 6.0 * spread
        return [min(max(v, lower), upper) for v in values]

    def _snapshot(self):
        if not self.scores:
            self.last_mean      = 0.0
            self.last_raw_std   = 0.0
            self.last_std       = 0.0
            self.last_threshold = self.min_threshold
            return 0.0, 0.0, 0.0, self.min_threshold

        bounded = self._bounded_scores()
        mean    = statistics.fmean(bounded)
        raw_std = statistics.pstdev(bounded) if len(bounded) > 1 else 0.0
        std     = raw_std if self.std_cap is None else min(raw_std, self.std_cap)

        # threshold is valid from the very first packet after seeding
        threshold = mean + self.k * std
        if not math.isfinite(threshold):
            threshold = self.last_threshold
        if self.max_threshold is not None:
            threshold = min(threshold, self.max_threshold)
        threshold = max(threshold, self.min_threshold)

        self.last_mean      = mean
        self.last_raw_std   = raw_std
        self.last_std       = std
        self.last_threshold = threshold
        return mean, raw_std, std, threshold

    def get_threshold(self):
        return self._snapshot()[3]

    def classify(self, score):
        value = float(score)
        if not math.isfinite(value):
            return "ANOMALY"
        _, _, _, threshold = self._snapshot()
        # dead-zone: must exceed threshold + margin to avoid borderline FP
        effective = threshold * (1.0 + self.margin)
        return "ANOMALY" if value > effective else "NORMAL"