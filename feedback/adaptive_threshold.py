from .feedback_types import FeedbackRecord


class FeedbackAdaptiveThreshold:
    """
    Adjusts the hard_override risk threshold based on the human feedback stream.

    Different from my_ids/adaptive_threshold.py, which adapts to the data stream.
    This class adapts exclusively to analyst FP/FN signals.

    FP feedback → threshold rises  (system is too sensitive; raise the bar)
    FN feedback → threshold falls  (system misses attacks; lower the bar)
    TP feedback → no change        (system is correct)

    Changes are bounded per step to resist feedback poisoning.
    """

    def __init__(
        self,
        initial=1.5,
        learning_rate=0.01,
        min_threshold=0.5,
        max_threshold=3.0,
        max_change_per_step=0.05,
    ):
        self.threshold = initial
        self.learning_rate = learning_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.max_change_per_step = max_change_per_step
        self._history = [initial]

    def update(self, feedback: FeedbackRecord):
        confidence = max(0.0, min(1.0, feedback.confidence))
        delta = min(self.learning_rate * confidence, self.max_change_per_step)

        if feedback.feedback_type == "fp":
            self.threshold = min(self.max_threshold, self.threshold + delta)
        elif feedback.feedback_type == "fn":
            self.threshold = max(self.min_threshold, self.threshold - delta)
        # "tp": no change

        self._history.append(self.threshold)

    def get_threshold(self):
        return self.threshold

    def get_history(self):
        return list(self._history)
