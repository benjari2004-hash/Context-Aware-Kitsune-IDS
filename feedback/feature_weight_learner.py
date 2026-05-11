import json
import os

from .feedback_types import FeedbackRecord


class FeatureWeightLearner:
    """
    Learns per-feature, per-context weight adjustments from
    counterfactual-guided analyst feedback.

    When an analyst marks a CF feature as 'actually normal for this context'
    (cf_accepted=False on a FP), the feature weight is decreased for that context.
    When the analyst confirms the feature is anomalous (cf_accepted=True on a TP),
    the weight is slightly increased.

    Context is a string derived from attack_type (e.g. "Mirai C2 Communication").
    Adjustments are persisted as JSON: {context: {feature: weight}}.
    """

    FEATURES = ("score", "risk", "freq")

    def __init__(
        self,
        adjustment_path="feature_adjustments.json",
        decrease_rate=0.85,
        increase_rate=1.05,
        min_weight=0.3,
        max_weight=2.0,
    ):
        self.adjustment_path = adjustment_path
        self.decrease_rate = decrease_rate
        self.increase_rate = increase_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adjustments = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if os.path.exists(self.adjustment_path):
            try:
                with open(self.adjustment_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save(self):
        os.makedirs(os.path.dirname(self.adjustment_path) or ".", exist_ok=True)
        with open(self.adjustment_path, "w", encoding="utf-8") as fh:
            json.dump(self.adjustments, fh, indent=2)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def _ensure_context(self, context):
        if context not in self.adjustments:
            self.adjustments[context] = {}

    def _clamp(self, weight):
        return max(self.min_weight, min(self.max_weight, weight))

    def process_feedback(self, feedback: FeedbackRecord, context: str):
        """
        Update feature weight based on CF feedback.

        FP + cf_accepted=False → feature is normal for context → decrease weight.
        TP + cf_accepted=True  → feature is genuinely anomalous → increase weight.
        All other combinations are informative but not acted on to avoid noise.
        """
        self._ensure_context(context)
        feature = feedback.cf_feature
        if feature not in self.FEATURES:
            return

        current = self.adjustments[context].get(feature, 1.0)

        if feedback.feedback_type == "fp" and not feedback.cf_accepted:
            current = self._clamp(current * self.decrease_rate)
        elif feedback.feedback_type == "tp" and feedback.cf_accepted:
            current = self._clamp(current * self.increase_rate)

        self.adjustments[context][feature] = current
        self._save()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_adjusted_risk(self, raw_risk: float, context: str, features_used: list):
        """
        Return raw_risk scaled by the product of learned weights for the
        requested features in the given context.
        """
        if context not in self.adjustments:
            return raw_risk
        weight_product = 1.0
        for feature in features_used:
            weight_product *= self.adjustments[context].get(feature, 1.0)
        return raw_risk * weight_product

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_adjustment_summary(self):
        lines = []
        for context, feats in sorted(self.adjustments.items()):
            lines.append(f"Context: {context}")
            for feat, w in sorted(feats.items()):
                direction = "↓ reduced" if w < 1.0 else "↑ increased" if w > 1.0 else "→ unchanged"
                lines.append(f"  {feat:6s} weight: {w:.4f}  ({direction} from 1.0)")
        return "\n".join(lines) if lines else "(no adjustments learned yet)"

    def get_weights_snapshot(self):
        """Return a flat copy: {context: {feature: weight}}."""
        return {ctx: dict(feats) for ctx, feats in self.adjustments.items()}
