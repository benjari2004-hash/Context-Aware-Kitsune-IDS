from dataclasses import dataclass, asdict
import time


@dataclass
class FeedbackRecord:
    packet_id: int
    entity_ip: str
    original_action: str
    feedback_type: str   # "fp", "tp", "fn"
    cf_feature: str      # which CF feature analyst reviewed (score / risk / freq)
    cf_accepted: bool    # True = CF correct; False = feature is actually normal for context
    analyst_id: str
    confidence: float    # 0.0 to 1.0
    timestamp: float = None
    notes: str = ""

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
