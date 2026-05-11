import json
import os
import time

from .feedback_types import FeedbackRecord


class FeedbackStore:
    """Append-only JSONL audit trail of analyst feedback."""

    def __init__(self, path="feedback_log.jsonl"):
        self.path = path

    def record(self, feedback: FeedbackRecord):
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(feedback.to_dict()) + "\n")

    def get_all(self):
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(FeedbackRecord.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        pass
        return records

    def get_by_entity(self, ip):
        return [r for r in self.get_all() if r.entity_ip == ip]

    def get_recent(self, hours=24):
        cutoff = time.time() - hours * 3600
        return [r for r in self.get_all() if r.timestamp >= cutoff]

    def summary(self):
        counts = {"fp": 0, "tp": 0, "fn": 0}
        for r in self.get_all():
            ft = r.feedback_type
            counts[ft] = counts.get(ft, 0) + 1
        return counts
