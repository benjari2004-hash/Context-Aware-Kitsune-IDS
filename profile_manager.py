import time

profiles = {}

def get_profile(entity_id):
    if entity_id not in profiles:
        profiles[entity_id] = {
            "mean": 0.0,
            "count": 0,
            "last_seen": time.time()
        }
    return profiles[entity_id]


def update_profile(profile, value):
    profile["count"] += 1
    profile["mean"] += (value - profile["mean"]) / profile["count"]
    profile["last_seen"] = time.time()


def compute_deviation(profile, value):
    return abs(value - profile["mean"])