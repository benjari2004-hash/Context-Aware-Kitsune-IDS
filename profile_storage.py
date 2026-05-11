# profile_storage.py

import json

def save_profiles(profiles, filename="profiles.json"):
    with open(filename, "w") as f:
        json.dump(profiles, f)


def load_profiles(filename="profiles.json"):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except:
        return {}