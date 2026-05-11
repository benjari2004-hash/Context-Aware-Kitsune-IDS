ACTION_MAP = {
    0: "ALLOW",
    1: "MONITOR",
    2: "ALERT_LOW",
    3: "ALERT_HIGH",
    4: "RATE_LIMIT",
    5: "BLOCK",
}

ATTACK_OVERRIDES = {
    "Mirai C2 Communication": "BLOCK",
    "DoS / Flood":            "RATE_LIMIT",
    "C2 Beacon":              "ALERT_HIGH",
}


def select_action(severity, attack_type):
    """
    Returns action string. Attack overrides take priority over severity matrix.
    """
    if attack_type in ATTACK_OVERRIDES:
        return ATTACK_OVERRIDES[attack_type]
    return ACTION_MAP.get(severity, "BLOCK")
