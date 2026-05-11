from .severity_calculator import calculate_severity
from .action_matrix import select_action, ATTACK_OVERRIDES

SEVERITY_LABELS = {
    0: "BENIGN",
    1: "SUSPICIOUS",
    2: "LOW",
    3: "MEDIUM",
    4: "HIGH",
    5: "CRITICAL",
}


def make_decision(score, risk, attack_type, freq):
    """
    Main entry point for the decision layer.
    Returns dict: {
      "severity": int 0-5,
      "severity_label": str,
      "action": str,
      "reason": str (1-2 sentences)
    }
    """
    severity = calculate_severity(score, risk, attack_type, freq)
    severity_label = SEVERITY_LABELS[severity]
    action = select_action(severity, attack_type)

    if attack_type in ATTACK_OVERRIDES:
        reason = (
            f"Attack type '{attack_type}' triggers mandatory {action} override. "
            f"Score {score:.3f} yields base severity {severity_label}."
        )
    else:
        reason = (
            f"Score {score:.3f} and risk {risk:.3f} map to {severity_label} severity. "
            f"Action {action} selected from severity matrix."
        )

    return {
        "severity":       severity,
        "severity_label": severity_label,
        "action":         action,
        "reason":         reason,
    }
