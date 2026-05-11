def calculate_severity(score, risk, attack_type, freq):
    """
    Returns severity level 0-5:
      0 BENIGN     — score < 0.15 AND risk < 0.5
      1 SUSPICIOUS — score 0.15-0.30 OR risk 0.5-1.0
      2 LOW        — score 0.30-0.50 OR risk 1.0-1.5
      3 MEDIUM     — score 0.50-0.70 OR risk 1.5-2.0
      4 HIGH       — score 0.70-0.90 OR risk 2.0-2.5
      5 CRITICAL   — score > 0.90 OR risk > 2.5
    Returns the higher of score-based or risk-based severity.
    """
    if score < 0.15:
        score_level = 0
    elif score < 0.30:
        score_level = 1
    elif score < 0.50:
        score_level = 2
    elif score < 0.70:
        score_level = 3
    elif score < 0.90:
        score_level = 4
    else:
        score_level = 5

    if risk < 0.5:
        risk_level = 0
    elif risk < 1.0:
        risk_level = 1
    elif risk < 1.5:
        risk_level = 2
    elif risk < 2.0:
        risk_level = 3
    elif risk < 2.5:
        risk_level = 4
    else:
        risk_level = 5

    return max(score_level, risk_level)
