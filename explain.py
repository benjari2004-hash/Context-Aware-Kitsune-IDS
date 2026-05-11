def explain(score, ctx):
    reasons = []

    if score > 1:
        reasons.append("High anomaly")

    if ctx["endpoint"] in ("/ssh", "/secure"):
        reasons.append("Sensitive service")

    if ctx["freq"] > 20:
        reasons.append("High request rate")

    return reasons