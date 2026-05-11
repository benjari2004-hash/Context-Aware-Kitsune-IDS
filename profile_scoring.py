# SEE config.py → profile_fusion_alpha for rationale
def combine_scores(score, deviation, alpha=0.7, max_deviation=10.0):
    norm_deviation = min(deviation / max_deviation, 1.0)
    raw = alpha * score + (1 - alpha) * norm_deviation
    return min(max(raw, 0.0), 1.0)