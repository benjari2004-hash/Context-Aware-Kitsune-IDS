# utils.py
# Shared helper functions for the Explainability Engine

def zscore(value, mean, std, min_std=1e-6):
    """Z-score of value relative to a baseline."""
    return (value - mean) / max(abs(std), min_std)


def deviation_label(z):
    """Convert z-score magnitude to a severity string."""
    az = abs(z)
    if az >= 3.0:   return "CRITICAL"
    elif az >= 2.5: return "HIGH"
    elif az >= 1.5: return "MEDIUM"
    elif az >= 0.8: return "LOW"
    return "NORMAL"


def direction_word(z):
    return "above" if z >= 0 else "below"


def pct_change(value, baseline, min_baseline=1e-6):
    return ((value - baseline) / max(abs(baseline), min_baseline)) * 100.0


def format_float(v, decimals=3):
    return f"{v:.{decimals}f}"


def top_k_indices(values, k):
    """Return indices of the k largest absolute values."""
    indexed = sorted(enumerate(values), key=lambda x: abs(x[1]), reverse=True)
    return [i for i, _ in indexed[:k]]


def seconds_ago_str(current_t, past_t):
    delta = max(0.0, current_t - past_t)
    if delta < 60:
        return f"{delta:.1f}s ago"
    return f"{int(delta // 60)}m {int(delta % 60)}s ago"


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))