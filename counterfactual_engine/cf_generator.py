from decision_layer.decision_engine import make_decision
from decision_layer.action_matrix import ATTACK_OVERRIDES

# ── PASS 1 / PASS 2 search ranges (9 candidates per feature) ──────────────
PERTURBABLE_FEATURES = ["score", "risk", "freq"]

SEARCH_RANGES = {
    "score": [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
    "risk":  [0.1, 0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5],
    "freq":  [1, 3, 5, 8, 12, 20, 35, 50, 80],
}

# ── PASS 4 reduced search ranges (5 candidates per feature, O(n²) budget) ──
PASS4_RANGES = {
    "score": [0.05, 0.15, 0.30, 0.50, 0.80],
    "risk":  [0.3, 0.8, 1.3, 2.0, 2.5],
    "freq":  [1, 5, 15, 35, 80],
}

PASS4_PAIRS = [("score", "risk"), ("score", "freq"), ("risk", "freq")]


# ────────────────────────────────────────────────────────────────────────────
# PASS 1 — single-feature flip
# ────────────────────────────────────────────────────────────────────────────

def _pass1_search(score, risk, attack_type, freq, current_action):
    """
    PASS 1: Try to flip action by perturbing score, risk, or freq.
    Returns result dict with found=True/False.
    """
    originals = {"score": score, "risk": risk, "freq": freq}
    candidates = []

    for feature in PERTURBABLE_FEATURES:
        orig_val = originals[feature]
        for candidate_val in SEARCH_RANGES[feature]:
            new_score = candidate_val if feature == "score" else score
            new_risk  = candidate_val if feature == "risk"  else risk
            new_freq  = candidate_val if feature == "freq"  else freq

            decision = make_decision(new_score, new_risk, attack_type, new_freq)
            if decision["action"] != current_action:
                denom = max(abs(orig_val), 1e-9)
                rel_change = abs(candidate_val - orig_val) / denom
                candidates.append({
                    "feature_changed":      feature,
                    "original_value":       orig_val,
                    "counterfactual_value": candidate_val,
                    "new_action":           decision["action"],
                    "rel_change":           rel_change,
                })

    if not candidates:
        return {"found": False}

    best = min(candidates, key=lambda x: x["rel_change"])

    orig_str = (str(int(round(best["original_value"])))
                if best["feature_changed"] == "freq"
                else f"{best['original_value']:.3f}")
    cf_str   = (str(int(best["counterfactual_value"]))
                if best["feature_changed"] == "freq"
                else f"{best['counterfactual_value']:.3f}")

    explanation = (
        f"If {best['feature_changed']} were {cf_str} "
        f"(instead of {orig_str}), action would be "
        f"{best['new_action']} instead of {current_action}"
    )

    return {
        "found":                True,
        "feature_changed":      best["feature_changed"],
        "original_value":       best["original_value"],
        "counterfactual_value": best["counterfactual_value"],
        "new_action":           best["new_action"],
        "rel_change":           best["rel_change"],
        "explanation":          explanation,
    }


# ────────────────────────────────────────────────────────────────────────────
# PASS 2 — override explanation
# ────────────────────────────────────────────────────────────────────────────

def _pass2_search(score, risk, attack_type, freq, current_action):
    """
    PASS 2: Explain via attack_type override removal.
    Only runs when attack_type is in ATTACK_OVERRIDES.

    Layer 1 — why the override fires and what action would be without it.
    Layer 2 — smallest score/risk change (no override) that further shifts action.
    """
    if attack_type not in ATTACK_OVERRIDES:
        return {"found": False}

    neutral = make_decision(score, risk, "", freq)
    no_override_action = neutral["action"]

    layer1 = (
        f"Action is {current_action} due to {attack_type} override. "
        f"Without the override, action would be {no_override_action} at current score/risk."
    )

    candidates = []
    for feature in ["score", "risk"]:
        orig_val = score if feature == "score" else risk
        for candidate_val in SEARCH_RANGES[feature]:
            new_score = candidate_val if feature == "score" else score
            new_risk  = candidate_val if feature == "risk"  else risk

            decision = make_decision(new_score, new_risk, "", freq)
            if decision["action"] != no_override_action:
                denom = max(abs(orig_val), 1e-9)
                rel = abs(candidate_val - orig_val) / denom
                candidates.append({
                    "feature": feature, "orig": orig_val,
                    "cf": candidate_val, "action": decision["action"], "rel": rel,
                })

    rel_change = 0.0
    if candidates:
        best = min(candidates, key=lambda x: x["rel"])
        feat = best["feature"]
        layer2 = (
            f"Additionally, if {feat} were {best['cf']:.3f} "
            f"(instead of {best['orig']:.3f}), action would be {best['action']}."
        )
        explanation     = f"{layer1} {layer2}"
        feature_changed = f"override+{feat}"
        rel_change      = best["rel"]
    else:
        explanation     = layer1
        feature_changed = "override"

    return {
        "found":                True,
        "feature_changed":      feature_changed,
        "original_value":       None,
        "counterfactual_value": None,
        "new_action":           no_override_action,
        "rel_change":           rel_change,
        "explanation":          explanation,
    }


# ────────────────────────────────────────────────────────────────────────────
# PASS 3 — temporal counterfactual
# ────────────────────────────────────────────────────────────────────────────

def _pass3_temporal(score, risk, attack_type, freq, current_action):
    """
    PASS 3: "What if this were the first packet from this entity?"
    Sets freq=1 and attack_type="" (no history, no classification yet),
    then checks whether the action would differ.

    Always runs on ANOMALY packets as an additional explanation layer.
    """
    temporal_decision = make_decision(score, risk, "", 1.0)
    new_action = temporal_decision["action"]

    if new_action != current_action:
        explanation = (
            f"Temporal: If this were the first packet from this entity "
            f"(freq=1, no attack history), action would be {new_action} "
            f"instead of {current_action}."
        )
        return {"found": True, "new_action": new_action, "explanation": explanation}

    return {"found": False}


# ────────────────────────────────────────────────────────────────────────────
# PASS 4 — multi-feature counterfactual
# ────────────────────────────────────────────────────────────────────────────

def _pass4_multifeature(score, risk, attack_type, freq, current_action):
    """
    PASS 4: Try pairs of feature changes simultaneously.
    Only called when PASS 1 found no single-feature flip.
    Uses a reduced 5-candidate search space per feature (75 evaluations total).
    """
    originals = {"score": score, "risk": risk, "freq": freq}
    candidates = []

    for feat1, feat2 in PASS4_PAIRS:
        orig1 = originals[feat1]
        orig2 = originals[feat2]

        for cand1 in PASS4_RANGES[feat1]:
            for cand2 in PASS4_RANGES[feat2]:
                new_score = (cand1 if feat1 == "score" else
                             cand2 if feat2 == "score" else score)
                new_risk  = (cand1 if feat1 == "risk"  else
                             cand2 if feat2 == "risk"  else risk)
                new_freq  = (cand1 if feat1 == "freq"  else
                             cand2 if feat2 == "freq"  else freq)

                decision = make_decision(new_score, new_risk, attack_type, new_freq)
                if decision["action"] != current_action:
                    rel1 = abs(cand1 - orig1) / max(abs(orig1), 1e-9)
                    rel2 = abs(cand2 - orig2) / max(abs(orig2), 1e-9)
                    candidates.append({
                        "feat1": feat1, "orig1": orig1, "cand1": cand1,
                        "feat2": feat2, "orig2": orig2, "cand2": cand2,
                        "new_action":  decision["action"],
                        "total_rel":   rel1 + rel2,
                    })

    if not candidates:
        return {"found": False}

    best = min(candidates, key=lambda x: x["total_rel"])

    def _fmt(feat, val):
        return str(int(val)) if feat == "freq" else f"{val:.3f}"

    explanation = (
        f"Multi: If {best['feat1']} were {_fmt(best['feat1'], best['cand1'])} "
        f"AND {best['feat2']} were {_fmt(best['feat2'], best['cand2'])} "
        f"(instead of {_fmt(best['feat1'], best['orig1'])} and "
        f"{_fmt(best['feat2'], best['orig2'])}), "
        f"action would be {best['new_action']} instead of {current_action}"
    )

    return {
        "found":       True,
        "new_action":  best["new_action"],
        "explanation": explanation,
        "total_rel":   best["total_rel"],
    }


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def generate_all_counterfactuals(score, risk, attack_type, freq, current_action):
    """
    Run all four passes and return three column strings:

      pass1    — PASS 1 (single-feature flip) OR PASS 4 (multi-feature, "Multi: " prefix)
      pass2    — PASS 2 (override explanation)
      temporal — PASS 3 (temporal CF, always computed independently)

    PASS 4 is only evaluated when PASS 1 finds no flip.
    PASS 3 runs unconditionally as an additional explanation layer.
    """
    p1 = _pass1_search(score, risk, attack_type, freq, current_action)
    p2 = _pass2_search(score, risk, attack_type, freq, current_action)
    p3 = _pass3_temporal(score, risk, attack_type, freq, current_action)

    if p1["found"]:
        pass1_str = p1["explanation"]
    else:
        p4 = _pass4_multifeature(score, risk, attack_type, freq, current_action)
        pass1_str = p4["explanation"] if p4["found"] else ""

    return {
        "pass1":    pass1_str,
        "pass2":    p2["explanation"] if p2["found"] else "",
        "temporal": p3["explanation"] if p3["found"] else "",
    }


def generate_counterfactual(score, risk, attack_type, freq, current_action):
    """Legacy single-result wrapper kept for backward compatibility."""
    p1 = _pass1_search(score, risk, attack_type, freq, current_action)
    if p1["found"]:
        return p1

    p2 = _pass2_search(score, risk, attack_type, freq, current_action)
    if p2["found"]:
        return p2

    return {
        "found": False, "feature_changed": None,
        "original_value": None, "counterfactual_value": None,
        "new_action": None,
        "explanation": "No counterfactual found within search range",
    }
