# main_ids.py
# Research-grade IDS pipeline with strict gating.
#
# ARCHITECTURE (enforced):
#   Kitsune score → risk → threshold → label
#   IF ANOMALY: classify → explain
#   IF NORMAL:  attack_type="Normal", no explanation, no reasons
#
# KEY FIXES:
#   - classifier NEVER runs for NORMAL traffic
#   - explain_pipeline NEVER runs for NORMAL traffic
#   - explain() reasons NEVER printed for NORMAL traffic
#   - profiles updated AFTER decision (no leakage)
#   - threshold seeded with adjusted_risk (same distribution as detection)
#   - adaptive threshold capped via max_threshold to prevent drift

import argparse
import csv
import datetime
import importlib.metadata
import json
import os
import subprocess
from kitsune_wrapper import KitsuneDetector
from adaptive_threshold import AdaptiveThreshold
from context_features import extract_context
from risk_engine import compute_risk
from profile_manager import get_profile, update_profile, compute_deviation
from explain import explain
from attack_classifier import classify_attack
from profile_scoring import combine_scores
import explain_pipeline
from decision_layer.decision_engine import make_decision
from counterfactual_engine.cf_generator import generate_all_counterfactuals

_feedback_learner = None

RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

MAX_PACKETS  = 100000
RESULTS_FILE = "results.csv"

FMgrace = 10000
ADgrace = 10000

MODE_CONFIG = {
    "NORMAL": {
        "threshold_mode": "low_noise",
        "window":         250,
        "alpha":          0.7,
        "endpoint_mult":  1.5,
        "rate_mult":      1.3,
        "rate_threshold": 50,         # SEE config.py → rate_threshold_normal for rationale
        "training_gate":  20000,
        # FIX: cap threshold to prevent unbounded drift → false negatives
        "max_threshold":  2.5,        # SEE config.py → adaptive_max_threshold_normal for rationale
    },
    "SENSITIVE": {
        "threshold_mode": "balanced",
        "window":         150,
        "alpha":          0.5,
        "endpoint_mult":  2.0,
        "rate_mult":      1.6,
        "rate_threshold": 30,
        "training_gate":  15000,
        "max_threshold":  1.8,
    },
    "STRICT": {
        "threshold_mode": "high_security",
        "window":         100,
        "alpha":          0.4,
        "endpoint_mult":  2.5,
        "rate_mult":      2.0,
        "rate_threshold": 15,
        "training_gate":  10000,
        "max_threshold":  1.2,
    },
}


def get_phase(i):
    if i <= FMgrace:
        return "📘 Feature Learning"
    elif i <= FMgrace + ADgrace:
        return "📗 Training"
    else:
        return "🔍 Detection"


def _compute_adjusted_risk(final_score, ctx, config, freq_deviation, length_signal, profile=None):
    """
    Single source of truth for risk computation.
    Used for both threshold seeding (training) and live detection.
    This ensures the threshold and the detection gate operate on
    the SAME distribution — fixing the calibration mismatch.
    """
    risk = compute_risk(
        final_score, ctx,
        endpoint_mult  = config["endpoint_mult"],
        rate_mult      = config["rate_mult"],
        rate_threshold = config["rate_threshold"],
        freq_deviation = freq_deviation,
        length_signal  = length_signal,
        profile        = profile,
    )
    bonus = 0.0
    if ctx["endpoint"] in ("/ssh", "/secure"):
        bonus += 0.02
    if freq_deviation > 5.0:
        bonus += 0.01
    if ctx["freq"] > config["rate_threshold"]:
        bonus += 0.01
    return risk, risk + bonus


def run(args):
    global FMgrace, ADgrace
    FMgrace       = args.fmgrace
    ADgrace       = args.adgrace
    config        = MODE_CONFIG[args.mode]
    training_gate = args.training_gate

    import config as _ids_config
    _profile_path   = args.profile or str(_ids_config.DEFAULT_PROFILE)
    traffic_profile = _ids_config.load_profile(_profile_path)

    print(f"🚀 Running my IDS... [MODE: {args.mode}]")

    # FIX: pass max_threshold so the adaptive threshold cannot drift
    # beyond a sensible ceiling — prevents unbounded false negative zone
    # OLD: KitsuneDetector(FMgrace=2000, ADgrace=2000) — hardcoded; KitNET
    #      entered detection at packet 4,001 while the pipeline labeled
    #      packets 1-20,000 as TRAIN, seeding the threshold with 15,984
    #      contaminated detection-phase scores.
    # NEW: FMgrace/ADgrace come from CLI args (default 10000/10000) so
    #      KitNET's training window aligns with the pipeline's training_gate.
    detector = KitsuneDetector(
        FMgrace   = args.fmgrace,
        ADgrace   = args.adgrace,
        pcap_path = args.pcap,
    )
    adaptive = AdaptiveThreshold(
        window        = config["window"],
        mode          = config["threshold_mode"],
        max_threshold = config.get("max_threshold", None),
    )

    i           = 0
    _seed_risks = []

    try:
        _git = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).decode().strip()
    except Exception:
        _git = 'unknown'
    _meta = (
        f"# run_timestamp={datetime.datetime.now().isoformat(timespec='seconds')}, "
        f"mode={args.mode}, fmgrace={args.fmgrace}, adgrace={args.adgrace}, "
        f"hard_override={args.hard_override}, training_gate={args.training_gate}, "
        f"numpy={importlib.metadata.version('numpy')}, "
        f"scapy={importlib.metadata.version('scapy')}, "
        f"git_commit={_git}"
    )

    csv_file = open(RESULTS_FILE, "w", newline="", encoding="utf-8")
    csv_file.write(_meta + "\n")
    writer   = csv.writer(csv_file)
    writer.writerow([
        "packet_id", "score", "risk", "threshold",
        "label", "attack_type", "attack_reason", "ara_active", "detection_path",
        "severity", "action", "cf_pass1", "cf_pass2", "cf_temporal"
    ])

    while True:
        i += 1
        if i >= MAX_PACKETS:
            print("🛑 Reached packet limit (100k)")
            break

        score, packet, features, ara_active = detector.process()
        if score == -1:
            print("✅ Finished PCAP.")
            break

        # ── 1. Context extraction ──
        X_context, ctx = extract_context(packet, profile=traffic_profile)
        entity_id      = ctx.get("src_ip", "unknown")
        timestamp      = packet.timestamp

        # safe defaults — overwritten in detection branch
        freq_deviation = 0.0
        length_signal  = 1.0
        final_score    = score
        attack_reason  = ""
        attack_type    = ""
        detection_path = ""

        # ── 2. ARA baseline update (training only) ──
        if i <= training_gate:
            explain_pipeline.update_ara_baseline(features)

        # ── 3. Pure freq profile value (no length/modulo pollution) ──
        value = ctx.get("freq", 0.0)

        # ══════════════════════════════════════════════
        # TRAINING BRANCH
        # ══════════════════════════════════════════════
        if i <= training_gate:
            final_score = score
            _, ar_seed  = _compute_adjusted_risk(
                final_score, ctx, config, 0.0, 1.0, profile=traffic_profile
            )
            _seed_risks.append(ar_seed)

            label         = "TRAIN"
            risk          = ar_seed
            adjusted_risk = ar_seed
            attack_type   = ""
            attack_reason = ""

            # Record temporal event with neutral z-scores
            explain_pipeline.record_temporal(
                entity_id, timestamp, ctx,
                freq_z=0.0, length_z=0.0, score=score
            )

            # Update BFDE baseline during training (safe — no decision yet)
            explain_pipeline.update_bfde(entity_id, ctx)

        # ══════════════════════════════════════════════
        # DETECTION BRANCH
        # ══════════════════════════════════════════════
        else:
            # Seed threshold once with adjusted_risk distribution
            if _seed_risks:
                adaptive.seed(_seed_risks)
                _seed_risks.clear()

            # Read profile BEFORE any update (no leakage)
            profile   = get_profile(entity_id)
            deviation = compute_deviation(profile, value)

            freq_deviation = abs(ctx["freq"] - profile["mean"])
            length_signal  = ctx.get("length", 500) / 500.0
            length_z       = (ctx.get("length", 500) - 500.0) / 150.0

            final_score = combine_scores(
                score, deviation, alpha=config["alpha"]
            )

            risk, adjusted_risk = _compute_adjusted_risk(
                final_score, ctx, config, freq_deviation, length_signal, profile=traffic_profile
            )

            if _feedback_learner is not None:
                context_key = attack_type or "unknown"
                risk = _feedback_learner.get_adjusted_risk(
                    risk, context_key, ["score", "risk"]
                )
                _bonus = 0.0
                if ctx["endpoint"] in ("/ssh", "/secure"):
                    _bonus += 0.02
                if freq_deviation > 5.0:
                    _bonus += 0.01
                if ctx["freq"] > config["rate_threshold"]:
                    _bonus += 0.01
                adjusted_risk = risk + _bonus

            # Record temporal event with real z-scores
            explain_pipeline.record_temporal(
                entity_id, timestamp, ctx,
                freq_z   = freq_deviation,
                length_z = length_z,
                score    = score,
            )

            # ── 4. Detection decision ──
            # Hard override for extreme cases; adaptive threshold for the rest.
            # FIX: override thresholds are high enough to not fire constantly.
            if adjusted_risk > args.hard_override:
                label          = "ANOMALY"
                detection_path = "hard_override_risk"
            elif final_score > 0.75:
                label          = "ANOMALY"
                detection_path = "hard_override_score"
            else:
                label          = adaptive.classify(adjusted_risk)
                detection_path = "adaptive_threshold" if label == "ANOMALY" else "normal"

            # Update threshold on EVERY packet (not just NORMAL)
            # so it does not drift upward during anomaly bursts
            adaptive.update(adjusted_risk)

            # ── 5. Classification — ONLY if ANOMALY ──
            # FIX: classifier is a LABELLER, not a detector.
            # It NEVER runs for NORMAL traffic.
            if label == "ANOMALY":
                attack_type, attack_reason = classify_attack(
                    final_score, adjusted_risk, ctx, freq_deviation
                )
            else:
                # FIX: NORMAL traffic gets no attack type and no reason
                attack_type   = "Normal"
                attack_reason = ""

            # ── 6. Explanation — ONLY if ANOMALY ──
            # FIX: explain_pipeline NEVER runs for NORMAL traffic
            if label == "ANOMALY":
                explanation = explain_pipeline.process(
                    entity_id      = entity_id,
                    timestamp      = timestamp,
                    ctx            = ctx,
                    feature_vector = features,
                    base_score     = score,
                    final_score    = final_score,
                    raw_risk       = risk,
                    adjusted_risk  = adjusted_risk,
                    freq_deviation = freq_deviation,
                    length_signal  = length_signal,
                    config         = config,
                    label          = label,
                    attack_type    = attack_type,
                    attack_reason  = attack_reason,
                )
                print(explanation["narrative"])
                print(explanation["rsdr"]["report_str"])

            # ── 7. Profile update LAST — after decision and explanation ──
            # FIX: updating before explanation would contaminate BFDE baseline
            explain_pipeline.update_bfde(entity_id, ctx)
            update_profile(profile, value)

        thr = adaptive.last_threshold

        # FIX: reasons printed ONLY for ANOMALY traffic
        reasons = explain(final_score, ctx) if label == "ANOMALY" else []

        decision = make_decision(final_score, adjusted_risk, attack_type, ctx.get("freq", 0.0))
        if label == "ANOMALY":
            cfs = generate_all_counterfactuals(
                final_score, adjusted_risk, attack_type, ctx.get("freq", 0.0),
                decision["action"]
            )
            cf1, cf2, cf3 = cfs["pass1"], cfs["pass2"], cfs["temporal"]
        else:
            cf1 = cf2 = cf3 = ""
        writer.writerow([
            i, round(final_score, 4), round(adjusted_risk, 4),
            round(thr, 4), label, attack_type, attack_reason, ara_active, detection_path,
            decision["severity"], decision["action"], cf1, cf2, cf3
        ])
        csv_file.flush()

        phase = get_phase(i)
        label_colored = (
            f"{RED}{label}{RESET}"  if label == "ANOMALY"
            else f"{GREEN}{label}{RESET}"
        )
        attack_str = f" | 🔴 {attack_type}" if label == "ANOMALY" else ""

        print(
            f"[{i}] {phase} | score={final_score:.4f} | "
            f"risk={adjusted_risk:.4f} | thr={thr:.4f} → "
            f"{label_colored}{attack_str}"
        )

        # FIX: reasons only printed when label is ANOMALY
        if reasons and label == "ANOMALY":
            print(f"{YELLOW}   reasons: {reasons}{RESET}")

    csv_file.close()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="Kitsune IDS pipeline")
    _parser.add_argument("--mode", choices=["NORMAL", "SENSITIVE", "STRICT"],
                         default="NORMAL")
    _parser.add_argument("--pcap", default=None, metavar="PATH",
                         help="PCAP file path (default: mirai.pcap in project root)")
    _parser.add_argument("--fmgrace", type=int, default=10000, metavar="N")
    _parser.add_argument("--adgrace", type=int, default=10000, metavar="N")
    _parser.add_argument("--hard-override", type=float, default=1.5,  # SEE config.py → hard_override_risk for rationale
                         dest="hard_override", metavar="F")
    _parser.add_argument("--training-gate", type=int, default=20000,
                         dest="training_gate", metavar="N")
    _parser.add_argument("--profile", default=None, metavar="PATH",
                         help="Traffic profile YAML (default: profiles/mirai.yaml)")
    _parser.add_argument("--feedback-weights", default=None, metavar="PATH",
                         dest="feedback_weights",
                         help="Load pre-computed feature weights JSON and apply to risk engine")
    _args = _parser.parse_args()
    import config as _cfg_mod
    _cfg_mod.init_profile(_args.profile)
    if _args.feedback_weights and os.path.exists(_args.feedback_weights):
        from feedback.feature_weight_learner import FeatureWeightLearner
        _learner = FeatureWeightLearner()
        with open(_args.feedback_weights, encoding="utf-8") as _fh:
            _learner.adjustments = json.load(_fh)
        print(f"[Feedback] Loaded weights from {_args.feedback_weights}")
        _feedback_learner = _learner
    run(_args)