# explain_pipeline.py
# Explainability orchestrator.
#
# FIX: process() must ONLY be called for ANOMALY packets.
#      Enforcement is in main_ids.py, but this module adds a guard.
# FIX: update_bfde() called by main_ids.py AFTER process()
#      to prevent data leakage into BFDE baseline.

import ara_explainer
import bfde_explainer
import tcce_explainer
import rsdr_explainer
import narrative_engine


def record_temporal(entity_id, timestamp, ctx, freq_z, length_z, score):
    """
    Record temporal event. Call every packet BEFORE profile update.
    Safe to call for NORMAL and TRAIN packets — builds causal context.
    """
    tcce_explainer.record_event(
        entity_id = entity_id,
        timestamp = float(timestamp),
        ctx       = ctx,
        freq_z    = float(freq_z),
        length_z  = float(length_z),
        score     = float(score),
    )


def update_ara_baseline(feature_vector):
    """
    Update ARA feature baseline. Call during training phase only.
    Requires real multi-dimensional feature vector — no-op for [score].
    """
    if feature_vector is not None and len(feature_vector) > 1:
        ara_explainer.update_baselines(feature_vector)


def update_bfde(entity_id, ctx):
    """
    Update BFDE behavioral profile.
    MUST be called AFTER process() to avoid leaking current observation
    into the baseline used for explanation.
    """
    bfde_explainer.update(entity_id, ctx)


def process(
    entity_id,
    timestamp,
    ctx,
    feature_vector,
    base_score,
    final_score,
    raw_risk,
    adjusted_risk,
    freq_deviation,
    length_signal,
    config,
    label,
    attack_type,
    attack_reason="",
):
    """
    Generate explanation for one ANOMALY packet.
    Uses PRE-UPDATE BFDE state — no data leakage.

    FIX: Guard ensures this never runs for NORMAL/TRAIN labels.

    Returns
    -------
    dict: ara, bfde, tcce, rsdr, narrative
    """
    # Guard: explanation only makes sense for confirmed anomalies
    if label != "ANOMALY":
        return {
            "ara":       {},
            "bfde":      {},
            "tcce":      {},
            "rsdr":      {"report_str": "", "confidence": "N/A", "signals_fired": 0},
            "narrative": "",
        }

    ara_result  = ara_explainer.explain(feature_vector, base_score)
    bfde_result = bfde_explainer.explain(entity_id, ctx)
    tcce_result = tcce_explainer.explain(entity_id, float(timestamp))

    rsdr_result = rsdr_explainer.explain(
        base_score     = base_score,
        final_score    = final_score,
        raw_risk       = raw_risk,
        adjusted_risk  = adjusted_risk,
        ctx            = ctx,
        freq_deviation = freq_deviation,
        length_signal  = length_signal,
        config         = config,
        label          = label,
        attack_type    = attack_type,
    )

    narrative = narrative_engine.generate(
        entity_id     = entity_id,
        timestamp     = float(timestamp),
        attack_type   = attack_type,
        ara_result    = ara_result,
        bfde_result   = bfde_result,
        tcce_result   = tcce_result,
        rsdr_result   = rsdr_result,
        ctx           = ctx,
        attack_reason = attack_reason,
    )

    return {
        "ara":       ara_result,
        "bfde":      bfde_result,
        "tcce":      tcce_result,
        "rsdr":      rsdr_result,
        "narrative": narrative,
    }