# test_explain.py
# Simulates a full explanation pipeline run without requiring Kitsune.
# Run with:  python test_explain.py

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import random
import explain_pipeline

random.seed(7)

# ── Simulate MODE_CONFIG entry (matches NORMAL mode) ──
MOCK_CONFIG = {
    "alpha":          0.7,
    "rate_threshold": 50,
    "rate_mult":      1.3,
    "endpoint_mult":  1.5,
}

# ── Simulate 60 training packets to build baselines ──
print("Building baselines (60 training packets)...")
for t in range(1, 61):
    entity = f"192.168.0.{(t % 4) + 1}"
    ctx = {
        "src_ip":   entity,
        "endpoint": random.choice(["/web", "/web", "/web", "/secure"]),
        "freq":     float(random.randint(1, 5)),
        "length":   random.randint(300, 600),
    }
    feat_vec = [random.gauss(0.3, 0.05) for _ in range(24)]
    explain_pipeline.record_temporal(
        entity_id = entity,
        timestamp = float(t),
        ctx       = ctx,
        freq_z    = 0.0,
        length_z  = 0.0,
        score     = random.uniform(0.01, 0.08),
    )

print("Baselines built.\n")

# ── Simulate an ANOMALY packet ──
ANOMALY_ENTITY    = "192.168.0.3"
ANOMALY_TIMESTAMP = 120.0

anomaly_ctx = {
    "src_ip":   ANOMALY_ENTITY,
    "endpoint": "/ssh",
    "freq":     12.0,
    "length":   820,
}

# simulate a preceding escalation in the temporal journal
for t_prev in [90.0, 100.0, 110.0]:
    explain_pipeline.record_temporal(
        entity_id = ANOMALY_ENTITY,
        timestamp = t_prev,
        ctx       = {**anomaly_ctx, "freq": t_prev / 10.0},
        freq_z    = 2.5,
        length_z  = 1.8,
        score     = 0.25 + (t_prev - 90) * 0.02,
    )

# the anomaly packet itself
anomaly_feat_vec = [random.gauss(0.8, 0.15) for _ in range(24)]

result = explain_pipeline.process(
    entity_id      = ANOMALY_ENTITY,
    timestamp      = ANOMALY_TIMESTAMP,
    ctx            = anomaly_ctx,
    feature_vector = anomaly_feat_vec,
    base_score     = 0.61,
    final_score    = 0.74,
    raw_risk       = 1.43,
    adjusted_risk  = 1.45,
    freq_deviation = 4.2,
    length_signal  = 1.64,
    config         = MOCK_CONFIG,
    label          = "ANOMALY",
    attack_type    = "Brute Force",
)

# ── Print all outputs ──
print("═" * 60)
print("ARA — Top Features:")
for f in result["ara"]["top_features"][:3]:
    print(f"  {f['name']}: observed={f['value']}, expected={f['expected']}, "
          f"contribution={f['contribution_pct']}%")

print("\nBFDE — Behavioral Deviations:")
for d in result["bfde"]["deviations"]:
    print(f"  {d['description']}")

print("\nTCCE — Causal Chain:")
for e in result["tcce"]["chain"]:
    print(f"  {e['description']}")

print("\nRSDR — Risk Audit:")
print(result["rsdr"]["report_str"])

print("\nNARRATIVE:")
print(result["narrative"])