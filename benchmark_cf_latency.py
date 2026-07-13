"""
benchmark_cf_latency.py
========================
Benchmarks the LatencyBoundedCFGenerator across all four budget modes
(STRICT, NORMAL, RELAXED, BATCH) on the ANOMALY rows from results.csv.

For each mode, reports:
  - n_tested       : number of anomaly packets run through the generator
  - avg_ms         : mean wall-clock latency per packet (milliseconds)
  - p50_ms / p95_ms: median and 95th-percentile latency
  - coverage_pct   : % of packets where cf["found"] == True
  - pass1_pct      : % resolved by Pass 1
  - pass2_pct      : % resolved by Pass 2
  - pass3_pct      : % resolved by Pass 3 (or exhausted without hit)
  - timeout_pct    : % that hit the time budget before finding a CF

Output: experiments/cf_latency_results.csv
"""

import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from counterfactual_engine.cf_realtime import LatencyBoundedCFGenerator

RESULTS_CSV  = os.path.join(_HERE, "results.csv")
OUTPUT_CSV   = os.path.join(_HERE, "experiments", "cf_latency_results.csv")
MODES        = ["STRICT", "NORMAL", "RELAXED", "BATCH"]


# ── Load anomaly rows from results.csv ──────────────────────────────────────

def _load_anomaly_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        filtered = (line for line in fh if not line.startswith("#"))
        reader   = csv.DictReader(filtered)
        for row in reader:
            if row["label"].strip().upper() == "ANOMALY":
                try:
                    rows.append({
                        "score":          float(row["score"]),
                        "risk":           float(row["risk"]),
                        "attack_type":    row.get("attack_type", "").strip(),
                        "freq":           float(row.get("risk", 1.0)),  # freq not stored; proxy with risk
                        "current_action": row["action"].strip(),
                    })
                except (ValueError, KeyError):
                    pass
    return rows


# ── Run benchmark for one mode ───────────────────────────────────────────────

def _run_mode(mode, rows):
    gen       = LatencyBoundedCFGenerator(mode=mode)
    latencies = []
    found_total = 0
    pass_tally  = {"pass1": 0, "pass2": 0, "pass3": 0, "timeout": 0}

    for r in rows:
        t0 = time.perf_counter()
        cf, pass_label, elapsed_ms = gen.generate(
            score          = r["score"],
            risk           = r["risk"],
            attack_type    = r["attack_type"],
            freq           = r["freq"],
            current_action = r["current_action"],
        )
        latencies.append(elapsed_ms)
        if cf.get("found"):
            found_total += 1

        if pass_label in pass_tally:
            pass_tally[pass_label] += 1
        else:
            pass_tally["timeout"] += 1

    n = len(latencies)
    if n == 0:
        return None

    latencies.sort()
    avg_ms = sum(latencies) / n
    p50_ms = latencies[int(n * 0.50)]
    p95_ms = latencies[min(int(n * 0.95), n - 1)]
    max_ms = latencies[-1]

    return {
        "mode":         mode,
        "n_tested":     n,
        "avg_ms":       round(avg_ms, 4),
        "p50_ms":       round(p50_ms, 4),
        "p95_ms":       round(p95_ms, 4),
        "max_ms":       round(max_ms, 4),
        "coverage_pct": round(found_total / n * 100, 2),
        "pass1_pct":    round(pass_tally["pass1"]   / n * 100, 2),
        "pass2_pct":    round(pass_tally["pass2"]   / n * 100, 2),
        "pass3_pct":    round(pass_tally["pass3"]   / n * 100, 2),
        "timeout_pct":  round(pass_tally["timeout"] / n * 100, 2),
        "total_found":  found_total,
        "total_timeout": pass_tally["timeout"],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"[benchmark] Loading anomaly rows from {RESULTS_CSV} ...")
    rows = _load_anomaly_rows(RESULTS_CSV)
    if not rows:
        print("[benchmark] ERROR: No anomaly rows found. Exiting.")
        sys.exit(1)
    print(f"[benchmark] {len(rows):,} anomaly packets loaded.\n")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    fieldnames = [
        "mode", "n_tested", "avg_ms", "p50_ms", "p95_ms", "max_ms",
        "coverage_pct", "pass1_pct", "pass2_pct", "pass3_pct", "timeout_pct",
        "total_found", "total_timeout",
    ]

    results = []
    for mode in MODES:
        print(f"[benchmark] Running mode={mode} on {len(rows):,} packets ...")
        t_start = time.perf_counter()
        stats   = _run_mode(mode, rows)
        t_done  = time.perf_counter() - t_start

        if stats is None:
            print(f"  SKIPPED (no data)")
            continue

        results.append(stats)
        print(f"  avg={stats['avg_ms']:.3f}ms  p95={stats['p95_ms']:.3f}ms  "
              f"coverage={stats['coverage_pct']:.1f}%  "
              f"p1={stats['pass1_pct']:.1f}%  p2={stats['pass2_pct']:.1f}%  "
              f"p3={stats['pass3_pct']:.1f}%  timeout={stats['timeout_pct']:.1f}%  "
              f"(wall={t_done:.1f}s)\n")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"[benchmark] Saved: {OUTPUT_CSV}")
    print()
    print("=" * 72)
    print(f"{'Mode':<10}  {'avg ms':>8}  {'p95 ms':>8}  {'coverage':>9}  "
          f"{'pass1':>6}  {'pass2':>6}  {'pass3':>6}  {'timeout':>8}")
    print("-" * 72)
    for r in results:
        print(f"{r['mode']:<10}  {r['avg_ms']:>8.3f}  {r['p95_ms']:>8.3f}  "
              f"{r['coverage_pct']:>8.1f}%  "
              f"{r['pass1_pct']:>5.1f}%  {r['pass2_pct']:>5.1f}%  "
              f"{r['pass3_pct']:>5.1f}%  {r['timeout_pct']:>7.1f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
