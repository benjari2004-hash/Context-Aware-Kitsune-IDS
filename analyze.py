import argparse
import csv
import re
from pathlib import Path


DEFAULT_RESULTS_FILE = Path(__file__).with_name("results_static.csv")
REQUIRED_FIELDS = {"packet_id", "score", "threshold", "label"}


def analyze(csv_file):
    total_packets = 0
    anomalies_count = 0
    path_counts = {
        "hard_override_risk":  0,
        "hard_override_score": 0,
        "adaptive_threshold":  0,
        "normal":              0,
    }
    ara_counts    = {"True": 0, "False": 0}
    action_counts  = {}
    cf_total       = 0
    cf1_count      = 0   # PASS 1 single-feature
    cf4_count      = 0   # PASS 4 multi-feature (detected by "Multi: " prefix in cf_pass1)
    cf2_count      = 0   # PASS 2 override
    cf3_change     = 0   # PASS 3 found (different action)
    cf3_same       = 0   # PASS 3 ran but action unchanged
    cf_any         = 0   # packets with at least one CF found
    cf_features    = {"score": 0, "risk": 0, "freq": 0, "override": 0}
    cf_rel_changes = []
    has_path_col   = False
    has_ara_col    = False
    has_action_col = False
    has_cf_col     = False   # legacy single column
    has_cf3_cols   = False   # new three-column layout

    with csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith('#'))
        missing_fields = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"CSV is missing required columns: {missing}")

        has_path_col   = "detection_path" in (reader.fieldnames or [])
        has_ara_col    = "ara_active"     in (reader.fieldnames or [])
        has_action_col = "action"         in (reader.fieldnames or [])
        has_cf_col     = "counterfactual" in (reader.fieldnames or [])
        has_cf3_cols   = "cf_pass1"       in (reader.fieldnames or [])

        for row in reader:
            total_packets += 1
            if row["label"] == "ANOMALY":
                anomalies_count += 1
            if has_path_col:
                path = row.get("detection_path", "")
                if path in path_counts:
                    path_counts[path] += 1
            if has_ara_col:
                ara_val = row.get("ara_active", "")
                if ara_val in ara_counts:
                    ara_counts[ara_val] += 1
            if has_action_col:
                act = row.get("action", "")
                if act:
                    action_counts[act] = action_counts.get(act, 0) + 1

            # ── New 3-column CF layout ──────────────────────────────────────
            if has_cf3_cols and row["label"] == "ANOMALY":
                cf_total += 1
                p1_str = row.get("cf_pass1", "")
                p2_str = row.get("cf_pass2", "")
                p3_str = row.get("cf_temporal", "")

                # PASS 4 is stored in cf_pass1 with "Multi: " prefix
                if p1_str.startswith("Multi: "):
                    cf4_count += 1
                elif p1_str:
                    # PASS 1 single-feature: "If {feature} were {cf} (instead of {orig}), ..."
                    cf1_count += 1
                    m1 = re.match(r"If (\w+) were ([\d.]+) \(instead of ([\d.]+)\)", p1_str)
                    if m1:
                        feat = m1.group(1)
                        cf_val   = float(m1.group(2))
                        orig_val = float(m1.group(3))
                        if feat in cf_features:
                            cf_features[feat] += 1
                        cf_rel_changes.append(abs(cf_val - orig_val) / max(abs(orig_val), 1e-9))

                if p2_str:
                    cf2_count += 1
                    cf_features["override"] += 1
                    m2 = re.search(r"if (\w+) were ([\d.]+) \(instead of ([\d.]+)\)", p2_str)
                    if m2:
                        feat2     = m2.group(1)
                        cf_val2   = float(m2.group(2))
                        orig_val2 = float(m2.group(3))
                        if feat2 in cf_features:
                            cf_features[feat2] += 1
                        cf_rel_changes.append(abs(cf_val2 - orig_val2) / max(abs(orig_val2), 1e-9))

                if p3_str:
                    cf3_change += 1
                else:
                    cf3_same += 1

                if p1_str or p2_str or p3_str:
                    cf_any += 1

            # ── Legacy single-column CF layout ─────────────────────────────
            elif has_cf_col and row["label"] == "ANOMALY":
                cf_total += 1
                cf_str = row.get("counterfactual", "")
                m1 = re.match(r"If (\w+) were ([\d.]+) \(instead of ([\d.]+)\)", cf_str)
                if m1:
                    cf1_count += 1
                    feat     = m1.group(1)
                    cf_val   = float(m1.group(2))
                    orig_val = float(m1.group(3))
                    if feat in cf_features:
                        cf_features[feat] += 1
                    cf_rel_changes.append(abs(cf_val - orig_val) / max(abs(orig_val), 1e-9))
                elif cf_str.startswith("Action is "):
                    cf2_count += 1
                    cf_features["override"] += 1
                    m2 = re.search(r"if (\w+) were ([\d.]+) \(instead of ([\d.]+)\)", cf_str)
                    if m2:
                        feat2     = m2.group(1)
                        cf_val2   = float(m2.group(2))
                        orig_val2 = float(m2.group(3))
                        if feat2 in cf_features:
                            cf_features[feat2] += 1
                        cf_rel_changes.append(abs(cf_val2 - orig_val2) / max(abs(orig_val2), 1e-9))
                if cf_str:
                    cf_any += 1

    ratio = anomalies_count / total_packets if total_packets else 0.0
    print(f"total packets: {total_packets}")
    print(f"anomalies count: {anomalies_count}")
    print(f"ratio: {ratio:.6f}")

    if has_ara_col:
        ara_total = sum(ara_counts.values())
        print("ara_active breakdown:")
        for val, count in ara_counts.items():
            pct = count / ara_total * 100 if ara_total else 0.0
            print(f"  {val}: {count} packets ({pct:.2f}%)")

    if has_path_col:
        detection_total = sum(path_counts.values())
        print("detection path breakdown:")
        for path, count in path_counts.items():
            pct = count / detection_total * 100 if detection_total else 0.0
            print(f"  {path}: {count} packets ({pct:.2f}%)")

    if has_action_col:
        action_total = sum(action_counts.values())
        print("action breakdown:")
        for act in ["ALLOW", "MONITOR", "ALERT_LOW", "ALERT_HIGH", "RATE_LIMIT", "BLOCK"]:
            count = action_counts.get(act, 0)
            pct = count / action_total * 100 if action_total else 0.0
            print(f"  {act:<12}{count} ({pct:.2f}%)")

    if has_cf3_cols or has_cf_col:
        cf_not_found = cf_total - cf_any
        any_pct      = cf_any      / cf_total * 100 if cf_total else 0.0
        no_pct       = cf_not_found / cf_total * 100 if cf_total else 0.0
        print("counterfactual coverage:")
        print(f"  Total ANOMALY packets: {cf_total}")
        print(f"  With any CF:           {cf_any} ({any_pct:.2f}%)")
        print(f"  Without any CF:        {cf_not_found} ({no_pct:.2f}%)")
        if has_cf3_cols:
            p14 = cf1_count + cf4_count
            p14_pct = p14 / cf_total * 100 if cf_total else 0.0
            p2_pct  = cf2_count / cf_total * 100 if cf_total else 0.0
            p3_pct  = cf3_change / cf_total * 100 if cf_total else 0.0
            print(f"  PASS 1 (single-feat):  {cf1_count}")
            print(f"  PASS 4 (multi-feat):   {cf4_count}  (combined P1+P4: {p14} / {p14_pct:.2f}%)")
            print(f"  PASS 2 (override):     {cf2_count} ({p2_pct:.2f}%)")
            print(f"  PASS 3 temporal insight: {cf3_change} changed action ({p3_pct:.2f}%) | {cf3_same} same action")
        else:
            print(f"  PASS 1: {cf1_count} | PASS 2: {cf2_count}")
        print("most common feature changed:")
        for feat in ["score", "risk", "freq", "override"]:
            print(f"  {feat}: {cf_features[feat]} times")
        avg_rel = sum(cf_rel_changes) / len(cf_rel_changes) if cf_rel_changes else 0.0
        print(f"average relative change to flip action: {avg_rel:.4f}")

    return ratio


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize IDS anomaly CSV output.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default=str(DEFAULT_RESULTS_FILE),
        help="CSV file to analyze.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze(Path(args.csv_file).resolve())


if __name__ == "__main__":
    main()
