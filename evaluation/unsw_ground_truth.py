"""
evaluation/unsw_ground_truth.py
================================
Converts UNSW-NB15 training-set CSV into ground_truth_unsw.csv for the
Kitsune IDS evaluation pipeline.

Usage:
    python evaluation/unsw_ground_truth.py [--input PATH] [--output PATH]

Default input:  unsw_train.csv   (project root)
Default output: evaluation/ground_truth_unsw.csv

UNSW-NB15 source:
    Moustafa, N. & Slay, J. (2015). UNSW-NB15: a comprehensive data set
    for network intrusion detection systems.  MilCIS 2015.
    https://research.unsw.edu.au/projects/unsw-nb15-dataset

How to obtain unsw_train.csv
-----------------------------
Option A — Kaggle (recommended):
    1. Create a free account at https://www.kaggle.com
    2. Search: "UNSW-NB15 dataset" (dataset by mrwellsdavid)
    3. Download UNSW_NB15_training-set.csv
    4. Place at: D:\\Kitsune-py-master\\Kitsune-py-master\\unsw_train.csv

Option B — UNSW official page:
    https://research.unsw.edu.au/projects/unsw-nb15-dataset
    Download "UNSW_NB15_training-set.csv" from the CSV files section.
"""

import argparse
import csv
import os
import sys

# ---------------------------------------------------------------------------
# Label mapping: UNSW-NB15 attack_cat  →  our binary label
# ---------------------------------------------------------------------------
UNSW_TO_OURS = {
    "Normal":          "NORMAL",
    "normal":          "NORMAL",
    "Fuzzers":         "ANOMALY",
    "Analysis":        "ANOMALY",
    "Backdoors":       "ANOMALY",
    "Backdoor":        "ANOMALY",
    "DoS":             "ANOMALY",
    "Exploits":        "ANOMALY",
    "Generic":         "ANOMALY",
    "Reconnaissance":  "ANOMALY",
    "Shellcode":       "ANOMALY",
    "Worms":           "ANOMALY",
}

HEADER_COMMENT = """\
# REAL LABELS from UNSW-NB15 dataset
# Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
# Reference: Moustafa & Slay, MilCIS 2015
# These are REAL ground truth labels, NOT synthetic
# Valid for publication
# Label column: binary (0=Normal, 1=Attack) mapped to NORMAL/ANOMALY
# attack_cat column: fine-grained category preserved in attack_category column
"""

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

DEFAULT_INPUT  = os.path.join(_ROOT, "unsw_train.csv")
DEFAULT_OUTPUT = os.path.join(_HERE, "ground_truth_unsw.csv")


def _detect_label_columns(fieldnames):
    """
    UNSW-NB15 has two label columns:
      'label'      — binary int (0=Normal, 1=Attack)
      'attack_cat' — string category (empty string for Normal rows)
    Return (binary_col, category_col) or raise ValueError.
    """
    lower = [f.strip().lower() for f in fieldnames]
    binary_col   = None
    category_col = None

    for orig, lo in zip(fieldnames, lower):
        if lo == "label":
            binary_col = orig
        elif lo in ("attack_cat", "attack_category", "category", "attackcategory"):
            category_col = orig

    if binary_col is None:
        raise ValueError(
            f"Cannot find 'label' column.  Available: {fieldnames}"
        )
    return binary_col, category_col


def convert(input_path: str, output_path: str) -> dict:
    """
    Read UNSW-NB15 CSV and write ground_truth_unsw.csv.
    Returns a summary dict.
    """
    if not os.path.exists(input_path):
        print(f"\n[ERROR] Input file not found: {input_path}")
        print(__doc__)
        sys.exit(1)

    total = 0
    normal_count = 0
    anomaly_count = 0
    unknown_count = 0
    category_counts: dict = {}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, newline="", encoding="utf-8-sig") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        fout.write(HEADER_COMMENT)
        reader  = csv.DictReader(fin)
        binary_col, category_col = _detect_label_columns(reader.fieldnames or [])
        writer  = csv.writer(fout)
        writer.writerow(["packet_id", "true_label", "attack_category"])

        for i, row in enumerate(reader, start=1):
            total += 1

            # Prefer category column; fall back to binary label
            if category_col and row.get(category_col, "").strip():
                raw_cat = row[category_col].strip()
            else:
                raw_cat = "Normal" if str(row.get(binary_col, "0")).strip() == "0" else "Generic"

            true_label = UNSW_TO_OURS.get(raw_cat)
            if true_label is None:
                # Try case-insensitive lookup
                for key, val in UNSW_TO_OURS.items():
                    if key.lower() == raw_cat.lower():
                        true_label = val
                        break
            if true_label is None:
                true_label = "ANOMALY"   # conservative: unknown categories treated as attacks
                unknown_count += 1

            if true_label == "NORMAL":
                normal_count += 1
            else:
                anomaly_count += 1

            category_counts[raw_cat] = category_counts.get(raw_cat, 0) + 1
            writer.writerow([i, true_label, raw_cat])

    normal_pct  = normal_count  / total * 100 if total else 0.0
    anomaly_pct = anomaly_count / total * 100 if total else 0.0

    return {
        "total":          total,
        "normal":         normal_count,
        "anomaly":        anomaly_count,
        "unknown_mapped": unknown_count,
        "normal_pct":     normal_pct,
        "anomaly_pct":    anomaly_pct,
        "categories":     category_counts,
        "output_path":    output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert UNSW-NB15 CSV to ground truth")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  metavar="PATH",
                        help=f"Path to UNSW_NB15_training-set.csv (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, metavar="PATH",
                        help=f"Path for output ground_truth_unsw.csv (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    print(f"[UNSW-NB15] Reading:  {args.input}")
    summary = convert(args.input, args.output)
    print(f"[UNSW-NB15] Written:  {args.output}")
    print()
    print("=" * 55)
    print("UNSW-NB15 Ground Truth Summary")
    print("=" * 55)
    print(f"  Total flows:   {summary['total']:>8,}")
    print(f"  NORMAL:        {summary['normal']:>8,}  ({summary['normal_pct']:.1f}%)")
    print(f"  ANOMALY:       {summary['anomaly']:>8,}  ({summary['anomaly_pct']:.1f}%)")
    if summary["unknown_mapped"]:
        print(f"  Unknown→ANOMALY: {summary['unknown_mapped']:>6,}  (conservative mapping)")
    print()
    print("  Attack categories:")
    for cat, cnt in sorted(summary["categories"].items(), key=lambda x: -x[1]):
        pct = cnt / summary["total"] * 100
        label = UNSW_TO_OURS.get(cat, "ANOMALY")
        print(f"    {cat:<22}  {cnt:>7,}  ({pct:5.1f}%)  → {label}")
    print("=" * 55)
    print("  NOTE: These are REAL ground truth labels — publishable.")
    print("=" * 55)


if __name__ == "__main__":
    main()
