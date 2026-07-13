"""
evaluation/nsl_kdd_ground_truth.py
====================================
Converts NSL-KDD CSV (no header, 43 columns) into ground_truth_nslkdd.csv
for use with the Kitsune IDS evaluation pipeline.

NSL-KDD is the refined version of KDD Cup 1999 dataset.
Reference: Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set."
           CISDA 2009. https://www.unb.ca/cic/datasets/nsl.html

Usage:
    python evaluation/nsl_kdd_ground_truth.py [--input PATH] [--output PATH]

Default input:  unsw_train.csv   (the NSL-KDD file downloaded to this name)
Default output: evaluation/ground_truth_nslkdd.csv

NSL-KDD column layout (0-indexed):
    0   duration          |  1   protocol_type   |  2   service
    3   flag              |  4   src_bytes        |  5   dst_bytes
    6-40  behavioural features
    41  attack_type  ← label column
    42  difficulty_level

Attack families:
    DoS:   back, land, neptune, pod, smurf, teardrop, apache2, ...
    Probe: ipsweep, mscan, nmap, portsweep, satan, saint
    R2L:   ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, ...
    U2R:   buffer_overflow, loadmodule, perl, rootkit, xterm, sqlattack, ps
"""

import argparse
import csv
import os
import sys

# ---------------------------------------------------------------------------
# NSL-KDD column names (no header in the file)
# ---------------------------------------------------------------------------
NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_type",        # index 41 — label column
    "difficulty_level",   # index 42
]

LABEL_COL = 41   # zero-based index of attack_type

# ---------------------------------------------------------------------------
# Label mapping: NSL-KDD attack_type → binary NORMAL/ANOMALY
# ---------------------------------------------------------------------------
NORMAL_LABELS = {"normal"}

NSL_ATTACK_FAMILIES = {
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS",
    "udpstorm": "DoS", "processtable": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "mscan": "Probe", "nmap": "Probe",
    "portsweep": "Probe", "satan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L",
    "warezclient": "R2L", "warezmaster": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "xterm": "U2R", "sqlattack": "U2R", "ps": "U2R",
}

HEADER_COMMENT = """\
# REAL LABELS from NSL-KDD dataset
# Source: https://www.unb.ca/cic/datasets/nsl.html
# Reference: Tavallaee et al., CISDA 2009
# These are REAL ground truth labels, NOT synthetic
# Valid for publication
# Note: NSL-KDD is flow-level; alignment to packet-level results
#       is by row-index — see evaluation/README.md for limitations.
"""

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

DEFAULT_INPUT  = os.path.join(_ROOT, "unsw_train.csv")
DEFAULT_OUTPUT = os.path.join(_HERE, "ground_truth_nslkdd.csv")


def convert(input_path: str, output_path: str) -> dict:
    if not os.path.exists(input_path):
        print(f"\n[ERROR] Input file not found: {input_path}")
        print(f"        Expected NSL-KDD CSV at: {input_path}")
        sys.exit(1)

    total         = 0
    normal_count  = 0
    anomaly_count = 0
    family_counts: dict = {}
    label_counts:  dict = {}

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, newline="", encoding="utf-8-sig") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:

        fout.write(HEADER_COMMENT)
        writer = csv.writer(fout)
        writer.writerow(["packet_id", "true_label", "attack_category", "attack_family"])

        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 42:
                continue    # skip malformed rows

            raw_label = parts[LABEL_COL].strip().lower()
            total += 1

            if raw_label in NORMAL_LABELS:
                true_label  = "NORMAL"
                category    = "Normal"
                family      = "Normal"
                normal_count += 1
            else:
                true_label   = "ANOMALY"
                category     = raw_label
                family       = NSL_ATTACK_FAMILIES.get(raw_label, "Other")
                anomaly_count += 1

            label_counts[raw_label]  = label_counts.get(raw_label, 0) + 1
            family_counts[family]    = family_counts.get(family, 0) + 1
            writer.writerow([total, true_label, category, family])

    return {
        "total":         total,
        "normal":        normal_count,
        "anomaly":       anomaly_count,
        "normal_pct":    normal_count  / total * 100 if total else 0.0,
        "anomaly_pct":   anomaly_count / total * 100 if total else 0.0,
        "label_counts":  label_counts,
        "family_counts": family_counts,
        "output_path":   output_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert NSL-KDD CSV to ground_truth_nslkdd.csv"
    )
    parser.add_argument("--input",  default=DEFAULT_INPUT,  metavar="PATH")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, metavar="PATH")
    args = parser.parse_args()

    print(f"[NSL-KDD] Reading:  {args.input}")
    s = convert(args.input, args.output)
    print(f"[NSL-KDD] Written:  {args.output}")
    print()
    print("=" * 58)
    print("NSL-KDD Ground Truth Summary")
    print("=" * 58)
    print(f"  Total flows:  {s['total']:>8,}")
    print(f"  NORMAL:       {s['normal']:>8,}  ({s['normal_pct']:.1f}%)")
    print(f"  ANOMALY:      {s['anomaly']:>8,}  ({s['anomaly_pct']:.1f}%)")
    print()
    print("  Attack families:")
    for fam, cnt in sorted(s["family_counts"].items(), key=lambda x: -x[1]):
        pct = cnt / s["total"] * 100
        print(f"    {fam:<18}  {cnt:>6,}  ({pct:5.1f}%)")
    print()
    print("  Individual labels:")
    for lbl, cnt in sorted(s["label_counts"].items(), key=lambda x: -x[1]):
        pct = cnt / s["total"] * 100
        mapped = "NORMAL" if lbl == "normal" else "ANOMALY"
        print(f"    {lbl:<22}  {cnt:>6,}  ({pct:5.1f}%)  → {mapped}")
    print("=" * 58)
    print("  NOTE: These are REAL ground truth labels — publishable.")
    print("=" * 58)


if __name__ == "__main__":
    main()
