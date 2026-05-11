# evaluate.py
# Ground truth evaluation framework for the Kitsune IDS pipeline.
#
# Usage:
#   python evaluate.py --labels ground_truth.csv
#   python evaluate.py --results results.csv --labels ground_truth.csv
#
# Ground truth CSV format:
#   packet_id,true_label
#   1,NORMAL
#   2,ANOMALY
#   ...
#   true_label must be NORMAL or ANOMALY (case-insensitive).
#
# TRAIN predictions are treated as NORMAL for evaluation purposes.
# All metrics are implemented manually — sklearn is not required.

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULTS_FILE  = Path(__file__).with_name("results.csv")
ROC_FILE      = Path(__file__).with_name("evaluate_roc.png")


# ─────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────

def load_results(path):
    """
    Returns dict[packet_id -> dict(score, risk, threshold, label, attack_type)]
    TRAIN labels are normalised to NORMAL so the caller never sees TRAIN.
    """
    path = Path(path)
    if not path.exists():
        sys.exit(f"[evaluate] results file not found: {path}")

    results = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(
            line for line in fh if not line.startswith("#")
        )
        for row in reader:
            pid   = int(row["packet_id"])
            label = row["label"].strip().upper()
            if label == "TRAIN":
                label = "NORMAL"
            results[pid] = {
                "score":       float(row["score"]),
                "risk":        float(row["risk"]),
                "threshold":   float(row["threshold"]),
                "label":       label,
                "attack_type": row.get("attack_type", ""),
            }
    return results


def load_ground_truth(path):
    """
    Returns dict[packet_id -> "NORMAL" | "ANOMALY"].
    Validates that every true_label is one of those two values.
    """
    path = Path(path)
    if not path.exists():
        sys.exit(f"[evaluate] labels file not found: {path}")

    labels = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(
            line for line in fh if not line.startswith("#")
        )
        if "packet_id" not in reader.fieldnames or "true_label" not in reader.fieldnames:
            sys.exit(
                "[evaluate] labels CSV must have columns: packet_id, true_label"
            )
        for row in reader:
            pid  = int(row["packet_id"])
            lbl  = row["true_label"].strip().upper()
            if lbl not in ("NORMAL", "ANOMALY"):
                sys.exit(
                    f"[evaluate] unknown true_label '{row['true_label']}' "
                    f"at packet_id={pid}. Expected NORMAL or ANOMALY."
                )
            labels[pid] = lbl
    return labels


# ─────────────────────────────────────────────────────────
# Metric computation (no sklearn)
# ─────────────────────────────────────────────────────────

def confusion(y_true, y_pred):
    """
    y_true / y_pred: lists of "NORMAL" | "ANOMALY"
    Positive class = ANOMALY.
    Returns TP, FP, TN, FN.
    """
    TP = FP = TN = FN = 0
    for t, p in zip(y_true, y_pred):
        if t == "ANOMALY" and p == "ANOMALY":
            TP += 1
        elif t == "NORMAL" and p == "ANOMALY":
            FP += 1
        elif t == "NORMAL" and p == "NORMAL":
            TN += 1
        else:
            FN += 1
    return TP, FP, TN, FN


def safe_div(num, den):
    return num / den if den else 0.0


def per_class_metrics(TP, FP, TN, FN):
    precision_anom = safe_div(TP, TP + FP)
    recall_anom    = safe_div(TP, TP + FN)   # TPR
    f1_anom        = safe_div(
        2 * precision_anom * recall_anom,
        precision_anom + recall_anom,
    )

    precision_norm = safe_div(TN, TN + FN)
    recall_norm    = safe_div(TN, TN + FP)
    f1_norm        = safe_div(
        2 * precision_norm * recall_norm,
        precision_norm + recall_norm,
    )

    fpr = safe_div(FP, FP + TN)   # fall-out
    fnr = safe_div(FN, FN + TP)   # miss rate

    accuracy = safe_div(TP + TN, TP + FP + TN + FN)

    macro_precision = (precision_anom + precision_norm) / 2
    macro_recall    = (recall_anom    + recall_norm)    / 2
    macro_f1        = (f1_anom        + f1_norm)        / 2

    return {
        "precision_anomaly": precision_anom,
        "recall_anomaly":    recall_anom,
        "f1_anomaly":        f1_anom,
        "precision_normal":  precision_norm,
        "recall_normal":     recall_norm,
        "f1_normal":         f1_norm,
        "macro_precision":   macro_precision,
        "macro_recall":      macro_recall,
        "macro_f1":          macro_f1,
        "fpr":               fpr,
        "fnr":               fnr,
        "accuracy":          accuracy,
    }


# ─────────────────────────────────────────────────────────
# ROC curve
# ─────────────────────────────────────────────────────────

def _roc_numpy(scores, y_true_binary):
    """
    Manual ROC curve using numpy.
    scores          : float array (higher = more anomalous)
    y_true_binary   : int array (1=ANOMALY, 0=NORMAL)
    Returns fpr_arr, tpr_arr, auc.
    """
    thresholds = np.unique(np.concatenate([scores, [np.inf]]))[::-1]
    fprs, tprs = [], []
    P = y_true_binary.sum()
    N = len(y_true_binary) - P
    for thr in thresholds:
        preds      = (scores >= thr).astype(int)
        tp         = int((preds * y_true_binary).sum())
        fp         = int((preds * (1 - y_true_binary)).sum())
        tprs.append(safe_div(tp, P) if P else 0.0)
        fprs.append(safe_div(fp, N) if N else 0.0)
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    # AUC via trapezoidal rule
    order = np.argsort(fprs)
    auc   = float(np.trapz(tprs[order], fprs[order]))
    return fprs, tprs, auc


def plot_roc(scores, y_true, out_path):
    """
    Tries sklearn; falls back to numpy implementation.
    scores  : list[float]
    y_true  : list["NORMAL"|"ANOMALY"]
    """
    scores_arr = np.array(scores, dtype=float)
    binary     = np.array([1 if lbl == "ANOMALY" else 0 for lbl in y_true], dtype=int)

    try:
        from sklearn.metrics import roc_curve, auc as sk_auc
        fpr_arr, tpr_arr, _ = roc_curve(binary, scores_arr)
        roc_auc = sk_auc(fpr_arr, tpr_arr)
        source  = "sklearn"
    except ImportError:
        fpr_arr, tpr_arr, roc_auc = _roc_numpy(scores_arr, binary)
        source = "numpy"

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_arr, tpr_arr, color="steelblue", lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Kitsune IDS — ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return roc_auc, source


# ─────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────

def print_report(TP, FP, TN, FN, metrics, n_matched, n_results, n_labels, roc_auc):
    W = 54
    sep = "+" + "-" * W + "+"

    def row(label, value, width=W):
        line = f"  {label}"
        return f"|{line:<{width}}|" if value is None else f"|{line:<30}{value:>22}|"

    print(sep)
    print(f"|{'  Kitsune IDS — Evaluation Report':^{W}}|")
    print(sep)
    print(row("Packets in results.csv",  f"{n_results}"))
    print(row("Packets in labels file",  f"{n_labels}"))
    print(row("Matched (inner join)",     f"{n_matched}"))
    print(sep)
    print(f"|{'  Confusion Matrix (positive = ANOMALY)':^{W}}|")
    print(sep)
    print(row("True Positives  (TP)",  f"{TP}"))
    print(row("False Positives (FP)",  f"{FP}"))
    print(row("True Negatives  (TN)",  f"{TN}"))
    print(row("False Negatives (FN)",  f"{FN}"))
    print(sep)
    print(f"|{'  Per-class Metrics':^{W}}|")
    print(sep)
    print(row("Precision  (ANOMALY)", f"{metrics['precision_anomaly']:.4f}"))
    print(row("Recall     (ANOMALY)", f"{metrics['recall_anomaly']:.4f}"))
    print(row("F1         (ANOMALY)", f"{metrics['f1_anomaly']:.4f}"))
    print(row("Precision  (NORMAL)",  f"{metrics['precision_normal']:.4f}"))
    print(row("Recall     (NORMAL)",  f"{metrics['recall_normal']:.4f}"))
    print(row("F1         (NORMAL)",  f"{metrics['f1_normal']:.4f}"))
    print(sep)
    print(f"|{'  Aggregate Metrics':^{W}}|")
    print(sep)
    print(row("Macro Precision",      f"{metrics['macro_precision']:.4f}"))
    print(row("Macro Recall",         f"{metrics['macro_recall']:.4f}"))
    print(row("Macro F1",             f"{metrics['macro_f1']:.4f}"))
    print(row("Accuracy",             f"{metrics['accuracy']:.4f}"))
    print(row("False Positive Rate",  f"{metrics['fpr']:.4f}"))
    print(row("False Negative Rate",  f"{metrics['fnr']:.4f}"))
    if roc_auc is not None:
        print(row("ROC AUC",          f"{roc_auc:.4f}"))
    print(sep)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Kitsune IDS predictions against ground truth labels."
    )
    parser.add_argument(
        "--results", default=str(RESULTS_FILE), metavar="PATH",
        help="Path to results.csv (default: results.csv next to this script)",
    )
    parser.add_argument(
        "--labels", required=True, metavar="PATH",
        help="Path to ground truth CSV: packet_id,true_label",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    truth   = load_ground_truth(args.labels)

    # Inner join on packet_id
    common_ids = sorted(set(results) & set(truth))
    if not common_ids:
        sys.exit("[evaluate] No matching packet_ids between results and labels.")

    y_pred   = [results[pid]["label"] for pid in common_ids]
    y_true   = [truth[pid]            for pid in common_ids]
    scores   = [results[pid]["score"] for pid in common_ids]

    TP, FP, TN, FN = confusion(y_true, y_pred)
    metrics = per_class_metrics(TP, FP, TN, FN)

    # ROC curve
    roc_auc = None
    has_both_classes = any(l == "ANOMALY" for l in y_true) and any(l == "NORMAL" for l in y_true)
    if has_both_classes:
        roc_auc, roc_src = plot_roc(scores, y_true, ROC_FILE)
        print(f"[evaluate] ROC curve saved to {ROC_FILE} (via {roc_src}, AUC={roc_auc:.4f})")
    else:
        print("[evaluate] ROC curve skipped — ground truth contains only one class.")

    print_report(
        TP, FP, TN, FN, metrics,
        n_matched = len(common_ids),
        n_results = len(results),
        n_labels  = len(truth),
        roc_auc   = roc_auc,
    )


if __name__ == "__main__":
    main()
