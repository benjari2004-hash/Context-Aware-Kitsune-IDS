"""
evaluation/unsw_evaluator.py
=============================
Metrics computation for the UNSW-NB15 evaluation pipeline.

All metrics are computed manually; sklearn is optional (used only for ROC-AUC
if available, falls back to a numpy trapezoidal implementation).

Positive class = ANOMALY (label == 1).
"""

from __future__ import annotations
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class EvalMetrics:
    """All evaluation metrics for one system run."""
    system_name:       str
    tp:                int
    fp:                int
    tn:                int
    fn:                int
    accuracy:          float
    precision_anomaly: float
    recall_anomaly:    float    # TPR / detection rate
    f1_anomaly:        float
    precision_normal:  float
    recall_normal:     float
    f1_normal:         float
    macro_f1:          float
    fpr:               float    # false positive rate
    fnr:               float    # false negative rate / miss rate
    roc_auc:           float
    threshold:         float    # anomaly score threshold used
    n_test:            int


def evaluate(
    system_name: str,
    y_true: np.ndarray,       # int array, 1=attack 0=normal
    y_pred: np.ndarray,       # int array, 1=anomaly 0=normal
    scores: np.ndarray,       # float array, higher = more anomalous
    threshold: float,
) -> EvalMetrics:
    """Compute full metrics from binary arrays."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec_a = _safe_div(tp, tp + fp)
    rec_a  = _safe_div(tp, tp + fn)
    f1_a   = _safe_div(2 * prec_a * rec_a, prec_a + rec_a)

    prec_n = _safe_div(tn, tn + fn)
    rec_n  = _safe_div(tn, tn + fp)
    f1_n   = _safe_div(2 * prec_n * rec_n, prec_n + rec_n)

    fpr     = _safe_div(fp, fp + tn)
    fnr     = _safe_div(fn, fn + tp)
    acc     = _safe_div(tp + tn, tp + fp + tn + fn)
    macro_f = (f1_a + f1_n) / 2.0

    has_both = y_true.sum() > 0 and (y_true == 0).sum() > 0
    roc_auc  = _compute_auc(scores, y_true) if has_both else 0.5

    return EvalMetrics(
        system_name       = system_name,
        tp                = tp,
        fp                = fp,
        tn                = tn,
        fn                = fn,
        accuracy          = acc,
        precision_anomaly = prec_a,
        recall_anomaly    = rec_a,
        f1_anomaly        = f1_a,
        precision_normal  = prec_n,
        recall_normal     = rec_n,
        f1_normal         = f1_n,
        macro_f1          = macro_f,
        fpr               = fpr,
        fnr               = fnr,
        roc_auc           = roc_auc,
        threshold         = threshold,
        n_test            = len(y_true),
    )


def print_metrics(m: EvalMetrics) -> None:
    W   = 56
    sep = "+" + "-" * W + "+"

    def row(label, value):
        return f"|  {label:<30}{value:>22}|"

    print(sep)
    print(f"|{'  ' + m.system_name:^{W}}|")
    print(sep)
    print(f"|{'  Confusion Matrix':^{W}}|")
    print(sep)
    print(row("TP (True Positives)",  f"{m.tp}"))
    print(row("FP (False Positives)", f"{m.fp}"))
    print(row("TN (True Negatives)",  f"{m.tn}"))
    print(row("FN (False Negatives)", f"{m.fn}"))
    print(sep)
    print(f"|{'  Detection Metrics (ANOMALY class)':^{W}}|")
    print(sep)
    print(row("Precision",            f"{m.precision_anomaly:.4f}"))
    print(row("Recall (TPR)",         f"{m.recall_anomaly:.4f}"))
    print(row("F1-score",             f"{m.f1_anomaly:.4f}"))
    print(sep)
    print(f"|{'  Aggregate':^{W}}|")
    print(sep)
    print(row("Accuracy",             f"{m.accuracy:.4f}"))
    print(row("Macro F1",             f"{m.macro_f1:.4f}"))
    print(row("False Positive Rate",  f"{m.fpr:.4f}"))
    print(row("False Negative Rate",  f"{m.fnr:.4f}"))
    print(row("ROC AUC",              f"{m.roc_auc:.4f}"))
    print(row("Threshold",            f"{m.threshold:.6f}"))
    print(sep)


def save_metrics_csv(metrics_list: List[EvalMetrics], path: Path) -> None:
    """Write a list of EvalMetrics objects to CSV (one row per system)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_list:
        return
    fieldnames = list(asdict(metrics_list[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow(asdict(m))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _compute_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """ROC-AUC via sklearn (preferred) or numpy trapz fallback."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except ImportError:
        pass

    # Manual implementation
    thresholds = np.unique(np.concatenate([scores, [np.inf]]))[::-1]
    P = int(y_true.sum())
    N = len(y_true) - P
    fprs, tprs = [], []
    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        tp    = int((preds * y_true).sum())
        fp    = int((preds * (1 - y_true)).sum())
        tprs.append(_safe_div(tp, P))
        fprs.append(_safe_div(fp, N))
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))
