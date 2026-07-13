"""
evaluation/unsw_dataset.py
===========================
Loads and splits the UNSW-NB15 training-set CSV into stratified
train / test DataFrames.  All label/feature separation happens here.

Returns plain numpy arrays — no pandas dependency in downstream modules.
"""

from __future__ import annotations
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from evaluation.config import ExperimentConfig, DEFAULT_CONFIG


@dataclass
class UNSWDataset:
    """Container returned by load_unsw_nb15()."""
    # raw string values for categoricals + float for numerics (train split)
    X_train_raw:   List[dict]   # list of row dicts, features only
    X_test_raw:    List[dict]
    y_train:       np.ndarray   # int (0=Normal, 1=Attack)
    y_test:        np.ndarray
    cat_train:     List[str]    # attack_cat for test rows
    cat_test:      List[str]
    feature_names: List[str]    # ordered list of feature column names
    n_total:       int
    n_normal:      int
    n_anomaly:     int


def load_unsw_nb15(cfg: ExperimentConfig = DEFAULT_CONFIG) -> UNSWDataset:
    """
    Load UNSW-NB15 CSV, validate, and return a stratified split.

    Raises FileNotFoundError if the CSV is absent.
    Raises ValueError if required columns are missing.
    """
    path = Path(cfg.unsw_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 CSV not found: {path}\n"
            "Download UNSW_NB15_training-set.csv and place it at that path."
        )

    rows_normal:  list = []
    rows_anomaly: list = []
    feature_names: list | None = None

    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        _validate_columns(fields, cfg)

        feature_names = _derive_feature_names(fields, cfg)

        for row in reader:
            label_val = row[cfg.label_col].strip()
            if label_val not in ("0", "1"):
                continue   # skip malformed rows

            feat_row = {k: row[k] for k in feature_names}
            cat      = row.get(cfg.attack_cat_col, "").strip()
            label    = int(label_val)

            if label == 0:
                rows_normal.append((feat_row, label, cat))
            else:
                rows_anomaly.append((feat_row, label, cat))

    if not rows_normal or not rows_anomaly:
        raise ValueError(
            "Dataset contains only one class. "
            f"Normal={len(rows_normal)}, Anomaly={len(rows_anomaly)}"
        )

    # Stratified split — same fraction from each class
    rng = random.Random(cfg.random_seed)
    train_rows, test_rows = _stratified_split(
        rows_normal, rows_anomaly, cfg.test_fraction, rng
    )

    X_train_raw, y_train_list, cat_train = zip(*train_rows)
    X_test_raw,  y_test_list,  cat_test  = zip(*test_rows)

    return UNSWDataset(
        X_train_raw   = list(X_train_raw),
        X_test_raw    = list(X_test_raw),
        y_train       = np.array(y_train_list, dtype=int),
        y_test        = np.array(y_test_list,  dtype=int),
        cat_train     = list(cat_train),
        cat_test      = list(cat_test),
        feature_names = feature_names,
        n_total       = len(rows_normal) + len(rows_anomaly),
        n_normal      = len(rows_normal),
        n_anomaly     = len(rows_anomaly),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_columns(fields: list, cfg: ExperimentConfig) -> None:
    required = {cfg.label_col, cfg.id_col, cfg.attack_cat_col}
    missing  = required - set(fields)
    if missing:
        raise ValueError(
            f"UNSW-NB15 CSV missing required columns: {missing}\n"
            f"Found: {fields}"
        )


def _derive_feature_names(fields: list, cfg: ExperimentConfig) -> list:
    """Return ordered feature column names (drop id, label, attack_cat)."""
    drop = {cfg.id_col, cfg.label_col, cfg.attack_cat_col}
    return [f for f in fields if f not in drop]


def _stratified_split(
    normal: list, anomaly: list, test_frac: float, rng: random.Random
) -> Tuple[list, list]:
    """Shuffle each class independently, then split by test_frac."""
    n_copy = list(normal)
    a_copy = list(anomaly)
    rng.shuffle(n_copy)
    rng.shuffle(a_copy)

    n_test_n = max(1, int(len(n_copy) * test_frac))
    n_test_a = max(1, int(len(a_copy) * test_frac))

    test  = n_copy[:n_test_n]  + a_copy[:n_test_a]
    train = n_copy[n_test_n:]  + a_copy[n_test_a:]

    # Shuffle combined splits so order is random
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def print_dataset_summary(ds: UNSWDataset) -> None:
    w = 55
    sep = "+" + "-" * w + "+"
    print(sep)
    print(f"|{'  UNSW-NB15 Dataset Summary':^{w}}|")
    print(sep)
    print(f"|  Total rows:          {ds.n_total:>8,}{'':>{w - 31}}|")
    print(f"|  Normal flows:        {ds.n_normal:>8,}  ({ds.n_normal/ds.n_total*100:5.1f}%){'':>{w - 43}}|")
    print(f"|  Attack flows:        {ds.n_anomaly:>8,}  ({ds.n_anomaly/ds.n_total*100:5.1f}%){'':>{w - 43}}|")
    print(f"|  Feature columns:     {len(ds.feature_names):>8,}{'':>{w - 31}}|")
    print(f"|  Train samples:       {len(ds.X_train_raw):>8,}{'':>{w - 31}}|")
    print(f"|  Test  samples:       {len(ds.X_test_raw):>8,}{'':>{w - 31}}|")
    print(sep)
