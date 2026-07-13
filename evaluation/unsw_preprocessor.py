"""
evaluation/unsw_preprocessor.py
=================================
Feature preprocessing for UNSW-NB15 data.

Pipeline (fit on train, applied to train + test):
  1. Parse all numeric columns to float (invalid/missing -> 0.0).
  2. One-hot encode categorical columns (proto, service, state).
  3. Clip inf / nan to 0.0.
  4. StandardScaler: zero-mean, unit-variance (fit on normal-traffic rows only,
     matching the KitNET philosophy of learning normal behaviour).

The fitted preprocessor is stored and reused for test data, preventing leakage.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from evaluation.config import ExperimentConfig, DEFAULT_CONFIG
from evaluation.unsw_dataset import UNSWDataset


@dataclass
class FittedPreprocessor:
    """Encapsulates all fitted state so it can be applied to new data."""
    feature_names:   List[str]           # original feature columns (before one-hot)
    cat_cols:        List[str]           # columns that were one-hot encoded
    numeric_cols:    List[str]           # columns that were scaled
    cat_vocab:       Dict[str, List[str]]  # col -> sorted list of seen categories
    scaler_mean:     np.ndarray
    scaler_std:      np.ndarray
    output_dim:      int                 # total features after encoding
    output_names:    List[str]           # names of every output feature


def fit_preprocessor(
    ds: UNSWDataset,
    cfg: ExperimentConfig = DEFAULT_CONFIG,
) -> FittedPreprocessor:
    """
    Fit the preprocessor on the TRAINING split of ds.
    The scaler is fit ONLY on normal (label=0) training rows to mimic
    the KitNET assumption that the baseline represents benign traffic.
    """
    cat_cols     = [c for c in cfg.cat_cols if c in ds.feature_names]
    numeric_cols = [c for c in ds.feature_names if c not in cat_cols]

    # Build vocabulary from ALL training rows (not just normal) to avoid
    # unknown-category errors on test data with attack-specific service values.
    cat_vocab: Dict[str, List[str]] = {}
    for col in cat_cols:
        seen = set()
        for row in ds.X_train_raw:
            seen.add(row.get(col, "").strip())
        cat_vocab[col] = sorted(seen)

    # Compute per-feature mean/std on NORMAL training rows only.
    normal_idx = np.where(ds.y_train == 0)[0]
    normal_rows = [ds.X_train_raw[i] for i in normal_idx]

    # We need the full encoded vectors to compute stats.
    enc_normal = _encode_rows(normal_rows, cat_cols, cat_vocab, numeric_cols)
    mean = enc_normal.mean(axis=0)
    std  = enc_normal.std(axis=0)
    std[std == 0] = 1.0   # avoid division by zero for constant features

    output_names = _build_output_names(cat_cols, cat_vocab, numeric_cols)

    return FittedPreprocessor(
        feature_names = ds.feature_names,
        cat_cols      = cat_cols,
        numeric_cols  = numeric_cols,
        cat_vocab     = cat_vocab,
        scaler_mean   = mean,
        scaler_std    = std,
        output_dim    = len(output_names),
        output_names  = output_names,
    )


def transform(
    rows: List[dict],
    prep: FittedPreprocessor,
) -> np.ndarray:
    """
    Apply the fitted preprocessor to a list of row dicts.
    Returns float32 array of shape (n_samples, output_dim).
    """
    enc = _encode_rows(rows, prep.cat_cols, prep.cat_vocab, prep.numeric_cols)
    scaled = (enc - prep.scaler_mean) / prep.scaler_std
    return scaled.astype(np.float32)


def transform_single(row: dict, prep: FittedPreprocessor) -> np.ndarray:
    """Transform one row dict.  Returns shape (output_dim,) float32 array."""
    return transform([row], prep)[0]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_float(value: str) -> float:
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except (ValueError, TypeError):
        return 0.0


def _encode_rows(
    rows: List[dict],
    cat_cols: List[str],
    cat_vocab: Dict[str, List[str]],
    numeric_cols: List[str],
) -> np.ndarray:
    """One-hot + numeric concat, no scaling applied."""
    n = len(rows)
    # Pre-compute total output width
    width = sum(len(cat_vocab[c]) for c in cat_cols) + len(numeric_cols)
    out = np.zeros((n, width), dtype=np.float64)

    for i, row in enumerate(rows):
        offset = 0
        for col in cat_cols:
            vocab = cat_vocab[col]
            val   = row.get(col, "").strip()
            if val in vocab:
                out[i, offset + vocab.index(val)] = 1.0
            # unknown category -> all zeros (handled gracefully)
            offset += len(vocab)
        for col in numeric_cols:
            out[i, offset] = _parse_float(row.get(col, "0"))
            offset += 1

    return out


def _build_output_names(
    cat_cols: List[str],
    cat_vocab: Dict[str, List[str]],
    numeric_cols: List[str],
) -> List[str]:
    names = []
    for col in cat_cols:
        for val in cat_vocab[col]:
            names.append(f"{col}_{val}")
    names.extend(numeric_cols)
    return names
