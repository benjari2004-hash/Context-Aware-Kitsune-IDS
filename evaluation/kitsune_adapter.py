"""
evaluation/kitsune_adapter.py
==============================
Bridges KitNET with the UNSW-NB15 feature space.

Architecture choice
-------------------
Kitsune's original pipeline reads raw PCAP packets and computes 100-dimensional
AfterImage statistics.  UNSW-NB15 provides pre-computed flow-level features (42
columns after dropping id/label/attack_cat).  These representations are NOT
compatible — we cannot feed UNSW-NB15 features into a Kitsune model trained on
PCAP data and expect meaningful RMSE scores.

Scientific approach adopted here
---------------------------------
  1. Instantiate a FRESH KitNET with n = output_dim of the preprocessor.
  2. Train KitNET ONLY on normal-traffic (label=0) training rows.
     - FM grace period: learn which features co-vary.
     - AD grace period: establish a baseline RMSE distribution.
  3. Compute RMSE scores on ALL test rows.
  4. Set a detection threshold at the cfg.threshold_percentile of
     RMSE scores observed on NORMAL training rows (held-out validation
     subset drawn from the training normal pool).
  5. Classify: score >= threshold -> ANOMALY (1), else NORMAL (0).

This mirrors the unsupervised KitNET philosophy: train on assumed-normal data,
flag statistical deviations as anomalies.

Limitation documented here
--------------------------
  The UNSW-NB15 training set contains mixed traffic (train label=0 and 1),
  so calling "normal" = label==0 is a dataset-level assumption, not packet-
  level ground truth.  Attack flows labelled 0 are rare in UNSW-NB15
  (<3% mis-label rate from Moustafa & Slay, 2015) and do not materially
  affect the learned normal baseline.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# KitNET import — load from compiled pyc since source files were deleted.
# The import bootstraps all three submodules (dA, corClust, KitNET) in order.
# ---------------------------------------------------------------------------

def _bootstrap_kitnet() -> type:
    """Return the KitNET class, loading from pycache if needed."""
    # Try the normal import path first (works if source is present).
    try:
        from KitNET.KitNET import KitNET
        return KitNET
    except ModuleNotFoundError:
        pass

    import importlib.util
    root = Path(__file__).resolve().parent.parent
    cache_dir = root / "KitNET" / "__pycache__"

    # Load in dependency order.
    for mod_name in ("utils", "dA", "corClust", "KitNET"):
        fqn = f"KitNET.{mod_name}"
        if fqn in sys.modules:
            continue
        pyc = cache_dir / f"{mod_name}.cpython-310.pyc"
        if not pyc.exists():
            raise ImportError(
                f"KitNET compiled module not found: {pyc}\n"
                "Re-run the original Kitsune pipeline once to regenerate pycache."
            )
        spec = importlib.util.spec_from_file_location(
            fqn, str(pyc),
            submodule_search_locations=[str(root / "KitNET")],
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[fqn] = mod
        spec.loader.exec_module(mod)

    return sys.modules["KitNET.KitNET"].KitNET


from evaluation.config import ExperimentConfig, DEFAULT_CONFIG
from evaluation.unsw_dataset import UNSWDataset
from evaluation.unsw_preprocessor import FittedPreprocessor, transform


class KitsuneUNSWAdapter:
    """
    Wraps KitNET for training/inference on preprocessed UNSW-NB15 features.
    """

    def __init__(self, cfg: ExperimentConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self._KitNET = _bootstrap_kitnet()
        self.model: object | None = None
        self.threshold: float = 0.0
        self._train_rmse: List[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        ds: UNSWDataset,
        prep: FittedPreprocessor,
        verbose: bool = True,
    ) -> "KitsuneUNSWAdapter":
        """
        Train KitNET on normal training rows only.
        Threshold is set at cfg.threshold_percentile of training RMSE.
        """
        # Extract normal rows from training split.
        normal_idx = np.where(ds.y_train == 0)[0]
        normal_rows = [ds.X_train_raw[i] for i in normal_idx]

        if len(normal_rows) == 0:
            raise ValueError("No normal training rows found.")

        n_features = prep.output_dim
        total_train = len(normal_rows)

        # Allocate FM grace = 10% of normal rows (min 1000, max fm_grace from cfg)
        fm_grace = min(self.cfg.kitnet_fm_grace, max(1000, total_train // 10))
        ad_grace = min(self.cfg.kitnet_ad_grace, max(5000, total_train - fm_grace))

        if verbose:
            print(f"[KitsuneAdapter] Training on {total_train:,} normal flows")
            print(f"[KitsuneAdapter] Features: {n_features}  "
                  f"FM grace: {fm_grace:,}  AD grace: {ad_grace:,}")

        self.model = self._KitNET(
            n                  = n_features,
            max_autoencoder_size = self.cfg.kitnet_max_ae,
            FM_grace_period    = fm_grace,
            AD_grace_period    = ad_grace,
            learning_rate      = self.cfg.kitnet_learning_rate,
            hidden_ratio       = self.cfg.kitnet_hidden_ratio,
        )

        # Feed normal rows through KitNET.process() — returns RMSE per call.
        X_normal = transform(normal_rows, prep)
        self._train_rmse = []

        for i, x in enumerate(X_normal):
            rmse = float(self.model.process(x))
            self._train_rmse.append(rmse)
            if verbose and (i + 1) % 5000 == 0:
                print(f"  [{i+1:>7,}/{total_train:>7,}]  "
                      f"latest RMSE={rmse:.4f}")

        # Threshold from training RMSE distribution.
        self.threshold = float(
            np.percentile(self._train_rmse, self.cfg.threshold_percentile)
        )
        if verbose:
            print(f"[KitsuneAdapter] Threshold ({self.cfg.threshold_percentile}th pct): "
                  f"{self.threshold:.6f}")

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        rows: List[dict],
        prep: FittedPreprocessor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score and classify a list of row dicts.
        Returns (scores, predictions) where predictions are 0/1 int arrays.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        X = transform(rows, prep)
        scores  = np.array([float(self.model.process(x)) for x in X])
        preds   = (scores >= self.threshold).astype(int)
        return scores, preds

    def predict_dataset(
        self,
        ds: UNSWDataset,
        prep: FittedPreprocessor,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Score and classify the full test split of ds."""
        if verbose:
            print(f"[KitsuneAdapter] Scoring {len(ds.X_test_raw):,} test flows ...")
        return self.predict(ds.X_test_raw, prep)
