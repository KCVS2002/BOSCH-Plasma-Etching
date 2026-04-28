"""Thin loader for split files written by `scripts/02_make_splits.py`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Split:
    """A loaded CV split file.

    Attributes:
        sample_fold_id: (N_samples,) int — fold each sample is val for.
        wafer_fold_id:  (N_wafers,)  int — fold each wafer is val for.
        wafer_keys:     (N_wafers,)  str — canonical wafer order.
        n_folds:        int
        method:         description string
    """
    sample_fold_id: np.ndarray
    wafer_fold_id: np.ndarray
    wafer_keys: np.ndarray
    n_folds: int
    method: str
    extras: dict

    def train_val_masks(self, fold: int) -> tuple[np.ndarray, np.ndarray]:
        val = self.sample_fold_id == fold
        train = ~val
        return train, val


def load_split(path: Path) -> Split:
    z = np.load(path, allow_pickle=False)
    extras = {k: z[k] for k in z.files
              if k not in {"sample_fold_id", "wafer_fold_id",
                           "wafer_keys", "n_folds", "method"}}
    return Split(
        sample_fold_id=z["sample_fold_id"],
        wafer_fold_id=z["wafer_fold_id"],
        wafer_keys=z["wafer_keys"],
        n_folds=int(z["n_folds"]),
        method=str(z["method"]),
        extras=extras,
    )
