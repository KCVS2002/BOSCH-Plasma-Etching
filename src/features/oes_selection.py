"""Per-fold OES wavelength selection by train-fold correlation.

Selects top-k wavelengths most correlated with the wafer-level target on the
TRAIN fold only. No leakage: only train wafers feed the correlation.

Available statistics (per wafer × wavelength):
  - "mean":      global mean over time
  - "late_mean": mean over late cycles (default 80..100, 1-based)
  - "drift":     late_mean − early_mean (default early = cycles 1..20)

All statistics are computed on log1p-transformed counts to match the model's
input transform (`fit_oes_normalizer` applies log1p before z-scoring).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _per_wafer_wavelength_stat(
    cache_root: Path,
    wafer_key: str,
    stat: str,
    late_start_cycle: int = 80,
    early_end_cycle: int = 20,
) -> np.ndarray:
    """Return (W,) — one wavelength-aggregated stat for one wafer (log1p domain)."""
    npz = np.load(cache_root / "wafers" / f"{wafer_key}.npz", allow_pickle=False)
    raw = npz["oes_data"]
    data = np.log1p(np.maximum(raw.astype(np.float64), 0.0))  # (T_raw, W)

    if stat == "mean":
        return data.mean(axis=0).astype(np.float32)

    starts = npz["oes_cycle_starts_idx"]
    ends = npz["oes_cycle_ends_idx"]
    n_cycles = len(starts)

    if stat == "late_mean":
        ls = max(int(late_start_cycle) - 1, 0)
        s = int(starts[ls])
        e = int(ends[n_cycles - 1])
        return data[s:e].mean(axis=0).astype(np.float32)

    if stat == "drift":
        ls = max(int(late_start_cycle) - 1, 0)
        ee = min(max(int(early_end_cycle), 1), n_cycles)
        s_late = int(starts[ls])
        e_late = int(ends[n_cycles - 1])
        s_early = int(starts[0])
        e_early = int(ends[ee - 1])
        late = data[s_late:e_late].mean(axis=0)
        early = data[s_early:e_early].mean(axis=0)
        return (late - early).astype(np.float32)

    raise ValueError(f"unknown stat {stat!r} (expected mean | late_mean | drift)")


def compute_oes_wavelength_scores(
    cache_root: Path,
    train_keys: Sequence[str],
    meas: pd.DataFrame,
    target: str,
    stat: str = "late_mean",
    late_start_cycle: int = 80,
    early_end_cycle: int = 20,
    cache_path: Path | None = None,
) -> np.ndarray:
    """|Pearson corr| of shape (W,) — wavelength stat vs wafer-mean target.

    Uses TRAIN wafers only.
    """
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path).astype(np.float32, copy=False)

    feats: list[np.ndarray] = []
    targets: list[float] = []
    for k in train_keys:
        feats.append(_per_wafer_wavelength_stat(
            cache_root, k, stat,
            late_start_cycle=late_start_cycle,
            early_end_cycle=early_end_cycle,
        ))
        meas_w = meas[meas["experiment_key"] == k]
        targets.append(float(meas_w[target].mean()))

    X = np.asarray(feats, dtype=np.float64)            # (n_train, W)
    y = np.asarray(targets, dtype=np.float64)          # (n_train,)

    X_c = X - X.mean(axis=0, keepdims=True)
    y_c = y - y.mean()
    num = (X_c * y_c[:, None]).sum(axis=0)             # (W,)
    den = np.sqrt((X_c ** 2).sum(axis=0) * (y_c ** 2).sum())
    corr = num / np.maximum(den, 1e-12)
    scores = np.abs(corr).astype(np.float32)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, scores)
    return scores


def oes_score_cache_path(
    cache_root: Path,
    train_keys: Sequence[str],
    target: str,
    stat: str,
    late_start_cycle: int,
    early_end_cycle: int,
) -> Path:
    payload = {
        "train_keys": sorted(str(k) for k in train_keys),
        "target": str(target),
        "stat": str(stat),
        "late_start_cycle": int(late_start_cycle),
        "early_end_cycle": int(early_end_cycle),
    }
    digest = hashlib.sha1(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    name = (
        f"{target}_{stat}_ls{int(late_start_cycle)}_"
        f"ee{int(early_end_cycle)}_{digest}.npy"
    )
    return Path(cache_root) / "features" / "oes_scores" / name


def select_top_k_wavelengths(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return ascending-sorted indices of top-k wavelengths by `scores`."""
    n = len(scores)
    k = min(int(top_k), n)
    top_idx = np.argpartition(scores, n - k)[n - k:]
    return np.sort(top_idx).astype(np.int32)
