"""Per-cycle aggregation of OES + Process arrays into compact statistics.

Designed for the XGBoost baseline (Phase 2). The DL pipeline (Phase 3) will
consume the raw cycle slices instead — these helpers exist so traditional ML
can ingest the data without exploding into 14k×3648 dimensions.

NaN handling: ~13 process channels (e.g. Gas6Flow, Heater5Temp) are not
recorded on most wafers and arrive as all-NaN columns. All reductions here
are nan-aware; channels that are entirely NaN summarise to 0.0 rather than
poisoning the design matrix.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np


def per_cycle_oes_band_means(
    oes_data: np.ndarray,                # (T_o, W) numeric
    cycle_starts: np.ndarray,            # (n_cycles,) int
    cycle_ends: np.ndarray,              # (n_cycles,) int — exclusive
    n_bands: int,
) -> np.ndarray:
    """Mean OES intensity per (cycle, wavelength-band).

    Returns array of shape (n_cycles, n_bands), float32. Empty cycle slices
    fall back to zero (cycle indices are constructed to be non-empty by the
    cache builder, but defensive zero keeps the pipeline robust).
    """
    n_cycles = len(cycle_starts)
    W = oes_data.shape[1]
    band_edges = np.linspace(0, W, n_bands + 1, dtype=np.int32)

    out = np.zeros((n_cycles, n_bands), dtype=np.float32)
    for c in range(n_cycles):
        s, e = int(cycle_starts[c]), int(cycle_ends[c])
        if e <= s:
            continue
        cycle = oes_data[s:e].astype(np.float32, copy=False)
        # mean over time axis first, then average over each band
        time_mean = cycle.mean(axis=0)            # (W,)
        for b in range(n_bands):
            out[c, b] = time_mean[band_edges[b]:band_edges[b + 1]].mean()
    return out


def per_cycle_process_means(
    proc_data: np.ndarray,               # (T_p, F) float (may contain NaN)
    cycle_starts: np.ndarray,
    cycle_ends: np.ndarray,
) -> np.ndarray:
    """Mean Process value per (cycle, channel). Shape (n_cycles, F).

    Uses nanmean so cycles with partially-NaN channels still produce a
    sensible mean. Fully-NaN channel → cycle entry stays NaN, picked up
    downstream by `summarise_cycle_series`.
    """
    n_cycles = len(cycle_starts)
    F = proc_data.shape[1]
    out = np.full((n_cycles, F), np.nan, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        for c in range(n_cycles):
            s, e = int(cycle_starts[c]), int(cycle_ends[c])
            if e <= s:
                continue
            out[c] = np.nanmean(proc_data[s:e], axis=0)
    return out


def summarise_cycle_series(
    series: np.ndarray,                  # (n_cycles,) per-cycle scalar
    early_n: int = 30,
    late_n: int = 30,
) -> dict[str, float]:
    """Reduce a 100-length per-cycle series to scalar summary stats.

    Returns: mean, std, min, max, slope (linear-regression coef vs cycle idx),
             early (mean of first `early_n`), late (mean of last `late_n`),
             drift (late - early).
    """
    keys = ("mean", "std", "min", "max", "slope", "early", "late", "drift")
    n = len(series)
    s = np.asarray(series, dtype=np.float32)
    valid = ~np.isnan(s)
    if n == 0 or not valid.any():
        return {k: 0.0 for k in keys}

    x = np.arange(n, dtype=np.float32)
    if valid.sum() >= 2:
        slope = float(np.polyfit(x[valid], s[valid], 1)[0])
    else:
        slope = 0.0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        early_slice = s[:early_n]
        late_slice = s[-late_n:]
        early = float(np.nanmean(early_slice)) if np.isfinite(early_slice).any() else 0.0
        late = float(np.nanmean(late_slice)) if np.isfinite(late_slice).any() else 0.0
        return {
            "mean": float(np.nanmean(s)),
            "std": float(np.nanstd(s)),
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
            "slope": slope,
            "early": early,
            "late": late,
            "drift": late - early,
        }


def extract_baseline_features_one_wafer(
    npz_path: Path,
    n_oes_bands: int = 10,
) -> dict[str, float]:
    """Build the baseline feature row for one wafer's NPZ.

    OES is reduced to `n_oes_bands` wavelength bands; for each band we keep
    8 across-cycle summary stats. Process: same 8 stats × 44 channels.
    """
    z = np.load(npz_path, allow_pickle=False)

    oes_data = z["oes_data"]
    proc_data = z["process_data"]
    proc_features = z["process_features"]

    oes_pc = per_cycle_oes_band_means(
        oes_data,
        z["oes_cycle_starts_idx"], z["oes_cycle_ends_idx"],
        n_bands=n_oes_bands,
    )
    proc_pc = per_cycle_process_means(
        proc_data,
        z["proc_cycle_starts_idx"], z["proc_cycle_ends_idx"],
    )

    feats: dict[str, float] = {"experiment_key": str(z["experiment_key"])}

    for b in range(n_oes_bands):
        for stat, val in summarise_cycle_series(oes_pc[:, b]).items():
            feats[f"oes_band{b:02d}_{stat}"] = val

    for f_idx in range(proc_pc.shape[1]):
        short = str(proc_features[f_idx]).removeprefix("Stat3_Etch_MV_")
        for stat, val in summarise_cycle_series(proc_pc[:, f_idx]).items():
            feats[f"proc_{short}_{stat}"] = val

    return feats
