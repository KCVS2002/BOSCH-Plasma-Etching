"""Build the per-wafer feature table for the XGBoost baseline (Phase 2).

The feature table is wafer-level (88 rows). Joining with the 89-points
measurement table happens in the training script and produces the per-sample
design matrix.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .cycle_stats import extract_baseline_features_one_wafer


def build_wafer_feature_table(
    cache_root: Path,
    n_oes_bands: int = 10,
    progress: bool = True,
) -> pd.DataFrame:
    """Iterate cache/<v>/wafers/*.npz, build a (88, n_features+1) DataFrame.

    First column is `experiment_key`; remaining columns are the baseline
    statistical features. Process channels that don't appear on every wafer
    (e.g. Gas6Flow, Heater5Temp — recorded on 9/88 wafers only) are dropped
    to avoid leaking the "channel availability" signal as a wafer/lot proxy.
    """
    wafer_dir = cache_root / "wafers"
    paths = sorted(wafer_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no wafer NPZs under {wafer_dir}")

    rows: list[dict] = []
    iterator = tqdm(paths, desc="extract features", unit="wafer") if progress else paths
    for p in iterator:
        rows.append(extract_baseline_features_one_wafer(p, n_oes_bands=n_oes_bands))

    df = pd.DataFrame(rows)
    keep = ["experiment_key"] + [
        c for c in df.columns if c != "experiment_key" and df[c].notna().all()
    ]
    dropped = sorted(set(df.columns) - set(keep))
    if dropped:
        print(f"  dropped {len(dropped)} channel columns absent on some wafers "
              f"(e.g. {dropped[:3]}{'...' if len(dropped) > 3 else ''})")
    return df[keep]


def load_or_build_features(
    cache_root: Path,
    feature_set: str,
    n_oes_bands: int = 10,
) -> pd.DataFrame:
    """Cached feature loader.

    Output path: cache/<v>/features/<feature_set>.parquet (csv fallback).
    """
    out_dir = cache_root / "features"
    out_dir.mkdir(exist_ok=True)
    pq = out_dir / f"{feature_set}.parquet"
    csv = out_dir / f"{feature_set}.csv"

    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)

    df = build_wafer_feature_table(cache_root, n_oes_bands=n_oes_bands)
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        df.to_csv(csv, index=False)
    return df
