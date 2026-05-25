"""Create lot-based kfold split NPZ from Dataset CSV without requiring cache.

Usage:
    python scripts/_make_lot_split.py --version v1 --kfolds 5 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds_csv = PROJECT_ROOT / "Dataset" / "Si_Oxide_etch_89_points.csv"
    if not ds_csv.exists():
        raise FileNotFoundError(f"measurements CSV not found: {ds_csv}")

    rows = []
    with ds_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    experiment_keys = [r["experiment_key"] for r in rows]
    lot_numbers = [int(r["lot_number"]) for r in rows]

    wafer_keys = np.array(sorted({k for k in experiment_keys}))

    # assign lots to folds by shuffling lots with seed then round-robin
    unique_lots = sorted(set(lot_numbers))
    random.seed(int(args.seed))
    random.shuffle(unique_lots)
    lot_to_fold = {lot: i % args.kfolds for i, lot in enumerate(unique_lots)}

    sample_fold = np.full(len(rows), -1, dtype=np.int8)
    for i, lot in enumerate(lot_numbers):
        sample_fold[i] = lot_to_fold[int(lot)]

    # wafer fold: ensure each wafer maps to a single fold
    wafer_idx = {k: i for i, k in enumerate(wafer_keys)}
    wafer_fold = np.full(len(wafer_keys), -1, dtype=np.int8)
    for i, k in enumerate(experiment_keys):
        w = wafer_idx[k]
        wafer_fold[w] = sample_fold[i]

    out_dir = PROJECT_ROOT / "cache" / args.version / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"kfold{args.kfolds}_lot.npz"

    method = f"GroupKFold(n_splits={args.kfolds}, group=lot)"
    np.savez(out_path,
             sample_fold_id=sample_fold,
             sample_wafer_idx=np.array([wafer_idx[k] for k in experiment_keys], dtype=np.int32),
             wafer_fold_id=wafer_fold,
             wafer_keys=wafer_keys.astype("<U32"),
             n_folds=np.int32(args.kfolds),
             method=np.array(method, dtype="<U128"),
    )
    print(f"Wrote split: {out_path}")

if __name__ == "__main__":
    main()
