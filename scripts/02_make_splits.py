"""Phase-1 entry: generate cross-validation splits from cache/<version>/measurements.

Two split files are emitted under cache/<version>/splits/:

  1. `kfold5_wafer.npz`  — main K=5 split for ALL experiments.
     Wafer-level grouping (89 points/wafer never split) + lot stratification
     (StratifiedGroupKFold) so each fold sees diverse conditioning conditions.

  2. `loo_lot.npz`       — Leave-One-Lot-Out (10 folds), auxiliary check for
     generalisation to unseen conditioning lots.

Each .npz contains, in canonical measurement-row order:
    sample_fold_id  (N_samples,) int8   — fold the sample belongs to (val side)
    sample_wafer_idx(N_samples,) int32  — index into wafer_keys for that sample
    wafer_fold_id   (N_wafers,)  int8   — fold the wafer belongs to
    wafer_keys      (N_wafers,)  str    — canonical wafer order
    n_folds         scalar int
    method          scalar str
    seed            scalar int           (kfold only)
    fold_lot_mapping(n_folds,) int       (loo only — which lot each fold holds out)

Run from project root:
    python scripts/02_make_splits.py --version v1
    python scripts/02_make_splits.py --version v1 --seed 42 --kfolds 5
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from src.utils import set_seed


def _load_measurements(cache_root: Path) -> pd.DataFrame:
    parquet = cache_root / "measurements.parquet"
    csv = cache_root / "measurements.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"no measurements file in {cache_root}")


def make_kfold_split(meas: pd.DataFrame, n_splits: int, seed: int) -> dict:
    """Lot-stratified GroupKFold over wafers.

    Sample y = lot_number (per row). Group = experiment_key (= wafer).
    Sklearn balances the lot distribution across folds while keeping every
    wafer's 89 points in exactly one fold.
    """
    sample_wafer = meas["experiment_key"].to_numpy()
    sample_lot = meas["lot_number"].to_numpy()

    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    sample_fold = np.full(len(meas), -1, dtype=np.int8)
    for fold_id, (_, val_idx) in enumerate(
        sgk.split(np.zeros(len(meas)), y=sample_lot, groups=sample_wafer)
    ):
        sample_fold[val_idx] = fold_id
    if (sample_fold == -1).any():
        raise RuntimeError("some samples were never assigned to a fold")

    return _to_split_dict(
        meas=meas,
        sample_fold=sample_fold,
        n_folds=n_splits,
        method=f"StratifiedGroupKFold(n_splits={n_splits}, stratify=lot, group=wafer)",
        seed=seed,
    )


def make_loo_lot_split(meas: pd.DataFrame) -> dict:
    """Leave-One-Lot-Out: 10 folds, each held-out fold == one lot.

    fold_id i corresponds to held-out lot fold_lot_mapping[i].
    """
    sample_lot = meas["lot_number"].to_numpy()
    sample_wafer = meas["experiment_key"].to_numpy()

    logo = LeaveOneGroupOut()
    sample_fold = np.full(len(meas), -1, dtype=np.int8)
    fold_lot: list[int] = []
    for fold_id, (_, val_idx) in enumerate(
        logo.split(np.zeros(len(meas)), groups=sample_lot)
    ):
        sample_fold[val_idx] = fold_id
        fold_lot.append(int(sample_lot[val_idx[0]]))
    if (sample_fold == -1).any():
        raise RuntimeError("some samples were never assigned to a fold")

    out = _to_split_dict(
        meas=meas,
        sample_fold=sample_fold,
        n_folds=len(fold_lot),
        method="LeaveOneGroupOut(group=lot)",
        seed=None,
    )
    out["fold_lot_mapping"] = np.asarray(fold_lot, dtype=np.int32)
    return out


def _to_split_dict(meas, sample_fold, n_folds, method, seed) -> dict:
    sample_wafer = meas["experiment_key"].to_numpy()
    wafer_keys, inverse = np.unique(sample_wafer, return_inverse=True)
    sample_wafer_idx = inverse.astype(np.int32)

    # wafer_fold_id: each wafer should land in exactly one fold (groups not split)
    wafer_fold = np.full(len(wafer_keys), -1, dtype=np.int8)
    for w_idx in range(len(wafer_keys)):
        rows = np.where(sample_wafer_idx == w_idx)[0]
        folds = np.unique(sample_fold[rows])
        if len(folds) != 1:
            raise RuntimeError(
                f"wafer {wafer_keys[w_idx]!r} spans folds {folds.tolist()} — group leak"
            )
        wafer_fold[w_idx] = folds[0]

    # Force fixed-width unicode dtype so .npz files load with allow_pickle=False.
    out = {
        "sample_fold_id": sample_fold,
        "sample_wafer_idx": sample_wafer_idx,
        "wafer_fold_id": wafer_fold,
        "wafer_keys": np.asarray(wafer_keys, dtype="<U32"),
        "n_folds": np.int32(n_folds),
        "method": np.asarray(method, dtype="<U128"),
    }
    if seed is not None:
        out["seed"] = np.int32(seed)
    return out


def _summarise(split: dict, meas: pd.DataFrame) -> dict:
    sample_fold = split["sample_fold_id"]
    wafer_fold = split["wafer_fold_id"]
    wafer_keys = split["wafer_keys"]
    n_folds = int(split["n_folds"])

    wafer_to_lot = (
        meas.drop_duplicates("experiment_key")
        .set_index("experiment_key")["lot_number"]
        .to_dict()
    )
    fold_rows = []
    for f in range(n_folds):
        val_wafer_mask = wafer_fold == f
        val_wafers = wafer_keys[val_wafer_mask]
        val_lots = sorted({wafer_to_lot[w] for w in val_wafers})
        fold_rows.append({
            "fold": f,
            "val_wafers": int(val_wafer_mask.sum()),
            "val_samples": int((sample_fold == f).sum()),
            "val_lots": val_lots,
        })
    return {"folds": fold_rows}


def _save(split: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **split)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1",
                        help="cache version (reads cache/<version>/measurements.*)")
    parser.add_argument("--kfolds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_root = PROJECT_ROOT / "cache" / args.version
    splits_dir = cache_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    meas = _load_measurements(cache_root)
    print(f"Loaded measurements: {meas.shape[0]} rows × {meas.shape[1]} cols "
          f"({meas['experiment_key'].nunique()} wafers, "
          f"{meas['lot_number'].nunique()} lots)")

    # ---- 1. K-fold (main) -----------------------------------------------
    kfold_path = splits_dir / f"kfold{args.kfolds}_wafer.npz"
    kfold = make_kfold_split(meas, n_splits=args.kfolds, seed=args.seed)
    _save(kfold, kfold_path)
    kfold_summary = _summarise(kfold, meas)
    print(f"\n[1/2] {kfold_path.relative_to(PROJECT_ROOT)}")
    print(f"      method: {str(kfold['method'])}")
    print(f"      seed:   {int(kfold['seed'])}")
    for r in kfold_summary["folds"]:
        print(f"      fold {r['fold']}: {r['val_wafers']:2d} wafers, "
              f"{r['val_samples']:5d} samples, lots={r['val_lots']}")

    # ---- 2. Leave-One-Lot-Out (auxiliary) -------------------------------
    loo_path = splits_dir / "loo_lot.npz"
    loo = make_loo_lot_split(meas)
    _save(loo, loo_path)
    loo_summary = _summarise(loo, meas)
    print(f"\n[2/2] {loo_path.relative_to(PROJECT_ROOT)}")
    print(f"      method: {str(loo['method'])}")
    for r, lot in zip(loo_summary["folds"], loo["fold_lot_mapping"].tolist()):
        print(f"      fold {r['fold']:2d} (lot {lot:2d}): "
              f"{r['val_wafers']:2d} wafers, {r['val_samples']:4d} samples")

    # ---- Manifest -------------------------------------------------------
    manifest = {
        "version": args.version,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": int(len(meas)),
        "n_wafers": int(meas["experiment_key"].nunique()),
        "n_lots": int(meas["lot_number"].nunique()),
        "kfold": {
            "file": kfold_path.name,
            "n_folds": int(kfold["n_folds"]),
            "seed": int(kfold["seed"]),
            "method": str(kfold["method"]),
            "folds": kfold_summary["folds"],
        },
        "loo_lot": {
            "file": loo_path.name,
            "n_folds": int(loo["n_folds"]),
            "method": str(loo["method"]),
            "fold_lot_mapping": loo["fold_lot_mapping"].tolist(),
            "folds": loo_summary["folds"],
        },
    }
    (splits_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved manifest: {(splits_dir / 'manifest.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
