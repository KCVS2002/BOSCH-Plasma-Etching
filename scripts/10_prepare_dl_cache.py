"""Prepare reusable DL preprocessing caches for cloud GPU runs.

This script is intentionally separate from experiment training:

  - ``dl_tensors`` stores fixed-size resampled cycle tensors once.
  - ``dl_normalizers`` stores fold-specific train-only normalizer stats.
  - ``dl_normalized`` optionally stores fold-specific normalized tensors.

Run from project root:
    .venv\\Scripts\\python.exe -m scripts.10_prepare_dl_cache --config configs/exp_dl_multimodal_oes_aux_mixup_ema_longrun_5fold.yaml --level normalized
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.data import (
    WaferCycleStore,
    dl_normalized_cache_root,
    dl_normalizer_cache_root,
    dl_tensor_cache_root,
)
from src.evaluation import load_split
from src.features import compute_oes_wavelength_scores, oes_score_cache_path


def _load_measurements(cache_root: Path) -> pd.DataFrame:
    pq = cache_root / "measurements.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(cache_root / "measurements.csv")


def _save_npz(path: Path, compressed: bool, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fold_ids_from_args(args, cfg: dict, split) -> list[int]:
    if args.folds:
        fold_ids = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
        if not fold_ids:
            raise ValueError("--folds was provided but no fold ids were parsed")
        bad = [f for f in fold_ids if f < 0 or f >= split.n_folds]
        if bad:
            raise ValueError(f"fold ids out of range for {split.n_folds} folds: {bad}")
        return fold_ids
    limit = cfg.get("experiment", {}).get("limit_folds", split.n_folds)
    n_folds = min(int(limit), split.n_folds) if limit is not None else split.n_folds
    return list(range(n_folds))


def _build_tensor_cache(
    *,
    cache_root: Path,
    tensor_root: Path,
    meas: pd.DataFrame,
    all_keys: list[str],
    t_o: int,
    t_p: int,
    overwrite: bool,
    compressed: bool,
) -> list[str]:
    store = WaferCycleStore(cache_root=cache_root, meas=meas, t_o=t_o, t_p=t_p)
    proc_names = store.discover_common_proc_channels(all_keys)
    wafer_dir = tensor_root / "wafers"
    wafer_dir.mkdir(parents=True, exist_ok=True)

    built = 0
    skipped = 0
    for key in tqdm(all_keys, desc="dl tensors", unit="wafer", ncols=100):
        out_path = wafer_dir / f"{key}.npz"
        if out_path.exists() and not overwrite:
            skipped += 1
            continue
        rec = store.load_wafer(key)
        _save_npz(
            out_path,
            compressed,
            oes_raw=rec.oes_raw.astype(np.float32, copy=False),
            proc_raw=rec.proc_raw.astype(np.float32, copy=False),
            oes_wavelengths=store.wavelengths.astype(np.float32),
            lot_number=np.asarray(rec.lot_number, dtype=np.int32),
        )
        del store._records[key]
        built += 1

    if store.wavelengths is None and all_keys:
        first_path = wafer_dir / f"{all_keys[0]}.npz"
        if first_path.exists():
            with np.load(first_path, allow_pickle=False) as z:
                store.wavelengths = z["oes_wavelengths"].astype(np.float32)

    manifest = {
        "kind": "dl_tensors",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "source_cache": str(cache_root.relative_to(PROJECT_ROOT)),
        "t_o": int(t_o),
        "t_p": int(t_p),
        "n_wafers": len(all_keys),
        "n_built": built,
        "n_skipped_existing": skipped,
        "n_wavelengths": int(len(store.wavelengths)) if store.wavelengths is not None else 0,
        "proc_kept_names": proc_names,
        "compressed": bool(compressed),
    }
    _write_json(tensor_root / "manifest.json", manifest)
    return proc_names


def _save_stats(store: WaferCycleStore, root: Path, fold: int) -> None:
    fold_root = root / f"fold{int(fold)}"
    _save_npz(fold_root / "stats.npz", False, **store.normalizer_state_arrays())


def _build_fold_caches(
    *,
    cache_root: Path,
    tensor_root: Path,
    normalizer_root: Path,
    normalized_root: Path | None,
    meas: pd.DataFrame,
    split,
    all_keys: list[str],
    fold_ids: list[int],
    t_o: int,
    t_p: int,
    xgb_feat_names: list[str] | None,
    per_wafer_norm: bool,
    overwrite: bool,
    compressed: bool,
) -> None:
    store = WaferCycleStore(
        cache_root=cache_root,
        meas=meas,
        t_o=t_o,
        t_p=t_p,
        xgb_feat_names=xgb_feat_names,
        per_wafer_norm=per_wafer_norm,
        tensor_cache_root=tensor_root,
        tensor_cache_required=True,
    )
    store.discover_common_proc_channels(all_keys)
    store.load_wafers(all_keys)

    normalizer_root.mkdir(parents=True, exist_ok=True)
    if normalized_root is not None:
        normalized_root.mkdir(parents=True, exist_ok=True)

    for fold in fold_ids:
        print(f"\n[fold {fold}] fit train-only normalizers")
        train_keys = split.wafer_keys[split.wafer_fold_id != fold].astype(str).tolist()
        store.fit_normalizers(train_keys)
        _save_stats(store, normalizer_root, fold)
        if normalized_root is None:
            continue

        print(f"[fold {fold}] apply and write normalized tensors")
        store.normalize_all(drop_raw=False, progress=True)
        _save_stats(store, normalized_root, fold)
        wafer_dir = normalized_root / f"fold{int(fold)}" / "wafers"
        for key in tqdm(all_keys, desc=f"fold {fold} normalized", unit="wafer", ncols=100):
            out_path = wafer_dir / f"{key}.npz"
            if out_path.exists() and not overwrite:
                continue
            rec = store[key]
            payload = {
                "oes": rec.oes.astype(np.float32, copy=False),
                "proc": rec.proc.astype(np.float32, copy=False),
                "points_X_norm": rec.points_X_norm.astype(np.float32, copy=False),
                "points_Y_norm": rec.points_Y_norm.astype(np.float32, copy=False),
                "points_si_norm": rec.points_si_norm.astype(np.float32, copy=False),
                "points_ox_norm": rec.points_ox_norm.astype(np.float32, copy=False),
                "lot_number": np.asarray(rec.lot_number, dtype=np.int32),
            }
            if rec.xgb_feats is not None:
                payload["xgb_feats"] = rec.xgb_feats.astype(np.float32, copy=False)
            _save_npz(out_path, compressed, **payload)

    manifest_base = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "source_cache": str(cache_root.relative_to(PROJECT_ROOT)),
        "tensor_cache": str(tensor_root.relative_to(PROJECT_ROOT)),
        "t_o": int(t_o),
        "t_p": int(t_p),
        "folds": fold_ids,
        "split_method": split.method,
        "proc_kept_names": list(store.proc_kept_names),
        "xgb_feat_names": xgb_feat_names or [],
    }
    _write_json(
        normalizer_root / "manifest.json",
        {"kind": "dl_normalizers", **manifest_base},
    )
    if normalized_root is not None:
        _write_json(
            normalized_root / "manifest.json",
            {
                "kind": "dl_normalized",
                **manifest_base,
                "per_wafer_norm": bool(per_wafer_norm),
                "compressed": bool(compressed),
            },
        )


def _precompute_oes_scores(
    *,
    cfg: dict,
    cache_root: Path,
    meas: pd.DataFrame,
    split,
    fold_ids: list[int],
) -> None:
    sel_cfg = cfg["data"].get("oes_band_selection")
    if not sel_cfg:
        return
    if str(sel_cfg.get("method", "correlation")) != "correlation":
        return
    stat = str(sel_cfg.get("stat", "late_mean"))
    late_start = int(sel_cfg.get("late_start_cycle", 80))
    early_end = int(sel_cfg.get("early_end_cycle", 20))
    targets = list(cfg["data"]["targets"])
    for target in targets:
        for fold in fold_ids:
            train_keys = split.wafer_keys[split.wafer_fold_id != fold].astype(str).tolist()
            path = oes_score_cache_path(
                cache_root,
                train_keys=train_keys,
                target=target,
                stat=stat,
                late_start_cycle=late_start,
                early_end_cycle=early_end,
            )
            if path.exists():
                continue
            print(f"[oes-score] target={target} fold={fold} stat={stat}")
            compute_oes_wavelength_scores(
                cache_root=cache_root,
                train_keys=train_keys,
                meas=meas,
                target=target,
                stat=stat,
                late_start_cycle=late_start,
                early_end_cycle=early_end,
                cache_path=path,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--level",
        choices=["tensors", "normalizers", "normalized"],
        default="normalized",
        help="highest cache level to build",
    )
    parser.add_argument("--folds", type=str, default=None, help="comma-separated fold ids")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="save tensor files with np.savez_compressed; slower but smaller",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = cfg["data"]
    cache_root = PROJECT_ROOT / "cache" / data_cfg["cache_version"]
    meas = _load_measurements(cache_root)
    split = load_split(cache_root / data_cfg["split_file"])

    t_o = int(data_cfg["t_o"])
    t_p = int(data_cfg["t_p"])
    xgb_feat_names = data_cfg.get("xgb_feat_names") or None
    per_wafer_norm = bool(data_cfg.get("per_wafer_norm", False))
    all_keys = sorted(set(meas["experiment_key"].astype(str).tolist()))
    fold_ids = _fold_ids_from_args(args, cfg, split)

    tensor_root = dl_tensor_cache_root(cache_root, t_o=t_o, t_p=t_p)
    normalizer_root = dl_normalizer_cache_root(
        cache_root,
        t_o=t_o,
        t_p=t_p,
        split_file=data_cfg["split_file"],
        xgb_feat_names=xgb_feat_names,
    )
    normalized_root = dl_normalized_cache_root(
        cache_root,
        t_o=t_o,
        t_p=t_p,
        split_file=data_cfg["split_file"],
        per_wafer_norm=per_wafer_norm,
        xgb_feat_names=xgb_feat_names,
    )

    print(f"Config: {args.config}")
    print(f"Cache: {cache_root.relative_to(PROJECT_ROOT)}")
    print(f"Level: {args.level}")
    print(f"Folds: {fold_ids}")
    print(f"Tensor cache: {tensor_root.relative_to(PROJECT_ROOT)}")
    if args.level in {"normalizers", "normalized"}:
        print(f"Normalizer cache: {normalizer_root.relative_to(PROJECT_ROOT)}")
    if args.level == "normalized":
        print(f"Normalized cache: {normalized_root.relative_to(PROJECT_ROOT)}")

    t0 = time.time()
    _build_tensor_cache(
        cache_root=cache_root,
        tensor_root=tensor_root,
        meas=meas,
        all_keys=all_keys,
        t_o=t_o,
        t_p=t_p,
        overwrite=args.overwrite,
        compressed=args.compressed,
    )
    if args.level in {"normalizers", "normalized"}:
        _build_fold_caches(
            cache_root=cache_root,
            tensor_root=tensor_root,
            normalizer_root=normalizer_root,
            normalized_root=normalized_root if args.level == "normalized" else None,
            meas=meas,
            split=split,
            all_keys=all_keys,
            fold_ids=fold_ids,
            t_o=t_o,
            t_p=t_p,
            xgb_feat_names=xgb_feat_names,
            per_wafer_norm=per_wafer_norm,
            overwrite=args.overwrite,
            compressed=args.compressed,
        )
    _precompute_oes_scores(cfg=cfg, cache_root=cache_root, meas=meas, split=split, fold_ids=fold_ids)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
