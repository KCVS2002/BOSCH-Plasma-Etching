"""Build a compact, portable inference bundle for the live competition demo.

Why a bundle:
  The Streamlit app must run on a venue laptop with NO access to Dataset/ or the
  full cache/. This script precomputes everything the app needs into demo/bundle/
  (gitignored, regenerable):

    bundle/
    ├── manifest.json              # wafer list, per-wafer metrics, model labels
    ├── checkpoints/
    │   ├── oxide_etch_fold0.pt    # 5-fold best model — app runs the LIVE forward
    │   ├── ...                       per wafer, using its OWN fold's checkpoint
    │   └── oxide_etch_fold4.pt
    └── wafers/<key>.npz           # per-wafer model-ready tensors + truths + baselines

  HELD-OUT HONESTY (why per-fold checkpoints):
    This is a 5-fold wafer GroupKFold. Each wafer is in exactly one fold's
    VALIDATION set and in the other 4 folds' TRAINING set. A wafer is a genuine
    held-out prediction only when scored by the model of the fold where it was
    validation. So every bundled wafer carries the checkpoint of its own fold,
    and the app loads that checkpoint to run the live forward. This lets ALL
    wafers (every fold) be demoed honestly, not just fold 0.

  oxide is the LIVE centerpiece (all folds): the app loads the wafer's fold
  checkpoint and runs the genuine model.forward on the stored (normalized) inputs.

  si is fold-0 ONLY: the si model is a single-fold experiment (no per-fold
  checkpoints), and it needs the full 3648-wavelength OES (≈187 MB/wafer), too big
  to ship. So si predictions are baked in for fold-0 wafers only, as
  spatial-dominant context (R²≈0.98) that motivates why oxide is the real VM task.

Run from project root:
    .venv\\Scripts\\python.exe -m demo.build_bundle              # ALL wafers, all folds
    .venv\\Scripts\\python.exe -m demo.build_bundle --folds 0    # just fold 0 (old behavior)
    .venv\\Scripts\\python.exe -m demo.build_bundle --wafers 2024-07-02_02,2024-08-22_02
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml

from src.data import WaferCycleStore
from src.demo import DLPredictor
from src.evaluation import load_split, regression_metrics
from src.features import load_or_build_features

# ── Canonical model sources (see memory/reference_experiments.md) ──────────────
OXIDE_EXP = "outputs/experiments/2026-05-28_04-19_dl-multimodal-oes-aux-mixup-ema-longrun-5fold"
SI_EXP = "outputs/experiments/2026-05-01_00-56_dl-multimodal-singlefold"
XGB_EXP = "outputs/experiments/2026-04-30_15-32_baseline-xgb"
SPLIT_FILE = "splits/kfold5_wafer.npz"
SI_FOLD = 0  # si model is single-fold; only fold-0 wafers are genuinely held-out for si
T_O, T_P = 128, 30


def _load_measurements(cache_root: Path) -> pd.DataFrame:
    pq = cache_root / "measurements.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(cache_root / "measurements.csv")


def _sorted_points(meas: pd.DataFrame, key: str) -> pd.DataFrame:
    """Same point ordering the store/predictor use: sort by (X, Y)."""
    return meas[meas["experiment_key"] == key].sort_values(["X", "Y"])


def _spatial_mean_baseline(
    meas: pd.DataFrame, train_keys: list[str], target: str
) -> np.ndarray:
    """(X,Y)-position lookup: mean target per point index over train wafers.

    The 89-point layout is identical across wafers, so sorting each wafer by
    (X, Y) aligns point i to the same physical site — averaging over train
    wafers gives the spatial-mean baseline (Lynn 2009 evaluation trap control).
    """
    stack = np.stack(
        [_sorted_points(meas, k)[target].to_numpy(dtype=np.float64) for k in train_keys],
        axis=0,
    )
    return stack.mean(axis=0).astype(np.float32)  # (89,)


def _xgb_predict(
    booster: xgb.Booster, feat_df: pd.DataFrame, meas_w: pd.DataFrame
) -> np.ndarray:
    feat_cols = [c for c in feat_df.columns if c != "experiment_key"]
    df = meas_w.merge(feat_df, on="experiment_key", how="left")
    feature_names = feat_cols + ["X", "Y"]
    X = df[feature_names].to_numpy(dtype=np.float32)
    dm = xgb.DMatrix(X, feature_names=feature_names)
    return booster.predict(dm).astype(np.float32)


def _m(t, p):
    return {k: round(float(v), 4) for k, v in regression_metrics(t, p).items()}


def build_oes_summary(out: Path, keys: list[str]) -> float:
    """Grand-mean OES preview image + a shared robust deviation scale.

    The app shows each wafer's (this wafer − mean wafer) deviation so the subtle
    but real between-wafer differences become visible; a single shared scale keeps
    the colors comparable as the presenter switches wafers. Saved to
    bundle/oes_grand_mean.npz (mean: (256,100) float32, scale: scalar).
    """
    imgs = [
        np.load(out / "wafers" / f"{k}.npz")["ox_oes"].mean(axis=1).T.astype(np.float32)
        for k in keys
    ]  # each (bands, cycles) — exactly the app's preview image
    grand = np.mean(imgs, axis=0).astype(np.float32)
    devs = np.stack([im - grand for im in imgs])
    scale = float(np.percentile(np.abs(devs), 99))
    np.savez_compressed(out / "oes_grand_mean.npz", mean=grand, scale=np.float32(scale))
    return scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wafers", type=str, default=None,
                    help="comma-separated experiment_keys (default: every wafer, all folds)")
    ap.add_argument("--folds", type=str, default=None,
                    help="comma-separated fold ids to bundle (default: all folds)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo" / "bundle")
    ap.add_argument("--cache-version", default="v1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cache_root = PROJECT_ROOT / "cache" / args.cache_version

    meas = _load_measurements(cache_root)
    split = load_split(cache_root / SPLIT_FILE)
    all_folds = sorted(int(f) for f in np.unique(split.wafer_fold_id))
    folds = ([int(f.strip()) for f in args.folds.split(",") if f.strip()]
             if args.folds else all_folds)
    wanted = ({k.strip() for k in args.wafers.split(",") if k.strip()}
              if args.wafers else None)

    # ── XGB baseline: per-wafer features are fold-independent; only the booster
    #    changes per fold. Build the feature table once, load boosters per fold. ─
    xgb_cfg = yaml.safe_load((PROJECT_ROOT / XGB_EXP / "config.yaml").read_text(encoding="utf-8"))
    feat_df = load_or_build_features(
        cache_root, xgb_cfg["data"]["feature_set"], int(xgb_cfg["data"]["n_oes_bands"])
    )

    # ── Wafer store (lazy; pop after each wafer to cap RAM at ~1 wafer's OES) ───
    store = WaferCycleStore(cache_root=cache_root, meas=meas, t_o=T_O, t_p=T_P)
    store.discover_common_proc_channels(sorted(set(meas["experiment_key"].astype(str))))
    print(f"  proc channels: {store.n_proc_channels}")

    out = args.out
    (out / "wafers").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    wafer_meta: list[dict] = []
    all_keys: list[str] = []
    si_keys: list[str] = []

    for fold in folds:
        val_keys = split.wafer_keys[split.wafer_fold_id == fold].astype(str).tolist()
        train_keys = split.wafer_keys[split.wafer_fold_id != fold].astype(str).tolist()
        demo_keys = sorted(k for k in val_keys if wanted is None or k in wanted)
        if not demo_keys:
            continue
        do_si = fold == SI_FOLD  # si genuinely held-out only for its single fold

        # oxide: this fold's held-out checkpoint (shipped + loaded live by the app)
        ckpt_rel = f"checkpoints/oxide_etch_fold{fold}.pt"
        ox_ckpt_src = PROJECT_ROOT / OXIDE_EXP / "checkpoints" / f"oxide_etch_fold{fold}.pt"
        shutil.copy(ox_ckpt_src, out / ckpt_rel)
        ox_pred = DLPredictor.from_checkpoint(ox_ckpt_src, device)

        spatial_ox = _spatial_mean_baseline(meas, train_keys, "oxide_etch")
        xgb_ox = xgb.Booster()
        xgb_ox.load_model(str(PROJECT_ROOT / XGB_EXP / "checkpoints" / f"oxide_etch_fold{fold}.json"))

        si_pred = spatial_si = xgb_si = None
        if do_si:
            si_pred = DLPredictor.from_checkpoint(
                PROJECT_ROOT / SI_EXP / "checkpoints" / f"si_etch_fold{SI_FOLD}.pt", device
            )
            spatial_si = _spatial_mean_baseline(meas, train_keys, "si_etch")
            xgb_si = xgb.Booster()
            xgb_si.load_model(str(PROJECT_ROOT / XGB_EXP / "checkpoints" / f"si_etch_fold{SI_FOLD}.json"))

        print(f"[fold {fold}] {len(demo_keys)} held-out wafers"
              f"{'  (+si)' if do_si else ''}  oxide OES={ox_pred.oes_band_idx.shape[0]}ch")

        for i, key in enumerate(demo_keys, 1):
            rec = store.load_wafer(key)
            X, Y = rec.points_X, rec.points_Y
            ox_true = rec.points_ox
            meas_w = _sorted_points(meas, key)

            # LIVE-path inputs for oxide (small: 256 bands), predicted by THIS fold.
            ox_inputs = ox_pred.preprocess(rec.oes_raw, rec.proc_raw, X, Y)
            ox_dl = ox_pred.forward(ox_inputs)
            xgb_ox_p = _xgb_predict(xgb_ox, feat_df, meas_w)

            npz: dict[str, np.ndarray] = dict(
                X=X.astype(np.float32), Y=Y.astype(np.float32),
                oxide_true=ox_true.astype(np.float32),
                oxide_pred_dl=ox_dl.astype(np.float32),
                oxide_pred_xgb=xgb_ox_p, oxide_pred_spatial=spatial_ox,
                lot_number=np.int64(rec.lot_number),
                **ox_inputs.to_npz_dict(prefix="ox_"),  # ox_oes (100,128,256), ox_proc, ox_xy
            )
            meta = {
                "key": key, "lot": int(rec.lot_number), "fold": fold,
                "has_si": do_si, "checkpoint": ckpt_rel,
                "oxide": {"dl": _m(ox_true, ox_dl), "xgb": _m(ox_true, xgb_ox_p),
                          "spatial": _m(ox_true, spatial_ox),
                          "true_mean": round(float(ox_true.mean()), 4)},
            }

            if do_si:
                si_true = rec.points_si
                si_dl = si_pred.predict_raw(rec.oes_raw, rec.proc_raw, X, Y)
                xgb_si_p = _xgb_predict(xgb_si, feat_df, meas_w)
                npz.update(
                    si_true=si_true.astype(np.float32),
                    si_pred_dl=si_dl.astype(np.float32),
                    si_pred_xgb=xgb_si_p, si_pred_spatial=spatial_si,
                )
                meta["si"] = {"dl": _m(si_true, si_dl), "xgb": _m(si_true, xgb_si_p),
                              "spatial": _m(si_true, spatial_si),
                              "true_mean": round(float(si_true.mean()), 4)}
                si_keys.append(key)

            np.savez_compressed(out / "wafers" / f"{key}.npz", **npz)
            wafer_meta.append(meta)
            all_keys.append(key)
            del store._records[key]  # free this wafer's full OES (~187 MB) before next
            print(f"  [f{fold} {i:2d}/{len(demo_keys)}] {key}  "
                  f"oxide R²(dl/xgb/sp)={meta['oxide']['dl']['r2']:.2f}/"
                  f"{meta['oxide']['xgb']['r2']:.2f}/{meta['oxide']['spatial']['r2']:.2f}")

    if not all_keys:
        raise SystemExit("no wafers matched the given --folds/--wafers filters")

    # keep wafers / per_wafer key-sorted so the app's selectbox is stable
    order = sorted(range(len(all_keys)), key=lambda j: all_keys[j])
    all_keys = [all_keys[j] for j in order]
    wafer_meta = [wafer_meta[j] for j in order]

    # ── Pooled metrics for the headline numbers (oxide: all wafers; si: fold-0) ─
    def _pool(keys, target_key, model_key):
        ts, ps = [], []
        for k in keys:
            z = np.load(out / "wafers" / f"{k}.npz")
            ts.append(z[f"{target_key}_true"]); ps.append(z[f"{target_key}_{model_key}"])
        return {kk: round(float(vv), 4)
                for kk, vv in regression_metrics(np.concatenate(ts), np.concatenate(ps)).items()}

    _mk = {"dl": "pred_dl", "xgb": "pred_xgb", "spatial": "pred_spatial"}
    manifest = {
        "folds": sorted(set(m["fold"] for m in wafer_meta)),
        "si_fold": SI_FOLD, "split_file": SPLIT_FILE,
        "n_wafers": len(all_keys), "wafers": all_keys, "si_wafers": sorted(si_keys),
        "models": {
            "oxide": {"label": "Cycle-Aware DL (best, aux+mixup+EMA)", "exp": OXIDE_EXP, "live": True},
            "si": {"label": "Cycle-Aware DL (single-fold)", "exp": SI_EXP, "live": False},
            "xgb": {"label": "XGBoost baseline", "exp": XGB_EXP},
            "spatial": {"label": "Spatial-mean baseline ((X,Y) lookup)"},
        },
        "pooled": {
            "oxide": {m: _pool(all_keys, "oxide", _mk[m]) for m in ("dl", "xgb", "spatial")},
            "si": ({m: _pool(si_keys, "si", _mk[m]) for m in ("dl", "xgb", "spatial")}
                   if si_keys else None),
        },
        "per_wafer": wafer_meta,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    oes_scale = build_oes_summary(out, all_keys)

    po = manifest["pooled"]["oxide"]
    print(f"\nDone → {out.relative_to(PROJECT_ROOT)}  "
          f"({len(all_keys)} wafers, folds {manifest['folds']}, si on {len(si_keys)})")
    print(f"  OES deviation scale (p99): ±{oes_scale:.3f}")
    print(f"  pooled oxide R²: DL={po['dl']['r2']:.3f}  "
          f"XGB={po['xgb']['r2']:.3f}  spatial={po['spatial']['r2']:.3f}")


if __name__ == "__main__":
    main()
