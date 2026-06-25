"""Build a compact, portable inference bundle for the live competition demo.

Why a bundle:
  The Streamlit app must run on a venue laptop with NO access to Dataset/ or the
  full cache/. This script precomputes everything the app needs into demo/bundle/
  (a few hundred MB, gitignored, regenerable):

    bundle/
    ├── manifest.json            # wafer list, per-wafer metrics, model labels
    ├── checkpoints/
    │   └── oxide_etch_fold0.pt  # 5-fold best model — app runs this LIVE
    └── wafers/<key>.npz         # per-wafer model-ready tensors + truths + baselines

  oxide is the LIVE centerpiece: the app loads the oxide checkpoint and runs the
  genuine model.forward on the stored (normalized) inputs on stage. si is the
  spatial-dominant context target (R²≈0.98) shown from precomputed predictions —
  its model needs the full 3648-wavelength OES (≈187 MB/wafer), too big to ship,
  so we bake si predictions in rather than re-running it live.

Both targets come from the SAME split (kfold5_wafer.npz, fold 0), so every
bundled wafer is a genuine held-out validation wafer for si AND oxide.

Run from project root:
    .venv\\python.exe -m demo.build_bundle                 # all 18 fold-0 val wafers
    .venv\\python.exe -m demo.build_bundle --wafers 2024-07-02_02,2024-08-22_02
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
FOLD = 0
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wafers", type=str, default=None,
                    help="comma-separated experiment_keys (default: all fold-0 val wafers)")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo" / "bundle")
    ap.add_argument("--cache-version", default="v1")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cache_root = PROJECT_ROOT / "cache" / args.cache_version

    meas = _load_measurements(cache_root)
    split = load_split(cache_root / SPLIT_FILE)
    val_keys = split.wafer_keys[split.wafer_fold_id == FOLD].astype(str).tolist()
    train_keys = split.wafer_keys[split.wafer_fold_id != FOLD].astype(str).tolist()

    demo_keys = (
        [k.strip() for k in args.wafers.split(",") if k.strip()]
        if args.wafers
        else sorted(val_keys)
    )
    bad = [k for k in demo_keys if k not in val_keys]
    if bad:
        raise SystemExit(f"not fold-{FOLD} validation wafers: {bad}")
    print(f"Bundling {len(demo_keys)} held-out wafers from fold {FOLD}")

    # ── Predictors ─────────────────────────────────────────────────────────────
    ox_pred = DLPredictor.from_checkpoint(
        PROJECT_ROOT / OXIDE_EXP / "checkpoints" / f"oxide_etch_fold{FOLD}.pt", device
    )
    si_pred = DLPredictor.from_checkpoint(
        PROJECT_ROOT / SI_EXP / "checkpoints" / f"si_etch_fold{FOLD}.pt", device
    )
    print(f"  oxide model: band-selected OES={ox_pred.oes_band_idx.shape[0]}ch  "
          f"si model: full OES (no band selection)")

    # ── Baselines ──────────────────────────────────────────────────────────────
    spatial_ox = _spatial_mean_baseline(meas, train_keys, "oxide_etch")
    spatial_si = _spatial_mean_baseline(meas, train_keys, "si_etch")

    xgb_cfg = yaml.safe_load((PROJECT_ROOT / XGB_EXP / "config.yaml").read_text(encoding="utf-8"))
    feat_df = load_or_build_features(
        cache_root, xgb_cfg["data"]["feature_set"], int(xgb_cfg["data"]["n_oes_bands"])
    )
    xgb_ox = xgb.Booster(); xgb_ox.load_model(str(PROJECT_ROOT / XGB_EXP / "checkpoints" / f"oxide_etch_fold{FOLD}.json"))
    xgb_si = xgb.Booster(); xgb_si.load_model(str(PROJECT_ROOT / XGB_EXP / "checkpoints" / f"si_etch_fold{FOLD}.json"))

    # ── Wafer store (lazy; pop after each wafer to cap RAM at ~1 wafer's OES) ───
    store = WaferCycleStore(cache_root=cache_root, meas=meas, t_o=T_O, t_p=T_P)
    store.discover_common_proc_channels(sorted(set(meas["experiment_key"].astype(str))))
    print(f"  proc channels: {store.n_proc_channels}")

    out = args.out
    (out / "wafers").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy(
        PROJECT_ROOT / OXIDE_EXP / "checkpoints" / f"oxide_etch_fold{FOLD}.pt",
        out / "checkpoints" / f"oxide_etch_fold{FOLD}.pt",
    )

    wafer_meta: list[dict] = []
    for i, key in enumerate(demo_keys, 1):
        rec = store.load_wafer(key)
        X, Y = rec.points_X, rec.points_Y
        ox_true, si_true = rec.points_ox, rec.points_si

        # LIVE-path inputs for oxide (small: 256 bands). si from full OES (precompute).
        ox_inputs = ox_pred.preprocess(rec.oes_raw, rec.proc_raw, X, Y)
        ox_dl = ox_pred.forward(ox_inputs)
        si_dl = si_pred.predict_raw(rec.oes_raw, rec.proc_raw, X, Y)

        meas_w = _sorted_points(meas, key)
        xgb_ox_p = _xgb_predict(xgb_ox, feat_df, meas_w)
        xgb_si_p = _xgb_predict(xgb_si, feat_df, meas_w)

        np.savez_compressed(
            out / "wafers" / f"{key}.npz",
            X=X.astype(np.float32), Y=Y.astype(np.float32),
            oxide_true=ox_true.astype(np.float32), si_true=si_true.astype(np.float32),
            oxide_pred_dl=ox_dl.astype(np.float32), si_pred_dl=si_dl.astype(np.float32),
            oxide_pred_xgb=xgb_ox_p, si_pred_xgb=xgb_si_p,
            oxide_pred_spatial=spatial_ox, si_pred_spatial=spatial_si,
            lot_number=np.int64(rec.lot_number),
            **ox_inputs.to_npz_dict(prefix="ox_"),  # ox_oes (100,128,256), ox_proc, ox_xy
        )

        def _m(t, p):
            return {k: round(float(v), 4) for k, v in regression_metrics(t, p).items()}

        wafer_meta.append({
            "key": key, "lot": int(rec.lot_number),
            "oxide": {"dl": _m(ox_true, ox_dl), "xgb": _m(ox_true, xgb_ox_p),
                      "spatial": _m(ox_true, spatial_ox),
                      "true_mean": round(float(ox_true.mean()), 4)},
            "si": {"dl": _m(si_true, si_dl), "xgb": _m(si_true, xgb_si_p),
                   "spatial": _m(si_true, spatial_si),
                   "true_mean": round(float(si_true.mean()), 4)},
        })
        del store._records[key]  # free this wafer's full OES (~187 MB) before next
        print(f"  [{i:2d}/{len(demo_keys)}] {key}  "
              f"oxide R²(dl/xgb/sp)={wafer_meta[-1]['oxide']['dl']['r2']:.2f}/"
              f"{wafer_meta[-1]['oxide']['xgb']['r2']:.2f}/{wafer_meta[-1]['oxide']['spatial']['r2']:.2f}")

    # ── Pooled metrics (all bundled points) for the headline numbers ───────────
    def _pool(target_key, model_key):
        ts, ps = [], []
        for k in demo_keys:
            z = np.load(out / "wafers" / f"{k}.npz")
            ts.append(z[f"{target_key}_true"]); ps.append(z[f"{target_key}_{model_key}"])
        return {kk: round(float(vv), 4)
                for kk, vv in regression_metrics(np.concatenate(ts), np.concatenate(ps)).items()}

    manifest = {
        "fold": FOLD, "split_file": SPLIT_FILE,
        "n_wafers": len(demo_keys), "wafers": demo_keys,
        "models": {
            "oxide": {"label": "Cycle-Aware DL (best, aux+mixup+EMA)", "exp": OXIDE_EXP, "live": True},
            "si": {"label": "Cycle-Aware DL (single-fold)", "exp": SI_EXP, "live": False},
            "xgb": {"label": "XGBoost baseline", "exp": XGB_EXP},
            "spatial": {"label": "Spatial-mean baseline ((X,Y) lookup)"},
        },
        "pooled": {
            "oxide": {m: _pool("oxide", {"dl": "pred_dl", "xgb": "pred_xgb", "spatial": "pred_spatial"}[m])
                      for m in ("dl", "xgb", "spatial")},
            "si": {m: _pool("si", {"dl": "pred_dl", "xgb": "pred_xgb", "spatial": "pred_spatial"}[m])
                   for m in ("dl", "xgb", "spatial")},
        },
        "per_wafer": wafer_meta,
        "checkpoint_oxide": f"checkpoints/oxide_etch_fold{FOLD}.pt",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    po = manifest["pooled"]["oxide"]
    print(f"\nDone → {out.relative_to(PROJECT_ROOT)}")
    print(f"  pooled oxide R²: DL={po['dl']['r2']:.3f}  "
          f"XGB={po['xgb']['r2']:.3f}  spatial={po['spatial']['r2']:.3f}")


if __name__ == "__main__":
    main()
