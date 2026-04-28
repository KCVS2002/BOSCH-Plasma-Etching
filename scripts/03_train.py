"""Phase-2/3 training entry: config-driven CV training of any registered model.

Pipeline (one config → one experiment folder):
  1. Read config YAML, set seed, create outputs/experiments/<ts>_<slug>/.
  2. Build / load wafer-level feature table (cached under cache/<v>/features/).
  3. Join with measurement table → per-sample design matrix.
  4. Load CV split file (cache/<v>/splits/...).
  5. For each target × fold: fit model, predict on val, record metrics.
  6. Aggregate, write metrics.json + checkpoints + NOTES.md update.

Run from project root:
    python scripts/03_train.py --config configs/exp_baseline_xgb.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yaml

from src.evaluation import aggregate_folds, load_split, regression_metrics
from src.features import load_or_build_features
from src.models import make_model
from src.utils import make_experiment_dir, set_seed


def _load_measurements(cache_root: Path) -> pd.DataFrame:
    pq = cache_root / "measurements.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(cache_root / "measurements.csv")


def _build_design_matrix(
    feat_df: pd.DataFrame, meas: pd.DataFrame, spatial: list[str], targets: list[str]
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], pd.DataFrame]:
    """Join wafer features with sample table → (X, y_per_target, feature_names, df)."""
    feat_cols = [c for c in feat_df.columns if c != "experiment_key"]
    df = meas.merge(feat_df, on="experiment_key", how="left", validate="many_to_one")
    if df[feat_cols].isna().any().any():
        missing = df[df[feat_cols].isna().any(axis=1)]["experiment_key"].unique()
        raise RuntimeError(
            f"feature join produced NaNs for wafers: {missing.tolist()[:5]}..."
        )
    feature_names = feat_cols + list(spatial)
    X = df[feature_names].to_numpy(dtype=np.float32)
    y = {t: df[t].to_numpy(dtype=np.float32) for t in targets}
    return X, y, feature_names, df


def _save_xgb_model(model, path: Path) -> None:
    """XGBoost has its own save format; fall back to joblib for others."""
    try:
        model.save_model(str(path.with_suffix(".json")))
    except AttributeError:
        import joblib
        joblib.dump(model, path.with_suffix(".pkl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path,
                        help="path to experiment config YAML")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    seed = int(config["experiment"]["seed"])
    set_seed(seed)

    exp_dir = make_experiment_dir(config["experiment"]["title"])
    shutil.copy(args.config, exp_dir / "config.yaml")

    log_lines: list[str] = []
    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"Experiment dir: {exp_dir.relative_to(PROJECT_ROOT)}")
    log(f"Config: {args.config}")

    # ---- Data ------------------------------------------------------------
    cache_root = PROJECT_ROOT / "cache" / config["data"]["cache_version"]
    log(f"\n[data] cache: {cache_root.relative_to(PROJECT_ROOT)}")

    feat_df = load_or_build_features(
        cache_root,
        feature_set=config["data"]["feature_set"],
        n_oes_bands=int(config["data"]["n_oes_bands"]),
    )
    log(f"[data] feature table: {feat_df.shape[0]} wafers × {feat_df.shape[1] - 1} features")

    meas = _load_measurements(cache_root)
    log(f"[data] measurements:  {meas.shape[0]} samples × {meas.shape[1]} cols")

    spatial = list(config["data"]["spatial_features"])
    targets = list(config["data"]["targets"])
    X, y_by_target, feature_names, df = _build_design_matrix(feat_df, meas, spatial, targets)
    log(f"[data] design matrix: {X.shape[0]} samples × {X.shape[1]} features")
    log(f"[data] targets: {targets}")

    split = load_split(cache_root / config["data"]["split_file"])
    log(f"[data] split: {config['data']['split_file']}  ({split.method}, n_folds={split.n_folds})")

    # ---- Train -----------------------------------------------------------
    ckpt_dir = exp_dir / "checkpoints"
    fold_csv_rows: list[dict] = []
    metrics_out: dict[str, dict] = {}

    for target in targets:
        log(f"\n=== Target: {target} ===")
        y = y_by_target[target]
        per_fold: list[dict] = []
        for f in range(split.n_folds):
            train_mask, val_mask = split.train_val_masks(f)
            t0 = time.time()
            model = make_model(config["model"]["name"], config["model"]["params"])
            model.fit(X[train_mask], y[train_mask])
            y_pred = model.predict(X[val_mask])
            m = regression_metrics(y[val_mask], y_pred)
            m["fold"] = f
            m["n_train"] = int(train_mask.sum())
            m["n_val"] = int(val_mask.sum())
            m["fit_seconds"] = round(time.time() - t0, 2)
            per_fold.append(m)
            fold_csv_rows.append({"target": target, **m})
            _save_xgb_model(model, ckpt_dir / f"{target}_fold{f}")
            log(f"  fold {f}: rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  "
                f"r2={m['r2']:.4f}  mape={m['mape_pct']:.2f}%  "
                f"({m['fit_seconds']:.1f}s, n_train={m['n_train']}, n_val={m['n_val']})")
        agg = aggregate_folds(per_fold)
        log(f"  AGG    rmse={agg['rmse_mean']:.4f}±{agg['rmse_std']:.4f}  "
            f"mae={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}  "
            f"r2={agg['r2_mean']:.4f}±{agg['r2_std']:.4f}  "
            f"mape={agg['mape_pct_mean']:.2f}%±{agg['mape_pct_std']:.2f}%")
        metrics_out[target] = {"per_fold": per_fold, "aggregate": agg}

    # ---- Persist outputs -------------------------------------------------
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(fold_csv_rows).to_csv(exp_dir / "logs" / "fold_metrics.csv", index=False)
    (exp_dir / "logs" / "stdout.log").write_text("\n".join(log_lines), encoding="utf-8")

    # Append a results block to the auto-seeded NOTES.md
    notes_path = exp_dir / "NOTES.md"
    notes_extra = ["\n## 자동 기록된 결과 (training script)\n"]
    for target, blk in metrics_out.items():
        a = blk["aggregate"]
        notes_extra.append(
            f"- **{target}**: RMSE={a['rmse_mean']:.4f}±{a['rmse_std']:.4f}, "
            f"MAE={a['mae_mean']:.4f}±{a['mae_std']:.4f}, "
            f"R²={a['r2_mean']:.4f}±{a['r2_std']:.4f}, "
            f"MAPE={a['mape_pct_mean']:.2f}±{a['mape_pct_std']:.2f}%"
        )
    notes_path.write_text(
        notes_path.read_text(encoding="utf-8") + "\n".join(notes_extra) + "\n",
        encoding="utf-8",
    )

    log(f"\nSaved: {(exp_dir / 'metrics.json').relative_to(PROJECT_ROOT)}")
    log(f"Saved: {(exp_dir / 'logs/fold_metrics.csv').relative_to(PROJECT_ROOT)}")
    log(f"Saved: {ckpt_dir.relative_to(PROJECT_ROOT)}/  ({len(targets) * split.n_folds} models)")


if __name__ == "__main__":
    main()
