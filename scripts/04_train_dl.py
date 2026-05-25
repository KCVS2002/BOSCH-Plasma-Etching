"""Phase-3 entry: train Cycle-Aware DL model with K-fold CV.

Pipeline (one config → one experiment folder):
  1. Read config YAML, set seed, create outputs/experiments/<ts>_<slug>/.
  2. Load cache/<v>/measurements + split file.
  3. For each target × fold:
     a. Build a WaferCycleStore covering ONLY this fold's wafers.
     b. Fit normalisers on TRAIN wafers, apply to all.
     c. Build SampleDataset for train/val.
     d. Train model with early stopping; log per-epoch RMSE.
     e. Reload best checkpoint; predict on val; record metrics.
  6. Aggregate, write metrics.json + checkpoints + NOTES.md update.

Run from project root:
    python scripts/04_train_dl.py --config configs/exp_dl_multimodal.yaml
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
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    Normalizer,
    WaferCycleStore,
    WaferDataset,
)
from src.evaluation import aggregate_folds, load_split, regression_metrics
from src.models import make_model
from src.utils import make_experiment_dir, set_seed


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _load_measurements(cache_root: Path) -> pd.DataFrame:
    pq = cache_root / "measurements.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(cache_root / "measurements.csv")


def _make_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"unknown loss {name!r}")


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _forward(model: nn.Module, batch: dict) -> torch.Tensor:
    return model(
        oes=batch.get("oes"),
        proc=batch.get("proc"),
        xy=batch.get("xy"),
        xgb_feat=batch.get("xgb_feat"),
    )


def _fold_lot(split, fold: int) -> int | None:
    """Return the held-out lot id when the split file provides one."""
    mapping = split.extras.get("fold_lot_mapping")
    if mapping is None:
        return None
    if fold >= len(mapping):
        return None
    return int(mapping[fold])


def _write_partial_outputs(
    *,
    exp_dir: Path,
    metrics_out: dict[str, dict],
    fold_csv_rows: list[dict],
    all_epoch_rows: list[dict],
    all_sample_rows: list[dict],
    log_lines: list[str],
) -> None:
    """Persist progress after each fold so interrupted long CV runs remain usable."""
    (exp_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(fold_csv_rows).to_csv(exp_dir / "logs" / "fold_metrics.csv", index=False)
    pd.DataFrame(all_epoch_rows).to_csv(exp_dir / "logs" / "epoch_log.csv", index=False)
    pd.DataFrame(all_sample_rows).to_csv(exp_dir / "logs" / "sample_predictions.csv", index=False)
    (exp_dir / "logs" / "stdout.log").write_text("\n".join(log_lines), encoding="utf-8")


# ----------------------------------------------------------------------
# Per-fold training
# ----------------------------------------------------------------------

def train_one_fold(
    *,
    target: str,
    fold: int,
    train_keys: list[str],
    val_keys: list[str],
    store: WaferCycleStore,
    cfg: dict,
    device: torch.device,
    log,
) -> tuple[dict, dict, dict, np.ndarray, list[dict]]:
    """Train one model on one (target, fold). Returns metrics, normalizer dicts,
    final best state_dict (cpu), and val predictions.

    Normalization is fit on TRAIN wafers of THIS fold only (no leakage to val).
    Raw tensors stay in RAM so we can re-fit per fold without reloading from
    disk. Re-fit is cached by fold id to avoid redundant work when iterating
    over (target, fold) pairs that share a fold.
    """
    modality = cfg["data"]["modality"]

    last_fold = getattr(store, "_last_fitted_fold", None)
    if last_fold != fold:
        log(f"    [fold-norm] fit on {len(train_keys)} train wafers (fold {fold})...")
        t_n0 = time.time()
        store.fit_normalizers(train_keys)
        store.normalize_all(drop_raw=False, progress=False)
        store._last_fitted_fold = fold
        log(f"    [fold-norm] done ({time.time()-t_n0:.1f}s)")
    else:
        log(f"    [fold-norm] reusing fold {fold} fit from previous target")

    target_stats = store.si_stats if target == "si_etch" else store.ox_stats
    log(f"    target {target} stats: mean={target_stats.mean:.4f} std={target_stats.std:.4f}")

    # ---- Build wafer-level datasets ----
    train_ds = WaferDataset(
        store=store, wafer_keys=train_keys, target=target, modality=modality
    )
    val_ds = WaferDataset(
        store=store, wafer_keys=val_keys, target=target, modality=modality
    )

    # batch_size = wafers per step. 89 points share the same wafer encoder
    # forward, so we don't need a large batch — 2~4 wafers gives 178~356
    # predictions per step on 8 GB GPU.
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"].get("num_workers", 0))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                          pin_memory=False)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw,
                        pin_memory=False)

    # ---- Build model (override proc in_channels with actual count) ----
    params = dict(cfg["model"]["params"])
    if params.get("proc_encoder") is not None and modality in ("proc", "multimodal"):
        params["proc_encoder"] = dict(params["proc_encoder"])
        params["proc_encoder"]["in_channels"] = store.n_proc_channels

    model = make_model(cfg["model"]["name"], params).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    n_epochs = int(cfg["training"]["epochs"])
    if cfg["training"].get("scheduler", "none") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)
    else:
        scheduler = None

    loss_name = cfg["training"].get("loss", "mse").lower()
    loss_fn = _make_loss(loss_name)
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    patience = int(cfg["training"].get("early_stop_patience", 12))

    best_val_rmse = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    epochs_since_best = 0
    epoch_log_rows: list[dict] = []

    for epoch in range(n_epochs):
        ep_t0 = time.time()
        # ---- train ----
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        train_bar = tqdm(
            train_dl,
            desc=f"  ep {epoch:3d} train",
            unit="step",
            leave=False,
            ncols=100,
        )
        for batch in train_bar:
            batch = _move_batch(batch, device)
            optim.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = _forward(model, batch)
                    loss = loss_fn(pred, batch["target"])
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                pred = _forward(model, batch)
                loss = loss_fn(pred, batch["target"])
                loss.backward()
                optim.step()
            bs_actual = batch["target"].shape[0]
            train_loss_sum += float(loss.item()) * bs_actual
            train_n += bs_actual
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = train_loss_sum / max(train_n, 1)

        # ---- val ----
        model.eval()
        val_preds: list[np.ndarray] = []
        val_truth: list[np.ndarray] = []
        val_bar = tqdm(
            val_dl,
            desc=f"  ep {epoch:3d} val  ",
            unit="step",
            leave=False,
            ncols=100,
        )
        with torch.no_grad():
            for batch in val_bar:
                batch = _move_batch(batch, device)
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = _forward(model, batch)
                else:
                    pred = _forward(model, batch)
                val_preds.append(pred.detach().float().cpu().numpy().reshape(-1))
                val_truth.append(batch["target"].detach().float().cpu().numpy().reshape(-1))
        v_pred = np.concatenate(val_preds)
        v_true = np.concatenate(val_truth)
        # Sanitise NaNs from training divergence so metrics call doesn't crash;
        # the loss spike is what we care about — metrics will look terrible.
        if not np.isfinite(v_pred).all():
            v_pred = np.nan_to_num(v_pred, nan=0.0, posinf=0.0, neginf=0.0)
        # Inverse-scale to raw target units before computing metrics so RMSE
        # is in μm and directly comparable to XGBoost baseline.
        v_pred_raw = target_stats.invert(v_pred)
        v_true_raw = target_stats.invert(v_true)
        val_m = regression_metrics(v_true_raw, v_pred_raw)

        if scheduler is not None:
            scheduler.step()

        # Convert train MSE (in normalized target space) → raw-μm RMSE so it's
        # directly comparable to val_rmse on the same y-axis. Exact only when
        # loss is plain MSE; for other losses fall back to NaN.
        if loss_name == "mse":
            train_rmse_raw = float(np.sqrt(max(train_loss, 0.0)) * target_stats.std)
        else:
            train_rmse_raw = float("nan")

        ep_dt = time.time() - ep_t0
        epoch_log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_rmse": round(train_rmse_raw, 6),
            "val_rmse": round(float(val_m["rmse"]), 6),
            "val_mae": round(float(val_m["mae"]), 6),
            "val_r2": round(float(val_m["r2"]), 6),
            "val_mape": round(float(val_m["mape_pct"]), 2),
            "lr": float(optim.param_groups[0]["lr"]),
            "elapsed_s": round(ep_dt, 2),
        })
        log(f"    ep {epoch:3d} ({ep_dt:.1f}s): train_loss={train_loss:.4f} "
            f"train_rmse={train_rmse_raw:.4f} val_rmse={val_m['rmse']:.4f} "
            f"val_r2={val_m['r2']:.4f} val_mape={val_m['mape_pct']:.2f}%")

        if np.isfinite(val_m["rmse"]) and val_m["rmse"] < best_val_rmse - 1e-8:
            best_val_rmse = float(val_m["rmse"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:
                log(f"    early stop @ epoch {epoch} (best {best_epoch}, "
                    f"val_rmse={best_val_rmse:.4f})")
                break

    # ---- Reload best, final val predictions ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    final_preds: list[np.ndarray] = []
    final_truth: list[np.ndarray] = []
    with torch.no_grad():
        for batch in val_dl:
            batch = _move_batch(batch, device)
            pred = _forward(model, batch)
            final_preds.append(pred.detach().float().cpu().numpy().reshape(-1))
            final_truth.append(batch["target"].detach().float().cpu().numpy().reshape(-1))
    v_pred = np.concatenate(final_preds)
    v_true = np.concatenate(final_truth)
    if not np.isfinite(v_pred).all():
        v_pred = np.nan_to_num(v_pred, nan=0.0, posinf=0.0, neginf=0.0)
    v_pred = target_stats.invert(v_pred)
    v_true = target_stats.invert(v_true)
    metrics = regression_metrics(v_true, v_pred)
    metrics["fold"] = fold
    metrics["best_epoch"] = best_epoch
    metrics["n_train"] = len(train_keys) * 89
    metrics["n_val"] = len(val_keys) * 89

    sample_rows: list[dict] = []
    point_idx_global = 0
    for wafer_key in val_keys:
        # Each wafer has exactly 89 measurement points
        true_row = v_true[point_idx_global:point_idx_global + 89]
        pred_row = v_pred[point_idx_global:point_idx_global + 89]

        for point_idx, (truth, pred) in enumerate(zip(true_row, pred_row)):
            sample_rows.append({
                "target": target,
                "fold": fold,
                "experiment_key": wafer_key,
                "point_idx": point_idx,
                "y_true": float(truth),
                "y_pred": float(pred),
                "residual": float(pred - truth),
                "abs_error": float(abs(pred - truth)),
            })

        point_idx_global += 89

    oes_n = store.oes_normalizer.state_dict() if store.oes_normalizer is not None else {}
    proc_n = store.proc_normalizer.state_dict() if store.proc_normalizer is not None else {}
    fold_stats = {
        "target_stats": target_stats.state_dict(),
        "x_stats": store.x_stats.state_dict() if store.x_stats is not None else {},
        "y_stats": store.y_stats.state_dict() if store.y_stats is not None else {},
        "proc_kept_names": list(store.proc_kept_names) if store.proc_kept_names is not None else [],
    }
    return (metrics, oes_n, proc_n, v_pred, fold_stats), best_state, epoch_log_rows, sample_rows


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--limit-folds", type=int, default=None,
                        help="override config limit_folds (e.g. --limit-folds 5 for full CV)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg["experiment"]["seed"])
    set_seed(seed)

    exp_dir = make_experiment_dir(cfg["experiment"]["title"])
    shutil.copy(args.config, exp_dir / "config.yaml")

    log_lines: list[str] = []
    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"Experiment dir: {exp_dir.relative_to(PROJECT_ROOT)}")
    log(f"Config: {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}  (CUDA={torch.cuda.is_available()})")

    cache_root = PROJECT_ROOT / "cache" / cfg["data"]["cache_version"]
    meas = _load_measurements(cache_root)
    split = load_split(cache_root / cfg["data"]["split_file"])

    # If the loaded split references wafers not present in this cache's
    # measurements (e.g. per-day caches), restrict the split to the
    # intersection so downstream code doesn't KeyError on missing wafers.
    meas_keys = sorted(set(meas["experiment_key"].astype(str).tolist()))
    split_keys = [str(k) for k in split.wafer_keys]
    if any(k not in meas_keys for k in split_keys):
        keep_idx = [i for i, k in enumerate(split_keys) if k in meas_keys]
        import numpy as _np
        new_wafer_keys = _np.array([split_keys[i] for i in keep_idx], dtype=object)
        new_wafer_fold_id = split.wafer_fold_id[keep_idx].astype(int)
        # Simplify sample_fold_id to zeros for the reduced set (89 samples per wafer)
        new_sample_fold_id = _np.zeros(len(new_wafer_keys) * 89, dtype=int)
        # Replace split with a reduced Split-like object (duck-typed)
        from src.evaluation.splits import Split as _Split
        split = _Split(
            sample_fold_id=new_sample_fold_id,
            wafer_fold_id=new_wafer_fold_id,
            wafer_keys=new_wafer_keys,
            n_folds=int(new_wafer_fold_id.max() + 1) if len(new_wafer_fold_id) else 1,
            method=(split.method + " (filtered)") if hasattr(split, 'method') else "filtered",
            extras=split.extras if hasattr(split, 'extras') else {},
        )
    # Priority: CLI --limit-folds > config experiment.limit_folds > default 1 (single fold)
    cfg_limit = cfg["experiment"].get("limit_folds", 1)
    cli_limit = args.limit_folds
    effective_limit = cli_limit if cli_limit is not None else cfg_limit
    n_folds = min(effective_limit, split.n_folds) if effective_limit is not None else split.n_folds
    log(f"Cache: {cache_root.relative_to(PROJECT_ROOT)}  |  split: {cfg['data']['split_file']} "
        f"({split.method}, running {n_folds}/{split.n_folds} folds)")

    targets = list(cfg["data"]["targets"])
    log(f"Targets: {targets}  |  modality: {cfg['data']['modality']}")

    # ---- Build store ONCE for all folds/targets ----
    # Per-fold normalisation is fit INSIDE train_one_fold (TRAIN wafers only,
    # no val leakage). Here we only discover common proc channels and load
    # raw resampled tensors into RAM so the per-fold fit doesn't re-hit disk.
    t_setup = time.time()
    all_keys = sorted(set(meas["experiment_key"].astype(str).tolist()))
    log(f"\n[setup] loading {len(all_keys)} wafers (one-time, shared across folds)...")
    xgb_feat_names = cfg["data"].get("xgb_feat_names") or None
    store = WaferCycleStore(
        cache_root=cache_root,
        meas=meas,
        t_o=int(cfg["data"]["t_o"]),
        t_p=int(cfg["data"]["t_p"]),
        xgb_feat_names=xgb_feat_names,
    )
    t_a = time.time()
    store.discover_common_proc_channels(all_keys)
    t_disc = time.time() - t_a
    t_a = time.time()
    store.load_wafers(all_keys)
    t_load = time.time() - t_a
    log(f"[setup] done in {time.time()-t_setup:.1f}s  "
        f"(discover={t_disc:.1f} load={t_load:.1f})")
    log(f"[setup] proc channels kept: {store.n_proc_channels}")
    log(f"[setup] XGB injection features: {store.n_xgb_feats} "
        f"({'disabled' if store.n_xgb_feats == 0 else ', '.join(store.xgb_feat_names[:3]) + '...'})")
    log(f"[setup] normalisation deferred to per-fold (no val leakage)")

    metrics_out: dict[str, dict] = {}
    fold_csv_rows: list[dict] = []
    all_epoch_rows: list[dict] = []
    all_sample_rows: list[dict] = []

    for target in targets:
        log(f"\n=== Target: {target} ===")
        per_fold: list[dict] = []
        for f in range(n_folds):
            heldout_lot = _fold_lot(split, f)
            fold_label = f"fold {f}" if heldout_lot is None else f"fold {f} (held-out lot {heldout_lot})"
            log(f"  -- {fold_label} --")
            t0 = time.time()

            # build train/val wafer key lists
            val_keys = split.wafer_keys[split.wafer_fold_id == f].astype(str).tolist()
            train_keys = split.wafer_keys[split.wafer_fold_id != f].astype(str).tolist()

            (m, oes_n, proc_n, v_pred, fold_stats), best_state, epoch_rows, sample_rows = train_one_fold(
                target=target,
                fold=f,
                train_keys=train_keys,
                val_keys=val_keys,
                store=store,
                cfg=cfg,
                device=device,
                log=log,
            )
            m["fit_seconds"] = round(time.time() - t0, 2)
            if heldout_lot is not None:
                m["heldout_lot"] = heldout_lot
            train_target_values = np.concatenate([
                store[k].points_ox if target == "oxide_etch" else store[k].points_si
                for k in train_keys
            ])
            val_target_values = np.concatenate([
                store[k].points_ox if target == "oxide_etch" else store[k].points_si
                for k in val_keys
            ])
            m["train_target_mean"] = float(train_target_values.mean())
            m["train_target_std"] = float(train_target_values.std(ddof=0))
            m["val_target_mean"] = float(val_target_values.mean())
            m["val_target_std"] = float(val_target_values.std(ddof=0))
            m["target_mean_shift"] = float(val_target_values.mean() - train_target_values.mean())
            per_fold.append(m)
            fold_csv_rows.append({"target": target, **m})
            for r in epoch_rows:
                row = {"target": target, "fold": f, **r}
                if heldout_lot is not None:
                    row["heldout_lot"] = heldout_lot
                all_epoch_rows.append(row)
            for r in sample_rows:
                if heldout_lot is not None:
                    r["heldout_lot"] = heldout_lot
            all_sample_rows.extend(sample_rows)

            ckpt_dir = exp_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"{target}_fold{f}.pt"
            torch.save({
                "state_dict": best_state,
                "config": cfg,
                "target": target,
                "fold": f,
                "heldout_lot": heldout_lot,
                "oes_normalizer": oes_n,
                "proc_normalizer": proc_n,
                "target_stats": fold_stats["target_stats"],
                "x_stats": fold_stats["x_stats"],
                "y_stats": fold_stats["y_stats"],
                "proc_kept_names": fold_stats["proc_kept_names"],
                "xgb_feat_names": list(store.xgb_feat_names) if store.xgb_feat_names else [],
                "xgb_normalizer": store.xgb_normalizer.state_dict() if store.xgb_normalizer else {},
                "metrics": m,
            }, ckpt_path)

            log(f"    final: rmse={m['rmse']:.4f}  mae={m['mae']:.4f}  "
                f"r2={m['r2']:.4f}  mape={m['mape_pct']:.2f}%  "
                f"({m['fit_seconds']:.1f}s, best_ep={m['best_epoch']})")

            partial_agg = aggregate_folds([
                {k: v for k, v in row.items() if k != "heldout_lot"}
                for row in per_fold
            ])
            metrics_out[target] = {"per_fold": per_fold, "aggregate": partial_agg}
            _write_partial_outputs(
                exp_dir=exp_dir,
                metrics_out=metrics_out,
                fold_csv_rows=fold_csv_rows,
                all_epoch_rows=all_epoch_rows,
                all_sample_rows=all_sample_rows,
                log_lines=log_lines,
            )

        agg = aggregate_folds([
            {k: v for k, v in row.items() if k != "heldout_lot"}
            for row in per_fold
        ])
        log(f"  AGG    rmse={agg['rmse_mean']:.4f}±{agg['rmse_std']:.4f}  "
            f"r2={agg['r2_mean']:.4f}±{agg['r2_std']:.4f}")
        metrics_out[target] = {"per_fold": per_fold, "aggregate": agg}

        _write_partial_outputs(
            exp_dir=exp_dir,
            metrics_out=metrics_out,
            fold_csv_rows=fold_csv_rows,
            all_epoch_rows=all_epoch_rows,
            all_sample_rows=all_sample_rows,
            log_lines=log_lines,
        )

    # ---- Persist outputs ----
    _write_partial_outputs(
        exp_dir=exp_dir,
        metrics_out=metrics_out,
        fold_csv_rows=fold_csv_rows,
        all_epoch_rows=all_epoch_rows,
        all_sample_rows=all_sample_rows,
        log_lines=log_lines,
    )

    notes_path = exp_dir / "NOTES.md"
    extra = ["\n## 자동 기록된 결과 (DL training script)\n"]
    for target, blk in metrics_out.items():
        a = blk["aggregate"]
        extra.append(
            f"- **{target}**: RMSE={a['rmse_mean']:.4f}±{a['rmse_std']:.4f}, "
            f"MAE={a['mae_mean']:.4f}±{a['mae_std']:.4f}, "
            f"R²={a['r2_mean']:.4f}±{a['r2_std']:.4f}, "
            f"MAPE={a['mape_pct_mean']:.2f}±{a['mape_pct_std']:.2f}%"
        )
    notes_path.write_text(
        notes_path.read_text(encoding="utf-8") + "\n".join(extra) + "\n",
        encoding="utf-8",
    )

    log(f"\nSaved: {(exp_dir / 'metrics.json').relative_to(PROJECT_ROOT)}")
    log(f"Saved: {(exp_dir / 'logs/fold_metrics.csv').relative_to(PROJECT_ROOT)}")
    log(f"Saved: {(exp_dir / 'checkpoints').relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
