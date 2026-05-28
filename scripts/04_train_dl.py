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
import copy
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
from src.features import (
    compute_oes_wavelength_scores,
    select_top_k_wavelengths,
)
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


def _forward(model: nn.Module, batch: dict, return_aux: bool = False):
    return model(
        oes=batch.get("oes"),
        proc=batch.get("proc"),
        xy=batch.get("xy"),
        xgb_feat=batch.get("xgb_feat"),
        return_aux=return_aux,
    )


def _apply_mixup(
    batch: dict,
    alpha: float,
    prob: float,
) -> tuple[dict, float]:
    """Wafer-level mixup augmentation (Zhang et al. 2018).

    Samples a single λ ~ Beta(α, α) and a single permutation per batch, then
    linearly interpolates EVERY tensor entry (inputs and target) so the
    supervision stays consistent with the mixed inputs:

        mixed = λ · batch + (1 - λ) · batch[perm]

    Why: the bimodal oxide_etch distribution (low 0.57-0.61 vs high 0.65-0.71)
    creates a mode-collapse attractor for the encoder — 11 high-mode val
    wafers in fold 4 collapsed to identical wafer_repr. Mixing pairs across
    the gap fills the continuum with synthetic intermediates, breaking the
    binary-classifier basin. Also smears lot boundaries (since cross-lot
    mixes are common), giving lot-invariance pressure for LOO-Lot evaluation.

    Linearity makes this transparent for the wafer-mean aux loss:
        mean(λ·t_A + (1-λ)·t_B) = λ·mean(t_A) + (1-λ)·mean(t_B)
    so the aux head receives consistent (input, target) supervision without
    any special handling.

    Skipped when B<2 (last batch in epoch when drop_last=False) or by the
    `prob` roll. Returns (batch, 1.0) in those cases — caller can detect
    "mixup skipped" via lam==1.0.
    """
    target = batch.get("target")
    if target is None or target.shape[0] < 2:
        return batch, 1.0
    if prob < 1.0 and float(np.random.random()) >= prob:
        return batch, 1.0

    lam = float(np.random.beta(alpha, alpha))
    bs = target.shape[0]
    perm = torch.randperm(bs, device=target.device)

    mixed: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == bs:
            mixed[k] = lam * v + (1.0 - lam) * v[perm]
        else:
            mixed[k] = v
    return mixed, lam


class WeightEMA:
    """Exponential moving average of model weights (Polyak averaging).

    Maintains a deep-copied shadow model whose parameters are an EMA of the
    live model's parameters. Validation and the saved best checkpoint use
    the shadow's weights, which smooths out the late-stage oscillation
    caused by mixup's gradient noise and the bimodal loss landscape.

    Update rule (after every optimizer step):
        ema_p ← d_eff · ema_p + (1 - d_eff) · model_p

    The effective decay ramps up early to avoid biasing toward the
    randomly-initialised shadow:
        d_eff = min(decay, (1 + step) / (10 + step))
    This matches PyTorch's swa_utils.get_ema_avg_fn convention.

    Buffers (e.g. Fourier `freqs` registered buffers) are deterministic
    constants in this project, so we simply copy them every step rather
    than averaging — keeping them in sync at no cost.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"ema decay must be in (0, 1), got {decay!r}")
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        # deepcopy fragments RNN weight memory — re-pack into a single flat
        # block so cuDNN doesn't trigger a UserWarning + repack on every val
        # forward. In-place ema.update() preserves this layout.
        for m in self.ema_model.modules():
            if isinstance(m, nn.RNNBase):
                m.flatten_parameters()
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d_eff = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        one_minus_d = 1.0 - d_eff
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(d_eff).add_(p.detach(), alpha=one_minus_d)
        # Sync non-trainable buffers (e.g. Fourier freqs); these are constants
        # in our model, so copy is correct (no averaging).
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


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

    # ---- Per-fold OES wavelength selection (no leakage: train wafers only) ----
    oes_band_idx: np.ndarray | None = None
    sel_cfg = cfg["data"].get("oes_band_selection")
    if sel_cfg and modality in ("oes", "multimodal"):
        method = str(sel_cfg.get("method", "correlation"))
        if method != "correlation":
            raise ValueError(
                f"unsupported oes_band_selection.method {method!r} "
                "(only 'correlation' implemented)"
            )
        stat = str(sel_cfg.get("stat", "late_mean"))
        top_k = int(sel_cfg["top_k"])
        t_b = time.time()
        log(f"    [oes-band] correlation selection: stat={stat}, top_k={top_k}, "
            f"train_wafers={len(train_keys)}")
        scores = compute_oes_wavelength_scores(
            cache_root=store.cache_root,
            train_keys=train_keys,
            meas=store.meas,
            target=target,
            stat=stat,
            late_start_cycle=int(sel_cfg.get("late_start_cycle", 80)),
            early_end_cycle=int(sel_cfg.get("early_end_cycle", 20)),
        )
        oes_band_idx = select_top_k_wavelengths(scores, top_k=top_k)
        log(f"    [oes-band] selected {len(oes_band_idx)}/{len(scores)} wavelengths "
            f"(max|corr|={float(scores.max()):.3f}, "
            f"min selected |corr|={float(scores[oes_band_idx].min()):.3f}, "
            f"{time.time()-t_b:.1f}s)")

    # ---- Build wafer-level datasets ----
    train_ds = WaferDataset(
        store=store, wafer_keys=train_keys, target=target, modality=modality,
        oes_band_idx=oes_band_idx,
    )
    val_ds = WaferDataset(
        store=store, wafer_keys=val_keys, target=target, modality=modality,
        oes_band_idx=oes_band_idx,
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

    # ---- Build model (override proc/oes in_channels with actual count) ----
    params = dict(cfg["model"]["params"])
    if params.get("proc_encoder") is not None and modality in ("proc", "multimodal"):
        params["proc_encoder"] = dict(params["proc_encoder"])
        params["proc_encoder"]["in_channels"] = store.n_proc_channels
    if (
        params.get("oes_encoder") is not None
        and modality in ("oes", "multimodal")
        and oes_band_idx is not None
    ):
        params["oes_encoder"] = dict(params["oes_encoder"])
        params["oes_encoder"]["in_channels"] = int(len(oes_band_idx))

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

    has_aux = getattr(model, "aux_head", None) is not None
    aux_weight = float(model.cfg.aux_wafer_loss_weight) if has_aux else 0.0
    if has_aux:
        log(f"    [aux] wafer-mean aux head ENABLED, weight={aux_weight}")

    # Mixup config (training.mixup section, all keys optional)
    mixup_cfg = cfg["training"].get("mixup") or {}
    mixup_enabled = bool(mixup_cfg.get("enabled", False))
    mixup_alpha = float(mixup_cfg.get("alpha", 0.2))
    mixup_prob = float(mixup_cfg.get("prob", 1.0))
    if mixup_enabled:
        log(f"    [mixup] ENABLED  alpha={mixup_alpha}  prob={mixup_prob}")

    # EMA config (training.ema section, all keys optional). When enabled,
    # validation and best_state are taken from the shadow EMA model.
    ema_cfg = cfg["training"].get("ema") or {}
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema: WeightEMA | None = WeightEMA(model, decay=ema_decay) if ema_enabled else None
    if ema is not None:
        log(f"    [ema] ENABLED  decay={ema_decay}  validates on shadow weights")

    # patience <= 0 disables early stopping (still tracks best_state every epoch)
    early_stop_active = patience > 0
    if not early_stop_active:
        log(f"    [early-stop] DISABLED (patience={patience}). Will run full {n_epochs} epochs.")

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
        # Track mean λ across batches in this epoch (only batches where mixup
        # was actually applied — skipped batches contribute lam=1.0 which we
        # exclude to avoid biasing the mean).
        mixup_lams_applied: list[float] = []
        train_bar = tqdm(
            train_dl,
            desc=f"  ep {epoch:3d} train",
            unit="step",
            leave=False,
            ncols=100,
        )
        for batch in train_bar:
            batch = _move_batch(batch, device)
            if mixup_enabled:
                batch, lam = _apply_mixup(batch, alpha=mixup_alpha, prob=mixup_prob)
                if lam != 1.0:
                    mixup_lams_applied.append(lam)
            optim.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    if has_aux:
                        pred, aux_pred = _forward(model, batch, return_aux=True)
                        point_loss = loss_fn(pred, batch["target"])
                        aux_loss = loss_fn(aux_pred, batch["target"].mean(dim=1))
                        loss = point_loss + aux_weight * aux_loss
                    else:
                        pred = _forward(model, batch)
                        loss = loss_fn(pred, batch["target"])
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                if has_aux:
                    pred, aux_pred = _forward(model, batch, return_aux=True)
                    point_loss = loss_fn(pred, batch["target"])
                    aux_loss = loss_fn(aux_pred, batch["target"].mean(dim=1))
                    loss = point_loss + aux_weight * aux_loss
                else:
                    pred = _forward(model, batch)
                    loss = loss_fn(pred, batch["target"])
                loss.backward()
                optim.step()
            # EMA update lives AFTER the optimizer step so the shadow tracks
            # post-update weights (the actual trajectory the optimizer takes).
            if ema is not None:
                ema.update(model)
            bs_actual = batch["target"].shape[0]
            # Log the per-point loss only (matches baseline `train_loss` semantics
            # and keeps `train_rmse` conversion below valid). Aux contribution is
            # already exerting its pressure through the combined backprop.
            log_loss_val = point_loss.item() if has_aux else loss.item()
            train_loss_sum += float(log_loss_val) * bs_actual
            train_n += bs_actual
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        train_loss = train_loss_sum / max(train_n, 1)

        # ---- val ----
        # When EMA is enabled, evaluate on the shadow weights. The shadow
        # tracks a smoothed trajectory through weight space → smoother val
        # curves → patience / best-state decisions become more robust.
        eval_model = ema.ema_model if ema is not None else model
        eval_model.eval()
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
                        pred = _forward(eval_model, batch)
                else:
                    pred = _forward(eval_model, batch)
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
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_rmse": round(train_rmse_raw, 6),
            "val_rmse": round(float(val_m["rmse"]), 6),
            "val_mae": round(float(val_m["mae"]), 6),
            "val_r2": round(float(val_m["r2"]), 6),
            "val_mape": round(float(val_m["mape_pct"]), 2),
            "lr": float(optim.param_groups[0]["lr"]),
            "elapsed_s": round(ep_dt, 2),
        }
        if mixup_enabled:
            n_applied = len(mixup_lams_applied)
            row["mixup_n_applied"] = n_applied
            row["mixup_lam_mean"] = (
                round(float(np.mean(mixup_lams_applied)), 4) if n_applied else float("nan")
            )
        epoch_log_rows.append(row)
        log(f"    ep {epoch:3d} ({ep_dt:.1f}s): train_loss={train_loss:.4f} "
            f"train_rmse={train_rmse_raw:.4f} val_rmse={val_m['rmse']:.4f} "
            f"val_r2={val_m['r2']:.4f} val_mape={val_m['mape_pct']:.2f}%")

        if np.isfinite(val_m["rmse"]) and val_m["rmse"] < best_val_rmse - 1e-8:
            best_val_rmse = float(val_m["rmse"])
            # When EMA is enabled, save shadow weights — they're what produced
            # this val_rmse, and they're what we want for final inference.
            state_source = ema.ema_model if ema is not None else model
            best_state = {k: v.detach().cpu().clone() for k, v in state_source.state_dict().items()}
            best_epoch = epoch
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if early_stop_active and epochs_since_best >= patience:
                log(f"    early stop @ epoch {epoch} (best {best_epoch}, "
                    f"val_rmse={best_val_rmse:.4f})")
                break

    # ---- Reload best, final val predictions ----
    if best_state is not None:
        model.load_state_dict(best_state)
        # state dict load can fragment LSTM weight memory — repack for fast
        # final inference (and downstream interpret scripts that load .pt).
        for m in model.modules():
            if isinstance(m, nn.RNNBase):
                m.flatten_parameters()
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
        "oes_band_idx": (
            oes_band_idx.tolist() if oes_band_idx is not None else None
        ),
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
    parser.add_argument("--folds", type=str, default=None,
                        help="comma-separated fold ids to run (e.g. 4 or 2,4); overrides --limit-folds")
    parser.add_argument("--seed", type=int, default=None,
                        help="override config experiment.seed (used for seed-sweep diagnostics)")
    parser.add_argument("--slug-suffix", type=str, default=None,
                        help="append a suffix to the experiment title slug (e.g. seed42)")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.seed is not None:
        cfg["experiment"]["seed"] = int(args.seed)
    if args.slug_suffix:
        cfg["experiment"]["title"] = f"{cfg['experiment']['title']} {args.slug_suffix}"
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
    # Priority: CLI --folds > CLI --limit-folds > config experiment.limit_folds > default 1
    cfg_limit = cfg["experiment"].get("limit_folds", 1)
    cli_limit = args.limit_folds
    if args.folds is not None:
        fold_ids = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
        if not fold_ids:
            raise ValueError("--folds was provided but no fold ids were parsed")
        bad = [f for f in fold_ids if f < 0 or f >= split.n_folds]
        if bad:
            raise ValueError(f"fold ids out of range for {split.n_folds} folds: {bad}")
    else:
        effective_limit = cli_limit if cli_limit is not None else cfg_limit
        n_folds = min(effective_limit, split.n_folds) if effective_limit is not None else split.n_folds
        fold_ids = list(range(n_folds))
    log(f"Cache: {cache_root.relative_to(PROJECT_ROOT)}  |  split: {cfg['data']['split_file']} "
        f"({split.method}, running folds {fold_ids} of {split.n_folds})")

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
    per_wafer_norm = bool(cfg["data"].get("per_wafer_norm", False))
    store = WaferCycleStore(
        cache_root=cache_root,
        meas=meas,
        t_o=int(cfg["data"]["t_o"]),
        t_p=int(cfg["data"]["t_p"]),
        xgb_feat_names=xgb_feat_names,
        per_wafer_norm=per_wafer_norm,
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
    if per_wafer_norm:
        log(f"[setup] per-wafer normalization ENABLED (removes wafer-level absolute offset)")

    metrics_out: dict[str, dict] = {}
    fold_csv_rows: list[dict] = []
    all_epoch_rows: list[dict] = []
    all_sample_rows: list[dict] = []

    base_seed = int(cfg["experiment"]["seed"])
    for target_idx, target in enumerate(targets):
        log(f"\n=== Target: {target} ===")
        per_fold: list[dict] = []
        for f in fold_ids:
            heldout_lot = _fold_lot(split, f)
            fold_label = f"fold {f}" if heldout_lot is None else f"fold {f} (held-out lot {heldout_lot})"
            log(f"  -- {fold_label} --")
            fold_seed = base_seed + target_idx * 1000 + f
            set_seed(fold_seed)
            log(f"    [seed] reset RNG for target/fold: {fold_seed}")
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
                "oes_band_idx": fold_stats["oes_band_idx"],
                "xgb_feat_names": list(store.xgb_feat_names) if store.xgb_feat_names else [],
                "xgb_normalizer": store.xgb_normalizer.state_dict() if store.xgb_normalizer else {},
                "metrics": m,
            }, ckpt_path)
            if fold_stats["oes_band_idx"] is not None:
                np.save(
                    exp_dir / "logs" / f"oes_band_idx_{target}_fold{f}.npy",
                    np.asarray(fold_stats["oes_band_idx"], dtype=np.int32),
                )

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
