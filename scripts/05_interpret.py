"""Post-hoc interpretability analysis for Virtual Metrology models.

Two analyses are produced:
  1. XGBoost SHAP  — which statistical features drive the prediction
                     (built-in XGBoost tree SHAP, no external shap library)
  2. DL gradient-based cycle attribution  — which of the 100 BOSCH cycles
     most influences the model output (gradient norm w.r.t. cycle embeddings)

Both figures are saved to the respective experiment's figures/ directory.

Run from project root:
    python -m scripts.05_interpret ^
        --dl-exp  outputs/experiments/2026-05-01_00-56_dl-multimodal-singlefold ^
        --xgb-exp outputs/experiments/2026-04-30_15-32_baseline-xgb ^
        --target  oxide_etch ^
        --fold    0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import yaml

from src.data import Normalizer, ScalarStats, WaferCycleStore
from src.evaluation import load_split
from src.features import load_or_build_features
from src.models import make_model


# ── colour palette ──────────────────────────────────────────────────────────
_CLR = {
    "OES":     "#1976D2",   # blue
    "Process": "#E64A19",   # deep orange
    "XY":      "#388E3C",   # green
    "static":  "#546E7A",   # blue-grey
    "temporal":"#FF7043",   # orange
}

# stat names produced by summarise_cycle_series
_TEMPORAL_STATS = {"slope", "early", "late", "drift"}
_STATIC_STATS   = {"mean",  "std",  "min",  "max"}


def _load_measurements(cache_root: Path) -> pd.DataFrame:
    pq = cache_root / "measurements.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    return pd.read_csv(cache_root / "measurements.csv")


def _feature_group(name: str) -> str:
    """Return 'OES' / 'Process' / 'XY' for a feature column name."""
    if name.startswith("oes_"):
        return "OES"
    if name.startswith("proc_"):
        return "Process"
    return "XY"


def _feature_stat_class(name: str) -> str:
    """Return 'static' / 'temporal' / 'coord' for a feature column name."""
    if name in ("X", "Y"):
        return "coord"
    stat = name.rsplit("_", 1)[-1]
    if stat in _TEMPORAL_STATS:
        return "temporal"
    return "static"


# ============================================================================
# 1.  XGBoost SHAP
# ============================================================================

def run_xgb_shap(
    exp_dir: Path,
    cache_root: Path,
    target: str,
    fold: int,
    feature_set: str,
    n_oes_bands: int,
    split_file: str,
) -> None:
    print(f"\n[XGB SHAP] target={target}  fold={fold}")

    # ── design matrix ───────────────────────────────────────────────────────
    feat_df = load_or_build_features(cache_root, feature_set, n_oes_bands)
    meas    = _load_measurements(cache_root)

    feat_cols    = [c for c in feat_df.columns if c != "experiment_key"]
    df           = meas.merge(feat_df, on="experiment_key", how="left")
    feature_names = feat_cols + ["X", "Y"]
    X            = df[feature_names].to_numpy(dtype=np.float32)

    split = load_split(cache_root / split_file)
    _, val_mask = split.train_val_masks(fold)
    X_val = X[val_mask]
    print(f"  val samples: {X_val.shape[0]}")

    # ── load model & compute SHAP ────────────────────────────────────────────
    ckpt_path = exp_dir / "checkpoints" / f"{target}_fold{fold}.json"
    booster   = xgb.Booster()
    booster.load_model(str(ckpt_path))

    dval      = xgb.DMatrix(X_val, feature_names=feature_names)
    shap_raw  = booster.predict(dval, pred_contribs=True)   # (n, n_feat+1)
    shap_vals = shap_raw[:, :-1]                             # drop bias col
    mean_abs  = np.abs(shap_vals).mean(axis=0)               # (n_feat,)

    feature_names_arr = np.array(feature_names)
    groups     = np.array([_feature_group(n)      for n in feature_names])
    stat_class = np.array([_feature_stat_class(n) for n in feature_names])

    # ── figure: two panels ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # ── LEFT: top-20 horizontal bar ──────────────────────────────────────────
    top_n    = min(20, len(mean_abs))
    top_idx  = np.argsort(mean_abs)[-top_n:]            # ascending → bottom of hbar
    top_vals = mean_abs[top_idx]
    top_lbls = feature_names_arr[top_idx]
    top_grps = groups[top_idx]

    bar_clr  = [_CLR[g] for g in top_grps]
    ax0 = axes[0]
    ax0.barh(range(top_n), top_vals, color=bar_clr, edgecolor="white", linewidth=0.4)
    ax0.set_yticks(range(top_n))
    ax0.set_yticklabels(top_lbls, fontsize=7.5)
    ax0.set_xlabel("Mean |SHAP value| (μm)", fontsize=11)
    ax0.set_title(f"Top {top_n} Features\nXGBoost — {target}", fontsize=12, fontweight="bold")
    ax0.grid(axis="x", linestyle="--", alpha=0.4)

    legend_patches = [mpatches.Patch(color=_CLR[g], label=g)
                      for g in ("OES", "Process", "XY")]
    ax0.legend(handles=legend_patches, loc="lower right", fontsize=9)

    # ── RIGHT: grouped contribution (OES/Proc) × (static/temporal) + XY ─────
    categories  = ["OES\nStatic", "OES\nTemporal",
                   "Process\nStatic", "Process\nTemporal",
                   "XY"]
    cat_masks = [
        (groups == "OES")     & (stat_class == "static"),
        (groups == "OES")     & (stat_class == "temporal"),
        (groups == "Process") & (stat_class == "static"),
        (groups == "Process") & (stat_class == "temporal"),
        (groups == "XY"),
    ]
    cat_totals = [mean_abs[m].sum() for m in cat_masks]
    cat_colors = [
        _CLR["OES"], _CLR["OES"],
        _CLR["Process"], _CLR["Process"],
        _CLR["XY"],
    ]
    cat_alpha  = [1.0, 0.55, 1.0, 0.55, 1.0]   # static=full, temporal=lighter

    ax1 = axes[1]
    bars = ax1.bar(categories, cat_totals, color=cat_colors,
                   alpha=1.0, edgecolor="black", linewidth=0.6)
    for bar, alpha in zip(bars, cat_alpha):
        bar.set_alpha(alpha)

    ax1.set_ylabel("Sum of Mean |SHAP| (μm)", fontsize=11)
    ax1.set_title("Grouped Feature Contribution\n(Static vs Temporal)", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # annotate bar values
    for bar, val in zip(bars, cat_totals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(cat_totals) * 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8.5,
        )

    static_patch  = mpatches.Patch(color="gray", alpha=1.0,  label="Static (mean/std/min/max)")
    temp_patch    = mpatches.Patch(color="gray", alpha=0.55, label="Temporal (slope/early/late/drift)")
    ax1.legend(handles=[static_patch, temp_patch], fontsize=9)

    fig.suptitle(
        f"XGBoost Feature Importance — {target.replace('_', ' ')} (fold {fold})",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    out_dir = exp_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"shap_{target}_fold{fold}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(PROJECT_ROOT)}")

    # print top-5 features per group for quick inspection
    for grp in ("OES", "Process", "XY"):
        mask    = groups == grp
        sub_idx = np.where(mask)[0]
        if len(sub_idx) == 0:
            continue
        order   = np.argsort(mean_abs[sub_idx])[::-1][:5]
        top5    = [(feature_names[sub_idx[i]], float(mean_abs[sub_idx[i]])) for i in order]
        print(f"  top-5 {grp}: " + "  |  ".join(f"{n}={v:.4f}" for n, v in top5))


# ============================================================================
# 2.  DL Gradient-Based Cycle Attribution
# ============================================================================

def _forward_from_cycles(model, cycle_emb: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """Run the LSTM + head on already-computed cycle embeddings.

    Mirrors CycleAwareBiLSTM.forward() but starts from cycle_emb so that
    retain_grad() on cycle_emb captures the gradient from the full head.
    """
    seq, _ = model.lstm(cycle_emb)                       # (B, 100, 2h)

    if model.cycle_pool is not None:
        wafer = model.cycle_pool(seq)                    # (B, 2h) learned attention
    else:
        wafer = seq.mean(dim=1)                          # (B, 2h) uniform mean

    if model.cfg.use_xy:
        xy_enc = model.xy_encoder(xy) if model.xy_encoder is not None else xy
        if model.use_film:
            return model.film_head(wafer, xy_enc)        # (B, n_pts)
        n_pts = xy_enc.shape[1]
        wafer_b = wafer.unsqueeze(1).expand(-1, n_pts, -1)
        full = torch.cat([wafer_b, xy_enc], dim=-1)
        return model.head(full).squeeze(-1)

    return model.head(wafer).squeeze(-1)                 # (B,)


def run_dl_attribution(
    exp_dir: Path,
    cache_root: Path,
    target: str,
    fold: int,
) -> None:
    print(f"\n[DL Attribution] target={target}  fold={fold}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    # ── load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = exp_dir / "checkpoints" / f"{target}_fold{fold}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg              = ckpt["config"]
    modality         = cfg["data"]["modality"]
    proc_kept_names  = list(ckpt["proc_kept_names"])

    oes_normalizer   = Normalizer.from_state(ckpt["oes_normalizer"])
    proc_normalizer  = Normalizer.from_state(ckpt["proc_normalizer"])
    x_stats          = ScalarStats(**ckpt["x_stats"])
    y_stats          = ScalarStats(**ckpt["y_stats"])

    # ── rebuild model ────────────────────────────────────────────────────────
    params = dict(cfg["model"]["params"])
    if params.get("proc_encoder") is not None:
        params["proc_encoder"] = dict(params["proc_encoder"])
        params["proc_encoder"]["in_channels"] = len(proc_kept_names)
    model = make_model(cfg["model"]["name"], params)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    print(f"  model loaded  (modality={modality})")

    # ── load val wafers ──────────────────────────────────────────────────────
    meas   = _load_measurements(cache_root)
    split  = load_split(cache_root / cfg["data"]["split_file"])
    val_keys = split.wafer_keys[split.wafer_fold_id == fold].astype(str).tolist()
    print(f"  val wafers: {len(val_keys)}")

    store = WaferCycleStore(
        cache_root=cache_root,
        meas=meas,
        t_o=int(cfg["data"]["t_o"]),
        t_p=int(cfg["data"]["t_p"]),
    )
    store.proc_kept_names = proc_kept_names
    store.load_wafers(val_keys, progress=True)

    # ── gradient attribution ─────────────────────────────────────────────────
    all_importances: list[np.ndarray] = []

    for key in val_keys:
        rec = store[key]

        oes_np  = oes_normalizer.apply(rec.oes_raw).astype(np.float32)   # (100, T_o, W)
        proc_np = proc_normalizer.apply(rec.proc_raw).astype(np.float32) # (100, T_p, F)
        X_n     = x_stats.apply(rec.points_X)
        Y_n     = y_stats.apply(rec.points_Y)
        xy_np   = np.stack([X_n, Y_n], axis=-1).astype(np.float32)       # (89, 2)

        oes_t  = torch.from_numpy(oes_np).unsqueeze(0).to(device)  if modality in ("oes",  "multimodal") else None
        proc_t = torch.from_numpy(proc_np).unsqueeze(0).to(device) if modality in ("proc", "multimodal") else None
        xy_t   = torch.from_numpy(xy_np).unsqueeze(0).to(device)

        # encode cycles, then intercept the embedding to track its gradient.
        # cuDNN RNN backward requires training mode on the LSTM module;
        # other layers stay in eval mode so BN/Dropout behave correctly.
        model.lstm.train()
        cycle_emb = model.encode_cycles(oes_t, proc_t)   # (1, 100, d)
        cycle_emb.retain_grad()

        pred = _forward_from_cycles(model, cycle_emb, xy_t)  # (1, n_pts)
        pred.sum().backward()
        model.lstm.eval()

        # L2 norm over embedding dim → scalar importance per cycle
        imp = cycle_emb.grad.norm(dim=-1).squeeze(0)          # (100,)
        all_importances.append(imp.detach().cpu().numpy())

    importances = np.stack(all_importances, axis=0)   # (n_val, 100)
    mean_imp    = importances.mean(axis=0)             # (100,)
    std_imp     = importances.std(axis=0)

    # normalise to [0, 1] for readability
    peak        = mean_imp.max() or 1.0
    mean_norm   = mean_imp / peak
    std_norm    = std_imp  / peak

    # ── plot ─────────────────────────────────────────────────────────────────
    cycles  = np.arange(1, 101)
    cmap    = plt.cm.RdYlBu_r                          # blue → red from early to late
    clrs    = [cmap(i / 99) for i in range(100)]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(cycles, mean_norm, color=clrs, width=0.85, alpha=0.85, zorder=2)
    ax.fill_between(
        cycles,
        np.maximum(mean_norm - std_norm, 0.0),
        mean_norm + std_norm,
        alpha=0.22, color="steelblue", label="±1 std (across val wafers)",
    )

    ax.set_xlim(0.5, 100.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Cycle Number  (1 = first BOSCH cycle after ignition)", fontsize=11)
    ax.set_ylabel("Relative Gradient Norm (normalised to peak)", fontsize=11)
    ax.set_title(
        f"Cycle-Level Attribution — DL Model\n"
        f"Target: {target.replace('_', ' ')}  |  Fold {fold}  |  n_val = {len(val_keys)} wafers",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)
    ax.legend(fontsize=10)

    # colour-bar for cycle progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 100))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.018, pad=0.01)
    cbar.set_label("Cycle number", fontsize=9)

    fig.tight_layout()
    out_dir  = exp_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"cycle_attribution_{target}_fold{fold}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.relative_to(PROJECT_ROOT)}")

    top10 = np.argsort(mean_imp)[-10:][::-1] + 1   # 1-indexed
    print(f"  top-10 cycles (by gradient norm): {sorted(top10.tolist())}")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dl-exp",  type=Path, required=True,
        help="path to DL experiment dir (relative to project root or absolute)",
    )
    parser.add_argument(
        "--xgb-exp", type=Path, required=True,
        help="path to XGBoost experiment dir",
    )
    parser.add_argument(
        "--target", default="oxide_etch",
        choices=["si_etch", "oxide_etch"],
        help="prediction target to analyse (default: oxide_etch)",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="fold index (default: 0)",
    )
    parser.add_argument(
        "--cache-version", default="v1",
        help="cache subdirectory under cache/ (default: v1)",
    )
    args = parser.parse_args()

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else PROJECT_ROOT / p

    dl_exp   = _resolve(args.dl_exp)
    xgb_exp  = _resolve(args.xgb_exp)
    cache_root = PROJECT_ROOT / "cache" / args.cache_version

    for p, label in ((dl_exp, "--dl-exp"), (xgb_exp, "--xgb-exp")):
        if not p.exists():
            raise FileNotFoundError(f"{label} path not found: {p}")

    xgb_cfg = yaml.safe_load((xgb_exp / "config.yaml").read_text(encoding="utf-8"))

    run_xgb_shap(
        exp_dir      = xgb_exp,
        cache_root   = cache_root,
        target       = args.target,
        fold         = args.fold,
        feature_set  = xgb_cfg["data"]["feature_set"],
        n_oes_bands  = int(xgb_cfg["data"]["n_oes_bands"]),
        split_file   = xgb_cfg["data"]["split_file"],
    )

    run_dl_attribution(
        exp_dir    = dl_exp,
        cache_root = cache_root,
        target     = args.target,
        fold       = args.fold,
    )

    print("\nAll done.")


if __name__ == "__main__":
    main()
