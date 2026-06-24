"""Generate final-presentation figures from saved experiment outputs.

The figures are intentionally presentation-oriented: wide layout, large labels,
and filenames prefixed with ``final_`` under ``outputs/figures``.

Run:
    .venv\\python.exe -m scripts.11_make_final_presentation_figures
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.gridspec import GridSpec

from src.evaluation import load_split
from src.features import load_or_build_features

CACHE = PROJECT_ROOT / "cache" / "v1"
OUT = PROJECT_ROOT / "outputs" / "figures"

XGB_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-27_14-00_baseline-xgb"
INITIAL_DL_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-21_02-09_dl-multimodal-5fold"
LONGRUN_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-26_23-24_dl-multimodal-oes-topk256-longrun-5fold"
AUX_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-27_04-16_dl-multimodal-oes-aux-wafer-mean-5fold"
MIXUP_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-28_03-16_dl-multimodal-oes-aux-mixup-5fold"
FINAL_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-28_04-19_dl-multimodal-oes-aux-mixup-ema-longrun-5fold"
SINGLE_FOLD_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-01_00-56_dl-multimodal-singlefold"
OES_ONLY_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-01_13-00_dl-oes-only-singlefold"
PROC_ONLY_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-04_11-18_dl-proc-only-singlefold"
ATTN_POOL_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-01_03-16_dl-oxide-v2-attn-singlefold"

PALETTE = {
    "xgb": "#4C78A8",
    "dl": "#E45756",
    "aux": "#72B7B2",
    "mixup": "#F58518",
    "ema": "#54A24B",
    "muted": "#6B7280",
    "dark": "#243447",
}

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "legend.frameon": False,
    }
)


@dataclass(frozen=True)
class Experiment:
    label: str
    path: Path
    color: str


EVOLUTION = [
    Experiment("XGB\nbaseline", XGB_EXP, PALETTE["xgb"]),
    Experiment("Initial DL\n5-fold", INITIAL_DL_EXP, "#9CA3AF"),
    Experiment("OES top-k\nlongrun", LONGRUN_EXP, "#7F7F7F"),
    Experiment("+ AUX\nwafer mean", AUX_EXP, PALETTE["aux"]),
    Experiment("+ Mixup\nwafer-level", MIXUP_EXP, PALETTE["mixup"]),
    Experiment("+ EMA\nfinal", FINAL_EXP, PALETTE["ema"]),
]


def _read_fold_metrics(exp: Path, target: str = "oxide_etch") -> pd.DataFrame:
    path = exp / "logs" / "fold_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df[df["target"] == target].copy()


def _save(fig: plt.Figure, name: str, written: list[tuple[str, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / name
    fig.savefig(out)
    plt.close(fig)
    written.append((name, str(out.relative_to(PROJECT_ROOT))))
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


def _aggregate_row(exp: Experiment) -> dict[str, object]:
    df = _read_fold_metrics(exp.path)
    return {
        "model": exp.label.replace("\n", " "),
        "r2_mean": df["r2"].mean(),
        "r2_std": df["r2"].std(ddof=0),
        "rmse_mean": df["rmse"].mean(),
        "rmse_std": df["rmse"].std(ddof=0),
        "mae_mean": df["mae"].mean(),
        "mape_mean": df["mape_pct"].mean(),
    }


def fig_model_evolution(written: list[tuple[str, str]]) -> None:
    rows = [_aggregate_row(exp) for exp in EVOLUTION]
    agg = pd.DataFrame(rows)
    x = np.arange(len(EVOLUTION))

    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    colors = [exp.color for exp in EVOLUTION]
    bars = ax.bar(
        x,
        agg["r2_mean"],
        yerr=agg["r2_std"],
        capsize=5,
        color=colors,
        edgecolor="#222",
        linewidth=0.7,
    )
    for i, exp in enumerate(EVOLUTION):
        fold_df = _read_fold_metrics(exp.path)
        jitter = np.linspace(-0.16, 0.16, len(fold_df))
        ax.scatter(
            np.full(len(fold_df), i) + jitter,
            fold_df["r2"],
            s=42,
            color="white",
            edgecolor="#222",
            linewidth=0.8,
            zorder=3,
        )
        for bar, value in zip([bars[i]], [agg.loc[i, "r2_mean"]]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.035,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                color="#111",
            )

    ax.axhline(0.551, color=PALETTE["xgb"], linestyle="--", linewidth=1.1, alpha=0.8)
    ax.text(0.1, 0.558, "XGB baseline mean", color=PALETTE["xgb"], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([exp.label for exp in EVOLUTION])
    ax.set_ylim(0.0, 0.86)
    ax.set_ylabel("oxide_etch R2 (5-fold wafer CV)")
    ax.set_title("Model evolution: from baseline to final cycle-aware DL", fontweight="bold")
    ax.text(
        0.01,
        0.03,
        "Dots show individual folds; bars show mean +- std.",
        transform=ax.transAxes,
        color=PALETTE["muted"],
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, "final_01_model_evolution_r2.png", written)


def fig_fold_comparison(written: list[tuple[str, str]]) -> None:
    xgb_df = _read_fold_metrics(XGB_EXP)
    final_df = _read_fold_metrics(FINAL_EXP)
    folds = sorted(final_df["fold"].unique())
    x = np.arange(len(folds))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.bar(x - width / 2, xgb_df.sort_values("fold")["r2"], width, label="XGBoost", color=PALETTE["xgb"])
    ax.bar(x + width / 2, final_df.sort_values("fold")["r2"], width, label="Final DL", color=PALETTE["ema"])
    for idx, row in final_df.sort_values("fold").iterrows():
        fold = int(row["fold"])
        ax.text(fold + width / 2, row["r2"] + 0.02, f"{row['r2']:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylim(0.0, 0.9)
    ax.set_ylabel("oxide_etch R2")
    ax.set_title("Fold-level comparison: XGBoost baseline vs final DL", fontweight="bold")
    ax.legend(loc="upper left")
    ax.text(
        0.62,
        0.08,
        "Fold 2 remains the main bottleneck;\nFold 4 collapse is largely recovered.",
        transform=ax.transAxes,
        fontsize=11,
        color=PALETTE["dark"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
    )
    fig.tight_layout()
    _save(fig, "final_02_fold_r2_xgb_vs_final_dl.png", written)


def fig_metric_improvement(written: list[tuple[str, str]]) -> None:
    xgb_df = _read_fold_metrics(XGB_EXP)
    final_df = _read_fold_metrics(FINAL_EXP)
    metrics = [
        ("RMSE", "rmse", "lower is better"),
        ("MAE", "mae", "lower is better"),
        ("MAPE (%)", "mape_pct", "lower is better"),
        ("R2", "r2", "higher is better"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8))
    for ax, (label, col, note) in zip(axes, metrics):
        xv = xgb_df[col].mean()
        dv = final_df[col].mean()
        ax.bar([0, 1], [xv, dv], color=[PALETTE["xgb"], PALETTE["ema"]], width=0.58)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["XGB", "Final DL"])
        ax.set_title(label, fontweight="bold")
        ax.text(0, xv, f"{xv:.3f}" if col == "r2" else f"{xv:.4f}", ha="center", va="bottom", fontsize=10)
        ax.text(1, dv, f"{dv:.3f}" if col == "r2" else f"{dv:.4f}", ha="center", va="bottom", fontsize=10)
        if col != "r2":
            improvement = (xv - dv) / xv * 100
            ax.text(
                0.5,
                0.88,
                f"{improvement:.1f}% lower",
                transform=ax.transAxes,
                ha="center",
                fontweight="bold",
                color=PALETTE["ema"],
            )
        else:
            improvement = dv - xv
            ax.text(
                0.5,
                0.88,
                f"+{improvement:.3f}",
                transform=ax.transAxes,
                ha="center",
                fontweight="bold",
                color=PALETTE["ema"],
            )
        ax.text(0.5, 0.78, note, transform=ax.transAxes, ha="center", fontsize=9, color=PALETTE["muted"])

    fig.suptitle("Final DL improves oxide_etch prediction over XGBoost", fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "final_03_metric_improvement_summary.png", written)


def _xgb_predictions(target: str = "oxide_etch") -> pd.DataFrame:
    feat_df = load_or_build_features(CACHE, feature_set="baseline_xgb_v1", n_oes_bands=10)
    meas = pd.read_csv(CACHE / "measurements.csv")
    df = meas.merge(feat_df, on="experiment_key", how="left", validate="many_to_one")
    feat_cols = [c for c in feat_df.columns if c != "experiment_key"]
    feature_names = feat_cols + ["X", "Y"]
    x_mat = df[feature_names].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32)
    split = load_split(CACHE / "splits" / "kfold5_wafer.npz")

    rows: list[pd.DataFrame] = []
    for fold in range(split.n_folds):
        _, val_mask = split.train_val_masks(fold)
        model = xgb.XGBRegressor()
        model.load_model(str(XGB_EXP / "checkpoints" / f"{target}_fold{fold}.json"))
        pred = model.predict(x_mat[val_mask])
        rows.append(
            pd.DataFrame(
                {
                    "target": target,
                    "fold": fold,
                    "experiment_key": df.loc[val_mask, "experiment_key"].to_numpy(),
                    "point_idx": df.loc[val_mask].groupby("experiment_key").cumcount().to_numpy(),
                    "y_true": y[val_mask],
                    "y_pred": pred,
                    "residual": pred - y[val_mask],
                    "abs_error": np.abs(pred - y[val_mask]),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _final_predictions() -> pd.DataFrame:
    pred = pd.read_csv(FINAL_EXP / "logs" / "sample_predictions.csv")
    return pred[pred["target"] == "oxide_etch"].copy()


def fig_pred_scatter(written: list[tuple[str, str]]) -> None:
    xgb_pred = _xgb_predictions()
    dl_pred = _final_predictions()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0), sharex=True, sharey=True)
    for ax, df, label, color in [
        (axes[0], xgb_pred, "XGBoost baseline", PALETTE["xgb"]),
        (axes[1], dl_pred, "Final DL", PALETTE["ema"]),
    ]:
        rmse = float(np.sqrt(np.mean((df["y_pred"] - df["y_true"]) ** 2)))
        r2 = 1.0 - np.var(df["y_pred"] - df["y_true"]) / np.var(df["y_true"])
        ax.scatter(df["y_true"], df["y_pred"], s=10, alpha=0.32, color=color, edgecolor="none")
        lo = min(df["y_true"].min(), df["y_pred"].min())
        hi = max(df["y_true"].max(), df["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], color="#222", linestyle="--", linewidth=1.1)
        ax.set_title(label, fontweight="bold", loc="left")
        ax.set_xlabel("True oxide_etch")
        ax.set_ylabel("Predicted oxide_etch")
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.04,
            0.96,
            f"n = {len(df):,}\npooled R2 = {r2:.3f}\npooled RMSE = {rmse:.4f}",
            transform=ax.transAxes,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
        )

    fig.suptitle("Prediction scatter: all validation points across 5 folds", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "final_04_pred_vs_true_xgb_vs_final_dl.png", written)


def fig_residuals(written: list[tuple[str, str]]) -> None:
    xgb_pred = _xgb_predictions()
    dl_pred = _final_predictions()

    fig = plt.figure(figsize=(14.0, 6.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.28)

    ax1 = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-0.16, 0.16, 80)
    ax1.hist(xgb_pred["residual"], bins=bins, alpha=0.58, color=PALETTE["xgb"], label="XGB")
    ax1.hist(dl_pred["residual"], bins=bins, alpha=0.58, color=PALETTE["ema"], label="Final DL")
    ax1.axvline(0, color="#222", linewidth=1)
    ax1.set_xlabel("Residual (pred - true)")
    ax1.set_ylabel("Count")
    ax1.set_title("Residual distribution", fontweight="bold", loc="left")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    grouped = [dl_pred.loc[dl_pred["fold"] == fold, "abs_error"] for fold in sorted(dl_pred["fold"].unique())]
    bp = ax2.boxplot(grouped, patch_artist=True, widths=0.6, showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor=PALETTE["ema"], alpha=0.65, edgecolor="#222")
    for med in bp["medians"]:
        med.set(color="#111", linewidth=1.6)
    ax2.set_xticks(np.arange(1, len(grouped) + 1))
    ax2.set_xticklabels([f"Fold {i}" for i in sorted(dl_pred["fold"].unique())])
    ax2.set_ylabel("Absolute error")
    ax2.set_title("Final DL absolute error by fold", fontweight="bold", loc="left")

    fig.suptitle("Error analysis for oxide_etch predictions", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "final_05_residual_analysis.png", written)


def fig_epoch_curves(written: list[tuple[str, str]]) -> None:
    epoch = pd.read_csv(FINAL_EXP / "logs" / "epoch_log.csv")
    epoch = epoch[epoch["target"] == "oxide_etch"].copy()
    fold_metrics = _read_fold_metrics(FINAL_EXP)

    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.0))
    ax1, ax2 = axes
    for fold, df in epoch.groupby("fold"):
        ax1.plot(df["epoch"], df["val_r2"], linewidth=1.8, label=f"Fold {fold}")
        best_ep = int(fold_metrics.loc[fold_metrics["fold"] == fold, "best_epoch"].iloc[0])
        best_row = df[df["epoch"] == best_ep]
        if not best_row.empty:
            ax1.scatter(best_ep, best_row["val_r2"].iloc[0], s=46, edgecolor="#111", zorder=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation R2")
    ax1.set_title("EMA validation curve by fold", fontweight="bold", loc="left")
    ax1.legend(ncol=2)

    for fold, df in epoch.groupby("fold"):
        ax2.plot(df["epoch"], df["train_rmse"], linewidth=1.6, alpha=0.9, label=f"Fold {fold}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Train RMSE")
    ax2.set_title("Training RMSE by fold", fontweight="bold", loc="left")
    ax2.legend(ncol=2)
    ax2.text(
        0.52,
        0.08,
        "Fold 2 peaks early (best epoch 39),\nwhile folds 3/4 keep improving late.",
        transform=ax2.transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.92),
    )

    fig.suptitle("Final DL learning dynamics: why long-run + EMA was useful", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "final_06_learning_curves_ema_longrun.png", written)


def fig_ablation_heatmap(written: list[tuple[str, str]]) -> None:
    matrix = []
    labels = []
    for exp in EVOLUTION:
        df = _read_fold_metrics(exp.path).sort_values("fold")
        matrix.append(df["r2"].to_numpy())
        labels.append(exp.label.replace("\n", " "))
    arr = np.vstack(matrix)

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0.1, vmax=0.82)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels([f"Fold {i}" for i in range(5)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if arr[i, j] > 0.55 else "#111"
            ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color=color, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("R2")
    ax.set_title("Ablation heatmap: which folds improved?", fontweight="bold", loc="left")
    fig.tight_layout()
    _save(fig, "final_07_ablation_fold_heatmap.png", written)


def fig_wafer_error_map(written: list[tuple[str, str]]) -> None:
    pred = _final_predictions()
    meas = pd.read_csv(CACHE / "measurements.csv")
    point_pos = (
        meas.sort_values(["experiment_key"])
        .groupby("experiment_key")
        .head(89)[["X", "Y"]]
        .reset_index(drop=True)
    )
    point_pos["point_idx"] = np.arange(len(point_pos))
    merged = pred.merge(point_pos, on="point_idx", how="left")
    pos = (
        merged.groupby(["point_idx", "X", "Y"], as_index=False)
        .agg(mean_abs_error=("abs_error", "mean"), mean_residual=("residual", "mean"))
    )
    radius = np.sqrt(pos["X"] ** 2 + pos["Y"] ** 2).max() * 1.06
    theta = np.linspace(0, 2 * np.pi, 240)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    configs = [
        ("Mean absolute error by wafer position", "mean_abs_error", "magma", None),
        ("Mean residual bias by wafer position", "mean_residual", "RdBu_r", 0.0),
    ]
    for ax, (title, col, cmap, center) in zip(axes, configs):
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="#6B7280", linewidth=1)
        if center is None:
            sc = ax.scatter(pos["X"], pos["Y"], c=pos[col], s=80, cmap=cmap, edgecolor="#111", linewidth=0.45)
        else:
            vmax = float(np.abs(pos[col]).max())
            sc = ax.scatter(
                pos["X"],
                pos["Y"],
                c=pos[col],
                s=80,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                edgecolor="#111",
                linewidth=0.45,
            )
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title, fontweight="bold", loc="left")
        cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
        cb.set_label("oxide_etch")
    fig.suptitle("Final DL spatial error map across 89 measurement positions", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "final_08_spatial_error_wafer_map.png", written)


def fig_aux_mixup_ema_explainer(written: list[tuple[str, str]]) -> None:
    steps = [
        ("Point loss only", "89 point targets\ncan fit local errors\nbut wafer-level mode may collapse", 0.596, "#7F7F7F"),
        ("AUX wafer mean", "forces wafer_repr\nto encode global etch level", 0.621, PALETTE["aux"]),
        ("Wafer mixup", "creates cross-wafer\nintermediate states", 0.644, PALETTE["mixup"]),
        ("EMA + 120ep", "averages noisy weights\nand keeps late improvement", 0.666, PALETTE["ema"]),
    ]
    fig = plt.figure(figsize=(13.8, 6.7))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.35, 0.95], hspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(steps))
    y = [s[2] for s in steps]
    colors = [s[3] for s in steps]
    ax.plot(x, y, color="#111", linewidth=2.2, zorder=1)
    ax.scatter(x, y, s=280, color=colors, edgecolor="#111", linewidth=1.0, zorder=2)
    for i, (title, desc, value, color) in enumerate(steps):
        ax.text(i, value + 0.022, f"{value:.3f}", ha="center", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps])
    ax.set_ylabel("Mean oxide_etch R2")
    ax.set_ylim(0.57, 0.705)
    ax.set_title("Why the final training recipe worked", fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0)

    ax_cards = fig.add_subplot(gs[1, 0])
    ax_cards.set_axis_off()
    ax_cards.set_xlim(0, 1)
    ax_cards.set_ylim(0, 1)
    card_x = np.linspace(0.12, 0.88, len(steps))
    for x_pos, (title, desc, _value, color) in zip(card_x, steps):
        ax_cards.text(
            x_pos,
            0.54,
            f"{title}\n{desc}",
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor=color,
                linewidth=2.0,
            ),
        )
    ax_cards.text(
        0.5,
        0.04,
        "Interpretation: the recipe first stabilizes wafer-level representation, then regularizes the training trajectory.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["muted"],
    )
    fig.tight_layout()
    _save(fig, "final_09_aux_mixup_ema_explainer.png", written)


def fig_pipeline_timeline(written: list[tuple[str, str]]) -> None:
    events = [
        ("Topic proposal", "1D-CNN pipeline\nsensor -> VM"),
        ("Professor feedback", "2D-CNN per cycle\n+ BiLSTM sequence"),
        ("si collapse", "XY dependence found\nFiLM + Fourier XY"),
        ("Single fold success", "oxide R2 ~0.73\nmultimodal selected"),
        ("5-fold audit", "fold 2/4 collapse\nrobustness problem"),
        ("Final recipe", "OES top-k + AUX\n+ mixup + EMA"),
    ]
    fig, ax = plt.subplots(figsize=(15.5, 4.9))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    xs = np.linspace(0.06, 0.94, len(events))
    y = 0.55
    ax.plot(xs, np.full_like(xs, y), color="#CBD5E1", linewidth=5, solid_capstyle="round")
    for i, ((title, desc), x) in enumerate(zip(events, xs), start=1):
        color = PALETTE["ema"] if i == len(events) else PALETTE["xgb"]
        ax.scatter([x], [y], s=520, color=color, edgecolor="#111", linewidth=1.0, zorder=3)
        ax.text(x, y, str(i), color="white", ha="center", va="center", fontweight="bold", fontsize=13)
        ax.text(x, 0.86, title, ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.text(x, 0.23, desc, ha="center", va="top", fontsize=10.5, color=PALETTE["dark"])
    ax.text(
        0.5,
        0.04,
        "Recommended story: each model change responds to a concrete failure mode discovered during validation.",
        ha="center",
        fontsize=11,
        color=PALETTE["muted"],
    )
    ax.text(
        0.0,
        0.98,
        "Research narrative: model development through failure diagnosis",
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        transform=ax.transAxes,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.97, bottom=0.06)
    _save(fig, "final_10_model_development_timeline.png", written)


def fig_modality_ablation(written: list[tuple[str, str]]) -> None:
    items = [
        ("OES-only", OES_ONLY_EXP, "#B279A2"),
        ("Process-only", PROC_ONLY_EXP, "#F58518"),
        ("Multimodal\nmean pool", SINGLE_FOLD_EXP, PALETTE["ema"]),
        ("Multimodal\nattention pool", ATTN_POOL_EXP, "#9CA3AF"),
    ]
    values = []
    rmses = []
    for _label, exp, _color in items:
        df = _read_fold_metrics(exp)
        values.append(float(df["r2"].iloc[0]))
        rmses.append(float(df["rmse"].iloc[0]))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = np.arange(len(items))
    bars = ax.bar(
        x,
        values,
        color=[c for *_rest, c in items],
        edgecolor="#111",
        linewidth=0.8,
        width=0.62,
    )
    for bar, r2, rmse in zip(bars, values, rmses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r2 + 0.025,
            f"R2 {r2:.3f}\nRMSE {rmse:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, *_ in items])
    ax.set_ylim(0.0, 0.84)
    ax.set_ylabel("oxide_etch R2 (single fold)")
    ax.set_title("Modality and pooling ablation: why multimodal mean-pool was selected", fontweight="bold", loc="left")
    ax.text(
        0.54,
        0.18,
        "Process carries strong signal,\nOES adds complementary information.\nAttention pooling underperformed mean pooling.",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
    )
    fig.tight_layout()
    _save(fig, "final_11_modality_pooling_ablation.png", written)


def fig_singlefold_to_kfold(written: list[tuple[str, str]]) -> None:
    single = _read_fold_metrics(SINGLE_FOLD_EXP)
    initial = _read_fold_metrics(INITIAL_DL_EXP).sort_values("fold")
    final = _read_fold_metrics(FINAL_EXP).sort_values("fold")

    fig = plt.figure(figsize=(13.5, 6.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.9, 1.35], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ["Single-fold\nsuccess", "Initial 5-fold\nmean", "Final 5-fold\nmean"]
    means = [float(single["r2"].iloc[0]), float(initial["r2"].mean()), float(final["r2"].mean())]
    stds = [0.0, float(initial["r2"].std(ddof=0)), float(final["r2"].std(ddof=0))]
    colors = [PALETTE["ema"], "#9CA3AF", PALETTE["mixup"]]
    ax1.bar(np.arange(3), means, yerr=stds, capsize=5, color=colors, edgecolor="#111", linewidth=0.8)
    for i, value in enumerate(means):
        ax1.text(i, value + 0.03, f"{value:.3f}", ha="center", fontweight="bold")
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0.0, 0.85)
    ax1.set_ylabel("oxide_etch R2")
    ax1.set_title("Why 5-fold validation changed the story", fontweight="bold", loc="left")

    ax2 = fig.add_subplot(gs[0, 1])
    folds = np.arange(5)
    ax2.plot(folds, initial["r2"], marker="o", linewidth=2.2, color="#9CA3AF", label="Initial DL 5-fold")
    ax2.plot(folds, final["r2"], marker="o", linewidth=2.2, color=PALETTE["ema"], label="Final DL")
    ax2.axhline(float(single["r2"].iloc[0]), color=PALETTE["ema"], linestyle="--", alpha=0.65, label="Single-fold result")
    for fold, value in zip(folds, initial["r2"]):
        if fold in (2, 4):
            ax2.text(fold, value - 0.055, f"collapse\n{value:.2f}", ha="center", fontsize=9, color=PALETTE["dark"])
    ax2.set_xticks(folds)
    ax2.set_xticklabels([f"Fold {i}" for i in folds])
    ax2.set_ylim(0.0, 0.85)
    ax2.set_ylabel("oxide_etch R2")
    ax2.set_title("Collapse diagnosis and recovery", fontweight="bold", loc="left")
    ax2.legend(loc="lower left")

    fig.suptitle("Single-fold success was not enough: robustness emerged as the main problem", fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, "final_12_singlefold_to_kfold_robustness.png", written)


def _diagram_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str,
    text_color: str = "#111827",
    fontsize: float = 11,
    weight: str = "bold",
) -> None:
    patch = plt.matplotlib.patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight=weight,
        linespacing=1.25,
    )


def _diagram_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#374151",
    lw: float = 1.8,
    rad: float = 0.0,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=14,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )


def _mini_wafer(
    ax: plt.Axes,
    cx: float,
    cy: float,
    r: float,
    mode: str,
    title: str,
) -> None:
    theta = np.linspace(0, 2 * np.pi, 220)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), color="#6B7280", linewidth=1.0)
    xs = np.linspace(-0.72, 0.72, 7)
    ys = np.linspace(-0.72, 0.72, 7)
    px, py, val = [], [], []
    for x in xs:
        for y in ys:
            if x * x + y * y <= 0.78 * 0.78:
                px.append(cx + r * x)
                py.append(cy + r * y)
                if mode == "same":
                    val.append(0.45)
                else:
                    val.append(0.25 + 0.5 * (x + 0.72) / 1.44 + 0.15 * np.sin(5 * y))
    cmap = "Greys" if mode == "same" else "viridis"
    ax.scatter(px, py, c=val, s=28, cmap=cmap, vmin=0, vmax=1, edgecolor="#111827", linewidth=0.25)
    ax.text(cx, cy - r - 0.055, title, ha="center", va="top", fontsize=9.5, color=PALETTE["dark"])


def fig_xy_film_comparison(written: list[tuple[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.965,
        "How XY information enters the regression head: raw concat vs FiLM modulation",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.925,
        "Problem context: initial DL predicted si_etch poorly because wafer-position dependence was too weakly represented.",
        ha="center",
        va="top",
        fontsize=11.5,
        color=PALETTE["muted"],
    )

    # Panel backgrounds
    _diagram_box(ax, (0.035, 0.095), 0.445, 0.765, "", "#F8FAFC", "#CBD5E1", fontsize=1)
    _diagram_box(ax, (0.52, 0.095), 0.445, 0.765, "", "#F0FDF4", "#86EFAC", fontsize=1)
    ax.text(0.257, 0.83, "Initial approach: XY concat", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.742, 0.83, "Improved approach: Fourier XY + FiLM", ha="center", fontsize=15, fontweight="bold")

    # Left: concat path
    _diagram_box(ax, (0.10, 0.685), 0.31, 0.08, "OES + Process encoders\n-> BiLSTM -> Mean pool", "#EDE9FE", PALETTE["dark"])
    _diagram_box(ax, (0.125, 0.555), 0.26, 0.075, "Wafer representation\none vector per wafer", "#DDD6FE", "#6D28D9")
    _diagram_box(ax, (0.135, 0.425), 0.24, 0.075, "Copy same wafer_repr\nto all 89 points", "#F3F4F6", "#6B7280")
    _diagram_box(ax, (0.055, 0.295), 0.17, 0.07, "Raw XY\n(89 x 2)", "#DCFCE7", "#15803D")
    _diagram_box(ax, (0.255, 0.295), 0.17, 0.07, "Concat\n[repr, x, y]", "#FEF3C7", "#D97706")
    _diagram_box(ax, (0.14, 0.165), 0.22, 0.075, "Regression head\nlearns spatial effect late", "#E5E7EB", "#374151")
    _diagram_arrow(ax, (0.255, 0.685), (0.255, 0.63))
    _diagram_arrow(ax, (0.255, 0.555), (0.255, 0.50))
    _diagram_arrow(ax, (0.255, 0.425), (0.312, 0.365), rad=-0.08)
    _diagram_arrow(ax, (0.14, 0.295), (0.255, 0.33), rad=-0.12, color="#15803D")
    _diagram_arrow(ax, (0.34, 0.295), (0.265, 0.24), rad=0.08)
    _mini_wafer(ax, 0.40, 0.245, 0.055, "same", "same base repr\nat every point")

    # Right: FiLM path
    _diagram_box(ax, (0.585, 0.685), 0.31, 0.08, "OES + Process encoders\n-> BiLSTM -> Mean pool", "#EDE9FE", PALETTE["dark"])
    _diagram_box(ax, (0.61, 0.555), 0.26, 0.075, "Wafer representation\none vector per wafer", "#DDD6FE", "#6D28D9")
    _diagram_box(ax, (0.545, 0.425), 0.18, 0.07, "XY coords\n(89 x 2)", "#DCFCE7", "#15803D")
    _diagram_box(ax, (0.755, 0.425), 0.18, 0.07, "Fourier encoder\n(89 x 64)", "#BBF7D0", "#15803D")
    _diagram_box(ax, (0.61, 0.315), 0.26, 0.08, "Generate gamma, beta\nper measurement point", "#DCFCE7", "#15803D")
    _diagram_box(ax, (0.59, 0.205), 0.30, 0.075, "FiLM: gamma * wafer_repr + beta\nper-point representation", "#D1FAE5", "#047857")
    _diagram_box(ax, (0.67, 0.105), 0.16, 0.065, "Regression\nhead", "#E5E7EB", "#374151")
    _diagram_arrow(ax, (0.74, 0.685), (0.74, 0.63))
    _diagram_arrow(ax, (0.74, 0.555), (0.70, 0.385), rad=0.08)
    _diagram_arrow(ax, (0.635, 0.425), (0.755, 0.46), color="#15803D")
    _diagram_arrow(ax, (0.845, 0.425), (0.74, 0.385), rad=-0.08, color="#15803D")
    _diagram_arrow(ax, (0.74, 0.315), (0.74, 0.280), color="#15803D")
    _diagram_arrow(ax, (0.75, 0.205), (0.75, 0.170), rad=-0.02)
    _mini_wafer(ax, 0.91, 0.315, 0.055, "varied", "position-specific\nrepresentation")

    # Bottom contrast strip
    _diagram_box(ax, (0.125, 0.015), 0.75, 0.055, "", "#FFFFFF", "#CBD5E1", fontsize=1)
    ax.text(
        0.5,
        0.043,
        "Key difference: concat asks the final head to infer spatial dependence; FiLM injects spatial dependence before prediction by modulating hidden features.",
        ha="center",
        va="center",
        fontsize=11.2,
        fontweight="bold",
        color="#111827",
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    _save(fig, "final_13_xy_concat_vs_film.png", written)


def fig_xy_vector_view(written: list[tuple[str, str]]) -> None:
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(15.5, 8.3))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.96,
        "XY conditioning: appending two coordinates vs modulating the whole representation",
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.918,
        "Both methods start from the same wafer representation, but XY affects the hidden vector at very different depths.",
        ha="center",
        va="top",
        fontsize=11.5,
        color=PALETTE["muted"],
    )

    def panel(x0: float, title: str, face: str, edge: str) -> None:
        patch = plt.matplotlib.patches.FancyBboxPatch(
            (x0, 0.09),
            0.43,
            0.76,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.8,
        )
        ax.add_patch(patch)
        ax.text(x0 + 0.215, 0.815, title, ha="center", va="center", fontsize=15, fontweight="bold")

    panel(0.045, "Initial: concatenate raw XY at the end", "#F8FAFC", "#CBD5E1")
    panel(0.525, "FiLM: project XY over the full vector", "#F0FDF4", "#86EFAC")

    def draw_vector(
        x: float,
        y: float,
        w: float,
        h: float,
        values: np.ndarray,
        cmap: str,
        label: str,
        edge: str = "#374151",
        label_y_offset: float = 0.032,
    ) -> None:
        n = len(values)
        gap = w * 0.002
        cell_w = (w - gap * (n - 1)) / n
        norm = (values - values.min()) / (values.max() - values.min() + 1e-9)
        cm = plt.get_cmap(cmap)
        for i, v in enumerate(norm):
            rect = plt.Rectangle(
                (x + i * (cell_w + gap), y),
                cell_w,
                h,
                facecolor=cm(0.12 + 0.78 * v),
                edgecolor="none",
            )
            ax.add_patch(rect)
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor=edge, linewidth=1.1)
        )
        ax.text(x + w / 2, y + h + label_y_offset, label, ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    base = np.sin(np.linspace(0, 3.5 * np.pi, 32)) + 0.25 * rng.normal(size=32)
    xy_small = np.array([0.2, 0.85])
    gamma = 0.75 + 0.55 * (np.sin(np.linspace(0, 4 * np.pi, 32) + 0.8) + 1) / 2
    beta = 0.35 * np.cos(np.linspace(0, 3 * np.pi, 32) - 0.5)
    film_vec = gamma * base + beta

    # Left panel: concat
    lx = 0.085
    draw_vector(lx, 0.60, 0.30, 0.07, base, "Purples", "wafer_repr: 256 hidden features")
    draw_vector(lx + 0.305, 0.60, 0.045, 0.07, xy_small, "Greens", "XY", label_y_offset=0.032)
    ax.text(lx + 0.175, 0.545, "[ wafer_repr_1 ... wafer_repr_256 | x | y ]", ha="center", fontsize=11.5, family="monospace")
    ax.annotate(
        "",
        xy=(lx + 0.175, 0.43),
        xytext=(lx + 0.175, 0.535),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.8, color="#374151", mutation_scale=15),
    )
    _diagram_box(
        ax,
        (lx + 0.045, 0.34),
        0.26,
        0.075,
        "Regression head must learn\nhow x,y interact with 256 features",
        "#E5E7EB",
        "#374151",
        fontsize=10.5,
    )
    ax.text(
        lx + 0.175,
        0.245,
        "XY has only two appended slots.\nThe original representation itself\nstays unchanged for every point.",
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["dark"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1"),
    )

    # Right panel: FiLM
    rx = 0.57
    draw_vector(rx, 0.68, 0.32, 0.055, base, "Purples", "wafer_repr")
    draw_vector(rx, 0.535, 0.32, 0.05, gamma, "Greens", "gamma(x,y): scale per feature")
    draw_vector(rx, 0.435, 0.32, 0.05, beta, "YlGn", "beta(x,y): shift per feature")
    ax.text(rx + 0.16, 0.61, "x,y -> Fourier encoder -> gamma, beta", ha="center", fontsize=10.5, color="#166534")
    ax.annotate(
        "",
        xy=(rx + 0.16, 0.405),
        xytext=(rx + 0.16, 0.515),
        arrowprops=dict(arrowstyle="-|>", linewidth=1.8, color="#15803D", mutation_scale=15),
    )
    draw_vector(rx, 0.31, 0.32, 0.065, film_vec, "viridis", "per-point representation = gamma * wafer_repr + beta")
    ax.text(
        rx + 0.16,
        0.235,
        "XY becomes a feature-wise transformation.\nEvery hidden dimension can be scaled or shifted\ndifferently for each wafer position.",
        ha="center",
        va="center",
        fontsize=11,
        color=PALETTE["dark"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#86EFAC"),
    )

    # Visual emphasis: 2 dims vs whole vector
    ax.text(0.255, 0.16, "2 extra dimensions", ha="center", fontsize=13, fontweight="bold", color="#D97706")
    ax.text(0.73, 0.16, "256-dimensional conditioning", ha="center", fontsize=13, fontweight="bold", color="#047857")
    ax.text(
        0.5,
        0.045,
        "Takeaway: concat gives XY to the final predictor; FiLM uses XY to reshape the representation before prediction.",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#CBD5E1"),
    )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    _save(fig, "final_14_xy_vector_concat_vs_film.png", written)


def _wafer_level_predictions(exp: Path) -> pd.DataFrame:
    pred = pd.read_csv(exp / "logs" / "sample_predictions.csv")
    pred = pred[pred["target"] == "oxide_etch"].copy()
    return (
        pred.groupby(["fold", "experiment_key"], as_index=False)
        .agg(
            y_true_mean=("y_true", "mean"),
            y_pred_mean=("y_pred", "mean"),
            mae=("abs_error", "mean"),
            true_point_std=("y_true", "std"),
            pred_point_std=("y_pred", "std"),
        )
    )


def fig_initial_kfold_collapse_diagnosis(written: list[tuple[str, str]]) -> None:
    initial_metrics = _read_fold_metrics(INITIAL_DL_EXP).sort_values("fold")
    initial_wafers = _wafer_level_predictions(INITIAL_DL_EXP)

    fig = plt.figure(figsize=(16.6, 8.8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.32], hspace=0.48, wspace=0.34)

    # A. Original k-fold collapse: best epoch + R2
    ax1 = fig.add_subplot(gs[0, 0])
    folds = initial_metrics["fold"].to_numpy()
    ax1.bar(folds, initial_metrics["best_epoch"], color="#9CC9E8", edgecolor="#315E7B", linewidth=0.7, label="Best epoch")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Best epoch")
    ax1.set_xticks(folds)
    ax1.set_title("Initial 5-fold: fold 2/4 failed early", loc="left", fontweight="bold")
    ax1b = ax1.twinx()
    ax1b.plot(folds, initial_metrics["r2"], marker="o", color="#0B6FB3", linewidth=2.4, label="Validation R2")
    ax1b.set_ylabel("Validation R2")
    ax1b.set_ylim(0.0, 0.86)
    for fold, ep, r2 in zip(folds, initial_metrics["best_epoch"], initial_metrics["r2"]):
        color = "#B91C1C" if fold in (2, 4) else "#0B6FB3"
        ax1b.text(fold, r2 + 0.035, f"{r2:.3f}", ha="center", fontsize=9.5, color=color, fontweight="bold")
        ax1.text(fold, ep + 1.0, f"{int(ep)}", ha="center", fontsize=9.5, color="#111827")
    ax1b.scatter([2, 4], initial_metrics.loc[initial_metrics["fold"].isin([2, 4]), "r2"], s=90, facecolor="white", edgecolor="#B91C1C", linewidth=2.0, zorder=4)

    # B. Wafer mean true vs predicted for problematic folds.
    ax2 = fig.add_subplot(gs[0, 1])
    colors = {2: "#F97316", 4: "#DC2626"}
    for fold in [2, 4]:
        df = initial_wafers[initial_wafers["fold"] == fold]
        ax2.scatter(
            df["y_true_mean"],
            df["y_pred_mean"],
            s=64,
            color=colors[fold],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.8,
            label=f"Initial fold {fold}",
        )
    lo = min(initial_wafers["y_true_mean"].min(), initial_wafers["y_pred_mean"].min()) - 0.01
    hi = max(initial_wafers["y_true_mean"].max(), initial_wafers["y_pred_mean"].max()) + 0.01
    ax2.plot([lo, hi], [lo, hi], color="#111827", linestyle="--", linewidth=1.2)
    ax2.set_xlim(lo, hi)
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("True wafer-mean oxide_etch")
    ax2.set_ylabel("Predicted wafer-mean oxide_etch")
    ax2.set_title("Different failure modes: fold 2 outlier vs fold 4 collapse", loc="left", fontweight="bold")
    ax2.legend(loc="upper left")
    outlier = initial_wafers[initial_wafers["experiment_key"] == "2024-08-22_04"]
    if not outlier.empty:
        row = outlier.iloc[0]
        ax2.annotate(
            "fold 2: unique wafer\n2024-08-22_04",
            xy=(row["y_true_mean"], row["y_pred_mean"]),
            xytext=(row["y_true_mean"] - 0.075, row["y_pred_mean"] + 0.025),
            arrowprops=dict(arrowstyle="-|>", color="#7C2D12", lw=1.4),
            fontsize=9.5,
            color="#7C2D12",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#FDBA74", alpha=0.95),
        )
    f4 = initial_wafers[initial_wafers["fold"] == 4]
    ax2.annotate(
        "fold 4: wafer means\ncompressed near one value",
        xy=(f4["y_true_mean"].mean(), f4["y_pred_mean"].mean()),
        xytext=(0.67, 0.585),
        arrowprops=dict(arrowstyle="-|>", color="#991B1B", lw=1.4),
        fontsize=9.5,
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#FCA5A5", alpha=0.95),
    )

    # C. Spread ratio: predicted wafer separation vs true wafer separation.
    ax3 = fig.add_subplot(gs[1, 0])
    rows = []
    for fold in range(5):
        df = initial_wafers[initial_wafers["fold"] == fold]
        rows.append(
            {
                "fold": fold,
                "true_spread": df["y_true_mean"].std(ddof=0),
                "pred_spread": df["y_pred_mean"].std(ddof=0),
                "spread_ratio": df["y_pred_mean"].std(ddof=0) / (df["y_true_mean"].std(ddof=0) + 1e-9),
            }
        )
    spread = pd.DataFrame(rows)
    x = np.arange(5)
    bars = ax3.bar(x, spread["spread_ratio"], color="#9CA3AF", edgecolor="#374151", linewidth=0.8)
    bars[2].set_color("#F97316")
    bars[4].set_color("#DC2626")
    ax3.axhline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"Fold {i}" for i in range(5)])
    ax3.set_ylabel("Predicted wafer-mean spread / true spread")
    ax3.set_title("Fold 4: predicted means are over-compressed", loc="left", fontweight="bold")
    for row in spread.itertuples(index=False):
        ax3.text(row.fold, row.spread_ratio + 0.035, f"{row.spread_ratio:.2f}", ha="center", fontsize=9.5, fontweight="bold")
    ax3.text(
        0.44,
        0.17,
        "A low ratio means the model predicts\nnearly the same wafer-level value even when\ntrue wafer means are spread out.",
        transform=ax3.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
    )

    # D. Worst wafer-level errors in the failed folds.
    ax4 = fig.add_subplot(gs[1, 1])
    failed = initial_wafers[initial_wafers["fold"].isin([2, 4])].copy()
    failed["label"] = failed["experiment_key"] + "  (fold " + failed["fold"].astype(str) + ")"
    worst = failed.sort_values("mae", ascending=False).head(10).iloc[::-1]
    colors = ["#F97316" if f == 2 else "#DC2626" for f in worst["fold"]]
    ax4.barh(np.arange(len(worst)), worst["mae"], color=colors, edgecolor="#374151", linewidth=0.5)
    ax4.set_yticks(np.arange(len(worst)))
    ax4.set_yticklabels(worst["label"], fontsize=8.8)
    ax4.set_xlabel("Wafer-level MAE")
    ax4.set_title("Largest wafer-level errors in failed folds", loc="left", fontweight="bold")
    for i, value in enumerate(worst["mae"]):
        ax4.text(value + 0.002, i, f"{value:.3f}", va="center", fontsize=8.8)
    ax4.text(
        0.45,
        0.08,
        "Fold 2 is dominated by a distinctive wafer\n(2024-08-22_04), while fold 4 shows many\nlow/high wafers pulled toward the middle.",
        transform=ax4.transAxes,
        fontsize=9.4,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
    )

    fig.suptitle(
        "Initial k-fold failure diagnosis: folds 2 and 4 failed for different reasons",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "final_15_initial_kfold_failure_diagnosis.png", written)


def write_index(written: list[tuple[str, str]]) -> None:
    rows = [_aggregate_row(exp) for exp in EVOLUTION]
    summary = pd.DataFrame(rows)
    summary_path = OUT / "final_model_summary.csv"
    summary.to_csv(summary_path, index=False)
    written.append(("final_model_summary.csv", str(summary_path.relative_to(PROJECT_ROOT))))

    index_path = OUT / "final_presentation_figure_index.md"
    lines = [
        "# Final Presentation Figure Index",
        "",
        "Generated by `python -m scripts.11_make_final_presentation_figures`.",
        "",
        "## Figures",
        "",
    ]
    for name, rel in written:
        lines.append(f"- `{name}`: `{rel}`")
    lines.extend(
        [
            "",
            "## Key final numbers",
            "",
            "- XGBoost oxide R2: 0.551 +- 0.083, RMSE: 0.0514 +- 0.0043",
            "- Final DL oxide R2: 0.666 +- 0.110, RMSE: 0.0440 +- 0.0073",
            "- Relative RMSE reduction: 14.4%",
        ]
    )
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved {index_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, str]] = []
    fig_model_evolution(written)
    fig_fold_comparison(written)
    fig_metric_improvement(written)
    fig_pred_scatter(written)
    fig_residuals(written)
    fig_epoch_curves(written)
    fig_ablation_heatmap(written)
    fig_wafer_error_map(written)
    fig_aux_mixup_ema_explainer(written)
    fig_pipeline_timeline(written)
    fig_modality_ablation(written)
    fig_singlefold_to_kfold(written)
    fig_xy_film_comparison(written)
    fig_xy_vector_view(written)
    fig_initial_kfold_collapse_diagnosis(written)
    write_index(written)
    print("\nFinal presentation figure package complete.")


if __name__ == "__main__":
    main()
