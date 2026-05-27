"""Lot-level validation analysis for DL model predictions.

Reads sample_predictions.csv from a 5-fold experiment, groups predictions
by lot (day), and produces per-lot metrics + visualizations.

Run from project root:
    python -m scripts.09_lot_validation --exp-dir outputs/experiments/2026-05-21_02-09_dl-multimodal-5fold
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import DAY_TO_LOT
from src.evaluation import regression_metrics

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
})


def experiment_key_to_lot(key: str) -> int:
    day = key.rsplit("_", 1)[0].replace("-", "_")
    return DAY_TO_LOT[day]


def experiment_key_to_day(key: str) -> str:
    return key.rsplit("_", 1)[0]


def compute_lot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, lot), g in df.groupby(["target", "lot"]):
        m = regression_metrics(g["y_true"].values, g["y_pred"].values)
        day = g["day"].iloc[0]
        rows.append({
            "target": target,
            "lot": lot,
            "day": day,
            "n_wafers": g["experiment_key"].nunique(),
            "n_points": len(g),
            "y_true_mean": g["y_true"].mean(),
            "y_true_std": g["y_true"].std(),
            **m,
        })
    return pd.DataFrame(rows).sort_values(["target", "lot"]).reset_index(drop=True)


def compute_wafer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, lot, wk), g in df.groupby(["target", "lot", "experiment_key"]):
        m = regression_metrics(g["y_true"].values, g["y_pred"].values)
        wafer_num = int(wk.rsplit("_", 1)[1])
        rows.append({
            "target": target,
            "lot": lot,
            "experiment_key": wk,
            "wafer_num": wafer_num,
            "n_points": len(g),
            **m,
        })
    return pd.DataFrame(rows).sort_values(["target", "lot", "wafer_num"]).reset_index(drop=True)


# -- Plotting -----------------------------------------------------------------

def plot_lot_bar(lot_df: pd.DataFrame, target: str, out_dir: Path) -> None:
    sub = lot_df[lot_df["target"] == target].sort_values("lot")
    lots = sub["lot"].values
    labels = [f"Lot {l}\n({d})" for l, d in zip(sub["lot"], sub["day"])]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"Lot-level Metrics — {target}", fontsize=13, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(lots)))

    ax = axes[0]
    ax.bar(range(len(lots)), sub["rmse"].values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(sub["rmse"].mean(), color="red", ls="--", lw=1, label=f"mean={sub['rmse'].mean():.4f}")
    ax.set_xticks(range(len(lots)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE by Lot")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(range(len(lots)), sub["r2"].values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(sub["r2"].mean(), color="red", ls="--", lw=1, label=f"mean={sub['r2'].mean():.4f}")
    ax.set_xticks(range(len(lots)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("R²")
    ax.set_title("R² by Lot")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(range(len(lots)), sub["mape_pct"].values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(sub["mape_pct"].mean(), color="red", ls="--", lw=1,
               label=f"mean={sub['mape_pct'].mean():.2f}%")
    ax.set_xticks(range(len(lots)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("MAPE (%)")
    ax.set_title("MAPE by Lot")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"lot_bar_{target}.png")
    plt.close(fig)


def plot_lot_scatter(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    sub = df[df["target"] == target]
    lots = sorted(sub["lot"].unique())
    n_lots = len(lots)
    ncols = 5
    nrows = (n_lots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    fig.suptitle(f"Predicted vs Actual by Lot — {target}", fontsize=13, fontweight="bold", y=1.01)
    axes_flat = np.atleast_2d(axes).flatten()

    for i, lot in enumerate(lots):
        ax = axes_flat[i]
        g = sub[sub["lot"] == lot]
        m = regression_metrics(g["y_true"].values, g["y_pred"].values)
        ax.scatter(g["y_true"], g["y_pred"], s=6, alpha=0.4, edgecolors="none")
        vmin = min(g["y_true"].min(), g["y_pred"].min())
        vmax = max(g["y_true"].max(), g["y_pred"].max())
        margin = (vmax - vmin) * 0.05
        ax.plot([vmin - margin, vmax + margin], [vmin - margin, vmax + margin],
                "r--", lw=0.8, alpha=0.7)
        ax.set_xlim(vmin - margin, vmax + margin)
        ax.set_ylim(vmin - margin, vmax + margin)
        day = g["day"].iloc[0]
        ax.set_title(f"Lot {lot} ({day})\nR²={m['r2']:.3f}  RMSE={m['rmse']:.4f}", fontsize=8)
        ax.set_xlabel("Actual", fontsize=7)
        ax.set_ylabel("Predicted", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / f"lot_scatter_{target}.png")
    plt.close(fig)


def plot_lot_residual_boxplot(df: pd.DataFrame, target: str, out_dir: Path) -> None:
    sub = df[df["target"] == target]
    lots = sorted(sub["lot"].unique())

    fig, ax = plt.subplots(figsize=(12, 5))
    data = [sub[sub["lot"] == lot]["residual"].values for lot in lots]
    bp = ax.boxplot(data, labels=[f"Lot {l}" for l in lots], patch_artist=True, showfliers=True,
                    flierprops={"markersize": 2, "alpha": 0.3})

    colors = plt.cm.tab10(np.linspace(0, 1, len(lots)))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.axhline(0, color="red", ls="--", lw=0.8)
    ax.set_ylabel("Residual (pred − actual)")
    ax.set_title(f"Residual Distribution by Lot — {target}", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    fig.savefig(out_dir / f"lot_residual_boxplot_{target}.png")
    plt.close(fig)


def plot_lot_drift(lot_df: pd.DataFrame, target: str, out_dir: Path) -> None:
    sub = lot_df[lot_df["target"] == target].sort_values("lot")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    x = range(len(sub))
    labels = [f"Lot {l}\n({d})" for l, d in zip(sub["lot"], sub["day"])]

    ln1 = ax1.plot(x, sub["rmse"], "o-", color="tab:blue", label="RMSE", markersize=6)
    ln2 = ax2.plot(x, sub["y_true_mean"], "s--", color="tab:orange", label="Target mean", markersize=6)
    ax2.fill_between(x,
                     sub["y_true_mean"] - sub["y_true_std"],
                     sub["y_true_mean"] + sub["y_true_std"],
                     alpha=0.15, color="tab:orange")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("RMSE", color="tab:blue")
    ax2.set_ylabel("Target mean ± std", color="tab:orange")
    ax1.set_title(f"Lot-level Drift Analysis — {target}", fontweight="bold")

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, fontsize=9, loc="upper left")

    plt.tight_layout()
    fig.savefig(out_dir / f"lot_drift_{target}.png")
    plt.close(fig)


def plot_wafer_heatmap(wafer_df: pd.DataFrame, target: str, metric: str,
                       out_dir: Path) -> None:
    sub = wafer_df[wafer_df["target"] == target].copy()
    lots = sorted(sub["lot"].unique())
    max_wafers = sub.groupby("lot")["wafer_num"].nunique().max()

    grid = np.full((len(lots), max_wafers), np.nan)
    for _, row in sub.iterrows():
        li = lots.index(row["lot"])
        wi = row["wafer_num"] - 1
        if wi < max_wafers:
            grid[li, wi] = row[metric]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r" if metric == "rmse" else "RdYlGn")
    ax.set_xticks(range(max_wafers))
    ax.set_xticklabels([f"W{i+1}" for i in range(max_wafers)], fontsize=8)
    ax.set_yticks(range(len(lots)))
    ax.set_yticklabels([f"Lot {l}" for l in lots], fontsize=9)
    ax.set_xlabel("Wafer")
    ax.set_title(f"Per-Wafer {metric.upper()} — {target}", fontweight="bold")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                fmt = f"{v:.4f}" if metric == "rmse" else f"{v:.3f}"
                ax.text(j, i, fmt, ha="center", va="center", fontsize=6,
                        color="white" if abs(v - np.nanmean(grid)) > np.nanstd(grid) else "black")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    fig.savefig(out_dir / f"lot_wafer_heatmap_{metric}_{target}.png")
    plt.close(fig)


# -- Main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, type=Path,
                        help="Path to experiment directory with logs/sample_predictions.csv")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = PROJECT_ROOT / exp_dir

    pred_path = exp_dir / "logs" / "sample_predictions.csv"
    if not pred_path.exists():
        print(f"ERROR: {pred_path} not found")
        sys.exit(1)

    df = pd.read_csv(pred_path)
    print(f"Loaded {len(df)} predictions from {pred_path.relative_to(PROJECT_ROOT)}")

    df["day"] = df["experiment_key"].apply(experiment_key_to_day)
    df["lot"] = df["experiment_key"].apply(experiment_key_to_lot)

    lot_df = compute_lot_metrics(df)
    wafer_df = compute_wafer_metrics(df)

    out_dir = exp_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    lot_df.to_csv(exp_dir / "logs" / "lot_metrics.csv", index=False)
    wafer_df.to_csv(exp_dir / "logs" / "wafer_metrics.csv", index=False)
    print(f"Saved lot_metrics.csv ({len(lot_df)} rows)")
    print(f"Saved wafer_metrics.csv ({len(wafer_df)} rows)")

    targets = sorted(df["target"].unique())
    for target in targets:
        print(f"\n=== {target} ===")
        sub = lot_df[lot_df["target"] == target].sort_values("lot")
        for _, row in sub.iterrows():
            print(f"  Lot {int(row['lot']):2d} ({row['day']}): "
                  f"{int(row['n_wafers']):2d} wafers, {int(row['n_points']):4d} pts | "
                  f"RMSE={row['rmse']:.4f}  R²={row['r2']:.4f}  MAPE={row['mape_pct']:.2f}%")

        overall = regression_metrics(
            df.loc[df["target"] == target, "y_true"].values,
            df.loc[df["target"] == target, "y_pred"].values,
        )
        print(f"  {'Overall':>20s}: RMSE={overall['rmse']:.4f}  "
              f"R²={overall['r2']:.4f}  MAPE={overall['mape_pct']:.2f}%")
        rmse_vals = sub["rmse"].values
        print(f"  {'Lot RMSE spread':>20s}: "
              f"best={rmse_vals.min():.4f} (Lot {int(sub.iloc[rmse_vals.argmin()]['lot'])})  "
              f"worst={rmse_vals.max():.4f} (Lot {int(sub.iloc[rmse_vals.argmax()]['lot'])})  "
              f"ratio={rmse_vals.max()/rmse_vals.min():.2f}x")

        plot_lot_bar(lot_df, target, out_dir)
        plot_lot_scatter(df, target, out_dir)
        plot_lot_residual_boxplot(df, target, out_dir)
        plot_lot_drift(lot_df, target, out_dir)
        plot_wafer_heatmap(wafer_df, target, "rmse", out_dir)
        plot_wafer_heatmap(wafer_df, target, "r2", out_dir)
        print(f"  Saved 6 figures for {target}")

    summary = {}
    for target in targets:
        sub = lot_df[lot_df["target"] == target]
        summary[target] = {
            "per_lot": sub.drop(columns=["target"]).to_dict(orient="records"),
            "lot_rmse_mean": float(sub["rmse"].mean()),
            "lot_rmse_std": float(sub["rmse"].std(ddof=0)),
            "lot_r2_mean": float(sub["r2"].mean()),
            "lot_r2_std": float(sub["r2"].std(ddof=0)),
            "lot_mape_mean": float(sub["mape_pct"].mean()),
            "lot_mape_std": float(sub["mape_pct"].std(ddof=0)),
            "worst_lot": int(sub.iloc[sub["rmse"].values.argmax()]["lot"]),
            "best_lot": int(sub.iloc[sub["rmse"].values.argmin()]["lot"]),
        }

    out_json = exp_dir / "lot_validation.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_json.relative_to(PROJECT_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
