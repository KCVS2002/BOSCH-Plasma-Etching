"""Generate the four presentation-quality figures for the mid-term slide deck.

Outputs (under outputs/figures/):
  pres_01_dataset_overview.png      # tensor summary + OES spectrum + process trace
  pres_02_oes_cycle_evolution.png   # cycle x wavelength heatmap (justifies cycle-aware DL)
  pres_03_target_overview.png       # 89-point wafer map + target distributions
  pres_04_pred_vs_true.png          # XGB & DL pred-vs-true scatter (si_etch, oxide_etch)

Run:
    .venv\\python.exe -m scripts.07_make_presentation_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib.gridspec import GridSpec

CACHE = PROJECT_ROOT / "cache" / "v1"
OUT = PROJECT_ROOT / "outputs" / "figures"
REP_WAFER = "2024-07-02_01"  # representative wafer for Figs 1-2
XGB_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-04-30_15-32_baseline-xgb"
DL_EXP = PROJECT_ROOT / "outputs" / "experiments" / "2026-05-01_00-56_dl-multimodal-singlefold"

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Fig 1 — Dataset overview
# ---------------------------------------------------------------------------
def fig_dataset_overview() -> None:
    print("[fig1] dataset overview…")
    d = np.load(CACHE / "wafers" / f"{REP_WAFER}.npz")
    oes = d["oes_data"]                       # (T_o, 3648)
    proc = d["process_data"]                  # (T_p, 44)
    proc_t = d["process_t_rel"]
    proc_feats = d["process_features"]
    wl = d["oes_wavelengths"]
    o_starts, o_ends = d["oes_cycle_starts_idx"], d["oes_cycle_ends_idx"]
    p_starts = d["proc_cycle_starts_idx"]

    # Count wafers from manifest
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    n_wafers = manifest["n_built"]
    meas = pd.read_csv(CACHE / "measurements.csv")
    n_points = meas[["X", "Y"]].drop_duplicates().shape[0]

    fig = plt.figure(figsize=(15, 5.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.2, 1.5], wspace=0.42)

    # ---- Panel A: dataset shape "card" -----------------------------------
    axA = fig.add_subplot(gs[0, 0])
    axA.set_axis_off()
    axA.set_title("Dataset shape", loc="left", fontweight="bold", pad=10)

    rows = [
        ("Wafers", f"{n_wafers}"),
        ("Cycles / wafer", "100"),
        ("OES wavelengths", "3,648  (186–884 nm)"),
        ("Process channels", "44 raw → 31 used"),
        ("Points / wafer", f"{n_points}"),
        ("Total samples", f"{len(meas):,}"),
        ("Targets", "si_etch, oxide_etch (μm)"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.92 - i * 0.115
        axA.text(0.02, y, k, transform=axA.transAxes, fontsize=10.5, color="#555")
        axA.text(0.66, y, v, transform=axA.transAxes, fontsize=10.5, fontweight="bold",
                 ha="left")

    # subtle tensor sketch beneath
    axA.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.08, boxstyle="round,pad=0.01",
        transform=axA.transAxes, facecolor="#eef3fa", edgecolor="#94a8c9", linewidth=0.8))
    axA.text(0.5, 0.06,
             "shape  ≈  88 × 100 × (3,648 λ + 31 ch)  ⊕  XY",
             transform=axA.transAxes, ha="center", va="center",
             fontsize=9.5, family="monospace", color="#23375f")

    # ---- Panel B: OES spectrum at one representative cycle ---------------
    axB = fig.add_subplot(gs[0, 1])
    cyc = 50
    s, e = int(o_starts[cyc]), int(o_ends[cyc])
    spec = oes[s:e].astype(np.float32).mean(axis=0)
    axB.plot(wl, spec, color="#2864c8", linewidth=0.7)
    axB.set_xlabel("Wavelength (nm)")
    axB.set_ylabel("Intensity (a.u.)")
    axB.set_title(f"OES spectrum (cycle {cyc})", loc="left", fontweight="bold")
    axB.set_xlim(wl.min(), wl.max())
    axB.grid(alpha=0.25)

    # annotate a few notable peaks (top 4)
    top_idx = np.argsort(spec)[-4:]
    for ix in top_idx:
        axB.annotate(f"{wl[ix]:.0f} nm", xy=(wl[ix], spec[ix]),
                     xytext=(5, 4), textcoords="offset points",
                     fontsize=8, color="#444")

    # ---- Panel C: Process traces with cycle boundaries -------------------
    axC = fig.add_subplot(gs[0, 2])
    chans = {
        "Pressure": "Stat3_Etch_MV_Pressure",
        "Gas2Flow (SF6)": "Stat3_Etch_MV_Gas2Flow",
        "SourceRF Power": "Stat3_Etch_MV_SourceRFLoadPower",
    }
    name_to_idx = {n: i for i, n in enumerate(proc_feats)}
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    # Use first 60 s for clarity (full wafer is too dense)
    tmask = proc_t < 60.0
    for (label, full), c in zip(chans.items(), colors):
        idx = name_to_idx[full]
        y = proc[:, idx]
        if np.isnan(y).all():
            continue
        # min-max scale to [0,1] for joint display
        valid = ~np.isnan(y)
        if valid.sum() < 2:
            continue
        ynorm = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-9)
        axC.plot(proc_t[tmask], ynorm[tmask], label=label, color=c, linewidth=0.9)

    # cycle-start markers in the same window
    cyc_t = d["proc_t_rel"][p_starts.astype(int)] if "proc_t_rel" in d.files else None
    if cyc_t is None:
        cyc_t = proc_t[p_starts.astype(int)]
    n_show = (cyc_t < 60.0).sum()
    for t in cyc_t[:n_show]:
        axC.axvline(t, color="#888", linestyle="--", linewidth=0.5, alpha=0.6)

    axC.set_xlabel("Time (s)")
    axC.set_ylabel("Normalized value")
    axC.set_title(f"Process traces (first 60 s, {n_show} cycles)",
                  loc="left", fontweight="bold")
    axC.set_xlim(0, 60)
    axC.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    axC.grid(alpha=0.25)

    fig.suptitle("Dataset overview — BOSCH plasma etching virtual metrology",
                 fontsize=13.5, fontweight="bold", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "pres_01_dataset_overview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Fig 2 — OES cycle x wavelength evolution
# ---------------------------------------------------------------------------
def fig_oes_cycle_evolution() -> None:
    print("[fig2] OES cycle×wavelength evolution…")
    d = np.load(CACHE / "wafers" / f"{REP_WAFER}.npz")
    oes = d["oes_data"]
    wl = d["oes_wavelengths"]
    o_starts, o_ends = d["oes_cycle_starts_idx"], d["oes_cycle_ends_idx"]
    proc = d["process_data"]
    proc_feats = d["process_features"]
    p_starts, p_ends = d["proc_cycle_starts_idx"], d["proc_cycle_ends_idx"]

    n_cycles = len(o_starts)
    n_w = oes.shape[1]
    cyc_mean = np.zeros((n_cycles, n_w), dtype=np.float32)
    for c in range(n_cycles):
        s, e = int(o_starts[c]), int(o_ends[c])
        cyc_mean[c] = oes[s:e].astype(np.float32).mean(axis=0)

    # Downsample wavelength axis to ~600 bins for sharper visual rendering
    n_bins = 600
    edges = np.linspace(0, n_w, n_bins + 1, dtype=int)
    cyc_binned = np.stack(
        [cyc_mean[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(n_bins)],
        axis=1,
    )
    wl_binned = np.array([wl[edges[i]:edges[i + 1]].mean() for i in range(n_bins)])

    # log scale to bring out emission lines
    img = np.log10(np.clip(cyc_binned, 1.0, None))

    # process cycle-mean for the right panel (use 31 channels excluding all-NaN)
    n_proc = proc.shape[1]
    proc_cyc = np.zeros((n_cycles, n_proc), dtype=np.float32)
    for c in range(n_cycles):
        s, e = int(p_starts[c]), int(p_ends[c])
        sl = proc[s:e]
        with np.errstate(invalid="ignore"):
            proc_cyc[c] = np.where(np.isnan(sl).all(axis=0), 0.0, np.nanmean(sl, axis=0))
    keep = ~np.isnan(proc).all(axis=0)  # 31 channels
    proc_cyc_keep = proc_cyc[:, keep]

    def _short(n: str) -> str:
        n = n.replace("Stat3_Etch_MV_", "")
        n = n.replace("PlatenRF", "PltRF").replace("SourceRF2", "SrcRF2").replace("SourceRF", "SrcRF")
        n = n.replace("ThermoCouple", "TC").replace("Heater", "Htr")
        n = n.replace("LoadCapacitor", "LdCap").replace("LoadPower", "LdPwr")
        n = n.replace("PeakToPeak", "PkPk").replace("ReflectedPower", "RefPwr")
        n = n.replace("TuningCapacitor", "TnCap")
        n = n.replace("ForeLinePressure", "ForePress").replace("Pressure", "Press")
        n = n.replace("HeliumBPFlow", "HeBPFlow").replace("HeliumBPPressure", "HeBPPress")
        n = n.replace("attenuatorRatio", "attRatio")
        n = n.replace("EpdIntensity", "EPDInt")
        return n
    short_names = [_short(str(n)) for n in proc_feats[keep]]
    # per-channel z-score so all dynamics visible
    p_mu = proc_cyc_keep.mean(axis=0, keepdims=True)
    p_sd = proc_cyc_keep.std(axis=0, keepdims=True) + 1e-9
    proc_z = (proc_cyc_keep - p_mu) / p_sd

    fig = plt.figure(figsize=(15.5, 6.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.05], wspace=0.45)

    # ---- left: OES heatmap ----------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        img, aspect="auto", origin="lower",
        extent=[wl_binned.min(), wl_binned.max(), 0, n_cycles],
        cmap="magma",
    )
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Cycle index")
    ax1.set_title(
        f"OES intensity per cycle — wafer {REP_WAFER}\n"
        "log10 mean intensity (cycle × λ)",
        loc="left", fontweight="bold")
    cb1 = fig.colorbar(im1, ax=ax1, fraction=0.045, pad=0.02)
    cb1.set_label("log10 intensity")

    # ---- right: process channels heatmap (z-score) ----------------------
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        proc_z.T, aspect="auto", origin="lower", cmap="RdBu_r",
        vmin=-3, vmax=3,
        extent=[0, n_cycles, 0, proc_z.shape[1]],
    )
    ax2.set_xlabel("Cycle index")
    ax2.set_ylabel("Process channel")
    ax2.set_yticks(np.arange(len(short_names)) + 0.5)
    ax2.set_yticklabels(short_names, fontsize=7.2)
    ax2.set_title("Process channel drift across cycles\n(per-channel z-score)",
                  loc="left", fontweight="bold")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.045, pad=0.02)
    cb2.set_label("z-score")

    fig.suptitle(
        "Cycle-level structure across modalities — "
        "motivation for cycle-aware temporal modeling",
        fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "pres_02_oes_cycle_evolution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Fig 3 — 89-point wafer map + target distributions
# ---------------------------------------------------------------------------
def fig_target_overview() -> None:
    print("[fig3] 89-point map + target distributions…")
    meas = pd.read_csv(CACHE / "measurements.csv")
    pos = meas.groupby(["X", "Y"], as_index=False).agg(
        si_mean=("si_etch", "mean"),
        ox_mean=("oxide_etch", "mean"),
        n=("si_etch", "size"),
    )

    fig = plt.figure(figsize=(16, 5.0))
    gs = GridSpec(
        1, 4, figure=fig, width_ratios=[1.0, 1.0, 0.95, 0.95], wspace=0.7,
    )

    # ---- wafer perimeter for context ------------------------------------
    rmax = np.sqrt((meas["X"] ** 2 + meas["Y"] ** 2)).max() * 1.05
    theta = np.linspace(0, 2 * np.pi, 200)

    def _wafer_panel(ax, c, label, cmap):
        ax.plot(rmax * np.cos(theta), rmax * np.sin(theta),
                color="#888", linewidth=0.8)
        sc = ax.scatter(pos["X"], pos["Y"], c=c, s=34, cmap=cmap,
                        edgecolor="black", linewidth=0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_title(label, loc="left", fontweight="bold")
        ax.grid(alpha=0.2)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("μm")

    ax1 = fig.add_subplot(gs[0, 0])
    _wafer_panel(ax1, pos["si_mean"], "si_etch (wafer-mean)", "viridis")

    ax2 = fig.add_subplot(gs[0, 1])
    _wafer_panel(ax2, pos["ox_mean"], "oxide_etch (wafer-mean)", "plasma")

    # ---- target distributions (separate panels) -------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(meas["si_etch"], bins=60, color="#2864c8", alpha=0.85)
    ax3.set_xlabel("si_etch (μm)")
    ax3.set_ylabel("Count")
    ax3.set_title("si_etch distribution", loc="left", fontweight="bold")
    ax3.text(0.04, 0.94,
             f"n = {len(meas):,}\n"
             f"μ = {meas['si_etch'].mean():.2f} μm\n"
             f"σ = {meas['si_etch'].std():.2f} μm",
             transform=ax3.transAxes, va="top", fontsize=9, family="monospace",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb",
                       boxstyle="round,pad=0.3"))
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(meas["oxide_etch"], bins=60, color="#d62728", alpha=0.85)
    ax4.set_xlabel("oxide_etch (μm)")
    ax4.set_ylabel("Count")
    ax4.set_title("oxide_etch distribution", loc="left", fontweight="bold")
    ax4.text(0.04, 0.94,
             f"n = {len(meas):,}\n"
             f"μ = {meas['oxide_etch'].mean():.3f} μm\n"
             f"σ = {meas['oxide_etch'].std():.3f} μm",
             transform=ax4.transAxes, va="top", fontsize=9, family="monospace",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb",
                       boxstyle="round,pad=0.3"))
    ax4.grid(alpha=0.25)

    fig.suptitle(
        f"Targets across the wafer — {len(pos)} measurement positions × "
        f"{n_wafers if (n_wafers := meas['experiment_key'].nunique()) else 88} wafers",
        fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT / "pres_03_target_overview.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Fig 4 — Pred vs True (XGB all-folds + DL fold0)
# ---------------------------------------------------------------------------
def _xgb_predictions_all_folds() -> dict[str, pd.DataFrame]:
    """Re-run inference per fold with saved XGB models. Returns {target: df}."""
    from src.evaluation import load_split
    from src.features import load_or_build_features

    feat_df = load_or_build_features(CACHE, feature_set="baseline_xgb_v1", n_oes_bands=10)
    meas = pd.read_csv(CACHE / "measurements.csv")
    df = meas.merge(feat_df, on="experiment_key", how="left", validate="many_to_one")
    feat_cols = [c for c in feat_df.columns if c != "experiment_key"]
    feature_names = feat_cols + ["X", "Y"]
    X = df[feature_names].to_numpy(dtype=np.float32)

    split = load_split(CACHE / "splits" / "kfold5_wafer.npz")
    targets = ["si_etch", "oxide_etch"]
    out: dict[str, pd.DataFrame] = {}

    for tgt in targets:
        y = df[tgt].to_numpy(dtype=np.float32)
        rows = []
        for f in range(split.n_folds):
            _, val_mask = split.train_val_masks(f)
            booster = xgb.XGBRegressor()
            booster.load_model(str(XGB_EXP / "checkpoints" / f"{tgt}_fold{f}.json"))
            y_pred = booster.predict(X[val_mask])
            rows.append(pd.DataFrame({
                "fold": f,
                "y_true": y[val_mask],
                "y_pred": y_pred,
            }))
        out[tgt] = pd.concat(rows, ignore_index=True)
    return out


def fig_pred_vs_true() -> None:
    print("[fig4] pred vs true (XGB + DL)…")
    xgb_preds = _xgb_predictions_all_folds()
    dl = pd.read_csv(DL_EXP / "logs" / "sample_predictions.csv")
    dl_preds = {t: dl[dl["target"] == t][["y_true", "y_pred"]].reset_index(drop=True)
                for t in ["si_etch", "oxide_etch"]}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.2))
    targets = ["si_etch", "oxide_etch"]
    sources = [("XGBoost (all 5 folds)", xgb_preds, "#1f77b4"),
               ("DL multimodal (fold 0)", dl_preds, "#d62728")]

    for r, tgt in enumerate(targets):
        for c, (src_label, src, color) in enumerate(sources):
            ax = axes[r, c]
            d = src[tgt]
            yt, yp = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
            err = yp - yt
            rmse = float(np.sqrt(np.mean(err ** 2)))
            r2 = 1 - np.var(err) / np.var(yt)
            ax.scatter(yt, yp, s=8, alpha=0.35, color=color, edgecolor="none")
            lim_lo = min(yt.min(), yp.min())
            lim_hi = max(yt.max(), yp.max())
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
                    color="#444", linewidth=1.0, linestyle="--", label="y = x")
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.set_aspect("equal")
            ax.set_xlabel(f"True {tgt} (μm)")
            ax.set_ylabel(f"Pred {tgt} (μm)")
            ax.set_title(f"{tgt}  ·  {src_label}", loc="left", fontweight="bold")
            ax.text(0.04, 0.96,
                    f"n = {len(yt):,}\nRMSE = {rmse:.4f}\nR² = {r2:.4f}",
                    transform=ax.transAxes, va="top", ha="left",
                    fontsize=9.5, family="monospace",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="#bbb",
                              boxstyle="round,pad=0.3"))
            ax.grid(alpha=0.25)
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Prediction vs True — XGBoost baseline vs Cycle-Aware DL multimodal",
        fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / "pres_04_pred_vs_true.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_dataset_overview()
    fig_oes_cycle_evolution()
    fig_target_overview()
    fig_pred_vs_true()
    print("\nAll four figures written to outputs/figures/")


if __name__ == "__main__":
    main()
