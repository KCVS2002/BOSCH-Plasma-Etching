"""Concept-explainer slide figures for the 4 fold-collapse remedies.

Style matches the "XY Conditioning" FiLM slide: white canvas, navy bold title,
rounded light panel, colored heatmap strips, minimal keyword labels, one bold
colored takeaway at the bottom. English labels, keywords only (the talk carries
the detail).

One PNG per technique:
  slide_oes_topk.png       — per-fold correlation wavelength selection (top-256)
  slide_aux_wafer_mean.png — wafer-mean auxiliary loss head
  slide_mixup.png          — wafer-level mixup fills the bimodal gap
  slide_ema.png            — Polyak weight EMA smooths the trajectory

Run:
    .venv\\Scripts\\python.exe notebooks/scratch/make_technique_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures"
EXP = ROOT / "outputs" / "experiments"
# real runs for the EMA figure: mixup-only (raw weights) vs mixup+EMA 120ep (shadow)
EMA_NOEMA_LOG = EXP / "2026-05-28_03-16_dl-multimodal-oes-aux-mixup-5fold" / "logs" / "epoch_log.csv"
EMA_EMA_LOG = EXP / "2026-05-28_04-19_dl-multimodal-oes-aux-mixup-ema-longrun-5fold" / "logs" / "epoch_log.csv"

# ---- palette (lifted from the FiLM slide) ----
NAVY = "#1b2a4a"
INK = "#2b3a52"
BLUE_TXT = "#2a5db0"
GREEN_TXT = "#2e8b57"
PANEL_BLUE_F = "#eef1f8"
PANEL_BLUE_E = "#b9c2dd"
PANEL_GREEN_F = "#e9f5ec"
PANEL_GREEN_E = "#a6d6b6"
GRAY_BOX_F = "#e8eaee"
GRAY_BOX_E = "#9aa3b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 200,
    "figure.dpi": 120,
})


# ---------------------------------------------------------------- helpers
def panel(fig, x, y, w, h, fill, edge, lw=2.0, pad=0.012, round_=0.025):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad},rounding_size={round_}",
        transform=fig.transFigure, facecolor=fill, edgecolor=edge,
        linewidth=lw, zorder=0,
    )
    fig.patches.append(p)


def box(fig, x, y, w, h, fill, edge, lw=1.6, round_=0.02):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.006,rounding_size={round_}",
        transform=fig.transFigure, facecolor=fill, edgecolor=edge,
        linewidth=lw, zorder=2,
    )
    fig.patches.append(p)


def strip(fig, x, y, w, h, data, cmap, vmin=None, vmax=None, sep=True,
          edge="#5a5a5a", elw=1.0):
    """Heatmap strip in figure coords. data shape (rows, cols)."""
    ax = fig.add_axes([x, y, w, h], zorder=3)
    data = np.atleast_2d(data)
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    if sep:
        n = data.shape[1]
        for i in range(1, n):
            ax.axvline(i - 0.5, color="white", lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(edge); s.set_linewidth(elw)
    return ax


def arrow(fig, x0, y0, x1, y1, color=INK, lw=2.2, style="-|>", mut=18):
    a = FancyArrowPatch(
        (x0, y0), (x1, y1), transform=fig.transFigure,
        arrowstyle=style, mutation_scale=mut, color=color, lw=lw, zorder=4,
    )
    fig.patches.append(a)


def title(fig, s):
    fig.text(0.5, 0.945, s, ha="center", va="center",
             fontsize=24, fontweight="bold", color=NAVY)


def takeaway(fig, s, color, x=0.5):
    fig.text(x, 0.055, s, ha="center", va="center",
             fontsize=16, fontweight="bold", color=color)


def label(fig, x, y, s, size=12.5, weight="bold", color=INK, ha="center"):
    fig.text(x, y, s, ha=ha, va="center", fontsize=size,
             fontweight=weight, color=color)


def new_fig():
    return plt.figure(figsize=(12.8, 7.2))


# ---------------------------------------------------------------- 1. OES top-k
def fig_oes_topk(rng):
    fig = new_fig()
    title(fig, "OES Wavelength Selection  (per fold, top-256)")

    panel(fig, 0.06, 0.13, 0.88, 0.76, PANEL_BLUE_F, PANEL_BLUE_E)

    W = 64  # display columns standing in for ~2048 wavelengths
    # a smooth spectrum-ish intensity row
    spec = (np.sin(np.linspace(0, 7, W)) * 0.5 + 0.5
            + rng.normal(0, 0.06, W)).clip(0, 1)

    # full spectrum strip
    label(fig, 0.5, 0.83, "All OES wavelengths  (~2048)", size=14)
    strip(fig, 0.10, 0.74, 0.80, 0.055, spec[None, :], "magma")

    # |corr| curve, train wafers only
    label(fig, 0.5, 0.66, "|corr|  with wafer-mean target   ·   train fold only",
          size=13, color=BLUE_TXT)
    ax = fig.add_axes([0.10, 0.40, 0.80, 0.22], zorder=3)
    corr = (np.abs(np.sin(np.linspace(0.5, 6.5, W) + 0.4))
            * np.linspace(0.4, 1.0, W) + rng.uniform(0, 0.18, W)).clip(0, 1)
    thresh = np.sort(corr)[-25]  # ~ top fraction for the W=64 illustration
    kept = corr >= thresh
    xs = np.arange(W)
    ax.bar(xs[~kept], corr[~kept], color="#c7cdda", width=0.85)
    ax.bar(xs[kept], corr[kept], color="#2a5db0", width=0.85)
    ax.axhline(thresh, color="#d1495b", lw=1.6, ls="--")
    ax.text(W - 0.5, thresh, "  top-256 cut", color="#d1495b", va="center",
            ha="left", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.7, W - 0.3); ax.set_ylim(0, 1.05)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#9aa3b2"); ax.spines["bottom"].set_color("#9aa3b2")

    arrow(fig, 0.5, 0.385, 0.5, 0.305)

    # kept-only strip feeding encoder
    label(fig, 0.5, 0.265, "Top-256 wavelengths  →  OES encoder", size=14,
          color=GREEN_TXT)
    sel = spec.copy()
    disp = np.where(kept, sel, np.nan)
    ax2 = fig.add_axes([0.10, 0.18, 0.80, 0.055], zorder=3)
    masked = np.ma.masked_invalid(disp[None, :])
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#f3f4f8")
    ax2.imshow(masked, cmap=cmap, aspect="auto", interpolation="nearest")
    for i in range(1, W):
        ax2.axvline(i - 0.5, color="white", lw=0.8)
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_edgecolor("#5a5a5a")

    takeaway(fig, "noise wavelengths dropped before the model sees them  ·  no val leakage",
             BLUE_TXT)
    fig.savefig(OUT / "slide_oes_topk.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 2. AUX
def _wafer_disk(fig, cx, cy, r, point_vals, outline="#9aa3b2"):
    """Draw a circular wafer map of 89 points at figure center (cx,cy).

    point_vals: array(89,) of colors, OR a single float for a uniform fill.
    Aspect is corrected for the 16:9 canvas so the disk stays round.
    """
    asp = 12.8 / 7.2
    ax = fig.add_axes([cx - r, cy - r * asp, 2 * r, 2 * r * asp], zorder=3)
    if np.isscalar(point_vals):
        circ = Circle((0, 0), 1.0, facecolor=plt.get_cmap("viridis")(point_vals),
                      edgecolor=outline, lw=1.6)
        ax.add_patch(circ)
    else:
        th = np.linspace(0, 2 * np.pi, 89, endpoint=False)
        rr = np.sqrt(np.linspace(0.02, 1.0, 89))
        np.random.default_rng(7).shuffle(rr)
        px, py = rr * np.cos(th), rr * np.sin(th)
        ax.scatter(px, py, c=point_vals, cmap="viridis", vmin=0, vmax=1,
                   s=24, edgecolor="white", linewidth=0.4)
        ax.add_patch(Circle((0, 0), 1.06, fill=False, color=outline, lw=1.6))
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect("equal")
    ax.axis("off")
    return ax


def fig_aux(rng):
    fig = new_fig()
    title(fig, "Wafer-Mean Auxiliary Loss")

    panel(fig, 0.06, 0.13, 0.88, 0.76, PANEL_GREEN_F, PANEL_GREEN_E)

    # shared wafer representation strip
    label(fig, 0.205, 0.78, "wafer_repr", size=14)
    repr_data = rng.uniform(0, 1, (1, 28))
    strip(fig, 0.10, 0.69, 0.21, 0.06, repr_data, "Purples")
    label(fig, 0.205, 0.645, "one vector per wafer", size=11, weight="normal",
          color=INK)

    # split into two heads
    arrow(fig, 0.315, 0.72, 0.43, 0.655)   # to point head
    arrow(fig, 0.315, 0.72, 0.43, 0.315)   # to aux head

    # ---- point head (main) ----
    box(fig, 0.43, 0.605, 0.155, 0.095, "#ffffff", "#9aa3b2")
    label(fig, 0.5075, 0.6525, "point head\n(+ xy / FiLM)", size=11.5)
    arrow(fig, 0.585, 0.6525, 0.665, 0.6525)
    label(fig, 0.80, 0.84, "89 per-point predictions", size=13, color=NAVY)
    _wafer_disk(fig, 0.80, 0.655, 0.075, rng.uniform(0, 1, 89))
    label(fig, 0.80, 0.50, "spatial map  →  L_point", size=12, color=INK)

    # ---- aux head ----
    box(fig, 0.43, 0.265, 0.155, 0.085, "#ffffff", "#9aa3b2")
    label(fig, 0.5075, 0.3075, "aux head", size=11.5)
    arrow(fig, 0.585, 0.3075, 0.665, 0.3075)
    label(fig, 0.80, 0.44, "1 wafer-mean prediction", size=13, color=NAVY)
    axm = _wafer_disk(fig, 0.80, 0.295, 0.075, 0.62)   # uniform fill = the mean
    axm.text(0, 0, "mean", ha="center", va="center", color="white",
             fontsize=11, fontweight="bold")
    label(fig, 0.80, 0.165, "global etch level  →  L_aux", size=12, color=INK)

    takeaway(fig, "L = L_point  +  w · L_aux      forces wafer_repr to encode the wafer-level mode",
             GREEN_TXT)
    fig.savefig(OUT / "slide_aux_wafer_mean.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 3. Mixup
def fig_mixup(rng):
    fig = new_fig()
    title(fig, "Wafer-Level Mixup")

    # left panel: bimodal gap ; right panel: filled
    panel(fig, 0.05, 0.13, 0.43, 0.74, PANEL_BLUE_F, PANEL_BLUE_E)
    panel(fig, 0.52, 0.13, 0.43, 0.74, PANEL_GREEN_F, PANEL_GREEN_E)
    label(fig, 0.265, 0.82, "Before  ·  bimodal target", size=15, color=NAVY)
    label(fig, 0.735, 0.82, "Mixup fills the gap", size=15, color=GREEN_TXT)

    low = rng.normal(0.59, 0.012, 400)
    high = rng.normal(0.68, 0.013, 400)
    bins = np.linspace(0.54, 0.73, 34)

    axL = fig.add_axes([0.09, 0.30, 0.355, 0.46], zorder=3)
    axL.hist(low, bins=bins, color="#2a5db0", alpha=0.85)
    axL.hist(high, bins=bins, color="#d1495b", alpha=0.85)
    axL.axvspan(0.61, 0.655, color="#9aa3b2", alpha=0.18)
    axL.text(0.6325, axL.get_ylim()[1] * 0.9, "gap", ha="center",
             color="#5a5a5a", fontsize=12, fontweight="bold")
    axL.text(0.59, -axL.get_ylim()[1] * 0.14, "low\nmode", ha="center",
             color="#2a5db0", fontsize=10.5, fontweight="bold")
    axL.text(0.68, -axL.get_ylim()[1] * 0.14, "high\nmode", ha="center",
             color="#d1495b", fontsize=10.5, fontweight="bold")
    axL.set_yticks([]); axL.set_xlabel("oxide_etch", fontsize=11, color=INK)
    for s in ["top", "right", "left"]:
        axL.spines[s].set_visible(False)
    axL.spines["bottom"].set_color("#9aa3b2")

    # mixing formula in the middle, above the arrow
    arrow(fig, 0.475, 0.50, 0.525, 0.50, lw=2.4)
    label(fig, 0.5, 0.585, "λ ~ Beta(α, α)", size=12.5, color=NAVY)
    label(fig, 0.5, 0.42, "mix =\nλ·A + (1−λ)·B", size=11.5, color=INK)

    mid = rng.uniform(0.61, 0.655, 260)
    axR = fig.add_axes([0.565, 0.30, 0.355, 0.46], zorder=3)
    axR.hist(low, bins=bins, color="#2a5db0", alpha=0.5)
    axR.hist(high, bins=bins, color="#d1495b", alpha=0.5)
    axR.hist(mid, bins=bins, color="#2e8b57", alpha=0.9)
    axR.text(0.6325, axR.get_ylim()[1] * 0.78, "synthetic\nintermediates",
             ha="center", color="#2e8b57", fontsize=10.5, fontweight="bold")
    axR.set_yticks([]); axR.set_xlabel("oxide_etch", fontsize=11, color=INK)
    for s in ["top", "right", "left"]:
        axR.spines[s].set_visible(False)
    axR.spines["bottom"].set_color("#9aa3b2")

    takeaway(fig, "mixes inputs AND target  ·  breaks the mode-collapse basin behind fold 2/4 collapse",
             GREEN_TXT)
    fig.savefig(OUT / "slide_mixup.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 4. EMA
def _val_curves(fold):
    no = pd.read_csv(EMA_NOEMA_LOG)
    em = pd.read_csv(EMA_EMA_LOG)
    no = no[no.fold == fold].sort_values("epoch")
    em = em[em.fold == fold].sort_values("epoch")
    return (no.epoch.values, no.val_rmse.values,
            em.epoch.values, em.val_rmse.values)


def _plot_val(ax, fold, legend=True, small=False):
    ne, nr, ee, er = _val_curves(fold)
    ax.plot(ne, nr, color="#9aa3b2", lw=1.4,
            label="raw weights (no EMA)")
    ax.plot(ee, er, color="#2e8b57", lw=2.4,
            label="EMA shadow")
    bi = int(np.argmin(er))
    ax.scatter([ee[bi]], [er[bi]], color="#2e8b57", s=55, zorder=5,
               edgecolor="white")
    nbi = int(np.argmin(nr))
    ax.scatter([ne[nbi]], [nr[nbi]], color="#7c8597", s=40, zorder=5,
               edgecolor="white")
    lo = min(nr.min(), er.min())
    ax.set_ylim(lo - 0.004, np.percentile(np.r_[nr, er], 88))
    ax.set_xlabel("epoch", fontsize=9.5 if small else 11, color=INK)
    ax.set_ylabel("val RMSE  (μm)", fontsize=9.5 if small else 11.5, color=INK)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#9aa3b2"); ax.spines["bottom"].set_color("#9aa3b2")
    ax.tick_params(labelsize=8 if small else 9, colors="#5a5a5a")
    if legend:
        ax.legend(loc="lower left", frameon=False, fontsize=10.5)
    return nr.min(), er.min()


def fig_ema(rng):
    fig = new_fig()
    title(fig, "Weight EMA  (Polyak averaging)")

    panel(fig, 0.06, 0.13, 0.88, 0.76, PANEL_GREEN_F, PANEL_GREEN_E)

    # ---- single large fold-0 val RMSE curve: raw vs EMA-shadow ----
    label(fig, 0.40, 0.82, "fold 0  ·  oxide_etch", size=14, color=NAVY)
    axw = fig.add_axes([0.12, 0.25, 0.56, 0.50], zorder=3)
    n0, e0 = _plot_val(axw, fold=0, legend=True)

    # update rule (right margin)
    label(fig, 0.815, 0.66, "ema ← d·ema\n+ (1−d)·θ", size=14, color=NAVY)
    label(fig, 0.815, 0.555, "decay  d = 0.999", size=12.5, color=INK)
    label(fig, 0.815, 0.50, "validate /\ncheckpoint on\nshadow weights", size=12,
          color=GREEN_TXT)

    takeaway(
        fig,
        f"smoother val trajectory  ·  best RMSE  {n0:.4f} → {e0:.4f}  (fold 0)",
        GREEN_TXT,
    )
    fig.savefig(OUT / "slide_ema.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_oes_topk(np.random.default_rng(1))
    fig_aux(np.random.default_rng(2))
    fig_mixup(np.random.default_rng(3))
    fig_ema(np.random.default_rng(4))
    print("saved 4 figures to", OUT)


if __name__ == "__main__":
    main()
