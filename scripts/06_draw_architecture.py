"""Architecture diagrams for XGBoost baseline and Cycle-Aware DL model.

Saves:
  outputs/figures/arch_xgboost.png
  outputs/figures/arch_dl_multimodal.png

Run:
    python -m scripts.06_draw_architecture
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C = {
    "oes":    "#1565C0",
    "proc":   "#C62828",
    "xy":     "#2E7D32",
    "shared": "#4527A0",
    "head":   "#37474F",
    "feat":   "#00695C",
    "xgb":    "#E65100",
    "bg":     "#F5F5F5",
}


def _box(ax, cx, cy, w, h, label, color,
         fontsize=11, text_color="white", alpha=0.93):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.04",
        facecolor=color, edgecolor="white",
        linewidth=2.0, alpha=alpha, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, label,
            ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold", multialignment="center",
            linespacing=1.5, zorder=4)


def _arrow(ax, x0, y0, x1, y1, color="#666", lw=2.0, rad=0.0):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        zorder=2,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
            mutation_scale=14,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# XGBoost Architecture
# ═══════════════════════════════════════════════════════════════════════════════

def draw_xgboost(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 15)
    ax.set_ylim(0.2, 9)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle("XGBoost Baseline Architecture",
                 fontsize=17, fontweight="bold", y=0.98, color="#1A1A1A")

    BH   = 0.90   # box height
    OW   = 4.2    # OES / Process branch width
    XW   = 2.0    # XY box width

    oes_x  = 3.0
    proc_x = 8.2
    xy_x   = 12.8

    # ── Row 1  y=8.0 : Raw inputs ────────────────────────────────────────────
    _box(ax, oes_x,  8.0, OW, BH,
         "OES Timeseries\n(~14,744 × 3,648 wavelengths)", C["oes"])
    _box(ax, proc_x, 8.0, OW, BH,
         "Process Timeseries\n(~3,245 × 44 channels)", C["proc"])

    # ── Row 2  y=6.5 : Cycle aggregation ─────────────────────────────────────
    _box(ax, oes_x,  6.5, OW, BH,
         "Per-cycle band mean\n(100 cycles × 10 OES bands)", C["feat"])
    _box(ax, proc_x, 6.5, OW, BH,
         "Per-cycle channel mean\n(100 cycles × 44 channels)", C["feat"])

    _arrow(ax, oes_x,  7.55, oes_x,  6.96)
    _arrow(ax, proc_x, 7.55, proc_x, 6.96)

    # ── Row 3  y=5.0 : Statistical summary + XY ──────────────────────────────
    _box(ax, oes_x,  5.0, OW, BH,
         "8 summary stats × 10 bands\n→ 80 OES features\n"
         "(mean / std / min / max / slope / early / late / drift)",
         C["feat"], fontsize=9.5)
    _box(ax, proc_x, 5.0, OW, BH,
         "8 summary stats × 44 channels\n→ 352 Process features\n"
         "(mean / std / min / max / slope / early / late / drift)",
         C["feat"], fontsize=9.5)
    _box(ax, xy_x,   5.0, XW, BH,
         "XY Coords\n(89 × 2)", C["xy"])

    _arrow(ax, oes_x,  6.04, oes_x,  5.46)
    _arrow(ax, proc_x, 6.04, proc_x, 5.46)

    # ── Row 4  y=3.3 : Concatenated feature vector ───────────────────────────
    _box(ax, 7.5, 3.3, 13.0, BH,
         "Concatenated Feature Vector"
         "     (80 OES  +  352 Process  +  2 XY  =  434 features)",
         C["head"])

    _arrow(ax, oes_x,  4.54, 3.2,  3.76, rad=-0.08)
    _arrow(ax, proc_x, 4.54, 7.5,  3.76)
    _arrow(ax, xy_x,   4.54, 11.8, 3.76, rad= 0.08)

    # ── Row 5  y=2.0 : XGBoost ────────────────────────────────────────────────
    _box(ax, 7.5, 2.0, 8.5, BH,
         "XGBoost Regressor\n"
         "800 trees  ·  max_depth = 6  ·  learning_rate = 0.05",
         C["xgb"])

    _arrow(ax, 7.5, 2.84, 7.5, 2.46)

    # ── Row 6  y=0.85 : Outputs ────────────────────────────────────────────────
    _box(ax, 5.8,  0.85, 3.2, 0.75, "si_etch  (μm)",    C["oes"],  fontsize=12)
    _box(ax, 10.0, 0.85, 3.6, 0.75, "oxide_etch  (μm)", C["proc"], fontsize=12)

    _arrow(ax, 6.5,  1.54, 5.8,  1.23)
    _arrow(ax, 8.5,  1.54, 10.0, 1.23)

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=C["oes"],  label="OES pathway"),
        mpatches.Patch(color=C["proc"], label="Process pathway"),
        mpatches.Patch(color=C["xy"],   label="Spatial (XY)"),
        mpatches.Patch(color=C["feat"], label="Feature engineering"),
        mpatches.Patch(color=C["xgb"],  label="XGBoost model"),
        mpatches.Patch(color=C["head"], label="Output"),
    ]
    ax.legend(handles=patches, loc="lower center", ncol=6,
              fontsize=9.5, framealpha=0.85,
              bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# DL Multimodal Architecture
# ═══════════════════════════════════════════════════════════════════════════════

def draw_dl_multimodal(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 13))
    ax.set_xlim(0, 16)
    ax.set_ylim(0.2, 13)
    ax.axis("off")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    fig.suptitle(
        "Cycle-Aware DL Architecture  (2D-CNN  +  Bi-LSTM  +  FiLM)",
        fontsize=17, fontweight="bold", y=0.99, color="#1A1A1A",
    )

    BH   = 0.85   # box height
    BW   = 4.2    # branch box width  (OES / Process)
    CW   = 6.0    # centre box width  (fusion → head)
    XW   = 2.8    # XY branch width

    oes_x  = 2.8
    proc_x = 11.0
    cx     = 6.9   # centre column
    xy_x   = 14.1  # XY branch

    # ── Row 1  y=12.0 : Inputs ────────────────────────────────────────────────
    _box(ax, oes_x,  12.0, BW, BH,
         "Per-Cycle OES Input\n(100 cycles × 128 steps × 3,648 λ)", C["oes"])
    _box(ax, proc_x, 12.0, BW, BH,
         "Per-Cycle Process Input\n(100 cycles × 30 steps × 31 ch)", C["proc"])

    # ── "same family" brace ───────────────────────────────────────────────────
    ax.annotate(
        "", xy=(oes_x + BW/2 + 0.15, 12.0),
        xytext=(proc_x - BW/2 - 0.15, 12.0),
        zorder=2,
        arrowprops=dict(arrowstyle="<->", color="#AAA", lw=1.3,
                        connectionstyle="arc3,rad=0"),
    )
    ax.text(cx, 12.18,
            "Same architecture family  ·  Independent weights",
            ha="center", va="bottom", fontsize=9, color="#777",
            style="italic", zorder=5)

    # ── Row 2  y=10.6 : CNN Encoders ─────────────────────────────────────────
    _box(ax, oes_x,  10.6, BW, BH,
         "2D-CNN Encoder\n(3 blocks · strided conv · BN · GELU)\nShared weights across 100 cycles",
         C["oes"], fontsize=10)
    _box(ax, proc_x, 10.6, BW, BH,
         "2D-CNN Encoder\n(3 blocks · strided conv · BN · GELU)\nShared weights across 100 cycles",
         C["proc"], fontsize=10)

    _arrow(ax, oes_x,  11.57, oes_x,  11.03)
    _arrow(ax, proc_x, 11.57, proc_x, 11.03)

    # ── Row 3  y=9.2 : Cycle Embeddings ──────────────────────────────────────
    _box(ax, oes_x,  9.2, BW, BH,
         "OES Cycle Embedding\n(100 × 128)", C["oes"])
    _box(ax, proc_x, 9.2, BW, BH,
         "Process Cycle Embedding\n(100 × 64)", C["proc"])

    _arrow(ax, oes_x,  10.17, oes_x,  9.62)
    _arrow(ax, proc_x, 10.17, proc_x, 9.62)

    # ── Row 4  y=7.7 : Fusion ────────────────────────────────────────────────
    _arrow(ax, oes_x  + BW/2, 8.77, cx - CW/2 + 0.5, 8.13, rad=-0.12)
    _arrow(ax, proc_x - BW/2, 8.77, cx + CW/2 - 0.5, 8.13, rad= 0.12)

    _box(ax, cx, 7.7, CW, BH,
         "Concat  →  FC  (GELU + LayerNorm)\nCycle Embedding  (100 × 128)", C["shared"])

    # ── Row 5  y=6.3 : Bi-LSTM ────────────────────────────────────────────────
    _arrow(ax, cx, 7.27, cx, 6.73)

    _box(ax, cx, 6.3, CW, BH,
         "Bidirectional LSTM\n(100 × 256  =  2 × 128 hidden units)", C["shared"])

    # ── Row 6  y=4.9 : Mean Pool  +  XY Input ────────────────────────────────
    _arrow(ax, cx, 5.87, cx, 5.33)

    _box(ax, cx, 4.9, CW, BH,
         "Mean Pool over 100 cycles\nWafer Representation  (256)", C["shared"])

    _box(ax, xy_x, 4.9, XW, BH,
         "XY Coords\n(89 × 2)", C["xy"])

    # ── Row 7  y=3.5 : FiLM  +  Fourier Encoder ──────────────────────────────
    _arrow(ax, cx, 4.47, cx, 3.93)

    _box(ax, xy_x, 3.5, XW, BH,
         "Fourier Encoder\n(89 × 64)", C["xy"])

    _arrow(ax, xy_x, 4.47,  xy_x, 3.93)
    # horizontal arrow from Fourier into FiLM
    _arrow(ax, xy_x - XW/2, 3.5,  cx + CW/2, 3.5)

    _box(ax, cx, 3.5, CW, BH,
         "FiLM Modulation\n( γ · wafer_repr + β )  per measurement point\n"
         "Per-Point Repr  (89 × 256)",
         C["shared"], fontsize=10)

    # ── Row 8  y=2.1 : Regression Head ────────────────────────────────────────
    _arrow(ax, cx, 3.07, cx, 2.53)

    _box(ax, cx, 2.1, CW, BH,
         "Regression Head\n(FC → GELU → Dropout(0.2) → FC)\n(89 × 1)",
         C["head"], fontsize=10)

    # ── Row 9  y=0.85 : Outputs ───────────────────────────────────────────────
    _arrow(ax, 5.5, 1.67, 4.5, 1.23)
    _arrow(ax, 8.3, 1.67, 9.3, 1.23)

    _box(ax, 4.0,  0.85, 3.0, 0.70, "si_etch  (89 pts, μm)",    C["oes"],  fontsize=11)
    _box(ax, 9.8,  0.85, 3.4, 0.70, "oxide_etch  (89 pts, μm)", C["proc"], fontsize=11)

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=C["oes"],    label="OES pathway"),
        mpatches.Patch(color=C["proc"],   label="Process pathway"),
        mpatches.Patch(color=C["xy"],     label="Spatial (XY)"),
        mpatches.Patch(color=C["shared"], label="Shared DL components"),
        mpatches.Patch(color=C["head"],   label="Regression head / output"),
    ]
    ax.legend(handles=patches, loc="lower center", ncol=5,
              fontsize=10, framealpha=0.85,
              bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    out_dir = PROJECT_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_xgboost(out_dir / "arch_xgboost.png")
    draw_dl_multimodal(out_dir / "arch_dl_multimodal.png")
    print("Done.")


if __name__ == "__main__":
    main()
