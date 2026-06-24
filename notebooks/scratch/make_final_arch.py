"""Final Cycle-Aware DL pipeline diagram (polished two-tone style).

Matches the refined architecture-slide aesthetic (white canvas, dark input
headers, light-fill processing nodes with colored borders, numbered purple
badges on the shared trunk) and adds the four fold-collapse remedies that
define the final model:

  - OES top-256 wavelength selection  (inference-path block in the OES branch)
  - Wafer-mean Aux head               (architecture branch off wafer_repr)
  - Wafer-level Mixup / Weight EMA / 120ep  (train-time recipe, dashed panel)

Output: outputs/figures/arch_dl_final.png

Run:
    .venv\\Scripts\\python.exe notebooks/scratch/make_final_arch.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "outputs" / "figures"

NAVY = "#1b2a4a"
C = {
    "oes_d": "#1565C0", "oes_l": "#E7F0FB",
    "proc_d": "#C62828", "proc_l": "#FCEAEA",
    "xy_d": "#2E7D32", "xy_l": "#E9F4EC",
    "sh_d": "#4527A0", "sh_l": "#EDE7F8", "sh_t": "#3A2A82", "badge": "#5B3FB5",
    "head_d": "#37474F",
    "sel_d": "#00695C", "sel_l": "#E2F1EF",
    "grey": "#9aa3b2", "ink": "#445",
}

plt.rcParams.update({"font.family": "DejaVu Sans"})


def _node(ax, cx, cy, w, h, title, sub, fill, edge, tcolor, scolor,
          lw=2.0, dashed=False, badge=None, title_size=11.5, sub_size=9.3):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
        linestyle=(":" if dashed else "-"),
    )
    ax.add_patch(p)
    tx = cx
    if badge is not None:
        bx = cx - w / 2 + 0.42
        ax.add_patch(Circle((bx, cy), 0.26, facecolor=C["badge"],
                            edgecolor="white", lw=1.4, zorder=5))
        ax.text(bx, cy, str(badge), ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", zorder=6)
        tx = cx + 0.32
    if sub:
        ax.text(tx, cy + h * 0.17, title, ha="center", va="center",
                fontsize=title_size, color=tcolor, fontweight="bold", zorder=4)
        ax.text(tx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=sub_size, color=scolor, zorder=4,
                multialignment="center", linespacing=1.35)
    else:
        ax.text(tx, cy, title, ha="center", va="center", fontsize=title_size,
                color=tcolor, fontweight="bold", zorder=4,
                multialignment="center", linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, color="#7a8290", lw=2.0, rad=0.0, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}",
                                mutation_scale=15,
                                linestyle=(":" if dashed else "-")))


def draw(out_path):
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_xlim(0, 16); ax.set_ylim(0, 14.3); ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.suptitle("Final Cycle-Aware DL Pipeline  (OES top-k · FiLM · Aux · Mixup · EMA)",
                 fontsize=18, fontweight="bold", y=0.975, color=NAVY)

    BW, CW, XW = 4.0, 5.8, 2.6
    oes_x, proc_x, cx, xy_x = 3.1, 10.4, 6.75, 13.6
    BH = 0.92

    # ---- inputs ----
    _node(ax, oes_x, 13.1, BW, BH, "Per-Cycle OES Input",
          "100 cycles × 128 steps × 3,648 λ", C["oes_d"], "white", "white", "#dce8f7")
    _node(ax, proc_x, 13.1, BW, BH, "Per-Cycle Process Input",
          "100 cycles × 30 steps × 31 ch", C["proc_d"], "white", "white", "#f6dada")

    ax.annotate("", xy=(oes_x + BW / 2 + 0.1, 13.1), xytext=(proc_x - BW / 2 - 0.1, 13.1),
                arrowprops=dict(arrowstyle="<->", color=C["grey"], lw=1.3))
    ax.text(cx, 13.78, "Same family  ·  independent weights",
            ha="center", va="bottom", fontsize=9, color="#888", style="italic")

    # ---- OES top-k selection (OES branch only) ----
    _node(ax, oes_x, 12.0, BW, 0.78, "Top-256 Wavelength Selection",
          "|corr| with wafer-mean · per fold · train-only", C["sel_l"], C["sel_d"],
          C["sel_d"], C["sel_d"], title_size=10.3, sub_size=8.6)
    _arrow(ax, oes_x, 12.64, oes_x, 12.41, color=C["oes_d"])
    _arrow(ax, oes_x, 11.61, oes_x, 11.27, color=C["oes_d"])
    _arrow(ax, proc_x, 12.64, proc_x, 11.27, color=C["proc_d"])

    # ---- encoders ----
    _node(ax, oes_x, 10.75, BW, BH, "2D-CNN Encoder",
          "3 blocks · strided conv · BN · GELU\nshared across 100 cycles · in=256 λ",
          C["oes_l"], C["oes_d"], C["oes_d"], C["oes_d"], title_size=10.8, sub_size=8.8)
    _node(ax, proc_x, 10.75, BW, BH, "2D-CNN Encoder",
          "3 blocks · strided conv · BN · GELU\nshared across 100 cycles",
          C["proc_l"], C["proc_d"], C["proc_d"], C["proc_d"], title_size=10.8, sub_size=8.8)
    _arrow(ax, oes_x, 10.29, oes_x, 10.0, color=C["oes_d"])
    _arrow(ax, proc_x, 10.29, proc_x, 10.0, color=C["proc_d"])

    # ---- embeddings ----
    _node(ax, oes_x, 9.55, BW, 0.8, "OES Cycle Embedding", "100 × 128",
          C["oes_l"], C["oes_d"], C["oes_d"], C["oes_d"], title_size=11)
    _node(ax, proc_x, 9.55, BW, 0.8, "Process Cycle Embedding", "100 × 64",
          C["proc_l"], C["proc_d"], C["proc_d"], C["proc_d"], title_size=11)

    # ---- fusion ----
    _arrow(ax, oes_x + BW / 2 - 0.3, 9.15, cx - CW / 2 + 0.5, 8.62, color=C["oes_d"], rad=-0.18)
    _arrow(ax, proc_x - BW / 2 + 0.3, 9.15, cx + CW / 2 - 0.5, 8.62, color=C["proc_d"], rad=0.18)

    _node(ax, cx, 8.2, CW, 0.85, "Concat  →  FC  (GELU + LayerNorm)",
          "Cycle Embedding  (100 × 128)", C["sh_l"], C["sh_d"], C["sh_t"], C["ink"],
          badge=1)
    _arrow(ax, cx, 7.77, cx, 7.45, color=C["sh_d"])

    _node(ax, cx, 7.0, CW, 0.85, "Bidirectional LSTM",
          "100 × 256  =  2 × 128 hidden units", C["sh_l"], C["sh_d"], C["sh_t"],
          C["ink"], badge=2)
    _arrow(ax, cx, 6.57, cx, 6.25, color=C["sh_d"])

    _node(ax, cx, 5.8, CW, 0.85, "Mean Pool over 100 cycles",
          "Wafer Representation  (256)", C["sh_l"], C["sh_d"], C["sh_t"], C["ink"],
          badge=3)

    # ---- Aux head branch (off wafer representation) ----
    _node(ax, 2.0, 5.8, 3.3, 0.95, "Aux Head  (FC → 1)",
          "wafer-mean prediction\ntrain-time  L_aux", C["sh_l"], C["sh_d"],
          C["sh_t"], C["ink"], dashed=True, title_size=10.5, sub_size=8.8)
    _arrow(ax, cx - CW / 2, 5.8, 3.65, 5.8, color=C["sh_d"], dashed=True)

    # ---- XY pathway (right) ----
    _node(ax, xy_x, 5.8, XW, 0.8, "XY Coords", "89 × 2", C["xy_l"], C["xy_d"],
          C["xy_d"], C["xy_d"], title_size=11)
    _node(ax, xy_x, 4.55, XW, 0.8, "Fourier Encoder", "89 × 64", C["xy_l"],
          C["xy_d"], C["xy_d"], C["xy_d"], title_size=11)
    _arrow(ax, xy_x, 5.4, xy_x, 4.95, color=C["xy_d"])
    _arrow(ax, xy_x - XW / 2, 4.55, cx + CW / 2, 4.55, color=C["xy_d"])

    _arrow(ax, cx, 5.37, cx, 5.0, color=C["sh_d"])
    _node(ax, cx, 4.55, CW, 0.92, "FiLM Modulation",
          "( γ · wafer_repr + β )  per measurement point\nPer-Point Repr  (89 × 256)",
          C["sh_l"], C["sh_d"], C["sh_t"], C["ink"], badge=4, title_size=11,
          sub_size=8.9)
    _arrow(ax, cx, 4.09, cx, 3.72, color=C["sh_d"])

    _node(ax, cx, 3.25, CW, 0.92, "Regression Head",
          "FC → GELU → Dropout(0.2) → FC\n(89 × 1)", C["head_d"], "white",
          "white", "#cfd6db", badge=5, title_size=11, sub_size=9)

    # ---- training recipe panel (train-time only, dashed) ----
    rx, ry = 13.2, 3.0
    _node(ax, rx, ry, 4.4, 2.05, "", "", "#F4F4F6", C["grey"],
          NAVY, C["ink"], dashed=True)
    ax.text(rx, ry + 0.74, "Training recipe", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(rx, ry + 0.46, "train-time only", ha="center", va="center",
            fontsize=8.6, color="#888", style="italic")
    for i, txt in enumerate([
        "• Wafer-level Mixup   λ ~ Beta(α, α)",
        "• Weight EMA   decay 0.999 (val on shadow)",
        "• 120 epochs · no early stop",
    ]):
        ax.text(rx - 2.0, ry + 0.08 - i * 0.42, txt, ha="left", va="center",
                fontsize=9.3, color=C["ink"])
    _arrow(ax, rx - 2.2, ry + 0.4, cx + CW / 2, 3.25, color=C["grey"], dashed=True,
           rad=0.12)

    # ---- outputs ----
    _arrow(ax, cx - 1.4, 2.79, 5.0, 2.35, color=C["oes_d"])
    _arrow(ax, cx + 1.4, 2.79, 8.5, 2.35, color=C["proc_d"])
    _node(ax, 4.5, 1.95, 3.0, 0.72, "si_etch  (89 pts, μm)", "", "white",
          C["oes_d"], C["oes_d"], C["oes_d"], title_size=11.5)
    _node(ax, 9.0, 1.95, 3.4, 0.72, "oxide_etch  (89 pts, μm)", "", "white",
          C["proc_d"], C["proc_d"], C["proc_d"], title_size=11.5)

    # ---- legend ----
    handles = [
        mpatches.Patch(color=C["oes_d"], label="OES pathway"),
        mpatches.Patch(color=C["proc_d"], label="Process pathway"),
        mpatches.Patch(color=C["sel_d"], label="Wavelength selection"),
        mpatches.Patch(color=C["xy_d"], label="Spatial (XY)"),
        mpatches.Patch(color=C["sh_d"], label="Shared DL trunk"),
        mpatches.Patch(color=C["head_d"], label="Regression head"),
        mpatches.Patch(facecolor="#F4F4F6", edgecolor=C["grey"], linestyle=":",
                       label="Train-time only"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=7, fontsize=9.2,
              framealpha=0.9, bbox_to_anchor=(0.5, -0.005))

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    draw(OUT / "arch_dl_final.png")
