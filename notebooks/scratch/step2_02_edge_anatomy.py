"""Step 2.2 — Understand WHERE the extra cycles come from.

For representative wafers with n_cycles = 100, 101, 102, plot:
  - SF6 signal at the START (first 40s) with rising edges marked
  - SF6 signal at the END (last 40s) with rising edges marked

Hypothesis: start has ignition artifacts and/or end has truncated cycle.
If we can identify a consistent trim rule (drop first k, last m), then
all wafers become exactly 100 cycles.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import list_wafers, load_process_wafer, segment_cycles_by_sf6
from src.data.loader import SF6_FEATURE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def main() -> None:
    summary = pd.read_csv(FIG_DIR / "step2_01_summary.csv")

    # Pick one wafer per n_cycles class
    reps = {
        100: summary[summary["n_cycles"] == 100].iloc[0],
        101: summary[summary["n_cycles"] == 101].iloc[0],
        102: summary[summary["n_cycles"] == 102].iloc[0],
    }

    wafers = {(w.day, w.wafer_num): w for w in list_wafers()}

    fig, axes = plt.subplots(3, 2, figsize=(16, 10))
    for row_idx, (n_cyc, rep) in enumerate(reps.items()):
        wk = wafers[(rep["day"], int(rep["wafer_num"]))]
        pw = load_process_wafer(wk)
        seg = segment_cycles_by_sf6(pw)
        sf6 = pw.column(SF6_FEATURE)
        rising_t = pw.t_rel[seg.sf6_rising_idx]

        # Start (first 40s)
        ax = axes[row_idx, 0]
        mask = pw.t_rel < 40
        ax.plot(pw.t_rel[mask], sf6[mask], color="#1565C0", lw=0.9)
        for rt in rising_t[rising_t < 40]:
            ax.axvline(rt, color="#EF5350", lw=1.0, alpha=0.7)
        ax.set_title(f"{wk.process_group} — n_cycles={n_cyc} — START (first 40s)")
        ax.set_xlabel("t_rel (s)")
        ax.set_ylabel("SF6 flow")
        ax.grid(alpha=0.3)

        # End (last 40s)
        ax = axes[row_idx, 1]
        t_end = pw.t_rel[-1]
        mask = pw.t_rel > t_end - 40
        ax.plot(pw.t_rel[mask], sf6[mask], color="#1565C0", lw=0.9)
        for rt in rising_t[rising_t > t_end - 40]:
            ax.axvline(rt, color="#EF5350", lw=1.0, alpha=0.7)
        ax.set_title(f"{wk.process_group} — n_cycles={n_cyc} — END (last 40s)")
        ax.set_xlabel("t_rel (s)")
        ax.set_ylabel("SF6 flow")
        ax.grid(alpha=0.3)

        # Print the first and last 3 rising-edge times for diagnostic
        print(f"\n{wk.process_group} (n_cycles={n_cyc})")
        print(f"  duration: {t_end:.2f}s")
        print(f"  first 3 rising: {rising_t[:3]}")
        print(f"  last  3 rising: {rising_t[-3:]}")
        print(f"  first 5 intervals: {np.diff(rising_t[:6])}")
        print(f"  last  5 intervals: {np.diff(rising_t[-6:])}")

    fig.tight_layout()
    path = FIG_DIR / "step2_02_edge_anatomy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
