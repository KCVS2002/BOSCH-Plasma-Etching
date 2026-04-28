"""Step 2.1 — Verify cycle segmentation consistency across ALL wafers.

For each of the 96 wafers, run segment_cycles_by_sf6() and collect:
  - n_cycles (expect 100 or 101)
  - median rising-to-rising interval (expect 6.000 s)
  - interval std (expect small)
  - whether any interval deviates by > 10% from 6s (anomaly flag)

Output:
  - outputs/figures/step2_01_cycle_counts_per_wafer.png
  - outputs/figures/step2_01_interval_std_per_wafer.png
  - prints a summary table; flags any anomalous wafer for manual inspection
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    wafers = list_wafers()
    print(f"Found {len(wafers)} wafers")

    rows = []
    for i, wk in enumerate(wafers):
        try:
            pw = load_process_wafer(wk)
            seg = segment_cycles_by_sf6(pw)
            iv = seg.intervals_s
            row = {
                "day": wk.day,
                "wafer_num": wk.wafer_num,
                "process_group": wk.process_group,
                "n_cycles": seg.n_cycles,
                "duration_s": float(pw.t_rel[-1]),
                "iv_median": float(np.median(iv)) if len(iv) else np.nan,
                "iv_std": float(np.std(iv)) if len(iv) else np.nan,
                "iv_min": float(np.min(iv)) if len(iv) else np.nan,
                "iv_max": float(np.max(iv)) if len(iv) else np.nan,
                "anomalous": bool(
                    len(iv) and (np.abs(iv - 6.0) > 0.6).any()
                ),  # >10% deviation
            }
        except Exception as e:
            row = {
                "day": wk.day,
                "wafer_num": wk.wafer_num,
                "process_group": wk.process_group,
                "n_cycles": -1,
                "duration_s": np.nan,
                "iv_median": np.nan,
                "iv_std": np.nan,
                "iv_min": np.nan,
                "iv_max": np.nan,
                "anomalous": True,
                "error": str(e),
            }
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(wafers)} processed...")

    df = pd.DataFrame(rows)

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Total wafers: {len(df)}")
    print(f"n_cycles distribution:")
    print(df["n_cycles"].value_counts().sort_index().to_string())
    print(f"\nDuration (seconds):")
    print(df["duration_s"].describe().to_string())
    print(f"\nInterval median (should be ~6.000 s):")
    print(df["iv_median"].describe().to_string())
    print(f"\nInterval std (should be small, <0.1):")
    print(df["iv_std"].describe().to_string())

    # Anomalies
    ano = df[df["anomalous"]]
    print(f"\nAnomalous wafers (interval deviates >10% from 6s): {len(ano)}")
    if len(ano):
        print(ano[["day", "wafer_num", "n_cycles", "iv_median", "iv_std", "iv_min", "iv_max"]].to_string())

    # Save summary CSV
    csv_path = FIG_DIR / "step2_01_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary: {csv_path.relative_to(PROJECT_ROOT)}")

    # Plot 1: n_cycles per wafer (bar)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#EF5350" if a else "#42A5F5" for a in df["anomalous"]]
    ax.bar(np.arange(len(df)), df["n_cycles"], color=colors, width=0.85)
    ax.axhline(100, color="#2E7D32", ls="--", lw=1.5, label="expected 100")
    ax.axhline(101, color="#66BB6A", ls=":", lw=1.5, label="observed 101")
    ax.set_xlabel("wafer index (sorted by day, wafer_num)")
    ax.set_ylabel("n_cycles detected (SF6 rising edges)")
    ax.set_title("Cycle segmentation: n_cycles per wafer (red = anomaly)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p1 = FIG_DIR / "step2_01_cycle_counts_per_wafer.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved: {p1.relative_to(PROJECT_ROOT)}")

    # Plot 2: interval std per wafer
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(np.arange(len(df)), df["iv_std"], color=colors, width=0.85)
    ax.set_xlabel("wafer index")
    ax.set_ylabel("interval std (s) — rising-to-rising")
    ax.set_title("Cycle interval stability per wafer (lower = more periodic)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = FIG_DIR / "step2_01_interval_std_per_wafer.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved: {p2.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
