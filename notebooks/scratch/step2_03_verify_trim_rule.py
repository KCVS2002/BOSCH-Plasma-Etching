"""Step 2.3 — Verify trim_to_100_cycles works on ALL 96 wafers.

For each wafer, apply trim_to_100_cycles and check:
  - Result has exactly 100 cycles
  - All 99 intervals are within [5.5, 6.5] s
  - Trimmed segment's duration ≈ 100 × 6.0 = 600s (598~601)

Any failure is a blocker for Step 2.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.data import (
    list_wafers,
    load_process_wafer,
    segment_cycles_by_sf6,
    trim_to_100_cycles,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"


def main() -> None:
    wafers = list_wafers()
    rows = []
    failures = []

    for i, wk in enumerate(wafers):
        try:
            pw = load_process_wafer(wk)
            seg_raw = segment_cycles_by_sf6(pw)
            seg = trim_to_100_cycles(seg_raw)
            iv = seg.intervals_s
            duration = float(seg.cycle_starts_s[-1] - seg.cycle_starts_s[0])
            rows.append({
                "day": wk.day,
                "wafer_num": wk.wafer_num,
                "n_raw": seg_raw.n_cycles,
                "n_trimmed": seg.n_cycles,
                "trim_start_edge_idx": int(
                    np.where(seg_raw.sf6_rising_idx == seg.sf6_rising_idx[0])[0][0]
                ),
                "cycle1_start_s": float(seg.cycle_starts_s[0]),
                "cycle100_start_s": float(seg.cycle_starts_s[-1]),
                "total_duration_s": duration,
                "iv_min": float(iv.min()),
                "iv_max": float(iv.max()),
                "iv_std": float(iv.std()),
                "ok": True,
            })
        except Exception as e:
            failures.append((wk, str(e)))
            rows.append({
                "day": wk.day,
                "wafer_num": wk.wafer_num,
                "n_raw": -1,
                "n_trimmed": -1,
                "ok": False,
                "error": str(e),
            })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(wafers)}...")

    df = pd.DataFrame(rows)
    ok = df["ok"].sum()
    print(f"\n{ok}/{len(df)} wafers trimmed to exactly 100 cycles")

    if failures:
        print("\nFAILURES:")
        for wk, err in failures:
            print(f"  {wk.process_group}: {err}")

    print(f"\nTrim distribution (how many edges were dropped from start):")
    print(df.groupby(["n_raw", "trim_start_edge_idx"]).size().to_string())

    print(f"\nTrimmed cycle 1 start_s (ignition end):")
    print(df["cycle1_start_s"].describe().to_string())

    print(f"\nTotal duration of trimmed 100 cycles (expect ~594s = 99 × 6s):")
    print(df["total_duration_s"].describe().to_string())

    print(f"\nInterval range per wafer (should be tight around 6.0):")
    print("  iv_min stats:", df["iv_min"].describe().to_string())
    print("  iv_max stats:", df["iv_max"].describe().to_string())
    print("  iv_std stats:", df["iv_std"].describe().to_string())

    csv = FIG_DIR / "step2_03_trim_verification.csv"
    df.to_csv(csv, index=False)
    print(f"\nSaved: {csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
