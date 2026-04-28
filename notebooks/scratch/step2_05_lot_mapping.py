"""Step 2.2 — Resolve experiment_key lot suffix via Lot_status.xlsx.

Goal: figure out the mapping between Day_YYYY_MM_DD and the lot number
that appears as `_LL` suffix in `experiment_key` (e.g., "2024-07-02_01").

Inspect Lot_status.xlsx structure, then verify that mapping with the
89-points measurement CSV's experiment_key column.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.data import DATASET_DIR, list_wafers, load_measurements_89


def main() -> None:
    # 1. Inspect Lot_status.xlsx (probably has multiple sheets)
    path = DATASET_DIR / "Lot_status.xlsx"
    xl = pd.ExcelFile(path)
    print(f"Lot_status.xlsx sheets: {xl.sheet_names}")
    for sn in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sn)
        print(f"\n--- sheet: {sn} ---")
        print(f"shape: {df.shape}")
        print(f"columns: {list(df.columns)}")
        print(df.head(15).to_string())

    # 2. Inspect what's in the 89-pts CSV experiment_key
    print("\n" + "=" * 72)
    meas = load_measurements_89()
    print("Measurements columns:", list(meas.columns))
    keys = sorted(meas["experiment_key"].unique())
    print(f"\nUnique experiment_keys ({len(keys)}):")
    for k in keys:
        cnt = (meas["experiment_key"] == k).sum()
        wfs = sorted(meas[meas["experiment_key"] == k]["wafer_number"].unique())
        print(f"  {k}: {cnt} rows, wafers={wfs}")

    # 3. Compare with Day_*.nc files we have
    print("\n" + "=" * 72)
    days = sorted({w.day for w in list_wafers()})
    print(f"Days from NetCDF: {days}")


if __name__ == "__main__":
    main()
