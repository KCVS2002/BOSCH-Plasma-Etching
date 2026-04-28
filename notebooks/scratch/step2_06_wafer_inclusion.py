"""Step 2.3 — Verify wafer inclusion: 96 NetCDF wafers vs 88 with measurements.

Match WaferKey.experiment_key against the measurement CSV. Build the
inclusion list (88 wafers we will train on) and identify which 8 are
excluded and which lot they belong to.
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.data import list_wafers, load_measurements_89


def main() -> None:
    wafers = list_wafers()
    meas = load_measurements_89()
    measured_keys = set(meas["experiment_key"].unique())

    rows = []
    for wk in wafers:
        rows.append({
            "lot_number": wk.lot_number,
            "day": wk.day,
            "wafer_num": wk.wafer_num,
            "experiment_key": wk.experiment_key,
            "has_measurements": wk.experiment_key in measured_keys,
        })
    df = pd.DataFrame(rows).sort_values(["lot_number", "wafer_num"]).reset_index(drop=True)

    print(f"Total NetCDF wafers: {len(df)}")
    print(f"Wafers with measurements: {df['has_measurements'].sum()}")
    print(f"Excluded (no measurements): {(~df['has_measurements']).sum()}")

    print("\nExcluded wafers:")
    print(df[~df["has_measurements"]].to_string(index=False))

    print("\nWafers per lot (with measurements):")
    print(df[df["has_measurements"]].groupby("lot_number").size().to_string())

    # Check: every measurement key has a corresponding NetCDF wafer
    netcdf_keys = {wk.experiment_key for wk in wafers}
    orphan = measured_keys - netcdf_keys
    if orphan:
        print(f"\nORPHAN measurements (no NetCDF): {sorted(orphan)}")
    else:
        print("\nAll 88 measurement keys have a NetCDF wafer ✓")


if __name__ == "__main__":
    main()
