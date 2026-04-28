"""One-shot: print structure of all NetCDF files + CSV/XLSX to understand dataset."""
import os, sys
from pathlib import Path
import netCDF4 as nc
import pandas as pd
import numpy as np

# netCDF4 on Windows can't handle non-ASCII paths → chdir to Dataset and use
# bare filenames. Redirect stdout to utf-8 so Korean prints safely.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DATA = Path(r"c:\Users\ljse2\Desktop\4-1\종합_설계_프로젝트\BOSCH Plasma-Etching\Dataset")
os.chdir(DATA)


def dump_nc(fname: str, max_vars: int = 30):
    path = Path(fname)
    print(f"\n{'='*70}\nFILE: {fname}  ({path.stat().st_size/1e6:.1f} MB)\n{'='*70}")
    with nc.Dataset(fname, "r") as ds:
        print(f"  Dimensions:")
        for dname, dim in ds.dimensions.items():
            print(f"    {dname}: {len(dim)}")
        print(f"  Global attrs: {list(ds.ncattrs())[:6]}")
        print(f"  Variables ({len(ds.variables)}):")
        for i, (vname, var) in enumerate(ds.variables.items()):
            if i >= max_vars:
                print(f"    ... (+{len(ds.variables)-max_vars} more)")
                break
            shape = var.shape
            dtype = var.dtype
            attrs = {a: var.getncattr(a) for a in var.ncattrs()[:3]}
            print(f"    [{vname}]  shape={shape}  dtype={dtype}  attrs={attrs}")
        # Groups (if any)
        if ds.groups:
            print(f"  Groups: {list(ds.groups.keys())[:5]}")
            for gname, g in list(ds.groups.items())[:2]:
                print(f"    Group '{gname}':")
                print(f"      dims: { {n: len(d) for n, d in g.dimensions.items()} }")
                print(f"      vars: {list(g.variables.keys())[:10]}")


# --- Dictionary files (likely metadata/schemas) ---
for f in ["Dictionary_OES.nc", "Dictionary_process.nc"]:
    dump_nc(f)

# --- One day-file (small structural peek) ---
day_files = sorted(Path(".").glob("Day_*.nc"))
print(f"\n\nFound {len(day_files)} Day_*.nc files")
dump_nc(day_files[0].name, max_vars=40)

# --- Process_data.nc ---
dump_nc("Process_data.nc", max_vars=40)

# --- CSV measurements ---
for f in ["Si_Oxide_etch_89_points.csv", "Si_Oxide_etch_9_points.csv"]:
    print(f"\n{'='*70}\nCSV: {f}\n{'='*70}")
    df = pd.read_csv(f)
    print(f"  shape: {df.shape}")
    print(f"  columns: {list(df.columns)}")
    print(f"  head:\n{df.head(3)}")
    print(f"  dtypes:\n{df.dtypes.to_string()}")

# --- Lot_status.xlsx ---
print(f"\n{'='*70}\nXLSX: Lot_status.xlsx\n{'='*70}")
xlsx = pd.ExcelFile("Lot_status.xlsx")
print(f"  sheets: {xlsx.sheet_names}")
for s in xlsx.sheet_names[:3]:
    df = xlsx.parse(s)
    print(f"  sheet '{s}': shape={df.shape}, cols={list(df.columns)[:10]}")
    print(df.head(3))
