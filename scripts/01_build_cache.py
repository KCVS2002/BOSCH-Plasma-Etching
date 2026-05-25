"""Phase-1 entry: raw NetCDF → cache/<version>/.

For each measured wafer (88 of 96):
  1. Load OES (uint16) + Process (float)
  2. Segment cycles by SF6 rising edges, trim to exactly 100 cycles
  3. Cross-correlate SF6 mask vs. OES total intensity → time offset
  4. Compute per-cycle (start, end_exclusive) indices for both modalities
  5. Save per-wafer NPZ under cache/<version>/wafers/<experiment_key>.npz

Aggregate outputs:
  - cache/<version>/measurements.parquet  — copy of 89-points table
  - cache/<version>/manifest.json         — wafer list + per-wafer summary
  - cache/<version>/README.md             — version description

Run from project root:
    python scripts/01_build_cache.py --version v1
    python scripts/01_build_cache.py --version v1 --limit 2     # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Project-root bootstrap so `from src.data import ...` works under direct
# execution (`python scripts/01_build_cache.py`). The leading digit makes
# `python -m scripts.01_build_cache` impossible (invalid identifier).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from tqdm import tqdm

from src.data import (
    align_oes_to_process,
    cycle_indices_oes,
    cycle_indices_proc,
    list_wafers,
    load_measurements_89,
    load_oes_wafer,
    load_process_wafer,
    segment_cycles_by_sf6,
    trim_to_100_cycles,
)


def build_one_wafer(wk, out_dir: Path) -> dict:
    t0 = time.time()
    oes = load_oes_wafer(wk)
    proc = load_process_wafer(wk)

    seg_raw = segment_cycles_by_sf6(proc)
    seg = trim_to_100_cycles(seg_raw)

    offset_s = align_oes_to_process(oes, proc)
    oes_starts, oes_ends = cycle_indices_oes(oes, seg.cycle_starts_s, offset_s)
    proc_starts, proc_ends = cycle_indices_proc(seg, n_total=len(proc.t_rel))

    # Sanity: each cycle should have ≥1 sample in both modalities
    if (oes_ends - oes_starts).min() < 1:
        raise ValueError(f"empty OES cycle slice on {wk.experiment_key}")
    if (proc_ends - proc_starts).min() < 1:
        raise ValueError(f"empty process cycle slice on {wk.experiment_key}")

    out_path = out_dir / f"{wk.experiment_key}.npz"
    np.savez_compressed(
        out_path,
        # Raw arrays (compressed)
        oes_data=oes.data.astype(np.uint16, copy=False),
        oes_t_rel=oes.t_rel.astype(np.float32),
        oes_wavelengths=oes.wavelengths.astype(np.float32),
        process_data=proc.data.astype(np.float32),
        process_t_rel=proc.t_rel.astype(np.float32),
        process_features=np.array([str(f) for f in proc.features]),
        # Cycle indices (end exclusive)
        oes_cycle_starts_idx=oes_starts,
        oes_cycle_ends_idx=oes_ends,
        proc_cycle_starts_idx=proc_starts,
        proc_cycle_ends_idx=proc_ends,
        cycle_starts_proc_s=seg.cycle_starts_s.astype(np.float32),
        # Alignment + metadata
        alignment_offset_s=np.float32(offset_s),
        experiment_key=np.array(wk.experiment_key),
        day=np.array(wk.day),
        wafer_num=np.int32(wk.wafer_num),
        lot_number=np.int32(wk.lot_number),
    )

    return {
        "experiment_key": wk.experiment_key,
        "day": wk.day,
        "wafer_num": wk.wafer_num,
        "lot_number": wk.lot_number,
        "n_oes_samples": int(oes.data.shape[0]),
        "n_oes_wavelengths": int(oes.data.shape[1]),
        "n_process_samples": int(proc.data.shape[0]),
        "n_process_features": int(proc.data.shape[1]),
        "n_cycles": int(seg.n_cycles),
        "alignment_offset_s": round(float(offset_s), 4),
        "oes_cycle_len_min": int((oes_ends - oes_starts).min()),
        "oes_cycle_len_max": int((oes_ends - oes_starts).max()),
        "proc_cycle_len_min": int((proc_ends - proc_starts).min()),
        "proc_cycle_len_max": int((proc_ends - proc_starts).max()),
        "elapsed_s": round(time.time() - t0, 2),
        "file_mb": round(out_path.stat().st_size / 1e6, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v1",
                        help="cache version (output goes to cache/<version>/)")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N measured wafers (smoke testing)")
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="only process wafers whose measurement 'day' equals this value (format: YYYY_MM_DD)",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="re-process wafers whose npz already exists")
    args = parser.parse_args()

    cache_root = PROJECT_ROOT / "cache" / args.version
    wafer_dir = cache_root / "wafers"
    wafer_dir.mkdir(parents=True, exist_ok=True)

    all_wafers = list_wafers()
    meas = load_measurements_89()
    # Optional day filter: restrict measurements and wafers to the specified day.
    # The measurements CSV doesn't have a `day` column, so match by the
    # `experiment_key` prefix (YYYY-MM-DD) derived from the provided
    # YYYY_MM_DD input.
    if args.day is not None:
        day_dash = args.day.replace("_", "-")
        meas = meas[meas["experiment_key"].str.startswith(day_dash)]
    measured_keys = set(meas["experiment_key"].unique())
    wafers = [w for w in all_wafers if w.experiment_key in measured_keys]
    wafers.sort(key=lambda w: (w.lot_number, w.wafer_num))
    if args.limit is not None:
        wafers = wafers[: args.limit]

    print(f"Building cache '{args.version}' for {len(wafers)} wafers")
    print(f"  output: {cache_root.relative_to(PROJECT_ROOT)}")

    manifest_entries: list[dict] = []
    failures: list[dict] = []
    skipped = 0
    for wk in tqdm(wafers, desc="wafers", unit="wafer"):
        out_path = wafer_dir / f"{wk.experiment_key}.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            entry = build_one_wafer(wk, wafer_dir)
            manifest_entries.append(entry)
        except Exception as e:
            failures.append({"experiment_key": wk.experiment_key, "error": repr(e)})
            tqdm.write(f"  FAIL {wk.experiment_key}: {e!r}")

    # Measurements snapshot — parquet preferred, csv fallback if pyarrow missing
    meas_path = cache_root / "measurements.parquet"
    try:
        meas.to_parquet(meas_path, index=False)
    except Exception:
        meas_path = cache_root / "measurements.csv"
        meas.to_csv(meas_path, index=False)

    manifest = {
        "version": args.version,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "n_wafers_total": len(wafers),
        "n_built": len(manifest_entries),
        "n_skipped_existing": skipped,
        "n_failed": len(failures),
        "wafers": manifest_entries,
        "failures": failures,
    }
    (cache_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = (
        f"# Cache `{args.version}`\n\n"
        f"- Built: {manifest['built_at']}\n"
        f"- Wafers: {manifest['n_built']} built, {manifest['n_skipped_existing']} skipped, "
        f"{manifest['n_failed']} failed (of {len(wafers)} requested)\n\n"
        f"## Per-wafer NPZ contents (`wafers/<experiment_key>.npz`)\n\n"
        f"| key | shape | dtype | meaning |\n"
        f"|---|---|---|---|\n"
        f"| `oes_data` | (T_o, 3648) | uint16 | raw OES counts |\n"
        f"| `oes_t_rel` | (T_o,) | float32 | seconds from first OES sample |\n"
        f"| `oes_wavelengths` | (3648,) | float32 | wavelength axis (nm) |\n"
        f"| `process_data` | (T_p, 44) | float32 | process channels |\n"
        f"| `process_t_rel` | (T_p,) | float32 | process-clock seconds (relative) |\n"
        f"| `process_features` | (44,) | str | feature names |\n"
        f"| `oes_cycle_starts_idx` | (100,) | int32 | OES cycle start indices |\n"
        f"| `oes_cycle_ends_idx` | (100,) | int32 | OES cycle end indices (exclusive) |\n"
        f"| `proc_cycle_starts_idx` | (100,) | int32 | process cycle start indices |\n"
        f"| `proc_cycle_ends_idx` | (100,) | int32 | process cycle end indices (exclusive) |\n"
        f"| `cycle_starts_proc_s` | (100,) | float32 | cycle starts on process clock (s) |\n"
        f"| `alignment_offset_s` | scalar | float32 | `process_time ≈ oes_t_rel + offset` |\n"
        f"| `experiment_key`, `day`, `wafer_num`, `lot_number` | scalars | — | metadata |\n\n"
        f"## Aggregate files\n\n"
        f"- `measurements.parquet` (or `.csv`): copy of `Si_Oxide_etch_89_points.csv`\n"
        f"- `manifest.json`: build metadata + per-wafer summary\n\n"
        f"Generated by `scripts/01_build_cache.py`.\n"
    )
    (cache_root / "README.md").write_text(readme, encoding="utf-8")

    print(f"\n=== Summary ===")
    print(f"  built:   {manifest['n_built']}")
    print(f"  skipped: {manifest['n_skipped_existing']}")
    print(f"  failed:  {manifest['n_failed']}")
    if manifest_entries:
        sizes = [e["file_mb"] for e in manifest_entries]
        offs = [e["alignment_offset_s"] for e in manifest_entries]
        elapsed = [e["elapsed_s"] for e in manifest_entries]
        print(f"  per-wafer file size (MB): "
              f"min={min(sizes):.1f}  median={np.median(sizes):.1f}  max={max(sizes):.1f}")
        print(f"  alignment offset (s):    "
              f"min={min(offs):.2f}  median={np.median(offs):.2f}  max={max(offs):.2f}")
        print(f"  per-wafer elapsed (s):   "
              f"min={min(elapsed):.1f}  median={np.median(elapsed):.1f}  max={max(elapsed):.1f}")
        print(f"  total cache size (MB):   {sum(sizes):.1f}")
    print(f"  manifest: cache/{args.version}/manifest.json")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
