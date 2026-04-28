"""Data loader for BOSCH Plasma-Etching dataset.

Reads NetCDF (OES, Process) and CSV (measurements), and segments the process
time-series into 100 BOSCH cycles using the SF6 gas flow as the definitional
signal.

Design notes (from Step-1 EDA, 2026-04-17):
  - OES is stored per-day in `Dataset/Day_YYYY_MM_DD.nc` under groups
    `Wafer_01 … Wafer_10` with vars `times`, `wavelengths`, `data`.
    OES `times` are UNIX timestamps (float64 seconds).
  - Process is stored in `Dataset/Process_data.nc` under groups
    `Day_YYYY_MM_DD_Wafer_NN` with vars `times`, `feature` (44 names),
    `data` (time, feature). `times` are tool-relative seconds.
  - Measurement targets live in `Si_Oxide_etch_89_points.csv` keyed by
    `experiment_key = 'YYYY-MM-DD_LL'` (lot label), `wafer_number`.
  - The two switching gases are Gas5Flow (SF6, duty ≈ 72%) and Gas4Flow
    (C4F8, duty ≈ 28%). Rising edges of SF6 define cycle starts.
  - Windows note: netCDF4 can't open files via non-ASCII paths on Windows,
    so `open_nc` first chdir's to the dataset directory and opens by basename.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import netCDF4 as nc
import numpy as np
import pandas as pd

DATASET_DIR = Path(__file__).resolve().parents[2] / "Dataset"


@contextmanager
def _cwd(path: Path) -> Iterator[None]:
    """chdir(path) on enter, restore on exit — works around netCDF4's
    inability to open non-ASCII paths on Windows."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


# ----------------------------------------------------------------------
# Keys & wafer enumeration
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class WaferKey:
    """Identifies one wafer across all three data sources."""
    day: str                  # e.g. "2024_07_02"
    wafer_num: int            # 1..10
    lot_number: int           # 1..10 (chronological order of days)
    day_file: str             # e.g. "Day_2024_07_02.nc"
    oes_group: str            # e.g. "Wafer_01"
    process_group: str        # e.g. "Day_2024_07_02_Wafer_01"
    experiment_key: str       # e.g. "2024-07-02_01"  (matches measurement CSV)


# Day → Lot mapping. Verified against Lot_status.xlsx (Step 2.2, 2026-04-24):
# the 10 lot dates are processed in chronological order, Lot 1 = first day.
DAY_TO_LOT: dict[str, int] = {
    "2024_07_02": 1,
    "2024_07_05": 2,
    "2024_07_09": 3,
    "2024_07_11": 4,
    "2024_07_19": 5,
    "2024_08_01": 6,
    "2024_08_05": 7,
    "2024_08_07": 8,
    "2024_08_21": 9,
    "2024_08_22": 10,
}


def wafer_key_to_experiment_key(day: str, wafer_num: int) -> str:
    """Convert Day_YYYY_MM_DD + wafer_num → experiment_key 'YYYY-MM-DD_WW'.

    The '_WW' suffix in experiment_key is the WAFER number (verified Step 2.2).
    Lot number is a separate column in the measurement CSV.
    """
    return f"{day.replace('_', '-')}_{wafer_num:02d}"


def list_wafers(dataset_dir: Path = DATASET_DIR) -> list[WaferKey]:
    """Enumerate all wafers found across the 10 Day_*.nc files."""
    out: list[WaferKey] = []
    with _cwd(dataset_dir):
        for day_file in sorted(Path(".").glob("Day_*.nc")):
            # Day_2024_07_02.nc → "2024_07_02"
            day = day_file.stem.removeprefix("Day_")
            with nc.Dataset(day_file.name, "r") as ds:
                for gname in ds.groups:
                    # Expect "Wafer_01"..."Wafer_10"
                    if not gname.startswith("Wafer_"):
                        continue
                    wnum = int(gname.split("_")[1])
                    out.append(WaferKey(
                        day=day,
                        wafer_num=wnum,
                        lot_number=DAY_TO_LOT[day],
                        day_file=day_file.name,
                        oes_group=gname,
                        process_group=f"Day_{day}_{gname}",
                        experiment_key=wafer_key_to_experiment_key(day, wnum),
                    ))
    return out


# ----------------------------------------------------------------------
# NetCDF loaders
# ----------------------------------------------------------------------

@dataclass
class OESWafer:
    times: np.ndarray        # (T,) UNIX seconds
    wavelengths: np.ndarray  # (W,) nm
    data: np.ndarray         # (T, W) uint16 intensity
    t_rel: np.ndarray        # (T,) seconds from wafer's first sample

    @property
    def duration(self) -> float:
        return float(self.t_rel[-1] - self.t_rel[0])

    @property
    def fs(self) -> float:
        """Sampling frequency (Hz)."""
        return float(1.0 / np.mean(np.diff(self.t_rel)))


@dataclass
class ProcessWafer:
    times: np.ndarray        # (T,) tool-relative seconds
    features: np.ndarray     # (F,) str names
    data: np.ndarray         # (T, F) float
    t_rel: np.ndarray        # (T,) seconds from first sample

    def column(self, name: str) -> np.ndarray:
        """Column by feature name (full or suffix after 'Stat3_Etch_MV_')."""
        names = np.asarray([str(n) for n in self.features])
        idx = np.where(names == name)[0]
        if len(idx) == 0:
            # allow suffix match
            suffixes = np.asarray([n.removeprefix("Stat3_Etch_MV_") for n in names])
            idx = np.where(suffixes == name)[0]
        if len(idx) == 0:
            raise KeyError(f"feature {name!r} not found. Available: {list(names)}")
        return self.data[:, idx[0]]


def load_oes_wafer(wk: WaferKey, dataset_dir: Path = DATASET_DIR) -> OESWafer:
    with _cwd(dataset_dir):
        with nc.Dataset(wk.day_file, "r") as ds:
            g = ds.groups[wk.oes_group]
            times = np.asarray(g.variables["times"][:])
            wl = np.asarray(g.variables["wavelengths"][:])
            data = np.asarray(g.variables["data"][:])  # (T, W)
    return OESWafer(times=times, wavelengths=wl, data=data, t_rel=times - times[0])


def load_process_wafer(wk: WaferKey, dataset_dir: Path = DATASET_DIR) -> ProcessWafer:
    with _cwd(dataset_dir):
        with nc.Dataset("Process_data.nc", "r") as ds:
            g = ds.groups[wk.process_group]
            times = np.asarray(g.variables["times"][:])
            feat = np.asarray(g.variables["feature"][:])
            data = np.asarray(g.variables["data"][:])  # (T, F)
    return ProcessWafer(times=times, features=feat, data=data, t_rel=times - times[0])


# ----------------------------------------------------------------------
# Measurement CSV
# ----------------------------------------------------------------------

def load_measurements_89(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Load the 89-points measurement table (7832 × 11).

    Columns: experiment_key, lot_number, wafer_number, X, Y,
             preox_thickness, postox_thickness, postox_thickness_nan,
             stepheight, oxide_etch, si_etch.
    """
    return pd.read_csv(dataset_dir / "Si_Oxide_etch_89_points.csv")


# ----------------------------------------------------------------------
# Cycle segmentation
# ----------------------------------------------------------------------

# Empirically identified from Step-1 EDA:
SF6_FEATURE = "Stat3_Etch_MV_Gas5Flow"   # duty ≈ 72% (4.5s ON / 6s cycle)
C4F8_FEATURE = "Stat3_Etch_MV_Gas4Flow"  # duty ≈ 28% (1.5s ON / 6s cycle)
EXPECTED_CYCLES = 100
CYCLE_PERIOD_S = 6.0


@dataclass
class CycleSegmentation:
    """Cycle boundaries derived from process-clock signals."""
    sf6_rising_idx: np.ndarray      # (N,) indices in ProcessWafer arrays where SF6 turns ON
    sf6_falling_idx: np.ndarray     # (N,) indices where SF6 turns OFF
    cycle_starts_s: np.ndarray      # (N,) process-clock seconds of each cycle start
    cycle_ends_s: np.ndarray        # (N,) process-clock seconds of each cycle end
    n_cycles: int

    @property
    def intervals_s(self) -> np.ndarray:
        return np.diff(self.cycle_starts_s)


def segment_cycles_by_sf6(pw: ProcessWafer, threshold_frac: float = 0.5) -> CycleSegmentation:
    """Detect BOSCH cycle boundaries from SF6 gas flow rising edges.

    Returns rising/falling edge indices (into pw.times) and their
    corresponding process-clock times. Rising-to-rising interval should
    be ≈ 6.0 s for valid cycles.
    """
    sf6 = pw.column(SF6_FEATURE).astype(np.float32)
    mx = float(sf6.max())
    if mx < 1.0:
        raise ValueError(f"SF6 signal appears constant/zero (max={mx})")
    on = (sf6 > mx * threshold_frac).astype(np.int8)
    diff = np.diff(on)
    rising = np.flatnonzero(diff == 1) + 1     # first index with SF6 ON
    falling = np.flatnonzero(diff == -1) + 1   # first index with SF6 OFF
    return CycleSegmentation(
        sf6_rising_idx=rising,
        sf6_falling_idx=falling,
        cycle_starts_s=pw.t_rel[rising] if len(rising) else np.array([]),
        cycle_ends_s=pw.t_rel[falling] if len(falling) else np.array([]),
        n_cycles=len(rising),
    )


# Empirical tolerance for "valid" cycle interval. BOSCH cycle is 6.0s;
# real data jitters ~±0.2s. 5.5–6.5 gives comfortable margin.
VALID_INTERVAL_MIN_S = 5.5
VALID_INTERVAL_MAX_S = 6.5
N_CYCLES_TARGET = 100


def trim_to_100_cycles(
    seg: CycleSegmentation,
    n_target: int = N_CYCLES_TARGET,
) -> CycleSegmentation:
    """Trim a segmentation to exactly `n_target` BOSCH cycles (end-anchored).

    Empirically (Step-2 verification, 2026-04-24):
      - 10/96 wafers have exactly 100 cycles (clean)
      - 83/96 have 101 cycles (one ignition spike at start, then 4.6s glitch)
      - 3/96 have 102 cycles (two ignition spikes at start)
      - All spurious edges are at the START; tails are always clean.
      - "Drop-and-take-100-clean" is impossible because dropping the bad
        head edges leaves only 98–99 clean edges, not 100. The 4.6 s
        glitch sits between two real edges, so it can't be cleanly
        excised without losing a cycle.

    End-anchored rule:
      Take the LAST `n_target` rising edges. Tail intervals are always
      clean ~6 s; any irregularity is concentrated in cycle 1, which the
      downstream encoder can flag/handle as the ignition-affected cycle.
    """
    t = seg.cycle_starts_s
    if len(t) < n_target:
        raise ValueError(f"only {len(t)} edges, need {n_target}")

    rising = seg.sf6_rising_idx[-n_target:]
    # Falling edges that fall inside the new rising-edge span (one falling
    # per rising; the tail-most falling may sit after the last rising).
    fmask = (seg.sf6_falling_idx >= rising[0]) & (
        seg.sf6_falling_idx <= seg.sf6_falling_idx.max()
    )
    falling = seg.sf6_falling_idx[fmask][:n_target]

    return CycleSegmentation(
        sf6_rising_idx=rising,
        sf6_falling_idx=falling,
        cycle_starts_s=t[-n_target:],
        cycle_ends_s=seg.cycle_ends_s[fmask][:n_target],
        n_cycles=n_target,
    )


# ----------------------------------------------------------------------
# OES ↔ Process time alignment
# ----------------------------------------------------------------------
# OES `times` are UNIX timestamps; Process `times` are tool-relative seconds.
# Their zero-points are unrelated, so we estimate an offset such that
#   process_time = oes.t_rel + alignment_offset_s
# by cross-correlating the SF6 ON-mask (process clock) against OES total
# intensity. The two signals have the same 6-second cycle period, so the
# correlation peak is sharp.

def align_oes_to_process(
    oes: OESWafer,
    proc: ProcessWafer,
    coarse_step_s: float = 0.5,
    refine_step_s: float = 0.05,
    search_range_s: tuple[float, float] = (-20.0, 120.0),
    threshold_frac: float = 0.5,
) -> float:
    """Return offset such that `process_time ≈ oes.t_rel + offset`.

    Uses two-stage cross-correlation (coarse grid → refined grid around best).
    The coarse grid covers the typical ignition-delay range; the refine stage
    pins it to ~5 cs precision, well below the 6 s cycle period.
    """
    sf6 = proc.column(SF6_FEATURE).astype(np.float32)
    sf6_max = float(sf6.max())
    if sf6_max < 1.0:
        raise ValueError(f"SF6 signal appears constant/zero (max={sf6_max})")
    sf6_on = (sf6 > sf6_max * threshold_frac).astype(np.float32)

    total_oes = oes.data.sum(axis=1).astype(np.float32)
    total_oes -= total_oes.mean()
    total_oes_norm = total_oes.std()
    if total_oes_norm < 1e-9:
        raise ValueError("OES total intensity is flat — alignment ill-defined")
    total_oes /= total_oes_norm

    def corr(offset_s: float) -> float:
        t_probe = oes.t_rel + offset_s
        sf6_at_oes = np.interp(t_probe, proc.t_rel, sf6_on, left=0.0, right=0.0)
        b = sf6_at_oes - sf6_at_oes.mean()
        b_std = b.std()
        if b_std < 1e-9:
            return 0.0
        b /= b_std
        return float((total_oes * b).mean())

    lo, hi = search_range_s
    coarse = np.arange(lo, hi + 0.5 * coarse_step_s, coarse_step_s)
    coarse_corrs = np.array([corr(o) for o in coarse])
    coarse_best = float(coarse[int(np.argmax(coarse_corrs))])

    refine = np.arange(
        coarse_best - coarse_step_s,
        coarse_best + coarse_step_s + 0.5 * refine_step_s,
        refine_step_s,
    )
    refine_corrs = np.array([corr(o) for o in refine])
    return float(refine[int(np.argmax(refine_corrs))])


# ----------------------------------------------------------------------
# Cycle index slicing
# ----------------------------------------------------------------------

def cycle_indices_proc(
    seg: CycleSegmentation,
    n_total: int,
    cycle_period_s: float = CYCLE_PERIOD_S,
) -> tuple[np.ndarray, np.ndarray]:
    """For each cycle, return (start_idx, end_idx_exclusive) into Process arrays.

    Cycle i spans from rising edge i to rising edge i+1. The last cycle uses
    the median rising-to-rising sample interval to extrapolate its end.
    """
    starts = np.asarray(seg.sf6_rising_idx, dtype=np.int32)
    if len(starts) < 2:
        raise ValueError(f"need ≥2 cycle starts, got {len(starts)}")
    median_iv = int(round(np.median(np.diff(starts))))
    last_end = min(int(starts[-1]) + median_iv, n_total)
    ends = np.concatenate([starts[1:], np.array([last_end], dtype=np.int32)])
    return starts, ends.astype(np.int32)


def cycle_indices_oes(
    oes: OESWafer,
    cycle_starts_proc_s: np.ndarray,
    offset_s: float,
    cycle_period_s: float = CYCLE_PERIOD_S,
) -> tuple[np.ndarray, np.ndarray]:
    """For each cycle, return (start_idx, end_idx_exclusive) into OES arrays.

    Cycle i covers process-clock interval [cycle_starts_proc_s[i],
    cycle_starts_proc_s[i+1]). The last cycle's end is extrapolated as
    cycle_starts_proc_s[-1] + cycle_period_s.
    """
    cs = np.asarray(cycle_starts_proc_s, dtype=np.float64)
    next_start = float(cs[-1]) + float(cycle_period_s)
    boundaries_proc = np.concatenate([cs, [next_start]])  # (n+1,)
    boundaries_oes_t = boundaries_proc - float(offset_s)  # in oes.t_rel coords
    edges_idx = np.searchsorted(oes.t_rel, boundaries_oes_t, side="left")
    starts = edges_idx[:-1].astype(np.int32)
    ends = edges_idx[1:].astype(np.int32)
    # Clamp to array bounds (last cycle end may sit past last sample)
    n = int(oes.t_rel.shape[0])
    ends = np.minimum(ends, n).astype(np.int32)
    starts = np.minimum(starts, n).astype(np.int32)
    return starts, ends


# ----------------------------------------------------------------------
# Smoke test when run directly
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    wafers = list_wafers()
    print(f"Found {len(wafers)} wafers across {len({w.day for w in wafers})} days")
    print(f"First: {wafers[0]}")
    print(f"Last:  {wafers[-1]}")

    wk = wafers[0]
    oes = load_oes_wafer(wk)
    proc = load_process_wafer(wk)
    print(f"\n{wk.oes_group}: OES {oes.data.shape} @ {oes.fs:.2f} Hz, dur {oes.duration:.1f}s")
    print(f"{wk.process_group}: Process {proc.data.shape}, dur {proc.t_rel[-1]:.1f}s")

    seg = segment_cycles_by_sf6(proc)
    print(f"\nCycles detected: {seg.n_cycles} (expected ≈ {EXPECTED_CYCLES})")
    if seg.n_cycles > 1:
        iv = seg.intervals_s
        print(f"Interval mean={iv.mean():.3f}s, median={np.median(iv):.3f}s, std={iv.std():.3f}s")

    meas = load_measurements_89()
    print(f"\nMeasurements: {meas.shape}, wafers with measurements: "
          f"{meas.groupby(['experiment_key', 'wafer_number']).ngroups}")
    print(f"Unique experiment_keys (first 5): {sorted(meas.experiment_key.unique())[:5]}")
