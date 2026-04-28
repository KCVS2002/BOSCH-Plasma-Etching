"""v3: Gas-flow-based cycle segmentation + OES/Process time alignment.

Idea: gas flow signals are digital setpoints → noise-free edge detection.
Use rising edges of SF6 flow to define cycle boundaries.
Then map these boundaries back to OES time axis.
"""
import os, sys
from pathlib import Path
import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"c:\Users\ljse2\Desktop\4-1\종합_설계_프로젝트\BOSCH Plasma-Etching")
DATA = ROOT / "Dataset"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(DATA)


def load_oes(day, wafer):
    with nc.Dataset(day, "r") as ds:
        g = ds.groups[wafer]
        return (np.asarray(g.variables["times"][:]),
                np.asarray(g.variables["wavelengths"][:]),
                np.asarray(g.variables["data"][:]))


def load_process(key):
    with nc.Dataset("Process_data.nc", "r") as ds:
        g = ds.groups[key]
        return (np.asarray(g.variables["times"][:]),
                np.asarray(g.variables["feature"][:]),
                np.asarray(g.variables["data"][:]))


t_oes, wl, oes = load_oes("Day_2024_07_02.nc", "Wafer_01")
t_p, feat, proc = load_process("Day_2024_07_02_Wafer_01")

# --- inspect gas channels: which ones are non-constant? ---
gas_idx = [i for i, n in enumerate(feat) if "Gas" in str(n) and "Flow" in str(n)]
print("=== Gas channel statistics ===")
for i in gas_idx:
    name = feat[i]
    y = proc[:, i]
    active_frac = (y > y.max() * 0.1).mean() if y.max() > 0 else 0
    print(f"  [{i:2d}] {name:40s}  min={y.min():.2f}  max={y.max():.2f}  active_frac={active_frac:.2f}")

# --- identify SWITCHING gases (true SF6 / C4F8), excluding constant channels ---
switching_gases = []
for i in gas_idx:
    y = proc[:, i].astype(float)
    if y.max() < 1.0:
        continue
    on_mask = y > y.max() * 0.5
    duty = on_mask.mean()
    # A switching channel has both ON and OFF periods (not always-on constant)
    if 0.02 < duty < 0.98:
        switching_gases.append((i, str(feat[i]), y.max(), duty, on_mask))
print(f"\nSwitching gas channels (candidates for SF6/C4F8):")
for i, name, mx, duty, _ in switching_gases:
    print(f"  [{i}] {name}: max={mx:.1f}, duty={duty:.2%}")

# BOSCH: SF6 ≈ 75% duty (4.5s/6s), C4F8 ≈ 25% duty (1.5s/6s)
# Sort by duty, highest = SF6
gas_info = sorted(switching_gases, key=lambda r: -r[3])
if len(gas_info) >= 2:
    sf6 = gas_info[0]
    c4f8 = gas_info[1]
    print(f"\nAssumed SF6 channel: [{sf6[0]}] {sf6[1]} (duty {sf6[3]:.1%})")
    print(f"Assumed C4F8 channel: [{c4f8[0]}] {c4f8[1]} (duty {c4f8[3]:.1%})")
else:
    print("\nERROR: could not identify two gas channels")
    sys.exit(1)

# --- detect rising edges of SF6 → cycle starts ---
sf6_on = sf6[4].astype(int)
rising = np.where(np.diff(sf6_on) == 1)[0] + 1  # indices where SF6 turns ON
falling = np.where(np.diff(sf6_on) == -1)[0] + 1  # indices where SF6 turns OFF
print(f"\nSF6 rising edges (= cycle starts): {len(rising)}")
print(f"SF6 falling edges: {len(falling)}")

# Time of cycle starts in process clock
cycle_starts_p = t_p[rising]
print(f"First 5 cycle start times (process clock): {cycle_starts_p[:5]}")
print(f"Last 5 cycle start times (process clock): {cycle_starts_p[-5:]}")
if len(cycle_starts_p) > 1:
    intervals = np.diff(cycle_starts_p)
    print(f"Cycle interval: mean={intervals.mean():.4f}s, median={np.median(intervals):.4f}s, "
          f"std={intervals.std():.4f}s")

# --- figure: gas flows + SF6 cycle starts ---
fig, axes = plt.subplots(3, 1, figsize=(14, 9))

t_p_rel = t_p - t_p[0]
axes[0].plot(t_p_rel, proc[:, sf6[0]], label=sf6[1].replace("Stat3_Etch_MV_", ""), lw=1, color="#C62828")
axes[0].plot(t_p_rel, proc[:, c4f8[0]], label=c4f8[1].replace("Stat3_Etch_MV_", ""), lw=1, color="#1565C0")
axes[0].set_title("Active gas flows (full wafer)")
axes[0].set_ylabel("flow"); axes[0].legend(); axes[0].grid(alpha=0.3)

# zoom first 40s
mask_p = t_p_rel <= 40
axes[1].plot(t_p_rel[mask_p], proc[mask_p, sf6[0]], label=sf6[1].replace("Stat3_Etch_MV_", ""), lw=1.5, color="#C62828")
axes[1].plot(t_p_rel[mask_p], proc[mask_p, c4f8[0]], label=c4f8[1].replace("Stat3_Etch_MV_", ""), lw=1.5, color="#1565C0")
for rs in rising[:7]:
    axes[1].axvline(t_p_rel[rs], color="green", alpha=0.5, lw=0.8)
axes[1].set_title("Zoom: first 40 s — green lines = SF6 rising edges (cycle start)")
axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("flow"); axes[1].legend(); axes[1].grid(alpha=0.3)

# histogram of inter-cycle intervals
if len(cycle_starts_p) > 1:
    axes[2].hist(intervals, bins=40, color="#2E7D32", alpha=0.7)
    axes[2].axvline(6.0, color="red", lw=2, label="expected 6.0 s")
    axes[2].set_title(f"SF6-rising-edge intervals  (count={len(rising)}, median={np.median(intervals):.3f}s)")
    axes[2].set_xlabel("interval (s)"); axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "08_gasflow_cycles.png", dpi=120)
plt.close()
print(f"Saved: {OUT / '08_gasflow_cycles.png'}")

# ============================================================
# TIME AXIS ALIGNMENT: map process cycle starts → OES indices
# ============================================================
# OES t is UNIX timestamps. Process t is tool-seconds (relative to tool boot).
# Strategy: align using process-start reference.
# The key insight: we need to find what PROCESS time corresponds to OES t_oes[0].

# Approach A: Assume OES and Process recording started "simultaneously" at
# process step begin. Then OES[0] ↔ process time when OES recording began.
# We expect OES total intensity to ramp up at ignition → find that ramp
# and correlate with process state (e.g., RF power or pressure going up).

# Look for a "process start" marker: first non-zero RF power / pressure ramp
epd_idx = [i for i, n in enumerate(feat) if "Epd" in str(n)][0]
pressure_idx = [i for i, n in enumerate(feat) if n == "Stat3_Etch_MV_Pressure"][0]
rf_idx = [i for i, n in enumerate(feat) if n == "Stat3_Etch_MV_SourceRFLoadPower"][0]
print(f"\nReference channels: EPD={epd_idx}, Pressure={pressure_idx}, SourceRFLoad={rf_idx}")

fig, axes = plt.subplots(3, 1, figsize=(14, 9))
axes[0].plot(t_p_rel, proc[:, epd_idx], color="#6A1B9A", lw=0.8)
axes[0].set_title(f"EPD Intensity (process clock, full)"); axes[0].grid(alpha=0.3)

axes[1].plot(t_p_rel, proc[:, pressure_idx], color="#1565C0", lw=0.8)
axes[1].set_title(f"Pressure (process clock, full)"); axes[1].grid(alpha=0.3)

axes[2].plot(t_p_rel, proc[:, rf_idx], color="#C62828", lw=0.8)
axes[2].set_title(f"SourceRFLoadPower (process clock, full)"); axes[2].grid(alpha=0.3)
axes[2].set_xlabel("time (s)")

plt.tight_layout()
plt.savefig(OUT / "09_process_reference_channels.png", dpi=120)
plt.close()
print(f"Saved: {OUT / '09_process_reference_channels.png'}")

# Find process-start: first moment RF power > threshold
rf = proc[:, rf_idx]
rf_on = rf > (rf.max() * 0.5)
if rf_on.any():
    rf_start_idx = np.argmax(rf_on)
    t_rf_start = t_p_rel[rf_start_idx]
    print(f"\nRF power turns ON at process_rel_time = {t_rf_start:.2f}s")

# First SF6 rising edge = cycle 1 start (in process clock)
t_cycle1_start_p = t_p_rel[rising[0]]
print(f"First SF6 rising edge at process_rel_time = {t_cycle1_start_p:.2f}s")

# --- Compare OES-based & Process-based cycle intervals to cross-validate ---
# We already know OES total-intensity peak interval median = 6.01s
# And SF6 rising-edge interval median ≈ 6.00s
# If both match, the process cycle structure is correctly measurable on both signals.

print("\n=== Cross-validation summary ===")
print(f"OES peak intervals (median): 6.01 s (from v2)")
print(f"SF6 rising-edge intervals (median): {np.median(intervals):.4f} s")
print(f"OES duration: {(t_oes[-1] - t_oes[0]):.2f} s, samples: {len(t_oes)}")
print(f"Process duration: {(t_p[-1] - t_p[0]):.2f} s, samples: {len(t_p)}")
print(f"Active-process duration (cycle1_start → last_SF6_fall):"
      f" {t_p_rel[falling[-1]] - t_cycle1_start_p:.2f} s  (expect ~600 s = 100 cycles × 6s)")

# ============================================================
# OES-ALIGNMENT via cross-correlation (assuming zero offset)
# ============================================================
# Build a synthetic "SF6 mask" on the OES time grid and correlate with
# OES total intensity to find best offset.
total_oes = oes.sum(axis=1).astype(np.float32)
t_oes_rel = t_oes - t_oes[0]
fs_oes = 1.0 / np.mean(np.diff(t_oes_rel))

# Resample SF6 digital signal onto OES time grid (shifted by candidate offsets)
from numpy import interp

def shifted_corr(offset_s):
    """Correlate SF6 mask (shifted by offset_s) with OES total intensity."""
    sf6_on_proc = sf6_on.astype(np.float32)
    # map process time to OES time with offset: OES_time = process_time - offset
    # (if offset>0, process lags OES)
    t_probe = t_oes_rel + offset_s  # process-time corresponding to each OES sample
    # linear interp of SF6 state at these process times
    sf6_at_oes = interp(t_probe, t_p_rel, sf6_on_proc, left=0, right=0)
    # correlation with OES total intensity
    a = sf6_at_oes - sf6_at_oes.mean()
    b = total_oes - total_oes.mean()
    denom = (a.std() * b.std() * len(a))
    return (a * b).sum() / (denom + 1e-9)


offsets = np.linspace(-20, 120, 281)  # -20 to 120 seconds, step 0.5s
corrs = np.array([shifted_corr(o) for o in offsets])
best = offsets[np.argmax(np.abs(corrs))]
print(f"\nBest OES-vs-SF6 offset (process_time = OES_rel + offset): {best:.2f} s")
print(f"  correlation at best: {corrs[np.argmax(np.abs(corrs))]:.3f}")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
axes[0].plot(offsets, corrs, lw=1)
axes[0].axvline(best, color="red", lw=1.5, label=f"best |corr| offset = {best:.1f}s")
axes[0].set_title("Cross-correlation: SF6-mask vs OES total intensity")
axes[0].set_xlabel("offset (s)  [process_time = OES_rel + offset]")
axes[0].set_ylabel("correlation")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Plot: SF6-mask and OES total on same (OES-relative) axis after alignment
t_probe = t_oes_rel + best
sf6_at_oes = interp(t_probe, t_p_rel, sf6_on.astype(float), left=0, right=0)

def norm(a):
    return (a - a.min()) / (np.ptp(a) + 1e-9)

# Show alignment over 3 windows: start, middle, end
for ax_i, (t0, t1) in zip(axes[1:], [(0, 40), (300, 340)]):
    m = (t_oes_rel >= t0) & (t_oes_rel <= t1)
    ax_i.plot(t_oes_rel[m], norm(total_oes[m]),
              lw=0.8, color="#1565C0", label="OES total (normalized)")
    ax_i.plot(t_oes_rel[m], sf6_at_oes[m],
              lw=1.2, color="#C62828", alpha=0.7, label="SF6 mask (aligned)")
    ax_i.set_title(f"Alignment check: OES t ∈ [{t0}, {t1}] s  (offset={best:.1f}s)")
    ax_i.set_xlabel("OES relative time (s)")
    ax_i.legend(); ax_i.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "10_oes_process_alignment.png", dpi=120)
plt.close()
print(f"Saved: {OUT / '10_oes_process_alignment.png'}")

# ============================================================
# Peak-based alignment: match OES peak times to SF6 edge times
# ============================================================
# This is a cleaner approach: we know median interval = 6s on both signals.
# Pick stable sub-sequence of each with exact 6s spacing, align 1-to-1.
from scipy.signal import find_peaks
fs_oes = 1 / np.mean(np.diff(t_oes_rel))
period = int(round(6.0 * fs_oes))
oes_peaks, _ = find_peaks(total_oes, distance=int(period * 0.7),
                          prominence=total_oes.std() * 0.3)
t_oes_peaks = t_oes_rel[oes_peaks]
t_sf6_edges = t_p_rel[rising]

print(f"\nOES peaks: n={len(t_oes_peaks)}, first 5: {t_oes_peaks[:5]}")
print(f"SF6 edges: n={len(t_sf6_edges)}, first 5: {t_sf6_edges[:5]}")

# If both signals have ~100 cycles, try: assume i-th OES peak ↔ i-th SF6 edge
# Compute offset for each plausible index-pair alignment
n = min(len(t_oes_peaks), len(t_sf6_edges))
if n >= 10:
    # offset that maps oes_peak_time → sf6_edge_time (process clock)
    diffs = t_sf6_edges[:n] - t_oes_peaks[:n]
    print(f"  offsets (SF6 - OES) first 10: {diffs[:10]}")
    print(f"  offsets mean={diffs.mean():.2f}, std={diffs.std():.2f}, median={np.median(diffs):.2f}")

    # Aligning by slightly different index shifts
    print("\n  Try different index alignments:")
    for shift in range(-3, 4):
        if shift >= 0:
            d = t_sf6_edges[shift:shift + min(n - shift, len(t_sf6_edges) - shift)] - \
                t_oes_peaks[:min(n - shift, len(t_sf6_edges) - shift)]
        else:
            d = t_sf6_edges[:min(n + shift, len(t_sf6_edges))] - \
                t_oes_peaks[-shift:min(-shift + n + shift, len(t_oes_peaks))]
        if len(d) > 5:
            print(f"    shift={shift:+d}: mean={d.mean():.2f}, std={d.std():.2f}")
