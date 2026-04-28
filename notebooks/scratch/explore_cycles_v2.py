"""v2: normalize times, inspect gas flow channels, verify 100-cycle structure."""
import os, sys
from pathlib import Path
import netCDF4 as nc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"c:\Users\ljse2\Desktop\4-1\종합_설계_프로젝트\BOSCH Plasma-Etching")
DATA = ROOT / "Dataset"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(DATA)


def load_oes(day, wafer):
    with nc.Dataset(day, "r") as ds:
        g = ds.groups[wafer]
        return g.variables["times"][:], g.variables["wavelengths"][:], g.variables["data"][:]


def load_process(key):
    with nc.Dataset("Process_data.nc", "r") as ds:
        g = ds.groups[key]
        return g.variables["times"][:], g.variables["feature"][:], g.variables["data"][:]


t_oes, wl, oes = load_oes("Day_2024_07_02.nc", "Wafer_01")
t_p, feat, proc = load_process("Day_2024_07_02_Wafer_01")

# Print feature names to find gas flows
print("=== All 44 process features ===")
for i, name in enumerate(feat):
    print(f"  [{i:2d}] {name}")

# Normalize times to their own t0
t_oes_rel = t_oes - t_oes[0]
t_p_rel = t_p - t_p[0]
print(f"\nOES duration: {t_oes_rel[-1]:.1f}s  (n={len(t_oes)})")
print(f"Process duration: {t_p_rel[-1]:.1f}s  (n={len(t_p)})")

total = oes.sum(axis=1)
target_wl = 704.0
idx704 = int(np.argmin(np.abs(wl - target_wl)))

# ---- Overview figure: OES total + single wavelength + process gas flows ----
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=False)

axes[0].plot(t_oes_rel, total, lw=0.5, color="#1565C0")
axes[0].set_title("OES total intensity — full wafer (t normalized)")
axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("sum"); axes[0].grid(alpha=0.3)

# Zoom 0-40s
mask_o = t_oes_rel <= 40
axes[1].plot(t_oes_rel[mask_o], total[mask_o], lw=0.8, color="#1565C0", label="total OES")
axes[1].set_title("OES total intensity — first 40 s  (expect 1 s ignition + ~6.5 cycles of 6 s)")
axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("sum"); axes[1].grid(alpha=0.3)
# Expected cycle grid
for cyc in range(7):
    axes[1].axvline(1 + cyc * 6, color="green", alpha=0.3, lw=0.6)
    axes[1].axvline(1 + cyc * 6 + 4.5, color="orange", alpha=0.3, lw=0.6)

axes[2].plot(t_oes_rel[mask_o], oes[mask_o, idx704], lw=0.8, color="#2E7D32")
axes[2].set_title(f"OES single wavelength {wl[idx704]:.1f} nm — first 40 s")
axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("intensity"); axes[2].grid(alpha=0.3)

# Process gas flows — find indices by name
gas_names = [(i, n) for i, n in enumerate(feat) if "Gas" in str(n) and "Flow" in str(n)]
print(f"\nGas flow channels: {gas_names}")

mask_p = t_p_rel <= 40
for i, name in gas_names[:6]:
    y = proc[mask_p, i] if proc.shape[0] == len(t_p) else proc[i, mask_p]
    axes[3].plot(t_p_rel[mask_p], y, lw=1.0, label=str(name).replace("Stat3_Etch_MV_", ""))
axes[3].set_title("Process gas flows — first 40 s (SF6/C4F8 switching visible?)")
axes[3].set_xlabel("time (s)"); axes[3].set_ylabel("flow"); axes[3].legend(fontsize=8)
axes[3].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "04_normalized_overview.png", dpi=120)
plt.close()
print(f"Saved: {OUT / '04_normalized_overview.png'}")

# ---- Cycle-count via find_peaks on OES total ----
fs = 1 / np.mean(np.diff(t_oes_rel))
period_samples = int(round(6.0 * fs))
peaks, _ = find_peaks(total, distance=int(period_samples * 0.7), prominence=total.std() * 0.3)
troughs, _ = find_peaks(-total, distance=int(period_samples * 0.7), prominence=total.std() * 0.3)
print(f"\nfs ≈ {fs:.2f} Hz, period_samples = {period_samples}")
print(f"Peaks found (expect ~100): {len(peaks)}")
print(f"Troughs found (expect ~100): {len(troughs)}")

# Plot peaks on full signal
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t_oes_rel, total, lw=0.4, color="#546E7A", alpha=0.7)
ax.plot(t_oes_rel[peaks], total[peaks], "rx", ms=3, label=f"peaks={len(peaks)}")
ax.plot(t_oes_rel[troughs], total[troughs], "bo", ms=2, label=f"troughs={len(troughs)}", alpha=0.5)
ax.set_title(f"OES total intensity: peak detection")
ax.set_xlabel("time (s)"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "05_peaks_full.png", dpi=120)
plt.close()
print(f"Saved: {OUT / '05_peaks_full.png'}")

# ---- Inter-peak interval histogram ----
if len(peaks) > 1:
    intervals = np.diff(t_oes_rel[peaks])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(intervals, bins=40, color="#1565C0", alpha=0.7)
    ax.axvline(6.0, color="red", lw=2, label="expected 6.0 s")
    ax.set_title(f"Inter-peak interval histogram (median={np.median(intervals):.2f}s)")
    ax.set_xlabel("interval (s)"); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "06_interval_hist.png", dpi=120)
    plt.close()
    print(f"Saved: {OUT / '06_interval_hist.png'}")
    print(f"  median interval = {np.median(intervals):.3f}s  (expect 6.0s)")
    print(f"  intervals: min={intervals.min():.2f}, max={intervals.max():.2f}")

# ---- Write a cycle spectra heatmap (100 cycles × wavelengths), roughly ----
# Assume first peak ≈ end of cycle 1 → chunk signal from ignition=1s in 6s blocks
ignition_end = 1.0
cycle_len = 6.0
oes_by_cycle = []
for c in range(100):
    t0, t1 = ignition_end + c * cycle_len, ignition_end + (c + 1) * cycle_len
    m = (t_oes_rel >= t0) & (t_oes_rel < t1)
    if m.sum() < 50:
        break
    oes_by_cycle.append(oes[m].mean(axis=0))
oes_by_cycle = np.array(oes_by_cycle)
print(f"\nCycle-averaged OES matrix shape: {oes_by_cycle.shape}  (expect ~(100, 3648))")

fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(oes_by_cycle, aspect="auto", origin="lower",
               extent=[wl[0], wl[-1], 0, len(oes_by_cycle)], cmap="viridis")
ax.set_title("Mean OES spectrum per cycle (cycle index vs wavelength)")
ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("cycle index")
plt.colorbar(im, ax=ax, label="intensity")
plt.tight_layout()
plt.savefig(OUT / "07_cycle_heatmap.png", dpi=120)
plt.close()
print(f"Saved: {OUT / '07_cycle_heatmap.png'}")
