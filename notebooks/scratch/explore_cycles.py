"""Verify BOSCH 100-cycle structure in OES & process signals for one wafer."""
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

DAY_FILE = "Day_2024_07_02.nc"
WAFER_GROUP = "Wafer_01"
PROC_KEY = "Day_2024_07_02_Wafer_01"


def load_oes(day_file: str, wafer: str):
    with nc.Dataset(day_file, "r") as ds:
        g = ds.groups[wafer]
        times = g.variables["times"][:]                 # (T,)
        wavelengths = g.variables["wavelengths"][:]     # (W,)
        data = g.variables["data"][:]                   # (T, W) or (W, T)
    return times, wavelengths, data


def load_process(proc_file: str, key: str):
    with nc.Dataset(proc_file, "r") as ds:
        g = ds.groups[key]
        times = g.variables["times"][:]
        feat = g.variables["feature"][:]
        data = g.variables["data"][:]
    return times, feat, data


print("=== OES loading ===")
t_oes, wl, oes = load_oes(DAY_FILE, WAFER_GROUP)
print(f"times shape={t_oes.shape}, dtype={t_oes.dtype}  range=[{t_oes.min():.3f}, {t_oes.max():.3f}]")
print(f"wavelengths shape={wl.shape}  range=[{wl.min():.2f}, {wl.max():.2f}] nm")
print(f"oes data shape={oes.shape}, dtype={oes.dtype}")
print(f"oes mean/std/min/max = {oes.mean():.2f}/{oes.std():.2f}/{oes.min():.2f}/{oes.max():.2f}")

# Sampling rate
if len(t_oes) > 1:
    dt = np.diff(t_oes)
    print(f"OES dt: mean={dt.mean():.4f}s, std={dt.std():.4e}s  → sampling ≈ {1/dt.mean():.1f} Hz")

print("\n=== Process loading ===")
t_p, feat, proc = load_process("Process_data.nc", PROC_KEY)
print(f"times shape={t_p.shape}  range=[{t_p.min():.3f}, {t_p.max():.3f}]")
print(f"feature shape={feat.shape}, dtype={feat.dtype}")
print(f"  first 5 features: {feat[:5]}")
print(f"proc data shape={proc.shape}")
if len(t_p) > 1:
    dt_p = np.diff(t_p)
    print(f"Process dt: mean={dt_p.mean():.4f}s  → sampling ≈ {1/dt_p.mean():.1f} Hz")

# ---- Find cycle structure: look at total OES intensity vs time ----
# Assumed orientation (T, W) — if wrong, flip.
if oes.shape[0] == len(t_oes):
    total = oes.sum(axis=1)      # total intensity per timestep
    oes_tw = oes                 # (T, W)
else:
    total = oes.sum(axis=0)
    oes_tw = oes.T

print(f"\nTotal-intensity signal shape: {total.shape}")

# ---- Plot 1: total OES intensity over time (full wafer) ----
fig, axes = plt.subplots(3, 1, figsize=(14, 9))
axes[0].plot(t_oes, total, lw=0.6, color="#1565C0")
axes[0].set_title("Total OES intensity vs time — full wafer (should show ignition + 100 cycles)")
axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("sum intensity")
axes[0].grid(alpha=0.3)

# Zoom into first 30s to see individual cycles
mask = t_oes <= 30
axes[1].plot(t_oes[mask], total[mask], lw=0.8, color="#C62828")
axes[1].set_title("Zoom: first 30 s (ignition ~1s, then SF6/C4F8 cycles of 6s)")
axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("sum intensity")
axes[1].grid(alpha=0.3)
# mark expected cycle boundaries
for cyc in range(5):
    ign = 1.0
    axes[1].axvline(ign + cyc * 6.0, color="green", alpha=0.4, lw=0.8)
    axes[1].axvline(ign + cyc * 6.0 + 4.5, color="orange", alpha=0.4, lw=0.8)
axes[1].axvline(1.0, color="k", alpha=0.6, lw=1.2, linestyle="--", label="ignition end")
axes[1].legend()

# FFT / autocorrelation hint — plot intensity at a specific wavelength
# (704 nm is SF6-related; find closest wavelength index)
target_wl = 704.0
idx = int(np.argmin(np.abs(wl - target_wl)))
print(f"  closest wavelength to {target_wl} nm: {wl[idx]:.2f} nm (idx={idx})")
axes[2].plot(t_oes[mask], oes_tw[mask, idx], lw=0.8, color="#2E7D32")
axes[2].set_title(f"Single wavelength {wl[idx]:.1f} nm, first 30 s")
axes[2].set_xlabel("time (s)"); axes[2].set_ylabel("intensity")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "01_oes_cycle_overview.png", dpi=120)
plt.close()
print(f"\nSaved: {OUT / '01_oes_cycle_overview.png'}")

# ---- Plot 2: Process parameters over time (look for gas flow switching) ----
fig, ax = plt.subplots(figsize=(14, 6))
# Plot first 8 features to spot any binary switching (SF6/C4F8 mass flow)
for i in range(min(8, proc.shape[0] if proc.shape[0] < proc.shape[1] else proc.shape[1])):
    if proc.shape[0] == 44:  # (feature, time)
        y = proc[i, :]
    else:
        y = proc[:, i]
    # Normalize for display
    rng = y.max() - y.min()
    yn = (y - y.min()) / (rng if rng > 0 else 1)
    ax.plot(t_p, yn + i * 1.1, lw=0.6, label=f"feat_{i}")
ax.set_xlim(0, 30)
ax.set_title("Process features (normalized, offset) — first 30 s")
ax.set_xlabel("time (s)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "02_process_first30s.png", dpi=120)
plt.close()
print(f"Saved: {OUT / '02_process_first30s.png'}")

# ---- Summary: how many peaks are there in total intensity? ----
# Simple: threshold-based cycle count using the derivative
from scipy.signal import find_peaks  # type: ignore
# Smooth & find peaks with period ≈ 6s (at ~24.5 Hz → ~147 samples)
fs = 1 / np.mean(np.diff(t_oes))
expected_period_samples = int(round(6.0 * fs))
peaks, _ = find_peaks(total, distance=int(expected_period_samples * 0.7))
print(f"\nRough cycle-peak count (expected ~100): {len(peaks)}")
print(f"Expected period: {expected_period_samples} samples (fs≈{fs:.1f} Hz)")

# Plot: peaks on full signal
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t_oes, total, lw=0.5, color="#546E7A")
ax.plot(t_oes[peaks], total[peaks], "rx", ms=4, label=f"peaks={len(peaks)}")
ax.set_title(f"Total OES intensity peaks (expected ~100 cycles)")
ax.set_xlabel("time (s)"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "03_oes_peaks.png", dpi=120)
plt.close()
print(f"Saved: {OUT / '03_oes_peaks.png'}")
