"""Fold 4 diagnosis: compare fold compositions, target distributions, and input statistics."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ---- Load data ----
sp = np.load(PROJECT_ROOT / "cache/v1/splits/kfold5_wafer.npz", allow_pickle=True)
fold_indices = sp["sample_fold_id"]  # per-sample fold assignment
wafer_fold_id = sp["wafer_fold_id"]  # per-wafer fold assignment
wafer_keys = sp["wafer_keys"]        # wafer name array

pq = PROJECT_ROOT / "cache/v1/measurements.parquet"
if pq.exists():
    meas = pd.read_parquet(pq)
else:
    meas = pd.read_csv(PROJECT_ROOT / "cache/v1/measurements.csv")

# Derive lot from experiment_key date part
DAY_TO_LOT = {
    "2024-07-02": 1, "2024-07-05": 2, "2024-07-09": 3, "2024-07-11": 4,
    "2024-07-19": 5, "2024-08-01": 6, "2024-08-05": 7, "2024-08-07": 8,
    "2024-08-21": 9, "2024-08-22": 10,
}
meas["day"] = meas["experiment_key"].str[:10]
meas["lot"] = meas["day"].map(DAY_TO_LOT)

print("=" * 70)
print("FOLD COMPOSITION ANALYSIS")
print("=" * 70)

# ---- Per-fold summary ----
for f in range(5):
    val_mask = fold_indices == f
    val = meas[val_mask]
    wafers = sorted(val["experiment_key"].unique())
    lots = sorted(val["lot"].unique()) if "lot" in meas.columns else []
    ox = val["oxide_etch"]
    si = val["si_etch"]

    print(f"\n--- Fold {f} (val) ---")
    print(f"  Samples: {val_mask.sum()}, Wafers: {len(wafers)}")
    print(f"  Lots: {lots}")
    print(f"  oxide_etch: mean={ox.mean():.4f}  std={ox.std():.4f}  "
          f"range=[{ox.min():.4f}, {ox.max():.4f}]")
    print(f"  si_etch:    mean={si.mean():.2f}  std={si.std():.2f}")
    print(f"  Wafers: {wafers}")

# ---- Cross-fold: lot distribution ----
print("\n" + "=" * 70)
print("LOT-FOLD CROSS TABLE (wafer count per lot per fold)")
print("=" * 70)

meas["fold"] = fold_indices
wafer_fold = meas.drop_duplicates("experiment_key")[["experiment_key", "lot", "fold"]]
ct = pd.crosstab(wafer_fold["lot"], wafer_fold["fold"], margins=True)
print(ct.to_string())

# ---- Per-lot oxide stats ----
print("\n" + "=" * 70)
print("PER-LOT OXIDE STATS")
print("=" * 70)

lot_stats = meas.groupby("lot")["oxide_etch"].agg(["mean", "std", "min", "max", "count"])
lot_stats.columns = ["mean", "std", "min", "max", "n_samples"]
print(lot_stats.to_string(float_format="%.4f"))

# ---- Fold 4 vs others: wafer-level oxide means ----
print("\n" + "=" * 70)
print("WAFER-LEVEL OXIDE MEAN: FOLD 4 VAL vs TRAINING")
print("=" * 70)

wafer_ox = meas.groupby("experiment_key")["oxide_etch"].mean()
wafer_fold_map = meas.drop_duplicates("experiment_key").set_index("experiment_key")["fold"]

for f in range(5):
    val_wafers = wafer_fold_map[wafer_fold_map == f].index
    train_wafers = wafer_fold_map[wafer_fold_map != f].index
    val_means = wafer_ox[val_wafers]
    train_means = wafer_ox[train_wafers]
    print(f"\nFold {f} val:   n={len(val_means):2d}  "
          f"wafer_ox_mean={val_means.mean():.4f}  std={val_means.std():.4f}  "
          f"range=[{val_means.min():.4f}, {val_means.max():.4f}]")
    print(f"Fold {f} train: n={len(train_means):2d}  "
          f"wafer_ox_mean={train_means.mean():.4f}  std={train_means.std():.4f}  "
          f"range=[{train_means.min():.4f}, {train_means.max():.4f}]")

# ---- Fold 4 specific: which wafers have extreme oxide values? ----
print("\n" + "=" * 70)
print("FOLD 4 VAL WAFERS: SORTED BY OXIDE MEAN")
print("=" * 70)

f4_wafers = wafer_fold_map[wafer_fold_map == 4].index
f4_detail = meas[meas["experiment_key"].isin(f4_wafers)].groupby("experiment_key").agg(
    lot=("lot", "first"),
    oxide_mean=("oxide_etch", "mean"),
    oxide_std=("oxide_etch", "std"),
    si_mean=("si_etch", "mean"),
    n=("oxide_etch", "count"),
).sort_values("oxide_mean")
print(f4_detail.to_string(float_format="%.4f"))

# ---- Compare: fold 4 val wafer oxide distribution vs global ----
print("\n" + "=" * 70)
print("OXIDE WAFER-MEAN DISTRIBUTION: ALL FOLDS")
print("=" * 70)

for f in range(5):
    fw = wafer_fold_map[wafer_fold_map == f].index
    fmeans = wafer_ox[fw]
    # percentiles
    print(f"Fold {f}: p10={fmeans.quantile(0.1):.4f}  p25={fmeans.quantile(0.25):.4f}  "
          f"p50={fmeans.quantile(0.5):.4f}  p75={fmeans.quantile(0.75):.4f}  "
          f"p90={fmeans.quantile(0.9):.4f}  IQR={fmeans.quantile(0.75)-fmeans.quantile(0.25):.4f}")

# ---- Key question: does fold 4 have wafers from lots NOT seen in training? ----
print("\n" + "=" * 70)
print("LOT OVERLAP: FOLD 4 VAL LOTS vs FOLD 4 TRAINING LOTS")
print("=" * 70)

f4_val_lots = set(meas[fold_indices == 4]["lot"].unique())
f4_train_lots = set(meas[fold_indices != 4]["lot"].unique())
print(f"Fold 4 val lots:   {sorted(f4_val_lots)}")
print(f"Fold 4 train lots: {sorted(f4_train_lots)}")
print(f"Val-only lots (not in train): {sorted(f4_val_lots - f4_train_lots)}")
print(f"Train-only lots (not in val): {sorted(f4_train_lots - f4_val_lots)}")

# Same analysis for fold 2 (also weak)
print("\n--- Same for Fold 2 ---")
f2_val_lots = set(meas[fold_indices == 2]["lot"].unique())
f2_train_lots = set(meas[fold_indices != 2]["lot"].unique())
print(f"Fold 2 val lots:   {sorted(f2_val_lots)}")
print(f"Fold 2 train lots: {sorted(f2_train_lots)}")
print(f"Val-only lots: {sorted(f2_val_lots - f2_train_lots)}")

# ---- OES/Process input-level statistics per wafer ----
print("\n" + "=" * 70)
print("INPUT SIGNAL STATISTICS: OES & PROCESS PER WAFER")
print("=" * 70)

cache_root = PROJECT_ROOT / "cache" / "v1" / "wafers"
wafer_stats = []
for wk in wafer_keys:
    npz_path = cache_root / f"{wk}.npz"
    if not npz_path.exists():
        continue
    d = np.load(npz_path, allow_pickle=True)
    oes_raw = d["oes_data"].astype(np.float32)       # (T_oes, 3648)
    proc_raw = d["process_data"].astype(np.float32)   # (T_proc, 44)
    oes_cstarts = d["oes_cycle_starts_idx"]            # (100,)
    oes_cends = d["oes_cycle_ends_idx"]                # (100,)
    proc_cstarts = d["proc_cycle_starts_idx"]
    proc_cends = d["proc_cycle_ends_idx"]

    # Per-cycle OES means (mean over time & wavelength within each cycle)
    oes_cycle_means = np.array([
        oes_raw[s:e].mean() for s, e in zip(oes_cstarts, oes_cends)
    ])
    proc_cycle_means = np.array([
        proc_raw[s:e].mean() for s, e in zip(proc_cstarts, proc_cends)
    ])

    # Early (0-20) vs Late (80-100)
    oes_early = oes_cycle_means[:20].mean()
    oes_late = oes_cycle_means[80:].mean()
    proc_early = proc_cycle_means[:20].mean()
    proc_late = proc_cycle_means[80:].mean()

    wafer_stats.append({
        "wafer": wk,
        "oes_mean": float(oes_cycle_means.mean()),
        "oes_std": float(oes_cycle_means.std()),
        "oes_early": float(oes_early),
        "oes_late": float(oes_late),
        "oes_drift": float(oes_late - oes_early),
        "oes_cycle_std": float(oes_cycle_means.std()),
        "proc_mean": float(proc_cycle_means.mean()),
        "proc_std": float(proc_cycle_means.std()),
        "proc_early": float(proc_early),
        "proc_late": float(proc_late),
        "proc_drift": float(proc_late - proc_early),
        "proc_cycle_std": float(proc_cycle_means.std()),
    })

ws = pd.DataFrame(wafer_stats)
ws["fold"] = [int(wafer_fold_id[i]) for i in range(len(wafer_keys))]
ws["lot"] = ws["wafer"].str[:10].map(DAY_TO_LOT)
ws["oxide_mean"] = [float(wafer_ox[wk]) for wk in wafer_keys]

print("\nPer-fold input signal means:")
fold_input = ws.groupby("fold")[["oes_mean", "oes_std", "oes_drift", "oes_cycle_std",
                                  "proc_mean", "proc_std", "proc_drift", "proc_cycle_std"]].agg(["mean", "std"])
print(fold_input.to_string(float_format="%.2f"))

print("\n\nFold 4 vs rest — signal comparison:")
f4 = ws[ws["fold"] == 4]
rest = ws[ws["fold"] != 4]
for col in ["oes_mean", "oes_std", "oes_drift", "oes_cycle_std",
            "proc_mean", "proc_std", "proc_drift", "proc_cycle_std"]:
    print(f"  {col:20s}  fold4={f4[col].mean():.4f}±{f4[col].std():.4f}  "
          f"rest={rest[col].mean():.4f}±{rest[col].std():.4f}")

# ---- Check: are the low-oxide lot1/2 wafers unique to fold 4? ----
print("\n" + "=" * 70)
print("LOW-OXIDE WAFERS (lot 1 & 2) DISTRIBUTION ACROSS FOLDS")
print("=" * 70)

low_ox = ws[ws["lot"].isin([1, 2])]
print(low_ox[["wafer", "fold", "lot", "oxide_mean", "oes_drift", "proc_drift"]].sort_values("oxide_mean").to_string(float_format="%.4f"))

# ---- Fold difficulty metric: correlation between oxide_mean and input features ----
print("\n" + "=" * 70)
print("WAFER-LEVEL CORRELATION: oxide_mean vs input features (per fold)")
print("=" * 70)

for f in range(5):
    fws = ws[ws["fold"] == f]
    if len(fws) < 5:
        continue
    corrs = fws[["oxide_mean", "oes_drift", "proc_drift", "oes_cycle_std", "proc_cycle_std"]].corr()["oxide_mean"]
    print(f"Fold {f} (n={len(fws)}): "
          f"oes_drift={corrs['oes_drift']:.3f}  proc_drift={corrs['proc_drift']:.3f}  "
          f"oes_cycle_std={corrs['oes_cycle_std']:.3f}  proc_cycle_std={corrs['proc_cycle_std']:.3f}")
