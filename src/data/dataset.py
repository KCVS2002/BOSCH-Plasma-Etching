"""PyTorch Dataset for cycle-aware DL training (Phase 3).

Two-tier caching:
  1. WaferCycleStore — lazily loads each wafer's NPZ once, builds the
     fixed-shape per-cycle tensors (100 × T × C), and caches them
     in-memory. The OES tensor is the bulk of the memory footprint
     (~88 × 100 × 128 × 3648 × 4 B ≈ 16 GB float32 if all loaded).
     For training we keep only train-fold wafers in memory at a time.
  2. SampleDataset — sample = one (wafer, measurement-point) pair.
     There are 88 × 89 = 7832 samples. Each sample shares its wafer's
     cycle tensors with the other 88 samples on the same wafer.

Normalization:
  - Computed once per fold from the TRAIN wafers only and applied to
    both train and val. Stats are stored as `Normalizer` instances.
  - OES: log1p first (heavy-tailed counts), then per-wavelength z-score.
  - Process: per-channel z-score; wholly-NaN channels are dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------
# Cycle tensor assembly
# ----------------------------------------------------------------------

def _resample_one_cycle(
    data: np.ndarray,        # (T_raw, C)
    n_target: int,
) -> np.ndarray:
    """Linearly interpolate a per-cycle slice onto `n_target` steps.

    `data` is the wafer-level array sliced to one cycle. If T_raw < 2
    (degenerate cycle, e.g. ignition-affected first cycle), tile the
    available samples up to length n_target.
    """
    t_raw = data.shape[0]
    if t_raw == 0:
        return np.zeros((n_target,) + data.shape[1:], dtype=np.float32)
    if t_raw == 1:
        return np.broadcast_to(data, (n_target,) + data.shape[1:]).astype(np.float32, copy=True)
    src_t = np.linspace(0.0, 1.0, t_raw, dtype=np.float32)
    dst_t = np.linspace(0.0, 1.0, n_target, dtype=np.float32)
    out = np.empty((n_target,) + data.shape[1:], dtype=np.float32)
    for c in range(data.shape[1]):
        out[:, c] = np.interp(dst_t, src_t, data[:, c].astype(np.float32))
    return out


def build_oes_cycle_tensor(npz: dict, t_o: int) -> np.ndarray:
    """(100, t_o, W) float32 — wafer's per-cycle OES, fixed time axis."""
    data = npz["oes_data"]                          # (T_raw, W)
    starts = npz["oes_cycle_starts_idx"]            # (100,)
    ends = npz["oes_cycle_ends_idx"]                # (100,)
    n = len(starts)
    out = np.empty((n, t_o, data.shape[1]), dtype=np.float32)
    for k in range(n):
        s, e = int(starts[k]), int(ends[k])
        out[k] = _resample_one_cycle(data[s:e], t_o)
    return out


def build_proc_cycle_tensor(
    npz: dict,
    t_p: int,
    keep_channel_idx: np.ndarray | None = None,
) -> np.ndarray:
    """(100, t_p, F) float32 — wafer's per-cycle Process, fixed time axis.

    Different wafers store different process_data widths (some 44 ch, some 31 ch),
    so column selection is by integer index — `keep_channel_idx` is the
    per-wafer index list resolved against THIS wafer's features array.
    """
    data = npz["process_data"]                      # (T_raw, F_this)
    if keep_channel_idx is not None:
        data = data[:, keep_channel_idx]
    starts = npz["proc_cycle_starts_idx"]
    ends = npz["proc_cycle_ends_idx"]
    n = len(starts)
    out = np.empty((n, t_p, data.shape[1]), dtype=np.float32)
    for k in range(n):
        s, e = int(starts[k]), int(ends[k])
        out[k] = _resample_one_cycle(data[s:e], t_p)
    return out


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------

@dataclass
class Normalizer:
    """Per-channel mean/std with log1p preprocessing flag."""
    mean: np.ndarray   # (C,) float32
    std: np.ndarray    # (C,) float32
    log1p: bool

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply to (..., C). Broadcasts mean/std across leading axes."""
        if self.log1p:
            x = np.log1p(np.maximum(x, 0.0).astype(np.float32))
        return (x - self.mean) / np.maximum(self.std, 1e-6)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "log1p": np.bool_(self.log1p)}

    @classmethod
    def from_state(cls, d: dict) -> "Normalizer":
        return cls(mean=np.asarray(d["mean"], dtype=np.float32),
                   std=np.asarray(d["std"], dtype=np.float32),
                   log1p=bool(d["log1p"]))


def fit_oes_normalizer(tensors: Iterable[np.ndarray]) -> Normalizer:
    """Per-wavelength z-score after log1p, computed across wafers/cycles/time."""
    sums = None
    sqs = None
    n_total = 0
    for t in tensors:
        x = np.log1p(np.maximum(t, 0.0).astype(np.float64))
        flat = x.reshape(-1, x.shape[-1])
        if sums is None:
            sums = flat.sum(axis=0)
            sqs = (flat * flat).sum(axis=0)
        else:
            sums += flat.sum(axis=0)
            sqs += (flat * flat).sum(axis=0)
        n_total += flat.shape[0]
    mean = (sums / n_total).astype(np.float32)
    var = (sqs / n_total - mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return Normalizer(mean=mean, std=std, log1p=True)


def fit_proc_normalizer(tensors: Iterable[np.ndarray]) -> Normalizer:
    """Per-channel z-score, no log."""
    sums = None
    sqs = None
    n_total = 0
    for t in tensors:
        x = t.astype(np.float64)
        flat = x.reshape(-1, x.shape[-1])
        if sums is None:
            sums = flat.sum(axis=0)
            sqs = (flat * flat).sum(axis=0)
        else:
            sums += flat.sum(axis=0)
            sqs += (flat * flat).sum(axis=0)
        n_total += flat.shape[0]
    mean = (sums / n_total).astype(np.float32)
    var = (sqs / n_total - mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
    return Normalizer(mean=mean, std=std, log1p=False)


# ----------------------------------------------------------------------
# Wafer cache + dataset
# ----------------------------------------------------------------------

@dataclass
class WaferRecord:
    experiment_key: str
    lot_number: int
    oes: np.ndarray | None    # (100, t_o, W) float32 normalized — None until normalize()
    proc: np.ndarray | None   # (100, t_p, F) float32 normalized
    oes_raw: np.ndarray       # (100, t_o, W) raw resampled (no normalization)
    proc_raw: np.ndarray
    points_X: np.ndarray      # (89,) raw mm coordinates
    points_Y: np.ndarray
    points_si: np.ndarray     # (89,) target in μm — raw scale
    points_ox: np.ndarray
    points_X_norm: np.ndarray | None = None  # filled by store.normalize_all()
    points_Y_norm: np.ndarray | None = None
    points_si_norm: np.ndarray | None = None
    points_ox_norm: np.ndarray | None = None
    # Optional wafer-level XGB features for DL injection (filled if store configured)
    xgb_feats_raw: np.ndarray | None = None  # (K,) float32 raw
    xgb_feats: np.ndarray | None = None      # (K,) float32 normalized


@dataclass
class ScalarStats:
    mean: float
    std: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / max(self.std, 1e-6)).astype(np.float32)

    def invert(self, x: np.ndarray) -> np.ndarray:
        return (x * max(self.std, 1e-6) + self.mean).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": float(self.mean), "std": float(self.std)}


class WaferCycleStore:
    """Loads + builds + caches per-wafer cycle tensors in memory.

    Use:
        store = WaferCycleStore(cache_root, t_o=128, t_p=30)
        store.load_wafers(experiment_keys)
        store.fit_normalizers(train_keys)
        store.normalize_all()
        wafer = store[exp_key]   # WaferRecord with .oes, .proc filled
    """

    def __init__(
        self,
        cache_root: Path,
        meas: "pd.DataFrame",  # noqa: F821
        t_o: int = 128,
        t_p: int = 30,
        proc_keep_channels: Sequence[str] | None = None,
        xgb_feat_names: Sequence[str] | None = None,
    ):
        self.cache_root = Path(cache_root)
        self.t_o = t_o
        self.t_p = t_p
        self.meas = meas
        self.proc_keep_channels = proc_keep_channels  # None = auto-discover
        self._records: dict[str, WaferRecord] = {}
        self.proc_kept_names: list[str] | None = None  # set by discover_common_proc_channels()
        self.wavelengths: np.ndarray | None = None
        self.oes_normalizer: Normalizer | None = None
        self.proc_normalizer: Normalizer | None = None
        self.x_stats: ScalarStats | None = None
        self.y_stats: ScalarStats | None = None
        self.si_stats: ScalarStats | None = None
        self.ox_stats: ScalarStats | None = None
        # XGB feature injection support (None = disabled)
        self.xgb_feat_names: list[str] | None = list(xgb_feat_names) if xgb_feat_names else None
        self._xgb_df: "pd.DataFrame | None" = None  # lazy-loaded feature table
        self.xgb_normalizer: Normalizer | None = None

    def _get_xgb_df(self) -> "pd.DataFrame":
        """Lazily load the wafer-level XGB feature table, indexed by experiment_key."""
        if self._xgb_df is None:
            import pandas as pd
            feat_dir = self.cache_root / "features"
            pq = feat_dir / "baseline_xgb_v1.parquet"
            csv = feat_dir / "baseline_xgb_v1.csv"
            df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv)
            missing = [c for c in (self.xgb_feat_names or []) if c not in df.columns]
            if missing:
                raise ValueError(f"XGB feature columns not found in feature table: {missing}")
            self._xgb_df = df.set_index("experiment_key")
        return self._xgb_df

    def discover_common_proc_channels(self, keys: Sequence[str]) -> list[str]:
        """Scan all wafers' `process_features` and choose channel names common
        to every wafer (and also non-NaN on every wafer). This must run before
        `load_wafer` so each wafer can resolve to identical channel order.

        Different wafers in cache/v1 store either 31 or 44 process channels
        — keeping their intersection avoids dropouts and matches the XGBoost
        baseline's "channels available on every wafer" rule.
        """
        all_sets: list[set[str]] = []
        not_all_nan_sets: list[set[str]] = []
        for k in keys:
            npz_path = self.cache_root / "wafers" / f"{k}.npz"
            z = np.load(npz_path, allow_pickle=False)
            names = [str(n) for n in z["process_features"]]
            data = z["process_data"]
            keep = ~np.all(np.isnan(data), axis=0)  # (F_this,) bool
            all_sets.append(set(names))
            not_all_nan_sets.append({names[i] for i in np.where(keep)[0]})
        common_names = set.intersection(*all_sets) & set.intersection(*not_all_nan_sets)
        if self.proc_keep_channels is not None:
            common_names &= set(self.proc_keep_channels)
        ordered = [n for n in sorted(common_names)]
        self.proc_kept_names = ordered
        return ordered

    def load_wafer(self, experiment_key: str) -> WaferRecord:
        """Load + resample one wafer; idempotent."""
        if experiment_key in self._records:
            return self._records[experiment_key]
        if self.proc_kept_names is None:
            raise RuntimeError(
                "call discover_common_proc_channels(...) before load_wafer(...)"
            )

        npz_path = self.cache_root / "wafers" / f"{experiment_key}.npz"
        z = np.load(npz_path, allow_pickle=False)
        # promote to a dict-of-arrays so resamplers don't keep the npz handle open
        npz = {k: z[k] for k in z.files}

        if self.wavelengths is None:
            self.wavelengths = npz["oes_wavelengths"].astype(np.float32)

        # Resolve kept-channel indices in THIS wafer's features array
        names = [str(n) for n in npz["process_features"]]
        name_to_idx = {n: i for i, n in enumerate(names)}
        try:
            keep_idx = np.array(
                [name_to_idx[n] for n in self.proc_kept_names], dtype=np.int32
            )
        except KeyError as e:
            raise RuntimeError(
                f"wafer {experiment_key} missing channel {e} present at discovery time"
            ) from None

        oes_raw = build_oes_cycle_tensor(npz, t_o=self.t_o)
        proc_raw = build_proc_cycle_tensor(npz, t_p=self.t_p, keep_channel_idx=keep_idx)
        # Replace any residual NaNs in proc with 0 (will be normalized away)
        np.nan_to_num(proc_raw, copy=False, nan=0.0)

        meas_w = self.meas[self.meas["experiment_key"] == experiment_key].sort_values(
            ["X", "Y"]
        ) if "X" in self.meas.columns else self.meas[
            self.meas["experiment_key"] == experiment_key
        ]
        rec = WaferRecord(
            experiment_key=experiment_key,
            lot_number=int(npz["lot_number"]),
            oes=None,
            proc=None,
            oes_raw=oes_raw,
            proc_raw=proc_raw,
            points_X=meas_w["X"].to_numpy(dtype=np.float32),
            points_Y=meas_w["Y"].to_numpy(dtype=np.float32),
            points_si=meas_w["si_etch"].to_numpy(dtype=np.float32),
            points_ox=meas_w["oxide_etch"].to_numpy(dtype=np.float32),
        )

        if self.xgb_feat_names is not None:
            xdf = self._get_xgb_df()
            rec.xgb_feats_raw = (
                xdf.loc[experiment_key, self.xgb_feat_names].to_numpy(dtype=np.float32)
            )

        self._records[experiment_key] = rec
        return rec

    def load_wafers(self, keys: Iterable[str], progress: bool = True) -> None:
        keys = list(keys)
        iterator = keys
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(keys, desc="  load wafers", unit="wafer", ncols=100)
            except ImportError:
                pass
        for k in iterator:
            self.load_wafer(k)

    def fit_normalizers(self, train_keys: Sequence[str]) -> None:
        oes_iter = (self._records[k].oes_raw for k in train_keys)
        self.oes_normalizer = fit_oes_normalizer(oes_iter)
        proc_iter = (self._records[k].proc_raw for k in train_keys)
        self.proc_normalizer = fit_proc_normalizer(proc_iter)

        if self.xgb_feat_names is not None:
            # Each wafer contributes one (K,) vector; fit_proc_normalizer handles that.
            xgb_iter = (
                self._records[k].xgb_feats_raw
                for k in train_keys
                if self._records[k].xgb_feats_raw is not None
            )
            self.xgb_normalizer = fit_proc_normalizer(xgb_iter)

        # Spatial + target stats — flat-aggregate over all points of train wafers.
        Xs = np.concatenate([self._records[k].points_X for k in train_keys])
        Ys = np.concatenate([self._records[k].points_Y for k in train_keys])
        sis = np.concatenate([self._records[k].points_si for k in train_keys])
        oxs = np.concatenate([self._records[k].points_ox for k in train_keys])
        self.x_stats = ScalarStats(mean=float(Xs.mean()), std=float(Xs.std() or 1.0))
        self.y_stats = ScalarStats(mean=float(Ys.mean()), std=float(Ys.std() or 1.0))
        self.si_stats = ScalarStats(mean=float(sis.mean()), std=float(sis.std() or 1.0))
        self.ox_stats = ScalarStats(mean=float(oxs.mean()), std=float(oxs.std() or 1.0))

    def normalize_all(self, drop_raw: bool = True, progress: bool = True) -> None:
        """Apply fitted normalizers to every loaded wafer.

        With `drop_raw=True` (default), the un-normalized arrays are released
        afterwards to halve peak memory — important because the OES tensors
        are large (≈180 MB per wafer at float32).
        """
        if self.oes_normalizer is None or self.proc_normalizer is None:
            raise RuntimeError("call fit_normalizers() first")
        if self.x_stats is None:
            raise RuntimeError("scalar stats missing; fit_normalizers should have set them")
        recs = list(self._records.values())
        iterator = recs
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(recs, desc="  normalize", unit="wafer", ncols=100)
            except ImportError:
                pass
        for rec in iterator:
            rec.oes = self.oes_normalizer.apply(rec.oes_raw).astype(np.float32)
            rec.proc = self.proc_normalizer.apply(rec.proc_raw).astype(np.float32)
            rec.points_X_norm = self.x_stats.apply(rec.points_X)
            rec.points_Y_norm = self.y_stats.apply(rec.points_Y)
            rec.points_si_norm = self.si_stats.apply(rec.points_si)
            rec.points_ox_norm = self.ox_stats.apply(rec.points_ox)
            if self.xgb_normalizer is not None and rec.xgb_feats_raw is not None:
                rec.xgb_feats = self.xgb_normalizer.apply(rec.xgb_feats_raw).astype(np.float32)
            if drop_raw:
                rec.oes_raw = np.empty(0, dtype=np.float32)
                rec.proc_raw = np.empty(0, dtype=np.float32)

    def __getitem__(self, key: str) -> WaferRecord:
        return self._records[key]

    def __contains__(self, key: str) -> bool:
        return key in self._records

    @property
    def n_proc_channels(self) -> int:
        if self.proc_kept_names is None:
            raise RuntimeError("call discover_common_proc_channels first")
        return len(self.proc_kept_names)

    @property
    def n_wavelengths(self) -> int:
        return 0 if self.wavelengths is None else int(len(self.wavelengths))

    @property
    def n_xgb_feats(self) -> int:
        return len(self.xgb_feat_names) if self.xgb_feat_names is not None else 0


class WaferDataset(Dataset):
    """One sample = one wafer + its 89 measurement points.

    The cycle encoder + LSTM run ONCE per wafer; the regression head is
    broadcast across the 89 (X, Y) points. This is dramatically more
    memory-efficient than treating each (wafer, point) as a separate
    sample, since the heavy OES forward (≈3 GB intermediate per wafer)
    isn't repeated 89×.

    Output dict keys (only those relevant to modality):
        oes:    (n_cycles, t_o, W) float32
        proc:   (n_cycles, t_p, F) float32
        xy:     (n_points, 2) float32
        target: (n_points,)   float32
    """

    def __init__(
        self,
        store: WaferCycleStore,
        wafer_keys: Sequence[str],
        target: str,
        modality: str = "multimodal",
    ):
        if modality not in {"oes", "proc", "multimodal"}:
            raise ValueError(f"unknown modality {modality!r}")
        if target not in {"si_etch", "oxide_etch"}:
            raise ValueError(f"target must be si_etch or oxide_etch, got {target!r}")
        self.store = store
        self.keys = list(wafer_keys)
        self.target = target
        self.target_attr_norm = "points_si_norm" if target == "si_etch" else "points_ox_norm"
        self.modality = modality

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.store[self.keys[idx]]
        if rec.points_X_norm is None or rec.points_Y_norm is None:
            raise RuntimeError("call store.normalize_all() before iterating dataset")
        xy = np.stack([rec.points_X_norm, rec.points_Y_norm], axis=-1).astype(np.float32)
        target_norm = getattr(rec, self.target_attr_norm)
        out: dict[str, torch.Tensor] = {
            "xy": torch.from_numpy(xy),                  # (89, 2) — z-scored
            "target": torch.from_numpy(target_norm),     # (89,)   — z-scored target
        }
        if self.modality in ("oes", "multimodal"):
            out["oes"] = torch.from_numpy(rec.oes)
        if self.modality in ("proc", "multimodal"):
            out["proc"] = torch.from_numpy(rec.proc)
        if rec.xgb_feats is not None:
            out["xgb_feat"] = torch.from_numpy(rec.xgb_feats)  # (K,) wafer-level
        return out


class SampleDataset(Dataset):
    """Per-sample dataset (legacy, kept for reference). Use WaferDataset
    for actual training — it's much more memory-efficient because the
    cycle encoder forward is shared across the 89 points of a wafer."""

    def __init__(
        self,
        store: WaferCycleStore,
        sample_wafer_keys: Sequence[str],
        sample_point_idx: np.ndarray,
        sample_targets: np.ndarray,
        sample_xy: np.ndarray,
        modality: str = "multimodal",
    ):
        if modality not in {"oes", "proc", "multimodal"}:
            raise ValueError(f"unknown modality {modality!r}")
        self.store = store
        self.keys = list(sample_wafer_keys)
        self.point_idx = np.asarray(sample_point_idx, dtype=np.int32)
        self.targets = np.asarray(sample_targets, dtype=np.float32)
        self.xy = np.asarray(sample_xy, dtype=np.float32)
        self.modality = modality
        if not (len(self.keys) == len(self.point_idx) == len(self.targets) == len(self.xy)):
            raise ValueError("sample arrays must align in length")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.store[self.keys[idx]]
        out: dict[str, torch.Tensor] = {
            "xy": torch.from_numpy(self.xy[idx]),
            "target": torch.tensor(float(self.targets[idx]), dtype=torch.float32),
        }
        if self.modality in ("oes", "multimodal"):
            out["oes"] = torch.from_numpy(rec.oes)
        if self.modality in ("proc", "multimodal"):
            out["proc"] = torch.from_numpy(rec.proc)
        return out


def build_sample_arrays(
    meas: "pd.DataFrame",  # noqa: F821
    target: str,
) -> dict[str, np.ndarray]:
    """Flatten the measurement DataFrame into aligned sample arrays."""
    keys = meas["experiment_key"].to_numpy().astype(str)
    # within-wafer point index
    point_idx = (
        meas.groupby("experiment_key").cumcount().to_numpy(dtype=np.int32)
    )
    xy = meas[["X", "Y"]].to_numpy(dtype=np.float32)
    targets = meas[target].to_numpy(dtype=np.float32)
    return {"keys": keys, "point_idx": point_idx, "xy": xy, "targets": targets}
