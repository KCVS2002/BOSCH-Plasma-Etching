"""Single-wafer inference for the Cycle-Aware DL Virtual Metrology model.

This wraps a trained checkpoint (produced by scripts/04_train_dl.py) so the
competition demo can turn one wafer's raw sensor tensors into the 89 predicted
etch values, exactly reproducing the training/validation forward pass:

    raw OES / Process cycle tensors  ──preprocess──▶  ModelInputs (normalized)
                                                            │
                                                       model.forward
                                                            │
                                                  per-point pred (z-scored)
                                                            │
                                                  target_stats.invert  ──▶  μm

`preprocess` and `forward` are split so the demo can pre-compute the (small)
normalized ModelInputs into a portable bundle and still run the genuine model
forward live on stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data import Normalizer, ScalarStats
from src.models import make_model


@dataclass
class ModelInputs:
    """Model-ready (already normalized) per-wafer tensors.

    Shapes match what WaferDataset feeds the model:
        oes:  (n_cycles, t_o, W_sel) float32 — log1p+z-scored, band-selected
        proc: (n_cycles, t_p, F)     float32 — z-scored
        xy:   (n_points, 2)          float32 — z-scored (X, Y)
    `oes` / `proc` may be None for single-modality models.
    """

    oes: np.ndarray | None
    proc: np.ndarray | None
    xy: np.ndarray

    def to_npz_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        d: dict[str, np.ndarray] = {f"{prefix}xy": self.xy.astype(np.float32)}
        if self.oes is not None:
            d[f"{prefix}oes"] = self.oes.astype(np.float32)
        if self.proc is not None:
            d[f"{prefix}proc"] = self.proc.astype(np.float32)
        return d


class DLPredictor:
    """Load one DL checkpoint and predict a wafer's 89 etch values.

    Use `from_checkpoint(path)` then either:
      • `predict_raw(oes_raw, proc_raw, X, Y)` — from raw resampled tensors, or
      • `preprocess(...)` once + `forward(inputs)` repeatedly (demo bundle path).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        modality: str,
        oes_normalizer: Normalizer | None,
        proc_normalizer: Normalizer | None,
        x_stats: ScalarStats,
        y_stats: ScalarStats,
        target_stats: ScalarStats,
        oes_band_idx: np.ndarray | None,
        target: str,
        fold: int,
        device: torch.device,
    ):
        self.model = model
        self.modality = modality
        self.oes_normalizer = oes_normalizer
        self.proc_normalizer = proc_normalizer
        self.x_stats = x_stats
        self.y_stats = y_stats
        self.target_stats = target_stats
        self.oes_band_idx = (
            np.asarray(oes_band_idx, dtype=np.int64) if oes_band_idx is not None else None
        )
        self.target = target
        self.fold = fold
        self.device = device

    # ------------------------------------------------------------------ #
    @classmethod
    def from_checkpoint(
        cls, ckpt_path: str | Path, device: torch.device | None = None
    ) -> "DLPredictor":
        ckpt_path = Path(ckpt_path)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        cfg = ckpt["config"]
        modality = cfg["data"]["modality"]
        proc_kept_names = list(ckpt.get("proc_kept_names") or [])
        oes_band_idx = ckpt.get("oes_band_idx")

        oes_norm = (
            Normalizer.from_state(ckpt["oes_normalizer"])
            if ckpt.get("oes_normalizer")
            else None
        )
        proc_norm = (
            Normalizer.from_state(ckpt["proc_normalizer"])
            if ckpt.get("proc_normalizer")
            else None
        )
        x_stats = ScalarStats(**ckpt["x_stats"])
        y_stats = ScalarStats(**ckpt["y_stats"])
        # 04_train_dl saves the per-fold target stats under "target_stats".
        target_stats = ScalarStats(**ckpt["target_stats"])

        # Rebuild model; override encoder in_channels with the actual kept
        # counts so the architecture matches the saved state_dict regardless
        # of what the YAML literal said.
        params = dict(cfg["model"]["params"])
        if params.get("proc_encoder") is not None and modality in ("proc", "multimodal"):
            params["proc_encoder"] = dict(params["proc_encoder"])
            params["proc_encoder"]["in_channels"] = len(proc_kept_names)
        if (
            params.get("oes_encoder") is not None
            and modality in ("oes", "multimodal")
            and oes_band_idx is not None
        ):
            params["oes_encoder"] = dict(params["oes_encoder"])
            params["oes_encoder"]["in_channels"] = int(len(oes_band_idx))

        model = make_model(cfg["model"]["name"], params)
        model.load_state_dict(ckpt["state_dict"])
        for m in model.modules():
            if isinstance(m, torch.nn.RNNBase):
                m.flatten_parameters()
        model.eval().to(device)

        return cls(
            model,
            modality=modality,
            oes_normalizer=oes_norm,
            proc_normalizer=proc_norm,
            x_stats=x_stats,
            y_stats=y_stats,
            target_stats=target_stats,
            oes_band_idx=oes_band_idx,
            target=str(ckpt.get("target", "")),
            fold=int(ckpt.get("fold", -1)),
            device=device,
        )

    # ------------------------------------------------------------------ #
    def preprocess(
        self,
        oes_raw: np.ndarray | None,
        proc_raw: np.ndarray | None,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> ModelInputs:
        """Raw resampled tensors → normalized, band-selected ModelInputs.

        Mirrors WaferCycleStore.normalize_all + WaferDataset.__getitem__:
        OES is log1p+z-scored over the FULL wavelength axis, THEN sliced to the
        selected band indices (per-channel z-score is elementwise, so the order
        is irrelevant — slicing the result equals slicing the normalizer).
        """
        oes_n = None
        if self.modality in ("oes", "multimodal"):
            if oes_raw is None or self.oes_normalizer is None:
                raise ValueError("model expects OES input but none/no normalizer given")
            oes_n = self.oes_normalizer.apply(oes_raw).astype(np.float32)
            if self.oes_band_idx is not None:
                oes_n = np.ascontiguousarray(oes_n[:, :, self.oes_band_idx])

        proc_n = None
        if self.modality in ("proc", "multimodal"):
            if proc_raw is None or self.proc_normalizer is None:
                raise ValueError("model expects Process input but none/no normalizer given")
            proc_n = self.proc_normalizer.apply(proc_raw).astype(np.float32)

        xy = np.stack(
            [self.x_stats.apply(np.asarray(X)), self.y_stats.apply(np.asarray(Y))],
            axis=-1,
        ).astype(np.float32)
        return ModelInputs(oes=oes_n, proc=proc_n, xy=xy)

    @torch.no_grad()
    def forward(self, inputs: ModelInputs) -> np.ndarray:
        """Run the genuine model forward; return 89 predictions in μm."""
        oes_t = (
            torch.from_numpy(inputs.oes).unsqueeze(0).to(self.device)
            if inputs.oes is not None
            else None
        )
        proc_t = (
            torch.from_numpy(inputs.proc).unsqueeze(0).to(self.device)
            if inputs.proc is not None
            else None
        )
        xy_t = torch.from_numpy(inputs.xy).unsqueeze(0).to(self.device)

        pred = self.model(oes=oes_t, proc=proc_t, xy=xy_t)
        if isinstance(pred, tuple):
            pred = pred[0]
        pred_np = pred.detach().float().cpu().numpy().reshape(-1)
        return self.target_stats.invert(pred_np)

    def predict_raw(
        self,
        oes_raw: np.ndarray | None,
        proc_raw: np.ndarray | None,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        return self.forward(self.preprocess(oes_raw, proc_raw, X, Y))
