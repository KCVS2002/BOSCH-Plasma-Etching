"""Per-cycle 2D-CNN encoder.

A single cycle is a 2D map of shape (T, C):
  - OES   cycle: (T_o=128, W=3648)
  - Process cycle: (T_p=30, F≈31)

We encode each cycle to a fixed-length embedding by treating it as a 2D
"image" and stacking Conv2d → BN → GELU → MaxPool blocks. Wafers are run
through the SAME encoder for all 100 cycles (weight sharing across the
cycle axis is enforced by reshaping (B, 100, T, C) → (B*100, 1, T, C)
before the conv stack).

Per the supervisor feedback (2026-04-24):
  - Same architecture FAMILY for OES and Process (no PCA, no MLP).
  - Independent weights — modalities have very different shapes and
    physical meaning of channels.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CycleEncoderConfig:
    in_time: int           # T  (input timesteps per cycle)
    in_channels: int       # C  (wavelengths or process channels)
    out_dim: int           # embedding size produced for one cycle
    base_channels: int = 16
    n_blocks: int = 3
    kernel_time: int = 3
    kernel_chan: int = 7   # wider along channel axis (helps for OES wavelength bands)
    stride_time: int = 2   # conv stride along time axis (replaces explicit pool)
    stride_chan: int = 4   # conv stride along channel axis — for OES this is the
                           # main memory lever; first conv goes from W=3648 to
                           # W/stride_chan immediately, halving the activation.
    dropout: float = 0.1
    norm_type: str = "batch"  # "batch" or "instance"


class CycleEncoder2D(nn.Module):
    """Encode a (T, C) per-cycle map → (out_dim,) embedding.

    Uses strided convolutions (no separate maxpool) to keep activation
    memory low — the first conv's output is already (T/stride_time,
    C/stride_chan) which matters when C=3648 wavelengths.
    """

    def __init__(self, cfg: CycleEncoderConfig):
        super().__init__()
        self.cfg = cfg

        layers: list[nn.Module] = []
        in_ch = 1
        out_ch = cfg.base_channels
        cur_t, cur_c = cfg.in_time, cfg.in_channels
        for b in range(cfg.n_blocks):
            kt = min(cfg.kernel_time, max(1, cur_t))
            kc = min(cfg.kernel_chan, max(1, cur_c))
            st = max(1, min(cfg.stride_time, cur_t))
            sc = max(1, min(cfg.stride_chan, cur_c))
            pad_t = kt // 2
            pad_c = kc // 2
            layers.append(nn.Conv2d(
                in_ch, out_ch,
                kernel_size=(kt, kc),
                stride=(st, sc),
                padding=(pad_t, pad_c),
            ))
            if cfg.norm_type == "instance":
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            else:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.GELU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout2d(cfg.dropout))
            cur_t = max(1, cur_t // st)
            cur_c = max(1, cur_c // sc)
            in_ch = out_ch
            out_ch = min(out_ch * 2, 128)

        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, cfg.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C) — one cycle per batch element.
        Returns:
            (B, out_dim)
        """
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, C), got {tuple(x.shape)}")
        x = x.unsqueeze(1)              # (B, 1, T, C)
        x = self.conv(x)                # (B, ch, T', C')
        x = self.pool(x).flatten(1)     # (B, ch)
        return self.proj(x)             # (B, out_dim)


class CycleSeriesEncoder(nn.Module):
    """Apply CycleEncoder2D independently to all 100 cycles of a wafer.

    Input shape: (B, n_cycles, T, C)  →  Output: (B, n_cycles, out_dim)
    Implementation: collapse (B, n_cycles) into the batch axis to reuse
    the same conv weights, then unflatten.
    """

    def __init__(self, cfg: CycleEncoderConfig):
        super().__init__()
        self.encoder = CycleEncoder2D(cfg)
        self.cfg = cfg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected (B, n_cycles, T, C), got {tuple(x.shape)}")
        b, n, t, c = x.shape
        flat = x.reshape(b * n, t, c)
        emb = self.encoder(flat)         # (B*n, out_dim)
        return emb.reshape(b, n, self.cfg.out_dim)
