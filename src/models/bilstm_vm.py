"""End-to-end Cycle-Aware DL Virtual Metrology model.

Pipeline (Phase 3, Exp 4 — main proposed method):

    OES cycle tensor      Process cycle tensor       (X, Y) spatial coord
    (B, 100, t_o, W)      (B, 100, t_p, F)           (B, n_pts, 2)
            |                       |                       |
       2D-CNN (shared              2D-CNN (shared          Fourier feature
       across cycles)              across cycles)          encoder
            |                       |                       |
    cycle embeddings        cycle embeddings           (B, n_pts, d_xy)
    (B, 100, d_o)           (B, 100, d_p)                   |
            |                       |                       |
            +-------- concat -------+                       |
                        |                                   |
                cycle fusion FC: (d_o+d_p) → d_cycle         |
                        |                                   |
                Bi-LSTM (B, 100, d_cycle) → (B, 2*h_lstm)    |
                        |                                   |
                        +-------- FiLM modulate ------------+
                                       |
                            per-point wafer repr (B, n_pts, 2h)
                                  ⊕ xy_enc (skip)
                                       |
                                Regression head
                                       |
                              (B, n_pts) scalar pred

Why FiLM + Fourier for xy:
  Each wafer has 89 measurement points sharing the SAME wafer_repr.  The
  only point-specific signal is (X, Y).  Without modulation the head can
  only differentiate points by 2 raw scalars — too weak compared to
  XGBoost's per-point splits.  We Fourier-encode (X, Y) to a high-dim
  representation and use it to FiLM-modulate the wafer_repr, giving each
  point its own affine-transformed wafer feature.

Single-modality variants (oes-only, proc-only) skip the missing branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .cycle_encoder import CycleEncoderConfig, CycleSeriesEncoder


@dataclass
class CycleVMConfig:
    """Configuration for the full BiLSTM + cycle encoder model.

    `oes_encoder` / `proc_encoder` are dicts (CycleEncoderConfig fields).
    Pass `None` to disable that modality.
    """
    oes_encoder: dict[str, Any] | None
    proc_encoder: dict[str, Any] | None
    cycle_fusion_dim: int = 128
    lstm_hidden: int = 128
    lstm_layers: int = 1
    lstm_dropout: float = 0.0
    head_hidden: int = 128
    head_dropout: float = 0.2
    use_xy: bool = True
    # --- xy representation ---
    xy_n_freqs: int = 6        # 0 = use raw xy (legacy). >0 = Fourier features.
    xy_enc_dim: int = 64       # output dim of xy encoder MLP
    use_film: bool = True      # FiLM-modulate wafer_repr by xy_enc per point
    # --- cycle aggregation ---
    pool: str = "mean"         # "mean" (uniform) or "attention" (learned weights)
    attn_hidden: int = 64      # hidden dim of attention scoring MLP
    # --- wafer-level XGB feature injection (before FiLM) ---
    # Set xgb_feat_dim=0 (default) to disable — identical to the original model.
    # When >0, the K XGB features are projected to xgb_proj_dim and concat'd to
    # wafer_repr BEFORE FiLM, enriching the shared wafer context each point sees.
    xgb_feat_dim: int = 0      # 0 = disabled (backward compat). Set to len(xgb_feat_names).
    xgb_proj_dim: int = 32     # projection dim for XGB features


class FourierFeatureEncoder(nn.Module):
    """Fourier-feature encoding for 2D coordinates.

    For each axis, project onto fixed log-spaced frequencies and take
    sin/cos. This lifts 2 raw scalars to (2 + 4*n_freqs) features that a
    small MLP then projects to `out_dim`. Inputs are assumed z-scored
    (typical range ±3) so frequencies 2^0…2^(n_freqs-1) capture both
    coarse and fine spatial variation across the wafer.

    Shapes: (B, n_pts, 2) → (B, n_pts, out_dim)
    """

    def __init__(self, n_freqs: int, out_dim: int):
        super().__init__()
        self.n_freqs = n_freqs
        self.out_dim = out_dim
        freqs = (2.0 ** torch.arange(n_freqs)).float()  # 1, 2, 4, ...
        self.register_buffer("freqs", freqs, persistent=False)
        in_dim = 2 + 4 * n_freqs
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (B, n_pts, 2)
        scaled = xy.unsqueeze(-1) * self.freqs                  # (B, n_pts, 2, F)
        sins = torch.sin(scaled)
        coss = torch.cos(scaled)
        fourier = torch.cat([sins, coss], dim=-1)               # (B, n_pts, 2, 2F)
        fourier = fourier.flatten(-2)                            # (B, n_pts, 4F)
        full = torch.cat([xy, fourier], dim=-1)                  # (B, n_pts, 2+4F)
        return self.proj(full)


class AttentionPool(nn.Module):
    """Learnable softmax attention over the cycle dimension.

    Replaces uniform mean pooling. The model learns per-cycle weights that
    can emphasise cycles carrying more target-relevant signal — useful when
    a target (e.g. oxide_etch) is dominated by specific phases of the
    100-cycle sequence rather than the global average.

    Input:  (B, n_cycles, d)
    Output: (B, d)
    """

    def __init__(self, d_in: int, d_hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)            # (B, n_cycles)
        weights = torch.softmax(scores, dim=-1)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, d)


class FiLMHead(nn.Module):
    """Per-point FiLM modulation of wafer_repr by xy_enc, then regress.

    Modulator: xy_enc → (γ, β), each shape (n_pts, wafer_dim).
    γ is centred at 1 (i.e. modulator outputs Δγ; effective γ = 1 + Δγ)
    so a fresh model starts close to "broadcast wafer_repr unchanged",
    matching the legacy non-FiLM path's initial behaviour.
    """

    def __init__(
        self,
        wafer_dim: int,
        xy_enc_dim: int,
        head_hidden: int,
        head_dropout: float,
        xgb_proj_dim: int = 0,
    ):
        super().__init__()
        self.wafer_dim = wafer_dim
        self.modulator = nn.Linear(xy_enc_dim, 2 * wafer_dim)
        # Initialise modulator output to ~0 so γ≈1, β≈0 at start
        nn.init.zeros_(self.modulator.weight)
        nn.init.zeros_(self.modulator.bias)
        # xgb_proj_dim > 0: XGB features are concat'd AFTER FiLM modulation.
        # FiLM only modulates the LSTM wafer_repr — XGB features bypass it entirely.
        self.regress = nn.Sequential(
            nn.Linear(wafer_dim + xy_enc_dim + xgb_proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(
        self,
        wafer: torch.Tensor,
        xy_enc: torch.Tensor,
        xgb_enc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # wafer:   (B, wafer_dim)
        # xy_enc:  (B, n_pts, xy_enc_dim)
        # xgb_enc: (B, xgb_proj_dim) or None — wafer-level, broadcast to all points
        n_pts = xy_enc.shape[1]
        gb = self.modulator(xy_enc)                              # (B, n_pts, 2*wd)
        d_gamma, beta = gb.chunk(2, dim=-1)                      # each (B, n_pts, wd)
        gamma = 1.0 + d_gamma
        wafer_b = wafer.unsqueeze(1).expand(-1, n_pts, -1)       # (B, n_pts, wd)
        modulated = gamma * wafer_b + beta
        parts = [modulated, xy_enc]                              # FiLM result first
        if xgb_enc is not None:
            # (B, proj_dim) → (B, n_pts, proj_dim): same XGB context for every point
            parts.append(xgb_enc.unsqueeze(1).expand(-1, n_pts, -1))
        full = torch.cat(parts, dim=-1)
        return self.regress(full).squeeze(-1)                    # (B, n_pts)


class CycleAwareBiLSTM(nn.Module):
    def __init__(self, cfg: CycleVMConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.oes_encoder is None and cfg.proc_encoder is None:
            raise ValueError("at least one of oes_encoder/proc_encoder must be set")

        emb_dim_total = 0
        if cfg.oes_encoder is not None:
            self.oes_enc = CycleSeriesEncoder(CycleEncoderConfig(**cfg.oes_encoder))
            emb_dim_total += int(cfg.oes_encoder["out_dim"])
        else:
            self.oes_enc = None
        if cfg.proc_encoder is not None:
            self.proc_enc = CycleSeriesEncoder(CycleEncoderConfig(**cfg.proc_encoder))
            emb_dim_total += int(cfg.proc_encoder["out_dim"])
        else:
            self.proc_enc = None

        self.fusion = nn.Sequential(
            nn.Linear(emb_dim_total, cfg.cycle_fusion_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.cycle_fusion_dim),
        )

        self.lstm = nn.LSTM(
            input_size=cfg.cycle_fusion_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
        )

        wafer_dim = 2 * cfg.lstm_hidden

        if cfg.pool == "attention":
            self.cycle_pool = AttentionPool(d_in=wafer_dim, d_hidden=cfg.attn_hidden)
        elif cfg.pool == "mean":
            self.cycle_pool = None
        else:
            raise ValueError(f"unknown pool={cfg.pool!r} (expected 'mean' or 'attention')")

        # XGB feature projection (K → xgb_proj_dim).
        # Injected AFTER FiLM into the regression head, NOT into wafer_repr.
        # This keeps FiLM's wafer_dim and zero-init unchanged from the baseline.
        #
        # Zero-init the Linear so xgb_enc ≈ 0 at epoch 0 → head sees the same
        # input distribution as the baseline (xgb portion is just zeros). The
        # gradient still flows back to xgb_proj weights via the head's xgb-portion
        # weights (which are random/non-zero), so xgb_proj learns to extract
        # signal gradually rather than disrupting training from epoch 0.
        # Mirrors FiLM's zero-init philosophy: start as identity, learn modulation.
        if cfg.xgb_feat_dim > 0:
            xgb_linear = nn.Linear(cfg.xgb_feat_dim, cfg.xgb_proj_dim)
            nn.init.zeros_(xgb_linear.weight)
            nn.init.zeros_(xgb_linear.bias)
            self.xgb_proj = nn.Sequential(
                xgb_linear,
                nn.GELU(),
            )
        else:
            self.xgb_proj = None

        xgb_head_dim = cfg.xgb_proj_dim if cfg.xgb_feat_dim > 0 else 0

        if cfg.use_xy:
            if cfg.xy_n_freqs > 0:
                self.xy_encoder = FourierFeatureEncoder(
                    n_freqs=cfg.xy_n_freqs, out_dim=cfg.xy_enc_dim,
                )
                xy_dim = cfg.xy_enc_dim
            else:
                self.xy_encoder = None
                xy_dim = 2
        else:
            self.xy_encoder = None
            xy_dim = 0

        self.use_film = bool(cfg.use_xy and cfg.use_film)
        if self.use_film:
            self.film_head = FiLMHead(
                wafer_dim=wafer_dim,
                xy_enc_dim=xy_dim,
                head_hidden=cfg.head_hidden,
                head_dropout=cfg.head_dropout,
                xgb_proj_dim=xgb_head_dim,
            )
            self.head = None
        else:
            head_in = wafer_dim + xy_dim + xgb_head_dim
            self.head = nn.Sequential(
                nn.Linear(head_in, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.head_hidden, 1),
            )
            self.film_head = None

    def encode_cycles(
        self,
        oes: torch.Tensor | None,
        proc: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return cycle-level embeddings (B, n_cycles, cycle_fusion_dim)."""
        embs: list[torch.Tensor] = []
        if self.oes_enc is not None:
            if oes is None:
                raise ValueError("model expects OES input")
            embs.append(self.oes_enc(oes))
        if self.proc_enc is not None:
            if proc is None:
                raise ValueError("model expects Process input")
            embs.append(self.proc_enc(proc))
        cyc = torch.cat(embs, dim=-1) if len(embs) > 1 else embs[0]
        return self.fusion(cyc)

    def forward(
        self,
        oes: torch.Tensor | None = None,
        proc: torch.Tensor | None = None,
        xy: torch.Tensor | None = None,
        xgb_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Wafer-level forward.

        Shapes:
            oes:      (B, n_cycles, T_o, W)
            proc:     (B, n_cycles, T_p, F)
            xy:       (B, n_points, 2)
            xgb_feat: (B, K) wafer-level XGB features — required if xgb_feat_dim > 0
        Returns:
            (B, n_points)
        """
        cycles = self.encode_cycles(oes, proc)               # (B, 100, d)
        seq, _ = self.lstm(cycles)                            # (B, 100, 2h)
        if self.cycle_pool is not None:
            wafer = self.cycle_pool(seq)                      # (B, 2h) attention
        else:
            wafer = seq.mean(dim=1)                           # (B, 2h) mean

        # Project XGB features for head injection (does NOT modify wafer_repr)
        if self.xgb_proj is not None:
            if xgb_feat is None:
                raise ValueError("model expects xgb_feat input (xgb_feat_dim > 0)")
            xgb_enc = self.xgb_proj(xgb_feat)   # (B, xgb_proj_dim)
        else:
            xgb_enc = None

        if not self.cfg.use_xy:
            # No xy at all — return one prediction per wafer
            assert self.head is not None
            return self.head(wafer).squeeze(-1)               # (B,)

        if xy is None:
            raise ValueError("model expects xy input (use_xy=True)")
        if xy.dim() != 3:
            raise ValueError(f"xy must be (B, n_points, 2), got {tuple(xy.shape)}")

        xy_enc = self.xy_encoder(xy) if self.xy_encoder is not None else xy

        if self.use_film:
            assert self.film_head is not None
            return self.film_head(wafer, xy_enc, xgb_enc=xgb_enc)   # (B, n_pts)

        # Legacy concat path (non-FiLM)
        n_pts = xy_enc.shape[1]
        wafer_b = wafer.unsqueeze(1).expand(-1, n_pts, -1)
        parts = [wafer_b, xy_enc]
        if xgb_enc is not None:
            parts.append(xgb_enc.unsqueeze(1).expand(-1, n_pts, -1))
        full = torch.cat(parts, dim=-1)
        assert self.head is not None
        return self.head(full).squeeze(-1)


def build_cycle_aware_bilstm(params: dict[str, Any]) -> CycleAwareBiLSTM:
    """Factory used by the model registry. `params` is the YAML model.params."""
    cfg = CycleVMConfig(
        oes_encoder=params.get("oes_encoder"),
        proc_encoder=params.get("proc_encoder"),
        cycle_fusion_dim=int(params.get("cycle_fusion_dim", 128)),
        lstm_hidden=int(params.get("lstm_hidden", 128)),
        lstm_layers=int(params.get("lstm_layers", 1)),
        lstm_dropout=float(params.get("lstm_dropout", 0.0)),
        head_hidden=int(params.get("head_hidden", 128)),
        head_dropout=float(params.get("head_dropout", 0.2)),
        use_xy=bool(params.get("use_xy", True)),
        xy_n_freqs=int(params.get("xy_n_freqs", 6)),
        xy_enc_dim=int(params.get("xy_enc_dim", 64)),
        use_film=bool(params.get("use_film", True)),
        pool=str(params.get("pool", "mean")),
        attn_hidden=int(params.get("attn_hidden", 64)),
        xgb_feat_dim=int(params.get("xgb_feat_dim", 0)),
        xgb_proj_dim=int(params.get("xgb_proj_dim", 32)),
    )
    return CycleAwareBiLSTM(cfg)
