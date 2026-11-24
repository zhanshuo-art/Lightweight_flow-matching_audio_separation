#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm import Mamba as _OfficialMamba
except Exception:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba as _OfficialMamba
    except Exception:
        _OfficialMamba = None


@dataclass
class MiniDiTConfig:
    input_size: tuple[int, int]
    patch_size: tuple[int, int]
    in_channels: int
    out_channels: int
    hidden_size: int
    depth: int
    num_heads: int
    mlp_ratio: float
    dropout: float
    num_classes: int
    mixer_type: str = "self_attention"
    mamba_backend: str = "torch"
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: float = 1.0
    hybrid_start: str = "self_attention"


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class SelfAttentionMixer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.backend_name = "self_attention"
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


class OfficialMambaMixer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        d_state: int,
        d_conv: int,
        expand: float,
    ) -> None:
        super().__init__()
        if _OfficialMamba is None:
            raise RuntimeError(
                "mamba_backend='mamba_ssm' was requested, but mamba-ssm is not installed. "
                "Install mamba-ssm/causal-conv1d on the CUDA server or use mamba_backend='torch'."
            )
        self.backend_name = "mamba_ssm"
        self.mamba = _OfficialMamba(
            d_model=hidden_size,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=float(expand),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


class TorchSelectiveSSMMixer(nn.Module):
    """Selective state-space mixer written with plain PyTorch ops (no fused kernel)."""

    def __init__(
        self,
        hidden_size: int,
        d_state: int,
        d_conv: int,
        expand: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backend_name = "torch_selective_ssm"
        self.hidden_size = int(hidden_size)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.inner_size = max(1, int(round(hidden_size * float(expand))))
        self.dt_rank = max(4, math.ceil(hidden_size / 16))

        self.in_proj = nn.Linear(hidden_size, 2 * self.inner_size)
        self.conv1d = nn.Conv1d(
            self.inner_size,
            self.inner_size,
            kernel_size=self.d_conv,
            groups=self.inner_size,
            padding=self.d_conv - 1,
        )
        self.x_proj = nn.Linear(self.inner_size, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.inner_size)
        self.out_proj = nn.Linear(self.inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.inner_size, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.inner_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        projected = self.x_proj(x_conv)
        dt_raw, b_raw, c_raw = torch.split(projected, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw)).clamp(max=10.0)
        b_raw = torch.tanh(b_raw)
        c_raw = torch.tanh(c_raw)

        a = -torch.exp(self.A_log).to(dtype=x.dtype, device=x.device)
        d = self.D.to(dtype=x.dtype, device=x.device)
        state = x.new_zeros(batch_size, self.inner_size, self.d_state)
        outputs = []

        # Input-dependent dt, B and C control how each token updates the state.
        for idx in range(seq_len):
            dt_i = dt[:, idx]
            x_i = x_conv[:, idx]
            b_i = b_raw[:, idx]
            c_i = c_raw[:, idx]
            dA = torch.exp(dt_i.unsqueeze(-1) * a.unsqueeze(0)).clamp(min=0.0, max=1.0)
            dB_x = dt_i.unsqueeze(-1) * b_i.unsqueeze(1) * x_i.unsqueeze(-1)
            state = dA * state + dB_x
            y_i = torch.sum(state * c_i.unsqueeze(1), dim=-1) + d * x_i
            outputs.append(y_i)

        y = torch.stack(outputs, dim=1)
        y = y * F.silu(z)
        return self.dropout(self.out_proj(y))


def build_mixer(
    mixer_type: str,
    hidden_size: int,
    num_heads: int,
    dropout: float,
    mamba_backend: str,
    mamba_d_state: int,
    mamba_d_conv: int,
    mamba_expand: float,
) -> nn.Module:
    if mixer_type == "self_attention":
        return SelfAttentionMixer(hidden_size, num_heads, dropout)
    if mixer_type == "mamba":
        backend = str(mamba_backend)
        if backend == "auto":
            backend = "mamba_ssm" if _OfficialMamba is not None else "torch"
        if backend == "mamba_ssm":
            return OfficialMambaMixer(hidden_size, mamba_d_state, mamba_d_conv, mamba_expand)
        if backend == "torch":
            return TorchSelectiveSSMMixer(hidden_size, mamba_d_state, mamba_d_conv, mamba_expand, dropout)
        raise ValueError(f"Unsupported mamba_backend: {mamba_backend}")
    raise ValueError(f"Unsupported mixer_type: {mixer_type}")


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        inner = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        mixer_type: str,
        mamba_backend: str,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mixer = build_mixer(
            mixer_type=mixer_type,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            mamba_backend=mamba_backend,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(hidden_size, mlp_ratio, dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.mixer(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class MiniDiT(nn.Module):
    def __init__(self, config: MiniDiTConfig) -> None:
        super().__init__()
        if config.mixer_type not in {"self_attention", "mamba", "hybrid"}:
            raise ValueError(f"Unsupported mixer_type for Week 4: {config.mixer_type}")
        if config.hybrid_start not in {"self_attention", "mamba"}:
            raise ValueError(f"Unsupported hybrid_start: {config.hybrid_start}")

        self.config = config
        self.input_size = tuple(config.input_size)
        self.patch_size = tuple(config.patch_size)
        self.padded_size = tuple(
            math.ceil(size / patch) * patch for size, patch in zip(self.input_size, self.patch_size)
        )
        self.grid_size = tuple(size // patch for size, patch in zip(self.padded_size, self.patch_size))
        num_patches = self.grid_size[0] * self.grid_size[1]

        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.label_embed = nn.Embedding(config.num_classes, config.hidden_size)
        block_mixer_types = self._resolve_block_mixer_types(config)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                    mixer_type=block_mixer_types[idx],
                    mamba_backend=config.mamba_backend,
                    mamba_d_state=config.mamba_d_state,
                    mamba_d_conv=config.mamba_d_conv,
                    mamba_expand=config.mamba_expand,
                )
                for idx in range(config.depth)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size),
        )
        self.unpatch = nn.ConvTranspose2d(
            config.hidden_size,
            config.out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.initialize_weights()

    @staticmethod
    def _resolve_block_mixer_types(config: MiniDiTConfig) -> list[str]:
        if config.mixer_type in {"self_attention", "mamba"}:
            return [config.mixer_type for _ in range(config.depth)]
        first = config.hybrid_start
        second = "mamba" if first == "self_attention" else "self_attention"
        return [first if idx % 2 == 0 else second for idx in range(config.depth)]

    def mixer_summary(self) -> list[dict[str, str]]:
        summary = []
        for idx, block in enumerate(self.blocks):
            mixer = block.mixer
            summary.append(
                {
                    "block": str(idx),
                    "class": type(mixer).__name__,
                    "backend": str(getattr(mixer, "backend_name", "unknown")),
                }
            )
        return summary

    def initialize_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.label_embed.weight, std=0.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.unpatch.weight)
        nn.init.zeros_(self.unpatch.bias)

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, freq, time = x.shape
        pad_freq = self.padded_size[0] - freq
        pad_time = self.padded_size[1] - time
        return F.pad(x, (0, pad_time, 0, pad_freq))

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        mixture_spec: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if xt.shape != mixture_spec.shape:
            raise ValueError(f"xt and mixture_spec shape mismatch: {tuple(xt.shape)} vs {tuple(mixture_spec.shape)}")
        if tuple(xt.shape[-2:]) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {tuple(xt.shape[-2:])}")

        x = torch.cat([xt, mixture_spec], dim=1)
        x = self.patch_embed(self._pad(x))
        batch_size, channels, grid_h, grid_w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        cond = self.t_embedder(t) + self.label_embed(labels)
        for block in self.blocks:
            x = block(x, cond)

        shift, scale = self.final_modulation(cond).chunk(2, dim=1)
        x = modulate(self.final_norm(x), shift, scale)
        x = x.transpose(1, 2).reshape(batch_size, channels, grid_h, grid_w)
        x = self.unpatch(x)
        return x[:, :, : self.input_size[0], : self.input_size[1]]
