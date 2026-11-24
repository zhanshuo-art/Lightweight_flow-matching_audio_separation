#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


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
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mixer = SelfAttentionMixer(hidden_size, num_heads, dropout)
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
        if config.mixer_type != "self_attention":
            raise ValueError(f"Unsupported mixer_type for Week 3 baseline: {config.mixer_type}")

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
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio, config.dropout)
                for _ in range(config.depth)
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
