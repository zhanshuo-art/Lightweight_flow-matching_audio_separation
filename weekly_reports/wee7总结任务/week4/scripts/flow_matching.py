#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch


def sample_flow_matching_tuple(
    start_spec: torch.Tensor,
    target_spec: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if start_spec.shape != target_spec.shape:
        raise ValueError(f"start_spec and target_spec shape mismatch: {tuple(start_spec.shape)} vs {tuple(target_spec.shape)}")
    if target_spec.ndim != 4:
        raise ValueError(f"Expected [B, C, F, T] target spectrogram, got {tuple(target_spec.shape)}")
    batch_size = target_spec.shape[0]
    t = torch.rand(batch_size, device=target_spec.device, dtype=target_spec.dtype, generator=generator)
    x0 = start_spec
    t_view = t.view(batch_size, 1, 1, 1)
    xt = (1.0 - t_view) * x0 + t_view * target_spec
    v_star = target_spec - x0
    return x0, xt, v_star, t
