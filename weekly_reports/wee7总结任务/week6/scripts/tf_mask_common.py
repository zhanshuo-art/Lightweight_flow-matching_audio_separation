#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEK5_SCRIPTS = PROJECT_ROOT / "week5" / "scripts"
if str(WEEK5_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(WEEK5_SCRIPTS))

from audio_features import STFTFeatureCodec  # noqa: E402


METHOD_ORDER = [
    "week5_direct",
    "tfmask_clip_0_1",
    "wiener_p1",
    "wiener_p2",
    "oracle_irm",
]

MODEL_METHODS = {"week5_direct", "tfmask_clip_0_1", "wiener_p1", "wiener_p2"}
POSTPROCESS_METHODS = {"tfmask_clip_0_1", "wiener_p1", "wiener_p2"}


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / (
        reference.pow(2).sum(dim=-1, keepdim=True) + eps
    )
    target = scale * reference
    noise = estimate - target
    return 10.0 * torch.log10((target.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def features_to_magnitude(features: torch.Tensor, codec: STFTFeatureCodec) -> torch.Tensor:
    return torch.expm1(codec.clamp_features(features).float() / float(codec.sigma)).clamp(min=0.0)


def magnitude_to_features(magnitude: torch.Tensor, codec: STFTFeatureCodec) -> torch.Tensor:
    features = torch.log1p(magnitude.clamp(min=0.0)) * float(codec.sigma)
    return codec.clamp_features(features)


def reduce_stats(values: torch.Tensor) -> torch.Tensor:
    return values.float().mean(dim=tuple(range(1, values.ndim)))


def nan_vector(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), float("nan"), device=device)


def apply_tf_mask_method(
    method: str,
    pred_spec: torch.Tensor,
    mixture_spec: torch.Tensor,
    target_spec: torch.Tensor,
    residual_spec: torch.Tensor,
    codec: STFTFeatureCodec,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if method not in METHOD_ORDER:
        raise ValueError(f"Unknown TF-mask method: {method}")

    pred_mag = features_to_magnitude(pred_spec, codec)
    mix_mag = features_to_magnitude(mixture_spec, codec)
    target_mag = features_to_magnitude(target_spec, codec)
    residual_mag = features_to_magnitude(residual_spec, codec)
    raw_mask = pred_mag / (mix_mag + eps)
    batch_size = pred_spec.shape[0]

    if method == "week5_direct":
        final_spec = codec.clamp_features(pred_spec)
        mask = raw_mask.clamp(0.0, 1.0)
        stats = {
            "clip_ratio": reduce_stats((raw_mask > 1.0).float()),
            "mask_mean": reduce_stats(mask),
            "mask_sparsity": reduce_stats((mask < 0.05).float()),
        }
        return final_spec, stats

    if method == "tfmask_clip_0_1":
        mask = raw_mask.clamp(0.0, 1.0)
        final_mag = mask * mix_mag
        final_spec = magnitude_to_features(final_mag, codec)
        stats = {
            "clip_ratio": reduce_stats((raw_mask > 1.0).float()),
            "mask_mean": reduce_stats(mask),
            "mask_sparsity": reduce_stats((mask < 0.05).float()),
        }
        return final_spec, stats

    if method in {"wiener_p1", "wiener_p2"}:
        power = 1.0 if method == "wiener_p1" else 2.0
        residual_proxy = (mix_mag - pred_mag).clamp(min=0.0)
        numerator = pred_mag.pow(power)
        denominator = numerator + residual_proxy.pow(power) + eps
        mask = (numerator / denominator).clamp(0.0, 1.0)
        final_mag = mask * mix_mag
        final_spec = magnitude_to_features(final_mag, codec)
        stats = {
            "clip_ratio": reduce_stats((raw_mask > 1.0).float()),
            "mask_mean": reduce_stats(mask),
            "mask_sparsity": reduce_stats((mask < 0.05).float()),
        }
        return final_spec, stats

    irm = target_mag.pow(2) / (target_mag.pow(2) + residual_mag.pow(2) + eps)
    final_mag = irm * mix_mag
    final_spec = magnitude_to_features(final_mag, codec)
    stats = {
        "clip_ratio": nan_vector(batch_size, pred_spec.device),
        "mask_mean": reduce_stats(irm),
        "mask_sparsity": reduce_stats((irm < 0.05).float()),
    }
    return final_spec, stats


def compute_irm_loss(
    endpoint_pred: torch.Tensor,
    mixture_spec: torch.Tensor,
    target_spec: torch.Tensor,
    residual_spec: torch.Tensor,
    codec: STFTFeatureCodec,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_mag = features_to_magnitude(endpoint_pred, codec)
    mix_mag = features_to_magnitude(mixture_spec, codec)
    target_mag = features_to_magnitude(target_spec, codec)
    residual_mag = features_to_magnitude(residual_spec, codec)
    soft_mask = (pred_mag / (mix_mag + eps)).clamp(0.0, 1.0)
    irm = target_mag.pow(2) / (target_mag.pow(2) + residual_mag.pow(2) + eps)
    return F.l1_loss(soft_mask, irm)


def safe_float(value: Any) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return output


def fmt_float(value: Any, digits: int = 5) -> str:
    number = safe_float(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.{digits}f}"


def average(rows: list[dict[str, Any]], key: str) -> float:
    values = [safe_float(row.get(key)) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    return mean(values) if values else float("nan")


def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)

