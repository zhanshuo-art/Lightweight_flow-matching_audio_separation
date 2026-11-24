#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
from torch import nn


SUPPORTED_REPRESENTATIONS = {"log_magnitude", "complex_signed_log"}


class STFTFeatureCodec(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        sigma: float,
        representation: str,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        clamp_min: float = -1.0,
        clamp_max: float = 1.0,
    ) -> None:
        super().__init__()
        if representation not in SUPPORTED_REPRESENTATIONS:
            raise ValueError(f"Unsupported STFT representation: {representation}")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.sigma = float(sigma)
        self.representation = str(representation)
        self.center = bool(center)
        self.pad_mode = str(pad_mode)
        self.normalized = bool(normalized)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)

    @property
    def channels(self) -> int:
        return 1 if self.representation == "log_magnitude" else 2

    def complex_stft(self, waveforms: torch.Tensor) -> torch.Tensor:
        if waveforms.ndim != 2:
            raise ValueError(f"Expected [B, T] waveforms, got {tuple(waveforms.shape)}")
        return torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=waveforms.device, dtype=waveforms.dtype),
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=self.normalized,
            return_complex=True,
        )

    def clamp_features(self, features: torch.Tensor) -> torch.Tensor:
        return torch.clamp(features, self.clamp_min, self.clamp_max)

    def _signed_log1p(self, values: torch.Tensor) -> torch.Tensor:
        compressed = torch.sign(values) * torch.log1p(torch.abs(values)) * self.sigma
        return self.clamp_features(compressed)

    def _signed_expm1(self, values: torch.Tensor) -> torch.Tensor:
        values = self.clamp_features(values)
        return torch.sign(values) * torch.expm1(torch.abs(values) / self.sigma)

    def waveform_to_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        complex_spec = self.complex_stft(waveforms)
        if self.representation == "log_magnitude":
            magnitude = complex_spec.abs()
            logmag = torch.log1p(magnitude) * self.sigma
            return self.clamp_features(logmag).unsqueeze(1)

        real = self._signed_log1p(complex_spec.real)
        imag = self._signed_log1p(complex_spec.imag)
        return torch.stack([real, imag], dim=1)

    def features_to_complex(self, features: torch.Tensor, mixture_waveforms: torch.Tensor | None = None) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"Expected [B, C, F, T] features, got {tuple(features.shape)}")

        if self.representation == "complex_signed_log":
            if features.shape[1] != 2:
                raise ValueError(f"Expected 2 channels for complex features, got {features.shape[1]}")
            real = self._signed_expm1(features[:, 0])
            imag = self._signed_expm1(features[:, 1])
            return torch.complex(real, imag)

        if features.shape[1] != 1:
            raise ValueError(f"Expected 1 channel for log-magnitude features, got {features.shape[1]}")
        if mixture_waveforms is None:
            raise ValueError("mixture_waveforms are required to reconstruct log-magnitude features.")
        pred_mag = torch.expm1(self.clamp_features(features).squeeze(1) / self.sigma)
        mixture_phase = torch.angle(self.complex_stft(mixture_waveforms))
        return torch.polar(pred_mag, mixture_phase)

    def features_to_log_magnitude(self, features: torch.Tensor, mixture_waveforms: torch.Tensor | None = None) -> torch.Tensor:
        complex_spec = self.features_to_complex(features, mixture_waveforms=mixture_waveforms)
        return torch.log1p(torch.abs(complex_spec))

    def reconstruct_waveform(
        self,
        features: torch.Tensor,
        mixture_waveforms: torch.Tensor | None,
        length: int,
    ) -> torch.Tensor:
        complex_spec = self.features_to_complex(features, mixture_waveforms=mixture_waveforms)
        return torch.istft(
            complex_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=features.device, dtype=features.dtype),
            center=self.center,
            normalized=self.normalized,
            length=length,
        )
