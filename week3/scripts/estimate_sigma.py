#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm

from common import load_config, write_json
from dataset import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate feature scaling sigma from training waveforms.")
    parser.add_argument("--config", default="week3/configs/minidit_self_attention.yaml")
    parser.add_argument("--num-waveforms", type=int, default=200)
    parser.add_argument("--pixels-per-waveform", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = Path(config["paths"]["data_root"])
    spec_cfg = config["spectrogram"]
    audio_cfg = config["audio"]
    manifest = load_jsonl(data_root / "metadata" / "manifest_train.jsonl")

    representation = str(spec_cfg["representation"])
    if representation == "log_magnitude":
        unique_paths = sorted({str(rec["mixture_path"]) for rec in manifest})
    elif representation == "complex_signed_log":
        unique_paths = sorted(
            {str(rec["mixture_path"]) for rec in manifest}
            | {str(rec["target_path"]) for rec in manifest}
        )
    else:
        raise ValueError(f"Unsupported representation for sigma estimation: {representation}")

    random.seed(int(config["seed"]))
    sampled_paths = random.sample(unique_paths, min(args.num_waveforms, len(unique_paths)))
    window = torch.hann_window(int(spec_cfg["win_length"]))
    values = []

    for rel_path in tqdm(sampled_paths, desc="Estimating sigma"):
        audio, sr = sf.read(str(data_root / rel_path), dtype="float32", always_2d=True)
        if int(sr) != int(audio_cfg["sample_rate"]):
            raise ValueError(f"Unexpected sample rate for {rel_path}: {sr}")
        wav = torch.from_numpy(audio.mean(axis=1)).float()
        stft = torch.stft(
            wav,
            n_fft=int(spec_cfg["n_fft"]),
            hop_length=int(spec_cfg["hop_length"]),
            win_length=int(spec_cfg["win_length"]),
            window=window,
            center=bool(spec_cfg["center"]),
            pad_mode=str(spec_cfg["pad_mode"]),
            normalized=bool(spec_cfg["normalized"]),
            return_complex=True,
        )
        if representation == "log_magnitude":
            compressed = torch.log1p(stft.abs()).flatten()
        else:
            complex_components = torch.view_as_real(stft)
            compressed = torch.log1p(torch.abs(complex_components)).flatten()
        idx = torch.randperm(compressed.numel())[: int(args.pixels_per_waveform)]
        values.append(compressed[idx])

    merged = torch.cat(values)
    stats = {
        "representation": representation,
        "sigma": 1.0 / torch.quantile(merged, 0.999).item(),
        "mean": merged.mean().item(),
        "p50": torch.quantile(merged, 0.50).item(),
        "p99": torch.quantile(merged, 0.99).item(),
        "p99.9": torch.quantile(merged, 0.999).item(),
        "num_waveforms": len(sampled_paths),
        "pixels_per_waveform": int(args.pixels_per_waveform),
    }
    stats_path = config["paths"].get("feature_stats_path", config["paths"].get("sigma_stats_path"))
    if stats_path is None:
        raise KeyError("Expected paths.feature_stats_path or paths.sigma_stats_path in config.")
    write_json(stats_path, stats)
    print(stats)
    print(f"Saved: {stats_path}")


if __name__ == "__main__":
    main()
