#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from audio_features import STFTFeatureCodec
from common import ensure_dir, load_config, load_torch_checkpoint, read_json, set_seed, write_json
from dataset import SeparationManifestDataset
from train_mamba_waveform_ft import build_feature_codec, build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Week5 target and residual audio from a trained Mini-DiT checkpoint.")
    parser.add_argument("--config", default="week5/configs/mamba_waveform_finetune.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    scale = (estimate * reference).sum(dim=-1, keepdim=True) / (
        reference.pow(2).sum(dim=-1, keepdim=True) + eps
    )
    target = scale * reference
    noise = estimate - target
    return 10.0 * torch.log10((target.pow(2).sum(dim=-1) + eps) / (noise.pow(2).sum(dim=-1) + eps))


def build_dataset(config: dict[str, Any], split: str, limit: int | None = None) -> SeparationManifestDataset:
    data_root = Path(config["paths"]["data_root"])
    class_names = list(config["dataset"]["class_names"])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    return SeparationManifestDataset(
        manifest_path=data_root / "metadata" / f"manifest_{split}.jsonl",
        data_root=data_root,
        class_to_idx=class_to_idx,
        sample_rate=int(config["audio"]["sample_rate"]),
        num_samples=int(config["audio"]["num_samples"]),
        limit=limit,
    )


@torch.no_grad()
def euler_sample(
    model: torch.nn.Module,
    start_spec: torch.Tensor,
    mixture_spec: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    feature_extractor: STFTFeatureCodec,
) -> torch.Tensor:
    x = start_spec.clone()
    dt = 1.0 / float(num_steps)
    batch_size = x.shape[0]
    for step in range(num_steps):
        t = torch.full((batch_size,), step / float(num_steps), device=x.device, dtype=x.dtype)
        velocity = model(x, t, mixture_spec, labels)
        x = x + dt * velocity
    return feature_extractor.clamp_features(x)


def save_triplet_figure(
    path: Path,
    mixture_logmag: torch.Tensor,
    target_logmag: torch.Tensor,
    pred_logmag: torch.Tensor,
    title: str,
) -> None:
    arrays = [
        ("Mixture", mixture_logmag.squeeze().detach().cpu().numpy()),
        ("Target", target_logmag.squeeze().detach().cpu().numpy()),
        ("Prediction", pred_logmag.squeeze().detach().cpu().numpy()),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmax = max(float(arr.max()) for _, arr in arrays) if arrays else 1.0
    for ax, (name, arr) in zip(axes, arrays):
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Frequency bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["paths"]["data_root"] = Path(args.data_root).resolve()

    set_seed(args.seed)
    device = resolve_device(args.device)
    stats_path = config["paths"].get("feature_stats_path", config["paths"].get("sigma_stats_path"))
    if stats_path is None:
        raise KeyError("Expected paths.feature_stats_path or paths.sigma_stats_path in config.")
    sigma = float(read_json(stats_path)["sigma"])
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(args.checkpoint).resolve().parent.parent / "samples"
    ensure_dir(output_dir)

    dataset = build_dataset(config, split=args.split, limit=args.num_samples)
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    mixture_waveform = batch["mixture_waveform"].to(device)
    target_waveform = batch["target_waveform"].to(device)
    residual_waveform = batch["residual_waveform"].to(device)
    labels = batch["label"].to(device)

    feature_extractor = build_feature_codec(config, sigma).to(device)
    model = build_model(config).to(device)
    checkpoint = load_torch_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mixture_spec = feature_extractor.waveform_to_features(mixture_waveform)
    target_spec = feature_extractor.waveform_to_features(target_waveform)
    flow_start = str(config["training"]["flow_start"])
    if flow_start != "mixture":
        raise ValueError(f"Unsupported flow_start for sampling: {flow_start}")
    pred_spec = euler_sample(model, mixture_spec, mixture_spec, labels, args.num_steps, feature_extractor)

    pred_waveform = feature_extractor.reconstruct_waveform(
        pred_spec,
        mixture_waveform,
        length=int(config["audio"]["num_samples"]),
    )
    pred_residual_waveform = mixture_waveform - pred_waveform

    rows = []
    sample_rate = int(config["audio"]["sample_rate"])
    for idx in range(len(dataset)):
        sample_dir = ensure_dir(output_dir / f"sample_{idx:02d}")
        rec = dataset.records[idx]
        sample_id = str(rec["sample_id"])
        target_label = str(rec["target_label"])
        source_labels = [str(label) for label in rec["source_labels"]]
        residual_label = next((label for label in source_labels if label != target_label), "")
        source_pair = " + ".join(source_labels)
        mixture_l1 = torch.mean(torch.abs(mixture_spec[idx] - target_spec[idx])).item()
        pred_l1 = torch.mean(torch.abs(pred_spec[idx] - target_spec[idx])).item()
        pred_target_wave_l1 = torch.mean(torch.abs(pred_waveform[idx] - target_waveform[idx])).item()
        pred_residual_wave_l1 = torch.mean(torch.abs(pred_residual_waveform[idx] - residual_waveform[idx])).item()
        pred_target_si_sdr = si_sdr(pred_waveform[idx : idx + 1], target_waveform[idx : idx + 1]).item()
        pred_residual_si_sdr = si_sdr(
            pred_residual_waveform[idx : idx + 1],
            residual_waveform[idx : idx + 1],
        ).item()

        sf.write(sample_dir / "mixture.wav", mixture_waveform[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "target.wav", target_waveform[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "residual_target.wav", residual_waveform[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "prediction_target.wav", pred_waveform[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "prediction_residual.wav", pred_residual_waveform[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "prediction.wav", pred_waveform[idx].detach().cpu().numpy(), sample_rate)
        save_triplet_figure(
            sample_dir / "spectrogram_triplet.png",
            feature_extractor.features_to_log_magnitude(mixture_spec[idx : idx + 1], mixture_waveform[idx : idx + 1]),
            feature_extractor.features_to_log_magnitude(target_spec[idx : idx + 1], target_waveform[idx : idx + 1]),
            feature_extractor.features_to_log_magnitude(pred_spec[idx : idx + 1], mixture_waveform[idx : idx + 1]),
            title=f"{sample_id} | target={target_label}",
        )
        rows.append(
            {
                "sample_id": sample_id,
                "target_label": target_label,
                "residual_label": residual_label,
                "source_pair": source_pair,
                "mixture_to_target_l1": mixture_l1,
                "prediction_to_target_l1": pred_l1,
                "improvement_l1": mixture_l1 - pred_l1,
                "prediction_target_wave_l1": pred_target_wave_l1,
                "prediction_target_si_sdr_db": pred_target_si_sdr,
                "prediction_residual_wave_l1": pred_residual_wave_l1,
                "prediction_residual_si_sdr_db": pred_residual_si_sdr,
            }
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "representation": str(config["spectrogram"]["representation"]),
        "flow_start": flow_start,
        "split": args.split,
        "num_samples": len(rows),
        "num_steps": int(args.num_steps),
        "mean_mixture_to_target_l1": sum(row["mixture_to_target_l1"] for row in rows) / len(rows),
        "mean_prediction_to_target_l1": sum(row["prediction_to_target_l1"] for row in rows) / len(rows),
        "mean_improvement_l1": sum(row["improvement_l1"] for row in rows) / len(rows),
        "mean_prediction_target_wave_l1": sum(row["prediction_target_wave_l1"] for row in rows) / len(rows),
        "mean_prediction_target_si_sdr_db": sum(row["prediction_target_si_sdr_db"] for row in rows) / len(rows),
        "mean_prediction_residual_wave_l1": sum(row["prediction_residual_wave_l1"] for row in rows) / len(rows),
        "mean_prediction_residual_si_sdr_db": sum(row["prediction_residual_si_sdr_db"] for row in rows) / len(rows),
        "samples": rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved samples to: {output_dir}")


if __name__ == "__main__":
    main()
