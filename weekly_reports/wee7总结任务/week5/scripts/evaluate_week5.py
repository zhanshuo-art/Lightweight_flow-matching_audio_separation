#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_features import STFTFeatureCodec
from common import ensure_dir, load_config, load_torch_checkpoint, read_json, set_seed, write_json
from sample_week5 import build_dataset, euler_sample
from train_mamba_waveform_ft import build_feature_codec, build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained Mini-DiT checkpoint on a full split.")
    parser.add_argument("--config", default="week5/configs/mamba_waveform_finetune.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--output-dir", required=True)
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def avg(key: str) -> float:
        return mean(float(row[key]) for row in rows)

    def ratio(key: str) -> float:
        return mean(float(row[key] > 0.0) for row in rows)

    return {
        "num_samples": len(rows),
        "spectrogram_l1": {
            "mixture_mean": avg("mixture_spec_l1"),
            "prediction_mean": avg("prediction_spec_l1"),
            "improvement_mean": avg("spectrogram_l1_improvement"),
            "improved_ratio": ratio("spectrogram_l1_improvement"),
        },
        "waveform_l1": {
            "mixture_mean": avg("mixture_wave_l1"),
            "prediction_mean": avg("prediction_wave_l1"),
            "improvement_mean": avg("waveform_l1_improvement"),
            "improved_ratio": ratio("waveform_l1_improvement"),
        },
        "si_sdr": {
            "mixture_mean_db": avg("mixture_si_sdr_db"),
            "prediction_mean_db": avg("prediction_si_sdr_db"),
            "improvement_mean_db": avg("si_sdr_improvement_db"),
            "improved_ratio": ratio("si_sdr_improvement_db"),
        },
        "residual_waveform_l1": {
            "prediction_mean": avg("prediction_residual_wave_l1"),
        },
        "residual_si_sdr": {
            "prediction_mean_db": avg("prediction_residual_si_sdr_db"),
        },
        "mixture_consistency_l1": avg("mixture_consistency_l1"),
    }


def grouped_summary(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)

    output: list[dict[str, Any]] = []
    for group_name, group_rows in sorted(groups.items()):
        summary = summarize_rows(group_rows)
        output.append(
            {
                key: group_name,
                "num_samples": summary["num_samples"],
                "mixture_spec_l1": summary["spectrogram_l1"]["mixture_mean"],
                "prediction_spec_l1": summary["spectrogram_l1"]["prediction_mean"],
                "spectrogram_l1_improvement": summary["spectrogram_l1"]["improvement_mean"],
                "spectrogram_l1_improved_ratio": summary["spectrogram_l1"]["improved_ratio"],
                "mixture_wave_l1": summary["waveform_l1"]["mixture_mean"],
                "prediction_wave_l1": summary["waveform_l1"]["prediction_mean"],
                "waveform_l1_improvement": summary["waveform_l1"]["improvement_mean"],
                "waveform_l1_improved_ratio": summary["waveform_l1"]["improved_ratio"],
                "mixture_si_sdr_db": summary["si_sdr"]["mixture_mean_db"],
                "prediction_si_sdr_db": summary["si_sdr"]["prediction_mean_db"],
                "si_sdr_improvement_db": summary["si_sdr"]["improvement_mean_db"],
                "si_sdr_improved_ratio": summary["si_sdr"]["improved_ratio"],
                "prediction_residual_wave_l1": summary["residual_waveform_l1"]["prediction_mean"],
                "prediction_residual_si_sdr_db": summary["residual_si_sdr"]["prediction_mean_db"],
                "mixture_consistency_l1": summary["mixture_consistency_l1"],
            }
        )
    return output


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["paths"]["data_root"] = Path(args.data_root).resolve()

    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    stats_path = config["paths"].get("feature_stats_path", config["paths"].get("sigma_stats_path"))
    if stats_path is None:
        raise KeyError("Expected paths.feature_stats_path or paths.sigma_stats_path in config.")
    sigma = float(read_json(stats_path)["sigma"])
    dataset = build_dataset(config, split=args.split, limit=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    feature_extractor = build_feature_codec(config, sigma).to(device)
    model = build_model(config).to(device)
    checkpoint = load_torch_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    flow_start = str(config["training"]["flow_start"])
    if flow_start != "mixture":
        raise ValueError(f"Unsupported flow_start for evaluation: {flow_start}")

    rows: list[dict[str, Any]] = []
    record_offset = 0
    for batch in tqdm(loader, desc=f"evaluate {args.split}"):
        mixture_waveform = batch["mixture_waveform"].to(device=device, non_blocking=True)
        target_waveform = batch["target_waveform"].to(device=device, non_blocking=True)
        residual_waveform = batch["residual_waveform"].to(device=device, non_blocking=True)
        labels = batch["label"].to(device=device, non_blocking=True)

        mixture_spec = feature_extractor.waveform_to_features(mixture_waveform)
        target_spec = feature_extractor.waveform_to_features(target_waveform)
        prediction_spec = euler_sample(
            model,
            mixture_spec,
            mixture_spec,
            labels,
            int(args.num_steps),
            feature_extractor,
        )
        prediction_waveform = feature_extractor.reconstruct_waveform(
            prediction_spec,
            mixture_waveform,
            length=int(config["audio"]["num_samples"]),
        )
        prediction_residual_waveform = mixture_waveform - prediction_waveform

        mixture_spec_l1 = torch.mean(torch.abs(mixture_spec - target_spec), dim=(1, 2, 3))
        prediction_spec_l1 = torch.mean(torch.abs(prediction_spec - target_spec), dim=(1, 2, 3))
        mixture_wave_l1 = torch.mean(torch.abs(mixture_waveform - target_waveform), dim=1)
        prediction_wave_l1 = torch.mean(torch.abs(prediction_waveform - target_waveform), dim=1)
        prediction_residual_wave_l1 = torch.mean(torch.abs(prediction_residual_waveform - residual_waveform), dim=1)
        mixture_consistency_l1 = torch.mean(
            torch.abs(prediction_waveform + prediction_residual_waveform - mixture_waveform),
            dim=1,
        )
        mixture_si_sdr = si_sdr(mixture_waveform, target_waveform)
        prediction_si_sdr = si_sdr(prediction_waveform, target_waveform)
        prediction_residual_si_sdr = si_sdr(prediction_residual_waveform, residual_waveform)

        batch_size = mixture_waveform.shape[0]
        for batch_idx in range(batch_size):
            rec = dataset.records[record_offset + batch_idx]
            source_labels = [str(label) for label in rec["source_labels"]]
            target_label = str(rec["target_label"])
            residual_label = next((label for label in source_labels if label != target_label), "")
            source_pair = " + ".join(source_labels)
            row = {
                "sample_id": str(rec["sample_id"]),
                "mix_id": str(rec["mix_id"]),
                "target_label": target_label,
                "residual_label": residual_label,
                "source_pair": source_pair,
                "mixture_spec_l1": float(mixture_spec_l1[batch_idx].cpu()),
                "prediction_spec_l1": float(prediction_spec_l1[batch_idx].cpu()),
                "spectrogram_l1_improvement": float(
                    (mixture_spec_l1[batch_idx] - prediction_spec_l1[batch_idx]).cpu()
                ),
                "mixture_wave_l1": float(mixture_wave_l1[batch_idx].cpu()),
                "prediction_wave_l1": float(prediction_wave_l1[batch_idx].cpu()),
                "waveform_l1_improvement": float(
                    (mixture_wave_l1[batch_idx] - prediction_wave_l1[batch_idx]).cpu()
                ),
                "mixture_si_sdr_db": float(mixture_si_sdr[batch_idx].cpu()),
                "prediction_si_sdr_db": float(prediction_si_sdr[batch_idx].cpu()),
                "si_sdr_improvement_db": float((prediction_si_sdr[batch_idx] - mixture_si_sdr[batch_idx]).cpu()),
                "prediction_residual_wave_l1": float(prediction_residual_wave_l1[batch_idx].cpu()),
                "prediction_residual_si_sdr_db": float(prediction_residual_si_sdr[batch_idx].cpu()),
                "mixture_consistency_l1": float(mixture_consistency_l1[batch_idx].cpu()),
            }
            rows.append(row)
        record_offset += batch_size

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "representation": str(config["spectrogram"]["representation"]),
        "flow_start": flow_start,
        "split": args.split,
        "num_steps": int(args.num_steps),
        **summarize_rows(rows),
    }

    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "samples.csv", rows)
    write_csv(output_dir / "by_label.csv", grouped_summary(rows, "target_label"))
    write_csv(output_dir / "by_pair.csv", grouped_summary(rows, "source_pair"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation to: {output_dir}")


if __name__ == "__main__":
    main()
