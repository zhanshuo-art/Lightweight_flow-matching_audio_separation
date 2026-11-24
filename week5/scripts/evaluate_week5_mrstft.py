#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import ensure_dir, load_config, load_torch_checkpoint, read_json
from sample_week5 import build_dataset, euler_sample
from train_mamba_waveform_ft import build_feature_codec, build_model, resolve_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "mamba_waveform_finetune.yaml"
DEFAULT_WEEK4 = ROOT.parent / "week4" / "runs" / "minidit_mamba_v1_bs48" / "checkpoints" / "best.pt"
DEFAULT_WEEK5 = ROOT / "runs" / "mamba_waveform_ft_v1" / "checkpoints" / "best.pt"
FIGURE_DIRS = [
    ROOT / "analysis" / "figures",
    ROOT / "\u5c55\u793appt\u751f\u6210" / "figure",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multi-resolution STFT L1 for Week4/Week5 models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--week4-checkpoint", default=str(DEFAULT_WEEK4))
    parser.add_argument("--week5-checkpoint", default=str(DEFAULT_WEEK5))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=str(ROOT / "analysis" / "mrstft_metrics"))
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def mrstft_l1(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    n_fft_values: tuple[int, ...] = (256, 512, 1024),
) -> dict[int | str, torch.Tensor]:
    values: dict[int | str, torch.Tensor] = {}
    total = None
    for n_fft in n_fft_values:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=estimate.device, dtype=estimate.dtype)
        est = torch.stft(
            estimate,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        ref = torch.stft(
            reference,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        value = torch.mean(torch.abs(torch.log1p(est.abs()) - torch.log1p(ref.abs())), dim=(1, 2))
        values[n_fft] = value
        total = value if total is None else total + value
    values["mean"] = total / float(len(n_fft_values))
    return values


def add_batch_values(store: dict[str, list[float]], prefix: str, values: dict[int | str, torch.Tensor]) -> None:
    for key, tensor in values.items():
        name = f"{prefix}_mrstft_l1" if key == "mean" else f"{prefix}_mrstft_l1_{key}"
        store.setdefault(name, []).extend(float(item) for item in tensor.detach().cpu())


@torch.no_grad()
def evaluate_model(
    *,
    model_name: str,
    checkpoint_path: Path,
    config: dict[str, Any],
    dataset,
    loader: DataLoader,
    feature_extractor,
    device: torch.device,
    num_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model = build_model(config).to(device)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    values: dict[str, list[float]] = {}
    sample_rows: list[dict[str, Any]] = []
    record_offset = 0
    for batch in tqdm(loader, desc=f"MR-STFT {model_name}"):
        mixture = batch["mixture_waveform"].to(device=device, non_blocking=True)
        target = batch["target_waveform"].to(device=device, non_blocking=True)
        residual = batch["residual_waveform"].to(device=device, non_blocking=True)
        labels = batch["label"].to(device=device, non_blocking=True)

        mixture_spec = feature_extractor.waveform_to_features(mixture)
        pred_spec = euler_sample(model, mixture_spec, mixture_spec, labels, int(num_steps), feature_extractor)
        pred_target = feature_extractor.reconstruct_waveform(
            pred_spec,
            mixture,
            length=int(config["audio"]["num_samples"]),
        )
        pred_residual = mixture - pred_target

        add_batch_values(values, "mixture_target", mrstft_l1(mixture, target))
        add_batch_values(values, "prediction_target", mrstft_l1(pred_target, target))
        add_batch_values(values, "prediction_residual", mrstft_l1(pred_residual, residual))

        batch_size = mixture.shape[0]
        pred_target_mrstft = mrstft_l1(pred_target, target)["mean"].detach().cpu()
        pred_residual_mrstft = mrstft_l1(pred_residual, residual)["mean"].detach().cpu()
        for batch_idx in range(batch_size):
            rec = dataset.records[record_offset + batch_idx]
            source_labels = [str(label) for label in rec["source_labels"]]
            target_label = str(rec["target_label"])
            residual_label = next((label for label in source_labels if label != target_label), "")
            sample_rows.append(
                {
                    "model": model_name,
                    "sample_id": str(rec["sample_id"]),
                    "mix_id": str(rec["mix_id"]),
                    "target_label": target_label,
                    "residual_label": residual_label,
                    "source_pair": " + ".join(source_labels),
                    "prediction_target_mrstft_l1": float(pred_target_mrstft[batch_idx]),
                    "prediction_residual_mrstft_l1": float(pred_residual_mrstft[batch_idx]),
                }
            )
        record_offset += batch_size

    summary: dict[str, Any] = {
        "model": model_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "num_steps": int(num_steps),
        "num_samples": len(dataset),
    }
    for key, items in sorted(values.items()):
        summary[key] = mean(items)
    return summary, sample_rows


def write_markdown(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    week4 = next(row for row in rows if row["model"] == "Week4 Mamba")
    week5 = next(row for row in rows if row["model"] == "Week5 Mamba FT")
    delta_target = float(week5["prediction_target_mrstft_l1"]) - float(week4["prediction_target_mrstft_l1"])
    delta_residual = float(week5["prediction_residual_mrstft_l1"]) - float(week4["prediction_residual_mrstft_l1"])
    lines = [
        "# Week5 Multi-resolution STFT 离线指标",
        "",
        "该指标不参与训练，只在 test split 上评估最终音频波形。计算方式为 `n_fft = 256 / 512 / 1024` 的 `log1p magnitude L1` 平均值。",
        "",
        "| Model | Samples | Target MR-STFT L1 ↓ | Residual MR-STFT L1 ↓ | Mixture Target MR-STFT L1 ↓ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {num_samples} | {prediction_target_mrstft_l1:.5f} | "
            "{prediction_residual_mrstft_l1:.5f} | {mixture_target_mrstft_l1:.5f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- Week5 相对 Week4 的 target MR-STFT L1 变化为 `{delta_target:+.5f}`。",
            f"- Week5 相对 Week4 的 residual MR-STFT L1 变化为 `{delta_residual:+.5f}`。",
            "- 这个指标从多尺度频谱角度补充 Spec L1 / Wave L1 / SI-SDR，帮助解释单一频谱 L1 与最终音频指标不完全一致的情况。",
        ]
    )
    (output_dir / "mrstft_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, Any]]) -> None:
    labels = [row["model"] for row in rows]
    target = [float(row["prediction_target_mrstft_l1"]) for row in rows]
    residual = [float(row["prediction_residual_mrstft_l1"]) for row in rows]
    x = range(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar([idx - width / 2 for idx in x], target, width=width, label="Target", color="#8e1620")
    ax.bar([idx + width / 2 for idx in x], residual, width=width, label="Residual", color="#2c7a5b")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("MR-STFT L1, lower is better")
    ax.set_title("Week4 vs Week5 multi-resolution STFT metric")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for idx, value in enumerate(target):
        ax.text(idx - width / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(residual):
        ax.text(idx + width / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    for out_dir in FIGURE_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "week5_mrstft_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(Path(args.output_dir))
    config = load_config(args.config)
    device = resolve_device(args.device)
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
    models = [
        ("Week4 Mamba", Path(args.week4_checkpoint)),
        ("Week5 Mamba FT", Path(args.week5_checkpoint)),
    ]

    summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for model_name, checkpoint in models:
        summary, rows = evaluate_model(
            model_name=model_name,
            checkpoint_path=checkpoint,
            config=config,
            dataset=dataset,
            loader=loader,
            feature_extractor=feature_extractor,
            device=device,
            num_steps=int(args.num_steps),
        )
        summaries.append(summary)
        sample_rows.extend(rows)

    write_json(output_dir / "mrstft_comparison_summary.json", {"rows": summaries})
    write_csv(output_dir / "mrstft_comparison_summary.csv", summaries)
    write_csv(output_dir / "mrstft_per_sample.csv", sample_rows)
    write_markdown(output_dir, summaries)
    plot(summaries)
    print(json.dumps({"rows": summaries}, ensure_ascii=False, indent=2))
    print(f"Saved MR-STFT outputs to {output_dir}")


if __name__ == "__main__":
    main()
