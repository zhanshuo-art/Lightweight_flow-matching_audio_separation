#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot final Week 3 Mini-DiT experiment results.")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_training_curves(metrics_path: Path, output_dir: Path) -> None:
    rows = read_csv(metrics_path)
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [
        ("loss", "Total loss"),
        ("velocity_loss", "Velocity loss"),
        ("endpoint_loss", "Endpoint loss"),
    ]
    for ax, (suffix, title) in zip(axes, specs):
        ax.plot(epochs, [float(row[f"train_{suffix}"]) for row in rows], label="train", linewidth=2)
        ax.plot(epochs, [float(row[f"val_{suffix}"]) for row in rows], label="val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("L1")
        ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def save_eval_overview(val_summary: dict[str, Any], test_summary: dict[str, Any] | None, output_dir: Path) -> None:
    splits = [("Val", val_summary)]
    if test_summary is not None:
        splits.append(("Test", test_summary))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("spectrogram_l1", "Frequency-feature L1", "mixture_mean", "prediction_mean"),
        ("waveform_l1", "Waveform L1", "mixture_mean", "prediction_mean"),
        ("si_sdr", "SI-SDR", "mixture_mean_db", "prediction_mean_db"),
    ]
    colors = ["#355070", "#EAAC8B"]
    for ax, (key, title, mix_key, pred_key) in zip(axes, metrics):
        x = list(range(len(splits)))
        width = 0.36
        mix_values = [summary[key][mix_key] for _, summary in splits]
        pred_values = [summary[key][pred_key] for _, summary in splits]
        ax.bar([idx - width / 2 for idx in x], mix_values, width=width, label="mixture", color=colors[0])
        ax.bar([idx + width / 2 for idx in x], pred_values, width=width, label="prediction", color=colors[1])
        ax.set_title(title)
        ax.set_xticks(x, [name for name, _ in splits])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "eval_overview.png", dpi=180)
    plt.close(fig)


def save_label_plot(
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]] | None,
    output_dir: Path,
) -> None:
    labels = [row["target_label"] for row in val_rows]
    val_map = {row["target_label"]: float(row["si_sdr_improvement_db"]) for row in val_rows}
    test_map = (
        {row["target_label"]: float(row["si_sdr_improvement_db"]) for row in test_rows}
        if test_rows is not None
        else None
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([idx - width / 2 for idx in x], [val_map[label] for label in labels], width=width, label="val", color="#6D597A")
    if test_map is not None:
        ax.bar(
            [idx + width / 2 for idx in x],
            [test_map[label] for label in labels],
            width=width,
            label="test",
            color="#B56576",
        )
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_title("SI-SDR improvement by target label")
    ax.set_ylabel("dB")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "si_sdr_improvement_by_label.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics)
    val_dir = Path(args.val_dir)
    test_dir = Path(args.test_dir) if args.test_dir else None

    val_summary = read_json(val_dir / "summary.json")
    test_summary = read_json(test_dir / "summary.json") if test_dir is not None else None
    val_by_label = read_csv(val_dir / "by_label.csv")
    test_by_label = read_csv(test_dir / "by_label.csv") if test_dir is not None else None

    save_training_curves(metrics_path, output_dir)
    save_eval_overview(val_summary, test_summary, output_dir)
    save_label_plot(val_by_label, test_by_label, output_dir)
    print(f"Saved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
