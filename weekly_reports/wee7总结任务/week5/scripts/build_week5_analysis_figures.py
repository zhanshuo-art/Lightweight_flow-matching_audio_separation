#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
METRICS_CSV = ROOT / "runs" / "mamba_waveform_ft_v1" / "metrics.csv"
COMPARISON_JSON = ROOT / "analysis" / "week5_comparison_summary.json"
CLASSWISE_CSV = ROOT / "analysis" / "week5_classwise_metrics.csv"
OUT_DIRS = [
    ROOT / "analysis" / "figures",
    ROOT / "\u5c55\u793appt\u751f\u6210" / "figure",
]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_all(fig: plt.Figure, name: str) -> None:
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves() -> None:
    rows = read_csv_rows(METRICS_CSV)
    epochs = [int(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    axes[0].plot(epochs, [float(r["train_loss"]) for r in rows], marker="o", label="train total")
    axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], marker="o", label="val total")
    axes[0].set_title("Week5 continued training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, [float(r["train_waveform_l1"]) for r in rows], marker="o", label="train waveform L1")
    axes[1].plot(epochs, [float(r["val_waveform_l1"]) for r in rows], marker="o", label="val waveform L1")
    axes[1].set_title("Waveform L1 auxiliary term")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Wave L1")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save_all(fig, "week5_training_curves.png")


def plot_metric_comparison() -> None:
    rows = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))["rows"]
    labels = [r["model"] for r in rows]
    metrics = [
        ("target_spec_l1", "Target Spec L1", "lower"),
        ("target_wave_l1", "Target Wave L1", "lower"),
        ("target_si_sdr_db", "Target SI-SDR (dB)", "higher"),
        ("residual_si_sdr_db", "Residual SI-SDR (dB)", "higher"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.0))
    colors = ["#7a7f87", "#8e1620"]
    for ax, (key, title, direction) in zip(axes, metrics):
        values = [float(r[key]) for r in rows]
        ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.22)
        for idx, value in enumerate(values):
            ax.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(
            0.5,
            -0.23,
            f"{direction} is better",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#62666d",
        )
    fig.suptitle("Week4 Mamba vs Week5 fine-tuned Mamba", y=1.04)
    fig.tight_layout()
    save_all(fig, "week5_test_metrics_comparison.png")


def plot_classwise() -> None:
    rows = read_csv_rows(CLASSWISE_CSV)
    rows = sorted(rows, key=lambda r: float(r["prediction_si_sdr_db"]), reverse=True)
    labels = [r["target_label"] for r in rows]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    values = [float(r["prediction_si_sdr_db"]) for r in rows]
    colors = ["#2c7a5b" if v >= 0 else "#8e1620" for v in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Week5 target SI-SDR by target label")
    ax.set_ylabel("SI-SDR (dB)")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    save_all(fig, "week5_classwise_si_sdr.png")

    rows = sorted(rows, key=lambda r: float(r["prediction_wave_l1"]))
    labels = [r["target_label"] for r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.bar(labels, [float(r["prediction_wave_l1"]) for r in rows], color="#8e1620")
    ax.set_title("Week5 target Wave L1 by target label")
    ax.set_ylabel("Wave L1")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    save_all(fig, "week5_classwise_wave_l1.png")


def main() -> None:
    plot_training_curves()
    plot_metric_comparison()
    plot_classwise()
    print("Saved Week5 analysis figures.")


if __name__ == "__main__":
    main()
