#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regenerate report summary figures from saved result files.

Run from the report_source directory:

    python make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE / "figures"

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["savefig.dpi"] = 200
matplotlib.rcParams["figure.dpi"] = 120

GREEN = "#3f7f5f"
BLUE = "#3a6ea5"
RED = "#9d2b30"


def _read_class_si_sdr() -> list[tuple[str, float]]:
    path = RESULTS / "week6_irm_ft_by_label.csv"
    rows = [
        r for r in csv.DictReader(path.open(encoding="utf-8"))
        if r["method"] == "tfmask_clip_0_1"
    ]
    pairs = [(r["target_label"], float(r["target_si_sdr_db"])) for r in rows]
    return sorted(pairs, key=lambda x: x[1])


def _read_si_sdr_progression() -> list[float]:
    files_and_paths = [
        ("week3_self_attention_summary.json", ("si_sdr", "prediction_mean_db")),
        ("week4_mamba_summary.json", ("si_sdr", "prediction_mean_db")),
        ("week5_mamba_ft_summary.json", ("si_sdr", "prediction_mean_db")),
    ]
    values: list[float] = []
    for filename, keys in files_and_paths:
        data = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
        values.append(float(data[keys[0]][keys[1]]))

    final = json.loads((RESULTS / "week6_irm_ft_summary.json").read_text(encoding="utf-8"))
    tfmask = next(row for row in final["summary"] if row["method"] == "tfmask_clip_0_1")
    values.append(float(tfmask["target_si_sdr_db"]))
    return values


def make_progression() -> None:
    stages = ["Week3\nSelf-Attn", "Week4\nMamba", "Week5\nWave L1", "Week6\nIRM+Mask"]
    values = _read_si_sdr_progression()
    colors = [RED, BLUE, GREEN, "#b8860b"]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(range(len(values)), values, color=RED, linewidth=2, zorder=1)
    ax.scatter(range(len(values)), values, c=colors, s=140, zorder=2, edgecolors="white")
    ax.axhline(0.0, color="0.6", linewidth=1)
    for i, v in enumerate(values):
        ax.annotate(f"{v:+.3f} dB", (i, v), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=11, color=colors[i])
    for i in range(len(values) - 1):
        d = values[i + 1] - values[i]
        ax.annotate(f"+{d:.3f} dB", ((i + i + 1) / 2, (values[i] + values[i + 1]) / 2),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=9, color="0.4")
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages)
    ax.set_ylabel("Target SI-SDR (dB)")
    ax.set_ylim(-0.9, 3.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "si_sdr_progression.png", bbox_inches="tight")
    plt.close(fig)


def make_classwise() -> None:
    pairs = _read_class_si_sdr()
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    colors = [GREEN if v >= 3 else (BLUE if v >= 0 else RED) for v in values]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.barh(range(len(values)), values, color=colors)
    ax.axvline(0.0, color="0.35", linewidth=1)
    for i, v in enumerate(values):
        ax.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target SI-SDR (dB)")
    ax.set_xlim(min(values) - 1.2, max(values) + 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "classwise_week6.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    make_progression()
    make_classwise()
    print("Regenerated si_sdr_progression.png and classwise_week6.png")


if __name__ == "__main__":
    main()
