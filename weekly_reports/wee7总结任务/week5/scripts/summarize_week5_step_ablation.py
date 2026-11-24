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
IN_DIR = ROOT / "analysis" / "inference_steps"
OUT_CSV = IN_DIR / "step_ablation_summary.csv"
OUT_MD = IN_DIR / "step_ablation_summary.md"
FIGURE_DIRS = [
    ROOT / "analysis" / "figures",
    ROOT / "\u5c55\u793appt\u751f\u6210" / "figure",
]


def read_summary(step: int) -> dict:
    path = IN_DIR / f"steps_{step}" / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation summary: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_summary(step: int, summary: dict) -> dict[str, float | int]:
    return {
        "num_steps": step,
        "num_samples": int(summary["num_samples"]),
        "target_spec_l1": float(summary["spectrogram_l1"]["prediction_mean"]),
        "target_wave_l1": float(summary["waveform_l1"]["prediction_mean"]),
        "target_si_sdr_db": float(summary["si_sdr"]["prediction_mean_db"]),
        "target_si_sdr_improvement_db": float(summary["si_sdr"]["improvement_mean_db"]),
        "residual_wave_l1": float(summary["residual_waveform_l1"]["prediction_mean"]),
        "residual_si_sdr_db": float(summary["residual_si_sdr"]["prediction_mean_db"]),
        "mixture_consistency_l1": float(summary["mixture_consistency_l1"]),
    }


def write_csv(rows: list[dict[str, float | int]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, float | int]]) -> None:
    best_si_sdr = max(rows, key=lambda row: float(row["target_si_sdr_db"]))
    best_wave = min(rows, key=lambda row: float(row["target_wave_l1"]))
    best_spec = min(rows, key=lambda row: float(row["target_spec_l1"]))
    sample_count = int(rows[0]["num_samples"])
    all_same_n = all(int(row["num_samples"]) == sample_count for row in rows)
    scope = (
        f"本实验使用 test split 前 {sample_count} 条 target records。"
        if all_same_n
        else "不同步数的样本数不完全一致，结果只作趋势参考。"
    )

    lines = [
        "# Week5 推理步数对比",
        "",
        scope,
        "",
        "| Steps | Samples | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {num_steps} | {num_samples} | {target_spec_l1:.5f} | {target_wave_l1:.5f} | "
            "{target_si_sdr_db:.3f} dB | {residual_si_sdr_db:.3f} dB |".format(**row)
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- Target SI-SDR 最好的步数是 `{int(best_si_sdr['num_steps'])}`，数值为 `{float(best_si_sdr['target_si_sdr_db']):.3f} dB`。",
            f"- Target Wave L1 最好的步数是 `{int(best_wave['num_steps'])}`，数值为 `{float(best_wave['target_wave_l1']):.5f}`。",
            f"- Target Spec L1 最好的步数是 `{int(best_spec['num_steps'])}`，数值为 `{float(best_spec['target_spec_l1']):.5f}`。",
            "- 如果多步没有稳定优于 1-step，说明当前瓶颈主要不在采样步数，而更可能在特征表示、phase 重建或模型容量。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, float | int]]) -> None:
    steps = [int(row["num_steps"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))

    axes[0].plot(steps, [float(row["target_spec_l1"]) for row in rows], marker="o", color="#7a7f87")
    axes[0].set_title("Target Spec L1")
    axes[0].set_xlabel("Inference steps")
    axes[0].set_ylabel("Lower is better")
    axes[0].grid(alpha=0.25)

    axes[1].plot(steps, [float(row["target_wave_l1"]) for row in rows], marker="o", color="#8e1620")
    axes[1].set_title("Target Wave L1")
    axes[1].set_xlabel("Inference steps")
    axes[1].set_ylabel("Lower is better")
    axes[1].grid(alpha=0.25)

    axes[2].plot(steps, [float(row["target_si_sdr_db"]) for row in rows], marker="o", color="#2c7a5b")
    axes[2].set_title("Target SI-SDR")
    axes[2].set_xlabel("Inference steps")
    axes[2].set_ylabel("dB, higher is better")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Week5 inference step ablation", y=1.03)
    fig.tight_layout()
    for out_dir in FIGURE_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "week5_inference_steps_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = [flatten_summary(step, read_summary(step)) for step in (1, 2, 4, 8)]
    write_csv(rows)
    write_markdown(rows)
    plot(rows)
    print(f"Saved step ablation summary to {OUT_CSV}")


if __name__ == "__main__":
    main()
