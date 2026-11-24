#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tf_mask_common import MODEL_METHODS, fmt_float, method_sort_key, read_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Week6 TF-mask summary markdown and comparison figure.")
    parser.add_argument("--analysis-dir", default="week6/analysis/tf_mask_postprocess")
    return parser.parse_args()


def f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def method_label(method: str) -> str:
    return {
        "week5_direct": "Week5 direct",
        "tfmask_clip_0_1": "TF-mask clip",
        "wiener_p1": "Wiener p=1",
        "wiener_p2": "Wiener p=2",
        "oracle_irm": "Oracle IRM",
    }.get(method, method)


def best_model_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["method"] in MODEL_METHODS]
    return max(candidates, key=lambda row: f(row, "target_si_sdr_db"), default=None)


def build_figure(rows: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(rows, key=lambda row: method_sort_key(row["method"]))
    labels = [method_label(row["method"]) for row in rows]
    x = range(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    metrics = [
        ("target_spec_l1", "Target Spec L1", "lower"),
        ("target_wave_l1", "Target Wave L1", "lower"),
        ("target_si_sdr_db", "Target SI-SDR (dB)", "higher"),
        ("residual_si_sdr_db", "Residual SI-SDR (dB)", "higher"),
    ]
    colors = ["#6574a6" if row["method"] != "oracle_irm" else "#b48b28" for row in rows]
    for ax, (key, title, direction) in zip(axes.flat, metrics):
        values = [f(row, key) for row in rows]
        ax.bar(x, values, color=colors)
        ax.set_title(f"{title} ({direction} is better)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    rows = sorted(rows, key=lambda row: method_sort_key(row["method"]))
    direct = next((row for row in rows if row["method"] == "week5_direct"), None)
    oracle = next((row for row in rows if row["method"] == "oracle_irm"), None)
    best = best_model_row(rows)
    lines = [
        "# Week6 TF-mask 后处理结果",
        "",
        "## 主结果",
        "",
        "| Method | Target Spec L1 ↓ | Target Wave L1 ↓ | Target SI-SDR ↑ | Residual SI-SDR ↑ | Clip ratio | Mask mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    method_label(row["method"]),
                    fmt_float(row.get("target_spec_l1"), 5),
                    fmt_float(row.get("target_wave_l1"), 5),
                    fmt_float(row.get("target_si_sdr_db"), 3),
                    fmt_float(row.get("residual_si_sdr_db"), 3),
                    fmt_float(row.get("clip_ratio"), 4),
                    fmt_float(row.get("mask_mean"), 4),
                ]
            )
            + " |"
        )

    lines.extend(["", "## 结论口径", ""])
    if direct is not None and best is not None:
        delta = f(best, "target_si_sdr_db") - f(direct, "target_si_sdr_db")
        res_delta = f(best, "residual_si_sdr_db") - f(direct, "residual_si_sdr_db")
        if best["method"] != "week5_direct" and (delta > 0.0 or res_delta > 0.0):
            lines.append(
                f"- 最好的非 oracle 方法是 `{best['method']}`，Target SI-SDR 相比 Week5 direct 变化 "
                f"{delta:+.3f} dB，Residual SI-SDR 变化 {res_delta:+.3f} dB。"
            )
        else:
            lines.append(
                "- 简单 TF-mask / Wiener-style 后处理没有稳定超过 Week5 direct；这说明当前瓶颈可能不只是 magnitude 越界。"
            )
    if direct is not None and oracle is not None:
        oracle_gap = f(oracle, "target_si_sdr_db") - f(direct, "target_si_sdr_db")
        lines.append(
            f"- `oracle_irm` 只作为诊断上界。它相对 Week5 direct 的 Target SI-SDR 差值为 {oracle_gap:+.3f} dB。"
        )
        if not math.isnan(oracle_gap) and oracle_gap >= 0.5:
            lines.append("- oracle gap 达到 0.5 dB 阈值，可以启动 Week6 IRM auxiliary continued training。")
        else:
            lines.append("- oracle gap 不足 0.5 dB 时，不建议再训练；应把 Week6 写成后处理负结果和瓶颈分析。")

    lines.extend(
        [
            "",
            "## 文件",
            "",
            "- `summary.csv`: 方法级别平均指标。",
            "- `samples.csv`: 每条 test record、每个方法的逐样本指标。",
            "- `by_label.csv`: 按 target label 的类别统计。",
            "- `week5_vs_tfmask_metrics.png`: 指标对比图。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir).resolve()
    summary_rows = read_csv(analysis_dir / "summary.csv")
    build_figure(summary_rows, analysis_dir / "week5_vs_tfmask_metrics.png")
    build_markdown(summary_rows, analysis_dir / "summary.md")
    print(f"Saved Week6 report to: {analysis_dir}")


if __name__ == "__main__":
    main()

