#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WEEK4_SAMPLES = ROOT / "analysis" / "eval_week4_mamba_1step" / "samples.csv"
WEEK5_LISTENING_SUMMARY = ROOT / "runs" / "mamba_waveform_ft_v1" / "samples_test_20_1step" / "summary.json"
LISTENING_ASSETS = ROOT / "analysis" / "week5_listening_demo" / "assets"
OUT_MD = ROOT / "analysis" / "week5_case_study.md"
OUT_CSV = ROOT / "analysis" / "week5_case_study.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_records() -> list[dict[str, Any]]:
    week4_by_id = {row["sample_id"]: row for row in read_csv(WEEK4_SAMPLES)}
    summary = json.loads(WEEK5_LISTENING_SUMMARY.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(summary["samples"]):
        sample_id = row["sample_id"]
        w4 = week4_by_id[sample_id]
        asset_dir = LISTENING_ASSETS / f"sample_{idx:02d}"
        w5_spec = float(row["prediction_to_target_l1"])
        w4_spec = float(w4["prediction_spec_l1"])
        w5_si_sdr = float(row["prediction_target_si_sdr_db"])
        w4_si_sdr = float(w4["prediction_si_sdr_db"])
        records.append(
            {
                "sample_name": f"sample_{idx:02d}",
                "sample_id": sample_id,
                "source_pair": row["source_pair"],
                "target_label": row["target_label"],
                "residual_label": row["residual_label"],
                "week4_target_spec_l1": w4_spec,
                "week5_target_spec_l1": w5_spec,
                "target_spec_l1_delta": w5_spec - w4_spec,
                "week4_target_si_sdr_db": w4_si_sdr,
                "week5_target_si_sdr_db": w5_si_sdr,
                "target_si_sdr_delta_db": w5_si_sdr - w4_si_sdr,
                "week5_residual_si_sdr_db": float(row["prediction_residual_si_sdr_db"]),
                "week5_target_wave_l1": float(row["prediction_target_wave_l1"]),
                "audio_dir": str(asset_dir.as_posix()),
                "mixture_wav": str((asset_dir / "mixture.wav").as_posix()),
                "target_wav": str((asset_dir / "target.wav").as_posix()),
                "residual_target_wav": str((asset_dir / "residual_target.wav").as_posix()),
                "week4_prediction_target_wav": str((asset_dir / "week4_prediction_target.wav").as_posix()),
                "week5_prediction_target_wav": str((asset_dir / "week5_prediction_target.wav").as_posix()),
                "week5_prediction_residual_wav": str((asset_dir / "week5_prediction_residual.wav").as_posix()),
            }
        )
    return records


def choose_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used: set[str] = set()
    chosen: list[dict[str, Any]] = []

    def add(kind: str, candidates: list[dict[str, Any]], n: int) -> None:
        for item in candidates:
            if item["sample_id"] in used:
                continue
            item = dict(item)
            item["case_type"] = kind
            item["explanation"] = explain_case(kind, item)
            chosen.append(item)
            used.add(item["sample_id"])
            if sum(row["case_type"] == kind for row in chosen) >= n:
                break

    success_candidates = sorted(
        [
            r
            for r in records
            if float(r["target_si_sdr_delta_db"]) > 0.0
            and float(r["week5_target_si_sdr_db"]) > 0.0
            and float(r["week5_residual_si_sdr_db"]) > 0.0
        ],
        key=lambda r: (
            float(r["week5_target_si_sdr_db"]) + float(r["week5_residual_si_sdr_db"]),
            float(r["target_si_sdr_delta_db"]),
        ),
        reverse=True,
    )
    failure_candidates = sorted(
        records,
        key=lambda r: (float(r["target_si_sdr_delta_db"]), float(r["week5_target_si_sdr_db"])),
    )
    borderline_candidates = sorted(
        [
            r
            for r in records
            if (float(r["target_spec_l1_delta"]) > 0.0 and float(r["target_si_sdr_delta_db"]) > 0.0)
            or (float(r["target_spec_l1_delta"]) < 0.0 and float(r["target_si_sdr_delta_db"]) < 0.0)
        ],
        key=lambda r: abs(float(r["target_spec_l1_delta"])) + abs(float(r["target_si_sdr_delta_db"])),
        reverse=True,
    )

    add("success", success_candidates, 3)
    add("failure", failure_candidates, 3)
    add("borderline", borderline_candidates, 2)
    return chosen


def explain_case(kind: str, row: dict[str, Any]) -> str:
    target = str(row["target_label"])
    residual = str(row["residual_label"])
    if kind == "success":
        return (
            f"Week5 target SI-SDR 比 Week4 提升 {float(row['target_si_sdr_delta_db']):.2f} dB。"
            f"该样例中 target={target}，residual={residual}，模型能较好利用 target label 抑制非目标声源。"
        )
    if kind == "failure":
        reason = "目标声源和干扰声源容易在时间或频谱上重叠"
        if target in {"car_horn", "siren"}:
            reason = "警报/鸣笛类声音持续且能量强，容易与背景残留混在一起"
        if target in {"clock_tick", "keyboard_typing", "glass_breaking"}:
            reason = "瞬态声音很短，预测幅度和 mixture phase 的误差会被放大"
        return (
            f"Week5 target SI-SDR 相对 Week4 下降 {abs(float(row['target_si_sdr_delta_db'])):.2f} dB。"
            f"可能原因是：{reason}。"
        )
    return (
        f"该样例的 Spec L1 变化为 {float(row['target_spec_l1_delta']):+.4f}，"
        f"但 SI-SDR 变化为 {float(row['target_si_sdr_delta_db']):+.2f} dB。"
        "这说明单一频谱 L1 与最终音频质量指标并不总是同向。"
    )


def write_markdown(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Week5 成功/失败案例分析",
        "",
        "本文件从现有 20 个试听样例中选择案例，不重新采样、不重新训练。选择依据为 Week4 Mamba 与 Week5 Mamba FT 的 target SI-SDR 差异，以及 Spec L1 与 SI-SDR 是否同向。",
        "",
        "| Type | Sample | Mixture | Target | Residual | Week4 SI-SDR | Week5 SI-SDR | Week5 Residual SI-SDR | 说明 |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {case_type} | {sample_name} / {sample_id} | {source_pair} | {target_label} | {residual_label} | "
            "{week4_target_si_sdr_db:.2f} | {week5_target_si_sdr_db:.2f} | {week5_residual_si_sdr_db:.2f} | {explanation} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 音频路径",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['case_type']} - {row['sample_name']} ({row['sample_id']})",
                "",
                f"- mixture: `{row['mixture_wav']}`",
                f"- target: `{row['target_wav']}`",
                f"- residual target: `{row['residual_target_wav']}`",
                f"- Week4 target: `{row['week4_prediction_target_wav']}`",
                f"- Week5 target: `{row['week5_prediction_target_wav']}`",
                f"- Week5 residual: `{row['week5_prediction_residual_wav']}`",
                "",
            ]
        )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = load_records()
    chosen = choose_cases(records)
    write_csv(OUT_CSV, chosen)
    write_markdown(chosen)
    print(f"Saved case study to {OUT_MD}")


if __name__ == "__main__":
    main()
