#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static Week5 target/residual listening demo.")
    parser.add_argument("--week5-samples", required=True)
    parser.add_argument("--week4-samples", default=None)
    parser.add_argument("--output-dir", default="week5/analysis/week5_listening_demo")
    parser.add_argument("--num-samples", type=int, default=20)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def audio_tag(path: str) -> str:
    return f'<audio controls preload="none" src="{html.escape(path)}"></audio>'


def build_html(summary: dict[str, Any], rows: list[dict[str, Any]], has_week4: bool) -> str:
    table_rows = []
    for idx, row in enumerate(rows):
        sample_name = f"sample_{idx:02d}"
        week4_cell = "-"
        if has_week4:
            week4_cell = audio_tag(f"assets/{sample_name}/week4_prediction_target.wav")
        table_rows.append(
            f"""
            <tr>
              <td>
                <div class="sample-name">{html.escape(sample_name)}</div>
                <div>ID: {html.escape(str(row.get("sample_id", "")))}</div>
                <div>Mixture: <b>{html.escape(str(row.get("source_pair", "")))}</b></div>
                <div>Target: <b>{html.escape(str(row.get("target_label", "")))}</b></div>
                <div>Residual: <b>{html.escape(str(row.get("residual_label", "")))}</b></div>
              </td>
              <td>{audio_tag(f"assets/{sample_name}/mixture.wav")}</td>
              <td>{audio_tag(f"assets/{sample_name}/target.wav")}</td>
              <td>{audio_tag(f"assets/{sample_name}/residual_target.wav")}</td>
              <td>{week4_cell}</td>
              <td>
                {audio_tag(f"assets/{sample_name}/week5_prediction_target.wav")}
                <div class="metric">Spec L1 {fmt(row.get("prediction_to_target_l1"), 5)}</div>
                <div class="metric">Wave L1 {fmt(row.get("prediction_target_wave_l1"), 5)}</div>
                <div class="metric">SI-SDR {fmt(row.get("prediction_target_si_sdr_db"), 2)} dB</div>
              </td>
              <td>
                {audio_tag(f"assets/{sample_name}/week5_prediction_residual.wav")}
                <div class="metric">Wave L1 {fmt(row.get("prediction_residual_wave_l1"), 5)}</div>
                <div class="metric">SI-SDR {fmt(row.get("prediction_residual_si_sdr_db"), 2)} dB</div>
              </td>
            </tr>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Week5 Audio Separation Listening Demo</title>
  <style>
    :root {{ --red:#8e1620; --text:#202124; --muted:#62666d; --line:#d9dde3; --bg:#f6f7f9; }}
    body {{ margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; color:var(--text); background:var(--bg); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 42px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .stats {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:12px; margin:18px 0; }}
    .card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .label {{ color:var(--muted); font-size:13px; }}
    .value {{ margin-top:6px; font-size:22px; font-weight:700; color:var(--red); }}
    .table-wrap {{ overflow-x:auto; background:#fff; border:1px solid var(--line); border-radius:8px; }}
    table {{ border-collapse:collapse; width:100%; min-width:1120px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:10px; vertical-align:top; text-align:left; }}
    th {{ background:#fafafa; font-size:13px; }}
    audio {{ width: 170px; max-width: 100%; }}
    .sample-name {{ font-weight:700; color:var(--red); margin-bottom:4px; }}
    .metric {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  </style>
</head>
<body>
<main>
  <h1>Week5 Target + Residual Listening Demo</h1>
  <p>
    这里展示 Week5 fine-tuned Mamba 模型的 test 样例。每行包含 mixture 组成、target label、
    residual label，以及真实 target/residual 和模型预测的 target/residual。
  </p>
  <div class="stats">
    <div class="card"><div class="label">Samples</div><div class="value">{len(rows)}</div></div>
    <div class="card"><div class="label">Mean Spec L1</div><div class="value">{fmt(summary.get("mean_prediction_to_target_l1"), 5)}</div></div>
    <div class="card"><div class="label">Mean Target Wave L1</div><div class="value">{fmt(summary.get("mean_prediction_target_wave_l1"), 5)}</div></div>
    <div class="card"><div class="label">Mean Target SI-SDR</div><div class="value">{fmt(summary.get("mean_prediction_target_si_sdr_db"), 2)} dB</div></div>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Mixture</th>
          <th>Target</th>
          <th>Residual Target</th>
          <th>Week4 Mamba Target</th>
          <th>Week5 Target</th>
          <th>Week5 Residual</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    week5_samples = Path(args.week5_samples).resolve()
    week4_samples = Path(args.week4_samples).resolve() if args.week4_samples else None
    output_dir = Path(args.output_dir).resolve()
    reset_dir(output_dir)

    summary = read_json(week5_samples / "summary.json")
    rows = list(summary.get("samples", []))[: int(args.num_samples)]
    assets_dir = output_dir / "assets"
    has_week4 = week4_samples is not None and week4_samples.exists()

    for idx, _row in enumerate(rows):
        sample_name = f"sample_{idx:02d}"
        src_dir = week5_samples / sample_name
        dst_dir = assets_dir / sample_name
        copy_if_exists(src_dir / "mixture.wav", dst_dir / "mixture.wav")
        copy_if_exists(src_dir / "target.wav", dst_dir / "target.wav")
        copy_if_exists(src_dir / "residual_target.wav", dst_dir / "residual_target.wav")
        copy_if_exists(src_dir / "prediction_target.wav", dst_dir / "week5_prediction_target.wav")
        copy_if_exists(src_dir / "prediction_residual.wav", dst_dir / "week5_prediction_residual.wav")
        if has_week4:
            copy_if_exists(week4_samples / sample_name / "prediction.wav", dst_dir / "week4_prediction_target.wav")

    (output_dir / "index.html").write_text(build_html(summary, rows, has_week4), encoding="utf-8", newline="\n")
    print(f"Saved listening demo to: {output_dir}")


if __name__ == "__main__":
    main()
