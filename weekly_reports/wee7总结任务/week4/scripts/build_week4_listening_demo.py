from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
WEEK4 = ROOT / "week4"
RUNS = WEEK4 / "runs"
DATA_ROOT = ROOT / "week2" / "data_esc50_2mix"
MAMBA_DIR = RUNS / "minidit_mamba_v1_bs48" / "samples_test_20_1step"
ATTN_DIR = RUNS / "minidit_self_attention_recheck_bs48" / "samples_test_20_1step"
LOCAL_DEMO_DIR = WEEK4 / "收尾材料" / "week4_listening_20_comparison"
HF_SPACE_DIR = WEEK4 / "hf_space_week4_audio_demo"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    estimate = estimate.astype(np.float64)
    reference = reference.astype(np.float64)
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    target = np.dot(estimate, reference) * reference / (np.dot(reference, reference) + eps)
    noise = estimate - target
    return float(10.0 * np.log10((np.sum(target**2) + eps) / (np.sum(noise**2) + eps)))


def audio_metrics(sample_dir: Path) -> dict[str, float]:
    target, _ = sf.read(sample_dir / "target.wav", dtype="float32")
    mixture, _ = sf.read(sample_dir / "mixture.wav", dtype="float32")
    prediction, _ = sf.read(sample_dir / "prediction.wav", dtype="float32")
    return {
        "mixture_wave_l1": float(np.mean(np.abs(mixture - target))),
        "prediction_wave_l1": float(np.mean(np.abs(prediction - target))),
        "mixture_si_sdr": si_sdr(mixture, target),
        "prediction_si_sdr": si_sdr(prediction, target),
    }


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_assets(asset_root: Path, sample_count: int) -> None:
    for idx in range(sample_count):
        sample_assets = asset_root / f"sample_{idx:02d}"
        sample_assets.mkdir(parents=True, exist_ok=True)
        mamba_sample = MAMBA_DIR / f"sample_{idx:02d}"
        attn_sample = ATTN_DIR / f"sample_{idx:02d}"

        shutil.copy2(mamba_sample / "mixture.wav", sample_assets / "mixture.wav")
        shutil.copy2(mamba_sample / "target.wav", sample_assets / "target.wav")
        shutil.copy2(mamba_sample / "prediction.wav", sample_assets / "mamba_prediction.wav")
        shutil.copy2(attn_sample / "prediction.wav", sample_assets / "self_attention_prediction.wav")
        shutil.copy2(mamba_sample / "spectrogram_triplet.png", sample_assets / "mamba_spectrogram.png")
        shutil.copy2(attn_sample / "spectrogram_triplet.png", sample_assets / "self_attention_spectrogram.png")


def format_num(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_rows(records: list[dict], mamba_summary: dict, attn_summary: dict) -> str:
    rows = []
    for idx, record in enumerate(records):
        sample_name = f"sample_{idx:02d}"
        mamba_sample = mamba_summary["samples"][idx]
        attn_sample = attn_summary["samples"][idx]
        mamba_metrics = audio_metrics(MAMBA_DIR / sample_name)
        attn_metrics = audio_metrics(ATTN_DIR / sample_name)

        source_labels = [str(label) for label in record["source_labels"]]
        target_label = str(record["target_label"])
        residual_labels = [label for label in source_labels if label != target_label]
        residual_label = residual_labels[0] if residual_labels else "same-class/unknown"
        mixture_text = " + ".join(source_labels)

        spec_winner = "Mamba" if mamba_sample["prediction_to_target_l1"] <= attn_sample["prediction_to_target_l1"] else "Self-Attn"
        wave_winner = "Mamba" if mamba_metrics["prediction_wave_l1"] <= attn_metrics["prediction_wave_l1"] else "Self-Attn"

        rows.append(
            f"""
            <tr>
              <td class="meta">
                <div class="sample-name">{html.escape(sample_name)}</div>
                <div>ID: {html.escape(str(record["sample_id"]))}</div>
                <div>Mixture: <b>{html.escape(mixture_text)}</b></div>
                <div>Target label: <b>{html.escape(target_label)}</b></div>
                <div>Residual label: {html.escape(residual_label)}</div>
              </td>
              <td><audio controls preload="none" src="assets/{sample_name}/mixture.wav"></audio></td>
              <td><audio controls preload="none" src="assets/{sample_name}/target.wav"></audio></td>
              <td>
                <audio controls preload="none" src="assets/{sample_name}/mamba_prediction.wav"></audio>
                <div class="metric">Spec L1 {format_num(mamba_sample["prediction_to_target_l1"], 5)}</div>
                <div class="metric">Wave L1 {format_num(mamba_metrics["prediction_wave_l1"], 5)}</div>
                <div class="metric">SI-SDR {format_num(mamba_metrics["prediction_si_sdr"], 2)} dB</div>
                <a href="assets/{sample_name}/mamba_spectrogram.png">spectrogram</a>
              </td>
              <td>
                <audio controls preload="none" src="assets/{sample_name}/self_attention_prediction.wav"></audio>
                <div class="metric">Spec L1 {format_num(attn_sample["prediction_to_target_l1"], 5)}</div>
                <div class="metric">Wave L1 {format_num(attn_metrics["prediction_wave_l1"], 5)}</div>
                <div class="metric">SI-SDR {format_num(attn_metrics["prediction_si_sdr"], 2)} dB</div>
                <a href="assets/{sample_name}/self_attention_spectrogram.png">spectrogram</a>
              </td>
              <td>
                <div>Spec: <b>{spec_winner}</b></div>
                <div>Wave: <b>{wave_winner}</b></div>
              </td>
            </tr>
            """
        )
    return "\n".join(rows)


def build_index_html(records: list[dict], mamba_summary: dict, attn_summary: dict) -> str:
    rows_html = build_rows(records, mamba_summary, attn_summary)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Week4 Audio Separation Listening Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --red: #9b1230;
      --ink: #202124;
      --muted: #5f6368;
      --line: #dadce0;
      --soft: #f7f7f8;
      --green: #087f5b;
    }}
    body {{
      margin: 28px;
      font-family: Arial, "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
      color: var(--ink);
      background: #fff;
    }}
    h1 {{ margin: 0 0 8px; color: var(--red); }}
    .summary {{
      max-width: 1100px;
      color: var(--muted);
      line-height: 1.65;
      margin-bottom: 18px;
    }}
    .summary strong {{ color: var(--ink); }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(3, minmax(180px, 1fr));
      gap: 12px;
      max-width: 960px;
      margin: 18px 0 22px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: var(--soft);
    }}
    .card .label {{ color: var(--muted); font-size: 13px; }}
    .card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 1180px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 10px; vertical-align: top; text-align: left; }}
    th {{ position: sticky; top: 0; background: #f1f3f4; z-index: 1; }}
    audio {{ width: 210px; max-width: 100%; }}
    a {{ color: var(--red); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }}
    .sample-name {{ font-weight: 700; color: var(--red); margin-bottom: 4px; }}
    .meta {{ min-width: 250px; line-height: 1.45; }}
    .metric {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}
    .note {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>Week4 音频分离试听对比</h1>
  <div class="summary">
    这里展示 test split 前 20 条 target records。每行说明原始 mixture 的两个类别、当前要分离的 target label，
    并提供 mixture、真实 target、Mamba prediction、self-attention prediction 四段音频。
    <br>
    这批 20 个样例上，Mamba 的平均谱图 L1 为 <strong>{mamba_summary["mean_prediction_to_target_l1"]:.5f}</strong>，
    self-attention 为 <strong>{attn_summary["mean_prediction_to_target_l1"]:.5f}</strong>。
  </div>

  <div class="cards">
    <div class="card"><div class="label">Mixture → Target mean Spec L1</div><div class="value">{mamba_summary["mean_mixture_to_target_l1"]:.5f}</div></div>
    <div class="card"><div class="label">Mamba mean Spec L1</div><div class="value">{mamba_summary["mean_prediction_to_target_l1"]:.5f}</div></div>
    <div class="card"><div class="label">Self-Attn mean Spec L1</div><div class="value">{attn_summary["mean_prediction_to_target_l1"]:.5f}</div></div>
  </div>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Sample / mixture info</th>
          <th>Mixture</th>
          <th>Target</th>
          <th>Mamba prediction</th>
          <th>Self-Attn prediction</th>
          <th>Lower loss</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
  <div class="note">
    注：Spec L1 是当前训练/评估直接使用的谱图指标；Wave L1 和 SI-SDR 是基于导出的 wav 重新计算的试听辅助指标。
    某些样例会出现“谱图 L1 更低但听感或 SI-SDR 更差”的情况，这也是 Week5 计划加入 waveform/STFT consistency loss 的原因。
  </div>
</body>
</html>
"""


def write_space_readme(path: Path) -> None:
    text = """---
title: Week4 Audio Separation Demo
emoji: 🎧
colorFrom: red
colorTo: gray
sdk: static
pinned: false
---

# Week4 Audio Separation Listening Demo

This static Space contains 20 test-set listening examples for the Week4 audio separation experiment.

Each row compares:

- mixture
- ground-truth target
- Mamba-style SSM prediction
- self-attention baseline prediction

The demo is generated from local experiment artifacts and is intended for course presentation and qualitative inspection.
"""
    (path / "README.md").write_text(text, encoding="utf-8")


def build_demo(target_dir: Path, records: list[dict], mamba_summary: dict, attn_summary: dict) -> None:
    reset_dir(target_dir)
    copy_assets(target_dir / "assets", len(records))
    (target_dir / "index.html").write_text(build_index_html(records, mamba_summary, attn_summary), encoding="utf-8")


def main() -> None:
    mamba_summary = read_json(MAMBA_DIR / "summary.json")
    attn_summary = read_json(ATTN_DIR / "summary.json")
    sample_count = min(int(mamba_summary["num_samples"]), int(attn_summary["num_samples"]), 20)
    records = read_jsonl(DATA_ROOT / "metadata" / "manifest_test.jsonl", sample_count)

    build_demo(LOCAL_DEMO_DIR, records, mamba_summary, attn_summary)
    build_demo(HF_SPACE_DIR, records, mamba_summary, attn_summary)
    write_space_readme(HF_SPACE_DIR)

    print(LOCAL_DEMO_DIR / "index.html")
    print(HF_SPACE_DIR / "index.html")


if __name__ == "__main__":
    main()
