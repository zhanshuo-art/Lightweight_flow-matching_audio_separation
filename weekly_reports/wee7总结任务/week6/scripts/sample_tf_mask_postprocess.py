#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from torch.utils.data import DataLoader

from tf_mask_common import MODEL_METHODS, apply_tf_mask_method, fmt_float, read_csv, si_sdr

from common import ensure_dir, load_config, load_torch_checkpoint, read_json, set_seed, write_json
from sample_week5 import build_dataset, euler_sample
from train_mamba_waveform_ft import build_feature_codec, build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample Week6 TF-mask postprocessed target/residual audio.")
    parser.add_argument("--config", default="week5/configs/mamba_waveform_finetune.yaml")
    parser.add_argument("--checkpoint", default="week5/runs/mamba_waveform_ft_v1/checkpoints/best.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--method", default="best", help="best or one of week5_direct/tfmask_clip_0_1/wiener_p1/wiener_p2")
    parser.add_argument("--summary-csv", default="week6/analysis/tf_mask_postprocess/summary.csv")
    parser.add_argument("--output-dir", default="week6/analysis/tf_mask_postprocess/listening_demo")
    return parser.parse_args()


def choose_method(requested: str, summary_csv: Path) -> str:
    if requested != "best":
        if requested not in MODEL_METHODS:
            raise ValueError(f"Unsupported sample method: {requested}")
        return requested
    rows = [row for row in read_csv(summary_csv) if row["method"] in MODEL_METHODS]
    if not rows:
        return "tfmask_clip_0_1"
    best = max(rows, key=lambda row: float(row["target_si_sdr_db"]))
    return str(best["method"])


def audio_tag(path: str) -> str:
    return f'<audio controls preload="none" src="{html.escape(path)}"></audio>'


def build_html(summary: dict[str, Any], rows: list[dict[str, Any]], method: str) -> str:
    table_rows = []
    for idx, row in enumerate(rows):
        sample_name = f"sample_{idx:02d}"
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
              <td>
                {audio_tag(f"assets/{sample_name}/week5_direct_target.wav")}
                <div class="metric">SI-SDR {fmt_float(row.get("direct_target_si_sdr_db"), 2)} dB</div>
              </td>
              <td>
                {audio_tag(f"assets/{sample_name}/week5_direct_residual.wav")}
                <div class="metric">SI-SDR {fmt_float(row.get("direct_residual_si_sdr_db"), 2)} dB</div>
              </td>
              <td>
                {audio_tag(f"assets/{sample_name}/week6_target.wav")}
                <div class="metric">Method {html.escape(method)}</div>
                <div class="metric">SI-SDR {fmt_float(row.get("week6_target_si_sdr_db"), 2)} dB</div>
              </td>
              <td>
                {audio_tag(f"assets/{sample_name}/week6_residual.wav")}
                <div class="metric">SI-SDR {fmt_float(row.get("week6_residual_si_sdr_db"), 2)} dB</div>
              </td>
            </tr>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Week6 TF-mask Listening Demo</title>
  <style>
    :root {{ --red:#8e1620; --text:#202124; --muted:#62666d; --line:#d9dde3; --bg:#f6f7f9; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; color:var(--text); background:var(--bg); }}
    main {{ max-width:1280px; margin:0 auto; padding:28px 18px 42px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    p {{ color:var(--muted); line-height:1.55; }}
    .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
    .card {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .label {{ color:var(--muted); font-size:13px; }}
    .value {{ margin-top:6px; font-size:22px; font-weight:700; color:var(--red); }}
    .table-wrap {{ overflow-x:auto; background:#fff; border:1px solid var(--line); border-radius:8px; }}
    table {{ border-collapse:collapse; width:100%; min-width:1240px; }}
    th, td {{ border-bottom:1px solid var(--line); padding:10px; vertical-align:top; text-align:left; }}
    th {{ background:#fafafa; font-size:13px; }}
    audio {{ width:160px; max-width:100%; }}
    .sample-name {{ font-weight:700; color:var(--red); margin-bottom:4px; }}
    .metric {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  </style>
</head>
<body>
<main>
  <h1>Week6 TF-mask Listening Demo</h1>
  <p>本页比较 Week5 direct 输出和 Week6 选定后处理方法。Week6 residual 仍由 mixture - prediction target 得到。</p>
  <div class="stats">
    <div class="card"><div class="label">Samples</div><div class="value">{len(rows)}</div></div>
    <div class="card"><div class="label">Method</div><div class="value">{html.escape(method)}</div></div>
    <div class="card"><div class="label">Direct Target SI-SDR</div><div class="value">{fmt_float(summary.get("mean_direct_target_si_sdr_db"), 2)} dB</div></div>
    <div class="card"><div class="label">Week6 Target SI-SDR</div><div class="value">{fmt_float(summary.get("mean_week6_target_si_sdr_db"), 2)} dB</div></div>
  </div>
  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Sample</th>
          <th>Mixture</th>
          <th>Target</th>
          <th>Residual Target</th>
          <th>Week5 Target</th>
          <th>Week5 Residual</th>
          <th>Week6 Target</th>
          <th>Week6 Residual</th>
        </tr>
      </thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </div>
</main>
</body>
</html>
"""


@torch.no_grad()
def main() -> None:
    args = parse_args()
    method = choose_method(args.method, Path(args.summary_csv).resolve())
    config = load_config(args.config)
    if args.data_root is not None:
        config["paths"]["data_root"] = Path(args.data_root).resolve()

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir).resolve())
    assets_dir = ensure_dir(output_dir / "assets")
    stats_path = config["paths"].get("feature_stats_path", config["paths"].get("sigma_stats_path"))
    if stats_path is None:
        raise KeyError("Expected paths.feature_stats_path or paths.sigma_stats_path in config.")
    sigma = float(read_json(stats_path)["sigma"])

    dataset = build_dataset(config, split=args.split, limit=args.num_samples)
    loader = DataLoader(dataset, batch_size=int(args.num_samples), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    mixture_waveform = batch["mixture_waveform"].to(device)
    target_waveform = batch["target_waveform"].to(device)
    residual_waveform = batch["residual_waveform"].to(device)
    labels = batch["label"].to(device)

    feature_extractor = build_feature_codec(config, sigma).to(device)
    model = build_model(config).to(device)
    checkpoint = load_torch_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mixture_spec = feature_extractor.waveform_to_features(mixture_waveform)
    target_spec = feature_extractor.waveform_to_features(target_waveform)
    residual_spec = feature_extractor.waveform_to_features(residual_waveform)
    pred_spec = euler_sample(model, mixture_spec, mixture_spec, labels, int(args.num_steps), feature_extractor)
    week6_spec, _stats = apply_tf_mask_method(method, pred_spec, mixture_spec, target_spec, residual_spec, feature_extractor)

    direct_target = feature_extractor.reconstruct_waveform(pred_spec, mixture_waveform, int(config["audio"]["num_samples"]))
    direct_residual = mixture_waveform - direct_target
    week6_target = feature_extractor.reconstruct_waveform(week6_spec, mixture_waveform, int(config["audio"]["num_samples"]))
    week6_residual = mixture_waveform - week6_target

    sample_rate = int(config["audio"]["sample_rate"])
    rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(dataset.records):
        sample_name = f"sample_{idx:02d}"
        sample_dir = ensure_dir(assets_dir / sample_name)
        source_labels = [str(label) for label in rec["source_labels"]]
        target_label = str(rec["target_label"])
        residual_label = next((label for label in source_labels if label != target_label), "")
        sf.write(sample_dir / "mixture.wav", mixture_waveform[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "target.wav", target_waveform[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "residual_target.wav", residual_waveform[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "week5_direct_target.wav", direct_target[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "week5_direct_residual.wav", direct_residual[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "week6_target.wav", week6_target[idx].detach().cpu().numpy(), sample_rate)
        sf.write(sample_dir / "week6_residual.wav", week6_residual[idx].detach().cpu().numpy(), sample_rate)
        rows.append(
            {
                "sample_id": str(rec["sample_id"]),
                "mix_id": str(rec["mix_id"]),
                "target_label": target_label,
                "residual_label": residual_label,
                "source_pair": " + ".join(source_labels),
                "direct_target_si_sdr_db": float(si_sdr(direct_target[idx : idx + 1], target_waveform[idx : idx + 1]).cpu()),
                "direct_residual_si_sdr_db": float(
                    si_sdr(direct_residual[idx : idx + 1], residual_waveform[idx : idx + 1]).cpu()
                ),
                "week6_target_si_sdr_db": float(si_sdr(week6_target[idx : idx + 1], target_waveform[idx : idx + 1]).cpu()),
                "week6_residual_si_sdr_db": float(
                    si_sdr(week6_residual[idx : idx + 1], residual_waveform[idx : idx + 1]).cpu()
                ),
            }
        )

    summary = {
        "method": method,
        "num_samples": len(rows),
        "mean_direct_target_si_sdr_db": sum(row["direct_target_si_sdr_db"] for row in rows) / len(rows),
        "mean_direct_residual_si_sdr_db": sum(row["direct_residual_si_sdr_db"] for row in rows) / len(rows),
        "mean_week6_target_si_sdr_db": sum(row["week6_target_si_sdr_db"] for row in rows) / len(rows),
        "mean_week6_residual_si_sdr_db": sum(row["week6_residual_si_sdr_db"] for row in rows) / len(rows),
        "samples": rows,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "index.html").write_text(build_html(summary, rows, method), encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved Week6 listening demo to: {output_dir}")


if __name__ == "__main__":
    main()

