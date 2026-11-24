#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tf_mask_common import METHOD_ORDER, apply_tf_mask_method, average, method_sort_key, si_sdr, write_csv

from common import ensure_dir, load_config, load_torch_checkpoint, read_json, set_seed, write_json
from sample_week5 import build_dataset, euler_sample
from train_mamba_waveform_ft import build_feature_codec, build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Week6 TF-mask postprocessing on Week5 Mamba predictions.")
    parser.add_argument("--config", default="week5/configs/mamba_waveform_finetune.yaml")
    parser.add_argument("--checkpoint", default="week5/runs/mamba_waveform_ft_v1/checkpoints/best.pt")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    parser.add_argument("--output-dir", default="week6/analysis/tf_mask_postprocess")
    return parser.parse_args()


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_samples": len(rows),
        "target_spec_l1": average(rows, "target_spec_l1"),
        "target_wave_l1": average(rows, "target_wave_l1"),
        "target_si_sdr_db": average(rows, "target_si_sdr_db"),
        "residual_wave_l1": average(rows, "residual_wave_l1"),
        "residual_si_sdr_db": average(rows, "residual_si_sdr_db"),
        "mixture_consistency_l1": average(rows, "mixture_consistency_l1"),
        "clip_ratio": average(rows, "clip_ratio"),
        "mask_mean": average(rows, "mask_mean"),
        "mask_sparsity": average(rows, "mask_sparsity"),
    }


def grouped_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)

    output: list[dict[str, Any]] = []
    for key_values, group_rows in sorted(groups.items(), key=lambda item: (method_sort_key(item[0][0]), item[0])):
        summary = summarize(group_rows)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(summary)
        output.append(row)
    return output


def add_direct_deltas(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    direct = next((row for row in summary_rows if row["method"] == "week5_direct"), None)
    if direct is None:
        return summary_rows
    direct_si = float(direct["target_si_sdr_db"])
    direct_res_si = float(direct["residual_si_sdr_db"])
    direct_wave = float(direct["target_wave_l1"])
    direct_spec = float(direct["target_spec_l1"])
    for row in summary_rows:
        row["target_si_sdr_delta_db"] = float(row["target_si_sdr_db"]) - direct_si
        row["residual_si_sdr_delta_db"] = float(row["residual_si_sdr_db"]) - direct_res_si
        row["target_wave_l1_delta"] = float(row["target_wave_l1"]) - direct_wave
        row["target_spec_l1_delta"] = float(row["target_spec_l1"]) - direct_spec
    return summary_rows


@torch.no_grad()
def main() -> None:
    args = parse_args()
    methods = list(dict.fromkeys(args.methods))
    unknown = [method for method in methods if method not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")

    config = load_config(args.config)
    if args.data_root is not None:
        config["paths"]["data_root"] = Path(args.data_root).resolve()

    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = ensure_dir(Path(args.output_dir).resolve())
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
    model = build_model(config).to(device)
    checkpoint = load_torch_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    flow_start = str(config["training"]["flow_start"])
    if flow_start != "mixture":
        raise ValueError(f"Unsupported flow_start for Week6 evaluation: {flow_start}")

    rows: list[dict[str, Any]] = []
    record_offset = 0
    for batch in tqdm(loader, desc=f"week6 tf-mask eval {args.split}"):
        mixture_waveform = batch["mixture_waveform"].to(device=device, non_blocking=True)
        target_waveform = batch["target_waveform"].to(device=device, non_blocking=True)
        residual_waveform = batch["residual_waveform"].to(device=device, non_blocking=True)
        labels = batch["label"].to(device=device, non_blocking=True)

        mixture_spec = feature_extractor.waveform_to_features(mixture_waveform)
        target_spec = feature_extractor.waveform_to_features(target_waveform)
        residual_spec = feature_extractor.waveform_to_features(residual_waveform)
        pred_spec = euler_sample(
            model,
            mixture_spec,
            mixture_spec,
            labels,
            int(args.num_steps),
            feature_extractor,
        )

        batch_size = mixture_waveform.shape[0]
        for method in methods:
            final_spec, stats = apply_tf_mask_method(
                method,
                pred_spec=pred_spec,
                mixture_spec=mixture_spec,
                target_spec=target_spec,
                residual_spec=residual_spec,
                codec=feature_extractor,
            )
            prediction_waveform = feature_extractor.reconstruct_waveform(
                final_spec,
                mixture_waveform,
                length=int(config["audio"]["num_samples"]),
            )
            prediction_residual_waveform = mixture_waveform - prediction_waveform

            target_spec_l1 = torch.mean(torch.abs(final_spec - target_spec), dim=(1, 2, 3))
            target_wave_l1 = torch.mean(torch.abs(prediction_waveform - target_waveform), dim=1)
            residual_wave_l1 = torch.mean(torch.abs(prediction_residual_waveform - residual_waveform), dim=1)
            mixture_consistency_l1 = torch.mean(
                torch.abs(prediction_waveform + prediction_residual_waveform - mixture_waveform),
                dim=1,
            )
            target_si_sdr = si_sdr(prediction_waveform, target_waveform)
            residual_si_sdr = si_sdr(prediction_residual_waveform, residual_waveform)

            for batch_idx in range(batch_size):
                rec = dataset.records[record_offset + batch_idx]
                source_labels = [str(label) for label in rec["source_labels"]]
                target_label = str(rec["target_label"])
                residual_label = next((label for label in source_labels if label != target_label), "")
                row = {
                    "method": method,
                    "sample_id": str(rec["sample_id"]),
                    "mix_id": str(rec["mix_id"]),
                    "target_label": target_label,
                    "residual_label": residual_label,
                    "source_pair": " + ".join(source_labels),
                    "target_spec_l1": float(target_spec_l1[batch_idx].cpu()),
                    "target_wave_l1": float(target_wave_l1[batch_idx].cpu()),
                    "target_si_sdr_db": float(target_si_sdr[batch_idx].cpu()),
                    "residual_wave_l1": float(residual_wave_l1[batch_idx].cpu()),
                    "residual_si_sdr_db": float(residual_si_sdr[batch_idx].cpu()),
                    "mixture_consistency_l1": float(mixture_consistency_l1[batch_idx].cpu()),
                    "clip_ratio": float(stats["clip_ratio"][batch_idx].cpu()),
                    "mask_mean": float(stats["mask_mean"][batch_idx].cpu()),
                    "mask_sparsity": float(stats["mask_sparsity"][batch_idx].cpu()),
                }
                rows.append(row)
        record_offset += batch_size

    summary_rows = add_direct_deltas(grouped_summary(rows, ("method",)))
    by_label_rows = grouped_summary(rows, ("method", "target_label"))
    by_pair_rows = grouped_summary(rows, ("method", "source_pair"))

    write_csv(output_dir / "samples.csv", rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "by_label.csv", by_label_rows)
    write_csv(output_dir / "by_pair.csv", by_pair_rows)
    summary_payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "num_steps": int(args.num_steps),
        "num_records": len(dataset),
        "methods": methods,
        "summary": summary_rows,
    }
    write_json(output_dir / "summary.json", summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    print(f"Saved Week6 TF-mask evaluation to: {output_dir}")


if __name__ == "__main__":
    main()

