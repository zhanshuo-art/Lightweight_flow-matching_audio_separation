#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    SPLIT_ORDER,
    compatibility_pairs,
    dataset_path,
    figures_dir,
    format_dict_table,
    load_config,
    metadata_dir,
    normalized_pair,
    output_root,
    peak,
    plot_clean_class_counts,
    plot_spectrogram,
    plot_target_counts,
    read_audio,
    read_jsonl,
    rms,
    selected_categories,
)


def manifest_records_by_split(root_meta: Path) -> dict[str, list[dict]]:
    return {split: read_jsonl(root_meta / f"manifest_{split}.jsonl") for split in SPLIT_ORDER}


def check_disjoint(clean_df: pd.DataFrame, column: str) -> tuple[bool, list[str]]:
    overlaps = []
    for i, split_a in enumerate(SPLIT_ORDER):
        set_a = set(clean_df[clean_df["split"] == split_a][column].astype(str))
        for split_b in SPLIT_ORDER[i + 1 :]:
            set_b = set(clean_df[clean_df["split"] == split_b][column].astype(str))
            shared = sorted(set_a & set_b)
            if shared:
                overlaps.append(f"{split_a}/{split_b} share {column}: {shared[:5]}")
    return len(overlaps) == 0, overlaps


def audio_integrity_checks(
    config: dict,
    mixture_df: pd.DataFrame,
    max_checks: int | None,
) -> dict:
    root = output_root(config)
    sample_rate = int(config["audio"]["sample_rate"])
    target_samples = int(round(sample_rate * float(config["audio"]["duration"])))
    additivity_tolerance = float(config["validation"]["additivity_tolerance"])
    peak_tolerance = float(config["validation"]["peak_tolerance"])

    if max_checks is not None and max_checks > 0:
        check_df = mixture_df.head(max_checks).copy()
    else:
        check_df = mixture_df.copy()

    stats = {
        "checked_mixtures": int(len(check_df)),
        "max_abs_error": 0.0,
        "mean_abs_error_max": 0.0,
        "max_peak": 0.0,
        "bad_additivity": 0,
        "bad_clipping": 0,
        "bad_shape": 0,
        "mixture_rms": [],
        "stem_rms": [],
    }

    for row in tqdm(check_df.to_dict("records"), desc="validate audio"):
        mixture, sr_mix = read_audio(dataset_path(row["mixture_path"], root))
        stem_0, sr_0 = read_audio(dataset_path(row["stem_0_path"], root))
        stem_1, sr_1 = read_audio(dataset_path(row["stem_1_path"], root))

        if (
            sr_mix != sample_rate
            or sr_0 != sample_rate
            or sr_1 != sample_rate
            or len(mixture) != target_samples
            or len(stem_0) != target_samples
            or len(stem_1) != target_samples
        ):
            stats["bad_shape"] += 1
            continue

        residual = mixture - (stem_0 + stem_1)
        max_err = float(np.max(np.abs(residual)))
        mean_err = float(np.mean(np.abs(residual)))
        stats["max_abs_error"] = max(stats["max_abs_error"], max_err)
        stats["mean_abs_error_max"] = max(stats["mean_abs_error_max"], mean_err)
        if max_err > additivity_tolerance:
            stats["bad_additivity"] += 1

        local_peak = max(peak(mixture), peak(stem_0), peak(stem_1))
        stats["max_peak"] = max(stats["max_peak"], local_peak)
        if local_peak > peak_tolerance + 1e-7:
            stats["bad_clipping"] += 1

        stats["mixture_rms"].append(rms(mixture))
        stats["stem_rms"].extend([rms(stem_0), rms(stem_1)])

    for key in ["mixture_rms", "stem_rms"]:
        values = np.asarray(stats[key], dtype=np.float64)
        stats[f"{key}_mean"] = float(values.mean()) if values.size else 0.0
        stats[f"{key}_min"] = float(values.min()) if values.size else 0.0
        stats[f"{key}_max"] = float(values.max()) if values.size else 0.0
        del stats[key]

    stats["additivity_pass"] = stats["bad_additivity"] == 0
    stats["clipping_pass"] = stats["bad_clipping"] == 0
    stats["shape_pass"] = stats["bad_shape"] == 0
    return stats


def make_figures(config: dict, clean_df: pd.DataFrame, manifest_records: list[dict], mixture_df: pd.DataFrame) -> None:
    fig_dir = figures_dir(config)
    categories = selected_categories(config)
    plot_clean_class_counts(clean_df, categories, fig_dir / "class_count_bar.png")
    plot_target_counts(manifest_records, categories, fig_dir / "target_count_bar.png")

    if mixture_df.empty:
        return
    split = str(config["validation"].get("figure_example_split", "train"))
    example_rows = mixture_df[mixture_df["split"] == split]
    if example_rows.empty:
        example_rows = mixture_df
    row = example_rows.iloc[0].to_dict()
    root = output_root(config)
    for key, title, filename in [
        ("stem_0_path", f"{row['mix_id']} stem_0: {row['stem_0_label']}", "example_stem_0_spectrogram.png"),
        ("stem_1_path", f"{row['mix_id']} stem_1: {row['stem_1_label']}", "example_stem_1_spectrogram.png"),
        ("mixture_path", f"{row['mix_id']} mixture", "example_mixture_spectrogram.png"),
    ]:
        audio, sr = read_audio(dataset_path(row[key], root))
        plot_spectrogram(audio, sr, title, fig_dir / filename)


def validate_2mix_dataset(
    config: dict,
    max_audio_checks: int | None = None,
    make_plots: bool = True,
) -> dict:
    root = output_root(config)
    meta_dir = metadata_dir(config)
    clean_path = meta_dir / "clean_clips.csv"
    index_path = meta_dir / "mixture_index.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Missing clean_clips.csv: {clean_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"Missing mixture_index.csv: {index_path}")

    clean_df = pd.read_csv(clean_path)
    mixture_df = pd.read_csv(index_path)
    manifests = manifest_records_by_split(meta_dir)
    all_manifest_records = [record for split in SPLIT_ORDER for record in manifests[split]]
    categories = selected_categories(config)

    clean_counts = (
        clean_df.groupby(["split", "category"]).size().rename("count").reset_index().to_dict("records")
        if not clean_df.empty
        else []
    )
    mixture_counts = mixture_df.groupby("split").size().to_dict() if not mixture_df.empty else {}
    manifest_counts = {split: len(records) for split, records in manifests.items()}
    target_counts = (
        pd.DataFrame(all_manifest_records).groupby(["split", "target_label"]).size().rename("count").reset_index()
        if all_manifest_records
        else pd.DataFrame(columns=["split", "target_label", "count"])
    )

    source_disjoint, source_overlap = check_disjoint(clean_df, "source_id")
    filename_disjoint, filename_overlap = check_disjoint(clean_df, "filename")
    src_file_disjoint, src_file_overlap = check_disjoint(clean_df, "src_file")
    split_leak_pass = source_disjoint and filename_disjoint and src_file_disjoint

    source_split = dict(zip(clean_df["source_id"].astype(str), clean_df["split"].astype(str)))
    bad_mixture_split_rows = []
    for row in mixture_df.to_dict("records"):
        expected_split = row["split"]
        if source_split.get(str(row["stem_0_source_id"])) != expected_split:
            bad_mixture_split_rows.append(row["mix_id"])
        if source_split.get(str(row["stem_1_source_id"])) != expected_split:
            bad_mixture_split_rows.append(row["mix_id"])

    pair_set = set(compatibility_pairs(config))
    bad_pairs = []
    for row in mixture_df.to_dict("records"):
        pair = normalized_pair(row["stem_0_label"], row["stem_1_label"])
        if pair not in pair_set:
            bad_pairs.append(f"{row['mix_id']}:{pair[0]}+{pair[1]}")

    if max_audio_checks is None:
        max_audio_checks = config["validation"].get("max_audio_checks", None)
    audio_stats = audio_integrity_checks(config, mixture_df, max_audio_checks)

    if make_plots:
        make_figures(config, clean_df, all_manifest_records, mixture_df)

    target_count_rows = (
        target_counts.sort_values(["split", "target_label"]).to_dict("records") if not target_counts.empty else []
    )
    clean_count_rows = sorted(clean_counts, key=lambda x: (x["split"], x["category"]))

    status = {
        "clean_source_count": int(len(clean_df)),
        "mixture_count_total": int(len(mixture_df)),
        "manifest_count_total": int(sum(manifest_counts.values())),
        "mixture_counts": {split: int(mixture_counts.get(split, 0)) for split in SPLIT_ORDER},
        "manifest_counts": {split: int(manifest_counts.get(split, 0)) for split in SPLIT_ORDER},
        "split_leak_pass": split_leak_pass and len(bad_mixture_split_rows) == 0,
        "compatibility_pass": len(bad_pairs) == 0,
        "source_overlap": source_overlap + filename_overlap + src_file_overlap,
        "bad_mixture_split_rows": bad_mixture_split_rows[:10],
        "bad_pairs": bad_pairs[:10],
        "audio": audio_stats,
    }

    report = build_report(
        config=config,
        status=status,
        clean_count_rows=clean_count_rows,
        target_count_rows=target_count_rows,
    )
    report_path = meta_dir / "validation_report.md"
    report_path.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    return status


def build_report(
    config: dict,
    status: dict,
    clean_count_rows: list[dict],
    target_count_rows: list[dict],
) -> str:
    counts = status["mixture_counts"]
    manifest_counts = status["manifest_counts"]
    audio = status["audio"]
    output = output_root(config)

    clean_table = format_dict_table(clean_count_rows, ["split", "category", "count"])
    target_table = format_dict_table(target_count_rows, ["split", "target_label", "count"])
    split_source_counts = []
    clean_by_split = {}
    for row in clean_count_rows:
        clean_by_split[row["split"]] = clean_by_split.get(row["split"], 0) + int(row["count"])
    for split in SPLIT_ORDER:
        split_source_counts.append({"split": split, "clean_sources": clean_by_split.get(split, 0)})

    pass_text = lambda ok: "PASS" if ok else "FAIL"
    lines = [
        "# Week 2 ESC-50 2-mix Validation Report",
        "",
        f"- output_root: `{output}`",
        f"- dataset: `{config['dataset']['name']}`",
        f"- sample_rate: `{config['audio']['sample_rate']}`",
        f"- duration: `{config['audio']['duration']}`",
        f"- selected_categories: `{', '.join(selected_categories(config))}`",
        "",
        "## Summary",
        "",
        f"- clean source count: {status['clean_source_count']}",
        f"- mixture count: train={counts['train']}, val={counts['val']}, test={counts['test']}, total={status['mixture_count_total']}",
        f"- manifest records: train={manifest_counts['train']}, val={manifest_counts['val']}, test={manifest_counts['test']}, total={status['manifest_count_total']}",
        f"- additivity check: {pass_text(audio['additivity_pass'])} (max_abs_error={audio['max_abs_error']:.8f}, tolerance={config['validation']['additivity_tolerance']})",
        f"- clipping check: {pass_text(audio['clipping_pass'])} (max_peak={audio['max_peak']:.6f}, limit={config['validation']['peak_tolerance']})",
        f"- shape/sample-rate check: {pass_text(audio['shape_pass'])}",
        f"- split leakage check: {pass_text(status['split_leak_pass'])}",
        f"- compatibility check: {pass_text(status['compatibility_pass'])}",
        "",
        "## Clean Source Counts",
        "",
        "```text",
        format_dict_table(split_source_counts, ["split", "clean_sources"]),
        "```",
        "",
        "```text",
        clean_table,
        "```",
        "",
        "## Target Counts",
        "",
        "```text",
        target_table,
        "```",
        "",
        "## Audio Statistics",
        "",
        f"- checked_mixtures: {audio['checked_mixtures']}",
        f"- max_abs_error: {audio['max_abs_error']:.8f}",
        f"- max_mean_abs_error: {audio['mean_abs_error_max']:.8f}",
        f"- max_peak: {audio['max_peak']:.6f}",
        f"- mixture_rms: min={audio['mixture_rms_min']:.6f}, mean={audio['mixture_rms_mean']:.6f}, max={audio['mixture_rms_max']:.6f}",
        f"- stem_rms: min={audio['stem_rms_min']:.6f}, mean={audio['stem_rms_mean']:.6f}, max={audio['stem_rms_max']:.6f}",
        "",
        "## Figures",
        "",
        "- `figures/class_count_bar.png`",
        "- `figures/target_count_bar.png`",
        "- `figures/example_stem_0_spectrogram.png`",
        "- `figures/example_stem_1_spectrogram.png`",
        "- `figures/example_mixture_spectrogram.png`",
    ]

    if status["source_overlap"]:
        lines.extend(["", "## Split Overlap Warnings", "", *[f"- {x}" for x in status["source_overlap"]]])
    if status["bad_mixture_split_rows"]:
        lines.extend(
            ["", "## Mixture Split Warnings", "", *[f"- {x}" for x in status["bad_mixture_split_rows"]]]
        )
    if status["bad_pairs"]:
        lines.extend(["", "## Compatibility Warnings", "", *[f"- {x}" for x in status["bad_pairs"]]])

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated ESC-50 2-mix dataset.")
    parser.add_argument("--config", default="week2/configs/esc50_2mix.yaml", help="Path to YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root directory.")
    parser.add_argument(
        "--max-audio-checks",
        type=int,
        default=None,
        help="Limit additivity/clipping checks for smoke tests. Omit or use 0 for all mixtures.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, output_root=args.output_root)
    max_checks = args.max_audio_checks
    if max_checks == 0:
        max_checks = None
    validate_2mix_dataset(config, max_audio_checks=max_checks, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
