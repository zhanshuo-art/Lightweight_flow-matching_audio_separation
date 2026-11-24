#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    SPLIT_ORDER,
    SPLIT_SEED_OFFSETS,
    compatibility_pairs,
    dataset_path,
    db_to_gain,
    load_config,
    make_dirs,
    metadata_dir,
    output_root,
    peak,
    read_audio,
    relative_path,
    rms,
    selected_categories,
    target_sample_count,
    write_wav,
)


def parse_count_overrides(args: argparse.Namespace, config: dict) -> dict[str, int]:
    counts = dict(config["mixing"]["mixtures_per_split"])
    if args.mixtures_train is not None:
        counts["train"] = int(args.mixtures_train)
    if args.mixtures_val is not None:
        counts["val"] = int(args.mixtures_val)
    if args.mixtures_test is not None:
        counts["test"] = int(args.mixtures_test)
    return {split: int(counts.get(split, 0)) for split in SPLIT_ORDER}


def prepare_output_locations(root: Path, counts: dict[str, int], overwrite: bool) -> None:
    mix_root = root / "rendered_mixtures"
    meta_dir = root / "metadata"
    existing = []
    for split, count in counts.items():
        split_dir = mix_root / split
        manifest = meta_dir / f"manifest_{split}.jsonl"
        if count <= 0:
            continue
        if split_dir.exists() and any(split_dir.iterdir()):
            existing.append(split_dir)
        if manifest.exists():
            existing.append(manifest)
    if existing and not overwrite:
        paths = "\n".join(str(p) for p in existing[:10])
        raise FileExistsError(f"Existing generated outputs found. Re-run with --overwrite.\n{paths}")

    if overwrite:
        for split, count in counts.items():
            if count <= 0:
                continue
            split_dir = mix_root / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            for filename in [f"manifest_{split}.jsonl"]:
                path = meta_dir / filename
                if path.exists():
                    path.unlink()
        for filename in ["mixture_index.csv"]:
            path = meta_dir / filename
            if path.exists():
                path.unlink()


def choose_balanced_pair(
    pairs: list[tuple[str, str]],
    target_counts: dict[str, int],
    rng: np.random.Generator,
) -> tuple[str, str]:
    scores = np.asarray([target_counts[a] + target_counts[b] for a, b in pairs], dtype=np.int64)
    min_score = int(scores.min())
    candidate_indices = np.flatnonzero(scores <= min_score + 2)
    idx = int(rng.choice(candidate_indices))
    return pairs[idx]


def load_source_audio(root: Path, clean_df: pd.DataFrame, sample_rate: int, target_samples: int) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    for row in tqdm(clean_df.to_dict("records"), desc="load clean audio"):
        path = dataset_path(row["clean_path"], root)
        audio, sr = read_audio(path)
        if sr != sample_rate:
            raise ValueError(f"Unexpected sample rate in clean source {path}: {sr}")
        if len(audio) != target_samples:
            raise ValueError(f"Unexpected source length in {path}: {len(audio)}")
        cache[row["source_id"]] = audio
    return cache


def generate_2mix_dataset(
    config: dict,
    counts: dict[str, int] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    make_dirs(config)
    root = output_root(config)
    meta_dir = metadata_dir(config)
    clean_path = meta_dir / "clean_clips.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Run prepare_clean_sources.py first: {clean_path}")

    pairs_path = meta_dir / "compatibility_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(f"Run build_compatibility_matrix.py first: {pairs_path}")

    clean_df = pd.read_csv(clean_path)
    categories = selected_categories(config)
    pairs = compatibility_pairs(config)
    if counts is None:
        counts = {split: int(config["mixing"]["mixtures_per_split"].get(split, 0)) for split in SPLIT_ORDER}

    prepare_output_locations(root, counts, overwrite)

    audio_cfg = config["audio"]
    sample_rate = int(audio_cfg["sample_rate"])
    duration = float(audio_cfg["duration"])
    target_samples = target_sample_count(config)
    gain_low, gain_high = [float(x) for x in audio_cfg["relative_gain_db_range"]]
    peak_limit = float(audio_cfg["peak_limit"])
    wav_subtype = str(audio_cfg.get("wav_subtype", "PCM_16"))
    save_peak_limit = peak_limit
    if wav_subtype.upper().startswith("PCM"):
        save_peak_limit = max(0.0, peak_limit - 2.0 / 32768.0)
    prompts = config["dataset"].get("target_prompts", {})
    base_seed = int(config["mixing"]["seed"])

    audio_cache = load_source_audio(root, clean_df, sample_rate, target_samples)
    source_rows = {
        split: {
            category: clean_df[(clean_df["split"] == split) & (clean_df["category"] == category)].to_dict("records")
            for category in categories
        }
        for split in SPLIT_ORDER
    }

    all_index_rows: list[dict] = []

    for split in SPLIT_ORDER:
        n_mix = int(counts.get(split, 0))
        if n_mix <= 0:
            continue

        valid_pairs = [
            pair
            for pair in pairs
            if len(source_rows[split].get(pair[0], [])) > 0 and len(source_rows[split].get(pair[1], [])) > 0
        ]
        if not valid_pairs:
            raise ValueError(f"No valid compatibility pairs for split {split}")

        target_counts = {category: 0 for category in categories}
        manifest_records: list[dict] = []
        split_dir = root / "rendered_mixtures" / split

        for idx in tqdm(range(1, n_mix + 1), desc=f"generate {split}"):
            mix_seed = base_seed + SPLIT_SEED_OFFSETS[split] + idx
            rng = np.random.default_rng(mix_seed)
            cat_a, cat_b = choose_balanced_pair(valid_pairs, target_counts, rng)
            if bool(rng.integers(0, 2)):
                cat_a, cat_b = cat_b, cat_a

            row_a = source_rows[split][cat_a][int(rng.integers(0, len(source_rows[split][cat_a])))]
            row_b = source_rows[split][cat_b][int(rng.integers(0, len(source_rows[split][cat_b])))]
            gain_db = float(rng.uniform(gain_low, gain_high))
            gain = db_to_gain(gain_db)

            stem_0 = audio_cache[row_a["source_id"]].copy()
            stem_1 = audio_cache[row_b["source_id"]].copy() * gain
            mixture = stem_0 + stem_1

            scale = 1.0
            peak_before = max(peak(mixture), peak(stem_0), peak(stem_1))
            if peak_before > save_peak_limit:
                scale = save_peak_limit / peak_before
                stem_0 *= scale
                stem_1 *= scale
                mixture *= scale

            mix_id = f"{split}_{idx:06d}"
            mix_dir = split_dir / f"mix_{idx:06d}"
            mixture_path = mix_dir / "mixture.wav"
            stem_0_path = mix_dir / "stem_0.wav"
            stem_1_path = mix_dir / "stem_1.wav"
            write_wav(mixture_path, mixture, sample_rate, wav_subtype)
            write_wav(stem_0_path, stem_0, sample_rate, wav_subtype)
            write_wav(stem_1_path, stem_1, sample_rate, wav_subtype)

            rel_mixture = relative_path(mixture_path, root)
            rel_stem_0 = relative_path(stem_0_path, root)
            rel_stem_1 = relative_path(stem_1_path, root)
            gains = [0.0, gain_db]
            labels = [cat_a, cat_b]
            source_ids = [row_a["source_id"], row_b["source_id"]]
            source_clean_paths = [row_a["clean_path"], row_b["clean_path"]]
            original_filenames = [row_a["filename"], row_b["filename"]]

            base_record = {
                "mix_id": mix_id,
                "mixture_path": rel_mixture,
                "source_labels": labels,
                "source_ids": source_ids,
                "source_clean_paths": source_clean_paths,
                "original_filenames": original_filenames,
                "gain_db": gains,
                "sample_rate": sample_rate,
                "duration": duration,
                "split": split,
                "seed": mix_seed,
                "global_seed": base_seed,
            }
            manifest_records.append(
                {
                    **base_record,
                    "sample_id": f"{mix_id}_t0",
                    "target_path": rel_stem_0,
                    "residual_path": rel_stem_1,
                    "target_label": cat_a,
                    "target_prompt": prompts.get(cat_a, cat_a.replace("_", " ")),
                }
            )
            manifest_records.append(
                {
                    **base_record,
                    "sample_id": f"{mix_id}_t1",
                    "target_path": rel_stem_1,
                    "residual_path": rel_stem_0,
                    "target_label": cat_b,
                    "target_prompt": prompts.get(cat_b, cat_b.replace("_", " ")),
                }
            )

            target_counts[cat_a] += 1
            target_counts[cat_b] += 1
            all_index_rows.append(
                {
                    "mix_id": mix_id,
                    "split": split,
                    "mixture_path": rel_mixture,
                    "stem_0_path": rel_stem_0,
                    "stem_1_path": rel_stem_1,
                    "stem_0_source_id": row_a["source_id"],
                    "stem_1_source_id": row_b["source_id"],
                    "stem_0_label": cat_a,
                    "stem_1_label": cat_b,
                    "stem_0_original_filename": row_a["filename"],
                    "stem_1_original_filename": row_b["filename"],
                    "gain_db_stem_0": 0.0,
                    "gain_db_stem_1": gain_db,
                    "scale": scale,
                    "mixture_peak": peak(mixture),
                    "stem_0_peak": peak(stem_0),
                    "stem_1_peak": peak(stem_1),
                    "mixture_rms": rms(mixture),
                    "stem_0_rms": rms(stem_0),
                    "stem_1_rms": rms(stem_1),
                    "seed": mix_seed,
                }
            )

        manifest_path = meta_dir / f"manifest_{split}.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
            for record in manifest_records:
                f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"{split}: mixtures={n_mix}, manifest_records={len(manifest_records)}")

    index_df = pd.DataFrame(all_index_rows)
    index_df.to_csv(meta_dir / "mixture_index.csv", index=False, encoding="utf-8")
    return index_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ESC-50 2-mix waveform dataset and manifests.")
    parser.add_argument("--config", default="week2/configs/esc50_2mix.yaml", help="Path to YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root directory.")
    parser.add_argument("--mixtures-train", type=int, default=None, help="Override train mixture count.")
    parser.add_argument("--mixtures-val", type=int, default=None, help="Override val mixture count.")
    parser.add_argument("--mixtures-test", type=int, default=None, help="Override test mixture count.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated mixtures/manifests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, output_root=args.output_root)
    counts = parse_count_overrides(args, config)
    generate_2mix_dataset(config, counts=counts, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
