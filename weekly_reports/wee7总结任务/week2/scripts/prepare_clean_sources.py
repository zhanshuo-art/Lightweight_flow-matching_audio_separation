#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import (
    fix_length,
    load_config,
    make_dirs,
    metadata_dir,
    output_root,
    peak,
    read_audio_mono,
    relative_path,
    resample_if_needed,
    rms,
    selected_categories,
    split_for_fold,
    target_sample_count,
    write_wav,
    write_yaml,
)


def prepare_clean_sources(config: dict, overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    make_dirs(config)
    root = output_root(config)
    meta_dir = metadata_dir(config)
    esc50_root = Path(config["paths"]["esc50_root"])
    esc50_meta_path = esc50_root / "meta" / "esc50.csv"
    audio_dir = esc50_root / "audio"

    if not esc50_meta_path.exists():
        raise FileNotFoundError(f"ESC-50 metadata not found: {esc50_meta_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"ESC-50 audio directory not found: {audio_dir}")

    audio_cfg = config["audio"]
    sample_rate = int(audio_cfg["sample_rate"])
    target_samples = target_sample_count(config)
    target_rms = float(audio_cfg["rms_target"])
    silence_threshold = float(audio_cfg["silence_rms_threshold"])
    peak_limit = float(audio_cfg["peak_limit"])
    wav_subtype = str(audio_cfg.get("wav_subtype", "PCM_16"))
    save_peak_limit = peak_limit
    if wav_subtype.upper().startswith("PCM"):
        save_peak_limit = max(0.0, peak_limit - 2.0 / 32768.0)
    categories = selected_categories(config)

    meta = pd.read_csv(esc50_meta_path)
    missing = sorted(set(categories) - set(meta["category"].unique()))
    if missing:
        raise ValueError(f"Selected categories not found in ESC-50 metadata: {missing}")

    rows: list[dict] = []
    skipped: list[dict] = []
    filtered = meta[meta["category"].isin(categories)].copy()
    filtered = filtered.sort_values(["fold", "category", "filename"]).reset_index(drop=True)

    for item in tqdm(filtered.to_dict("records"), desc="prepare clean sources"):
        split = split_for_fold(config, int(item["fold"]))
        if split is None:
            skipped.append({**item, "status": "skipped", "reason": "fold_not_in_config"})
            continue

        source_id = Path(item["filename"]).stem
        source_path = audio_dir / item["filename"]
        clean_path = root / "clean_sources" / split / item["category"] / item["filename"]

        try:
            raw_audio, source_sr = read_audio_mono(source_path)
            original_rms = rms(raw_audio)
            original_peak = peak(raw_audio)
            if original_rms < silence_threshold:
                skipped.append(
                    {
                        **item,
                        "source_id": source_id,
                        "split": split,
                        "status": "skipped",
                        "reason": f"rms_below_{silence_threshold}",
                        "original_rms": original_rms,
                        "original_peak": original_peak,
                    }
                )
                continue

            audio = resample_if_needed(raw_audio, source_sr, sample_rate)
            audio, length_status = fix_length(audio, target_samples)
            audio = audio / max(rms(audio), 1e-12) * target_rms

            peak_scale = 1.0
            clean_peak_before = peak(audio)
            if clean_peak_before > save_peak_limit:
                peak_scale = save_peak_limit / clean_peak_before
                audio = audio * peak_scale

            clean_rms = rms(audio)
            clean_peak = peak(audio)
            if overwrite or not clean_path.exists():
                write_wav(clean_path, audio, sample_rate, wav_subtype)

            rows.append(
                {
                    "source_id": source_id,
                    "filename": item["filename"],
                    "fold": int(item["fold"]),
                    "split": split,
                    "target": int(item["target"]),
                    "category": item["category"],
                    "esc10": bool(item["esc10"]),
                    "src_file": item["src_file"],
                    "take": item["take"],
                    "original_path": source_path.resolve().relative_to(esc50_root.resolve()).as_posix(),
                    "clean_path": relative_path(clean_path, root),
                    "original_sample_rate": int(source_sr),
                    "sample_rate": sample_rate,
                    "num_samples": int(target_samples),
                    "duration": round(target_samples / sample_rate, 6),
                    "original_rms": original_rms,
                    "original_peak": original_peak,
                    "clean_rms": clean_rms,
                    "clean_peak": clean_peak,
                    "peak_scale": peak_scale,
                    "length_status": length_status,
                    "status": "ok",
                }
            )
        except Exception as exc:  # Keep one bad clip from breaking the whole pool.
            skipped.append(
                {
                    **item,
                    "source_id": source_id,
                    "split": split,
                    "status": "skipped",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )

    clean_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    clean_df.to_csv(meta_dir / "clean_clips.csv", index=False, encoding="utf-8")
    skipped_df.to_csv(meta_dir / "skipped_clean_clips.csv", index=False, encoding="utf-8")
    write_yaml(meta_dir / "preprocessing_config.yaml", config)

    expected = len(filtered)
    print(f"clean clips: {len(clean_df)} / {expected}")
    if not clean_df.empty:
        print(clean_df.groupby(["split", "category"]).size().unstack(fill_value=0).to_string())
    if not skipped_df.empty:
        print("skipped clips:")
        print(skipped_df[["filename", "category", "reason"]].to_string(index=False))
    return clean_df, skipped_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ESC-50 clean source pool for 2-mix separation.")
    parser.add_argument("--config", default="week2/configs/esc50_2mix.yaml", help="Path to YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clean source wav files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, output_root=args.output_root)
    prepare_clean_sources(config, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
