#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import pandas as pd

from common import load_config, metadata_dir
from validate_2mix_dataset import make_figures, manifest_records_by_split


def plot_dataset_examples(config: dict) -> None:
    meta_dir = metadata_dir(config)
    clean_df = pd.read_csv(meta_dir / "clean_clips.csv")
    mixture_df = pd.read_csv(meta_dir / "mixture_index.csv")
    manifests = manifest_records_by_split(meta_dir)
    manifest_records = [record for records in manifests.values() for record in records]
    make_figures(config, clean_df, manifest_records, mixture_df)
    print("figures regenerated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Week 2 dataset figures.")
    parser.add_argument("--config", default="week2/configs/esc50_2mix.yaml", help="Path to YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, output_root=args.output_root)
    plot_dataset_examples(config)


if __name__ == "__main__":
    main()
