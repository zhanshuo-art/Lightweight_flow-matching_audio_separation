#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

from build_compatibility_matrix import build_compatibility_matrix
from generate_2mix_dataset import generate_2mix_dataset, parse_count_overrides
from prepare_clean_sources import prepare_clean_sources
from validate_2mix_dataset import validate_2mix_dataset
from common import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete Week 2 ESC-50 2-mix data pipeline.")
    parser.add_argument("--config", default="week2/configs/esc50_2mix.yaml", help="Path to YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root directory.")
    parser.add_argument("--mixtures-train", type=int, default=None, help="Override train mixture count.")
    parser.add_argument("--mixtures-val", type=int, default=None, help="Override val mixture count.")
    parser.add_argument("--mixtures-test", type=int, default=None, help="Override test mixture count.")
    parser.add_argument("--max-audio-checks", type=int, default=None, help="Limit audio checks; 0 means all.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, output_root=args.output_root)
    counts = parse_count_overrides(args, config)
    max_checks = args.max_audio_checks
    if max_checks == 0:
        max_checks = None

    prepare_clean_sources(config, overwrite=args.overwrite)
    build_compatibility_matrix(config)
    generate_2mix_dataset(config, counts=counts, overwrite=args.overwrite)
    validate_2mix_dataset(config, max_audio_checks=max_checks, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
