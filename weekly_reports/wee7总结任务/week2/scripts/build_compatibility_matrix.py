#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import pandas as pd

from common import compatibility_pairs, load_config, make_dirs, metadata_dir, selected_categories


def build_compatibility_matrix(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    make_dirs(config)
    categories = selected_categories(config)
    pair_set = set(compatibility_pairs(config))

    matrix = pd.DataFrame(0, index=categories, columns=categories, dtype=int)
    rows = []
    for a, b in sorted(pair_set):
        if a not in categories or b not in categories:
            raise ValueError(f"Compatibility pair contains category outside selected set: {(a, b)}")
        matrix.loc[a, b] = 1
        matrix.loc[b, a] = 1
        rows.append(
            {
                "class_a": a,
                "class_b": b,
                "compatible": 1,
                "reason": "semantic co-occurrence allowed by Week 2 plan",
            }
        )

    meta_dir = metadata_dir(config)
    matrix.to_csv(meta_dir / "compatibility_matrix.csv", encoding="utf-8")
    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(meta_dir / "compatibility_pairs.csv", index=False, encoding="utf-8")
    print(f"compatibility pairs: {len(pairs_df)}")
    print(matrix.to_string())
    return matrix, pairs_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ESC-50 category compatibility matrix.")
    parser.add_argument("--config", default="week2/configs/esc50_2mix.yaml", help="Path to YAML config.")
    parser.add_argument("--output-root", default=None, help="Override output root directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, output_root=args.output_root)
    build_compatibility_matrix(config)


if __name__ == "__main__":
    main()
