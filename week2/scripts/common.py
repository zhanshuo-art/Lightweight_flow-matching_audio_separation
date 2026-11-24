#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from scipy import signal


SPLIT_ORDER = ["train", "val", "test"]
SPLIT_SEED_OFFSETS = {"train": 0, "val": 100_000, "test": 200_000}


def load_config(config_path: str | Path, output_root: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = config_path.parent
    paths = config.setdefault("paths", {})
    for key in ("esc50_root", "output_root"):
        if key not in paths:
            raise KeyError(f"Missing paths.{key} in {config_path}")
        raw_path = Path(paths[key])
        paths[key] = raw_path if raw_path.is_absolute() else (base_dir / raw_path).resolve()

    if output_root is not None:
        output_root = Path(output_root)
        paths["output_root"] = output_root if output_root.is_absolute() else output_root.resolve()

    paths["config_path"] = config_path
    return config


def as_yamlable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: as_yamlable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [as_yamlable(v) for v in obj]
    return obj


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(as_yamlable(data), f, allow_unicode=True, sort_keys=False)


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["output_root"]).resolve()


def metadata_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "metadata"


def figures_dir(config: dict[str, Any]) -> Path:
    return output_root(config) / "figures"


def selected_categories(config: dict[str, Any]) -> list[str]:
    return list(config["dataset"]["selected_categories"])


def split_for_fold(config: dict[str, Any], fold: int) -> str | None:
    for split, folds in config["splits"].items():
        if int(fold) in [int(x) for x in folds]:
            return split
    return None


def target_sample_count(config: dict[str, Any]) -> int:
    audio_cfg = config["audio"]
    return int(round(float(audio_cfg["sample_rate"]) * float(audio_cfg["duration"])))


def rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def peak(audio: np.ndarray) -> float:
    audio = np.asarray(audio)
    return float(np.max(np.abs(audio))) if audio.size else 0.0


def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def read_audio_mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1, dtype=np.float32)
    return mono.astype(np.float32, copy=False), int(sample_rate)


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1, dtype=np.float32)
    return np.asarray(data, dtype=np.float32), int(sample_rate)


def resample_if_needed(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if int(source_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32)
    divisor = math.gcd(int(source_sr), int(target_sr))
    up = int(target_sr) // divisor
    down = int(source_sr) // divisor
    resampled = signal.resample_poly(audio, up, down)
    return np.asarray(resampled, dtype=np.float32)


def fix_length(audio: np.ndarray, target_samples: int) -> tuple[np.ndarray, str]:
    n = int(audio.shape[0])
    if n == target_samples:
        return np.asarray(audio, dtype=np.float32), "exact"
    if n > target_samples:
        return np.asarray(audio[:target_samples], dtype=np.float32), f"trimmed_from_{n}"
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[:n] = audio
    return padded, f"padded_from_{n}"


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int, subtype: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), int(sample_rate), subtype=subtype)


def relative_path(path: str | Path, root: str | Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def dataset_path(path: str | Path, root: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalized_pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((str(a), str(b))))  # type: ignore[return-value]


def compatibility_pairs(config: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = []
    for a, b in config["mixing"]["compatibility_pairs"]:
        pair = normalized_pair(a, b)
        if pair[0] != pair[1]:
            pairs.append(pair)
    return sorted(set(pairs))


def make_dirs(config: dict[str, Any]) -> None:
    root = output_root(config)
    for rel in ["clean_sources", "rendered_mixtures", "metadata", "figures"]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def format_dict_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(empty)"
    widths = {col: max(len(str(col)), *(len(str(row.get(col, ""))) for row in rows)) for col in columns}
    header = " | ".join(str(col).ljust(widths[col]) for col in columns)
    sep = "-+-".join("-" * widths[col] for col in columns)
    body = [" | ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns) for row in rows]
    return "\n".join([header, sep, *body])


def plot_clean_class_counts(clean_clips: pd.DataFrame, categories: list[str], path: str | Path) -> None:
    counts = (
        clean_clips.groupby(["category", "split"]).size().rename("count").reset_index()
        if not clean_clips.empty
        else pd.DataFrame(columns=["category", "split", "count"])
    )
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottoms = np.zeros(len(categories))
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}
    for split in SPLIT_ORDER:
        values = []
        for category in categories:
            match = counts[(counts["category"] == category) & (counts["split"] == split)]
            values.append(int(match["count"].iloc[0]) if not match.empty else 0)
        ax.bar(categories, values, bottom=bottoms, label=split, color=colors.get(split))
        bottoms += np.asarray(values)
    ax.set_ylabel("clean source count")
    ax.set_xlabel("category")
    ax.set_title("Clean source count by category")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_target_counts(manifest_records: list[dict[str, Any]], categories: list[str], path: str | Path) -> None:
    df = pd.DataFrame(manifest_records)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    if df.empty:
        values = [0 for _ in categories]
        ax.bar(categories, values, color="#4C78A8")
    else:
        counts = df.groupby("target_label").size()
        values = [int(counts.get(category, 0)) for category in categories]
        ax.bar(categories, values, color="#4C78A8")
    ax.set_ylabel("target record count")
    ax.set_xlabel("target label")
    ax.set_title("Target count in manifest")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_spectrogram(audio: np.ndarray, sample_rate: int, title: str, path: str | Path) -> None:
    freqs, times, zxx = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=1024,
        noverlap=768,
        boundary=None,
    )
    spec = np.log1p(np.abs(zxx))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    mesh = ax.pcolormesh(times, freqs, spec, shading="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
