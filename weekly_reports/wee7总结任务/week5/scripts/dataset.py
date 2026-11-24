#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class SeparationManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        data_root: str | Path,
        class_to_idx: dict[str, int],
        sample_rate: int,
        num_samples: int,
        limit: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.records = load_jsonl(manifest_path)
        if limit is not None:
            self.records = self.records[: int(limit)]
        self.class_to_idx = dict(class_to_idx)
        self.sample_rate = int(sample_rate)
        self.num_samples = int(num_samples)

    def __len__(self) -> int:
        return len(self.records)

    def _load_waveform(self, relative_path: str) -> torch.Tensor:
        path = self.data_root / relative_path
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if int(sr) != self.sample_rate:
            raise ValueError(f"Unexpected sample rate for {path}: {sr} != {self.sample_rate}")
        mono = audio.mean(axis=1, dtype=np.float32)
        if mono.shape[0] < self.num_samples:
            padded = np.zeros(self.num_samples, dtype=np.float32)
            padded[: mono.shape[0]] = mono
            mono = padded
        elif mono.shape[0] > self.num_samples:
            mono = mono[: self.num_samples]
        return torch.from_numpy(mono.copy())

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        target_label = str(rec["target_label"])
        if target_label not in self.class_to_idx:
            raise KeyError(f"Unknown target label: {target_label}")
        return {
            "mixture_waveform": self._load_waveform(rec["mixture_path"]),
            "target_waveform": self._load_waveform(rec["target_path"]),
            "residual_waveform": self._load_waveform(rec["residual_path"]),
            "label": torch.tensor(self.class_to_idx[target_label], dtype=torch.long),
            "sample_id": str(rec["sample_id"]),
            "mix_id": str(rec["mix_id"]),
            "target_label": target_label,
            "source_labels": [str(label) for label in rec.get("source_labels", [])],
            "residual_label": next(
                (str(label) for label in rec.get("source_labels", []) if str(label) != target_label),
                "",
            ),
        }
