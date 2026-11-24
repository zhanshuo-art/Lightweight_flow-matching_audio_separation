#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export selected test examples from the final model.

The script scores records with Target SI-SDR and retains at most one record per
target label. These selected examples are intended for listening, not for
estimating aggregate test performance.

Run from the code root (the directory that holds week2/.../week6/):

    python week6/scripts/export_demo_samples.py \
        --checkpoint checkpoint/week6_best.pt \
        --data-root week2/data_esc50_2mix \
        --output-dir samples_demo --scan 200 --pick 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]  # 代码根目录（含 week2.../week6）
sys.path.insert(0, str(ROOT / "week5" / "scripts"))
sys.path.insert(0, str(ROOT / "week6" / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ensure_dir, load_config, load_torch_checkpoint, read_json, set_seed
from dataset import SeparationManifestDataset
from tf_mask_common import apply_tf_mask_method, si_sdr
from train_mamba_waveform_ft import build_feature_codec, build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export good separation demos from the final model.")
    parser.add_argument("--config", default="week5/configs/mamba_waveform_finetune.yaml")
    parser.add_argument("--checkpoint", default="checkpoint/week6_best.pt")
    parser.add_argument("--data-root", default="week2/data_esc50_2mix")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--scan", type=int, default=200, help="number of test records to score")
    parser.add_argument("--pick", type=int, default=4, help="number of demos to export")
    parser.add_argument("--method", default="tfmask_clip_0_1")
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--output-dir", default="samples_demo")
    return parser.parse_args()


@torch.no_grad()
def euler_sample(model, start_spec, mixture_spec, labels, num_steps, codec):
    x = start_spec.clone()
    dt = 1.0 / float(num_steps)
    batch = x.shape[0]
    for step in range(num_steps):
        t = torch.full((batch,), step / float(num_steps), device=x.device, dtype=x.dtype)
        x = x + dt * model(x, t, mixture_spec, labels)
    return codec.clamp_features(x)


def save_triplet(path, mixture, target, prediction, title):
    arrays = [("Mixture", mixture), ("Target", target), ("Prediction", prediction)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmax = max(float(a.max()) for _, a in arrays)
    for ax, (name, arr) in zip(axes, arrays):
        im = ax.imshow(arr, aspect="auto", origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(name)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Frequency bin")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["paths"]["data_root"] = Path(args.data_root).resolve()

    set_seed(args.seed)
    device = resolve_device(args.device)
    sigma = float(read_json(config["paths"]["sigma_stats_path"])["sigma"])

    class_names = list(config["dataset"]["class_names"])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    data_root = Path(config["paths"]["data_root"])
    dataset = SeparationManifestDataset(
        manifest_path=data_root / "metadata" / f"manifest_{args.split}.jsonl",
        data_root=data_root,
        class_to_idx=class_to_idx,
        sample_rate=int(config["audio"]["sample_rate"]),
        num_samples=int(config["audio"]["num_samples"]),
        limit=args.scan,
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    codec = build_feature_codec(config, sigma).to(device)
    model = build_model(config).to(device)
    checkpoint = load_torch_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    scored: list[dict] = []
    base = 0
    for batch in loader:
        mix_wav = batch["mixture_waveform"].to(device)
        tgt_wav = batch["target_waveform"].to(device)
        res_wav = batch["residual_waveform"].to(device)
        labels = batch["label"].to(device)

        mix_spec = codec.waveform_to_features(mix_wav)
        tgt_spec = codec.waveform_to_features(tgt_wav)
        res_spec = codec.waveform_to_features(res_wav)
        pred_spec = euler_sample(model, mix_spec, mix_spec, labels, args.num_steps, codec)
        final_spec, _ = apply_tf_mask_method(
            args.method, pred_spec, mix_spec, tgt_spec, res_spec, codec
        )
        pred_wav = codec.reconstruct_waveform(final_spec, mix_wav, length=mix_wav.shape[-1])
        sdr = si_sdr(pred_wav, tgt_wav)
        for i in range(mix_wav.shape[0]):
            idx = base + i
            rec = dataset.records[idx]
            scored.append({
                "idx": idx,
                "sample_id": str(rec["sample_id"]),
                "target_label": str(rec["target_label"]),
                "source_labels": [str(s) for s in rec["source_labels"]],
                "si_sdr": float(sdr[i].item()),
                "mixture": mix_wav[i].detach().cpu().numpy(),
                "target": tgt_wav[i].detach().cpu().numpy(),
                "prediction": pred_wav[i].detach().cpu().numpy(),
                "mix_lm": codec.features_to_log_magnitude(mix_spec[i:i+1], mix_wav[i:i+1]).squeeze().cpu().numpy(),
                "tgt_lm": codec.features_to_log_magnitude(tgt_spec[i:i+1], tgt_wav[i:i+1]).squeeze().cpu().numpy(),
                "pred_lm": codec.features_to_log_magnitude(final_spec[i:i+1], mix_wav[i:i+1]).squeeze().cpu().numpy(),
            })
        base += mix_wav.shape[0]

    # Retain at most one record per label before ranking by SI-SDR.
    best_per_label: dict[str, dict] = {}
    for s in scored:
        lab = s["target_label"]
        if lab not in best_per_label or s["si_sdr"] > best_per_label[lab]["si_sdr"]:
            best_per_label[lab] = s
    picks = sorted(best_per_label.values(), key=lambda x: -x["si_sdr"])[: args.pick]

    out = ensure_dir(Path(args.output_dir).resolve())
    sr = int(config["audio"]["sample_rate"])
    lines = ["分离效果试听样例\n\n",
             f"模型：{args.checkpoint}（IRM辅助训练）\n",
             f"后处理：{args.method}\n数据：{args.split} split，{args.num_steps}-step inference\n",
             f"候选范围：按manifest顺序扫描前{args.scan}条records\n",
             "样例按Target SI-SDR选取，仅用于试听，不代表测试集整体水平。\n\n"]
    for rank, s in enumerate(picks):
        name = f"demo_{rank:02d}_{s['target_label']}"
        d = ensure_dir(out / name)
        sf.write(d / "mixture.wav", s["mixture"], sr)
        sf.write(d / "target.wav", s["target"], sr)
        sf.write(d / "prediction.wav", s["prediction"], sr)
        save_triplet(d / "spectrogram_triplet.png", s["mix_lm"], s["tgt_lm"], s["pred_lm"],
                     title=f"{s['sample_id']} | target={s['target_label']} | SI-SDR={s['si_sdr']:.2f} dB")
        pair = " + ".join(s["source_labels"])
        lines.append(f"{name}：{pair}，目标{s['target_label']}，{s['si_sdr']:.2f} dB\n")
        print(f"{name}: SI-SDR={s['si_sdr']:.2f} dB  ({pair})")

    (out / "样例说明.txt").write_text("".join(lines), encoding="utf-8")
    print(f"\nSaved {len(picks)} demos to: {out}")


if __name__ == "__main__":
    main()
