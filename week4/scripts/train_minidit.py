#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_features import STFTFeatureCodec
from common import (
    append_csv,
    count_trainable_parameters,
    ensure_dir,
    load_config,
    load_torch_checkpoint,
    read_json,
    set_seed,
    stringify_paths,
    write_json,
)
from dataset import SeparationManifestDataset
from flow_matching import sample_flow_matching_tuple
from modeling_minidit import MiniDiT, MiniDiTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Week 3 self-attention Mini-DiT baseline.")
    parser.add_argument("--config", default="week3/configs/minidit_self_attention.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--overfit-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_datasets(config: dict[str, Any], overfit_samples: int | None = None) -> tuple[SeparationManifestDataset, SeparationManifestDataset]:
    data_root = Path(config["paths"]["data_root"])
    class_names = list(config["dataset"]["class_names"])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    kwargs = {
        "data_root": data_root,
        "class_to_idx": class_to_idx,
        "sample_rate": int(config["audio"]["sample_rate"]),
        "num_samples": int(config["audio"]["num_samples"]),
    }
    train_ds = SeparationManifestDataset(
        manifest_path=data_root / "metadata" / "manifest_train.jsonl",
        limit=overfit_samples,
        **kwargs,
    )
    val_ds = SeparationManifestDataset(
        manifest_path=data_root / "metadata" / "manifest_val.jsonl",
        limit=overfit_samples,
        **kwargs,
    )
    return train_ds, val_ds


def build_model(config: dict[str, Any]) -> MiniDiT:
    spec_cfg = config["spectrogram"]
    model_cfg = config["model"]
    cfg = MiniDiTConfig(
        input_size=tuple(spec_cfg["input_size"]),
        patch_size=tuple(model_cfg["patch_size"]),
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        hidden_size=int(model_cfg["hidden_size"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        num_classes=len(config["dataset"]["class_names"]),
        mixer_type=str(model_cfg["mixer_type"]),
        mamba_backend=str(model_cfg.get("mamba_backend", "torch")),
        mamba_d_state=int(model_cfg.get("mamba_d_state", 16)),
        mamba_d_conv=int(model_cfg.get("mamba_d_conv", 4)),
        mamba_expand=float(model_cfg.get("mamba_expand", 1.0)),
        hybrid_start=str(model_cfg.get("hybrid_start", "self_attention")),
    )
    return MiniDiT(cfg)


def build_feature_codec(config: dict[str, Any], sigma: float) -> STFTFeatureCodec:
    return STFTFeatureCodec(
        sigma=sigma,
        **{k: v for k, v in config["spectrogram"].items() if k != "input_size"},
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mix_wav = batch["mixture_waveform"].to(device=device, non_blocking=True)
    tgt_wav = batch["target_waveform"].to(device=device, non_blocking=True)
    labels = batch["label"].to(device=device, non_blocking=True)
    return mix_wav, tgt_wav, labels


def run_epoch(
    model: nn.Module,
    feature_extractor: STFTFeatureCodec,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float,
    use_amp: bool,
    flow_start: str,
    velocity_loss_weight: float,
    endpoint_loss_weight: float,
    max_steps: int | None,
    desc: str,
    fixed_seed: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = []
    velocity_losses = []
    endpoint_losses = []
    iterator = tqdm(loader, desc=desc, leave=False)
    autocast_enabled = use_amp and device.type == "cuda"
    generator = None
    if fixed_seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(fixed_seed))

    for step, batch in enumerate(iterator, start=1):
        mix_wav, tgt_wav, labels = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                mixture_spec = feature_extractor.waveform_to_features(mix_wav)
                target_spec = feature_extractor.waveform_to_features(tgt_wav)
                if flow_start != "mixture":
                    raise ValueError(f"Unsupported flow_start for Week 3 baseline: {flow_start}")
                _, xt, v_star, t = sample_flow_matching_tuple(mixture_spec, target_spec, generator=generator)
                v_pred = model(xt, t, mixture_spec, labels)
                velocity_loss = F.l1_loss(v_pred, v_star)
                t_view = t.view(-1, 1, 1, 1)
                endpoint_pred = xt + (1.0 - t_view) * v_pred
                endpoint_loss = F.l1_loss(endpoint_pred, target_spec)
                loss = velocity_loss_weight * velocity_loss + endpoint_loss_weight * endpoint_loss

            if is_train:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

        totals.append(float(loss.detach().cpu()))
        velocity_losses.append(float(velocity_loss.detach().cpu()))
        endpoint_losses.append(float(endpoint_loss.detach().cpu()))
        iterator.set_postfix(
            loss=f"{totals[-1]:.4f}",
            fm=f"{velocity_losses[-1]:.4f}",
            end=f"{endpoint_losses[-1]:.4f}",
        )
        if max_steps is not None and step >= max_steps:
            break

    count = max(len(totals), 1)
    return {
        "loss": sum(totals) / count,
        "velocity_loss": sum(velocity_losses) / count,
        "endpoint_loss": sum(endpoint_losses) / count,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": stringify_paths(config),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.data_root is not None:
        config["paths"]["data_root"] = Path(args.data_root).resolve()
    if args.output_dir is not None:
        config["paths"]["output_dir"] = Path(args.output_dir).resolve()

    train_cfg = config["training"]
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.max_train_steps is not None:
        train_cfg["max_train_steps"] = args.max_train_steps
    if args.max_val_steps is not None:
        train_cfg["max_val_steps"] = args.max_val_steps

    set_seed(int(config["seed"]))
    device = resolve_device(args.device)
    output_dir = ensure_dir(config["paths"]["output_dir"])
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_path = output_dir / "metrics.csv"
    stats_path = config["paths"].get("feature_stats_path", config["paths"].get("sigma_stats_path"))
    if stats_path is None:
        raise KeyError("Expected paths.feature_stats_path or paths.sigma_stats_path in config.")
    sigma_stats = read_json(stats_path)
    sigma = float(sigma_stats["sigma"])

    train_ds, val_ds = build_datasets(config, overfit_samples=args.overfit_samples)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=pin_memory,
        drop_last=False,
    )

    feature_extractor = build_feature_codec(config, sigma).to(device)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["use_amp"]) and device.type == "cuda")

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume is not None:
        checkpoint = load_torch_checkpoint(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    write_json(
        output_dir / "run_summary.json",
        {
            "device": str(device),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "sigma": sigma,
            "representation": str(config["spectrogram"]["representation"]),
            "mixer_type": str(config["model"]["mixer_type"]),
            "mixer_summary": model.mixer_summary() if hasattr(model, "mixer_summary") else [],
            "parameters": count_trainable_parameters(model),
            "config_path": str(config["paths"]["config_path"]),
        },
    )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Sigma: {sigma:.6f}")
    print(f"Output dir: {output_dir}")

    epochs = 1 if args.dry_run else int(train_cfg["epochs"])
    max_train_steps = int(train_cfg["max_train_steps"]) if train_cfg["max_train_steps"] is not None else None
    max_val_steps = int(train_cfg["max_val_steps"]) if train_cfg["max_val_steps"] is not None else None

    for epoch in range(start_epoch, start_epoch + epochs):
        start = time.time()
        train_metrics = run_epoch(
            model=model,
            feature_extractor=feature_extractor,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=float(train_cfg["grad_clip_norm"]),
            use_amp=bool(train_cfg["use_amp"]),
            flow_start=str(train_cfg["flow_start"]),
            velocity_loss_weight=float(train_cfg["velocity_loss_weight"]),
            endpoint_loss_weight=float(train_cfg["endpoint_loss_weight"]),
            max_steps=max_train_steps,
            desc=f"train {epoch}",
        )
        val_metrics = run_epoch(
            model=model,
            feature_extractor=feature_extractor,
            loader=val_loader,
            device=device,
            optimizer=None,
            scaler=None,
            grad_clip_norm=float(train_cfg["grad_clip_norm"]),
            use_amp=bool(train_cfg["use_amp"]),
            flow_start=str(train_cfg["flow_start"]),
            velocity_loss_weight=float(train_cfg["velocity_loss_weight"]),
            endpoint_loss_weight=float(train_cfg["endpoint_loss_weight"]),
            max_steps=max_val_steps,
            desc=f"val {epoch}",
            fixed_seed=int(train_cfg["fixed_validation_seed"]),
        )
        elapsed = time.time() - start
        append_csv(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_velocity_loss": train_metrics["velocity_loss"],
                "train_endpoint_loss": train_metrics["endpoint_loss"],
                "val_loss": val_metrics["loss"],
                "val_velocity_loss": val_metrics["velocity_loss"],
                "val_endpoint_loss": val_metrics["endpoint_loss"],
                "seconds": elapsed,
            },
        )
        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_fm={train_metrics['velocity_loss']:.6f} "
            f"train_end={train_metrics['endpoint_loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_fm={val_metrics['velocity_loss']:.6f} "
            f"val_end={val_metrics['endpoint_loss']:.6f} "
            f"time={elapsed:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, best_val_loss, config)
        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, best_val_loss, config)

    print("Training finished.")


if __name__ == "__main__":
    main()
