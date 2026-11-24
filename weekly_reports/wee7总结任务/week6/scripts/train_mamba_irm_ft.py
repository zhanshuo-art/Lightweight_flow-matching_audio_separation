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

from tf_mask_common import PROJECT_ROOT, compute_irm_loss

from common import append_csv, count_trainable_parameters, ensure_dir, load_config, load_torch_checkpoint, read_json, set_seed, stringify_paths, write_json
from flow_matching import sample_flow_matching_tuple
from train_mamba_waveform_ft import build_datasets, build_feature_codec, build_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week6 Mamba continued training with waveform and IRM auxiliary losses.")
    parser.add_argument("--config", default="week5/configs/mamba_waveform_finetune.yaml")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--waveform-loss-weight", type=float, default=0.1)
    parser.add_argument("--irm-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-val-steps", type=int, default=None)
    parser.add_argument("--overfit-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mix_wav = batch["mixture_waveform"].to(device=device, non_blocking=True)
    tgt_wav = batch["target_waveform"].to(device=device, non_blocking=True)
    res_wav = batch["residual_waveform"].to(device=device, non_blocking=True)
    labels = batch["label"].to(device=device, non_blocking=True)
    return mix_wav, tgt_wav, res_wav, labels


def run_epoch(
    model: nn.Module,
    feature_extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    grad_clip_norm: float,
    use_amp: bool,
    flow_start: str,
    velocity_loss_weight: float,
    endpoint_loss_weight: float,
    waveform_loss_weight: float,
    irm_loss_weight: float,
    num_samples: int,
    max_steps: int | None,
    desc: str,
    fixed_seed: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals: list[float] = []
    flow_losses: list[float] = []
    velocity_losses: list[float] = []
    endpoint_losses: list[float] = []
    waveform_losses: list[float] = []
    irm_losses: list[float] = []
    iterator = tqdm(loader, desc=desc, leave=False)
    autocast_enabled = use_amp and device.type == "cuda"
    generator = None
    if fixed_seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(fixed_seed))

    for step, batch in enumerate(iterator, start=1):
        mix_wav, tgt_wav, res_wav, labels = move_batch(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                mixture_spec = feature_extractor.waveform_to_features(mix_wav)
                target_spec = feature_extractor.waveform_to_features(tgt_wav)
                residual_spec = feature_extractor.waveform_to_features(res_wav)
                if flow_start != "mixture":
                    raise ValueError(f"Unsupported flow_start for Week6 IRM fine-tuning: {flow_start}")
                _, xt, v_star, t = sample_flow_matching_tuple(mixture_spec, target_spec, generator=generator)
                v_pred = model(xt, t, mixture_spec, labels)
                velocity_loss = F.l1_loss(v_pred, v_star)
                t_view = t.view(-1, 1, 1, 1)
                endpoint_pred = xt + (1.0 - t_view) * v_pred
                endpoint_loss = F.l1_loss(endpoint_pred, target_spec)
                flow_loss = velocity_loss_weight * velocity_loss + endpoint_loss_weight * endpoint_loss

            with torch.amp.autocast(device_type=device.type, enabled=False):
                prediction_waveform = feature_extractor.reconstruct_waveform(
                    endpoint_pred.float(),
                    mix_wav.float(),
                    length=int(num_samples),
                )
                waveform_loss = F.l1_loss(prediction_waveform, tgt_wav.float())
                irm_loss = compute_irm_loss(
                    endpoint_pred.float(),
                    mixture_spec.float(),
                    target_spec.float(),
                    residual_spec.float(),
                    feature_extractor,
                )
                loss = flow_loss.float() + float(waveform_loss_weight) * waveform_loss + float(irm_loss_weight) * irm_loss

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
        flow_losses.append(float(flow_loss.detach().cpu()))
        velocity_losses.append(float(velocity_loss.detach().cpu()))
        endpoint_losses.append(float(endpoint_loss.detach().cpu()))
        waveform_losses.append(float(waveform_loss.detach().cpu()))
        irm_losses.append(float(irm_loss.detach().cpu()))
        iterator.set_postfix(
            loss=f"{totals[-1]:.4f}",
            flow=f"{flow_losses[-1]:.4f}",
            wav=f"{waveform_losses[-1]:.4f}",
            irm=f"{irm_losses[-1]:.4f}",
        )
        if max_steps is not None and step >= max_steps:
            break

    count = max(len(totals), 1)
    return {
        "loss": sum(totals) / count,
        "flow_loss": sum(flow_losses) / count,
        "velocity_loss": sum(velocity_losses) / count,
        "endpoint_loss": sum(endpoint_losses) / count,
        "waveform_l1": sum(waveform_losses) / count,
        "irm_l1": sum(irm_losses) / count,
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
    output_dir = Path(args.output_dir).resolve() if args.output_dir else PROJECT_ROOT / "week6" / "runs" / "mamba_irm_ft_v1"
    config["paths"]["output_dir"] = output_dir

    train_cfg = config["training"]
    train_cfg["epochs"] = int(args.epochs)
    train_cfg["batch_size"] = int(args.batch_size)
    train_cfg["num_workers"] = int(args.num_workers)
    train_cfg["learning_rate"] = float(args.learning_rate)
    train_cfg["waveform_loss_weight"] = float(args.waveform_loss_weight)
    train_cfg["irm_loss_weight"] = float(args.irm_loss_weight)
    train_cfg["max_train_steps"] = args.max_train_steps
    train_cfg["max_val_steps"] = args.max_val_steps

    init_checkpoint = args.init_checkpoint
    if init_checkpoint is None:
        init_checkpoint = str((PROJECT_ROOT / "week5" / "runs" / "mamba_waveform_ft_v1" / "checkpoints" / "best.pt").resolve())

    set_seed(int(config["seed"]))
    device = resolve_device(args.device)
    output_dir = ensure_dir(output_dir)
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    metrics_path = output_dir / "metrics.csv"
    stats_path = config["paths"].get("feature_stats_path", config["paths"].get("sigma_stats_path"))
    if stats_path is None:
        raise KeyError("Expected paths.feature_stats_path or paths.sigma_stats_path in config.")
    sigma = float(read_json(stats_path)["sigma"])

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
        print(f"Resumed from: {args.resume}")
    else:
        checkpoint = load_torch_checkpoint(init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Initialized model weights from: {init_checkpoint}")

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
            "init_checkpoint": str(init_checkpoint),
            "waveform_loss_weight": float(train_cfg["waveform_loss_weight"]),
            "irm_loss_weight": float(train_cfg["irm_loss_weight"]),
        },
    )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
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
            waveform_loss_weight=float(train_cfg["waveform_loss_weight"]),
            irm_loss_weight=float(train_cfg["irm_loss_weight"]),
            num_samples=int(config["audio"]["num_samples"]),
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
            waveform_loss_weight=float(train_cfg["waveform_loss_weight"]),
            irm_loss_weight=float(train_cfg["irm_loss_weight"]),
            num_samples=int(config["audio"]["num_samples"]),
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
                "train_flow_loss": train_metrics["flow_loss"],
                "train_velocity_loss": train_metrics["velocity_loss"],
                "train_endpoint_loss": train_metrics["endpoint_loss"],
                "train_waveform_l1": train_metrics["waveform_l1"],
                "train_irm_l1": train_metrics["irm_l1"],
                "val_loss": val_metrics["loss"],
                "val_flow_loss": val_metrics["flow_loss"],
                "val_velocity_loss": val_metrics["velocity_loss"],
                "val_endpoint_loss": val_metrics["endpoint_loss"],
                "val_waveform_l1": val_metrics["waveform_l1"],
                "val_irm_l1": val_metrics["irm_l1"],
                "seconds": elapsed,
            },
        )
        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_flow={train_metrics['flow_loss']:.6f} "
            f"train_wav={train_metrics['waveform_l1']:.6f} "
            f"train_irm={train_metrics['irm_l1']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_flow={val_metrics['flow_loss']:.6f} "
            f"val_wav={val_metrics['waveform_l1']:.6f} "
            f"val_irm={val_metrics['irm_l1']:.6f} "
            f"time={elapsed:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, best_val_loss, config)
        save_checkpoint(checkpoints_dir / "last.pt", model, optimizer, epoch, best_val_loss, config)

    print("Week6 IRM fine-tuning finished.")


if __name__ == "__main__":
    main()

