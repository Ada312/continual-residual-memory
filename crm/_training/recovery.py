#!/usr/bin/env python3
"""Stage 1 training for utterance-local residual recovery."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from crm.recovery import ResidualSpeechProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train constrained residual speech recovery.")
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--validation-speakers", nargs="+", default=["p226", "p287"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=1,
        help="Run held-out validation every N epochs and at the final epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--max-gain", type=float, default=0.1)
    parser.add_argument("--initial-gain", type=float, default=0.005)
    parser.add_argument("--time-loss-weight", type=float, default=1.0)
    parser.add_argument("--complex-loss-weight", type=float, default=0.35)
    parser.add_argument("--logmag-loss-weight", type=float, default=0.2)
    parser.add_argument("--projection-loss-weight", type=float, default=0.15)
    parser.add_argument("--sisdr-loss-weight", type=float, default=0.01)
    parser.add_argument("--segmental-snr-loss-weight", type=float, default=0.0)
    parser.add_argument("--trust-loss-weight", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def load_mono(path: Path, sample_rate: int = 16_000) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path)
    audio = audio.mean(dim=0).float()
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


class PairedResidualDataset(Dataset):
    def __init__(
        self,
        clean_dir: Path,
        noisy_dir: Path,
        source_dir: Path,
        files: list[str],
        segment_length: int,
        training: bool,
    ):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.source_dir = source_dir
        self.files = files
        self.segment_length = segment_length
        self.training = training

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        filename = self.files[index]
        clean = load_mono(self.clean_dir / filename)
        noisy = load_mono(self.noisy_dir / filename)
        source = load_mono(self.source_dir / filename)
        length = min(clean.numel(), noisy.numel(), source.numel())
        clean, noisy, source = clean[:length], noisy[:length], source[:length]
        if length < self.segment_length:
            padding = self.segment_length - length
            clean = F.pad(clean, (0, padding))
            noisy = F.pad(noisy, (0, padding))
            source = F.pad(source, (0, padding))
            length = self.segment_length
        if self.training:
            start = random.randint(0, length - self.segment_length)
        else:
            start = (length - self.segment_length) // 2
        end = start + self.segment_length
        clean, noisy, source = clean[start:end], noisy[start:end], source[start:end]
        if self.training:
            amplitude = 10.0 ** random.uniform(-0.15, 0.05)
            clean, noisy, source = clean * amplitude, noisy * amplitude, source * amplitude
        return clean, noisy, source


def split_files(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    clean_files = {path.name for path in args.clean_dir.glob("*.wav")}
    noisy_files = {path.name for path in args.noisy_dir.glob("*.wav")}
    source_files = {path.name for path in args.source_dir.glob("*.wav")}
    files = sorted(clean_files & noisy_files & source_files)
    validation_speakers = set(args.validation_speakers)
    validation = [name for name in files if name.split("_")[0] in validation_speakers]
    training = [name for name in files if name.split("_")[0] not in validation_speakers]
    if not training or not validation:
        raise RuntimeError(
            f"empty split: training={len(training)}, validation={len(validation)}, total={len(files)}"
        )
    return training, validation


def normalized_si_sdr_loss(estimate: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    clean_zm = clean - clean.mean(dim=-1, keepdim=True)
    estimate_zm = estimate - estimate.mean(dim=-1, keepdim=True)
    scale = (estimate_zm * clean_zm).sum(-1, keepdim=True) / clean_zm.square().sum(
        -1, keepdim=True
    ).clamp_min(1e-7)
    target = scale * clean_zm
    ratio = target.square().sum(-1) / (estimate_zm - target).square().sum(-1).clamp_min(1e-7)
    return -10.0 * torch.log10(ratio.clamp_min(1e-7)).mean()


def differentiable_segmental_snr_loss(
    estimate: torch.Tensor,
    clean: torch.Tensor,
    sample_rate: int = 16_000,
    frame_seconds: float = 0.03,
    overlap: float = 0.75,
) -> torch.Tensor:
    """Differentiable counterpart of the clipped SNRseg evaluation metric."""
    frame_length = int(round(frame_seconds * sample_rate))
    hop_length = int(math.floor((1.0 - overlap) * frame_seconds * sample_rate))
    if frame_length < 1 or hop_length < 1:
        raise ValueError("segmental-SNR frame and hop lengths must be positive")
    if estimate.shape != clean.shape:
        raise ValueError("estimate and clean must have identical shapes")
    if estimate.shape[-1] < frame_length:
        padding = frame_length - estimate.shape[-1]
        estimate = F.pad(estimate, (0, padding))
        clean = F.pad(clean, (0, padding))

    samples = torch.arange(
        1, frame_length + 1, device=estimate.device, dtype=estimate.dtype
    )
    window = 0.5 * (
        1.0 - torch.cos(2.0 * torch.pi * samples / (frame_length + 1))
    )
    clean_frames = clean.unfold(-1, frame_length, hop_length) * window
    estimate_frames = estimate.unfold(-1, frame_length, hop_length) * window
    signal_energy = clean_frames.square().sum(dim=-1)
    error_energy = (clean_frames - estimate_frames).square().sum(dim=-1)
    eps = torch.finfo(estimate.dtype).eps
    segmental_snr = 10.0 * torch.log10(
        (signal_energy + eps) / (error_energy + eps)
    ).clamp(-10.0, 35.0)
    if segmental_snr.shape[-1] > 1:
        segmental_snr = segmental_snr[..., :-1]
    return -segmental_snr.mean()


def compute_loss(model, core_model, clean, noisy, source, args):
    estimate, gain, auxiliary = model(noisy, source)
    clean_spectrum = core_model.stft(clean)
    clean_scale = clean.abs().mean().clamp_min(1e-4)
    clean_spec_scale = clean_spectrum.abs().mean().clamp_min(1e-4)
    time_loss = (estimate - clean).abs().mean() / clean_scale
    complex_loss = (auxiliary["estimate_spectrum"] - clean_spectrum).abs().mean() / clean_spec_scale
    logmag_loss = F.l1_loss(
        torch.log(auxiliary["estimate_spectrum"].abs().clamp_min(1e-5)),
        torch.log(clean_spectrum.abs().clamp_min(1e-5)),
    )

    residual = auxiliary["residual_spectrum"]
    missing = clean_spectrum - auxiliary["source_spectrum"]
    oracle = (residual.conj() * missing).real / residual.abs().square().clamp_min(1e-7)
    lower_bound = -float(getattr(core_model, "negative_gain", 0.0))
    oracle = oracle.clamp(lower_bound, core_model.max_gain).detach()
    weights = residual.abs().square()
    weights = (weights / weights.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-7)).clamp_max(10.0)
    projection_loss = (
        weights
        * F.smooth_l1_loss(
            gain / core_model.max_gain, oracle / core_model.max_gain, reduction="none"
        )
    ).mean()
    sisdr_loss = normalized_si_sdr_loss(estimate, clean)
    segmental_snr_loss = differentiable_segmental_snr_loss(estimate, clean)
    trust_loss = gain.abs().mean() / core_model.max_gain
    loss = (
        args.time_loss_weight * time_loss
        + args.complex_loss_weight * complex_loss
        + args.logmag_loss_weight * logmag_loss
        + args.projection_loss_weight * projection_loss
        + args.sisdr_loss_weight * sisdr_loss
        + args.segmental_snr_loss_weight * segmental_snr_loss
        + args.trust_loss_weight * trust_loss
    )
    stats = {
        "loss": loss.detach(),
        "time": time_loss.detach(),
        "complex": complex_loss.detach(),
        "logmag": logmag_loss.detach(),
        "projection": projection_loss.detach(),
        "sisdr_loss": sisdr_loss.detach(),
        "segmental_snr_loss": segmental_snr_loss.detach(),
        "gain": gain.abs().mean().detach(),
    }
    return loss, stats


def reduce_stats(totals: dict[str, torch.Tensor], count: int, device: torch.device):
    keys = sorted(totals)
    values = torch.stack([totals[key] for key in keys] + [torch.tensor(float(count), device=device)])
    if dist.is_initialized():
        dist.all_reduce(values)
    denominator = values[-1].clamp_min(1.0)
    return {key: float(value / denominator) for key, value in zip(keys, values[:-1])}


def run_epoch(model, loader, optimizer, device, training: bool, args):
    model.train(training)
    totals: dict[str, torch.Tensor] = {}
    count = 0
    iterator = tqdm(loader, disable=dist.is_initialized() and dist.get_rank() != 0)
    for clean, noisy, source in iterator:
        clean, noisy, source = clean.to(device), noisy.to(device), source.to(device)
        with torch.set_grad_enabled(training):
            core_model = model.module if hasattr(model, "module") else model
            loss, stats = compute_loss(model, core_model, clean, noisy, source, args)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_size = clean.shape[0]
        for key, value in stats.items():
            totals[key] = totals.get(key, torch.zeros((), device=device)) + value * batch_size
        count += batch_size
        iterator.set_postfix(loss=float(loss.detach()), gain=float(stats["gain"]))
    return reduce_stats(totals, count, device)


def main() -> None:
    args = parse_args()
    if args.validation_interval < 1:
        raise ValueError("--validation-interval must be positive")
    loss_weights = {
        name: value
        for name, value in vars(args).items()
        if name.endswith("_loss_weight")
    }
    if any(value < 0.0 for value in loss_weights.values()):
        raise ValueError(f"loss weights must be non-negative: {loss_weights}")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    training_files, validation_files = split_files(args)
    segment_length = int(16_000 * args.segment_seconds)
    train_set = PairedResidualDataset(
        args.clean_dir, args.noisy_dir, args.source_dir, training_files, segment_length, True
    )
    validation_set = PairedResidualDataset(
        args.clean_dir, args.noisy_dir, args.source_dir, validation_files, segment_length, False
    )
    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    validation_sampler = DistributedSampler(validation_set, shuffle=False) if world_size > 1 else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        sampler=validation_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    bare_model = ResidualSpeechProjector(
        channels=args.channels,
        max_gain=args.max_gain,
        initial_gain=args.initial_gain,
    ).to(device)
    model = DistributedDataParallel(bare_model, device_ids=[local_rank]) if world_size > 1 else bare_model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_validation = float("inf")
    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"filename": training_files}).to_csv(
            args.out_dir / "training_subset.csv", index=False
        )
        (args.out_dir / "config.json").write_text(
            json.dumps({**vars(args), "clean_dir": str(args.clean_dir), "noisy_dir": str(args.noisy_dir),
                        "source_dir": str(args.source_dir), "out_dir": str(args.out_dir),
                        "training_files": len(training_files), "validation_files": len(validation_files)},
                       indent=2) + "\n",
            encoding="utf-8",
        )

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats = run_epoch(model, train_loader, optimizer, device, True, args)
        should_validate = (
            (epoch + 1) % args.validation_interval == 0
            or epoch + 1 == args.epochs
        )
        validation_stats = (
            run_epoch(model, validation_loader, optimizer, device, False, args)
            if should_validate
            else {}
        )
        scheduler.step()
        row = {"epoch": epoch + 1, **{f"train_{k}": v for k, v in train_stats.items()},
               **{f"validation_{k}": v for k, v in validation_stats.items()}}
        if rank == 0:
            history.append(row)
            print(json.dumps(row))
            state = (model.module if hasattr(model, "module") else model).state_dict()
            torch.save(state, args.out_dir / "last.th")
            if should_validate and validation_stats["loss"] < best_validation:
                best_validation = validation_stats["loss"]
                torch.save(state, args.out_dir / "best.th")
            pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
