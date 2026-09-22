#!/usr/bin/env python3
"""Stage 2 training for the paper's memory-conditioned refinement."""
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch_pesq import PesqLoss
import torch_pesq.bark as torch_pesq_bark
import torch_pesq.loss as torch_pesq_loss
import torch_pesq.loudness as torch_pesq_loudness

from crm.memory import CausalPrototypeResidualNoiseMemory
from crm.recovery import UtteranceLocalRecovery
from crm.refinement import MemoryConditionedRefinement

# torch-pesq 0.1.x resolves string shape annotations as globals on Python 3.11.
for _module in (torch_pesq_bark, torch_pesq_loss, torch_pesq_loudness):
    for _dimension in ("batch", "sample", "frame", "band", "bark"):
        setattr(_module, _dimension, _dimension)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: train memory-conditioned refinement on EARS-WHAM."
    )
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, default=None)
    parser.add_argument("--validation-speakers", nargs="+", default=["p100", "p101"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--memory-dropout", type=float, default=0.2)
    parser.add_argument("--counterfactual-memory-weight", type=float, default=0.1)
    parser.add_argument("--counterfactual-change-weight", type=float, default=0.1)
    parser.add_argument("--noise-frame-fraction", type=float, default=0.3)
    parser.add_argument("--memory-warmup", type=int, default=20)
    parser.add_argument("--memory-prototypes", type=int, default=2)
    parser.add_argument("--novelty-threshold", type=float, default=0.35)
    parser.add_argument("--posterior-temperature", type=float, default=0.5)
    parser.add_argument("--reliability-power", type=float, default=0.0)
    parser.add_argument("--posterior-confidence-power", type=float, default=0.0)
    parser.add_argument("--delta-rank", type=int, default=4)
    parser.add_argument("--memory-delta-gain", type=float, default=0.08)
    parser.add_argument("--torch-pesq-loss-weight", type=float, default=2.0)
    parser.add_argument("--sisdr-loss-weight", type=float, default=0.03)
    parser.add_argument("--segmental-snr-loss-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_mono(path: Path, sample_rate: int = 16_000) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path)
    audio = audio.mean(dim=0).float()
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio




def load_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=r"\s+", names=["stem", "noise", "snr"])
    frame["filename"] = frame["stem"] + ".wav"
    return frame


def available_files(args: argparse.Namespace, metadata: pd.DataFrame) -> list[str]:
    clean = {path.name for path in args.clean_dir.glob("*.wav")}
    noisy = {path.name for path in args.noisy_dir.glob("*.wav")}
    source = {path.name for path in args.source_dir.glob("*.wav")}
    metadata_files = set(metadata["filename"])
    return sorted(clean & noisy & source & metadata_files)




def stft(audio: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        audio,
        n_fft=512,
        hop_length=128,
        window=torch.hann_window(512),
        return_complex=True,
    )


def build_contexts(
    noisy_dir: Path,
    source_dir: Path,
    metadata: pd.DataFrame,
    files: list[str],
    noise_frame_fraction: float,
    memory_warmup: int,
    memory_prototypes: int,
    novelty_threshold: float,
    description: str,
) -> dict[str, dict[str, torch.Tensor | int]]:
    """Build causal K_train contexts from grouped EARS-WHAM histories."""
    allowed = set(files)
    subset = metadata[metadata["filename"].isin(allowed)].copy()
    locations = subset["noise"].astype(str).str.extract(r"(loc\d+)", expand=False)
    subset["context_group"] = locations.fillna(subset["noise"].astype(str))
    streams = [
        sorted(group["filename"].tolist())
        for _, group in subset.groupby("context_group", sort=True)
    ]

    def compute_observation(filename: str) -> tuple[str, torch.Tensor]:
        noisy = load_mono(noisy_dir / filename)
        source = load_mono(source_dir / filename)
        length = min(noisy.numel(), source.numel())
        source_spectrum = stft(source[:length])
        residual_spectrum = stft(noisy[:length] - source[:length])
        source_power = source_spectrum.abs().square()
        residual_power = residual_spectrum.abs().square()
        occupancy = source_power.sum(0) / (
            source_power.sum(0) + residual_power.sum(0) + 1e-8
        )
        selected = max(
            1, int(math.ceil(occupancy.numel() * noise_frame_fraction))
        )
        indices = torch.topk(occupancy, selected, largest=False).indices
        observation = torch.log(
            residual_power[:, indices] + 1e-8
        ).median(dim=1).values.cpu()
        return filename, observation

    previous_torch_threads = torch.get_num_threads()
    torch.set_num_threads(8)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            observations = dict(
                tqdm(
                    executor.map(compute_observation, files),
                    total=len(files),
                    desc=f"{description} observations",
                )
            )
    finally:
        torch.set_num_threads(previous_torch_threads)

    contexts: dict[str, dict[str, torch.Tensor | int]] = {}
    for stream_files in tqdm(streams, desc=description):
        memory = CausalPrototypeResidualNoiseMemory(
            prototypes=memory_prototypes,
            noise_frame_fraction=noise_frame_fraction,
            warmup_utterances=memory_warmup,
            novelty_threshold=novelty_threshold,
        )
        for filename in stream_files:
            memory_prev = memory.context()
            contexts[filename] = {
                "mean": memory_prev.means,
                "variance": memory_prev.variances,
                "count": memory_prev.counts,
            }
            memory.update_observation(observations[filename])
    if len(contexts) != len(files):
        raise RuntimeError(f"built {len(contexts)} contexts for {len(files)} files")
    return contexts


class ContinualResidualDataset(Dataset):
    def __init__(
        self,
        clean_dir: Path,
        noisy_dir: Path,
        source_dir: Path,
        files: list[str],
        contexts: dict[str, dict[str, torch.Tensor | int]],
        segment_length: int,
        training: bool,
    ):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.source_dir = source_dir
        self.files = files
        self.contexts = contexts
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
        start = (
            random.randint(0, length - self.segment_length)
            if self.training
            else (length - self.segment_length) // 2
        )
        end = start + self.segment_length
        clean, noisy, source = clean[start:end], noisy[start:end], source[start:end]
        if self.training:
            amplitude = 10.0 ** random.uniform(-0.15, 0.05)
            clean, noisy, source = clean * amplitude, noisy * amplitude, source * amplitude

        context = self.contexts[filename]
        return (
            clean,
            noisy,
            source,
            context["mean"],
            context["variance"],
            torch.as_tensor(context["count"], dtype=torch.float32),
        )


def per_sample_normalized_si_sdr_loss(
    estimate: torch.Tensor, clean: torch.Tensor
) -> torch.Tensor:
    clean_zm = clean - clean.mean(dim=-1, keepdim=True)
    estimate_zm = estimate - estimate.mean(dim=-1, keepdim=True)
    scale = (estimate_zm * clean_zm).sum(-1, keepdim=True) / clean_zm.square().sum(
        -1, keepdim=True
    ).clamp_min(1e-7)
    target = scale * clean_zm
    ratio = target.square().sum(-1) / (estimate_zm - target).square().sum(-1).clamp_min(1e-7)
    return -10.0 * torch.log10(ratio.clamp_min(1e-7))


def normalized_si_sdr_loss(estimate: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    return per_sample_normalized_si_sdr_loss(estimate, clean).mean()


def per_sample_differentiable_segmental_snr_loss(
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

    dtype = estimate.dtype
    device = estimate.device
    samples = torch.arange(1, frame_length + 1, device=device, dtype=dtype)
    window = 0.5 * (
        1.0 - torch.cos(2.0 * torch.pi * samples / (frame_length + 1))
    )
    clean_frames = clean.unfold(-1, frame_length, hop_length) * window
    estimate_frames = estimate.unfold(-1, frame_length, hop_length) * window
    signal_energy = clean_frames.square().sum(dim=-1)
    error_energy = (clean_frames - estimate_frames).square().sum(dim=-1)
    eps = torch.finfo(dtype).eps
    segmental_snr = 10.0 * torch.log10(
        (signal_energy + eps) / (error_energy + eps)
    )
    segmental_snr = segmental_snr.clamp(-10.0, 35.0)
    if segmental_snr.shape[-1] > 1:
        segmental_snr = segmental_snr[..., :-1]
    return -segmental_snr.mean(dim=-1)


def differentiable_segmental_snr_loss(
    estimate: torch.Tensor,
    clean: torch.Tensor,
    sample_rate: int = 16_000,
    frame_seconds: float = 0.03,
    overlap: float = 0.75,
) -> torch.Tensor:
    return per_sample_differentiable_segmental_snr_loss(
        estimate,
        clean,
        sample_rate=sample_rate,
        frame_seconds=frame_seconds,
        overlap=overlap,
    ).mean()




def per_sample_reconstruction_proxy(
    model: MemoryConditionedRefinement,
    estimate: torch.Tensor,
    clean: torch.Tensor,
) -> torch.Tensor:
    """Source-reference proxy used only to create counterfactual training regrets."""
    clean_spectrum = model.stft(clean)
    estimate_spectrum = model.stft(estimate)
    clean_scale = clean.abs().mean(dim=-1).clamp_min(1e-4)
    clean_spec_scale = clean_spectrum.abs().mean(dim=(-2, -1)).clamp_min(1e-4)
    time_error = (estimate - clean).abs().mean(dim=-1) / clean_scale
    complex_error = (
        (estimate_spectrum - clean_spectrum).abs().mean(dim=(-2, -1))
        / clean_spec_scale
    )
    logmag_error = F.l1_loss(
        torch.log(estimate_spectrum.abs().clamp_min(1e-5)),
        torch.log(clean_spectrum.abs().clamp_min(1e-5)),
        reduction="none",
    ).mean(dim=(-2, -1))
    return time_error + 0.35 * complex_error + 0.2 * logmag_error






















def compute_loss(
    model: MemoryConditionedRefinement,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    source: torch.Tensor,
    memory_mean: torch.Tensor,
    memory_variance: torch.Tensor,
    memory_count: torch.Tensor,
    *,
    pesq_loss_fn: PesqLoss,
    pesq_loss_weight: float,
    sisdr_loss_weight: float,
    segmental_snr_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Paper base objective for memory-conditioned refinement."""
    estimate, gain, auxiliary = model(
        noisy, source, memory_mean, memory_variance, memory_count
    )
    clean_spectrum = model.stft(clean)
    clean_scale = clean.abs().mean().clamp_min(1e-4)
    clean_spec_scale = clean_spectrum.abs().mean().clamp_min(1e-4)
    time_loss = (estimate - clean).abs().mean() / clean_scale
    complex_loss = (
        (auxiliary["estimate_spectrum"] - clean_spectrum).abs().mean()
        / clean_spec_scale
    )
    logmag_loss = F.l1_loss(
        torch.log(auxiliary["estimate_spectrum"].abs().clamp_min(1e-5)),
        torch.log(clean_spectrum.abs().clamp_min(1e-5)),
    )

    residual_spectrum = auxiliary["residual_spectrum"]
    missing_spectrum = clean_spectrum - auxiliary["source_spectrum"]
    weights = residual_spectrum.abs().square()
    weights = (
        weights / weights.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    ).clamp_max(10.0)
    oracle_gain = (
        residual_spectrum.conj() * missing_spectrum
    ).real / residual_spectrum.abs().square().clamp_min(1e-7)
    oracle_gain = oracle_gain.clamp(0.0, model.max_gain).detach()
    projection_loss = (
        weights
        * F.smooth_l1_loss(
            gain / model.max_gain,
            oracle_gain / model.max_gain,
            reduction="none",
        )
    ).mean()
    sisdr_loss = normalized_si_sdr_loss(estimate, clean)
    segmental_snr_loss = differentiable_segmental_snr_loss(estimate, clean)
    trust_loss = gain.abs().mean() / model.max_gain
    perceptual_loss = pesq_loss_fn(clean, estimate).mean()
    loss = (
        time_loss
        + 0.35 * complex_loss
        + 0.2 * logmag_loss
        + 0.15 * projection_loss
        + sisdr_loss_weight * sisdr_loss
        + segmental_snr_loss_weight * segmental_snr_loss
        + 0.01 * trust_loss
        + pesq_loss_weight * perceptual_loss
    )
    return loss, {
        "loss": loss.detach(),
        "time": time_loss.detach(),
        "complex": complex_loss.detach(),
        "logmag": logmag_loss.detach(),
        "projection": projection_loss.detach(),
        "sisdr_loss": sisdr_loss.detach(),
        "segmental_snr_loss": segmental_snr_loss.detach(),
        "gain": gain.abs().mean().detach(),
        "perceptual_loss": perceptual_loss.detach(),
    }


def run_epoch(
    model: MemoryConditionedRefinement,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    memory_dropout: float,
    pesq_loss_fn: PesqLoss,
    pesq_loss_weight: float,
    sisdr_loss_weight: float,
    segmental_snr_loss_weight: float,
    counterfactual_memory_weight: float,
    counterfactual_change_weight: float,
) -> dict[str, float]:
    """Run the paper's matched/shuffled/empty-memory Stage 2 objective."""
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    count = 0
    iterator = tqdm(loader, desc="train" if training else "validation")
    for batch in iterator:
        clean, noisy, source, memory_mean, memory_variance, memory_count = [
            value.to(device, non_blocking=True) for value in batch
        ]
        if training and memory_dropout > 0.0:
            keep = (torch.rand_like(memory_count) >= memory_dropout).float()
            memory_count = memory_count * keep

        with torch.set_grad_enabled(training):
            loss, stats = compute_loss(
                model,
                clean,
                noisy,
                source,
                memory_mean,
                memory_variance,
                memory_count,
                pesq_loss_fn=pesq_loss_fn,
                pesq_loss_weight=pesq_loss_weight,
                sisdr_loss_weight=sisdr_loss_weight,
                segmental_snr_loss_weight=segmental_snr_loss_weight,
            )
            matched_regret = torch.zeros((), device=device)
            shuffled_regret = torch.zeros((), device=device)
            shuffled_consistency = torch.zeros((), device=device)

            if training and counterfactual_memory_weight > 0.0 and clean.shape[0] > 1:
                with torch.no_grad():
                    empty_count = torch.zeros_like(memory_count)
                    empty_estimate, _, _ = model(
                        noisy, source, memory_mean, memory_variance, empty_count
                    )
                    empty_proxy = per_sample_reconstruction_proxy(
                        model, empty_estimate, clean
                    )

                matched_estimate, _, _ = model(
                    noisy, source, memory_mean, memory_variance, memory_count
                )
                matched_proxy = per_sample_reconstruction_proxy(
                    model, matched_estimate, clean
                )
                matched_regret = F.relu(matched_proxy - empty_proxy).mean()

                shift = int(torch.randint(1, clean.shape[0], (), device=device).item())
                shuffled_estimate, _, _ = model(
                    noisy,
                    source,
                    memory_mean.roll(shift, dims=0),
                    memory_variance.roll(shift, dims=0),
                    memory_count.roll(shift, dims=0),
                )
                shuffled_proxy = per_sample_reconstruction_proxy(
                    model, shuffled_estimate, clean
                )
                shuffled_excess = shuffled_proxy - empty_proxy
                shuffled_regret = F.relu(shuffled_excess).mean()
                harmful_probability = torch.sigmoid(
                    shuffled_excess.detach() / 0.02
                )
                normalized_change = (
                    (shuffled_estimate - empty_estimate).abs().mean(dim=-1)
                    / clean.abs().mean(dim=-1).clamp_min(1e-4)
                )
                shuffled_consistency = (
                    harmful_probability * normalized_change
                ).mean()
                loss = loss + counterfactual_memory_weight * (
                    matched_regret
                    + shuffled_regret
                    + counterfactual_change_weight * shuffled_consistency
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    5.0,
                )
                optimizer.step()

        stats["matched_regret"] = matched_regret.detach()
        stats["counterfactual_regret"] = shuffled_regret.detach()
        stats["counterfactual_change"] = shuffled_consistency.detach()
        stats["optimization_loss"] = loss.detach()
        batch_size = clean.shape[0]
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + float(value) * batch_size
        count += batch_size
        iterator.set_postfix(loss=float(loss.detach()), gain=float(stats["gain"]))
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if args.delta_rank < 1:
        raise ValueError("--delta-rank must be positive")
    if not 0.0 <= args.memory_dropout < 1.0:
        raise ValueError("--memory-dropout must be in [0, 1)")
    if args.counterfactual_memory_weight < 0.0:
        raise ValueError("--counterfactual-memory-weight must be non-negative")
    if args.counterfactual_change_weight < 0.0:
        raise ValueError("--counterfactual-change-weight must be non-negative")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metadata = load_metadata(args.metadata)
    files = available_files(args, metadata)
    validation_speakers = set(args.validation_speakers)
    validation_files = [
        name for name in files if name.split("_")[0] in validation_speakers
    ]
    validation_set = set(validation_files)
    training_files = [name for name in files if name not in validation_set]
    if not training_files or not validation_files:
        raise RuntimeError(
            f"empty split: training={len(training_files)}, "
            f"validation={len(validation_files)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    context_cache = args.context_cache or args.out_dir / "memory_contexts.pt"
    context_cache.parent.mkdir(parents=True, exist_ok=True)
    if context_cache.exists():
        context_payload = torch.load(
            context_cache, map_location="cpu", weights_only=True
        )
        training_contexts = context_payload["training"]
        validation_contexts = context_payload["validation"]
    else:
        training_contexts = build_contexts(
            args.noisy_dir, args.source_dir, metadata, training_files,
            args.noise_frame_fraction, args.memory_warmup,
            args.memory_prototypes, args.novelty_threshold,
            "build training memory",
        )
        validation_contexts = build_contexts(
            args.noisy_dir, args.source_dir, metadata, validation_files,
            args.noise_frame_fraction, args.memory_warmup,
            args.memory_prototypes, args.novelty_threshold,
            "build validation memory",
        )
        torch.save(
            {"training": training_contexts, "validation": validation_contexts},
            context_cache,
        )

    recovery_config = json.loads(
        (args.recovery_checkpoint.parent / "config.json").read_text(encoding="utf-8")
    )
    recovery = UtteranceLocalRecovery(
        channels=int(recovery_config["channels"]),
        max_gain=float(recovery_config["max_gain"]),
        initial_gain=float(recovery_config["initial_gain"]),
    )
    recovery.load_state_dict(
        torch.load(args.recovery_checkpoint, map_location="cpu", weights_only=True)
    )
    model = MemoryConditionedRefinement(
        channels=int(recovery_config["channels"]),
        max_gain=float(recovery_config["max_gain"]),
        initial_gain=float(recovery_config["initial_gain"]),
        memory_warmup=args.memory_warmup,
        prototypes=args.memory_prototypes,
        posterior_temperature=args.posterior_temperature,
        reliability_power=args.reliability_power,
        posterior_confidence_power=args.posterior_confidence_power,
        readout_mass=None,
        delta_rank=args.delta_rank,
        memory_delta_gain=args.memory_delta_gain,
    )
    model.initialize_from_static(recovery)
    model.requires_grad_(False)
    model.delta_basis.requires_grad_(True)
    model.delta_coordinates.requires_grad_(True)
    model.to(device)

    pesq_loss_fn = PesqLoss(0.5, sample_rate=16_000).to(device)
    segment_length = int(16_000 * args.segment_seconds)
    training_set = ContinualResidualDataset(
        args.clean_dir, args.noisy_dir, args.source_dir, training_files,
        training_contexts, segment_length, True,
    )
    validation_set = ContinualResidualDataset(
        args.clean_dir, args.noisy_dir, args.source_dir, validation_files,
        validation_contexts, segment_length, False,
    )
    training_loader = DataLoader(
        training_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    serializable = {
        **vars(args),
        "clean_dir": str(args.clean_dir),
        "noisy_dir": str(args.noisy_dir),
        "source_dir": str(args.source_dir),
        "metadata": str(args.metadata),
        "recovery_checkpoint": str(args.recovery_checkpoint),
        "out_dir": str(args.out_dir),
        "context_cache": str(context_cache),
        "readout_mass": None,
        "channels": int(recovery_config["channels"]),
        "max_gain": float(recovery_config["max_gain"]),
        "initial_gain": float(recovery_config["initial_gain"]),
        "training_files": len(training_files),
        "validation_files": len(validation_files),
        "trainable_parameters": sum(p.numel() for p in trainable_parameters),
    }
    (args.out_dir / "config.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )

    history = []
    best_validation = float("inf")
    for epoch in range(args.epochs):
        training_stats = run_epoch(
            model, training_loader, optimizer, device, args.memory_dropout,
            pesq_loss_fn, args.torch_pesq_loss_weight, args.sisdr_loss_weight,
            args.segmental_snr_loss_weight, args.counterfactual_memory_weight,
            args.counterfactual_change_weight,
        )
        validation_stats = run_epoch(
            model, validation_loader, None, device, 0.0, pesq_loss_fn,
            args.torch_pesq_loss_weight, args.sisdr_loss_weight,
            args.segmental_snr_loss_weight, 0.0,
            args.counterfactual_change_weight,
        )
        scheduler.step()
        selection_value = validation_stats["loss"]
        row = {
            "epoch": epoch + 1,
            **{f"train_{key}": value for key, value in training_stats.items()},
            **{f"validation_{key}": value for key, value in validation_stats.items()},
            "validation_selection_score": selection_value,
        }
        history.append(row)
        print(json.dumps(row))
        torch.save(model.state_dict(), args.out_dir / "last.th")
        if selection_value < best_validation:
            best_validation = selection_value
            torch.save(model.state_dict(), args.out_dir / "best.th")
        pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)


if __name__ == "__main__":
    main()
