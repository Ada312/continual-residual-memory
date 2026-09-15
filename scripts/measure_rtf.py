#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from backbones.cmgan.transform_util import get_transforms
from backbones.cmgan import model as _cmgan_registration  # noqa: F401 - registry side effect
from backbones.registry import ModelRegistry
from crm.memory import DynamicCausalPrototypeResidualNoiseMemory
from crm.refinement import LowRankResidualDeltaPrototypeContinualResidualSpeechProjector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CMGAN and CMGAN+CRM RTF.")
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--cmgan-checkpoint", type=Path, required=True)
    parser.add_argument("--crm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-files", type=int, default=10)
    parser.add_argument("--memory-prototypes", type=int, default=64)
    parser.add_argument("--readout-mass", type=float, default=0.95)
    parser.add_argument("--candidate-threshold", type=float, default=0.7)
    parser.add_argument("--immediate-novelty-threshold", type=float, default=0.7)
    parser.add_argument("--novelty-patience", type=int, default=2)
    parser.add_argument("--merge-threshold", type=float, default=0.05)
    parser.add_argument("--retirement-horizon", type=int, default=20)
    parser.add_argument("--stream-order-manifest", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def build_transform(device: torch.device):
    config = OmegaConf.create(
        {
            "fs": 16_000,
            "transforms": ["power_norm", "TF=COMP"],
            "power_compress": 0.3,
            "stft": {"fft_length": 400, "hop_length": 100, "window": "hamming"},
        }
    )
    return get_transforms(config).to(device)


def build_cmgan(checkpoint: Path, device: torch.device):
    config = OmegaConf.create({"gen_depth": 4, "fs": 16_000, "load": None})
    model = ModelRegistry()["CMGAN", config]
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)


def build_crm(checkpoint: Path, device: torch.device, args: argparse.Namespace):
    config_path = checkpoint.parent / "config.json"
    if not config_path.exists() and checkpoint.stem.endswith("_best"):
        config_path = checkpoint.with_name(
            checkpoint.stem.removesuffix("_best") + "_config.json"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = LowRankResidualDeltaPrototypeContinualResidualSpeechProjector(
        channels=int(config["channels"]),
        max_gain=float(config["max_gain"]),
        initial_gain=float(config["initial_gain"]),
        memory_warmup=int(config["memory_warmup"]),
        prototypes=args.memory_prototypes,
        posterior_temperature=float(config["posterior_temperature"]),
        reliability_power=float(config.get("reliability_power", 0.0)),
        posterior_confidence_power=float(config.get("posterior_confidence_power", 0.0)),
        readout_mass=args.readout_mass,
        delta_rank=int(config.get("delta_rank", 4)),
        memory_delta_gain=float(config.get("memory_delta_gain", 0.02)),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    memory = DynamicCausalPrototypeResidualNoiseMemory(
        prototypes=args.memory_prototypes,
        noise_frame_fraction=float(config["noise_frame_fraction"]),
        warmup_utterances=int(config["memory_warmup"]),
        novelty_threshold=float(config["novelty_threshold"]),
        novelty_patience=args.novelty_patience,
        candidate_threshold=args.candidate_threshold,
        immediate_novelty_threshold=args.immediate_novelty_threshold,
        merge_threshold=args.merge_threshold,
        retirement_horizon=args.retirement_horizon,
    )
    return model.to(device).eval().requires_grad_(False), memory


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_files(noisy_dir: Path, manifest: Path | None) -> list[Path]:
    if manifest is None:
        return sorted(noisy_dir.glob("*.wav"))
    import pandas as pd

    frame = pd.read_csv(manifest).sort_values("test_order")
    if frame["filename"].duplicated().any() or frame["test_order"].duplicated().any():
        raise ValueError("stream order manifest contains duplicate entries")
    files = [noisy_dir / str(name) for name in frame["filename"]]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"manifest references missing files: {missing[:5]}")
    return files


def load_mono(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(path)
    audio = audio.mean(dim=0, keepdim=True).float()
    if sample_rate != 16_000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16_000)
    return audio


def cmgan_forward(model, transform, noisy: torch.Tensor) -> torch.Tensor:
    features, reconstruction = transform(noisy)
    estimate_features = model.evaluate(features.unsqueeze(0))
    estimate = transform.reconstruct(estimate_features, reconstruction)
    return estimate[..., : noisy.shape[-1]]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the RTF benchmark")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cmgan = build_cmgan(args.cmgan_checkpoint, device)
    transform = build_transform(device)
    crm, memory = build_crm(args.crm_checkpoint, device, args)
    files = ordered_files(args.noisy_dir, args.stream_order_manifest)
    if len(files) <= args.warmup_files:
        raise ValueError("the evaluation set must be larger than the warm-up prefix")

    trial_rows = []
    with torch.inference_mode():
        for repeat in range(args.repeats):
            memory.reset()
            source_seconds = 0.0
            dynamic_seconds = 0.0
            audio_seconds = 0.0
            for index, path in enumerate(tqdm(files, desc=f"CMGAN RTF repeat {repeat + 1}")):
                noisy = load_mono(path).to(device)

                torch.cuda.synchronize(device)
                start = time.perf_counter()
                _ = cmgan_forward(cmgan, transform, noisy)
                torch.cuda.synchronize(device)
                source_elapsed = time.perf_counter() - start

                torch.cuda.synchronize(device)
                start = time.perf_counter()
                source = cmgan_forward(cmgan, transform, noisy)
                context = memory.context()
                estimate, _, auxiliary = crm(
                    noisy,
                    source,
                    context.means.unsqueeze(0),
                    context.variances.unsqueeze(0),
                    context.counts.to(torch.float32).unsqueeze(0),
                )
                memory.update(
                    auxiliary["source_spectrum"][0], auxiliary["residual_spectrum"][0]
                )
                _ = estimate
                torch.cuda.synchronize(device)
                dynamic_elapsed = time.perf_counter() - start

                if index >= args.warmup_files:
                    source_seconds += source_elapsed
                    dynamic_seconds += dynamic_elapsed
                    audio_seconds += noisy.shape[-1] / 16_000.0
            trial_rows.append({
                "repeat": repeat,
                "audio_seconds": audio_seconds,
                "source_seconds": source_seconds,
                "dynamic_seconds": dynamic_seconds,
                "source_rtf": source_seconds / audio_seconds,
                "dynamic_rtf": dynamic_seconds / audio_seconds,
            })

    source_rtf = sum(row["source_rtf"] for row in trial_rows) / len(trial_rows)
    dynamic_rtf = sum(row["dynamic_rtf"] for row in trial_rows) / len(trial_rows)
    source_seconds = sum(row["source_seconds"] for row in trial_rows) / len(trial_rows)
    dynamic_seconds = sum(row["dynamic_seconds"] for row in trial_rows) / len(trial_rows)
    audio_seconds = trial_rows[0]["audio_seconds"]
    report = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "float32",
        "batch_size": 1,
        "timing_scope": "waveform-to-waveform, including transforms and causal memory update; disk I/O excluded",
        "files_total": len(files),
        "warmup_files": args.warmup_files,
        "repeats": args.repeats,
        "seed": args.seed,
        "memory_prototypes": args.memory_prototypes,
        "context_prototypes": args.memory_prototypes,
        "readout_mass": args.readout_mass,
        "candidate_threshold": args.candidate_threshold,
        "immediate_novelty_threshold": args.immediate_novelty_threshold,
        "novelty_patience": args.novelty_patience,
        "merge_threshold": args.merge_threshold,
        "retirement_horizon": args.retirement_horizon,
        "stream_order_manifest": str(args.stream_order_manifest) if args.stream_order_manifest else None,
        "stream_order_manifest_sha256": sha256(args.stream_order_manifest) if args.stream_order_manifest else None,
        "cmgan_checkpoint": str(args.cmgan_checkpoint),
        "cmgan_checkpoint_sha256": sha256(args.cmgan_checkpoint),
        "crm_checkpoint": str(args.crm_checkpoint),
        "crm_checkpoint_sha256": sha256(args.crm_checkpoint),
        "benchmark_script_sha256": sha256(Path(__file__).resolve()),
        "files_timed": len(files) - args.warmup_files,
        "audio_seconds": audio_seconds,
        "source_seconds": source_seconds,
        "dynamic_seconds": dynamic_seconds,
        "source_rtf": source_rtf,
        "dynamic_rtf": dynamic_rtf,
        "absolute_rtf_overhead": dynamic_rtf - source_rtf,
        "relative_time_overhead_percent": 100.0
        * (dynamic_seconds / source_seconds - 1.0),
        "trials": trial_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
