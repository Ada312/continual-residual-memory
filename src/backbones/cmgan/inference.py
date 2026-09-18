#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

from backbones.cmgan.transform_util import get_transforms
from backbones.cmgan import model as _cmgan_registration  # noqa: F401 - registry side effect
from backbones.registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache outputs from the frozen original CMGAN checkpoint."
    )
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_transform(sample_rate: int):
    cfg = OmegaConf.create(
        {
            "fs": sample_rate,
            "transforms": ["power_norm", "TF=COMP"],
            "power_compress": 0.3,
            "stft": {"fft_length": 400, "hop_length": 100, "window": "hamming"},
        }
    )
    return get_transforms(cfg)


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path)
    audio = audio.mean(dim=0, keepdim=True).float()
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model_cfg = OmegaConf.create(
        {"gen_depth": 4, "fs": args.sample_rate, "load": None}
    )
    model = ModelRegistry()["CMGAN", model_cfg]
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval().requires_grad_(False)
    transform = build_transform(args.sample_rate).to(device)

    files = sorted(args.noisy_dir.glob("*.wav"))
    files = files[local_rank::world_size]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for path in tqdm(files, desc=f"source rank {local_rank}", position=local_rank):
            output_path = args.output_dir / path.name
            if output_path.exists() and not args.overwrite:
                continue
            noisy = load_audio(path, args.sample_rate).to(device)
            features, reconstruction = transform(noisy)
            estimate_features = model.evaluate(features.unsqueeze(0))
            estimate = transform.reconstruct(estimate_features, reconstruction)
            estimate = estimate[..., : noisy.shape[-1]].clamp(-1.0, 1.0)
            torchaudio.save(output_path, estimate.cpu(), args.sample_rate)

    if local_rank == 0:
        print(
            f"checkpoint={args.checkpoint} world_size={world_size} "
            f"total_files={len(list(args.noisy_dir.glob('*.wav')))}"
        )


if __name__ == "__main__":
    main()
