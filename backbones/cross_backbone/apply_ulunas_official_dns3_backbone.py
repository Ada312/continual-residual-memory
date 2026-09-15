#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
ULUNAS_ROOT = Path(
    os.environ.get("UL_UNAS_REPO", ROOT.parent / "streaming_enhancers" / "ul-unas")
)
OFFICIAL_SHA256 = "9a0656bf8e88d36865792d3065fed9bc33d5ab85e11ae3a1ae085b928fdb49a8"
if str(ULUNAS_ROOT) not in sys.path:
    sys.path.insert(0, str(ULUNAS_ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the official DNS3-trained UL-UNAS checkpoint."
    )
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--total-limit", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_mono(path: Path, sample_rate: int = 16_000) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path)
    audio = audio.mean(dim=0).float()
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def main() -> None:
    args = parse_args()
    from ulunas import ULUNAS
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    actual_sha256 = sha256(args.checkpoint)
    if actual_sha256 != OFFICIAL_SHA256:
        raise RuntimeError(
            f"refusing non-official UL-UNAS checkpoint: {actual_sha256}"
        )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if set(checkpoint) != {"epoch", "optimizer", "scheduler", "model"}:
        raise RuntimeError(f"unexpected official checkpoint keys: {checkpoint.keys()}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = ULUNAS().to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    files = sorted(args.noisy_dir.glob("*.wav"))
    if args.total_limit is not None:
        files = files[: args.total_limit]
    files = files[args.shard_index :: args.num_shards]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError("no input WAV files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for noisy_path in tqdm(files, desc=f"UL-UNAS shard {args.shard_index}"):
            output_path = args.output_dir / noisy_path.name
            if output_path.exists() and not args.overwrite:
                continue
            noisy = load_mono(noisy_path).to(device)
            estimate = model(noisy.unsqueeze(0)).squeeze(0)
            torchaudio.save(
                output_path,
                estimate.clamp(-1.0, 1.0).cpu().unsqueeze(0),
                16_000,
            )


if __name__ == "__main__":
    main()
