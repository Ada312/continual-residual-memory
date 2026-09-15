#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
FASTENHANCER_ROOT = Path(
    os.environ.get("FASTENHANCER_REPO", ROOT.parent / "fastenhancer")
)
if str(FASTENHANCER_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTENHANCER_ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the official FastEnhancer checkpoint."
    )
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speakers", nargs="*", default=[])
    parser.add_argument("--total-limit", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path)
    audio = audio.mean(dim=0).float()
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def main() -> None:
    args = parse_args()
    from utils import get_hparams
    from wrappers import get_wrapper
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    hparams = get_hparams(config_dir=str(args.config), base_dir=str(args.config.parent))
    wrapper = get_wrapper(hparams.wrapper)(hparams, device=device)
    wrapper.load(path=str(args.checkpoint))
    model = wrapper.model.eval().requires_grad_(False)
    sample_rate = int(wrapper.sr)

    speakers = set(args.speakers)
    files = sorted(args.noisy_dir.glob("*.wav"))
    if speakers:
        files = [path for path in files if path.stem.split("_")[0] in speakers]
    if args.total_limit is not None:
        files = files[: args.total_limit]
    files = files[args.shard_index :: args.num_shards]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError("no input wav files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for noisy_path in tqdm(files, desc=f"FastEnhancer-B shard {args.shard_index}"):
            output_path = args.output_dir / noisy_path.name
            if output_path.exists() and not args.overwrite:
                continue
            noisy = load_mono(noisy_path, sample_rate).to(device)
            original_length = noisy.numel()
            if original_length < 513:
                noisy = torch.nn.functional.pad(noisy, (0, 513 - original_length))
            estimate, _ = model(noisy.unsqueeze(0))
            estimate = estimate.squeeze(0)[:original_length]
            torchaudio.save(
                output_path,
                estimate.clamp(-1.0, 1.0).cpu().unsqueeze(0),
                sample_rate,
            )


if __name__ == "__main__":
    main()
