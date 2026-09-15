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
GTCRN_ROOT = Path(
    os.environ.get("GTCRN_REPO", ROOT.parent / "streaming_enhancers" / "gtcrn")
)
if str(GTCRN_ROOT) not in sys.path:
    sys.path.insert(0, str(GTCRN_ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the official GTCRN VCTK-DEMAND checkpoint."
    )
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speakers", nargs="*", default=[])
    parser.add_argument("--total-limit", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_mono(path: Path, sample_rate: int = 16_000) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path)
    audio = audio.mean(dim=0).float()
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def main() -> None:
    args = parse_args()
    from gtcrn import GTCRN
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GTCRN().to(device).eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"])
    window = torch.hann_window(512, device=device).sqrt()

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
        for noisy_path in tqdm(files, desc=f"GTCRN shard {args.shard_index}"):
            output_path = args.output_dir / noisy_path.name
            if output_path.exists() and not args.overwrite:
                continue
            noisy = load_mono(noisy_path).to(device)
            spectrum = torch.stft(
                noisy, 512, 256, 512, window, return_complex=False
            )
            enhanced_spectrum = model(spectrum.unsqueeze(0))[0]
            enhanced_spectrum = torch.view_as_complex(enhanced_spectrum.contiguous())
            estimate = torch.istft(
                enhanced_spectrum,
                512,
                256,
                512,
                window,
                length=noisy.numel(),
                return_complex=False,
            )
            torchaudio.save(
                output_path, estimate.clamp(-1.0, 1.0).cpu().unsqueeze(0), 16_000
            )


if __name__ == "__main__":
    main()
