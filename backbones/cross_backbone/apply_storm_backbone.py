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
STORM_ROOT = Path(os.environ.get("STORM_REPO", ROOT / ".deps" / "storm"))
if str(STORM_ROOT) not in sys.path:
    sys.path.insert(0, str(STORM_ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the official StoRM WSJ0+CHiME3 checkpoint."
    )
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--corrector", choices=("ald", "langevin", "none"), default="ald")
    parser.add_argument("--corrector-steps", type=int, default=1)
    parser.add_argument("--snr", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--speakers", nargs="*", default=[])
    parser.add_argument("--total-limit", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_mono(path: Path, sample_rate: int = 16_000) -> torch.Tensor:
    audio, source_rate = torchaudio.load(path)
    audio = audio.mean(dim=0, keepdim=True).float()
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def file_seed(base_seed: int, path: Path) -> int:
    digest = hashlib.sha256(path.name.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "little")) % (2**31)


def main() -> None:
    args = parse_args()
    from sgmse.model import StochasticRegenerationModel
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.corrector_steps < 1:
        raise ValueError("--corrector-steps must be positive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = StochasticRegenerationModel.load_from_checkpoint(
        str(args.checkpoint),
        base_dir="",
        batch_size=1,
        num_workers=0,
        kwargs={"gpu": False},
        map_location="cpu",
    )
    model.eval(no_ema=False).to(device).requires_grad_(False)

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
        for noisy_path in tqdm(files, desc=f"StoRM shard {args.shard_index}"):
            output_path = args.output_dir / noisy_path.name
            if output_path.exists() and not args.overwrite:
                continue

            torch.manual_seed(file_seed(args.seed, noisy_path))
            noisy = load_mono(noisy_path).to(device)
            estimate = model.enhance(
                noisy,
                corrector=args.corrector,
                N=args.steps,
                corrector_steps=args.corrector_steps,
                snr=args.snr,
            )
            torchaudio.save(
                output_path,
                estimate.clamp(-1.0, 1.0).cpu().reshape(1, -1),
                16_000,
            )


if __name__ == "__main__":
    main()
