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
FLOWSE_ROOT = Path(os.environ.get("FLOWSE_REPO", ROOT / ".deps" / "flowmse"))
if str(FLOWSE_ROOT) not in sys.path:
    sys.path.insert(0, str(FLOWSE_ROOT))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the official ICASSP 2025 FlowSE checkpoint."
    )
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nfe", type=int, default=5)
    parser.add_argument("--reverse-start", type=float, default=1.0)
    parser.add_argument("--reverse-end", type=float, default=0.03)
    parser.add_argument("--speakers", nargs="*", default=[])
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


def main() -> None:
    args = parse_args()
    from flowmse.model import VFModel
    from flowmse.sampling import get_white_box_solver
    from flowmse.util.other import pad_spec
    if args.nfe < 1:
        raise ValueError("--nfe must be positive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = VFModel.load_from_checkpoint(
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
    files = files[args.shard_index :: args.num_shards]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise RuntimeError("no input wav files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for noisy_path in tqdm(files, desc="FlowSE"):
            output_path = args.output_dir / noisy_path.name
            if output_path.exists() and not args.overwrite:
                continue
            noisy = load_mono(noisy_path).to(device)
            length = noisy.shape[-1]
            norm = noisy.abs().max().clamp_min(1e-7)
            normalized = noisy / norm
            condition = model._forward_transform(model._stft(normalized))
            condition = pad_spec(condition.unsqueeze(0))
            sampler = get_white_box_solver(
                "euler",
                model.ode,
                model,
                Y=condition,
                Y_prior=condition,
                T_rev=args.reverse_start,
                t_eps=args.reverse_end,
                N=args.nfe,
            )
            sample, _ = sampler()
            estimate = model.to_audio(sample.squeeze(), length).squeeze() * norm
            torchaudio.save(
                output_path,
                estimate.clamp(-1.0, 1.0).cpu().unsqueeze(0),
                16_000,
            )


if __name__ == "__main__":
    main()
