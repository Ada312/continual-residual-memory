#!/usr/bin/env python3
"""Ordered causal CRM inference: read, infer, save output, write observation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torchaudio

from crm.model import CRM, load_recovery, load_refinement, new_memory, read_config


def ordered_ids(manifest: Path) -> list[str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda row: int(row["test_order"]))
    ids = [row["filename"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("manifest must contain unique utterance filenames")
    if [int(row["test_order"]) for row in rows] != list(range(len(rows))):
        raise ValueError("manifest test_order must be contiguous and zero-based")
    return ids


def load_audio(path: Path) -> torch.Tensor:
    audio, sample_rate = torchaudio.load(path)
    audio = audio.mean(dim=0).float()
    if sample_rate != 16_000:
        audio = torchaudio.functional.resample(audio, sample_rate, 16_000)
    return audio


def save_memory(path: Path, crm: CRM) -> None:
    memory = crm.memory
    state = {key: getattr(memory, key).clone() for key in
             ("means", "variances", "counts", "usage", "last_used")}
    state.update(candidate=memory.candidate.clone() if memory.candidate is not None else None,
                 candidate_count=int(memory.candidate_count), step=int(memory.step))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "crm_dynamic_memory_v1",
                "prototypes": memory.prototypes, "frequency_bins": memory.frequency_bins,
                "noise_frame_fraction": memory.noise_frame_fraction,
                "warmup_utterances": memory.warmup_utterances,
                "novelty_threshold": memory.novelty_threshold,
                "novelty_patience": memory.novelty_patience,
                "candidate_threshold": memory.candidate_threshold,
                "immediate_novelty_threshold": memory.immediate_novelty_threshold,
                "merge_threshold": memory.merge_threshold,
                "retirement_horizon": memory.retirement_horizon,
                "processed_utterances": memory.step, "state": state}, path)


def load_memory(path: Path, crm: CRM) -> None:
    memory = crm.memory
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"format": "crm_dynamic_memory_v1",
                "prototypes": memory.prototypes, "frequency_bins": memory.frequency_bins,
                "noise_frame_fraction": memory.noise_frame_fraction,
                "warmup_utterances": memory.warmup_utterances,
                "novelty_threshold": memory.novelty_threshold,
                "novelty_patience": memory.novelty_patience,
                "candidate_threshold": memory.candidate_threshold,
                "immediate_novelty_threshold": memory.immediate_novelty_threshold,
                "merge_threshold": memory.merge_threshold,
                "retirement_horizon": memory.retirement_horizon}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"incompatible memory state: {key}")
    state = payload["state"]
    for key in ("means", "variances", "counts", "usage", "last_used"):
        if state[key].shape != getattr(memory, key).shape or state[key].dtype != getattr(memory, key).dtype:
            raise ValueError(f"incompatible memory tensor: {key}")
        getattr(memory, key).copy_(state[key])
    candidate = state["candidate"]
    if candidate is not None and (candidate.shape != (memory.frequency_bins,) or
                                  candidate.dtype != memory.means.dtype):
        raise ValueError("incompatible memory candidate")
    memory.candidate = candidate.clone() if candidate is not None else None
    memory.candidate_count = int(state["candidate_count"])
    memory.step = int(state["step"])
    if memory.step != payload["processed_utterances"] or memory.step < 1 or int(memory.counts.sum()) < 1:
        raise ValueError("invalid memory step/count")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("no_memory", "crm"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cmgan/final.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-memory-state", type=Path)
    parser.add_argument("--save-memory-state", type=Path)
    parser.add_argument("--limit", type=int, help="Smoke test only; never use in a formal run")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.mode == "no_memory" and (args.initial_memory_state or args.save_memory_state):
        parser.error("no-memory recovery cannot read or write prototype state")
    config = read_config(args.config)
    torch.manual_seed(int(config["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ids = ordered_ids(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        ids = ids[:args.limit]
    for filename in ids:
        if not (args.noisy_dir / filename).is_file() or not (args.source_dir / filename).is_file():
            raise FileNotFoundError(f"missing noisy/Source waveform for {filename}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "crm":
        crm = CRM(load_refinement(args.checkpoint, config, device), new_memory(config))
        if args.initial_memory_state:
            load_memory(args.initial_memory_state, crm)
    else:
        projector = load_recovery(args.checkpoint, config, device)
    for filename in ids:
        output = args.output_dir / filename
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite for a complete ordered rerun: {output}")
        noisy_waveform = load_audio(args.noisy_dir / filename)
        backbone_estimate = load_audio(args.source_dir / filename)
        length = min(noisy_waveform.numel(), backbone_estimate.numel())
        noisy_waveform = noisy_waveform[:length].to(device)
        backbone_estimate = backbone_estimate[:length].to(device)
        with torch.inference_mode():
            if args.mode == "crm":
                enhanced, _, auxiliary = crm.infer(noisy_waveform, backbone_estimate)
            else:
                enhanced, _, _ = projector(noisy_waveform[None], backbone_estimate[None])
            torchaudio.save(output, enhanced.clamp(-1.0, 1.0).cpu(), 16_000)
            if args.mode == "crm":
                crm.write(auxiliary)
    if args.mode == "crm" and args.save_memory_state:
        save_memory(args.save_memory_state, crm)
    print(json.dumps({"mode": args.mode, "processed": len(ids), "memory_steps":
                      crm.memory.step if args.mode == "crm" else 0}))


if __name__ == "__main__":
    main()
