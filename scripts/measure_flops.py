#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.flop_counter import FlopCounterMode, flop_registry
from tqdm import tqdm

from scripts.measure_rtf import (
    build_cmgan,
    build_crm,
    build_transform,
    cmgan_forward,
    load_mono,
    ordered_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CMGAN and CMGAN+CRM counted FLOPs under the frozen RTF protocol."
    )
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--cmgan-checkpoint", type=Path, required=True)
    parser.add_argument("--crm-checkpoint", type=Path, required=True)
    parser.add_argument("--rtf-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup-files", type=int, default=10)
    parser.add_argument("--memory-prototypes", type=int, default=64)
    parser.add_argument("--readout-mass", type=float, default=0.95)
    parser.add_argument("--candidate-threshold", type=float, default=0.7)
    parser.add_argument("--immediate-novelty-threshold", type=float, default=0.7)
    parser.add_argument("--novelty-patience", type=int, default=2)
    parser.add_argument("--merge-threshold", type=float, default=0.05)
    parser.add_argument("--retirement-horizon", type=int, default=20)
    parser.add_argument("--stream-order-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def serialize_counts(counts: dict[object, int]) -> dict[str, int]:
    return {str(operation): int(value) for operation, value in counts.items()}


def add_counts(total: dict[str, int], current: dict[str, int]) -> None:
    for operation, value in current.items():
        total[operation] += value


def counted_call(modules, function):
    counter = FlopCounterMode(mods=modules, display=False)
    with counter:
        result = function()
    counts = serialize_counts(counter.get_flop_counts()["Global"])
    return result, int(counter.get_total_flops()), counts


def verify_rtf_protocol(args: argparse.Namespace, files: list[Path]) -> dict:
    rtf = json.loads(args.rtf_artifact.read_text(encoding="utf-8"))
    expected = {
        "precision": "float32",
        "batch_size": 1,
        "files_total": len(files),
        "warmup_files": args.warmup_files,
        "seed": args.seed,
        "memory_prototypes": args.memory_prototypes,
        "context_prototypes": args.memory_prototypes,
        "readout_mass": args.readout_mass,
        "candidate_threshold": args.candidate_threshold,
        "immediate_novelty_threshold": args.immediate_novelty_threshold,
        "novelty_patience": args.novelty_patience,
        "merge_threshold": args.merge_threshold,
        "retirement_horizon": args.retirement_horizon,
    }
    mismatches = {
        key: {"rtf": rtf.get(key), "flops": value}
        for key, value in expected.items()
        if rtf.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"FLOPs/RTF protocol mismatch: {mismatches}")
    return rtf


def main() -> None:
    args = parse_args()
    if args.warmup_files < 0:
        raise ValueError("--warmup-files must be non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    files = ordered_files(args.noisy_dir, args.stream_order_manifest)
    if len(files) <= args.warmup_files:
        raise ValueError("the evaluation set must be larger than the warm-up prefix")
    rtf = verify_rtf_protocol(args, files)

    cmgan = build_cmgan(args.cmgan_checkpoint, device)
    transform = build_transform(device)
    crm, memory = build_crm(args.crm_checkpoint, device, args)

    rows = []
    source_operation_totals: dict[str, int] = defaultdict(int)
    dynamic_operation_totals: dict[str, int] = defaultdict(int)
    memory.reset()

    with torch.inference_mode():
        for index, path in enumerate(tqdm(files, desc="CMGAN/CRM FLOPs")):
            noisy = load_mono(path).to(device)
            audio_seconds = noisy.shape[-1] / 16_000.0

            if index < args.warmup_files:
                _ = cmgan_forward(cmgan, transform, noisy)
                source = cmgan_forward(cmgan, transform, noisy)
                context = memory.context()
                _, _, auxiliary = crm(
                    noisy,
                    source,
                    context.means.unsqueeze(0),
                    context.variances.unsqueeze(0),
                    context.counts.to(torch.float32).unsqueeze(0),
                )
                memory.update(
                    auxiliary["source_spectrum"][0],
                    auxiliary["residual_spectrum"][0],
                )
                continue

            _, source_flops, source_counts = counted_call(
                cmgan, lambda: cmgan_forward(cmgan, transform, noisy)
            )

            def dynamic_forward():
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
                    auxiliary["source_spectrum"][0],
                    auxiliary["residual_spectrum"][0],
                )
                return estimate

            _, dynamic_flops, dynamic_counts = counted_call(
                [cmgan, crm], dynamic_forward
            )
            add_counts(source_operation_totals, source_counts)
            add_counts(dynamic_operation_totals, dynamic_counts)
            rows.append(
                {
                    "test_order": index,
                    "filename": path.name,
                    "audio_seconds": audio_seconds,
                    "source_counted_flops": source_flops,
                    "dynamic_counted_flops": dynamic_flops,
                    "absolute_counted_flops_overhead": dynamic_flops - source_flops,
                    "source_gflops_per_audio_second": source_flops / audio_seconds / 1e9,
                    "dynamic_gflops_per_audio_second": dynamic_flops / audio_seconds / 1e9,
                }
            )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_utterance_path = output_dir / "cmgan_vs_crm_k64_per_utterance.csv"
    with per_utterance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audio_seconds = sum(float(row["audio_seconds"]) for row in rows)
    source_flops = sum(int(row["source_counted_flops"]) for row in rows)
    dynamic_flops = sum(int(row["dynamic_counted_flops"]) for row in rows)
    report = {
        "status": "VERIFIED AGAINST FROZEN RTF PROTOCOL",
        "measurement_class": "PROFILER-COUNTED FLOPs",
        "profiler": "torch.utils.flop_counter.FlopCounterMode",
        "flop_convention": "one multiply-add is two FLOPs",
        "coverage": (
            "PyTorch native registered formulas; observed counted operators are listed "
            "in operator_breakdown"
        ),
        "not_counted": [
            "STFT/iSTFT FFT arithmetic",
            "elementwise activation, normalization, arithmetic, and complex operations",
            "top-k, sorting, median, indexing, comparisons, and prototype control flow",
            "memory copies and CPU/GPU transfers",
        ],
        "interpretation": (
            "Use as a reproducible lower-bound count for registered tensor operators, "
            "not as complete end-to-end hardware work. RTF remains the end-to-end runtime measure."
        ),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "float32",
        "batch_size": 1,
        "execution_scope": rtf["timing_scope"],
        "files_total": len(files),
        "warmup_files": args.warmup_files,
        "files_profiled": len(rows),
        "audio_seconds_profiled": audio_seconds,
        "seed": args.seed,
        "memory_prototypes": args.memory_prototypes,
        "context_prototypes": args.memory_prototypes,
        "readout_mass": args.readout_mass,
        "candidate_threshold": args.candidate_threshold,
        "immediate_novelty_threshold": args.immediate_novelty_threshold,
        "novelty_patience": args.novelty_patience,
        "merge_threshold": args.merge_threshold,
        "retirement_horizon": args.retirement_horizon,
        "source_counted_flops_total": source_flops,
        "dynamic_counted_flops_total": dynamic_flops,
        "absolute_counted_flops_overhead_total": dynamic_flops - source_flops,
        "relative_counted_flops_overhead_percent": 100.0
        * (dynamic_flops / source_flops - 1.0),
        "source_gflops_per_audio_second": source_flops / audio_seconds / 1e9,
        "dynamic_gflops_per_audio_second": dynamic_flops / audio_seconds / 1e9,
        "absolute_gflops_per_audio_second_overhead": (dynamic_flops - source_flops)
        / audio_seconds
        / 1e9,
        "source_mean_gflops_per_utterance": source_flops / len(rows) / 1e9,
        "dynamic_mean_gflops_per_utterance": dynamic_flops / len(rows) / 1e9,
        "operator_breakdown": {
            "source": dict(sorted(source_operation_totals.items())),
            "dynamic": dict(sorted(dynamic_operation_totals.items())),
        },
        "registered_formula_operators": sorted(str(item) for item in flop_registry),
        "stream_order_manifest": str(args.stream_order_manifest.resolve()),
        "cmgan_checkpoint": str(args.cmgan_checkpoint.resolve()),
        "crm_checkpoint": str(args.crm_checkpoint.resolve()),
        "rtf_artifact": str(args.rtf_artifact.resolve()),
        "per_utterance_csv": str(per_utterance_path),
        "command": shlex.join(sys.argv),
    }
    output_path = output_dir / "cmgan_vs_crm_k64_flops.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
