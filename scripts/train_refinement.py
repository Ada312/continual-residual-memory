#!/usr/bin/env python3
"""Stage 2: train memory-conditioned refinement; recovery stays frozen."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cmgan/refinement_training.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (
        config["tau_cf"] != 0.02
        or config["readout_mass"] is not None
        or config["context_stream_mode"] != "grouped"
        or config["context_grouping"] != "wham_location"
    ):
        raise ValueError(
            "paper training requires tau_cf=0.02, grouped WHAM-location "
            "histories, and no readout-mass truncation"
        )
    if not args.dry_run and not (args.recovery_checkpoint.parent / "config.json").exists():
        raise FileNotFoundError("recovery checkpoint directory must contain config.json")
    command = [
        sys.executable, "-m", "crm._training.refinement",
        "--clean-dir", str(args.clean_dir), "--noisy-dir", str(args.noisy_dir),
        "--source-dir", str(args.source_dir), "--metadata", str(args.metadata),
        "--recovery-checkpoint", str(args.recovery_checkpoint),
        "--out-dir", str(args.output_dir), "--context-cache", str(args.context_cache),
        "--device", args.device,
    ]
    for key in (
        "epochs", "batch_size", "num_workers", "segment_seconds", "learning_rate",
        "weight_decay", "memory_dropout", "counterfactual_memory_weight",
        "counterfactual_change_weight", "noise_frame_fraction", "memory_warmup",
        "novelty_threshold", "reliability_power", "posterior_confidence_power",
        "torch_pesq_loss_weight", "sisdr_loss_weight", "segmental_snr_loss_weight",
        "seed",
    ):
        command.extend(("--" + key.replace("_", "-"), str(config[key])))
    for key, option in (("k_train", "--memory-prototypes"), ("tau_mem", "--posterior-temperature"),
                        ("d", "--delta-rank"), ("alpha_dyn", "--memory-delta-gain")):
        command.extend((option, str(config[key])))
    command.extend(("--validation-speakers", *config["validation_speakers"]))
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
