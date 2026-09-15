#!/usr/bin/env python3
"""Stage 1: train utterance-local residual recovery on EARS-WHAM."""

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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cmgan/recovery_training.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    command = [
        sys.executable, "-m", "crm._training.recovery",
        "--clean-dir", str(args.clean_dir), "--noisy-dir", str(args.noisy_dir),
        "--source-dir", str(args.source_dir), "--out-dir", str(args.output_dir),
    ]
    for key in (
        "epochs", "batch_size", "num_workers", "segment_seconds", "learning_rate",
        "weight_decay",
        "channels", "max_gain", "initial_gain", "validation_interval", "seed",
        "time_loss_weight", "complex_loss_weight", "logmag_loss_weight",
        "projection_loss_weight", "sisdr_loss_weight", "segmental_snr_loss_weight",
        "trust_loss_weight",
    ):
        command.extend(("--" + key.replace("_", "-"), str(config[key])))
    command.extend(("--validation-speakers", *config["validation_speakers"]))
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
