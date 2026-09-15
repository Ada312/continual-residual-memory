#!/usr/bin/env python3
"""CMGAN main comparison on the paper's DNS, EARS-D, and Libri-MUSAN streams."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {"dns": "dns.csv", "ears_d": "ears_d.csv", "libri_musan": "musan_music.csv"}


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=MANIFESTS, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--cmgan-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint", type=Path, default=ROOT / "checkpoints/crm/static_best.th")
    parser.add_argument("--refinement-checkpoint", type=Path, default=ROOT / "checkpoints/crm/dynamic_best.th")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "manifests" / MANIFESTS[args.dataset]
    source = args.output_dir / "backbone/wav"
    run([sys.executable, "backbones/cmgan/inference.py", "--noisy-dir", str(args.noisy_dir),
         "--output-dir", str(source), "--checkpoint", str(args.cmgan_checkpoint),
         "--sample-rate", "16000", "--overwrite"], args.dry_run)
    for mode, checkpoint, subdir in (
        ("no_memory", args.recovery_checkpoint, "crm_no_memory"),
        ("crm", args.refinement_checkpoint, "crm"),
    ):
        run([sys.executable, "scripts/infer_stream.py", "--mode", mode,
             "--manifest", str(manifest), "--noisy-dir", str(args.noisy_dir),
             "--source-dir", str(source), "--checkpoint", str(checkpoint),
             "--output-dir", str(args.output_dir / subdir / "wav"),
             "--device", args.device, "--overwrite"], args.dry_run)
    for subdir in ("backbone", "crm_no_memory", "crm"):
        run([sys.executable, "metrics/evaluate.py", "--clean-dir", str(args.clean_dir),
             "--noisy-dir", str(args.noisy_dir),
             "--denoised-dir", str(args.output_dir / subdir / "wav"),
             "--out-dir", str(args.output_dir / subdir / "metrics"),
             "--method", subdir, "--references", str(manifest),
             "--fs", "16000", "--workers", str(args.workers)], args.dry_run)


if __name__ == "__main__":
    main()
