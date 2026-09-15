#!/usr/bin/env python3
"""One frozen Source backbone and its independently trained CRM on DNS."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "storm50": ("apply_storm_backbone.py", "STORM_REPO", [
        "--steps", "50", "--corrector", "ald", "--corrector-steps", "1",
        "--snr", "0.5", "--seed", "1337"]),
    "gtcrn": ("apply_gtcrn_backbone.py", "GTCRN_REPO", []),
    "fastenhancer_b": ("apply_fastenhancer_backbone.py", "FASTENHANCER_REPO", []),
    "flowse": ("apply_flowse_backbone.py", "FLOWSE_REPO", [
        "--nfe", "5", "--reverse-start", "1.0", "--reverse-end", "0.03"]),
    "ul_unas": ("apply_ulunas_official_dns3_backbone.py", "UL_UNAS_REPO", []),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=WRAPPERS, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, help="Required by FastEnhancer-B")
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    script, env_name, settings = WRAPPERS[args.backbone]
    if args.backbone == "fastenhancer_b" and args.backbone_config is None:
        parser.error("--backbone-config is required for FastEnhancer-B")
    output = args.output_dir / args.backbone
    manifest = ROOT / "manifests/dns.csv"
    env = {**os.environ, env_name: str(args.upstream_repo)}

    def run(command: list[str]) -> None:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, env=env, check=True)

    source = output / "source/wav"
    source_cmd = [sys.executable, "backbones/cross_backbone/" + script,
                  "--noisy-dir", str(args.noisy_dir), "--checkpoint", str(args.backbone_checkpoint),
                  "--output-dir", str(source), "--device", args.device, *settings]
    if args.backbone_config:
        source_cmd.extend(("--config", str(args.backbone_config)))
    run([*source_cmd, "--overwrite"])
    run([sys.executable, "scripts/infer_stream.py", "--mode", "crm",
         "--manifest", str(manifest), "--noisy-dir", str(args.noisy_dir),
         "--source-dir", str(source), "--checkpoint",
         str(ROOT / "checkpoints/crm/cross_backbone" / args.backbone / "dynamic_best.th"),
         "--output-dir", str(output / "dynamic/wav"), "--device", args.device, "--overwrite"])
    for name in ("source", "dynamic"):
        run([sys.executable, "metrics/evaluate.py", "--clean-dir", str(args.clean_dir),
             "--noisy-dir", str(args.noisy_dir),
             "--denoised-dir", str(output / name / "wav"),
             "--out-dir", str(output / name / "metrics"), "--method", name,
             "--references", str(manifest), "--fs", "16000", "--workers", str(args.workers)])


if __name__ == "__main__":
    main()
