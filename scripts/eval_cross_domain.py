#!/usr/bin/env python3
"""Carry mature Libri-MUSAN prototype memory into causal DNS evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musan-noisy-dir", type=Path, required=True)
    parser.add_argument("--musan-source-dir", type=Path, required=True)
    parser.add_argument("--dns-noisy-dir", type=Path, required=True)
    parser.add_argument("--dns-source-dir", type=Path, required=True)
    parser.add_argument("--dns-clean-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/crm/dynamic_best.th")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    state_path = args.output_dir / "musan_final_memory.pt"

    def run(command: list[str]) -> None:
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    for dataset, noisy, source, extra in (
        ("musan_music", args.musan_noisy_dir, args.musan_source_dir,
         ["--save-memory-state", str(state_path)]),
        ("dns", args.dns_noisy_dir, args.dns_source_dir,
         ["--initial-memory-state", str(state_path)]),
    ):
        run([sys.executable, "scripts/infer_stream.py", "--mode", "crm",
             "--manifest", str(ROOT / "data/manifests" / f"{dataset}.csv"),
             "--noisy-dir", str(noisy), "--source-dir", str(source),
             "--output-dir", str(args.output_dir / dataset / "wav"),
             "--checkpoint", str(args.checkpoint), "--device", args.device,
             "--overwrite", *extra])
    run([sys.executable, "metrics/evaluate.py", "--clean-dir", str(args.dns_clean_dir),
         "--noisy-dir", str(args.dns_noisy_dir),
         "--denoised-dir", str(args.output_dir / "dns/wav"),
         "--out-dir", str(args.output_dir / "dns/metrics"), "--method", "crm_cross_domain",
         "--references", str(ROOT / "data/manifests/dns.csv"), "--fs", "16000",
         "--workers", str(args.workers)])


if __name__ == "__main__":
    main()
