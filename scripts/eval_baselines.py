#!/usr/bin/env python3
"""Reproduce official LaDen/MPol TTA with pinned SETTA and frozen stream order."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = {"dns": "dns.csv", "ears_d": "ears_d.csv", "libri_musan": "musan_music.csv"}
EXPECTED = {
    "run_da.py": "c9599108b6d2c50974a3acf88be91f7eefd2fd095ccc30c72adea47cb414ed9f",
    "adaptation/laden.py": "40a8789cd75059e9eefc92677ad82fc211162b572702a7b79a5192fa4674c85a",
    "adaptation/mpol.py": "3d35303549e1d2197b3db87929a92a26c11ac58a0ba0df51f38ae0d519c830e9",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=MANIFESTS, required=True)
    parser.add_argument("--method", choices=("laden", "mpol"), required=True)
    parser.add_argument("--setta-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Contains clean/ and noisy/")
    parser.add_argument("--cmgan-checkpoint", type=Path, required=True)
    parser.add_argument("--foundation-map", type=Path, default=ROOT / "checkpoints/baselines/WavLM_EARS_map.th")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        for name, expected in EXPECTED.items():
            actual = hashlib.sha256((args.setta_repo / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"SETTA snapshot mismatch: {name}; apply patches/setta_final_baselines.patch")
    manifest = ROOT / "manifests" / MANIFESTS[args.dataset]
    target = args.output_dir / args.dataset / args.method
    command = [
        sys.executable, str(args.setta_repo / "run_da.py"),
        "--config-name", args.method + "_cmgan",
        "hydra.run.dir=" + str(target),
        "data.path=" + str(args.dataset_root),
        "data.testset.path=" + str(args.dataset_root),
        "data.testset.name=VoiceBank",
        "+data.testset.order_manifest=" + str(manifest),
        "data.test_batch_size=1", "data.n_workers=1", "+seed=1337",
        "+eval.save_denoised=True", "logs=dummy",
        "model.load=" + str(args.cmgan_checkpoint), "metrics=[]", "async_metrics=False",
    ]
    if args.method == "laden":
        command.append("adaptation.foundation_map=" + str(args.foundation_map))
    print(" ".join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=args.setta_repo, check=True)
    metrics = [
        sys.executable, "metrics/evaluate.py", "--clean-dir", str(args.dataset_root / "clean"),
        "--noisy-dir", str(args.dataset_root / "noisy"),
        "--denoised-dir", str(target / "denoised"),
        "--out-dir", str(target / "metrics"), "--method", args.method,
        "--references", str(manifest), "--fs", "16000", "--workers", str(args.workers),
    ]
    print(" ".join(metrics))
    if not args.dry_run:
        subprocess.run(metrics, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
