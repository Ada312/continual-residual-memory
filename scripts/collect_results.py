#!/usr/bin/env python3
"""Assemble the three paper streams from evaluated per-utterance CSVs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"dns": "dns", "ears_d": "ears_d", "libri_musan": "musan_music"}
METHODS = {"source": "backbone", "static": "crm_no_memory", "dynamic": "crm"}
METRICS = ("PESQ", "STOI", "C_sig", "C_bak", "C_ovl", "SSNR", "SISDR")


def validate(csv_path: Path, manifest: Path) -> None:
    with manifest.open(newline="", encoding="utf-8") as handle:
        ids = [row["filename"] for row in csv.DictReader(handle)]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"Filename", *METRICS}.issubset(reader.fieldnames or []):
            raise ValueError(f"missing seven-metric schema: {csv_path}")
        rows = list(reader)
    actual = [row["Filename"] for row in rows]
    if len(ids) != len(actual) or len(set(actual)) != len(actual) or set(ids) != set(actual):
        raise ValueError(f"missing, duplicate or extra utterance in {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root / "main/per_utterance"
    output.mkdir(parents=True, exist_ok=True)
    for dataset, file_prefix in DATASETS.items():
        manifest = ROOT / "manifests" / f"{file_prefix}.csv"
        for name, public_name in METHODS.items():
            source = args.main_root / dataset / public_name / "metrics" / f"{public_name}_per_file_metrics.csv"
            validate(source, manifest)
            shutil.copyfile(source, output / f"{file_prefix}_{name}.csv")
        for baseline in ("laden", "mpol"):
            source = args.baseline_root / dataset / baseline / "metrics" / f"{baseline}_per_file_metrics.csv"
            validate(source, manifest)
            shutil.copyfile(source, output / f"{file_prefix}_{baseline}.csv")
    print(f"three-domain paired metric CSVs: {output}")


if __name__ == "__main__":
    main()
