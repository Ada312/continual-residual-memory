#!/usr/bin/env python3
"""Regenerate the five-backbone paper table from paired per-file metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKBONES = {
    "storm50": "StoRM-50", "gtcrn": "GTCRN", "fastenhancer_b": "FastEnhancer-B",
    "flowse": "FlowSE", "ul_unas": "UL-UNAS",
}
METRICS = ["PESQ", "STOI", "C_sig", "C_bak", "C_ovl", "SSNR", "SISDR"]


def aggregate(run: Path, manifest: Path) -> pd.DataFrame:
    with manifest.open(newline="", encoding="utf-8") as handle:
        ordered = sorted(csv.DictReader(handle), key=lambda row: int(row["test_order"]))
    ids = [row["filename"] for row in ordered]
    if len(ids) != 150 or len(set(ids)) != len(ids):
        raise ValueError("expected 150 unique DNS utterances in frozen manifest")
    rows = []
    for folder, backbone in BACKBONES.items():
        for method in ("source", "dynamic"):
            path = run / folder / f"{method}_per_file_metrics.csv"
            if not path.exists():
                path = run / folder / method / "metrics" / f"{method}_per_file_metrics.csv"
            frame = pd.read_csv(path)
            if (len(frame) != len(ids) or frame.Filename.duplicated().any()
                    or set(frame.Filename) != set(ids) or frame[METRICS].isna().any().any()):
                raise ValueError(f"unpaired/invalid DNS metrics: {path}")
            rows.append({"Backbone": backbone, "Method": method.title(),
                         **{metric: frame[metric].mean() for metric in METRICS}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/cross_backbone")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/dns.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = aggregate(args.run, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
