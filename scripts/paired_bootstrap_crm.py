#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = {
    "PESQ": "PESQ",
    "STOI": "STOI",
    "CSIG": "C_sig",
    "CBAK": "C_bak",
    "COVL": "C_ovl",
    "SSNR": "SSNR",
    "SISDR": "SISDR",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Utterance-paired bootstrap for CRM Dynamic minus Static.")
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    static = pd.read_csv(args.static)
    dynamic = pd.read_csv(args.dynamic)
    if static["Filename"].duplicated().any() or dynamic["Filename"].duplicated().any():
        raise RuntimeError("duplicate Filename in metric input")
    metric_columns = list(METRICS.values())
    joined = static[["Filename", *metric_columns]].merge(
        dynamic[["Filename", *metric_columns]],
        on="Filename",
        suffixes=("_static", "_dynamic"),
        validate="one_to_one",
    )
    if len(joined) != len(static) or len(joined) != len(dynamic):
        raise RuntimeError(f"unpaired metric inputs: static={len(static)}, dynamic={len(dynamic)}, paired={len(joined)}")
    if joined.filter(regex="_(static|dynamic)$").isna().any().any():
        raise RuntimeError("NaN in paired metric inputs")
    rng = np.random.default_rng(args.seed)
    n = len(joined)
    rows = []
    for metric, column in METRICS.items():
        deltas = joined[f"{column}_dynamic"].to_numpy(float) - joined[f"{column}_static"].to_numpy(float)
        means = np.empty(args.iterations, dtype=np.float64)
        block = 1000
        for start in range(0, args.iterations, block):
            count = min(block, args.iterations - start)
            indices = rng.integers(0, n, size=(count, n))
            means[start:start + count] = deltas[indices].mean(axis=1)
        rows.append({
            "metric": metric,
            "utterances": n,
            "dynamic_minus_static_mean": float(deltas.mean()),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
            "wins": int((deltas > 0).sum()),
            "ties": int((deltas == 0).sum()),
            "losses": int((deltas < 0).sum()),
        })
    args.output.mkdir(parents=True, exist_ok=True)
    paired_path = args.output / "paired_inputs.csv"
    joined.to_csv(paired_path, index=False)
    pd.DataFrame(rows).to_csv(args.output / "paired_bootstrap_ci.csv", index=False)
    metadata = {
        "sample_unit": "utterance",
        "comparison": "paired Dynamic - Static",
        "seed": args.seed,
        "iterations": args.iterations,
        "static_csv": str(args.static),
        "static_csv_sha256": sha256(args.static),
        "dynamic_csv": str(args.dynamic),
        "dynamic_csv_sha256": sha256(args.dynamic),
        "paired_csv": str(paired_path),
        "paired_csv_sha256": sha256(paired_path),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
    }
    (args.output / "bootstrap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
