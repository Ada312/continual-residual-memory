#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Utterance-paired bootstrap for full CRM minus utterance-local recovery."
        )
    )
    parser.add_argument("--recovery", type=Path)
    parser.add_argument("--crm", type=Path)
    parser.add_argument("--static", dest="legacy_static", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dynamic", dest="legacy_dynamic", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    recovery_path = args.recovery or args.legacy_static
    crm_path = args.crm or args.legacy_dynamic
    if recovery_path is None or crm_path is None:
        parser.error("--recovery and --crm are required")
    if args.recovery and args.legacy_static and args.recovery != args.legacy_static:
        parser.error("--recovery conflicts with legacy --static")
    if args.crm and args.legacy_dynamic and args.crm != args.legacy_dynamic:
        parser.error("--crm conflicts with legacy --dynamic")

    recovery = pd.read_csv(recovery_path)
    crm = pd.read_csv(crm_path)
    if recovery["Filename"].duplicated().any() or crm["Filename"].duplicated().any():
        raise RuntimeError("duplicate Filename in metric input")
    metric_columns = list(METRICS.values())
    joined = recovery[["Filename", *metric_columns]].merge(
        crm[["Filename", *metric_columns]],
        on="Filename",
        suffixes=("_recovery", "_crm"),
        validate="one_to_one",
    )
    if len(joined) != len(recovery) or len(joined) != len(crm):
        raise RuntimeError(
            "unpaired metric inputs: "
            f"recovery={len(recovery)}, crm={len(crm)}, paired={len(joined)}"
        )
    if joined.filter(regex="_(recovery|crm)$").isna().any().any():
        raise RuntimeError("NaN in paired metric inputs")
    rng = np.random.default_rng(args.seed)
    n = len(joined)
    rows = []
    for metric, column in METRICS.items():
        deltas = joined[f"{column}_crm"].to_numpy(float) - joined[
            f"{column}_recovery"
        ].to_numpy(float)
        means = np.empty(args.iterations, dtype=np.float64)
        block = 1000
        for start in range(0, args.iterations, block):
            count = min(block, args.iterations - start)
            indices = rng.integers(0, n, size=(count, n))
            means[start:start + count] = deltas[indices].mean(axis=1)
        rows.append({
            "metric": metric,
            "utterances": n,
            "crm_minus_recovery_mean": float(deltas.mean()),
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
        "comparison": "paired full CRM - utterance-local recovery",
        "seed": args.seed,
        "iterations": args.iterations,
        "recovery_csv": str(recovery_path),
        "crm_csv": str(crm_path),
        "paired_csv": str(paired_path),
        "script": str(Path(__file__).resolve()),
    }
    (args.output / "bootstrap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
