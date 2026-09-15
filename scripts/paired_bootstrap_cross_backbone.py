#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


BACKBONES = {
    "storm50": "StoRM-50",
    "gtcrn": "GTCRN",
    "fastenhancer_b": "FastEnhancer-B",
    "flowse": "FlowSE",
    "ul_unas": "UL-UNAS",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_ci(
    deltas: np.ndarray, iterations: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    block = 1000
    for start in range(0, iterations, block):
        count = min(block, iterations - start)
        indices = rng.integers(0, len(deltas), size=(count, len(deltas)))
        means[start : start + count] = deltas[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Formal utterance-paired PESQ bootstrap for cross-backbone Dynamic minus Source."
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Write results separately from frozen reference inputs")
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    run = args.run.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else run
    statistics_dir = output_dir / "statistics"
    statistics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    inputs: list[dict[str, object]] = []

    for directory, display_name in BACKBONES.items():
        source_path = run / directory / "source/metrics/source_per_file_metrics.csv"
        dynamic_path = run / directory / "dynamic/metrics/dynamic_per_file_metrics.csv"
        if not source_path.exists():
            source_path = run / directory / "source_per_file_metrics.csv"
        if not dynamic_path.exists():
            dynamic_path = run / directory / "dynamic_per_file_metrics.csv"
        source = pd.read_csv(source_path)
        dynamic = pd.read_csv(dynamic_path)
        for label, frame in (("Source", source), ("Dynamic", dynamic)):
            if frame["Filename"].duplicated().any():
                raise RuntimeError(f"{display_name} {label}: duplicate Filename")
            if frame[["Filename", "PESQ"]].isna().any().any():
                raise RuntimeError(f"{display_name} {label}: NaN in paired inputs")

        paired = source[["Filename", "PESQ"]].merge(
            dynamic[["Filename", "PESQ"]],
            on="Filename",
            suffixes=("_source", "_dynamic"),
            validate="one_to_one",
        )
        if len(paired) != len(source) or len(paired) != len(dynamic):
            raise RuntimeError(
                f"{display_name}: unpaired inputs "
                f"source={len(source)}, dynamic={len(dynamic)}, paired={len(paired)}"
            )
        if len(paired) != 150:
            raise RuntimeError(f"{display_name}: expected 150 DNS utterances, got {len(paired)}")

        paired["PESQ_delta_dynamic_minus_source"] = (
            paired["PESQ_dynamic"] - paired["PESQ_source"]
        )
        paired_dir = statistics_dir / directory
        paired_dir.mkdir(parents=True, exist_ok=True)
        paired_path = paired_dir / "paired_inputs.csv"
        paired.to_csv(paired_path, index=False)

        deltas = paired["PESQ_delta_dynamic_minus_source"].to_numpy(dtype=float)
        low, high = bootstrap_ci(deltas, args.iterations, args.seed)
        row = {
            "backbone": display_name,
            "metric": "PESQ",
            "comparison": "Dynamic - Source",
            "utterances": len(paired),
            "bootstrap_iterations": args.iterations,
            "bootstrap_seed": args.seed,
            "mean_delta": float(deltas.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "wins": int((deltas > 0).sum()),
            "ties": int((deltas == 0).sum()),
            "losses": int((deltas < 0).sum()),
            "source_mean": float(paired["PESQ_source"].mean()),
            "dynamic_mean": float(paired["PESQ_dynamic"].mean()),
            "source_csv": str(source_path),
            "source_csv_sha256": sha256(source_path),
            "dynamic_csv": str(dynamic_path),
            "dynamic_csv_sha256": sha256(dynamic_path),
            "paired_csv": str(paired_path),
            "paired_csv_sha256": sha256(paired_path),
        }
        rows.append(row)
        inputs.append(
            {
                "backbone": display_name,
                "source_csv": str(source_path),
                "source_csv_sha256": row["source_csv_sha256"],
                "dynamic_csv": str(dynamic_path),
                "dynamic_csv_sha256": row["dynamic_csv_sha256"],
                "paired_csv": str(paired_path),
                "paired_csv_sha256": row["paired_csv_sha256"],
            }
        )

    output_path = output_dir / "cross_backbone_bootstrap.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    script_path = Path(__file__).resolve()
    metadata = {
        "status": "FROZEN FORMAL ARTIFACT",
        "sample_unit": "utterance",
        "dataset": "DNS 2020 synthetic no-reverb",
        "metric": "PESQ",
        "comparison": "paired Dynamic - Source",
        "confidence_interval": "95% percentile bootstrap",
        "iterations": args.iterations,
        "seed": args.seed,
        "rng": "numpy.random.default_rng; independently reinitialized with the fixed seed for each backbone",
        "output_csv": str(output_path),
        "output_csv_sha256": sha256(output_path),
        "script": str(script_path),
        "script_sha256": sha256(script_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "inputs": inputs,
    }
    metadata_path = output_dir / "cross_backbone_bootstrap_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(rows)[["backbone", "utterances", "mean_delta", "ci95_low", "ci95_high"]].to_string(index=False))
    print(f"\nCSV SHA256: {metadata['output_csv_sha256']}")


if __name__ == "__main__":
    main()
