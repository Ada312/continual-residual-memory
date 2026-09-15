#!/usr/bin/env python3
"""Plot smoothed per-utterance DNS CRM-minus-Static PESQ/COVL gains."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_N = 150


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(manifest_path: Path, static_path: Path, dynamic_path: Path):
    with manifest_path.open(newline="") as handle:
        manifest = sorted(csv.DictReader(handle), key=lambda row: int(row["test_order"]))
    if len(manifest) != EXPECTED_N:
        raise RuntimeError(f"DNS manifest has {len(manifest)} rows; expected {EXPECTED_N}")
    expected_ids = [row["filename"] for row in manifest]
    if [int(row["test_order"]) for row in manifest] != list(range(EXPECTED_N)):
        raise RuntimeError("DNS test_order must be contiguous and zero-based")

    methods = {}
    for method, path in (("static", static_path), ("dynamic", dynamic_path)):
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        ids = [row["Filename"] for row in rows]
        if ids != expected_ids:
            raise RuntimeError(f"{method}: IDs/order do not match frozen DNS manifest")
        methods[method] = rows
    return manifest, methods


def causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(
        [values[max(0, i - window + 1) : i + 1].mean() for i in range(len(values))]
    )


def draw(methods, metric_column: str, metric_label: str, window: int, output: Path):
    static = np.asarray([float(row[metric_column]) for row in methods["static"]])
    dynamic = np.asarray([float(row[metric_column]) for row in methods["dynamic"]])
    gain = dynamic - static
    smoothed = causal_moving_average(gain, window)
    x = np.arange(1, EXPECTED_N + 1)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    })
    fig, axis = plt.subplots(figsize=(4.0, 3.07))
    axis.axhline(0, color="#555555", linewidth=1.0, linestyle="--")
    axis.plot(x, smoothed, color="#73866B", linewidth=1.8)
    axis.set_xlim(-5, 155)
    axis.set_xticks([0, 50, 100, 150])
    axis.set_xlabel("# Utterance")
    axis.set_ylabel(rf"$\Delta${metric_label} $\uparrow$")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=1.0, length=4)
    axis.spines["left"].set_linewidth(1.0)
    axis.spines["bottom"].set_linewidth(1.0)
    lower = min(0.0, float(gain.min()))
    upper = max(0.0, float(gain.max()))
    padding = 0.06 * (upper - lower)
    axis.set_ylim(lower - padding, upper + padding)
    fig.tight_layout(pad=0.25)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return gain


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=here.parent / "manifests/dns.csv")
    parser.add_argument("--static", type=Path, default=here.parent / "results/main/per_utterance/dns_static.csv")
    parser.add_argument("--dynamic", type=Path, default=here.parent / "results/main/per_utterance/dns_dynamic.csv")
    parser.add_argument("--output-dir", type=Path, default=here.parent / "outputs/figures")
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()
    if args.window < 1:
        raise ValueError("--window must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, methods = load_inputs(args.manifest, args.static, args.dynamic)
    pesq = draw(methods, "PESQ", "PESQ", args.window, args.output_dir / "pesq.png")
    covl = draw(methods, "C_ovl", "COVL", args.window, args.output_dir / "covl.png")
    metadata = {
        "definition": "per-utterance Dynamic/CRM minus Static",
        "utterances": EXPECTED_N,
        "order": "manifest test_order, converted from 0-based to x=1..150",
        "smoothing": f"causal trailing moving average, window={args.window}, no future samples",
        "raw_points_visible": False,
        "input_sha256": {
            "manifest": sha256(args.manifest),
            "static": sha256(args.static),
            "dynamic": sha256(args.dynamic),
        },
        "raw_gain_summary": {
            "pesq_min": float(pesq.min()), "pesq_max": float(pesq.max()),
            "covl_min": float(covl.min()), "covl_max": float(covl.max()),
        },
    }
    (args.output_dir / "plot_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
