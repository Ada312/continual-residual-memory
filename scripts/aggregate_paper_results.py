#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aggregate_cross_backbone import aggregate as aggregate_cross_backbone


METRICS = ["PESQ", "STOI", "C_sig", "C_bak", "C_ovl", "SSNR", "SISDR"]
DATASETS = ["dns", "ears_d", "musan_music"]
METHODS = ["source", "laden", "mpol", "static", "dynamic"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper main tables and Dynamic-Static deltas from per-file CSVs."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main_rows = []
    delta_rows = []
    for dataset in DATASETS:
        frames = {}
        for method in METHODS:
            path = args.results_root / "main" / "per_utterance" / f"{dataset}_{method}.csv"
            frame = pd.read_csv(path).sort_values("Filename").reset_index(drop=True)
            frames[method] = frame
            row = {"dataset": dataset, "method": method, "num_files": len(frame)}
            row.update({metric: frame[metric].mean() for metric in METRICS})
            main_rows.append(row)

        static = frames["static"]
        dynamic = frames["dynamic"]
        if not static["Filename"].equals(dynamic["Filename"]):
            raise ValueError(f"{dataset}: Static/Dynamic filenames are not paired")
        for metric in METRICS:
            delta = dynamic[metric] - static[metric]
            delta_rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "mean_delta": delta.mean(),
                    "wins": int((delta > 0).sum()),
                    "ties": int((delta == 0).sum()),
                    "losses": int((delta < 0).sum()),
                }
            )

    main = pd.DataFrame(main_rows)
    main = main.rename(columns={"C_sig": "CSIG", "C_bak": "CBAK", "C_ovl": "COVL"})
    main["method"] = main["method"].replace({"source": "backbone", "static": "crm_no_memory", "dynamic": "crm"})
    main.to_csv(args.output_dir / "paper_main_results.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(
        args.output_dir / "dynamic_minus_static.csv", index=False
    )

    cross = aggregate_cross_backbone(args.results_root / "cross_backbone", Path(__file__).resolve().parents[1] / "manifests/dns.csv")
    cross.to_csv(args.output_dir / "paper_cross_backbone_results.csv", index=False)
    print(f"wrote paper tables to {args.output_dir}")


if __name__ == "__main__":
    main()
