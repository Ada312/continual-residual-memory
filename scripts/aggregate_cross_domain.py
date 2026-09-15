#!/usr/bin/env python3
"""Generate the paper's DNS cross-domain table from frozen paired metrics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = {"PESQ": "PESQ", "COVL": "C_ovl", "SI-SDR": "SISDR"}
CONDITIONS = {"Matched history": "matched_dns", "Cross-domain history": "musan_to_dns_switch"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, default=ROOT / "results/transition/dns_all_methods_paired.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/dns.csv")
    parser.add_argument("--switch-csv", type=Path, help="Optional newly evaluated CRM cross-domain per-file metrics")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = sorted(csv.DictReader(handle), key=lambda row: int(row["test_order"]))
    ids = [row["filename"] for row in manifest]
    if len(ids) != 150 or len(set(ids)) != len(ids):
        raise ValueError("expected 150 unique DNS utterances")
    frame = pd.read_csv(args.paired)
    if len(frame) != 150 or frame.Filename.tolist() != ids or frame.test_order.tolist() != list(range(150)):
        raise ValueError("paired metrics must follow the frozen DNS order without missing/duplicate IDs")
    if args.switch_csv is not None:
        switch = pd.read_csv(args.switch_csv)
        if len(switch) != 150 or switch.Filename.duplicated().any() or set(switch.Filename) != set(ids):
            raise ValueError("cross-domain metrics do not pair one-to-one with frozen DNS")
        switch = switch.set_index("Filename").loc[ids]
        for column in METRICS.values():
            frame[f"musan_to_dns_switch_{column}"] = switch[column].to_numpy()
    rows = []
    for label, prefix in CONDITIONS.items():
        values = {"history_condition": label, "n": len(ids)}
        for name, column in METRICS.items():
            data = frame[f"{prefix}_{column}"]
            if data.isna().any():
                raise ValueError(f"missing {name} in {label}")
            values[name] = data.mean()
        rows.append(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
