#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate CRM prototype-state storage.")
    parser.add_argument("--prototypes", type=int, default=64)
    parser.add_argument("--frequency-bins", type=int, default=257)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    k, f = args.prototypes, args.frequency_bins
    parts = {
        "means_float32": k * f * 4,
        "variances_float32": k * f * 4,
        "counts_int64": k * 8,
        "usage_float32": k * 4,
        "last_used_int64": k * 8,
    }
    persistent = sum(parts.values())
    candidate = f * 4
    report = {
        "prototypes": k,
        "frequency_bins": f,
        "parts_bytes": parts,
        "persistent_bytes": persistent,
        "persistent_kib": persistent / 1024,
        "candidate_bytes": candidate,
        "maximum_with_candidate_kib": (persistent + candidate) / 1024,
        "paper_rounded_maximum_kib": round((persistent + candidate) / 1024, 2),
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
