#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


DNS_OUTPUT_PATTERN = re.compile(r"dns2020_no_reverb_fileid_(\d+)\.wav$")
RAW_FILE_ID_PATTERN = re.compile(r"_fileid_(\d+)\.wav$")


def index_by_file_id(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)

    index: dict[int, Path] = {}
    for path in sorted(directory.glob("*.wav")):
        match = RAW_FILE_ID_PATTERN.search(path.name)
        if match is None:
            continue
        file_id = int(match.group(1))
        if file_id in index:
            raise ValueError(
                f"duplicate file ID {file_id} in {directory}: "
                f"{index[file_id].name}, {path.name}"
            )
        index[file_id] = path
    return index


def required_file_ids(rows: Iterable[dict[str, str]]) -> list[tuple[str, int]]:
    required: list[tuple[str, int]] = []
    seen_filenames: set[str] = set()
    seen_ids: set[int] = set()
    for row in rows:
        filename = row["filename"]
        match = DNS_OUTPUT_PATTERN.fullmatch(filename)
        if match is None:
            raise ValueError(f"unexpected DNS manifest name: {filename}")
        file_id = int(match.group(1))
        if filename in seen_filenames or file_id in seen_ids:
            raise ValueError(f"duplicate DNS manifest entry: {filename}")
        seen_filenames.add(filename)
        seen_ids.add(file_id)
        required.append((filename, file_id))
    return required


def resolve_required_pairs(
    required: Iterable[tuple[str, int]],
    clean_index: dict[int, Path],
    noisy_index: dict[int, Path],
) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for filename, file_id in required:
        clean = clean_index.get(file_id)
        noisy = noisy_index.get(file_id)
        if clean is None:
            raise FileNotFoundError(f"missing clean file ID {file_id}")
        if noisy is None:
            raise FileNotFoundError(f"missing noisy file ID {file_id}")
        pairs.append((filename, clean, noisy))
    return pairs


def prepare_dns(dns_root: Path, manifest: Path, output_root: Path) -> None:
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 150:
        raise ValueError(f"expected 150 DNS entries, found {len(rows)}")

    orders = [int(row["test_order"]) for row in rows]
    if orders != list(range(150)):
        raise ValueError("DNS manifest test_order must be contiguous from 0 to 149")

    required = required_file_ids(rows)
    clean_index = index_by_file_id(dns_root / "clean")
    noisy_index = index_by_file_id(dns_root / "noisy")
    pairs = resolve_required_pairs(required, clean_index, noisy_index)

    destinations: list[tuple[Path, Path]] = []
    for filename, clean, noisy in pairs:
        destinations.extend(
            (
                (clean, output_root / "clean" / filename),
                (noisy, output_root / "noisy" / filename),
            )
        )
    for _, destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)

    (output_root / "clean").mkdir(parents=True, exist_ok=True)
    (output_root / "noisy").mkdir(parents=True, exist_ok=True)
    for source, destination in destinations:
        destination.symlink_to(source.resolve())

    print(f"linked {len(pairs)} DNS pairs under {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the fixed DNS 2020 synthetic no-reverb test set."
    )
    parser.add_argument("--dns-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/dns.csv")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    prepare_dns(args.dns_root, args.manifest, args.output_root)


if __name__ == "__main__":
    main()
