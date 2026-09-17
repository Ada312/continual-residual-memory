#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic flat training view from the official "
            "EARS-WHAM v1 train split and complete validation split."
        )
    )
    parser.add_argument(
        "--ears-wham-root",
        type=Path,
        default=Path("data/ears_benchmark_v1_inputs/EARS-WHAM"),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=ROOT / "data/manifests/training/subset_manifest.json",
    )
    parser.add_argument("--overwrite-links", action="store_true")
    return parser.parse_args()


def speech_style(speech_file: str) -> str:
    stem = Path(speech_file).stem
    if stem.startswith("emo_"):
        return "emotion"
    for prefix in (
        "sentences",
        "rainbow",
        "freeform",
        "nonverbal",
        "interjection",
        "vegetative",
        "melodic",
    ):
        if stem == prefix or stem.startswith(prefix + "_"):
            return prefix
    return stem.split("_", maxsplit=1)[0]


def snr_bin(snr_db: float) -> str:
    lower = 5.0 * math.floor((snr_db + 2.5) / 5.0) - 2.5
    upper = lower + 5.0
    return f"[{lower:g},{upper:g})"


def load_split(root: Path, split: str) -> list[dict]:
    csv_path = root / f"{split}.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["split"] = split
        row["snr_db"] = float(row["snr_dB"])
        row["speech_start"] = int(row["speech_start"])
        row["speech_end"] = int(row["speech_end"])
        row["noise_start"] = int(row["noise_start"])
        row["noise_end"] = int(row["noise_end"])
        clean_name = f'{row["id"]}.wav'
        noisy_name = f'{row["id"]}_{row["snr_db"]:.1f}dB.wav'
        row["clean_path"] = str(root / split / "clean" / row["speaker"] / clean_name)
        row["noisy_path"] = str(root / split / "noisy" / row["speaker"] / noisy_name)
        if not Path(row["clean_path"]).is_file() or not Path(row["noisy_path"]).is_file():
            raise FileNotFoundError(
                f'missing v1 pair for {split} row {row["id"]}: '
                f'{row["clean_path"]}, {row["noisy_path"]}'
            )
    return rows


def load_fixed_selection(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError(f"selection manifest contains no rows: {path}")
    return rows


def select_fixed_rows(
    source_rows: list[dict], fixed_rows: list[dict], training_role: str
) -> list[dict]:
    source = {}
    for row in source_rows:
        key = (row["split"], row["speaker"], str(row["id"]))
        if key in source:
            raise RuntimeError(f"duplicate EARS-WHAM source row: {key}")
        source[key] = row
    selected = []
    seen = set()
    for fixed in fixed_rows:
        if fixed["training_role"] != training_role:
            continue
        key = (fixed["split"], fixed["speaker"], str(fixed["id"]))
        if key in seen:
            raise RuntimeError(f"duplicate fixed EARS-WHAM row: {key}")
        seen.add(key)
        if key not in source:
            raise RuntimeError(f"fixed EARS-WHAM row is unavailable: {key}")
        row = dict(source[key])
        for field in ("speech_file", "speech_start", "speech_end", "noise_file"):
            if str(row[field]) != str(fixed[field]):
                raise RuntimeError(
                    f"fixed EARS-WHAM row differs at {key} field {field}: "
                    f"{row[field]} != {fixed[field]}"
                )
        row["sample_id"] = fixed["sample_id"]
        selected.append(row)
    return selected


def ensure_link(link: Path, target: Path, overwrite: bool) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        if link.resolve() == target:
            return
        if not overwrite:
            raise FileExistsError(f"link points to a different target: {link}")
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"refusing to replace non-symlink path: {link}")
    link.symlink_to(os.path.relpath(target, start=link.parent))


def summarize(rows: list[dict]) -> dict:
    durations = [(row["noise_end"] - row["noise_start"]) for row in rows]
    return {
        "pairs": len(rows),
        "hours": sum(durations) / 48_000.0 / 3600.0,
        "speakers": len({row["speaker"] for row in rows}),
        "noise_recordings": len({row["noise_file"] for row in rows}),
        "speech_styles": dict(
            sorted(Counter(speech_style(row["speech_file"]) for row in rows).items())
        ),
        "snr_bins": dict(sorted(Counter(snr_bin(row["snr_db"]) for row in rows).items())),
        "snr_min": min(row["snr_db"] for row in rows),
        "snr_max": max(row["snr_db"] for row in rows),
    }


def main() -> None:
    args = parse_args()
    train_pool = load_split(args.ears_wham_root, "train")
    validation_pool = load_split(args.ears_wham_root, "valid")
    fixed_rows = load_fixed_selection(args.selection_manifest)
    selected_train = select_fixed_rows(train_pool, fixed_rows, "train")
    validation_rows = select_fixed_rows(validation_pool, fixed_rows, "validation")
    if len(selected_train) != args.train_size:
        raise RuntimeError(
            f"fixed selection has {len(selected_train)} training rows, "
            f"not requested {args.train_size}"
        )

    clean_dir = args.out_dir / "clean"
    noisy_dir = args.out_dir / "noisy"
    source_dir = args.out_dir / "source"
    for directory in (clean_dir, noisy_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_rows = []
    metadata_lines = []
    seen_ids: set[str] = set()
    for row in selected_train + validation_rows:
        identifier = row["sample_id"]
        if identifier in seen_ids:
            raise RuntimeError(f"sample-id collision: {identifier}")
        seen_ids.add(identifier)
        ensure_link(clean_dir / identifier, Path(row["clean_path"]), args.overwrite_links)
        ensure_link(noisy_dir / identifier, Path(row["noisy_path"]), args.overwrite_links)
        public_row = {
            key: value
            for key, value in row.items()
            if key not in {"clean_path", "noisy_path", "sample_id"}
        }
        output_rows.append(
            {
                "sample_id": identifier,
                "training_role": "train" if row["split"] == "train" else "validation",
                "speech_style": speech_style(row["speech_file"]),
                "snr_bin": snr_bin(row["snr_db"]),
                **public_row,
            }
        )
        metadata_lines.append(
            f'{Path(identifier).stem} {row["noise_file"]} {row["snr_db"]:.6f}'
        )

    source_speakers = sorted({row["speaker"] for row in selected_train})
    validation_speakers = sorted({row["speaker"] for row in validation_rows})
    overlap = sorted(set(source_speakers) & set(validation_speakers))
    if overlap:
        raise RuntimeError(f"speaker leakage between train and validation: {overlap}")

    audit = {
        "description": "SETTA-aligned EARS-WHAM v1 source-only adapter subset",
        "ears_benchmark_repository": "https://github.com/sp-uhh/ears_benchmark",
        "ears_wham_root": str(args.ears_wham_root.resolve()),
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_strategy": "exact identities from the repository's fixed selection metadata",
        "requested_train_size": args.train_size,
        "train_pool": summarize(train_pool),
        "selected_train": summarize(selected_train),
        "validation": summarize(validation_rows),
        "source_speakers": source_speakers,
        "validation_speakers": validation_speakers,
        "speaker_overlap": overlap,
        "checkpoint_policy": "frozen original checkpoints/cmgan_ears.th only",
        "target_domain_data_used": False,
        "rows": output_rows,
    }
    (args.out_dir / "metadata.txt").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8"
    )
    (args.out_dir / "subset_manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "validation_speakers.txt").write_text(
        "\n".join(validation_speakers) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: audit[key] for key in ("selected_train", "validation", "speaker_overlap")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
