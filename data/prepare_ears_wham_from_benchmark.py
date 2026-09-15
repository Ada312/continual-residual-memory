#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path


EARS_BENCHMARK_V1_COMMIT = "97020e6"
EARS_BENCHMARK_V1_GENERATOR_SHA256 = (
    "7b863b756331cf95aa40916f02a293dd35b4f978efe8f4dcec8fe90e18f8c151"
)


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
    parser.add_argument("--seed", type=int, default=1337)
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


def stable_digest(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


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


def allocate_proportional_quotas(
    strata: dict[tuple[str, str, str], list[dict]], target: int
) -> dict[tuple[str, str, str], int]:
    total = sum(len(rows) for rows in strata.values())
    if target <= 0 or target > total:
        raise ValueError(f"train-size must be in [1, {total}], got {target}")
    exact = {key: target * len(rows) / total for key, rows in strata.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(quotas.values())
    order = sorted(
        strata,
        key=lambda key: (exact[key] - quotas[key], len(strata[key]), key),
        reverse=True,
    )
    for key in order[:remaining]:
        quotas[key] += 1
    assert sum(quotas.values()) == target
    return quotas


def select_train_rows(rows: list[dict], target: int, seed: int) -> list[dict]:
    strata: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["speaker"], speech_style(row["speech_file"]), snr_bin(row["snr_db"]))
        strata[key].append(row)
    quotas = allocate_proportional_quotas(strata, target)

    noise_counts: Counter[str] = Counter()
    selected: list[dict] = []
    stratum_order = sorted(
        strata,
        key=lambda key: (-quotas[key], stable_digest(seed, "|".join(key))),
    )
    for key in stratum_order:
        candidates = list(strata[key])
        for _ in range(quotas[key]):
            candidates.sort(
                key=lambda row: (
                    noise_counts[row["noise_file"]],
                    stable_digest(seed, f'{row["split"]}|{row["speaker"]}|{row["id"]}'),
                )
            )
            chosen = candidates.pop(0)
            selected.append(chosen)
            noise_counts[chosen["noise_file"]] += 1
    return sorted(selected, key=lambda row: (row["speaker"], int(row["id"])))


def sample_id(row: dict, seed: int) -> str:
    identity = f'{row["split"]}|{row["speaker"]}|{row["id"]}'
    return f'{row["speaker"]}_{stable_digest(seed, identity)[:16]}.wav'


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
    validation_rows = load_split(args.ears_wham_root, "valid")
    selected_train = select_train_rows(train_pool, args.train_size, args.seed)

    clean_dir = args.out_dir / "clean"
    noisy_dir = args.out_dir / "noisy"
    source_dir = args.out_dir / "source"
    for directory in (clean_dir, noisy_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output_rows = []
    metadata_lines = []
    seen_ids: set[str] = set()
    for row in selected_train + validation_rows:
        identifier = sample_id(row, args.seed)
        if identifier in seen_ids:
            raise RuntimeError(f"sample-id collision: {identifier}")
        seen_ids.add(identifier)
        ensure_link(clean_dir / identifier, Path(row["clean_path"]), args.overwrite_links)
        ensure_link(noisy_dir / identifier, Path(row["noisy_path"]), args.overwrite_links)
        public_row = {
            key: value
            for key, value in row.items()
            if key not in {"clean_path", "noisy_path"}
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
        "ears_benchmark_commit": EARS_BENCHMARK_V1_COMMIT,
        "generator_sha256": EARS_BENCHMARK_V1_GENERATOR_SHA256,
        "ears_wham_root": str(args.ears_wham_root.resolve()),
        "selection_seed": args.seed,
        "selection_strategy": (
            "proportional speaker x speech_style x 5dB_SNR_bin strata; "
            "within-stratum greedy WHAM recording diversity"
        ),
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
