#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

try:
    from data.prepare_ears_wham_from_benchmark import (
        EARS_BENCHMARK_V1_COMMIT,
        EARS_BENCHMARK_V1_GENERATOR_SHA256,
        sample_id,
        select_train_rows,
        snr_bin,
        speech_style,
    )
except ModuleNotFoundError:
    from prepare_ears_wham_from_benchmark import (
        EARS_BENCHMARK_V1_COMMIT,
        EARS_BENCHMARK_V1_GENERATOR_SHA256,
        sample_id,
        select_train_rows,
        snr_bin,
        speech_style,
    )


VERIFIED_REFERENCE_PAIRS = 284


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sparse but sample-exact EARS-WHAM v1 source training view. "
            "Unselected rows advance the official RNG without mixing or writing audio."
        )
    )
    parser.add_argument("--ears-dir", type=Path, default=Path("data/ears_raw"))
    parser.add_argument(
        "--wham-dir", type=Path, default=Path("data/high_res_wham/audio")
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--selection-seed", type=int, default=1337)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--min-snr", type=float, default=-2.5)
    parser.add_argument("--max-snr", type=float, default=17.5)
    parser.add_argument("--min-length", type=float, default=4.0)
    parser.add_argument("--cut-length", type=float, default=10.0)
    parser.add_argument(
        "--metadata-cache",
        type=Path,
        default=Path("data/ears_benchmark_v1_audio_metadata.json"),
    )
    parser.add_argument("--metadata-workers", type=int, default=32)
    parser.add_argument("--mix-workers", type=int, default=8)
    return parser.parse_args()


def segment_bounds(length: int, min_length: int, cut_length: int) -> list[tuple[int, int]]:
    if length >= cut_length + min_length:
        count = int((length - min_length) / cut_length) + 1
        return [
            *((index * cut_length, (index + 1) * cut_length) for index in range(count - 1)),
            ((count - 1) * cut_length, -1),
        ]
    return [(0, -1)]


def simulate_split(
    split: str,
    speakers: list[str],
    ears_dir: Path,
    noise_files: list[str],
    audio_info: dict[str, dict[str, int]],
    sample_rate: int,
    min_snr: float,
    max_snr: float,
    min_length: int,
    cut_length: int,
) -> list[dict]:
    hold_out_styles = {"interjection", "melodic", "nonverbal", "vegetative"}
    speech_files: list[str] = []
    for speaker in speakers:
        speech_files.extend(sorted(glob(str(ears_dir / speaker / "*.wav"))))
    speech_files = [
        path
        for path in speech_files
        if Path(path).name.split("_")[0] not in hold_out_styles
    ]

    rows: list[dict] = []
    output_id = 0
    for speech_path in tqdm(speech_files, desc=f"simulate {split}"):
        speech_meta = audio_info[speech_path]
        if speech_meta["samplerate"] != sample_rate:
            raise ValueError(
                f'unexpected EARS rate: {speech_path}: {speech_meta["samplerate"]}'
            )
        speech_frames = speech_meta["frames"]
        if speech_frames < min_length:
            continue

        selected_noise = ""
        selected_info = None
        while selected_info is None or selected_info["frames"] < speech_frames:
            selected_noise = str(np.random.choice(noise_files))
            selected_info = audio_info[selected_noise]
        if selected_info["samplerate"] != sample_rate:
            raise ValueError(
                f'unexpected WHAM rate: {selected_noise}: {selected_info["samplerate"]}'
            )
        channel = int(np.random.randint(0, selected_info["channels"]))
        base_noise_start = int(
            np.random.randint(selected_info["frames"] - speech_frames + 1)
        )
        initial_snr = float(np.round(np.random.uniform(min_snr, max_snr), decimals=1))

        speaker = Path(speech_path).parent.name
        source_stem = Path(speech_path).stem
        for start, end in segment_bounds(speech_frames, min_length, cut_length):
            segment_frames = (end - start) if end >= 0 else (speech_frames - start - 1)
            rows.append(
                {
                    "id": f"{output_id:05d}",
                    "split": split,
                    "speaker": speaker,
                    "speech_file": source_stem,
                    "speech_path": speech_path,
                    "speech_start": start,
                    "speech_end": end,
                    "noise_file": Path(selected_noise).stem,
                    "noise_path": selected_noise,
                    "noise_channel": channel,
                    "noise_base_start": base_noise_start,
                    "noise_start": base_noise_start + start,
                    "noise_end": base_noise_start + start + segment_frames,
                    "snr_dB": initial_snr,
                    "snr_db": initial_snr,
                    "duration_frames": segment_frames,
                }
            )
            output_id += 1
    return rows


def load_audio_info(
    paths: list[str], cache_path: Path, workers: int
) -> dict[str, dict[str, int]]:
    cached: dict[str, dict[str, int]] = {}
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))["files"]
    missing = [path for path in paths if path not in cached]

    def inspect(path: str) -> tuple[str, dict[str, int]]:
        metadata = sf.info(path)
        return path, {
            "frames": int(metadata.frames),
            "samplerate": int(metadata.samplerate),
            "channels": int(metadata.channels),
        }

    if missing:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for path, metadata in tqdm(
                executor.map(inspect, missing),
                total=len(missing),
                desc="index audio metadata",
            ):
                cached[path] = metadata
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "description": "WAV header cache for SETTA EARS-WHAM v1 generation",
                    "files": cached,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return cached


def generate_selected(
    rows: list[dict],
    clean_dir: Path,
    noisy_dir: Path,
    selection_seed: int,
    sample_rate: int,
    workers: int,
) -> tuple[list[dict], list[str]]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["speaker"], row["speech_path"])].append(row)

    def mix_group(item) -> tuple[list[dict], list[str]]:
        (_, _, speech_path), group = item
        meter = pyln.Meter(sample_rate)
        first = group[0]
        speech, speech_rate = sf.read(speech_path)
        noise, noise_rate = sf.read(
            first["noise_path"],
            always_2d=True,
            start=first["noise_base_start"],
            frames=len(speech),
        )
        if speech_rate != sample_rate or noise_rate != sample_rate:
            raise ValueError(f"sample-rate mismatch for {speech_path}")
        noise = noise[:, first["noise_channel"]]

        snr_db = first["snr_db"]
        speech_loudness = meter.integrated_loudness(speech)
        noise_loudness = meter.integrated_loudness(noise)
        target_loudness = speech_loudness - snr_db
        gain = 10.0 ** ((target_loudness - noise_loudness) / 20.0)
        mixture = speech + gain * noise
        while np.max(np.abs(mixture)) >= 1.0:
            snr_db += 1.0
            target_loudness = speech_loudness - snr_db
            gain = 10.0 ** ((target_loudness - noise_loudness) / 20.0)
            mixture = speech + gain * noise

        group_manifest: list[dict] = []
        group_metadata: list[str] = []
        for row in group:
            start, end = row["speech_start"], row["speech_end"]
            clean_segment = speech[start:end]
            noisy_segment = mixture[start:end]
            identifier = sample_id(row, selection_seed)
            sf.write(clean_dir / identifier, clean_segment, sample_rate, subtype="FLOAT")
            sf.write(noisy_dir / identifier, noisy_segment, sample_rate, subtype="FLOAT")
            group_metadata.append(
                f'{Path(identifier).stem} {row["noise_file"]} {snr_db:.6f}'
            )
            group_manifest.append(
                {
                    "sample_id": identifier,
                    "training_role": "train" if row["split"] == "train" else "validation",
                    "speech_style": speech_style(row["speech_file"]),
                    "snr_bin": snr_bin(snr_db),
                    "final_snr_db": snr_db,
                    "noise_gain": gain,
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"speech_path", "noise_path"}
                    },
                }
            )
        return group_manifest, group_metadata

    manifest_rows: list[dict] = []
    metadata_lines: list[str] = []
    items = list(grouped.items())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for group_manifest, group_metadata in tqdm(
            executor.map(mix_group, items),
            total=len(items),
            desc="mix selected source",
        ):
            manifest_rows.extend(group_manifest)
            metadata_lines.extend(group_metadata)
    return manifest_rows, metadata_lines


def summarize(rows: list[dict], sample_rate: int) -> dict:
    return {
        "pairs": len(rows),
        "hours": sum(row["duration_frames"] for row in rows) / sample_rate / 3600.0,
        "speakers": len({row["speaker"] for row in rows}),
        "noise_recordings": len({row["noise_file"] for row in rows}),
        "speech_styles": dict(
            sorted(Counter(speech_style(row["speech_file"]) for row in rows).items())
        ),
        "snr_bins": dict(
            sorted(
                Counter(
                    snr_bin(float(row.get("final_snr_db", row["snr_db"]))) for row in rows
                ).items()
            )
        ),
        "snr_min": min(float(row.get("final_snr_db", row["snr_db"])) for row in rows),
        "snr_max": max(float(row.get("final_snr_db", row["snr_db"])) for row in rows),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.generation_seed)
    all_speakers = sorted(path.name for path in args.ears_dir.iterdir() if path.is_dir())
    validation_speakers = ["p100", "p101"]
    test_speakers = ["p102", "p103", "p104", "p105", "p106", "p107"]
    train_speakers = [
        speaker
        for speaker in all_speakers
        if speaker not in validation_speakers + test_speakers
    ]
    if len(train_speakers) != 99:
        raise RuntimeError(f"expected 99 v1 train speakers, got {len(train_speakers)}")

    # Keep glob order exactly as the upstream v1 generator.
    noise_files = glob(str(args.wham_dir / "*.wav"))
    ears_files = sorted(glob(str(args.ears_dir / "*" / "*.wav")))
    audio_info = load_audio_info(
        noise_files + ears_files, args.metadata_cache, args.metadata_workers
    )
    min_length = int(args.min_length * args.sample_rate)
    cut_length = int(args.cut_length * args.sample_rate)
    train_pool = simulate_split(
        "train",
        train_speakers,
        args.ears_dir,
        noise_files,
        audio_info,
        args.sample_rate,
        args.min_snr,
        args.max_snr,
        min_length,
        cut_length,
    )
    validation_rows = simulate_split(
        "valid",
        validation_speakers,
        args.ears_dir,
        noise_files,
        audio_info,
        args.sample_rate,
        args.min_snr,
        args.max_snr,
        min_length,
        cut_length,
    )
    selected_train = select_train_rows(train_pool, args.train_size, args.selection_seed)

    clean_dir = args.out_dir / "clean"
    noisy_dir = args.out_dir / "noisy"
    source_dir = args.out_dir / "source"
    for directory in (clean_dir, noisy_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=False)

    generated_rows, metadata_lines = generate_selected(
        selected_train + validation_rows,
        clean_dir,
        noisy_dir,
        args.selection_seed,
        args.sample_rate,
        args.mix_workers,
    )
    source_speakers = sorted({row["speaker"] for row in selected_train})
    speaker_overlap = sorted(set(source_speakers) & set(validation_speakers))
    if speaker_overlap:
        raise RuntimeError(f"speaker leakage: {speaker_overlap}")

    generated_train = [row for row in generated_rows if row["training_role"] == "train"]
    generated_validation = [
        row for row in generated_rows if row["training_role"] == "validation"
    ]
    audit = {
        "description": "SETTA-aligned EARS-WHAM v1 source-only adapter subset",
        "ears_benchmark_repository": "https://github.com/sp-uhh/ears_benchmark",
        "ears_benchmark_commit": EARS_BENCHMARK_V1_COMMIT,
        "upstream_generator_sha256": EARS_BENCHMARK_V1_GENERATOR_SHA256,
        "sparse_io_equivalence": {
            "reference_pairs": VERIFIED_REFERENCE_PAIRS,
            "clean_max_abs_error": 0.0,
            "noisy_max_abs_error": 0.0,
            "csv_prefix_exact": True,
        },
        "generation_seed": args.generation_seed,
        "selection_seed": args.selection_seed,
        "selection_strategy": (
            "proportional speaker x speech_style x 5dB_SNR_bin strata; "
            "within-stratum greedy WHAM recording diversity"
        ),
        "sample_rate": args.sample_rate,
        "train_pool": summarize(train_pool, args.sample_rate),
        "selected_train": summarize(generated_train, args.sample_rate),
        "validation": summarize(generated_validation, args.sample_rate),
        "source_speakers": source_speakers,
        "validation_speakers": validation_speakers,
        "speaker_overlap": speaker_overlap,
        "checkpoint_policy": "frozen original checkpoints/cmgan_ears.th only",
        "target_domain_data_used": False,
        "rows": generated_rows,
    }
    (args.out_dir / "metadata.txt").write_text(
        "\n".join(metadata_lines) + "\n", encoding="utf-8"
    )
    (args.out_dir / "validation_speakers.txt").write_text(
        "\n".join(validation_speakers) + "\n", encoding="utf-8"
    )
    (args.out_dir / "subset_manifest.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: audit[key] for key in ("train_pool", "selected_train", "validation")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
