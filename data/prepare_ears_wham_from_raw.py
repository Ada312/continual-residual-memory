#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
PAPER_SAMPLE_RATE = 48_000
EXPECTED_TRAIN = 8_192
EXPECTED_VALIDATION = 632
EXPECTED_VALIDATION_SPEAKERS = {"p100", "p101"}
REQUIRED_FIELDS = {
    "sample_id",
    "training_role",
    "final_snr_db",
    "noise_gain",
    "split",
    "speaker",
    "speech_file",
    "speech_start",
    "speech_end",
    "noise_file",
    "noise_channel",
    "noise_base_start",
    "noise_start",
    "noise_end",
    "snr_dB",
    "snr_db",
    "duration_frames",
}


@dataclass(frozen=True)
class AudioInfo:
    frames: int
    sample_rate: int
    channels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically reconstruct the paper EARS-WHAM training and "
            "held-out data from raw EARS, WHAM 48 kHz, and the fixed manifest."
        )
    )
    parser.add_argument(
        "--ears-dir",
        type=Path,
        required=True,
        help="EARS root containing one directory per speaker, e.g. p001/.",
    )
    parser.add_argument(
        "--wham-dir",
        type=Path,
        required=True,
        help="WHAM 48 kHz directory containing the referenced WAV files.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=ROOT / "data/manifests/training/subset_manifest.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"fixed selection manifest not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"manifest has no row list: {path}")
    return document, rows


def audio_info(path: Path) -> AudioInfo:
    info = sf.info(path)
    return AudioInfo(
        frames=int(info.frames),
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
    )


def paper_compatible_slice(waveform: np.ndarray, start: int, end: int) -> np.ndarray:
    if end == -1:
        # The dataset used for the paper applied Python's [start:-1] slice for
        # final/unsplit segments. Preserve that one-sample omission exactly.
        return waveform[start:-1]
    return waveform[start:end]


def expected_segment_frames(speech_frames: int, start: int, end: int) -> int:
    if end == -1:
        return speech_frames - start - 1
    return end - start


def resolve_inputs(
    rows: list[dict[str, Any]], ears_dir: Path, wham_dir: Path
) -> tuple[dict[tuple[str, str], Path], dict[str, Path]]:
    speech_paths = {
        (str(row["speaker"]), str(row["speech_file"])): (
            ears_dir / str(row["speaker"]) / f"{row['speech_file']}.wav"
        )
        for row in rows
    }
    noise_paths = {
        str(row["noise_file"]): wham_dir / f"{row['noise_file']}.wav" for row in rows
    }
    missing_speech = sorted(
        str(path) for path in speech_paths.values() if not path.is_file()
    )
    missing_noise = sorted(
        str(path) for path in noise_paths.values() if not path.is_file()
    )
    if missing_speech or missing_noise:
        details = []
        if missing_speech:
            details.append(
                f"missing {len(missing_speech)} EARS files; first: {missing_speech[0]}"
            )
        if missing_noise:
            details.append(
                f"missing {len(missing_noise)} WHAM files; first: {missing_noise[0]}"
            )
        raise FileNotFoundError("; ".join(details))
    return speech_paths, noise_paths


def validate_manifest(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("fixed selection manifest is empty")

    missing_fields = [
        (index, sorted(REQUIRED_FIELDS - set(row)))
        for index, row in enumerate(rows)
        if REQUIRED_FIELDS - set(row)
    ]
    if missing_fields:
        index, fields = missing_fields[0]
        raise ValueError(f"manifest row {index} is missing fields: {fields}")

    identifiers = [str(row["sample_id"]) for row in rows]
    duplicate_ids = sorted(
        identifier for identifier, count in Counter(identifiers).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(f"duplicate sample_id in fixed manifest: {duplicate_ids[0]}")
    if any(
        Path(identifier).name != identifier or not identifier.endswith(".wav")
        for identifier in identifiers
    ):
        raise ValueError("sample_id values must be plain .wav filenames")

    roles = Counter(str(row["training_role"]) for row in rows)
    expected_roles = {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}
    if dict(roles) != expected_roles:
        raise ValueError(f"expected role counts {expected_roles}, got {dict(roles)}")

    expected_splits = {"train": "train", "validation": "valid"}
    for index, row in enumerate(rows):
        role = str(row["training_role"])
        if role not in expected_splits or row["split"] != expected_splits[role]:
            raise ValueError(
                f"manifest row {index} has inconsistent role/split: "
                f"{role!r}/{row['split']!r}"
            )
        for key in ("final_snr_db", "noise_gain", "snr_dB", "snr_db"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"manifest row {index} has non-finite {key}")
        if float(row["noise_gain"]) <= 0.0:
            raise ValueError(f"manifest row {index} has non-positive noise_gain")
        initial_snr = float(row["snr_db"])
        final_snr = float(row["final_snr_db"])
        if float(row["snr_dB"]) != initial_snr:
            raise ValueError(
                f"manifest row {index} has inconsistent initial SNR fields"
            )
        snr_adjustment = final_snr - initial_snr
        if snr_adjustment < 0.0 or not math.isclose(
            snr_adjustment, round(snr_adjustment), abs_tol=1e-9
        ):
            raise ValueError(
                f"manifest row {index} has invalid clipping SNR adjustment"
            )

    train_speakers = {
        str(row["speaker"]) for row in rows if row["training_role"] == "train"
    }
    validation_speakers = {
        str(row["speaker"]) for row in rows if row["training_role"] == "validation"
    }
    overlap = sorted(train_speakers & validation_speakers)
    if overlap:
        raise ValueError(f"train/held-out speaker overlap: {overlap}")
    if validation_speakers != EXPECTED_VALIDATION_SPEAKERS:
        raise ValueError(
            "expected validation speakers "
            f"{sorted(EXPECTED_VALIDATION_SPEAKERS)}, got {sorted(validation_speakers)}"
        )


def validate_audio_inputs(
    rows: list[dict[str, Any]],
    speech_paths: dict[tuple[str, str], Path],
    noise_paths: dict[str, Path],
    workers: int,
) -> None:
    speech_items = list(speech_paths.items())
    noise_items = list(noise_paths.items())

    def inspect(item: tuple[Any, Path]) -> tuple[Any, AudioInfo]:
        key, path = item
        return key, audio_info(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        speech_info = dict(
            tqdm(
                executor.map(inspect, speech_items),
                total=len(speech_items),
                desc="validate EARS",
            )
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        noise_info = dict(
            tqdm(
                executor.map(inspect, noise_items),
                total=len(noise_items),
                desc="validate WHAM",
            )
        )

    for key, info in speech_info.items():
        if info.sample_rate != PAPER_SAMPLE_RATE:
            raise ValueError(
                f"EARS file {speech_paths[key]} is {info.sample_rate} Hz; "
                f"expected {PAPER_SAMPLE_RATE} Hz"
            )
        if info.channels != 1:
            raise ValueError(
                f"EARS file {speech_paths[key]} has {info.channels} channels; expected mono"
            )
    for key, info in noise_info.items():
        if info.sample_rate != PAPER_SAMPLE_RATE:
            raise ValueError(
                f"WHAM file {noise_paths[key]} is {info.sample_rate} Hz; "
                f"expected {PAPER_SAMPLE_RATE} Hz"
            )

    group_fields: dict[tuple[str, str], tuple[Any, ...]] = {}
    for index, row in enumerate(rows):
        speech_key = (str(row["speaker"]), str(row["speech_file"]))
        noise_key = str(row["noise_file"])
        speech = speech_info[speech_key]
        noise = noise_info[noise_key]
        start = int(row["speech_start"])
        end = int(row["speech_end"])
        duration = int(row["duration_frames"])
        channel = int(row["noise_channel"])
        base_start = int(row["noise_base_start"])
        noise_start = int(row["noise_start"])
        noise_end = int(row["noise_end"])

        if start < 0 or end < -1 or (end >= 0 and end > speech.frames):
            raise ValueError(f"manifest row {index} has invalid speech bounds")
        expected_duration = expected_segment_frames(speech.frames, start, end)
        if duration <= 0 or duration != expected_duration:
            raise ValueError(
                f"manifest row {index} duration mismatch: {duration} != {expected_duration}"
            )
        if channel < 0 or channel >= noise.channels:
            raise ValueError(f"manifest row {index} has invalid WHAM channel {channel}")
        if base_start < 0 or base_start + speech.frames > noise.frames:
            raise ValueError(f"manifest row {index} has invalid WHAM base offset")
        if noise_start != base_start + start or noise_end != noise_start + duration:
            raise ValueError(
                f"manifest row {index} has inconsistent WHAM segment bounds"
            )

        group_value = (
            noise_key,
            channel,
            base_start,
            float(row["final_snr_db"]),
            float(row["noise_gain"]),
        )
        previous = group_fields.setdefault(speech_key, group_value)
        if group_value != previous:
            raise ValueError(
                f"manifest rows for {speech_key} disagree on mixture construction"
            )

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        grouped.setdefault((str(row["speaker"]), str(row["speech_file"])), row)

    def validate_mixture(item: tuple[tuple[str, str], dict[str, Any]]) -> None:
        speech_key, row = item
        speech, _ = sf.read(speech_paths[speech_key])
        noise, _ = sf.read(
            noise_paths[str(row["noise_file"])],
            always_2d=True,
            start=int(row["noise_base_start"]),
            frames=len(speech),
        )
        noise = noise[:, int(row["noise_channel"])]
        meter = pyln.Meter(PAPER_SAMPLE_RATE)
        speech_loudness = meter.integrated_loudness(speech)
        noise_loudness = meter.integrated_loudness(noise)
        final_snr = float(row["snr_db"])
        gain = 10.0 ** ((speech_loudness - final_snr - noise_loudness) / 20.0)
        mixture = speech + gain * noise
        while np.max(np.abs(mixture)) >= 1.0:
            final_snr += 1.0
            gain = 10.0 ** ((speech_loudness - final_snr - noise_loudness) / 20.0)
            mixture = speech + gain * noise
        if final_snr != float(row["final_snr_db"]) or not math.isclose(
            gain, float(row["noise_gain"]), rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError(
                f"raw audio content is incompatible with the fixed manifest for "
                f"{speech_key}: reconstructed final_snr/noise_gain "
                f"{final_snr}/{gain} != "
                f"{row['final_snr_db']}/{row['noise_gain']}"
            )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in tqdm(
            executor.map(validate_mixture, grouped.items()),
            total=len(grouped),
            desc="validate mixture metadata",
        ):
            pass


def generate_dataset(
    rows: list[dict[str, Any]],
    speech_paths: dict[tuple[str, str], Path],
    noise_paths: dict[str, Path],
    out_dir: Path,
    workers: int,
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["speaker"]), str(row["speech_file"]))].append(row)

    clean_dir = out_dir / "clean"
    noisy_dir = out_dir / "noisy"
    source_dir = out_dir / "source"
    for directory in (clean_dir, noisy_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=False)

    def mix_group(item: tuple[tuple[str, str], list[dict[str, Any]]]) -> int:
        speech_key, group = item
        first = group[0]
        speech, speech_rate = sf.read(speech_paths[speech_key])
        noise, noise_rate = sf.read(
            noise_paths[str(first["noise_file"])],
            always_2d=True,
            start=int(first["noise_base_start"]),
            frames=len(speech),
        )
        if speech_rate != PAPER_SAMPLE_RATE or noise_rate != PAPER_SAMPLE_RATE:
            raise ValueError(f"sample-rate changed after validation for {speech_key}")
        if len(noise) != len(speech):
            raise ValueError(f"short WHAM read for {speech_key}")

        noise_channel = noise[:, int(first["noise_channel"])]
        mixture = speech + float(first["noise_gain"]) * noise_channel
        for row in group:
            start = int(row["speech_start"])
            end = int(row["speech_end"])
            clean_segment = paper_compatible_slice(speech, start, end)
            noisy_segment = paper_compatible_slice(mixture, start, end)
            expected = int(row["duration_frames"])
            if len(clean_segment) != expected or len(noisy_segment) != expected:
                raise RuntimeError(
                    f"generated length mismatch for {row['sample_id']}: "
                    f"{len(clean_segment)}/{len(noisy_segment)} != {expected}"
                )
            identifier = str(row["sample_id"])
            sf.write(
                clean_dir / identifier,
                clean_segment,
                PAPER_SAMPLE_RATE,
                subtype="FLOAT",
            )
            sf.write(
                noisy_dir / identifier,
                noisy_segment,
                PAPER_SAMPLE_RATE,
                subtype="FLOAT",
            )
        return len(group)

    generated = 0
    items = list(grouped.items())
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for count in tqdm(
            executor.map(mix_group, items),
            total=len(items),
            desc="reconstruct paper data",
        ):
            generated += count
    if generated != len(rows):
        raise RuntimeError(f"generated {generated} files, expected {len(rows)}")


def write_metadata(
    rows: list[dict[str, Any]], manifest_path: Path, out_dir: Path
) -> None:
    metadata = [
        f"{Path(str(row['sample_id'])).stem} {row['noise_file']} "
        f"{float(row['final_snr_db']):.6f}"
        for row in rows
    ]
    validation_speakers = sorted(
        {str(row["speaker"]) for row in rows if row["training_role"] == "validation"}
    )
    (out_dir / "metadata.txt").write_text("\n".join(metadata) + "\n", encoding="utf-8")
    (out_dir / "validation_speakers.txt").write_text(
        "\n".join(validation_speakers) + "\n", encoding="utf-8"
    )
    shutil.copyfile(manifest_path, out_dir / "subset_manifest.json")


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.out_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.out_dir}")

    _, rows = load_manifest(args.selection_manifest)
    validate_manifest(rows)
    speech_paths, noise_paths = resolve_inputs(rows, args.ears_dir, args.wham_dir)
    validate_audio_inputs(rows, speech_paths, noise_paths, args.workers)

    args.out_dir.mkdir(parents=True)
    try:
        generate_dataset(rows, speech_paths, noise_paths, args.out_dir, args.workers)
        write_metadata(rows, args.selection_manifest, args.out_dir)
    except Exception:
        shutil.rmtree(args.out_dir)
        raise

    role_counts = Counter(str(row["training_role"]) for row in rows)
    print(
        json.dumps(
            {
                "output": str(args.out_dir.resolve()),
                "sample_rate": PAPER_SAMPLE_RATE,
                "train": role_counts["train"],
                "held_out": role_counts["validation"],
                "source_outputs_generated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
