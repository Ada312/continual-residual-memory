#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono_float32(path: Path, expected_sr: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != expected_sr:
        raise ValueError(f"{path}: expected {expected_sr} Hz, got {sample_rate}")
    return audio.mean(axis=1, dtype=np.float32)


def tile_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if not len(audio):
        raise ValueError("cannot tile empty audio")
    repeats = (length + len(audio) - 1) // len(audio)
    return np.tile(audio, repeats)[:length]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the frozen LibriSpeech test-clean + MUSAN Music set."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--librispeech-root", type=Path, required=True)
    parser.add_argument("--musan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    clean_dir = args.output_root / "clean"
    noisy_dir = args.output_root / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["test_order"]))
    if args.limit is not None:
        rows = rows[: args.limit]

    for row in rows:
        sample_rate = int(row["sample_rate"])
        length = int(row["num_samples"])
        clean = load_mono_float32(
            args.librispeech_root / row["librispeech_relpath"], sample_rate
        )
        clean = clean[:length]
        if len(clean) != length:
            raise ValueError(f'{row["utt_id"]}: short clean input')

        music = load_mono_float32(args.musan_root / row["music_relpath"], sample_rate)
        offset = int(row["crop_offset"])
        if row["music_tiled"].lower() == "true":
            segment = tile_to_length(music, offset + length)[offset : offset + length]
        else:
            segment = music[offset : offset + length]
        if len(segment) != length:
            raise ValueError(f'{row["utt_id"]}: short music segment')

        # The frozen generator centered each selected segment in float32.
        segment = segment - segment.mean(dtype=np.float32)
        noisy = (
            clean + np.float32(row["music_scale"]) * segment
        ) * np.float32(row["global_gain"])

        clean_path = clean_dir / f'{row["utt_id"]}.wav'
        noisy_path = noisy_dir / f'{row["utt_id"]}.wav'
        sf.write(clean_path, np.clip(clean, -1.0, 1.0), sample_rate, subtype="PCM_16")
        sf.write(noisy_path, np.clip(noisy, -1.0, 1.0), sample_rate, subtype="PCM_16")

    print(f"generated {len(rows)} pairs under {args.output_root}")


if __name__ == "__main__":
    main()
