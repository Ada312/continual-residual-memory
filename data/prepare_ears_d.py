#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio.functional as AF
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
TEST_SPEAKERS = ["p102", "p103", "p104", "p105", "p106", "p107"]
EMOTIONS_STYLES = [
    "adoration", "amazement", "amusement", "anger", "confusion", "contentment",
    "cuteness", "desire", "disappointment", "disgust", "distress",
    "embarassment", "extasy", "fast", "fear", "guilt", "highpitch", "interest",
    "loud", "lowpitch", "neutral", "pain", "pride", "realization", "relief",
    "regular", "sadness", "serenity", "slow", "whisper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export deterministic EARS-W v1 and SETTA EARS-D test pairs. "
            "The EARS-W stage follows ears_benchmark v1; EARS-D follows "
            "SETTA datasets/ears_demand at 16 kHz."
        )
    )
    parser.add_argument("--ears-dir", type=Path, default=Path("data/ears_raw"))
    parser.add_argument(
        "--wham-dir", type=Path, default=Path("data/high_res_wham/audio")
    )
    parser.add_argument(
        "--test-files",
        type=Path,
        default=ROOT / "data/manifests/protocol/ears_benchmark_v1_test_files.json",
    )
    parser.add_argument(
        "--demand-index",
        type=Path,
        default=ROOT / "data/manifests/protocol/demand_16k_index.csv",
    )
    parser.add_argument(
        "--demand-dir", type=Path, default=Path("data/demand/16k")
    )
    parser.add_argument(
        "--audio-metadata-cache",
        type=Path,
        default=None,
        help="Optional sf.info cache; audio is inspected directly when omitted.",
    )
    parser.add_argument("--ears-w-out", type=Path, required=True)
    parser.add_argument("--ears-d-out", type=Path, default=None)
    parser.add_argument("--source-rate", type=int, default=48_000)
    parser.add_argument("--target-rate", type=int, default=16_000)
    parser.add_argument("--ears-w-min-snr", type=float, default=0.0)
    parser.add_argument("--ears-w-max-snr", type=float, default=20.0)
    parser.add_argument("--ears-d-min-snr", type=float, default=-2.5)
    parser.add_argument("--ears-d-max-snr", type=float, default=17.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite-existing-dir", action="store_true")
    return parser.parse_args()


def find_emotion_style(stem: str) -> str | None:
    lowered = stem.lower()
    return next((value for value in EMOTIONS_STYLES if value in lowered), None)


def build_ears_w_rows(args: argparse.Namespace) -> list[dict]:
    test_spec = json.loads(args.test_files.read_text(encoding="utf-8"))
    if list(test_spec) != TEST_SPEAKERS:
        raise RuntimeError(f"unexpected EARS test speakers: {list(test_spec)}")

    cached = (
        json.loads(args.audio_metadata_cache.read_text(encoding="utf-8"))["files"]
        if args.audio_metadata_cache is not None
        else {}
    )

    def metadata(path: str | Path) -> dict:
        key = str(path)
        if key in cached:
            return cached[key]
        info = sf.info(path)
        return {
            "frames": int(info.frames),
            "samplerate": int(info.samplerate),
            "channels": int(info.channels),
        }

    noise_files = glob(str(args.wham_dir / "*.wav"))
    if not noise_files:
        raise RuntimeError(f"no WHAM files in {args.wham_dir}")
    noise_info = {path: metadata(path) for path in noise_files}

    test_files = [
        str(args.ears_dir / speaker / f"{stem}.wav")
        for speaker in TEST_SPEAKERS
        for stem in test_spec[speaker]
    ]
    np.random.seed(42)
    np.random.shuffle(test_files)
    snr_bins = np.linspace(args.ears_w_min_snr, args.ears_w_max_snr, 13)
    counters = Counter({value: 0 for value in EMOTIONS_STYLES})
    rows: list[dict] = []

    for speech_path in test_files:
        path = Path(speech_path)
        speaker = path.parent.name
        stem = path.stem
        speech_frames = metadata(path)["frames"]
        selected_noise = str(np.random.choice(noise_files))
        selected_info = noise_info[selected_noise]
        if selected_info["samplerate"] != args.source_rate:
            raise RuntimeError(f"unexpected WHAM rate: {selected_noise}")
        channel = int(np.random.randint(0, selected_info["channels"]))

        for start, end in test_spec[speaker][stem]:
            length = (end - start) if end >= 0 else (speech_frames - start - 1)
            if length > 29 * args.source_rate:
                continue
            while selected_info["frames"] < length:
                selected_noise = str(np.random.choice(noise_files))
                selected_info = noise_info[selected_noise]
                channel = int(np.random.randint(0, selected_info["channels"]))
            noise_start = int(np.random.randint(selected_info["frames"] - length + 1))
            style = find_emotion_style(stem)
            if style is None:
                initial_snr = float(
                    np.round(
                        np.random.uniform(args.ears_w_min_snr, args.ears_w_max_snr), 1
                    )
                )
            else:
                index = counters[style] % 12
                counters[style] += 1
                initial_snr = float(
                    np.round(np.random.uniform(snr_bins[index], snr_bins[index + 1]), 1)
                )
            rows.append(
                {
                    "id": len(rows),
                    "speaker": speaker,
                    "speech_file": stem,
                    "speech_path": str(path),
                    "speech_start": int(start),
                    "speech_end": int(end),
                    "num_samples_48k": int(length),
                    "noise_file": Path(selected_noise).name,
                    "noise_path": selected_noise,
                    "noise_channel": channel,
                    "noise_start": noise_start,
                    "initial_snr_db": initial_snr,
                    "emotion_style": style,
                }
            )
    return rows[: args.limit] if args.limit is not None else rows


def resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    return AF.resample(tensor, source_rate, target_rate).numpy()


def export_ears_w(args: argparse.Namespace, rows: list[dict]) -> list[dict]:
    clean_dir = args.ears_w_out / "clean"
    noisy_dir = args.ears_w_out / "noisy"
    clean_16k_dir = args.ears_w_out / "clean_16k"
    for directory in (clean_dir, noisy_dir, clean_16k_dir):
        directory.mkdir(parents=True, exist_ok=args.overwrite_existing_dir)
    ramp_samples = int(0.010 * args.source_rate)
    ramp = np.linspace(0, 1, ramp_samples)

    def export(row: dict) -> dict:
        speech, speech_rate = sf.read(
            row["speech_path"],
            start=row["speech_start"],
            frames=row["num_samples_48k"],
        )
        noise, noise_rate = sf.read(
            row["noise_path"],
            always_2d=True,
            start=row["noise_start"],
            frames=row["num_samples_48k"],
        )
        if speech_rate != args.source_rate or noise_rate != args.source_rate:
            raise RuntimeError("EARS-W source sample-rate mismatch")
        noise = noise[:, row["noise_channel"]]
        meter = pyln.Meter(args.source_rate)
        speech_loudness = meter.integrated_loudness(speech)
        noise_loudness = meter.integrated_loudness(noise)
        snr = row["initial_snr_db"]
        gain = 10.0 ** ((speech_loudness - snr - noise_loudness) / 20.0)
        mixture = speech + gain * noise
        while np.max(np.abs(mixture)) >= 1.0:
            snr += 1.0
            gain = 10.0 ** ((speech_loudness - snr - noise_loudness) / 20.0)
            mixture = speech + gain * noise

        speech = speech.copy()
        mixture = mixture.copy()
        speech[:ramp_samples] *= ramp
        speech[-ramp_samples:] *= ramp[::-1]
        mixture[:ramp_samples] *= ramp
        mixture[-ramp_samples:] *= ramp[::-1]
        name = f'{row["speaker"]}_{row["id"]:05d}.wav'
        sf.write(clean_dir / name, speech, args.source_rate, subtype="FLOAT")
        sf.write(noisy_dir / name, mixture, args.source_rate, subtype="FLOAT")
        clean_16k = resample(speech, args.source_rate, args.target_rate)
        sf.write(clean_16k_dir / name, clean_16k, args.target_rate, subtype="FLOAT")
        return {
            **row,
            "filename": name,
            "final_snr_db": snr,
            "noise_gain": gain,
            "num_samples_16k": len(clean_16k),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        exported = list(
            tqdm(executor.map(export, rows), total=len(rows), desc="export EARS-W v1")
        )
    manifest = {
        "description": "ears_benchmark v1 complete EARS-W standard test split",
        "ears_benchmark_repository": "https://github.com/sp-uhh/ears_benchmark",
        "generation_seed": 42,
        "snr_range_db": [args.ears_w_min_snr, args.ears_w_max_snr],
        "source_sample_rate": args.source_rate,
        "clean_16k_resampler": "torchaudio.functional.resample",
        "pairs": len(exported),
        "speakers": TEST_SPEAKERS,
        "hours": sum(row["num_samples_48k"] for row in exported)
        / args.source_rate
        / 3600,
        "rows": exported,
    }
    (args.ears_w_out / "mix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return exported


def export_ears_d(args: argparse.Namespace, ears_w_rows: list[dict]) -> list[dict]:
    demand = pd.read_csv(args.demand_index)
    test_indices = demand.index[demand["split"] == "test"].to_numpy()
    if not np.array_equal(test_indices, np.arange(test_indices[0], test_indices[-1] + 1)):
        raise RuntimeError("SETTA EARS-D assumes a contiguous DEMAND test index")
    if set(demand.loc[test_indices, "dir"]) != {
        "SPSQUARE", "PCAFETER", "OMEETING", "OHALLWAY"
    }:
        raise RuntimeError("unexpected DEMAND test environments")

    ordered = sorted(ears_w_rows, key=lambda row: row["id"])
    np.random.seed(123)
    snrs = (
        np.random.rand(len(ordered))
        * (args.ears_d_max_snr - args.ears_d_min_snr)
        + args.ears_d_min_snr
    )
    selected = (
        np.random.rand(len(ordered)) * len(test_indices) + int(test_indices[0])
    ).astype(int)
    clean_dir = args.ears_d_out / "clean"
    noisy_dir = args.ears_d_out / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=args.overwrite_existing_dir)
    noisy_dir.mkdir(parents=True, exist_ok=args.overwrite_existing_dir)

    def export(item: tuple[int, dict]) -> dict:
        position, clean_row = item
        noise_row = demand.iloc[selected[position]]
        name = clean_row["filename"]
        clean, clean_rate = sf.read(args.ears_w_out / "clean_16k" / name)
        noise_path = args.demand_dir / noise_row["dir"] / noise_row["file"]
        noise, noise_rate = sf.read(
            noise_path,
            start=int(noise_row["start"]),
            frames=30 * args.target_rate,
            always_2d=True,
        )
        if clean_rate != args.target_rate or noise_rate != args.target_rate:
            raise RuntimeError("EARS-D sample-rate mismatch")
        noise = noise[:, 0][: len(clean)]
        if len(noise) != len(clean):
            raise RuntimeError(f"DEMAND segment too short for {name}")
        meter = pyln.Meter(args.target_rate)
        clean_loudness = meter.integrated_loudness(clean)
        snr = float(snrs[position])
        gain = 10.0 ** (
            (clean_loudness - snr - float(noise_row["loudness"])) / 20.0
        )
        noisy = clean + gain * noise
        sf.write(clean_dir / name, clean, args.target_rate, subtype="FLOAT")
        sf.write(noisy_dir / name, noisy, args.target_rate, subtype="FLOAT")
        return {
            "position": position,
            "filename": name,
            "speaker": clean_row["speaker"],
            "snr_db": snr,
            "clean_loudness_lufs": clean_loudness,
            "noise_index": int(selected[position]),
            "noise_environment": noise_row["dir"],
            "noise_file": noise_row["file"],
            "noise_start_16k": int(noise_row["start"]),
            "indexed_noise_loudness_lufs": float(noise_row["loudness"]),
            "noise_gain": gain,
            "num_samples": len(clean),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        exported = list(
            tqdm(
                executor.map(export, enumerate(ordered)),
                total=len(ordered),
                desc="export SETTA EARS-D",
            )
        )
    manifest = {
        "description": "SETTA-aligned complete EARS-D standard test split",
        "setta_repository": "https://github.com/tobiaaa/SETTA",
        "construction": (
            "official ears_benchmark v1 p102-p107 clean test; SETTA "
            "datasets/ears_demand/index.py and dataset.py; seed 123; DEMAND "
            "SPSQUARE/PCAFETER/OMEETING/OHALLWAY; -2.5 to 17.5 dB LUFS SNR"
        ),
        "ordering_policy": (
            "official ears_benchmark v1 global output id; SETTA did not publish its "
            "generated test_index.csv and used unsorted os.listdir"
        ),
        "sample_rate": args.target_rate,
        "generation_seed": 123,
        "snr_range_db": [args.ears_d_min_snr, args.ears_d_max_snr],
        "pairs": len(exported),
        "speakers": TEST_SPEAKERS,
        "noise_environments": sorted({row["noise_environment"] for row in exported}),
        "snr_min": min(row["snr_db"] for row in exported),
        "snr_max": max(row["snr_db"] for row in exported),
        "hours": sum(row["num_samples"] for row in exported)
        / args.target_rate
        / 3600,
        "rows": exported,
    }
    (args.ears_d_out / "mix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return exported


def main() -> None:
    args = parse_args()
    rows = build_ears_w_rows(args)
    ears_w = export_ears_w(args, rows)
    ears_d = export_ears_d(args, ears_w) if args.ears_d_out is not None else []
    print(
        json.dumps(
            {
                "ears_w_pairs": len(ears_w),
                "ears_d_pairs": len(ears_d),
                "speakers": TEST_SPEAKERS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
