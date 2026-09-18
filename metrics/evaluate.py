#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from _frozen.composite import Composite
from _frozen.pesq import PESQ
from _frozen.sisdr import SISDR
from _frozen.ssnr import SSNR
from _frozen.stoi import STOI


_WORKER_METRICS = None


def metric_cfg(fs: int):
    return SimpleNamespace(data=SimpleNamespace(fs=fs))


def make_metrics(fs: int):
    cfg = metric_cfg(fs)
    device = torch.device("cpu")
    return [
        PESQ(cfg, device),
        STOI(cfg, device),
        Composite(cfg, device),
        SSNR(cfg, device),
        SISDR(cfg, device),
    ]


def init_worker(fs: int) -> None:
    global _WORKER_METRICS
    torch.set_num_threads(1)
    _WORKER_METRICS = make_metrics(fs)


def load_mono(path: Path, fs: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != fs:
        wav = torchaudio.functional.resample(wav, sr, fs)
    return wav.squeeze(0).unsqueeze(0)


def score_file(clean_path: Path, noisy_path: Path, denoised_path: Path, fs: int) -> dict:
    x_clean = load_mono(clean_path, fs)
    x_noisy = load_mono(noisy_path, fs)
    x_denoised = load_mono(denoised_path, fs)
    n = min(x_clean.shape[-1], x_noisy.shape[-1], x_denoised.shape[-1])
    x_clean = x_clean[..., :n]
    x_noisy = x_noisy[..., :n]
    x_denoised = x_denoised[..., :n]

    row = {"Filename": clean_path.name}
    metric_names = ["PESQ", "STOI", "CSIG", "CBAK", "COVL", "SSNR", "SISDR"]
    if (
        n < fs // 4
        or torch.max(torch.abs(x_clean)).item() < 1e-7
        or torch.max(torch.abs(x_denoised)).item() < 1e-7
    ):
        for name in metric_names:
            row[name] = float("nan")
        row["metric_errors"] = "skipped before metrics: too short or near-silent audio"
        return row

    metrics = _WORKER_METRICS if _WORKER_METRICS is not None else make_metrics(fs)
    for metric in metrics:
        names = metric.names()
        try:
            result = metric(x_clean, x_noisy, x_denoised, {}).squeeze()
            if len(names) == 1:
                row[names[0]] = result.item()
            else:
                for name, value in zip(names, result):
                    row[name] = value.item()
        except Exception as exc:
            for name in names:
                row[name] = float("nan")
            row.setdefault("metric_errors", [])
            row["metric_errors"].append(f"{metric.__class__.__name__}: {exc}")
    if isinstance(row.get("metric_errors"), list):
        row["metric_errors"] = " | ".join(row["metric_errors"])
    return row


def score_job(args):
    def timeout_handler(signum, frame):
        raise TimeoutError("metric computation timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)
    try:
        return score_file(*args)
    except Exception as exc:
        clean_path = args[0]
        row = {"Filename": clean_path.name, "metric_errors": str(exc)}
        for name in ["PESQ", "STOI", "CSIG", "CBAK", "COVL", "SSNR", "SISDR"]:
            row[name] = float("nan")
        return row
    finally:
        signal.alarm(0)


def collect_jobs(clean_dir: Path, noisy_dir: Path, denoised_dir: Path, references: Path | None, fs: int):
    if references is not None:
        refs = pd.read_csv(references)
        if "reference_text" in refs.columns:
            special = {"<OTHER>", "<NOISE>", "<SIL>", "<MUSIC>"}
            refs = refs[~refs["reference_text"].astype(str).str.strip().isin(special)].copy()
        if "duration_sec" in refs.columns:
            refs = refs[refs["duration_sec"] <= 60].copy()
        names = refs["filename"].tolist()
    else:
        names = sorted(p.name for p in clean_dir.glob("*.wav"))

    jobs = []
    for name in names:
        clean_path = clean_dir / name
        noisy_path = noisy_dir / name
        denoised_path = denoised_dir / name
        if clean_path.exists() and noisy_path.exists() and denoised_path.exists():
            jobs.append((clean_path, noisy_path, denoised_path, fs))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PESQ/STOI/Composite/SSNR/SISDR for an existing audio directory.")
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--denoised-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--references", type=Path, default=None)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=min(max(os.cpu_count() or 1, 1), 16))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = collect_jobs(args.clean_dir, args.noisy_dir, args.denoised_dir, args.references, args.fs)
    if not jobs:
        raise RuntimeError("No matching clean/noisy/denoised wav triplets found.")

    per_file_path = args.out_dir / f"{args.method}_per_file_metrics.csv"
    completed: set[str] = set()
    if per_file_path.exists():
        try:
            completed_df = pd.read_csv(per_file_path, usecols=["Filename"])
            completed = set(completed_df["Filename"].astype(str).tolist())
        except Exception:
            completed = set()
    if completed:
        jobs = [job for job in jobs if job[0].name not in completed]

    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.fs,),
    ) as ex:
        for row in tqdm(ex.map(score_job, jobs), total=len(jobs), desc=args.method):
            rows.append(row)
            pd.DataFrame([row]).to_csv(
                per_file_path,
                mode="a",
                index=False,
                header=not per_file_path.exists(),
            )

    if per_file_path.exists():
        df = pd.read_csv(per_file_path)
    else:
        df = pd.DataFrame(rows)
    summary = df.mean(numeric_only=True).to_dict()
    summary["Method"] = args.method
    summary["num_files"] = int(len(df))

    df.to_csv(per_file_path, index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(args.out_dir / f"{args.method}_summary_metrics.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
