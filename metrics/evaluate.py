#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

if __package__:
    from ._frozen.composite import Composite
    from ._frozen.pesq import PESQ
    from ._frozen.sisdr import SISDR
    from ._frozen.ssnr import SSNR
    from ._frozen.stoi import STOI
else:
    from _frozen.composite import Composite
    from _frozen.pesq import PESQ
    from _frozen.sisdr import SISDR
    from _frozen.ssnr import SSNR
    from _frozen.stoi import STOI


_WORKER_METRICS = None
METRIC_COLUMNS = ["PESQ", "STOI", "C_sig", "C_bak", "C_ovl", "SSNR", "SISDR"]


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
    if (
        n < fs // 4
        or torch.max(torch.abs(x_clean)).item() < 1e-7
        or torch.max(torch.abs(x_denoised)).item() < 1e-7
    ):
        for name in METRIC_COLUMNS:
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
        for name in METRIC_COLUMNS:
            row[name] = float("nan")
        return row
    finally:
        signal.alarm(0)


def collect_jobs(
    clean_dir: Path,
    noisy_dir: Path,
    denoised_dir: Path,
    references: Path | None,
    fs: int,
) -> tuple[list[tuple[Path, Path, Path, int]], list[str]]:
    if references is not None:
        refs = pd.read_csv(references)
        if "filename" not in refs.columns:
            raise ValueError("reference manifest must contain a filename column")
        names = refs["filename"].astype(str).tolist()
        duplicates = (
            refs.loc[refs["filename"].duplicated(), "filename"].astype(str).tolist()
        )
        if duplicates:
            raise ValueError(f"duplicate filename in reference manifest: {duplicates[0]}")
    else:
        names = sorted(p.name for p in clean_dir.glob("*.wav"))
    if not names:
        raise RuntimeError("No evaluation samples were selected.")

    jobs = []
    missing: list[str] = []
    for name in names:
        clean_path = clean_dir / name
        noisy_path = noisy_dir / name
        denoised_path = denoised_dir / name
        absent = [
            label
            for label, path in (
                ("clean", clean_path),
                ("noisy", noisy_path),
                ("denoised", denoised_path),
            )
            if not path.is_file()
        ]
        if absent:
            missing.append(f"{name} ({', '.join(absent)})")
        else:
            jobs.append((clean_path, noisy_path, denoised_path, fs))
    if missing:
        raise FileNotFoundError(
            f"missing evaluation files for {len(missing)} sample(s); first: {missing[0]}"
        )
    return jobs, names


def validate_results(
    frame: pd.DataFrame,
    expected_names: list[str],
    *,
    require_complete: bool,
) -> pd.DataFrame:
    required = {"Filename", *METRIC_COLUMNS}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"metric CSV is missing columns: {missing_columns}")
    if frame["Filename"].duplicated().any():
        duplicate = frame.loc[frame["Filename"].duplicated(), "Filename"].iloc[0]
        raise RuntimeError(f"duplicate metric row for {duplicate}")

    expected = set(expected_names)
    actual = set(frame["Filename"].astype(str))
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected:
        raise RuntimeError(f"metric CSV contains unexpected sample: {unexpected[0]}")
    if require_complete and missing:
        raise RuntimeError(
            f"metric CSV is missing {len(missing)} sample(s); first: {missing[0]}"
        )

    if "metric_errors" in frame.columns:
        errors = frame["metric_errors"].fillna("").astype(str).str.strip()
        if errors.ne("").any():
            failed = frame.loc[errors.ne(""), "Filename"].iloc[0]
            detail = errors[errors.ne("")].iloc[0]
            raise RuntimeError(f"metric failure for {failed}: {detail}")

    numeric = frame[METRIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy())
    if not finite.all():
        row_index, column_index = np.argwhere(~finite)[0]
        filename = frame.iloc[int(row_index)]["Filename"]
        metric = METRIC_COLUMNS[int(column_index)]
        raise RuntimeError(f"non-finite {metric} result for {filename}")

    order = {name: index for index, name in enumerate(expected_names)}
    return frame.assign(
        _test_order=frame["Filename"].astype(str).map(order)
    ).sort_values("_test_order").drop(columns="_test_order").reset_index(drop=True)


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
    jobs, expected_names = collect_jobs(
        args.clean_dir, args.noisy_dir, args.denoised_dir, args.references, args.fs
    )

    per_file_path = args.out_dir / f"{args.method}_per_file_metrics.csv"
    completed: set[str] = set()
    if per_file_path.exists():
        completed_df = validate_results(
            pd.read_csv(per_file_path), expected_names, require_complete=False
        )
        completed = set(completed_df["Filename"].astype(str).tolist())
    if completed:
        jobs = [job for job in jobs if job[0].name not in completed]

    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.fs,),
    ) as ex:
        for row in tqdm(ex.map(score_job, jobs), total=len(jobs), desc=args.method):
            validated = validate_results(
                pd.DataFrame([row]), [str(row["Filename"])], require_complete=True
            )
            row = validated.iloc[0].to_dict()
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
    df = validate_results(df, expected_names, require_complete=True)
    summary = df.mean(numeric_only=True).to_dict()
    summary["Method"] = args.method
    summary["num_files"] = int(len(df))

    df.to_csv(per_file_path, index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(args.out_dir / f"{args.method}_summary_metrics.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
