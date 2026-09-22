from pathlib import Path

import pandas as pd
import pytest

from metrics.evaluate import METRIC_COLUMNS, collect_jobs, validate_results


def metric_frame(filenames: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Filename": filename, **{metric: 1.0 for metric in METRIC_COLUMNS}}
            for filename in filenames
        ]
    )


def test_collect_jobs_requires_every_manifest_file(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    noisy = tmp_path / "noisy"
    denoised = tmp_path / "denoised"
    for directory in (clean, noisy, denoised):
        directory.mkdir()
    (clean / "a.wav").touch()
    (noisy / "a.wav").touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filename\na.wav\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="a.wav \\(denoised\\)"):
        collect_jobs(clean, noisy, denoised, manifest, 16_000)


def test_collect_jobs_rejects_duplicate_manifest_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("filename\na.wav\na.wav\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate filename"):
        collect_jobs(tmp_path, tmp_path, tmp_path, manifest, 16_000)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("PESQ", float("nan"), "non-finite PESQ"),
        ("STOI", float("inf"), "non-finite STOI"),
    ],
)
def test_validate_results_rejects_non_finite_metrics(
    column: str, value: float, message: str
) -> None:
    frame = metric_frame(["a.wav"])
    frame.loc[0, column] = value

    with pytest.raises(RuntimeError, match=message):
        validate_results(frame, ["a.wav"], require_complete=True)


def test_validate_results_rejects_metric_errors_and_missing_rows() -> None:
    failed = metric_frame(["a.wav"])
    failed["metric_errors"] = ["PESQ failed"]
    with pytest.raises(RuntimeError, match="metric failure"):
        validate_results(failed, ["a.wav"], require_complete=True)

    with pytest.raises(RuntimeError, match="missing 1 sample"):
        validate_results(metric_frame(["a.wav"]), ["a.wav", "b.wav"], require_complete=True)


def test_validate_results_preserves_manifest_order() -> None:
    ordered = validate_results(
        metric_frame(["b.wav", "a.wav"]),
        ["a.wav", "b.wav"],
        require_complete=True,
    )
    assert ordered["Filename"].tolist() == ["a.wav", "b.wav"]


def test_collect_jobs_does_not_filter_manifest_rows(tmp_path: Path) -> None:
    for directory_name in ("clean", "noisy", "denoised"):
        directory = tmp_path / directory_name
        directory.mkdir()
        (directory / "a.wav").touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "filename,reference_text,duration_sec\na.wav,<NOISE>,61\n",
        encoding="utf-8",
    )

    jobs, names = collect_jobs(
        tmp_path / "clean",
        tmp_path / "noisy",
        tmp_path / "denoised",
        manifest,
        16_000,
    )

    assert names == ["a.wav"]
    assert len(jobs) == 1
