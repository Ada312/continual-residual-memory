from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "data/prepare_dns.py"
SPEC = importlib.util.spec_from_file_location("prepare_dns", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

index_by_file_id = MODULE.index_by_file_id
resolve_required_pairs = MODULE.resolve_required_pairs


def touch(directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.touch()
    return path


def test_descriptive_noisy_filename_matches_file_id(tmp_path: Path) -> None:
    noisy = touch(
        tmp_path / "noisy",
        "clnsp126_3Wjw0nadnM4_snr15_tl-22_fileid_0.wav",
    )

    assert index_by_file_id(tmp_path / "noisy") == {0: noisy}


def test_missing_required_file_id_fails(tmp_path: Path) -> None:
    clean = touch(tmp_path / "clean", "clean_fileid_0.wav")

    with pytest.raises(FileNotFoundError, match="missing noisy file ID 0"):
        resolve_required_pairs([("dns2020_no_reverb_fileid_0.wav", 0)], {0: clean}, {})


def test_duplicate_matching_file_id_fails(tmp_path: Path) -> None:
    noisy_dir = tmp_path / "noisy"
    touch(noisy_dir, "first_fileid_7.wav")
    touch(noisy_dir, "second_fileid_7.wav")

    with pytest.raises(ValueError, match="duplicate file ID 7"):
        index_by_file_id(noisy_dir)


def test_clean_and_noisy_ids_cannot_be_silently_mismatched(tmp_path: Path) -> None:
    clean = touch(tmp_path / "clean", "clean_fileid_3.wav")
    noisy = touch(tmp_path / "noisy", "description_fileid_4.wav")

    with pytest.raises(FileNotFoundError, match="missing noisy file ID 3"):
        resolve_required_pairs(
            [("dns2020_no_reverb_fileid_3.wav", 3)],
            {3: clean},
            {4: noisy},
        )
