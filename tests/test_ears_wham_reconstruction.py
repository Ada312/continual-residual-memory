from __future__ import annotations

import json
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "data/prepare_ears_wham_from_raw.py"
SPEC = importlib.util.spec_from_file_location("prepare_ears_wham_from_raw", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

expected_segment_frames = MODULE.expected_segment_frames
paper_compatible_slice = MODULE.paper_compatible_slice
validate_manifest = MODULE.validate_manifest


def test_paper_compatible_final_segment_omits_last_sample() -> None:
    waveform = np.arange(8)

    segment = paper_compatible_slice(waveform, 3, -1)

    np.testing.assert_array_equal(segment, np.array([3, 4, 5, 6]))
    assert len(segment) == expected_segment_frames(len(waveform), 3, -1)


def test_fixed_training_manifest_is_valid_and_complete() -> None:
    manifest = json.loads(
        (ROOT / "data/manifests/training/subset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = manifest["rows"]

    validate_manifest(rows)

    assert Counter(row["training_role"] for row in rows) == {
        "train": 8_192,
        "validation": 632,
    }
    assert len({row["sample_id"] for row in rows}) == 8_824
