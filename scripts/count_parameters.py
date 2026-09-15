#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.measure_rtf import build_cmgan, build_crm


def count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Count frozen CMGAN/CRM parameters.")
    parser.add_argument("--cmgan-checkpoint", type=Path, required=True)
    parser.add_argument("--crm-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    cmgan = build_cmgan(args.cmgan_checkpoint, device)
    crm_args = argparse.Namespace(
        memory_prototypes=64,
        readout_mass=0.95,
        novelty_patience=2,
        candidate_threshold=0.7,
        immediate_novelty_threshold=0.7,
        merge_threshold=0.05,
        retirement_horizon=20,
    )
    crm, _ = build_crm(args.crm_checkpoint, device, crm_args)
    state = torch.load(args.crm_checkpoint, map_location="cpu", weights_only=True)
    static = count(crm.input) + count(crm.blocks) + count(crm.output)
    dynamic_active = count(crm.delta_basis) + count(crm.delta_coordinates)
    report = {
        "cmgan_generator": count(cmgan.generator),
        "static": static,
        "dynamic_active": dynamic_active,
        "functional_crm_extra": static + dynamic_active,
        "full_dynamic_checkpoint_elements": sum(value.numel() for value in state.values()),
        "test_time_trainable": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
