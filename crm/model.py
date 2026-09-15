"""Paper-facing CRM interface: M_(t-1) read, infer, then M_t write."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .memory import PrototypeMemory
from .recovery import UtteranceLocalRecovery
from .refinement import MemoryConditionedRefinement


def load_recovery(checkpoint: Path, config: dict, device: torch.device):
    projector = UtteranceLocalRecovery(
        channels=int(config["channels"]),
        max_gain=float(config["max_gain"]),
        initial_gain=float(config["initial_gain"]),
    )
    projector.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return projector.to(device).eval().requires_grad_(False)


def load_refinement(checkpoint: Path, config: dict, device: torch.device):
    """Load final state_dict unchanged, including the inactive legacy memory_adapter."""
    projector = MemoryConditionedRefinement(
        channels=int(config["channels"]),
        max_gain=float(config["max_gain"]),
        initial_gain=float(config["initial_gain"]),
        memory_warmup=int(config["memory_warmup"]),
        prototypes=int(config["k_max"]),
        posterior_temperature=float(config["tau_mem"]),
        reliability_power=float(config["reliability_power"]),
        posterior_confidence_power=float(config["posterior_confidence_power"]),
        readout_mass=float(config["gamma_read"]),
        delta_rank=int(config["d"]),
        memory_delta_gain=float(config["alpha_dyn"]),
    )
    projector.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return projector.to(device).eval().requires_grad_(False)


def new_memory(config: dict) -> PrototypeMemory:
    return PrototypeMemory(
        prototypes=int(config["k_max"]),
        frequency_bins=int(config["n_fft"]) // 2 + 1,
        noise_frame_fraction=float(config["noise_frame_fraction"]),
        warmup_utterances=int(config["memory_warmup"]),
        novelty_threshold=float(config["novelty_threshold"]),
        novelty_patience=int(config["novelty_patience"]),
        candidate_threshold=float(config["candidate_threshold"]),
        immediate_novelty_threshold=float(config["immediate_novelty_threshold"]),
        merge_threshold=float(config["merge_threshold"]),
        retirement_horizon=int(config["retirement_horizon"]),
    )


class CRM:
    """Maintain prototype state; separate inference from post-output writing."""

    def __init__(self, projector: MemoryConditionedRefinement, memory: PrototypeMemory):
        self.projector = projector
        self.memory = memory

    @torch.inference_mode()
    def infer(self, noisy_waveform: torch.Tensor, backbone_estimate: torch.Tensor):
        memory_prev = self.memory.context()
        device = noisy_waveform.device
        x_dyn, g_dyn, auxiliary = self.projector(
            noisy_waveform.unsqueeze(0),
            backbone_estimate.unsqueeze(0),
            memory_prev.means.unsqueeze(0),
            memory_prev.variances.unsqueeze(0),
            memory_prev.counts.to(device=device, dtype=torch.float32).unsqueeze(0),
            external_memory_strength=1.0,
        )
        return x_dyn, g_dyn, auxiliary

    @torch.inference_mode()
    def write(self, auxiliary: dict[str, torch.Tensor]) -> int:
        """Call only after the enhanced output has been produced/saved."""
        return self.memory.update(
            auxiliary["source_spectrum"][0], auxiliary["residual_spectrum"][0]
        )


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
