from __future__ import annotations

import math

import torch
import torch.nn as nn

from .recovery import ResidualSpeechProjector


class PrototypeContinualResidualSpeechProjector(ResidualSpeechProjector):
    """Memory readout and context maps Z_t for the final refinement branch."""

    memory_channels = 6

    def __init__(
        self,
        channels: int = 32,
        max_gain: float = 0.1,
        initial_gain: float = 0.005,
        n_fft: int = 512,
        hop_length: int = 128,
        compression: float = 0.3,
        memory_warmup: int = 20,
        prototypes: int = 4,
        posterior_temperature: float = 0.5,
        reliability_power: float = 0.0,
        posterior_confidence_power: float = 0.0,
        readout_prototypes: int | None = None,
        readout_mass: float | None = None,
    ):
        super().__init__(
            channels=channels,
            max_gain=max_gain,
            initial_gain=initial_gain,
            n_fft=n_fft,
            hop_length=hop_length,
            compression=compression,
        )
        self.memory_warmup = int(memory_warmup)
        self.prototypes = int(prototypes)
        self.posterior_temperature = float(posterior_temperature)
        self.reliability_power = float(reliability_power)
        if not 0.0 <= posterior_confidence_power <= 1.0:
            raise ValueError("posterior_confidence_power must be in [0, 1]")
        self.posterior_confidence_power = float(posterior_confidence_power)
        if readout_prototypes is not None and readout_prototypes < 1:
            raise ValueError("readout_prototypes must be positive")
        self.readout_prototypes = (
            None if readout_prototypes is None else int(readout_prototypes)
        )
        if readout_mass is not None and not 0.0 < readout_mass <= 1.0:
            raise ValueError("readout_mass must be in (0, 1]")
        if readout_prototypes is not None and readout_mass is not None:
            raise ValueError(
                "readout_prototypes and readout_mass are mutually exclusive"
            )
        self.readout_mass = None if readout_mass is None else float(readout_mass)
        self.memory_adapter = nn.Sequential(
            nn.Conv2d(self.memory_channels, channels, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1, bias=False),
        )
        nn.init.normal_(self.memory_adapter[-1].weight, mean=0.0, std=1e-3)

    def initialize_from_static(self, static: ResidualSpeechProjector) -> None:
        if self.input.out_channels != static.input.out_channels:
            raise ValueError("static and prototype projectors need the same channel width")
        self.input.load_state_dict(static.input.state_dict())
        self.blocks.load_state_dict(static.blocks.state_dict())
        self.output.load_state_dict(static.output.state_dict())
        nn.init.normal_(self.memory_adapter[-1].weight, mean=0.0, std=1e-3)

    def _memory_features(
        self,
        residual_spectrum: torch.Tensor,
        memory_means: torch.Tensor,
        memory_variances: torch.Tensor,
        memory_counts: torch.Tensor,
        prototypes: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, frequency, time = residual_spectrum.shape
        means = memory_means.to(residual_spectrum.device).reshape(
            batch, prototypes, frequency
        )
        variances = memory_variances.to(residual_spectrum.device).reshape(
            batch, prototypes, frequency
        )
        counts = memory_counts.to(residual_spectrum.device).reshape(
            batch, prototypes
        )
        active = counts > 0
        available = active.any(dim=1)

        log_power = torch.log(residual_spectrum.abs().square().clamp_min(1e-8))
        observation = log_power.median(dim=-1).values
        centered_observation = observation - observation.mean(dim=1, keepdim=True)
        centered_means = means - means.mean(dim=2, keepdim=True)
        shape_distance = (
            (centered_means - centered_observation.unsqueeze(1)).square()
            / (variances + 0.5)
        ).mean(dim=2)
        level_distance = (
            means.mean(dim=2) - observation.mean(dim=1, keepdim=True)
        ).square()
        distances = shape_distance + 0.15 * level_distance
        distances = distances.masked_fill(~active, 1e4)
        if (
            self.readout_prototypes is not None
            and self.readout_prototypes < prototypes
        ):
            nearest = distances.topk(
                k=self.readout_prototypes, dim=1, largest=False
            ).indices
            readout_mask = torch.zeros_like(active)
            readout_mask.scatter_(1, nearest, True)
            active = active & readout_mask
            distances = distances.masked_fill(~active, 1e4)
        elif self.readout_mass is not None and self.readout_mass < 1.0:
            preliminary_logits = -distances / max(
                self.posterior_temperature, 1e-3
            )
            preliminary = torch.softmax(preliminary_logits, dim=1)
            preliminary = preliminary * active.to(preliminary.dtype)
            preliminary = preliminary / preliminary.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            sorted_probability, sorted_indices = preliminary.sort(
                dim=1, descending=True
            )
            cumulative = sorted_probability.cumsum(dim=1)
            keep_sorted = (
                cumulative - sorted_probability
            ) < self.readout_mass
            readout_mask = torch.zeros_like(active)
            readout_mask.scatter_(1, sorted_indices, keep_sorted)
            active = active & readout_mask
            distances = distances.masked_fill(~active, 1e4)
        logits = -distances / max(self.posterior_temperature, 1e-3)
        posterior = torch.softmax(logits, dim=1) * active.to(logits.dtype)
        posterior = posterior / posterior.sum(dim=1, keepdim=True).clamp_min(1e-8)

        aggregate_mean = (posterior.unsqueeze(-1) * means).sum(dim=1)
        aggregate_variance = (
            posterior.unsqueeze(-1)
            * (variances + (means - aggregate_mean.unsqueeze(1)).square())
        ).sum(dim=1)
        total_count = counts.sum(dim=1, keepdim=True)
        total_confidence = (
            torch.log1p(total_count)
            / math.log1p(max(1, self.memory_warmup))
        ).clamp(0.0, 1.0)
        inverse_effective_count = (
            posterior.square() / counts.clamp_min(1.0)
        ).sum(dim=1, keepdim=True)
        effective_sample_count = torch.where(
            available.unsqueeze(1),
            inverse_effective_count.clamp_min(1e-8).reciprocal(),
            torch.zeros_like(inverse_effective_count),
        )
        posterior_confidence = (
            torch.log1p(effective_sample_count)
            / math.log1p(max(1, self.memory_warmup))
        ).clamp(0.0, 1.0)
        confidence_ratio = torch.where(
            total_confidence > 0,
            posterior_confidence / total_confidence.clamp_min(1e-8),
            torch.zeros_like(total_confidence),
        ).clamp(0.0, 1.0)
        confidence = total_confidence * confidence_ratio.pow(
            self.posterior_confidence_power
        )
        active_count = active.sum(dim=1, keepdim=True).clamp_min(1)
        entropy = -(posterior * torch.log(posterior.clamp_min(1e-8))).sum(dim=1, keepdim=True)
        entropy = entropy / torch.log(active_count.to(logits.dtype)).clamp_min(1.0)
        minimum_distance = distances.min(dim=1, keepdim=True).values
        similarity = torch.exp(-minimum_distance).clamp(0.0, 1.0)
        mean_distance = (
            distances.masked_fill(~active, 0.0).sum(dim=1, keepdim=True)
            / active_count.to(distances.dtype)
        )
        distance_variance = (
            (distances - mean_distance).square().masked_fill(~active, 0.0)
            .sum(dim=1, keepdim=True)
            / active_count.to(distances.dtype)
        )
        relative_distance_margin = (
            (mean_distance - minimum_distance)
            / distance_variance.sqrt().clamp_min(1e-4)
        )
        relative_distance_margin = relative_distance_margin * (
            active_count > 1
        ).to(relative_distance_margin.dtype)
        posterior_peak = posterior.max(dim=1, keepdim=True).values
        uniform_peak = active_count.to(posterior.dtype).reciprocal()
        normalized_posterior_peak = torch.where(
            active_count > 1,
            (posterior_peak - uniform_peak)
            / (1.0 - uniform_peak).clamp_min(1e-4),
            torch.zeros_like(posterior_peak),
        ).clamp(0.0, 1.0)

        profile_shape = aggregate_mean - aggregate_mean.mean(dim=1, keepdim=True)
        profile_shape = (profile_shape / 4.0).clamp(-2.0, 2.0)
        deviation = (log_power - aggregate_mean.unsqueeze(-1)) / (
            aggregate_variance.sqrt().unsqueeze(-1) + 0.5
        )
        deviation = (deviation / 4.0).clamp(-2.0, 2.0)
        stability = (1.0 / (aggregate_variance.sqrt() + 1.0)).clamp(0.0, 1.0)

        features = torch.stack(
            (
                profile_shape.unsqueeze(-1).expand(-1, -1, time),
                deviation,
                stability.unsqueeze(-1).expand(-1, -1, time),
                confidence.unsqueeze(-1).expand(-1, frequency, time),
                similarity.unsqueeze(-1).expand(-1, frequency, time),
                entropy.unsqueeze(-1).expand(-1, frequency, time),
            ),
            dim=1,
        )
        gate = available.to(features.dtype).reshape(batch, 1, 1, 1)
        features = features * gate * confidence.reshape(batch, 1, 1, 1)
        return features, {
            "prototype_posterior": posterior,
            "prototype_distance": distances,
            "memory_aggregate_mean": aggregate_mean,
            "memory_aggregate_variance": aggregate_variance,
            "memory_similarity": similarity,
            "memory_minimum_distance": minimum_distance,
            "memory_mean_active_distance": mean_distance,
            "memory_relative_distance_margin": relative_distance_margin,
            "memory_posterior_peak": posterior_peak,
            "memory_normalized_posterior_peak": normalized_posterior_peak,
            "memory_entropy": entropy,
            "memory_confidence": confidence,
            "memory_total_confidence": total_confidence,
            "memory_posterior_confidence": posterior_confidence,
            "memory_effective_sample_count": effective_sample_count,
            "memory_effective_prototypes": active.sum(dim=1, keepdim=True),
        }

    def memory_features(
        self,
        residual_spectrum: torch.Tensor,
        memory_means: torch.Tensor,
        memory_variances: torch.Tensor,
        memory_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._memory_features(
            residual_spectrum,
            memory_means,
            memory_variances,
            memory_counts,
            self.prototypes,
        )

    def forward(
        self,
        noisy: torch.Tensor,
        source: torch.Tensor,
        memory_means: torch.Tensor | None = None,
        memory_variances: torch.Tensor | None = None,
        memory_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source_spectrum = self.stft(source)
        residual_spectrum = self.stft(noisy - source)
        local_features = self.features(source_spectrum, residual_spectrum)
        if memory_means is None:
            shape = (noisy.shape[0], self.prototypes, residual_spectrum.shape[1])
            memory_means = torch.zeros(shape, device=noisy.device)
            memory_variances = torch.ones(shape, device=noisy.device)
            memory_counts = torch.zeros(
                noisy.shape[0], self.prototypes, device=noisy.device
            )
        if memory_variances is None or memory_counts is None:
            raise ValueError("prototype means, variances, and counts are required together")
        context_features, posterior_stats = self.memory_features(
            residual_spectrum, memory_means, memory_variances, memory_counts
        )
        hidden = self.blocks(self.input(local_features))
        static_logit = self.output(hidden).squeeze(1)
        memory_delta = 2.0 * torch.tanh(self.memory_adapter(context_features).squeeze(1))
        reliability = posterior_stats["memory_similarity"].pow(
            self.reliability_power
        )
        reliability = reliability * (
            posterior_stats["memory_confidence"] > 0
        ).to(reliability.dtype)
        memory_delta = memory_delta * reliability.unsqueeze(-1)
        gain = self.max_gain * torch.sigmoid(static_logit + memory_delta)
        estimate_spectrum = source_spectrum + gain * residual_spectrum
        estimate = self.istft(estimate_spectrum, source.shape[-1])
        auxiliary = {
            "source_spectrum": source_spectrum,
            "residual_spectrum": residual_spectrum,
            "estimate_spectrum": estimate_spectrum,
            "memory_features": context_features,
            "memory_delta": memory_delta,
            "memory_reliability": reliability,
            **posterior_stats,
        }
        return estimate, gain, auxiliary


class LowRankResidualDeltaPrototypeContinualResidualSpeechProjector(
    PrototypeContinualResidualSpeechProjector
):
    """Memory-conditioned signed low-rank correction to the local gain.

    The frozen utterance-local projector produces the base residual gain. Current
    utterance features define local correction bases, while causal memory defines
    only their coordinates. Their product is therefore history-identifiable and
    exactly zero when no prior observation exists.
    """

    def __init__(
        self,
        *args,
        delta_rank: int = 4,
        memory_delta_gain: float = 0.02,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if delta_rank < 1:
            raise ValueError("delta_rank must be positive")
        if not 0.0 < memory_delta_gain <= self.max_gain:
            raise ValueError("memory_delta_gain must be in (0, max_gain]")
        self.delta_rank = int(delta_rank)
        self.memory_delta_gain = float(memory_delta_gain)
        channels = self.input.out_channels
        self.delta_basis = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
            nn.SiLU(),
            nn.Conv2d(channels, self.delta_rank, 1, bias=False),
        )
        self.delta_coordinates = nn.Sequential(
            nn.Conv2d(
                self.memory_channels, channels, 3, padding=1, bias=False
            ),
            nn.SiLU(),
            nn.Conv2d(
                channels, channels, 3, padding=1, groups=channels, bias=False
            ),
            nn.SiLU(),
            nn.Conv2d(channels, self.delta_rank, 1, bias=False),
        )
        self._reset_delta_heads()

    def _reset_delta_heads(self) -> None:
        nn.init.normal_(self.delta_basis[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.delta_coordinates[-1].weight)

    def initialize_from_static(self, static: ResidualSpeechProjector) -> None:
        super().initialize_from_static(static)
        self._reset_delta_heads()

    def forward(
        self,
        noisy: torch.Tensor,
        source: torch.Tensor,
        memory_means: torch.Tensor | None = None,
        memory_variances: torch.Tensor | None = None,
        memory_counts: torch.Tensor | None = None,
        external_memory_strength: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source_spectrum = self.stft(source)
        residual_spectrum = self.stft(noisy - source)
        local_features = self.features(source_spectrum, residual_spectrum)
        if memory_means is None:
            shape = (noisy.shape[0], self.prototypes, residual_spectrum.shape[1])
            memory_means = torch.zeros(shape, device=noisy.device)
            memory_variances = torch.ones(shape, device=noisy.device)
            memory_counts = torch.zeros(
                noisy.shape[0], self.prototypes, device=noisy.device
            )
        if memory_variances is None or memory_counts is None:
            raise ValueError(
                "prototype means, variances, and counts are required together"
            )

        context_features, memory_stats = self.memory_features(
            residual_spectrum, memory_means, memory_variances, memory_counts
        )
        hidden = self.blocks(self.input(local_features))
        static_logit = self.output(hidden).squeeze(1)
        static_gain = self.max_gain * torch.sigmoid(static_logit)
        reliability = memory_stats["memory_similarity"].pow(
            self.reliability_power
        )
        reliability = reliability * (
            memory_stats["memory_confidence"] > 0
        ).to(reliability.dtype)
        if external_memory_strength is None:
            memory_strength = torch.ones_like(reliability)
        else:
            memory_strength = torch.as_tensor(
                external_memory_strength,
                dtype=reliability.dtype,
                device=reliability.device,
            )
            if memory_strength.ndim == 1:
                memory_strength = memory_strength.unsqueeze(1)
            memory_strength = torch.broadcast_to(
                memory_strength, reliability.shape
            ).clamp(0.0, 2.0)

        basis = torch.tanh(self.delta_basis(hidden.detach()))
        coordinates = torch.tanh(self.delta_coordinates(context_features))
        interaction = (basis * coordinates).sum(dim=1) / math.sqrt(self.delta_rank)
        delta_gain = self.memory_delta_gain * torch.tanh(interaction)
        delta_gain = (
            delta_gain
            * reliability.unsqueeze(-1)
            * memory_strength.unsqueeze(-1)
        )
        gain = (static_gain + delta_gain).clamp(0.0, self.max_gain)
        estimate_spectrum = source_spectrum + gain * residual_spectrum
        estimate = self.istft(estimate_spectrum, source.shape[-1])
        return estimate, gain, {
            "source_spectrum": source_spectrum,
            "residual_spectrum": residual_spectrum,
            "estimate_spectrum": estimate_spectrum,
            "local_hidden": hidden,
            "static_gain": static_gain,
            "memory_delta": delta_gain,
            "memory_delta_gain": delta_gain,
            "memory_delta_basis": basis,
            "memory_delta_coordinates": coordinates,
            "memory_reliability": reliability,
            "external_memory_strength": memory_strength,
            **memory_stats,
        }


MemoryConditionedRefinement = LowRankResidualDeltaPrototypeContinualResidualSpeechProjector
