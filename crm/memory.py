from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class PrototypeNoiseMemoryContext:
    means: torch.Tensor
    variances: torch.Tensor
    counts: torch.Tensor

    def clone(self) -> "PrototypeNoiseMemoryContext":
        return PrototypeNoiseMemoryContext(
            self.means.clone(), self.variances.clone(), self.counts.clone()
        )


class CausalPrototypeResidualNoiseMemory:
    """Online robust mixture of residual-noise spectral prototypes."""

    def __init__(
        self,
        prototypes: int = 4,
        frequency_bins: int = 257,
        noise_frame_fraction: float = 0.3,
        warmup_utterances: int = 20,
        maximum_step: float = 2.0,
        novelty_threshold: float = 0.35,
        level_weight: float = 0.15,
    ):
        if prototypes < 1:
            raise ValueError("prototypes must be positive")
        if not 0.0 < noise_frame_fraction <= 1.0:
            raise ValueError("noise_frame_fraction must be in (0, 1]")
        self.prototypes = int(prototypes)
        self.frequency_bins = int(frequency_bins)
        self.noise_frame_fraction = float(noise_frame_fraction)
        self.warmup_utterances = int(warmup_utterances)
        self.maximum_step = float(maximum_step)
        self.novelty_threshold = float(novelty_threshold)
        self.level_weight = float(level_weight)
        self.reset()

    def reset(self) -> None:
        self.means = torch.zeros(self.prototypes, self.frequency_bins)
        self.variances = torch.ones(self.prototypes, self.frequency_bins)
        self.counts = torch.zeros(self.prototypes, dtype=torch.long)

    def context(self) -> PrototypeNoiseMemoryContext:
        return PrototypeNoiseMemoryContext(
            self.means.clone(), self.variances.clone(), self.counts.clone()
        )

    def _observation(
        self, source_spectrum: torch.Tensor, residual_spectrum: torch.Tensor
    ) -> torch.Tensor:
        source_power = source_spectrum.abs().square()
        residual_power = residual_spectrum.abs().square()
        source_fraction = source_power.sum(0) / (
            source_power.sum(0) + residual_power.sum(0) + 1e-8
        )
        selected = max(1, int(math.ceil(source_fraction.numel() * self.noise_frame_fraction)))
        indices = torch.topk(source_fraction, selected, largest=False).indices
        return torch.log(residual_power[:, indices] + 1e-8).median(dim=1).values.cpu()

    def _distances(self, observation: torch.Tensor) -> torch.Tensor:
        centered_observation = observation - observation.mean()
        centered_means = self.means - self.means.mean(dim=1, keepdim=True)
        shape_distance = (
            (centered_means - centered_observation).square()
            / (self.variances + 0.5)
        ).mean(dim=1)
        level_distance = (self.means.mean(dim=1) - observation.mean()).square()
        distances = shape_distance + self.level_weight * level_distance
        return distances.masked_fill(self.counts == 0, float("inf"))

    @torch.no_grad()
    def change_score(
        self, source_spectrum: torch.Tensor, residual_spectrum: torch.Tensor
    ) -> float:
        if not torch.any(self.counts > 0):
            return 0.0
        observation = self._observation(source_spectrum, residual_spectrum)
        return float(self._distances(observation).min())

    @torch.no_grad()
    def update(
        self, source_spectrum: torch.Tensor, residual_spectrum: torch.Tensor
    ) -> int:
        if source_spectrum.ndim != 2 or residual_spectrum.ndim != 2:
            raise ValueError("memory update expects [frequency, time] spectra")
        if source_spectrum.shape != residual_spectrum.shape:
            raise ValueError("source and residual spectra must have the same shape")
        observation = self._observation(source_spectrum, residual_spectrum)
        return self.update_observation(observation)

    @torch.no_grad()
    def update_observation(self, observation: torch.Tensor) -> int:
        """Update from a precomputed residual observation in causal stream order."""
        observation = observation.detach().float().cpu()
        if observation.shape != (self.frequency_bins,):
            raise ValueError(
                f"expected observation [{self.frequency_bins}], got {tuple(observation.shape)}"
            )
        inactive = torch.where(self.counts == 0)[0]
        if torch.all(self.counts == 0):
            selected = 0
        else:
            distances = self._distances(observation)
            selected = int(torch.argmin(distances))
            if inactive.numel() and float(distances[selected]) > self.novelty_threshold:
                selected = int(inactive[0])

        if self.counts[selected] == 0:
            self.means[selected].copy_(observation)
            self.variances[selected].fill_(1.0)
            self.counts[selected] = 1
            return selected

        rate = 1.0 / min(int(self.counts[selected]) + 1, self.warmup_utterances)
        innovation = (observation - self.means[selected]).clamp(
            -self.maximum_step, self.maximum_step
        )
        self.means[selected].add_(rate * innovation)
        self.variances[selected].mul_(1.0 - rate).add_(rate * innovation.square())
        self.variances[selected].clamp_(min=0.05, max=9.0)
        self.counts[selected] += 1
        return selected


class DynamicCausalPrototypeResidualNoiseMemory(CausalPrototypeResidualNoiseMemory):
    """Budgeted prototype memory with conservative online birth and retirement.

    A new mode is admitted only when consecutive novel observations agree with one
    another. Unrelated outliers therefore do not overwrite useful stream state. When
    the budget is full, only a prototype that has remained unused for a configurable
    horizon can be replaced.
    """

    def __init__(
        self,
        prototypes: int = 4,
        frequency_bins: int = 257,
        noise_frame_fraction: float = 0.3,
        warmup_utterances: int = 20,
        maximum_step: float = 2.0,
        novelty_threshold: float = 0.35,
        level_weight: float = 0.15,
        novelty_patience: int = 2,
        candidate_threshold: float | None = None,
        immediate_novelty_threshold: float | None = None,
        merge_threshold: float = 0.05,
        retirement_horizon: int = 20,
    ):
        if novelty_patience < 1:
            raise ValueError("novelty_patience must be positive")
        if retirement_horizon < 1:
            raise ValueError("retirement_horizon must be positive")
        self.novelty_patience = int(novelty_patience)
        self.candidate_threshold = float(
            novelty_threshold if candidate_threshold is None else candidate_threshold
        )
        if (
            immediate_novelty_threshold is not None
            and immediate_novelty_threshold <= novelty_threshold
        ):
            raise ValueError(
                "immediate_novelty_threshold must exceed novelty_threshold"
            )
        self.immediate_novelty_threshold = (
            None
            if immediate_novelty_threshold is None
            else float(immediate_novelty_threshold)
        )
        self.merge_threshold = float(merge_threshold)
        self.retirement_horizon = int(retirement_horizon)
        super().__init__(
            prototypes=prototypes,
            frequency_bins=frequency_bins,
            noise_frame_fraction=noise_frame_fraction,
            warmup_utterances=warmup_utterances,
            maximum_step=maximum_step,
            novelty_threshold=novelty_threshold,
            level_weight=level_weight,
        )

    def reset(self) -> None:
        super().reset()
        self.usage = torch.zeros(self.prototypes)
        self.last_used = torch.full((self.prototypes,), -1, dtype=torch.long)
        self.step = 0
        self.candidate: torch.Tensor | None = None
        self.candidate_count = 0

    def _observation_distance(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> float:
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        shape = (left_centered - right_centered).square().mean()
        level = (left.mean() - right.mean()).square()
        return float(shape + self.level_weight * level)

    def _clear_candidate(self) -> None:
        self.candidate = None
        self.candidate_count = 0

    def current_novelty_threshold(self) -> float:
        return self.novelty_threshold

    def novelty_detection_ready(self) -> bool:
        return True

    def _record_observed_distance(self, distance: float) -> None:
        del distance

    def _initialize(self, index: int, observation: torch.Tensor) -> None:
        self.means[index].copy_(observation)
        self.variances[index].fill_(1.0)
        self.counts[index] = 1
        self.usage[index] = 1.0
        self.last_used[index] = self.step

    def _merge_redundant(self) -> None:
        active = torch.where(self.counts > 0)[0].tolist()
        best: tuple[float, int, int] | None = None
        for offset, left in enumerate(active):
            for right in active[offset + 1 :]:
                distance = self._observation_distance(
                    self.means[left], self.means[right]
                )
                if best is None or distance < best[0]:
                    best = (distance, left, right)
        if best is None or best[0] >= self.merge_threshold:
            return

        _, left, right = best
        left_count = float(self.counts[left])
        right_count = float(self.counts[right])
        total = left_count + right_count
        mean = (
            left_count * self.means[left] + right_count * self.means[right]
        ) / total
        variance = (
            left_count
            * (self.variances[left] + (self.means[left] - mean).square())
            + right_count
            * (self.variances[right] + (self.means[right] - mean).square())
        ) / total
        self.means[left].copy_(mean)
        self.variances[left].copy_(variance.clamp(0.05, 9.0))
        self.counts[left] += self.counts[right]
        self.usage[left] += self.usage[right]
        self.last_used[left] = torch.maximum(
            self.last_used[left], self.last_used[right]
        )
        self.means[right].zero_()
        self.variances[right].fill_(1.0)
        self.counts[right] = 0
        self.usage[right] = 0.0
        self.last_used[right] = -1

    @torch.no_grad()
    def update(
        self, source_spectrum: torch.Tensor, residual_spectrum: torch.Tensor
    ) -> int:
        if source_spectrum.ndim != 2 or residual_spectrum.ndim != 2:
            raise ValueError("memory update expects [frequency, time] spectra")
        if source_spectrum.shape != residual_spectrum.shape:
            raise ValueError("source and residual spectra must have the same shape")

        self.step += 1
        self.usage.mul_(0.995)
        observation = self._observation(source_spectrum, residual_spectrum)
        active = torch.where(self.counts > 0)[0]
        if active.numel() == 0:
            self._initialize(0, observation)
            return 0

        distances = self._distances(observation)
        selected = int(torch.argmin(distances))
        selected_distance = float(distances[selected])
        novelty_threshold = self.current_novelty_threshold()
        novelty_ready = self.novelty_detection_ready()
        is_novel = novelty_ready and selected_distance > novelty_threshold
        # Bootstrap before novelty decisions are enabled. Once ready, retain only
        # matched distances so rejected candidates cannot inflate future thresholds.
        if not novelty_ready or not is_novel:
            self._record_observed_distance(selected_distance)
        if is_novel:
            immediate_birth = (
                self.immediate_novelty_threshold is not None
                and selected_distance > self.immediate_novelty_threshold
            )
            if immediate_birth:
                self.candidate = observation.clone()
                self.candidate_count = self.novelty_patience
            elif (
                self.candidate is None
                or self._observation_distance(observation, self.candidate)
                > self.candidate_threshold
            ):
                self.candidate = observation.clone()
                self.candidate_count = 1
            else:
                count = self.candidate_count
                self.candidate = (count * self.candidate + observation) / (count + 1)
                self.candidate_count += 1

            if self.candidate_count < self.novelty_patience:
                return -1
            inactive = torch.where(self.counts == 0)[0]
            replacement: int | None = int(inactive[0]) if inactive.numel() else None
            if replacement is None:
                ages = self.step - self.last_used
                stale = torch.where(ages >= self.retirement_horizon)[0]
                if stale.numel():
                    stale_scores = ages[stale].float() - 0.01 * self.usage[stale]
                    replacement = int(stale[torch.argmax(stale_scores)])
            if replacement is None:
                return -1
            self._initialize(replacement, self.candidate)
            self._clear_candidate()
            self._merge_redundant()
            return replacement

        self._clear_candidate()
        rate = 1.0 / min(int(self.counts[selected]) + 1, self.warmup_utterances)
        innovation = (observation - self.means[selected]).clamp(
            -self.maximum_step, self.maximum_step
        )
        self.means[selected].add_(rate * innovation)
        self.variances[selected].mul_(1.0 - rate).add_(
            rate * innovation.square()
        )
        self.variances[selected].clamp_(min=0.05, max=9.0)
        self.counts[selected] += 1
        self.usage[selected] += 1.0
        self.last_used[selected] = self.step
        self._merge_redundant()
        return selected


PrototypeMemory = DynamicCausalPrototypeResidualNoiseMemory
