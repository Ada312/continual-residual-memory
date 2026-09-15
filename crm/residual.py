"""Backbone estimate, discarded residual, and paper TF features Phi_t."""

from __future__ import annotations

import torch

from .recovery import UtteranceLocalRecovery


def residual_representation(
    noisy_waveform: torch.Tensor,
    backbone_estimate: torch.Tensor,
    projector: UtteranceLocalRecovery,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return r_t, X_hat_t, R_t, and seven-channel Phi_t without changing the model."""
    residual = noisy_waveform - backbone_estimate
    x_hat_stft = projector.stft(backbone_estimate)
    residual_stft = projector.stft(residual)
    tf_features = projector.features(x_hat_stft, residual_stft)
    return residual, x_hat_stft, residual_stft, tf_features
