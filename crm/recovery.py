from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedResidualBlock(nn.Module):
    def __init__(self, channels: int, time_dilation: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.depthwise = nn.Conv2d(
            channels * 2,
            channels * 2,
            kernel_size=3,
            padding=(1, time_dilation),
            dilation=(1, time_dilation),
            groups=channels * 2,
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.depthwise(self.in_proj(F.silu(self.norm(x))))
        value, gate = z.chunk(2, dim=1)
        return x + self.out_proj(torch.tanh(value) * torch.sigmoid(gate))


class ResidualSpeechProjector(nn.Module):
    """Utterance-local residual recovery (P_s and the g_static gain head)."""

    def __init__(
        self,
        channels: int = 32,
        max_gain: float = 0.1,
        initial_gain: float = 0.005,
        n_fft: int = 512,
        hop_length: int = 128,
        compression: float = 0.3,
    ):
        super().__init__()
        if not 0.0 < initial_gain < max_gain:
            raise ValueError("initial_gain must be between zero and max_gain")
        self.max_gain = float(max_gain)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.compression = float(compression)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

        self.input = nn.Conv2d(7, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[GatedResidualBlock(channels, dilation) for dilation in (1, 2, 4, 8, 16, 32)]
        )
        self.output = nn.Conv2d(channels, 1, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        prior = initial_gain / max_gain
        nn.init.constant_(self.output.bias, math.log(prior / (1.0 - prior)))

    def stft(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(audio.device),
            return_complex=True,
        )

    def istft(self, spectrum: torch.Tensor, length: int) -> torch.Tensor:
        return torch.istft(
            spectrum,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(spectrum.device),
            length=length,
        )

    def features(
        self, source_spectrum: torch.Tensor, residual_spectrum: torch.Tensor
    ) -> torch.Tensor:
        source_mag = source_spectrum.abs().clamp_min(1e-7)
        residual_mag = residual_spectrum.abs().clamp_min(1e-7)
        source_comp = source_mag.pow(self.compression)
        residual_comp = residual_mag.pow(self.compression)
        phase_delta = torch.angle(residual_spectrum) - torch.angle(source_spectrum)
        log_ratio = torch.log(residual_mag / source_mag).clamp(-6.0, 6.0) / 6.0
        return torch.stack(
            (
                source_comp * torch.cos(torch.angle(source_spectrum)),
                source_comp * torch.sin(torch.angle(source_spectrum)),
                residual_comp * torch.cos(torch.angle(residual_spectrum)),
                residual_comp * torch.sin(torch.angle(residual_spectrum)),
                log_ratio,
                torch.cos(phase_delta),
                torch.sin(phase_delta),
            ),
            dim=1,
        )

    def forward(
        self, noisy: torch.Tensor, source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        source_spectrum = self.stft(source)
        residual_spectrum = self.stft(noisy - source)
        hidden = self.blocks(self.input(self.features(source_spectrum, residual_spectrum)))
        gain = self.max_gain * torch.sigmoid(self.output(hidden)).squeeze(1)
        estimate_spectrum = source_spectrum + gain * residual_spectrum
        estimate = self.istft(estimate_spectrum, source.shape[-1])
        auxiliary = {
            "source_spectrum": source_spectrum,
            "residual_spectrum": residual_spectrum,
            "estimate_spectrum": estimate_spectrum,
        }
        return estimate, gain, auxiliary


# Keep the checkpoint-bearing class/attribute names unchanged.
UtteranceLocalRecovery = ResidualSpeechProjector
