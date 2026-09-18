import pyloudnorm as pyln
import torch
import torch.nn as nn


class Norm(nn.Module):
    def __init__(self, reconstruct):
        """
        Transform to normalize input w.r.t Z-score
        """
        super().__init__()
        self.reconstruct_en = reconstruct

    def forward(self, x):
        std, mu = torch.std_mean(x, dim=-1, keepdim=True)

        x = x - mu
        x = x / (std + 1e-10)

        return x, {'mu': mu, 'std': std}

    def reconstruct(self, x, mu, std):
        if not self.reconstruct_en:
            return x
        x = x * std.to(x.device)
        x = x + mu.to(x.device)

        return x


class PowerNorm(nn.Module):
    def __init__(self, reconstruct):
        super().__init__()

        self.reconstruct_en = reconstruct
    
    def forward(self, x):
        c = torch.sqrt(x.size(-1) / (torch.sum((x**2.0), dim=-1, keepdim=True) + 1e-8) + 1e-8)
        x = x * c
                    
        return x, {'c': c}

    def reconstruct(self, x, c):
        if not self.reconstruct_en:
            return x
        if x.dim() < c.dim():
            c = c[..., 0]
        x = x / (c.to(x.device) + 1e-10)
        return x 

class LoudNorm(nn.Module):
    def __init__(self, reconstruct, fs):
        super().__init__()
        self.reconstruct_en = reconstruct
        self.meter = pyln.Meter(fs)

    def forward(self, x):
        loudness = self.meter.integrated_loudness(x.squeeze().numpy())
        x = self.normalize(x, loudness, -30)
        return x, {'loudness': loudness}

    def reconstruct(self, x, loudness):
        x = self.normalize(x, -30, loudness)
        return x

    def normalize(self, x, input_loudness, target_loudness):
        # Yanked from pyloudnorm
        delta_loudness = target_loudness - input_loudness
        gain = 10 ** (delta_loudness / 20.0)
        if not isinstance(gain, torch.Tensor):
            gain = torch.tensor(gain, dtype=torch.float32)
        x = x * gain[..., None].to(x.device)
        return x
