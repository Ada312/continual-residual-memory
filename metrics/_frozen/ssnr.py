import torch

from ._base import Metric
from .composite import SNRseg


class SSNR(Metric):
    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        x_clean = x_clean.cpu().squeeze(dim=1).numpy()
        x_denoised = x_denoised.cpu().squeeze(dim=1).numpy()
        update_vals = []
        # Iterate over batch
        for x_cl, x_de in zip(x_clean, x_denoised):
            score = SNRseg(x_cl, x_de, self.fs)
            update_vals.append(score)
            self.append(score)

        return torch.tensor(update_vals)
