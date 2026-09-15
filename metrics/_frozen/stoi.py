# Adapted from https://github.com/mpariente/pystoi/blob/master/pystoi/stoi.py
import pystoi
import torch

from ._base import Metric


class STOI(Metric):
    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        update_vals = []
        # Iterate over batch
        for x_cl, x_de in zip(x_clean, x_denoised):
            score = pystoi.stoi(x_cl.squeeze().cpu().numpy(),
                                x_de.squeeze().cpu().numpy(),
                                self.fs)
            self.append(score)
            update_vals.append(score)

        return torch.tensor(update_vals)
