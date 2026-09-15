import numpy as np
from ._base import Metric


class SISDR(Metric):
    def update(self, x_clean, x_noisy, x_denoised, kwargs):

        x_clean = x_clean.cpu().numpy()
        x_noisy = x_noisy.cpu().numpy()
        x_denoised = x_denoised.cpu().numpy()
    
        scale = (x_clean * x_denoised).sum(1, keepdims=True) / ((x_clean ** 2).sum(1, keepdims=True) + 1e-8)

        sig = np.linalg.norm(scale * x_clean, axis=1, keepdims=True)
        diff = np.linalg.norm(scale * x_clean - x_denoised, axis=1, keepdims=True)

        sisdr = (sig / (diff + 1e-8)) ** 2
        sisdr = 10 * np.log10(sisdr)
        sisdr = sisdr.mean()

        self.append(sisdr)

        return sisdr
