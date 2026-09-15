import math
import pesq
import torch
import torchaudio.functional as AF

from ._base import Metric


class PESQ(Metric):
    def update(self, x_clean, x_noisy, x_denoised, kwargs):
        update_vals = []
        # Iterate over batch
        for x_cl, x_deno in zip(x_clean, x_denoised):
            if self.fs != 16_000:
                x_cl = AF.resample(x_cl, self.fs, 16_000)
                x_deno = AF.resample(x_deno, self.fs, 16_000)

            # PESQ can crash / throw NoUtterancesError on near-silent segments.
            try:
                score = pesq.pesq(
                    16_000,
                    x_cl.squeeze().cpu().numpy(),
                    x_deno.squeeze().cpu().numpy(),
                    # mode can be omitted (defaults vary), but explicit is safer:
                    # 'wb' for 16k wideband
                    'wb'
                )
            except Exception:
                # Skip problematic utterances: store NaN so aggregation can ignore it
                score = float("nan")

            # Keep tensor type consistent
            self.append(torch.tensor(score))
            update_vals.append(score)

        return torch.tensor(update_vals)