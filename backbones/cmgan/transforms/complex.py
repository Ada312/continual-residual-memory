import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Complex(nn.Module):
    def __init__(self, name, stft_cfg, power_compress=False, reconstruct=True):
        """Complex Transformations

        Can be used to combine multiple complex transformations
        e.g. to get a concatenation of the magnitude and the complex components
        use the configuration "TF=MAG|COMP"
        
        Possible values: [MAG, COMP, PHASE]

        Args:
            name (str): configuration
            n_fft (int): FFT-length
            hop (int): STFT window hop length
        """
        super().__init__()
        groups = name[3:].split('|')
        self.transforms = [self.parse_key(key) for key in groups]
        self.n_fft = stft_cfg.fft_length
        self.hop = stft_cfg.hop_length
        if stft_cfg.window == 'hann':
            self.window = torch.hann_window(self.n_fft)
        elif stft_cfg.window == 'hamming':
            self.window = torch.hamming_window(self.n_fft)
        else:
            raise ValueError
        self.power_compress = power_compress
        if power_compress is not None:
            self.power_compress = power_compress
        else:
            self.power_compress = False

        self.reconstruct_en = reconstruct
        if not self.reconstruct_en and self.power_compress:
            logger.warning('Power Compression is always reveresed during reconstruction')

    def parse_key(self, key):
        if key == 'MAG':
            return self._magnitude
        elif key == 'COMP':
            return self._complex
        elif key == 'PHASE':
            return self._phase
        else:
            raise ValueError(f'Complex transform {key} unknown')

    def _magnitude(self, x):
        return x.abs()[..., None]

    def _complex(self, x):
        x = torch.view_as_real(x)
        return x

    def _phase(self, x):
        return x.angle()[..., None]

    def _power_compress(self, x):
        mag = x.abs().clamp(min=1e-8) ** self.power_compress
        return torch.polar(mag, x.angle())

    def forward(self, x):
        pad_val = x.shape[1] % self.hop
        if pad_val != 0:
            x = F.pad(x, (0, self.hop - pad_val))

        x_tf = torch.stft(x,
                          self.n_fft,
                          self.hop,
                          window=self.window.to(x.device),
                          return_complex=True,
                          onesided=True)

        if self.power_compress:
            x_tf = self._power_compress(x_tf)

        x_out = torch.concat([trans(x_tf) for trans in self.transforms], -1).squeeze(0)
        return x_out.permute((2, 0, 1)), {}

    def reconstruct(self, x):
        x_comp = torch.complex(x[:, 0], x[:, 1])
        if self.power_compress:
            mag = x_comp.abs() ** (1.0 / self.power_compress)
            x_comp = torch.polar(mag, x_comp.angle())

        if not self.reconstruct_en:
            return x
        x = torch.istft(x_comp,
                        self.n_fft,
                        self.hop,
                        window=self.window.to(x.device),
                        onesided=True)
        return x
