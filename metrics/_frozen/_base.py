import os
from abc import ABC, abstractmethod

import torch

SYMB_DICT = {
    'up': u'\u2191',
    'down': u'\u2193',
}


class Metric(ABC):
    def __init__(self, cfg, device):
        """Metric abstract baseclass

        Must overwrite update and result methods
        """
        super().__init__()
        self._data = []
        self.device = device
        self.fs = cfg.data.fs
        self.objective = 'up'

    @property
    def data(self):
        if isinstance(self._data, list):
            return self._data
        else:
            return list(self._data)

    @data.setter
    def data(self, item):
        self._data = item

    @abstractmethod
    def update(self, x_clean, x_noisy, x_denoised, kwargs) -> torch.Tensor:
        # Most metrics won't need all inputs, but need consistent call signature
        pass

    def result(self):
        return torch.stack(self.data).mean(0)

    def std(self):
        return torch.stack(self.data).std(0)

    def reset(self):
        self.data = []

    def append(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self._data.append(value)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            return self.update(*args, **kwargs)

    def __repr__(self) -> str:
        output = []
        result = self.result()
        stds = self.std()
        if result.shape == torch.Size([]):
            result = result[None]

        if stds.shape == torch.Size([]):
            stds = stds[None]

        for name, val, std in zip(self.names(), result, stds):
            if os.environ.get('STD', 'False') == 'True':
                output.append(f'{name}{SYMB_DICT[self.objective]}: {val:.4f} ± {std:.3f}')
            else:
                output.append(f'{name}{SYMB_DICT[self.objective]}: {val:.4f}')

        return '\n'.join(output)

    def names(self):
        return [self.__class__.__name__]
