import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import ModelRegistry
from .discriminator import Discriminator
from .generator import TSCNet

FEATURE_DICT = {16_000: 201,
                48_000: 601}


@ModelRegistry.register
class CMGAN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.generator = TSCNet(depth=cfg.gen_depth, num_features=FEATURE_DICT[cfg.fs])
        self.discriminator = Discriminator(16)

        self.return_mask = False

    def forward(self, x):
        return self.generate(x)

    def generate(self, x):
        x = x.transpose(2, 3)
        if self.return_mask:
            (gen_r, gen_i), mask = self.generator(x, self.return_mask)
            gen_r, gen_i = gen_r.transpose(2, 3), gen_i.transpose(2, 3)
            return torch.concat((gen_r, gen_i), 1), mask.transpose(2, 3) 
        else:
            gen_r, gen_i = self.generator(x, self.return_mask)

        gen_r, gen_i = gen_r.transpose(2, 3), gen_i.transpose(2, 3)
        return torch.concat((gen_r, gen_i), 1) 


    def discriminate(self, x_clean, x_noisy, x_denoised):
        clean_mag = x_clean.norm(dim=1, keepdim=True)
        denoised_mag = x_denoised.norm(dim=1, keepdim=True)
        pred_1 = self.discriminator(clean_mag, clean_mag)
        pred_2 = self.discriminator(clean_mag, denoised_mag)
        return torch.concat((pred_1, pred_2))

    @torch.no_grad()
    def evaluate(self, x, max_length=2000):
        batched = False
        if x.shape[-1] > max_length:
            batched = True
            batch_size = max_length // 2
            pad_val = round((1.0 - (x.shape[-1] / batch_size) % 1.0) * batch_size)
            x = F.pad(x, (0, pad_val))
            x = x.unfold(-1, batch_size, batch_size)
            x = x[0].permute(2, 0, 1, 3)

        if x.shape[0] > 5:
            outputs = []
            for x in x:
                outputs.append(self.__call__(x[None]))
            x_denoised = torch.concat(outputs)
        else:
            x_denoised = self.__call__(x)

        if batched:
            x_denoised = x_denoised.permute(1, 2, 0, 3)
            x_denoised = x_denoised.flatten(2,3)[None, ..., :-pad_val]

        return x_denoised

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if strict:
            keys = set(self.state_dict().keys())
            load_keys = set(state_dict.keys())
            if load_keys != keys:
                # Support checkpoints created with nn.DataParallel on generator/discriminator.
                state_dict = {
                    name.replace('generator.module.', 'generator.')
                    .replace('discriminator.module.', 'discriminator.'): val
                    for name, val in state_dict.items()
                }
                load_keys = set(state_dict.keys())
                # Legacy compatibility: only expand if keys still mismatch and checkpoint uses
                # `...TSCB_1...` style while current model expects `...TSCB.TSCB_1...`.
                if load_keys != keys:
                    state_dict = {self._maybe_expand_key(name): val for name, val in state_dict.items()}

        return super().load_state_dict(state_dict, strict, assign)

    def _maybe_expand_key(self, key):
        if '.TSCB.TSCB_' in key:
            return key
        groups = key.split('.')
        groups_out = []
        for i in groups:
            if i.startswith('TSCB_'):
                groups_out.append('TSCB')
            groups_out.append(i)

        return '.'.join(groups_out)


    def get_adapt_groups(self, groups):
        parameters = []

        for key, val in self.named_parameters():
            if 'norm' in groups and 'norm' in key:
                parameters.append(val)
            elif 'output' in groups and 'complex_decoder.conv' in key:
                parameters.append(val)
            elif 'input' in groups and 'dense_encoder.conv_1' in key:
                parameters.append(val)
            elif 'prelu' in groups and 'prelu' in key:
                parameters.append(val)
            elif 'fn' in groups and 'fn' in key:
                parameters.append(val)
            elif 'freq' in groups and 'freq' in key:
                parameters.append(val)

        return parameters
