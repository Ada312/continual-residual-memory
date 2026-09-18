import torchaudio
from torch import nn

from . import transforms

def get_transforms(cfg):
    """Get Transforms from config"""
    if cfg.transforms is None or len(cfg.transforms) == 0:
        return nn.Identity()

    transforms = []
    reconstructions = []
    for name in cfg.transforms:
        out = get_transform(name, cfg)
        if isinstance(out, (list, tuple)):
            transform, recon = out
        else:
            transform = out
            def recon(x):
                return x

        transforms.append(transform)
        reconstructions.append(recon)

    return TransformChain(transforms)


def get_transform(name, cfg):
    
    # If name starts with "*", do not reconstruct
    reconstruct = True
    if name.startswith('*'):
        reconstruct = False
        name = name.lstrip('*')

    if name == 'norm':
        return transforms.Norm(reconstruct)
    elif name == 'power_norm':
        return transforms.PowerNorm(reconstruct)
    elif name == 'loud_norm':
        return transforms.LoudNorm(reconstruct, cfg.fs)
    elif name == 'stft':
        return TransformWrapper(torchaudio.transforms.Spectrogram(), reconstruct)
    elif name == 'mel_stft':
        return TransformWrapper(torchaudio.transforms.MelSpectrogram(cfg.fs), reconstruct)
    elif name == 'log':
        return TransformWrapper(transforms.Log(), reconstruct)
    elif name == 'identity':
        return TransformWrapper(nn.Identity(), reconstruct)
    elif name.startswith('TF'):
        return transforms.Complex(name, cfg.stft, power_compress=cfg.power_compress, reconstruct=reconstruct)
    else:
        raise NameError(f'Transform "{name}" unknown')


class TransformWrapper(nn.Module):
    def __init__(self, module, reconstruct):
        super().__init__()
        self.module = module
        self.reconstruct_en = reconstruct

    def forward(self, x):
        output = self.module(x)
        if isinstance(output, (list, tuple)):
            return output
        else:
            return output, {}

    def reconstruct(self, x, **kwargs):
        if not self.reconstruct_en:
            return x

        if hasattr(self.module, 'reconstruct'):
            return self.module.reconstruct(x, **kwargs)
        else:
            return x


class TransformChain(nn.Module):
    def __init__(self, transforms):
        super().__init__()

        self.transforms = transforms

    def reconstruct(self, x, kwargs_list):
        assert len(self.transforms) == len(kwargs_list)
        for transform, kwargs in zip(reversed(self.transforms), reversed(kwargs_list)):
            x = transform.reconstruct(x, **kwargs)
    
        return x

    def forward(self, x):
        recon_kwargs = []
        for transform in self.transforms:
            x, recon_kwarg = transform(x)
            recon_kwargs.append(recon_kwarg)

        return x, recon_kwargs
