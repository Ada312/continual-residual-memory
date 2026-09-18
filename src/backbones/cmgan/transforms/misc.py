import torch
import torch.nn as nn


class Log(nn.Module):
    """
    Transform for logarithmic mapping
    """
    def forward(self, x):
        return torch.log(x)
