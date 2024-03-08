from math import log, pi
from typing import Iterable, Union

import pandas as pd
import torch

__all__ = [
    'JointEmbedding', 'NaNEncoder', 'PositionEmbedding', 'TimeEmbedding',
    'TimeLastLayerNorm'
]


class NaNEncoder(torch.nn.Module):
    '''
    This module replaces NaN values in tensors with zeros, and appends a mask
    along the channel dimension specifying which values were NaNs, doubling the
    number of channels. It is used as a preprocessing step.
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        This method expects to receive tensors in NCT arrangement.
        '''
        is_nan = torch.isnan(x)
        x[is_nan] = 0
        return torch.cat((x, is_nan.to(x.dtype)), dim=1)


class JointEmbedding(torch.nn.ModuleList):
    '''
    This takes a list of multiple time embeddings and applies them sequentially
    to the input sequence.
    '''
    def _init(self):
        for embedding in self:
            embedding._init()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for embedding in self:
            x = embedding(x, t)
        return x


class PositionEmbedding(torch.nn.Module):
    '''
    This layer attaches a positional embedding to the input sequence.
    '''
    def __init__(self, dim: int, scale: int = 1):
        '''
        Args:
            dim (int): Number of input channels.
            scale (int): Expected average distance between time samples. This
            is used to scale the embedding appropriately.
        '''
        super().__init__()
        divisor = (torch.arange(0, dim, 2) * (-log(10000.) / dim)).exp()
        self.register_buffer('divisor', divisor)
        self.scale = scale
        self.linear = torch.nn.Conv1d(dim, dim, 1)

    def _init(self):
        torch.nn.init.kaiming_normal_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if (t.shape[2] != x.shape[2]):
            raise ValueError(f'Mismatch in time length: {x.shape}, {t.shape}')
        t = ((t - t[:, :, 0:1]) // self.scale).float()
        embed = (t * self.divisor.view(1, -1, 1))
        embed = torch.cat((embed.sin(), embed.cos()), dim=1)
        return x + self.linear(embed)


class TimeEmbedding(torch.nn.Module):
    '''
    This layer attaches a temporal embedding to the input sequence.
    '''
    def __init__(self, dim: int, frequencies: Iterable[Union[str, int]]):
        super().__init__()
        self.embed = torch.nn.ModuleDict({
            f: torch.nn.Conv1d(2, dim, 1) for f in frequencies
        })
        self.frequencies = frequencies
        self.wavelengths = [_get_wavelength(f) for f in frequencies]

    def _init(self):
        for conv in self.embed.values():
            torch.nn.init.kaiming_normal_(conv.weight)
            conv.weight /= len(self.embed)
            torch.nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for f, wl in zip(self.frequencies, self.wavelengths):
            t_f = 2 * pi * (t % wl).float() / wl
            x = x + self.embed[f](torch.cat((t_f.sin(), t_f.cos()), dim=1))
        return x


def _get_wavelength(f: Union[int, str]) -> int:
    if isinstance(f, int):
        return f
    elif f == 'W':
        return pd.tseries.frequencies.to_offset('D').nanos * 7
    elif f == 'Y':
        return pd.tseries.frequencies.to_offset('D').nanos * 365
    else:
        return pd.tseries.frequencies.to_offset(f).nanos


class TimeLastLayerNorm(torch.nn.LayerNorm):
    '''
    This is an implementation of layer norm that expects the tensor to have the
    channel dimension as the 1st dimension instead of the last dimension, and
    the time as the last dimension instead of the 1st.
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, -1)).transpose(1, -1)
