from math import log, pi
from typing import Iterable

import torch

__all__ = [
    'NaNEncoder', 'PositionEmbedding', 'TimeEmbedding', 'TimeLastLayerNorm'
]


class NaNEncoder(torch.nn.Module):
    '''
    This module replaces NaN values in tensors with random values, and
    appends a mask along the channel dimension specifying which values were
    NaNs. It is used as a preprocessing step.
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        This method expects to receive tensors in NCT arrangement.
        '''
        is_nan = torch.isnan(x)
        x[is_nan] = 0
        return torch.cat((x, is_nan.to(x.dtype)), dim=1)


class PositionEmbedding(torch.nn.Module):
    '''
    This layer attaches a positional embedding to the input sequence.
    '''
    def __init__(self, dim: int):
        '''
        Args:
            dim (int): Number of input channels.
        '''
        super().__init__()
        divisor = (torch.arange(0, dim, 2) * (-log(10000.) / dim)).exp()
        self.register_buffer('divisor', divisor)

    def _init(self):
        return

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if (t.shape[2] != x.shape[2]):
            raise ValueError(f'Mismatch in time length: {x.shape}, {t.shape}')
        embed = (t * self.divisor.view(1, -1, 1))
        embed = torch.cat((embed.sin(), embed.cos()), dim=1)
        return x + embed


class TimeEmbedding(torch.nn.Module):
    '''
    This layer attaches a temporal embedding to the input sequence.
    '''
    wavelengths = {
        'W': 604_800_000_000_000,
        'D':  86_400_000_000_000,
        'H':   3_600_000_000_000,
        'm':      60_000_000_000,
        's':       1_000_000_000,
    }

    def __init__(self, dim: int, frequencies: Iterable[str]):
        super().__init__()
        self.embed = torch.nn.ModuleDict({
            f: torch.nn.Conv1d(2, dim, 1) for f in frequencies
        })
        self.frequencies = frequencies

    def _init(self):
        for conv in self.embed.values():
            torch.nn.init.kaiming_normal_(conv.weight)
            conv.weight /= len(self.embed)
            torch.nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for f in self.frequencies:
            wl = self.wavelengths[f]
            t_f = 2 * pi * (t % wl).float() / wl
            x = x + self.embed[f](torch.cat((t_f.sin(), t_f.cos()), dim=1))
        return x


class TimeLastLayerNorm(torch.nn.LayerNorm):
    '''
    This is an implementation of layer norm that expects the tensor to have the
    channel dimension as the 1st dimension instead of the last dimension, and
    the time as the last dimension instead of the 1st.
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, -1)).transpose(1, -1)
