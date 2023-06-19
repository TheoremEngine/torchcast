import math

import torch

__all__ = ['NaNEncoder', 'TimeEmbedding', 'TimeLastLayerNorm']


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


class TimeEmbedding(torch.nn.Module):
    '''
    This layer attaches a time embedding to one or more input sequences.
    '''
    def __init__(self, dim: int):
        '''
        Args:
            dim (int): Number of input channels.
        '''
        super().__init__()
        divisor = (torch.arange(0, dim, 2) * (-math.log(10000.) / dim)).exp()
        self.register_buffer('divisor', divisor)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if (t.shape[2] != x.shape[2]):
            raise ValueError(f'Mismatch in time length: {x.shape}, {t.shape}')
        embed = (t * self.divisor.view(1, -1, 1))
        embed = torch.cat((embed.sin(), embed.cos()), dim=1)
        return x + embed


class TimeLastLayerNorm(torch.nn.LayerNorm):
    '''
    This is an implementation of layer norm that expects the tensor to have the
    channel dimension as the 1st dimension instead of the last dimension, and
    the time as the last dimension instead of the 1st.
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, -1)).transpose(1, -1)
