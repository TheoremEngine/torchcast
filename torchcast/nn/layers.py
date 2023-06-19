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
    def __init__(self, dim: int, max_sequence_length: int = 5000):
        '''
        Args:
            dim (int): Number of input channels.
            max_sequence_length (int): The maximum sequence length.
        '''
        super().__init__()

        t = torch.arange(max_sequence_length).unsqueeze(1)
        divisor = (torch.arange(0, dim, 2) * (-math.log(10000.) / dim)).exp()

        # time_embedding must be of arrangement 1CT.
        time_embedding = torch.zeros(1, dim, max_sequence_length)
        time_embedding[0, 0::2, :] = (t * divisor).sin().T
        time_embedding[0, 1::2, :] = (t * divisor).cos().T
        self.register_buffer('time_embedding', time_embedding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] > self.time_embedding.shape[2]:
            raise ValueError('Length of sequences exceed maximum value')
        return x + self.time_embedding[:, :, :x.shape[2]]


class TimeLastLayerNorm(torch.nn.LayerNorm):
    '''
    This is an implementation of layer norm that expects the tensor to have the
    channel dimension as the 1st dimension instead of the last dimension, and
    the time as the last dimension instead of the 1st.
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.transpose(1, -1)).transpose(1, -1)
