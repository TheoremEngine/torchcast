import math

import torch

__all__ = [
    'NaNEncoder', 'TimeEmbedding', 'TransformerLayer',
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
        x[is_nan] = torch.rand_like(x[is_nan], dtype=torch.float32)
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


class TransformerLayer(torch.nn.Module):
    '''
    This :class:`torch.nn.Module` replaces `torch.nn.TransformerDecoderLayer`,
    providing a module that consists of a single encoder layer incorporating
    self-attention and feed-forward layer.
    '''
    def __init__(self, dim: int, num_heads: int, hidden_dim: int,
                 dropout: float = 0.1):
        '''
        Args:
            dim (int): Channel dimension of the input.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Channel dimension of the hidden layers.
            dropout (float): Dropout probability.
        '''
        super().__init__()

        # Implementation of self-attention component
        self.self_attn = torch.nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.drop = torch.nn.Dropout(dropout)
        self.norm1 = torch.nn.LayerNorm(dim)

        # Implementation of feed-forward component
        self.ff_block = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout),
        )
        self.norm2 = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (:class:`torch.Tensor`): The input sequence to be decoded. This
            should be in arrangement NCT.
        '''
        # Permute NCT -> NTC
        x = x.permute(0, 2, 1)
        # Self-attention component
        x = x + self.drop(self.self_attn(x, x, x, need_weights=False)[0])
        # Feed-forward component
        x = self.norm2(x + self.ff_block(self.norm1(x)))
        # Permute NTC -> NCT
        x = x.permute(0, 2, 1)
        return x
