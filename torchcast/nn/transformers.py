from typing import Callable, Optional

import torch

from .layers import TimeLastLayerNorm


__all__ = ['Decoder', 'DecoderLayer', 'Encoder', 'EncoderLayer']


class Decoder(torch.nn.ModuleList):
    '''
    This module provides a stack of :class:`DecoderLayer`s.
    '''
    def __init__(self, dim: int, num_layers: int,
                 hidden_dim: Optional[int] = None, num_heads: int = 8,
                 dropout: float = 0.1, norm: Callable = TimeLastLayerNorm):
        '''
        Args:
            dim (int): Channel dimension of the input.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Channel dimension of the hidden layers.
            dropout (float): Dropout probability.
            norm (callable): A function for constructing a normalization layer.
                This should expect the dimension as an argument and return the
                layer.
        '''
        hidden_dim = hidden_dim or (dim * 4)
        super().__init__([
            DecoderLayer(dim, num_heads, hidden_dim, dropout=dropout,
                         norm=norm)
            for _ in range(num_layers)
        ])
        self.norm = norm(dim)

    def _init(self):
        with torch.no_grad():
            for i in range(len(self) - 1):
                self[i]._init()

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        for i in range(len(self) - 1):
            x = self[i](x, cross)
        return self.norm(x)


class DecoderLayer(torch.nn.Module):
    '''
    This module replaces `torch.nn.TransformerDecoderLayer`, providing a module
    that consists of a single decoder layer incorporating self-attention,
    cross-attention, and feed-forward layer.
    '''
    def __init__(self, dim: int, num_heads: int, hidden_dim: int,
                 dropout: float = 0.1, norm: Callable = TimeLastLayerNorm):
        '''
        Args:
            dim (int): Channel dimension of the input.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Channel dimension of the hidden layers.
            dropout (float): Dropout probability.
        '''
        super().__init__()

        self.drop = torch.nn.Dropout(dropout)

        # Implementation of self-attention component
        self.self_attn = torch.nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = norm(dim)

        # Implementation of cross-attention component
        self.cross_attn = torch.nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = norm(dim)

        # Implementation of feed-forward component
        self.ff_block = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout),
        )
        self.norm3 = norm(dim)

    def _init(self):
        with torch.no_grad():
            self.self_attn._reset_parameters()
            self.cross_attn._reset_parameters()
            torch.nn.init.kaiming_normal_(self.ff_block[0].weight)
            torch.nn.init.zeros_(self.ff_block[0].bias)
            torch.nn.init.kaiming_normal_(self.ff_block[3].weight)
            torch.nn.init.zeros_(self.ff_block[3].bias)

    def forward(self, x: torch.Tensor, cross: torch.Tensor):
        # Our normalization layers expect the tensor in NCT arrangement, as is
        # standard, while our other layers expect the tensor in NTC
        # arrangement. Hence, we need to repeatedly permute the tensors. Note
        # that if using LayerNorm, the default option, these permutations are
        # actually a no-op as they will be undone inside the TimeLastLayerNorm
        # forward method. However, we need to be careful with the skip
        # connections to make sure the tensor is in the same arrangement.

        x, cross = x.permute(0, 2, 1), cross.permute(0, 2, 1)  # NTC
        # Self-attention component
        attn, _ = self.self_attn(x, x, x, need_weights=False)
        x = (x + self.drop(attn)).permute(0, 2, 1)  # NCT
        x = self.norm1(x).permute(0, 2, 1)  # NTC
        # Cross-attention component
        attn, _ = self.cross_attn(x, cross, cross, need_weights=False)
        x = (x + self.drop(attn)).permute(0, 2, 1)  # NCT
        x = self.norm2(x).permute(0, 2, 1)  # NTC
        # Feed-forward component
        x = (x + self.ff_block(x)).permute(0, 2, 1)  # NCT
        return self.norm3(x)


class Encoder(torch.nn.ModuleList):
    '''
    This module provides a stack of :class:`EncoderLayer`s.
    '''
    def __init__(self, dim: int, num_layers: int,
                 hidden_dim: Optional[int] = None, num_heads: int = 8,
                 dropout: float = 0.1, norm: Callable = TimeLastLayerNorm):
        '''
        Args:
            dim (int): Channel dimension of the input.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Channel dimension of the hidden layers.
            dropout (float): Dropout probability.
            norm (callable): A function for constructing a normalization layer.
                This should expect the dimension as an argument and return the
                layer.
        '''
        hidden_dim = hidden_dim or (dim * 4)
        super().__init__([
            EncoderLayer(dim, num_heads, hidden_dim, dropout=dropout,
                         norm=norm)
            for _ in range(num_layers)
        ])
        self.norm = norm(dim)

    def _init(self):
        with torch.no_grad():
            for i in range(len(self) - 1):
                self[i]._init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self) - 1):
            x = self[i](x)
        return self.norm(x)


class EncoderLayer(torch.nn.Module):
    '''
    This module replaces `torch.nn.TransformerEncoderLayer`, providing a module
    that consists of a single encoder layer incorporating self-attention and
    feed-forward layer.
    '''
    def __init__(self, dim: int, num_heads: int, hidden_dim: int,
                 dropout: float = 0.1, norm: Callable = TimeLastLayerNorm):
        '''
        Args:
            dim (int): Channel dimension of the input.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Channel dimension of the hidden layers.
            dropout (float): Dropout probability.
            norm (callable): A function for constructing a normalization layer.
                This should expect the dimension as an argument and return the
                layer.
        '''
        super().__init__()

        self.drop = torch.nn.Dropout(dropout)

        # Implementation of self-attention component
        self.self_attn = torch.nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = norm(dim)

        # Implementation of feed-forward component
        self.ff_block = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout),
        )
        self.norm2 = norm(dim)

    def _init(self):
        with torch.no_grad():
            self.self_attn._reset_parameters()
            torch.nn.init.kaiming_normal_(self.ff_block[0].weight)
            torch.nn.init.zeros_(self.ff_block[0].bias)
            torch.nn.init.kaiming_normal_(self.ff_block[3].weight)
            torch.nn.init.zeros_(self.ff_block[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x (:class:`torch.Tensor`): The input sequence to be decoded. This
                should be in arrangement NCT.
        '''
        # Our normalization layers expect the tensor in NCT arrangement, as is
        # standard, while our other layers expect the tensor in NTC
        # arrangement. Hence, we need to repeatedly permute the tensors. Note
        # that if using LayerNorm, the default option, these permutations are
        # actually a no-op as they will be undone inside the TimeLastLayerNorm
        # forward method. However, we need to be careful with the skip
        # connections to make sure the tensor is in the same arrangement.

        x = x.permute(0, 2, 1)  # NTC
        # Self-attention component
        attn, _ = self.self_attn(x, x, x, need_weights=False)
        x = (x + self.drop(attn)).permute(0, 2, 1)  # NCT
        x = self.norm1(x).permute(0, 2, 1)  # NTC
        # Feed-forward component
        x = (x + self.ff_block(x)).permute(0, 2, 1)  # NCT
        return self.norm2(x)
