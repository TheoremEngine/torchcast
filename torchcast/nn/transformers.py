from typing import Optional

import torch


__all__ = ['Decoder', 'DecoderLayer', 'Encoder', 'EncoderLayer']


class Decoder(torch.nn.ModuleList):
    '''
    This module provides a stack of :class:`DecoderLayer`s.
    '''
    def __init__(self, dim: int, num_layers: int,
                 hidden_dim: Optional[int] = None, num_heads: int = 8):
        '''
        Args:
            dim (int): Channel dimension of the input.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Channel dimension of the hidden layers.
            dropout (float): Dropout probability.
        '''
        hidden_dim = hidden_dim or (dim * 4)
        super().__init__([
            DecoderLayer(dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(dim)

    def _init(self):
        with torch.no_grad():
            for i in range(len(self) - 1):
                self[i]._init()

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        for i in range(len(self) - 1):
            x = self[i](x, cross)
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)


class DecoderLayer(torch.nn.Module):
    '''
    This module replaces `torch.nn.TransformerDecoderLayer`, providing a module
    that consists of a single decoder layer incorporating self-attention,
    cross-attention, and feed-forward layer.
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

        self.drop = torch.nn.Dropout(dropout)

        # Implementation of self-attention component
        self.self_attn = torch.nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = torch.nn.LayerNorm(dim)

        # Implementation of cross-attention component
        self.cross_attn = torch.nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = torch.nn.LayerNorm(dim)

        # Implementation of feed-forward component
        self.ff_block = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.ReLU(True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout),
        )
        self.norm3 = torch.nn.LayerNorm(dim)

    def _init(self):
        with torch.no_grad():
            self.self_attn._reset_parameters()
            self.cross_attn._reset_parameters()
            torch.nn.init.kaiming_normal_(self.ff_block[0].weight)
            torch.nn.init.zeros_(self.ff_block[0].bias)
            torch.nn.init.kaiming_normal_(self.ff_block[3].weight)
            torch.nn.init.zeros_(self.ff_block[3].bias)

    def forward(self, x: torch.Tensor, cross: torch.Tensor):
        # Permute NCT -> NTC
        x, cross = x.permute(0, 2, 1), cross.permute(0, 2, 1)
        # Self-attention component
        attn, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(attn))
        # Cross-attention component
        attn, _ = self.cross_attn(x, cross, cross, need_weights=False)
        x = self.norm2(x + self.drop(attn))
        # Feed-forward component
        x = self.norm3(x + self.ff_block(x))
        # Permute NTC -> NCT
        x = x.permute(0, 2, 1)
        return x


class Encoder(torch.nn.ModuleList):
    '''
    This module provides a stack of :class:`EncoderLayer`s.
    '''
    def __init__(self, dim: int, num_layers: int,
                 hidden_dim: Optional[int] = None, num_heads: int = 8):
        hidden_dim = hidden_dim or (dim * 4)
        super().__init__([
            EncoderLayer(dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])
        self.norm = torch.nn.LayerNorm(dim)

    def _init(self):
        with torch.no_grad():
            for i in range(len(self) - 1):
                self[i]._init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self) - 1):
            x = self[i](x)
        return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)


class EncoderLayer(torch.nn.Module):
    '''
    This module replaces `torch.nn.TransformerEncoderLayer`, providing a module
    that consists of a single encoder layer incorporating self-attention and
    feed-forward layer.
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

        self.drop = torch.nn.Dropout(dropout)

        # Implementation of self-attention component
        self.self_attn = torch.nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
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
        # Permute NCT -> NTC
        x = x.permute(0, 2, 1)
        # Self-attention component
        attn, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(attn))
        # Feed-forward component
        x = self.norm2(x + self.ff_block(x))
        # Permute NTC -> NCT
        x = x.permute(0, 2, 1)
        return x
