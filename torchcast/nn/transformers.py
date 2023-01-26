import torch

from .layers import (
    NaNEncoder,
    TimeEmbedding,
    TransformerLayer
)

__all__ = ['EncoderTransformer']


class EncoderTransformer(torch.nn.Module):
    '''
    Provides a complete encoder-only transformer network for time series
    forecasting.
    '''
    def __init__(self, num_input_channels: int, num_output_channels: int,
                 hidden_dim: int, num_layers: int, num_heads: int = 8,
                 token_size: int = 1, one_hot_encode_nan_inputs: bool = False):
        super().__init__()

        self.nan_encoder = NaNEncoder() if one_hot_encode_nan_inputs else None
        m = 2 if one_hot_encode_nan_inputs else 1
        self.project = torch.nn.Conv1d(
            m * num_input_channels, hidden_dim, token_size, stride=token_size,
        )
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.main = torch.nn.Sequential(
            *[TransformerLayer(hidden_dim, num_heads, hidden_dim * 4)
              for _ in range(num_layers)]
        )
        self.out = torch.nn.ConvTranspose1d(
            hidden_dim, num_output_channels, token_size, stride=token_size
        )

        self._init()

    def _init(self):
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(self.project.weight)
            torch.nn.init.zeros_(self.project.bias)
            torch.nn.init.kaiming_normal_(self.out.weight)
            torch.nn.init.zeros_(self.out.bias)

            for layer in self.main:
                layer.self_attn._reset_parameters()
                torch.nn.init.kaiming_normal_(layer.ff_block[0].weight)
                torch.nn.init.zeros_(layer.ff_block[0].bias)
                torch.nn.init.kaiming_normal_(layer.ff_block[3].weight)
                torch.nn.init.zeros_(layer.ff_block[3].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nan_encoder is not None:
            x = self.nan_encoder(x)

        x = self.project(x)
        x = self.time_embedding(x)
        x = self.main(x)

        return self.out(x)
