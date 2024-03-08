from typing import Callable, Optional, Union

import torch

from .layers import NaNEncoder, PositionEmbedding, TimeLastLayerNorm
from .transformers import Decoder, Encoder

__all__ = ['EncoderDecoderTransformer', 'EncoderTransformer']


class EncoderDecoderTransformer(torch.nn.Module):
    '''
    Provides a complete encoder-decoder transformer network for time series
    forecasting.
    '''
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_encoder_layers: int, num_decoder_layers: int,
                 exogenous_dim: int = 0, num_heads: int = 8,
                 one_hot_encode_nan_inputs: bool = False, dropout: float = 0.1,
                 embedding: Optional[torch.nn.Module] = None,
                 norm: Callable = TimeLastLayerNorm):
        '''
        Args:
            in_dim (int): Number of channels in the input time series.
            hidden_dim (int): Number of channels in hidden layers.
            out_dim (int): Number of channels in the output time series.
            num_encoder_layers (int): Number of transformer layers in the
                encoder.
            num_decoder_layers (int): Number of transformer layers in the
                decoder.
            exogenous_dim (int): The number of channels in the exogenous data
                used as additional inputs for the output tokens.
            num_heads (int): Number of heads per transformer.
            one_hot_encode_nan_inputs (bool): If provided, expect NaNs to be in
                inputs, and use one-hot encoding prior to the projection to
                handle them.
            dropout (float): Dropout probability.
            embedding (optional, :class:`torch.nn.Module`): A time embedding
                layer. If not provided, a :class:`PositionEmbedding` will be
                used.
            norm (callable): A function for constructing a normalization layer.
                This should expect the dimension as an argument and return the
                layer.
        '''
        super().__init__()

        # NaN encoder
        self.nan_encoder = NaNEncoder() if one_hot_encode_nan_inputs else None

        # Projection of input values
        m = 2 if one_hot_encode_nan_inputs else 1
        self.proj = torch.nn.Conv1d(m * in_dim, hidden_dim, 1)

        # Projection of exogenous inputs
        if exogenous_dim:
            self.proj_exogenous = torch.nn.Conv1d(
                m * exogenous_dim, hidden_dim, 1
            )
        else:
            self.proj_exogenous = None

        # Time embedding
        if embedding is None:
            embedding = PositionEmbedding(hidden_dim)
        self.embedding = embedding

        # Main modules
        self.encoder = Encoder(
            hidden_dim, num_encoder_layers, num_heads=num_heads,
            dropout=dropout, norm=norm
        )
        self.decoder = Decoder(
            hidden_dim, num_encoder_layers, num_heads=num_heads,
            dropout=dropout, norm=norm
        )

        # Output projection
        self.out = torch.nn.Conv1d(hidden_dim, out_dim, 1)
        self.mask_token = torch.nn.Parameter(torch.randn(hidden_dim))

        with torch.no_grad():
            self._init()

    def _init(self):
        for layer in [self.proj, self.proj_exogenous]:
            if layer is not None:
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        if self.out is not None:
            std = 1. / self.out.weight.shape[0]
            torch.nn.init.normal_(self.out.weight, 0., std)
            torch.nn.init.zeros_(self.out.bias)

        self.embedding._init()
        self.encoder._init()
        self.decoder._init()

    def forward(self, x_in: torch.Tensor, t_in: Optional[torch.Tensor] = None,
                exog_in: Optional[torch.Tensor] = None,
                t_out: Optional[Union[int, torch.Tensor]] = None,
                exog_out: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Args:
            x_in (:class:`torch.Tensor`): The time series observed so far.
            t_in (optional, :class:`torch.Tensor`): The times of the time
                series observed so far. If not provided, integer time steps
                will be used.
            exog_in (optional, :class:`torch.Tensor`): The exogenous data for
                the observed time series.
            t_out (optional, int or :class:`torch.Tensor`): The times of the
                time series to be forecasted. If an integer is provided,
                integer time steps will be used. If not provided, x_out must be
                provided, and the number of time steps will be inferred from
                its shape.
            exog_out (optional, :class:`torch.Tensor`): The exogenous data for
                the forecast region of the time series.
        '''
        # Apply NaN encoding
        if self.nan_encoder is not None:
            x_in = self.nan_encoder(x_in)
            if exog_in is not None:
                exog_in = self.nan_encoder(exog_in)
            if exog_out is not None:
                exog_out = self.nan_encoder(exog_out)

        # Get times
        if t_in is None:
            t_in = torch.arange(x_in.shape[2], device=x_in.device)
            t_in = t_in.view(1, 1, -1).repeat(x_in.shape[0], 1, 1)
        if t_out is None and exog_out is None:
            raise ValueError('t_out must be provided')
        elif t_out is None:
            t_out = exog_out.shape[2]
        if isinstance(t_out, int):
            # TODO: Should be based on t_in, not shape
            t_out = torch.arange(
                x_in.shape[2], x_in.shape[2] + t_out, device=x_in.device
            )
            t_out = t_out.view(1, 1, -1).repeat(x_in.shape[0], 1, 1)

        x_in = self.proj(x_in)
        mask = self.mask_token.view(1, -1, 1)
        x_out = mask.repeat(t_out.shape[0], 1, t_out.shape[2])

        # Add exogenous variables
        if self.proj_exogenous is not None:
            if (exog_out is None) or (exog_in is None):
                raise ValueError(
                    'Exogenous variables expected but not received'
                )
            x_in = self.proj_exogenous(exog_in) + x_in
            x_out = self.proj_exogenous(exog_out) + x_out
        elif (exog_out is not None) or (exog_in is not None):
            raise ValueError(
                'Exogenous variables received but not expected'
            )

        # Prep inputs to encoder and decoder
        x = torch.cat((x_in, x_out), dim=2)
        t = torch.cat((t_in, t_out), dim=2)
        x = self.embedding(x, t)
        x_in, x_out = x[:, :, :x_in.shape[2]], x[:, :, x_in.shape[2]:]

        # Now that the tokens are prepped, pass through the encoder.
        encoder_x = self.encoder(x_in)

        # And then the decoder.
        x = self.decoder(x_out, encoder_x)

        # And then the output projection.
        return self.out(x)


class EncoderTransformer(torch.nn.Module):
    '''
    Provides a complete encoder-only transformer network for time series
    forecasting.
    '''
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_encoder_layers: int, num_classes: int = 0,
                 exogenous_dim: int = 0, num_heads: int = 8,
                 one_hot_encode_nan_inputs: bool = False, dropout: float = 0.1,
                 embedding: Optional[torch.nn.Module] = None,
                 norm: Callable = TimeLastLayerNorm):
        '''
        Args:
            in_dim (int): Number of channels in the input time series.
            hidden_dim (int): Number of channels in hidden layers.
            out_dim (int): Number of channels in the output time series.
            num_encoder_layers (int): Number of transformer layers in the
                encoder.
            num_classes (int): If provided, include a classifier
                token predicting this many classes.
            exogenous_dim (int): The number of channels in the exogenous data
                used as additional inputs for the output tokens.
            num_heads (int): Number of heads per transformer.
            token_size (int): Spatial width of each token.
            one_hot_encode_nan_inputs (bool): If provided, expect NaNs to be in
                inputs, and use one-hot encoding prior to the projection to
                handle them.
            dropout (float): Dropout probability.
            embedding (optional, :class:`torch.nn.Module`): A time embedding
                layer. If not provided, a :class:`PositionEmbedding` will be
                used.
            norm (callable): A function for constructing a normalization layer.
                This should expect the dimension as an argument and return the
                layer.
        '''
        super().__init__()

        # NaN encoder
        self.nan_encoder = NaNEncoder() if one_hot_encode_nan_inputs else None

        # Projection of input values
        m = 2 if one_hot_encode_nan_inputs else 1
        self.proj = torch.nn.Conv1d(m * in_dim, hidden_dim, 1)

        # Projection of exogenous inputs
        if exogenous_dim:
            self.proj_exogenous = torch.nn.Conv1d(
                m * exogenous_dim, hidden_dim, 1
            )
        else:
            self.proj_exogenous = None

        # Time embedding
        if embedding is None:
            embedding = PositionEmbedding(hidden_dim)
        self.embedding = embedding

        # Main module
        self.main = Encoder(
            hidden_dim, num_encoder_layers, num_heads=num_heads,
            dropout=dropout, norm=norm,
        )

        # Output projection
        if out_dim:
            self.out = torch.nn.ConvTranspose1d(
                hidden_dim, out_dim, 1
            )
            self.mask_token = torch.nn.Parameter(torch.randn(hidden_dim))
        else:
            self.out = self.mask_token = None

        # Classifier head
        if num_classes:
            self.class_token = torch.nn.Parameter(torch.randn(hidden_dim))
            self.class_proj = torch.nn.Linear(hidden_dim, num_classes)
        else:
            self.class_token = self.class_proj = None

        with torch.no_grad():
            self._init()

    def _init(self):
        for layer in [self.proj, self.proj_exogenous, self.class_proj]:
            if layer is not None:
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        if self.class_token is not None:
            torch.nn.init.zeros_(self.class_token)

        if self.out is not None:
            std = 1. / self.out.weight.shape[0]
            torch.nn.init.normal_(self.out.weight, 0., std)
            torch.nn.init.zeros_(self.out.bias)

        self.embedding._init()
        self.main._init()

    def forward(self, x_in: torch.Tensor, t_in: Optional[torch.Tensor] = None,
                exog_in: Optional[torch.Tensor] = None,
                t_out: Optional[Union[int, torch.Tensor]] = None,
                exog_out: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Args:
            x_in (:class:`torch.Tensor`): The time series observed so far.
            t_in (optional, :class:`torch.Tensor`): The times of the time
                series observed so far. If not provided, integer time steps
                will be used.
            exog_in (optional, :class:`torch.Tensor`): The exogenous data for
                the observed time series.
            t_out (optional, int or :class:`torch.Tensor`): The times of the
                time series to be forecasted. If an integer is provided,
                integer time steps will be used. If not provided, x_out must be
                provided, and the number of time steps will be inferred from
                its shape.
            exog_out (optional, :class:`torch.Tensor`): The exogenous data for
                the forecast region of the time series.
        '''
        # Apply NaN encoding
        if self.nan_encoder is not None:
            x_in = self.nan_encoder(x_in)
            if exog_in is not None:
                exog_in = self.nan_encoder(exog_in)
            if exog_out is not None:
                exog_out = self.nan_encoder(exog_out)

        # Get times
        if t_in is None:
            t_in = torch.arange(x_in.shape[2], device=x_in.device)
            t_in = t_in.view(1, 1, -1).repeat(x_in.shape[0], 1, 1)
        if t_out is None and exog_out is None:
            raise ValueError('t_out must be provided')
        elif t_out is None:
            t_out = exog_out.shape[2]
        if isinstance(t_out, int):
            t_out = torch.arange(
                x_in.shape[2], x_in.shape[2] + t_out, device=x_in.device
            )
            t_out = t_out.view(1, 1, -1).repeat(x_in.shape[0], 1, 1)
        t = torch.cat((t_in, t_out), dim=2)

        x_in = self.proj(x_in)
        mask = self.mask_token.view(1, -1, 1)
        x_out = mask.repeat(x_in.shape[0], 1, t_out.shape[2])

        # Add exogenous variables
        if self.proj_exogenous is not None:
            if exog_out is None:
                raise ValueError(
                    'Exogenous variables expected but not received'
                )
            x_out = x_out + self.proj_exogenous(exog_out)
        elif exog_out is not None:
            raise ValueError(
                'Exogenous variables received but not expected'
            )

        # Prep inputs to encoder and decoder
        x = torch.cat((x_in, x_out), dim=2)
        x = self.embedding(x, t)

        # Add class token if needed.
        if self.class_token is not None:
            token = self.class_token.view(1, -1, 1).repeat(x.shape[0], 1, 1)
            x = torch.cat((token, x), dim=2)

        # Now that the embeddings are prepped, pass through the encoder.
        x = self.main(x)

        out = tuple()

        # Classifier head
        if self.class_proj is not None:
            class_token, x = x[:, :, 0], x[:, :, 1:]
            out = out + (self.class_proj(class_token),)

        # Output projection
        if self.out is not None:
            out = out + (self.out(x[:, :, -t_out.shape[2]:]),)

        return out[0] if (len(out) == 1) else out
