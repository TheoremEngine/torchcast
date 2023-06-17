from typing import Optional

import torch

from .layers import NaNEncoder, TimeEmbedding
from .transformers import Decoder, Encoder

__all__ = ['EncoderDecoderTransformer', 'EncoderTransformer']


class EncoderDecoderTransformer(torch.nn.Module):
    '''
    Provides a complete encoder-decoder transformer network for time series
    forecasting.
    '''
    def __init__(self, series_dim: int, hidden_dim: int,
                 num_encoder_layers: int, num_decoder_layers: int,
                 exogenous_dim: int = 0, predict_ahead: int = 1,
                 num_heads: int = 8, one_hot_encode_nan_inputs: bool = False):
        '''
        Args:
            series_dim (int): Number of channels in the time series.
            hidden_dim (int): Number of channels in hidden layers.
            num_encoder_layers (int): Number of transformer layers in the
            encoder.
            num_decoder_layers (int): Number of transformer layers in the
            decoder.
            exogenous_dim (int): The number of channels in the exogenous data
            used in prediction.
            predict_ahead (int): If provided, how far ahead the network will
            predict, in time steps.
            num_heads (int): Number of heads per transformer.
            token_size (int): Spatial width of each token.
            one_hot_encode_nan_inputs (bool): If provided, expect NaNs to be in
            inputs, and use one-hot encoding prior to the projection to handle
            them.
        '''
        super().__init__()

        self.nan_encoder = NaNEncoder() if one_hot_encode_nan_inputs else None
        m = 2 if one_hot_encode_nan_inputs else 1
        self.proj = torch.nn.Conv1d(m * series_dim, hidden_dim, 1)
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.encoder = Encoder(
            hidden_dim, num_encoder_layers, num_heads=num_heads
        )
        self.decoder = Decoder(
            hidden_dim, num_encoder_layers, num_heads=num_heads
        )

        if exogenous_dim:
            self.proj_exogenous = torch.nn.Conv1d(
                m * exogenous_dim, hidden_dim, 1
            )
        else:
            self.proj_exogenous = None

        self.out = torch.nn.ConvTranspose1d(
            hidden_dim, series_dim, 1
        )
        self.mask_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.predict_ahead = predict_ahead

        self._init()

    def _init(self):
        with torch.no_grad():
            for layer in [self.proj, self.proj_exogenous, self.out]:
                if layer is not None:
                    torch.nn.init.kaiming_normal_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

            self.out.weight /= 8.
            self.encoder._init()
            self.decoder._init()

    def forward(self, x: torch.Tensor,
                exogenous: Optional[torch.Tensor] = None):
        '''
        Args:
            x (:class:`torch.Tensor`): The time series observed so far. This
            should not include exogenous data, if present.
            exogenous (:class:`torch.Tensor`): This should contain the
            exogenous data for both the observed time series and the forecast
            region of the time series. This will be used to determine how far
            to predict ahead.
        '''
        if self.nan_encoder is not None:
            x = self.nan_encoder(x)
            if exogenous is not None:
                exogenous = self.nan_encoder(exogenous)

        x = self.proj(x)

        # Attach mask tokens to time series embedding.
        if self.predict_ahead is not None:
            mask = self.mask_token.view(1, -1, 1)
            mask = mask.repeat(x.shape[0], 1, self.predict_ahead)
            x = torch.cat((x, mask), dim=2)

        # Add exogenous data to time series embedding.
        if self.proj_exogenous is not None:
            if exogenous is None:
                raise ValueError(
                    'Exogenous variables expected but not received'
                )
            x += self.proj_exogenous(exogenous)
        elif exogenous is not None:
            raise ValueError(
                'Exogenous variables received but not expected'
            )

        x = self.time_embedding(x)
        encoder_x = x[:, :, :-self.predict_ahead]
        decoder_x = x[:, :, -self.predict_ahead:]

        # Now that the embeddings are prepped, pass through the encoder.
        encoder_x = self.encoder(encoder_x)
        # And then the decoder.
        x = self.decoder(decoder_x, encoder_x)

        return self.out(x)


class EncoderTransformer(torch.nn.Module):
    '''
    Provides a complete encoder-only transformer network for time series
    forecasting.
    '''
    def __init__(self, series_dim: int, hidden_dim: int, num_layers: int,
                 exogenous_dim: int = 0, predict_ahead: int = 0,
                 num_classes: int = 0, num_heads: int = 8,
                 one_hot_encode_nan_inputs: bool = False):
        '''
        Args:
            series_dim (int): Number of channels in the time series.
            hidden_dim (int): Number of channels in hidden layers.
            num_layers (int): Number of transformer layers.
            exogenous_dim (int): The number of channels in the exogenous data
            used in prediction.
            predict_ahead (int): If provided, how far ahead the network will
            predict, in time steps.
            num_classes (int): If provided, include a classifier
            token predicting this many classes.
            num_heads (int): Number of heads per transformer.
            token_size (int): Spatial width of each token.
            one_hot_encode_nan_inputs (bool): If provided, expect NaNs to be in
            inputs, and use one-hot encoding prior to the projection to handle
            them.
        '''
        super().__init__()

        self.nan_encoder = NaNEncoder() if one_hot_encode_nan_inputs else None
        m = 2 if one_hot_encode_nan_inputs else 1
        self.proj = torch.nn.Conv1d(m * series_dim, hidden_dim, 1)
        self.time_embedding = TimeEmbedding(hidden_dim)
        self.main = Encoder(hidden_dim, num_layers, num_heads=num_heads)

        if exogenous_dim:
            self.proj_exogenous = torch.nn.Conv1d(
                m * exogenous_dim, hidden_dim, 1
            )
        else:
            self.proj_exogenous = None

        if predict_ahead:
            self.out = torch.nn.ConvTranspose1d(
                hidden_dim, series_dim, 1
            )
            self.mask_token = torch.nn.Parameter(torch.randn(hidden_dim))
            self.predict_ahead = predict_ahead
        else:
            self.out = self.mask_token = self.predict_ahead = None

        if num_classes:
            self.class_token = torch.nn.Parameter(torch.randn(hidden_dim))
            self.class_proj = torch.nn.Linear(hidden_dim, num_classes)
        else:
            self.class_token = self.class_proj = None

        self._init()

    def _init(self):
        with torch.no_grad():
            for layer in [self.proj, self.proj_exogenous, self.out,
                          self.class_proj]:
                if layer is not None:
                    torch.nn.init.kaiming_normal_(layer.weight)
                    torch.nn.init.zeros_(layer.bias)

            if self.out is not None:
                self.out.weight /= 8.

            if self.class_token is not None:
                torch.nn.init.zeros_(self.class_token)

            self.main._init()

    def forward(self, x: torch.Tensor,
                exogenous: Optional[torch.Tensor] = None):
        '''
        Args:
            x (:class:`torch.Tensor`): The time series observed so far. This
            should not include exogenous data, if present.
            exogenous (:class:`torch.Tensor`): This should contain the
            exogenous data for both the observed time series and the forecast
            region of the time series. This will be used to determine how far
            to predict ahead.
        '''
        if self.nan_encoder is not None:
            x = self.nan_encoder(x)
            if exogenous is not None:
                exogenous = self.nan_encoder(exogenous)

        x = self.proj(x)

        # Attach mask tokens to time series embedding.
        if self.predict_ahead is not None:
            mask = self.mask_token.view(1, -1, 1)
            mask = mask.repeat(x.shape[0], 1, self.predict_ahead)
            x = torch.cat((x, mask), dim=2)

        # Add exogenous data to time series embedding.
        if self.proj_exogenous is not None:
            if exogenous is None:
                raise ValueError(
                    'Exogenous variables expected but not received'
                )
            x += self.proj_exogenous(exogenous)

        x = self.time_embedding(x)

        # Add class token if needed.
        if self.class_token is not None:
            token = self.class_token.view(1, -1, 1).repeat(x.shape[0], 1, 1)
            x = torch.cat((token, x), dim=2)

        # Now that the embeddings are prepped, pass through the encoder.
        x = self.main(x)

        out = tuple()

        if self.class_proj is not None:
            class_token, x = x[:, :, 0], x[:, :, 1:]
            out = out + (self.class_proj(class_token),)

        if self.out is not None:
            out = out + (self.out(x)[:, :, -self.predict_ahead:],)

        return out[0] if (len(out) == 1) else out
