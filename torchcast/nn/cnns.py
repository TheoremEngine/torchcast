import torch

from .hooks import max_norm_constraint
from .layers import NaNEncoder

__all__ = ['EEGNet']


class EEGNet(torch.nn.Sequential):
    '''
    Based on:

        https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
        https://github.com/vlawhern/arl-eegmodels
    '''
    def __init__(self, num_input_channels: int, num_output_channels: int,
                 hidden_dim: int = 8, one_hot_encode_nan_inputs: bool = False,
                 window_size: int = 128):
        if one_hot_encode_nan_inputs:
            num_input_channels *= 2
            blocks = [NaNEncoder()]
        else:
            blocks = []

        blocks += [
            torch.nn.Conv1d(num_input_channels, hidden_dim, 64, bias=False,
                            padding='same'),
            torch.nn.BatchNorm1d(hidden_dim),
            # Depthwise convolution
            # TODO: Add max_norm constraint
            torch.nn.Conv1d(hidden_dim, hidden_dim * 2, 64, groups=hidden_dim,
                            bias=False, padding='same'),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.AvgPool1d(4),
            torch.nn.Dropout(0.5),
            # Separable convolution
            torch.nn.Conv1d(hidden_dim * 2, hidden_dim * 2, 16,
                            groups=(hidden_dim * 2), bias=False,
                            padding='same'),
            torch.nn.Conv1d(hidden_dim * 2, hidden_dim * 2, 1, bias=False),
            torch.nn.BatchNorm1d(hidden_dim * 2),
            torch.nn.ELU(),
            torch.nn.AvgPool1d(8),
            torch.nn.Dropout(0.5),
            torch.nn.Flatten(),
            # TODO: Add max_norm constraint
            torch.nn.Linear(hidden_dim // 16, num_output_channels),
        ]
        super().__init__(*blocks)
