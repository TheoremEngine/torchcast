from datetime import datetime, timedelta
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from ..data import Metadata, TensorSeriesDataset
from ._file_readers import parse_tsf
from .utils import _download_and_extract, _split_7_1_2

__all__ = ['SanFranciscoTrafficDataset']

TRAFFIC_URL = 'https://zenodo.org/record/4656132/files/traffic_hourly_dataset.zip'  # noqa
TRAFFIC_FILE_NAME = 'traffic_hourly_dataset.tsf'


class SanFranciscoTrafficDataset(TensorSeriesDataset):
    '''
    San Francisco traffic dataset, taken from:

        https://pems.dot.ca.gov

        https://arxiv.org/abs/1703.07015
    '''
    def __init__(self, path: Optional[str] = None, split: str = 'all',
                 download: Union[str, bool] = True, scale: bool = True,
                 columns_as_channels: bool = True,
                 transform: Optional[Callable] = None,
                 input_margin: Optional[int] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (optional, str): Path to find the dataset at.
            split (str): What split of the data to return. The splits are taken
                from Zeng et al. Choices: 'all', 'train', 'val', 'test'.
            scale (bool): Whether to normalize the data, as in the benchmark.
            download (bool): Whether to download the dataset if it is not
                already available.
            columns_as_channels (bool): If true, each column is treated as a
                separate channel. If false, each column is treated as a
                separate series.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            input_margin (optional, int): The amount of margin to include on
                the left-hand side of the dataset, as it is used as an input to
                the model.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        buff = _download_and_extract(
            TRAFFIC_URL,
            TRAFFIC_FILE_NAME,
            path,
            download=download,
        )

        data, _ = parse_tsf(buff.read())
        data = torch.from_numpy(data).permute(1, 0, 2)

        # The data starts at 2015-01-01 00:00:00. Per the README at:
        # https://github.com/laiguokun/multivariate-time-series-data
        # The data is taken hourly.
        t = pd.date_range(
            datetime(2015, 1, 1),
            datetime(2015, 1, 1) + timedelta(hours=(data.shape[2] - 1)),
            freq='H',
        )
        t = torch.from_numpy(t.astype(np.int64).values).view(1, 1, -1)

        if scale:
            train_data = _split_7_1_2('train', input_margin, data)
            mean, std = train_data.mean((0, 2)), train_data.std((0, 2))
            data = (data - mean.reshape(1, -1, 1)) / std.reshape(1, -1, 1)

        data, t = _split_7_1_2(split, input_margin, data, t)

        if not columns_as_channels:
            data = data.permute(1, 0, 2)

        super().__init__(
            t, data,
            transform=transform,
            return_length=return_length,
            metadata=[Metadata(name='Datetime'), Metadata(name='Rate')],
        )
