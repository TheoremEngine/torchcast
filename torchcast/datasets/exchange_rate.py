from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..data import TensorSeriesDataset
from .utils import _download_and_extract, _split_7_1_2

__all__ = ['ExchangeRateDataset']

EXCHANGE_RATE_URL = 'https://github.com/laiguokun/multivariate-time-series-data/raw/master/exchange_rate/exchange_rate.txt.gz'  # noqa
EXCHANGE_RATE_FILE_NAME = 'exchange_rate.txt'


class ExchangeRateDataset(TensorSeriesDataset):
    '''
    This is a record of currency exchange rates, taken from:

        https://github.com/laiguokun/multivariate-time-series-data

        https://arxiv.org/abs/1703.07015
    '''
    def __init__(self, path: Optional[str] = None, split: str = 'all',
                 scale: bool = True, download: Union[bool, str] = True,
                 transform: Optional[Callable] = None,
                 input_margin: Optional[int] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (optional, str): Path to find the dataset at.
            split (str): What split of the data to return. The splits are taken
            from Zeng et al. Choices: 'all', 'train', 'val', 'test'.
            scale (bool): Whether to normalize the data, as in the benchmark.
            download (bool or str): Whether to download the dataset if it is
            not already available. Choices: True, False, 'force'.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            input_margin (optional, int): The amount of margin to include on
            the left-hand side of the dataset, as it is used as an input to the
            model.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        buff = _download_and_extract(
            EXCHANGE_RATE_URL,
            EXCHANGE_RATE_FILE_NAME,
            path,
            download=download
        )

        df = pd.read_csv(buff, header=None)

        data = np.array(df, dtype=np.float32).T
        data = data.reshape(1, *data.shape)
        # In the pre-processing applied by Zeng et al., the last two channels
        # are swapped. To ensure replicability, we repeat that here.
        data = data[:, [0, 1, 2, 3, 4, 5, 7, 6], :]

        if scale:
            train_data = _split_7_1_2('train', input_margin, data)
            mean, std = train_data.mean((0, 2)), train_data.std((0, 2))
            data = (data - mean.reshape(1, -1, 1)) / std.reshape(1, -1, 1)

        data = _split_7_1_2(split, input_margin, data)

        super().__init__(
            data,
            transform=transform,
            return_length=return_length,
        )
