import os
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
    def __init__(self, path: str, split: str = 'all',
                 download: Union[bool, str] = False,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (str): Path to find the dataset at. This should be a
            directory, as the dataset consists of two files.
            split (str): What split of the data to return. The splits are taken
            from Zeng et al. Choices: 'all', 'train', 'val', 'test'.
            download (bool or str): Whether to download the dataset if it is
            not already available. Choices: True, False, 'force'.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        if os.path.isdir(path):
            path = os.path.join(path, EXCHANGE_RATE_FILE_NAME)
        if ((download == 'force') or (download and not os.path.exists(path))):
            path = _download_and_extract(
                EXCHANGE_RATE_URL, path, file_name=EXCHANGE_RATE_FILE_NAME,
            )
        if os.path.exists(path):
            df = pd.read_csv(path, header=None)
        else:
            df = pd.read_csv(EXCHANGE_RATE_URL, header=None)

        data = np.array(df, dtype=np.float32).T
        data = data.reshape(1, *data.shape)
        # In the pre-processing applied by Zeng et al., the last two channels
        # are swapped. To ensure replicability, we repeat that here.
        data = data[:, [0, 1, 2, 3, 4, 5, 7, 6], :]

        data = _split_7_1_2(split, data)

        super().__init__(
            data,
            transform=transform,
            return_length=return_length,
        )
