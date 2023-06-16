from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..data import TensorSeriesDataset
from .utils import _download_and_extract, _split_7_1_2

__all__ = ['ElectricityLoadDataset']

ELECTRICITY_LOAD_URL = 'https://github.com/laiguokun/multivariate-time-series-data/raw/master/electricity/electricity.txt.gz'  # noqa
ELECTRICITY_LOAD_FILE_NAME = 'electricity.txt'


class ElectricityLoadDataset(TensorSeriesDataset):
    '''
    Electricity Load dataset, obtained from:

        https://github.com/laiguokun/multivariate-time-series-data

    This is derived from:

        https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

    But the data has been subsetted and pre-processed. This implementation is
    based on:

        https://github.com/cure-lab/LTSF-Linear
    '''
    def __init__(self, path: Optional[str] = None, split: str = 'all',
                 download: Union[str, bool] = True, scale: bool = True,
                 transform: Optional[Callable] = None,
                 input_margin: Optional[int] = 336,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (optional, str): Path to find the dataset at.
            split (str): What split of the data to return. The splits are taken
            from Zeng et al. Choices: 'all', 'train', 'val', 'test'.
            download (bool): Whether to download the dataset if it is not
            already available.
            scale (bool): Whether to normalize the data, as in the benchmark.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            input_margin (optional, int): The amount of margin to include on
            the left-hand side of the dataset, as it is used as an input to the
            model.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        buff = _download_and_extract(
            ELECTRICITY_LOAD_URL,
            ELECTRICITY_LOAD_FILE_NAME,
            path,
            download=download,
        )

        df = pd.read_csv(buff, header=None)

        data = np.array(df, dtype=np.float32).T
        data = data.reshape(1, *data.shape)

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
