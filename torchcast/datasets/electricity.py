from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_and_extract, _split_7_1_2

__all__ = ['ElectricityLoadDataset']

ELECTRICITY_LOAD_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'  # noqa
ELECTRICITY_LOAD_FILE_NAME = 'LD2011_2014.txt'


class ElectricityLoadDataset(TensorSeriesDataset):
    '''
    Electricity Load dataset from:

    https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

    This implementation is based on:

        https://github.com/cure-lab/LTSF-Linear
    '''
    def __init__(self, path: Optional[str] = None, split: str = 'all',
                 download: Union[str, bool] = True,
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

        df = pd.read_csv(buff, delimiter=';', decimal=',')

        date = pd.to_datetime(df.pop('Unnamed: 0'), format='%Y-%m-%d %H:%M:%S')
        date = np.array(date, dtype=np.int64).reshape(1, 1, -1)

        data = np.array(df, dtype=np.float32).T
        data = data.reshape(1, *data.shape)

        date, data = _split_7_1_2(split, input_margin, date, data)

        super().__init__(
            date, data,
            transform=transform,
            return_length=return_length,
            metadata=[Metadata(name='Datetime'),
                      Metadata(name='Electricity Load')],
        )
