import os
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..data import TensorSeriesDataset
from .utils import _download_and_extract

__all__ = ['ElectricityLoadDataset']

ELECTRICITY_LOAD_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'  # noqa
ELECTRICITY_LOAD_FILE_NAME = 'LD2011_2014.txt'


class ElectricityLoadDataset(TensorSeriesDataset):
    '''
    Electricity Load dataset from:

    https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
    '''
    def __init__(self, path: str, download: Union[str, bool] = False,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (str): Path to find the dataset at.
            download (bool): Whether to download the dataset if it is not
            already available.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        if os.path.isdir(path):
            path = os.path.join(path, ELECTRICITY_LOAD_FILE_NAME)
        if (not os.path.exists(path)) or (download == 'force'):
            if download:
                path = _download_and_extract(
                    ELECTRICITY_LOAD_URL, path,
                    file_name=ELECTRICITY_LOAD_FILE_NAME
                )
            else:
                raise FileNotFoundError(
                    f'Electricity load dataset not found at: {path}'
                )

        df = pd.read_csv(path, delimiter=';', decimal=',')

        date = pd.to_datetime(df.pop('Unnamed: 0'), format='%Y-%m-%d %H:%M:%S')
        date = np.array(date, dtype=np.int64).reshape(1, 1, -1)

        data = np.array(df, dtype=np.float32).T
        data = data.reshape(1, *data.shape)

        super().__init__(
            date, data,
            transform=transform,
            return_length=return_length,
        )
