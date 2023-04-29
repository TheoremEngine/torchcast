import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_and_extract

__all__ = ['AirQualityDataset']

AIR_QUALITY_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'  # noqa
AIR_QUALITY_FILE_NAME = 'AirQualityUCI.csv'


class AirQualityDataset(TensorSeriesDataset):
    '''
    This is the De Vito et al. air quality dataset.
    '''
    def __init__(self, path: str, download: bool = False,
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
            path = os.path.join(path, AIR_QUALITY_FILE_NAME)
        if not os.path.exists(path):
            if download:
                path = _download_and_extract(
                    AIR_QUALITY_URL, path, file_name=AIR_QUALITY_FILE_NAME
                )
            else:
                raise FileNotFoundError(
                    f'Air Quality dataset not found at: {path}'
                )

        # This will return a dictionary mapping keys to lists
        df = pd.read_csv(path, delimiter=';', decimal=',')
        # Drop empty columns and rows
        df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
        # Extract time.
        t = df.pop('Date') + ' ' + df.pop('Time')
        t = pd.to_datetime(t, format='%d/%m/%Y %H.%M.%S')
        # This converts time to nanoseconds, and we want seconds.
        t = torch.from_numpy(np.array(t.astype(np.int64))) // 1_000_000_000
        # Coerce to NCT arrangement
        t = t.view(1, 1, -1)
        # A value of -200 denotes a NaN
        df[df == -200] = float('nan')
        # Convert to torch.tensor and coerce to NCT arrangement
        data = torch.from_numpy(np.array(df, dtype=np.float32))
        data = data.permute(1, 0).unsqueeze(0)
        super().__init__(
            t, data,
            transform=transform,
            return_length=return_length,
            metadata=[None, Metadata(channel_names=list(df.columns))],
        )
