from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from ..data import Metadata, TensorSeriesDataset
from .utils import _decode, _download_and_extract

__all__ = ['AirQualityDataset']

AIR_QUALITY_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'  # noqa
AIR_QUALITY_FILE_NAME = 'AirQualityUCI.csv'


class AirQualityDataset(TensorSeriesDataset):
    '''
    This is the `De Vito et al. 2008
    <https://doi.org/10.1016/J.SNB.2007.09.060>`__ air quality dataset.
    '''
    def __init__(self, path: Optional[str] = None,
                 download: Union[bool, str] = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (optional, str): Path to find the dataset at.
            download (bool or str): Whether to download the dataset if it is
                not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        buff = _download_and_extract(
            AIR_QUALITY_URL,
            AIR_QUALITY_FILE_NAME,
            path,
            download=download,
        )
        buff = _decode(buff)

        # This will return a dictionary mapping keys to lists
        df = pd.read_csv(buff, delimiter=';', decimal=',')
        # Drop empty columns and rows
        df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)

        # Extract time.
        t = df.pop('Date') + ' ' + df.pop('Time')
        t = pd.to_datetime(t, format='%d/%m/%Y %H.%M.%S')
        t = torch.from_numpy(np.array(t.astype(np.int64)))
        # Coerce to NCT arrangement
        t = t.view(1, 1, -1)
        time_meta = Metadata(name='Datetime')

        # A value of -200 denotes a NaN
        df[df == -200] = float('nan')
        # Convert to torch.tensor and coerce to NCT arrangement
        data = torch.from_numpy(np.array(df, dtype=np.float32))
        data = data.permute(1, 0).unsqueeze(0)
        data_meta = Metadata(name='Data', channel_names=list(df.columns))

        super().__init__(
            t, data,
            transform=transform,
            return_length=return_length,
            metadata=[time_meta, data_meta],
        )
