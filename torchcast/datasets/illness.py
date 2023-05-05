import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..data import Metadata, TensorSeriesDataset

__all__ = ['ILIDataset']


class ILIDataset(TensorSeriesDataset):
    '''
    This dataset describes both the raw number of patients with influenza-like
    symptoms and the ratio of those patients to the total number of patients in
    the US, obtained from the CDC. This must be manually downloaded from:

        https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

    To download this dataset, click "Download Data". Unselect "WHO/NREVSS" and
    select the desired seasons, then click "Download Data".
    '''
    def __init__(self, path: str, transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (str): Path to find the dataset at.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        df = pd.read_csv(path, skiprows=1, na_values='X')

        # Drop unneeded columns
        del df['REGION TYPE'], df['REGION']

        # Extract dates
        date = np.array(df[['YEAR', 'WEEK']]).T.reshape(1, 2, -1)
        date_meta = Metadata(name='Date', channel_names=['YEAR', 'WEEK'])
        del df['YEAR'], df['WEEK']

        # Convert data columns to float
        for col in df.columns:
            df[col] = df[col].astype(np.float32)
        data = np.array(df).T.reshape(1, 11, -1)
        data_meta = Metadata(name='Data', channel_names=df.columns)

        super().__init__(
            date, data,
            transform=transform,
            return_length=return_length,
            metadata=[date_meta, data_meta],
        )
