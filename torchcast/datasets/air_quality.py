from datetime import datetime
import os
from typing import Callable, Optional

import torch

from ..data import TensorSeriesDataset
from .utils import _download_and_extract, _load_csv_file

__all__ = ['AirQualityDataset']

AIR_QUALITY_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip'  # noqa
AIR_QUALITY_FILE_NAME = 'AirQualityUCI.csv'

AIR_QUALITY_KEYS = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T',
    'RH', 'AH'
]


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
        data = _load_csv_file(path, delimiter=';')
        # Drop empty column.
        data.pop('')
        # Drop empty rows.
        data = {k: [v for v in vs if v] for k, vs in data.items()}
        # Extract time.
        t = [
            (datetime.strptime(f'{d} {t}', '%d/%m/%Y %H.%M.%S')
             - datetime.min).seconds
            for d, t in zip(data.pop('Date'), data.pop('Time'))
        ]
        # All other values should be castable to float, except that the decimal
        # point is a ',' instead of '.'. Easily fixed.
        data = [
            [float(v.replace(',', '.')) for v in data[k]]
            for k in AIR_QUALITY_KEYS
        ]
        data = [t] + data
        data = torch.tensor(data, dtype=torch.float32)
        # A value of -200 denotes a NaN
        data[data == -200] = float('nan')
        # Coerce to NCT arrangement
        data = data.unsqueeze(0)
        super().__init__(
            data,
            transform=transform,
            return_length=return_length,
            channel_names=AIR_QUALITY_KEYS,
        )
