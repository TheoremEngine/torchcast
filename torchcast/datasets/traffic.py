import os
from typing import Callable, Optional, Union

import torch

from ..data import TensorSeriesDataset
from .utils import _download_and_extract, load_tsf_file

__all__ = ['SanFranciscoTrafficDataset']

TRAFFIC_URL = 'https://zenodo.org/record/4656132/files/traffic_hourly_dataset.zip'  # noqa
TRAFFIC_FILE_NAME = 'traffic_hourly_dataset.tsf'


class SanFranciscoTrafficDataset(TensorSeriesDataset):
    '''
    San Francisco traffic dataset, taken from:

    https://pems.dot.ca.gov

    https://arxiv.org/abs/1703.07015
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
            path = os.path.join(path, TRAFFIC_FILE_NAME)
        if (not os.path.exists(path)) or (download == 'force'):
            if download:
                path = _download_and_extract(
                    TRAFFIC_URL, path,
                    file_name=TRAFFIC_FILE_NAME
                )
            else:
                raise FileNotFoundError(
                    f'San Francisco traffic dataset not found at: {path}'
                )

        data, _ = load_tsf_file(path)

        super().__init__(
            torch.from_numpy(data).unsqueeze(0),
            transform=transform,
            return_length=return_length,
        )
