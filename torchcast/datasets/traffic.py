from typing import Callable, Optional, Union

import torch

from ..data import TensorSeriesDataset
from ._file_readers import parse_tsf
from .utils import _download_and_extract, _split_7_1_2

__all__ = ['SanFranciscoTrafficDataset']

TRAFFIC_URL = 'https://zenodo.org/record/4656132/files/traffic_hourly_dataset.zip'  # noqa
TRAFFIC_FILE_NAME = 'traffic_hourly_dataset.tsf'


class SanFranciscoTrafficDataset(TensorSeriesDataset):
    '''
    San Francisco traffic dataset, taken from:

    https://pems.dot.ca.gov

    https://arxiv.org/abs/1703.07015
    '''
    def __init__(self, path: Optional[str] = None, split: str = 'all',
                 scale: bool = True, download: Union[str, bool] = True,
                 transform: Optional[Callable] = None,
                 input_margin: Optional[int] = None,
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
            TRAFFIC_URL,
            TRAFFIC_FILE_NAME,
            path,
            download=download,
        )

        data, _ = parse_tsf(buff.read())
        data = torch.from_numpy(data).permute(1, 0, 2)

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
