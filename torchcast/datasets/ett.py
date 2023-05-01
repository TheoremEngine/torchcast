import os
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_and_extract

__all__ = ['ElectricityTransformerDataset']

ETT_URL = 'https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/{name}'

ETT_FILE_NAMES = {
    '15min': ['ETTm1.csv', 'ETTm2.csv'],
    'hourly': ['ETTh1.csv', 'ETTh2.csv'],
}

COLUMN_NAME_MAP = {
    'HUFL': 'High Useful Load',
    'HULL': 'High Useless Load',
    'MUFL': 'Middle Useful Load',
    'MULL': 'Middle Useless Load',
    'LUFL': 'Low Useful Load',
    'LULL': 'Low Useless Load',
    'OT': 'Oil Temperature'
}


class ElectricityTransformerDataset(TensorSeriesDataset):
    '''
    This is the Zhou et al. electricity transformer dataset, obtained from:

        https://github.com/zhouhaoyi/ETDataset
    '''
    def __init__(self, path: str, task: str = '15min',
                 download: Union[bool, str] = False,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (str): Path to find the dataset at. This should be a
            directory, as the dataset consists of two files.
            task (str): Whether to download the hourly dataset or the every 15
            minute dataset. Choices: 'hourly', '15min'.
            download (bool or str): Whether to download the dataset if it is
            not already available. Choices: True, False, 'force'.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        if task not in ETT_FILE_NAMES:
            raise ValueError(task)

        dfs = []

        for site, name in enumerate(ETT_FILE_NAMES[task]):
            file_path = os.path.join(path, name)
            url = ETT_URL.format(name=name)
            if (
                (download == 'force') or
                (download and (not os.path.exists(file_path)))
            ):
                _download_and_extract(url, file_path, file_name=name)

            if os.path.exists(file_path):
                dfs.append(pd.read_csv(file_path))
            else:
                dfs.append(pd.read_csv(url))
            dates = dfs[-1].pop('date')

        dates = pd.to_datetime(dates, format='%Y-%m-%d %H:%M:%S')
        dates = np.array(dates, dtype=np.int64).reshape(1, 1, -1)

        target = [np.array(df.pop('OT'), dtype=np.float32) for df in dfs]
        target = np.stack(target, axis=0).reshape(2, 1, -1)
        target_meta = Metadata(channel_names=['Oil Temperature'])

        channel_names = [COLUMN_NAME_MAP[col] for col in dfs[0].columns]
        pred = [np.array(df, dtype=np.float32).T for df in dfs]
        pred = np.stack(pred, axis=0)
        pred_meta = Metadata(channel_names=channel_names)

        super().__init__(
            dates, pred, target,
            transform=transform,
            return_length=return_length,
            metadata=[None, pred_meta, target_meta],
        )
