from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_and_extract

__all__ = ['ElectricityTransformerDataset']

ETT_URL = 'https://github.com/zhouhaoyi/ETDataset/raw/main/ETT-small/{name}'

ETT_FILE_NAMES = {
    '15min': ['ETTm1.csv', 'ETTm2.csv'],
    '15min-1': ['ETTm1.csv'],
    '15min-2': ['ETTm2.csv'],
    'hourly': ['ETTh1.csv', 'ETTh2.csv'],
    'hourly-1': ['ETTh1.csv'],
    'hourly-2': ['ETTh2.csv'],
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

DATA_SPLITS = {
    'train': (0, 12 * 30 * 24),
    'val': (12 * 30 * 24, 16 * 30 * 24),
    'test': (16 * 30 * 24, 20 * 40 * 24),
}


class ElectricityTransformerDataset(TensorSeriesDataset):
    '''
    This is the Zhou et al. electricity transformer dataset, obtained from:

        https://github.com/zhouhaoyi/ETDataset

    This implementation is based on:

        https://github.com/cure-lab/LTSF-Linear
    '''
    def __init__(self, path: Optional[str] = None, task: str = '15min',
                 split: str = 'all', download: Union[bool, str] = True,
                 scale: bool = True,
                 columns_as_channels: bool = True,
                 transform: Optional[Callable] = None,
                 input_margin: Optional[int] = 336,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (optional, str): Path to find the dataset at. This should be a
            directory, as the dataset consists of multiple files.
            task (str): Whether to retrieve the hourly dataset or the every 15
            minute dataset, and whether to retrieve one file or two. Choices:
            'hourly', 'hourly-1', 'hourly-2', '15min', '15min-1', '15min-2'.
            split (str): What split of the data to return. The splits are taken
            from Zeng et al. Choices: 'all', 'train', 'val', 'test'.
            download (bool or str): Whether to download the dataset if it is
            not already available. Choices: True, False, 'force'.
            scale (bool): Whether to normalize the data, as in the benchmark.
            columns_as_channels (bool): If true, each column is treated as a
            separate channel. If false, each column is treated as a separate
            series.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            input_margin (optional, int): The amount of margin to include on
            the left-hand side of the dataset, as it is used as an input to the
            model.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        if task not in ETT_FILE_NAMES:
            raise ValueError(task)

        dfs = []
        file_names = ETT_FILE_NAMES[task]

        for name in file_names:
            url = ETT_URL.format(name=name)
            buff = _download_and_extract(url, name, path, download=download)
            dfs.append(pd.read_csv(buff))
            dates = dfs[-1].pop('date')

        dates = pd.to_datetime(dates, format='%Y-%m-%d %H:%M:%S')
        dates = np.array(dates, dtype=np.int64).reshape(1, 1, -1)
        date_meta = Metadata(name='Datetime')

        target = [np.array(df.pop('OT'), dtype=np.float32) for df in dfs]
        target = np.stack(target, axis=0).reshape(len(file_names), 1, -1)
        target_meta = Metadata(
            name='Target',
            channel_names=['Oil Temperature'],
        )

        channel_names = [COLUMN_NAME_MAP[col] for col in dfs[0].columns]
        pred = [np.array(df, dtype=np.float32).T for df in dfs]
        pred = np.stack(pred, axis=0)
        pred_meta = Metadata(
            name='Predictors',
            channel_names=channel_names,
        )

        if scale:
            t_0, t_1 = DATA_SPLITS['train']
            if task.startswith('15min'):
                t_0, t_1 = t_0 * 4, t_1 * 4
            pred_mean = pred[:, :, t_0:t_1].mean((0, 2)).reshape(1, -1, 1)
            pred_std = pred[:, :, t_0:t_1].std((0, 2)).reshape(1, -1, 1)
            pred = (pred - pred_mean) / pred_std
            target_mean = target[:, :, t_0:t_1].mean((0, 2)).reshape(1, -1, 1)
            target_std = target[:, :, t_0:t_1].std((0, 2)).reshape(1, -1, 1)
            target = (target - target_mean) / target_std

        # This uses a custom split, unlike the other datasets from Zeng et al.
        if split in {'train', 'val', 'test'}:
            t_0, t_1 = DATA_SPLITS[split]
            if task.startswith('15min'):
                t_0, t_1 = t_0 * 4, t_1 * 4
            if split in {'val', 'test'}:
                t_0 -= (input_margin or 0)
            dates = dates[:, :, t_0:t_1]
            pred = pred[:, :, t_0:t_1]
            target = target[:, :, t_0:t_1]
        elif split != 'all':
            raise ValueError(split)

        if not columns_as_channels:
            data = data.permute(1, 0, 2)

        super().__init__(
            dates, pred, target,
            transform=transform,
            return_length=return_length,
            metadata=[date_meta, pred_meta, target_meta],
        )
