import os
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch

from ..data import Metadata, TensorSeriesDataset

__all__ = ['LargeSTDataset']


_TASK_TO_DISTRICTS = {
    'gla': [7, 8, 12],
    'gba': [4],
    'sd': [11],
}


class LargeSTDataset(TensorSeriesDataset):
    '''
    This dataset is the LargeST traffic dataset from `Liu et al. 2023
    <https://arxiv.org/abs/2306.08259>`__. It must be downloaded manually from:

        https://www.kaggle.com/datasets/liuxu77/largest

    And unzipped.
    '''
    def __init__(self, root_path: str, years: Union[Iterable[int], int],
                 task: str = 'ca', split: str = 'all', scale: bool = True,
                 resample_time: bool = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            root_path (str): Path to find the dataset at.
            years (int or iterable of ints): Which years to load.
            task (str): Which dataset to retrieve. Choices: 'ca' (meaning all),
                'gba', 'gla', 'sd'.
            split (str): What split of the data to return. The splits are taken
                from Liu et al. Choices: 'all', 'train', 'val', 'test'.
            scale (bool): Whether to normalize the data, as in the benchmark.
            resample_time (bool): Whether to resample time to 15 minute
                intervals, as in Liu et al.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if not os.path.isdir(root_path):
            raise NotADirectoryError(root_path)
        if task not in {'ca', 'gba', 'gla', 'sd'}:
            raise ValueError(task)

        if isinstance(years, int):
            years = [years]

        # Create task-specific list of channels
        if task == 'ca':
            task_ids = None
        else:
            meta_path = os.path.join(root_path, 'ca_meta.csv')
            meta_df = pd.read_csv(meta_path)
            mask = (meta_df['District'] == _TASK_TO_DISTRICTS[task][0])
            for district in _TASK_TO_DISTRICTS[task][1:]:
                mask = mask | (meta_df['District'] == district)
            task_ids = meta_df[mask]['ID'].astype(str).values.tolist()

        data = []
        for year in years:
            path = os.path.join(root_path, f'ca_his_raw_{year}.h5')
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            df = pd.read_hdf(path)
            # Subset on sensor IDs
            if task_ids is not None:
                df = df[task_ids]
            # Apply pre-processing from notebook
            if resample_time:
                df = df.resample('15min').mean().round(0)
            data.append(df)
        data = pd.concat(data)

        meta = [
            Metadata(name='Time'),
            Metadata(name='Data', channel_names=df.columns.tolist()),
        ]

        # Coerce to 1CT arrangement
        t = data.index.values.astype(np.int64).reshape(1, 1, -1)
        data = np.expand_dims(data.values.T.astype(np.float32), axis=0)
        t, data = torch.from_numpy(t), torch.from_numpy(data)

        num_samples = t.shape[2] - return_length + 1
        i_train = round(num_samples * 0.6)
        i_val = i_train + round(num_samples * 0.2)

        if scale:
            std, mean = torch.std_mean(data[:, :, :i_train - 1])
            data = (data - mean) / std

        if split == 'train':
            t = t[:, :, :i_train + return_length - 1]
            data = data[:, :, :i_train + return_length - 1]
        elif split == 'val':
            t = t[:, :, i_train:i_val + return_length - 1]
            data = data[:, :, i_train:i_val + return_length - 1]
        elif split == 'test':
            t, data = t[:, :, i_val:], data[:, :, i_val:]
        elif split != 'all':
            raise ValueError(split)

        super().__init__(
            t, data, return_length=return_length, transform=transform,
            metadata=meta,
        )
