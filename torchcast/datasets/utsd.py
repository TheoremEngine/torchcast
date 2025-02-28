from functools import lru_cache
import json
import os
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from ..data import TensorSeriesDataset
from .utils import _download_and_extract

__all__ = ['UTSDDataset']


class UTSDDataset(TensorSeriesDataset):
    '''
    This is the `UTSD <https://github.com/thuml/Large-Time-Series-Model>`__
    pre-training dataset, first published in `Liu et al. 2024
    <https://arxiv.org/abs/2402.02368>`__.
    '''
    def __init__(self, task: str, split: str = 'default',
                 path: Optional[str] = None,
                 download: Union[bool, str] = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            task (str): Which dataset to retrieve.
            split (str): Which split to retrieve; choose from 'default',
                'UTSD-1G', 'UTSD-2G', 'UTSD-12G'.
            path (optional, str): Path to find the dataset at.
            download (bool or str): Whether to download the dataset if it is
                not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        manifest = _get_utsd_manifest()

        if split not in manifest:
            raise ValueError(
                f'Did not recognize split {split}; choose from '
                f'{tuple(manifest.keys())}'
            )
        if task not in manifest[split]:
            raise ValueError(
                f'Did not recognize task {task}; choose from '
                f'{tuple(manifest[split].keys())}'
            )

        # First, fetch the data from HuggingFace
        dfs = []
        for remote_path in manifest[split][task]:
            buff = _download_and_extract(
                remote_path,
                os.path.basename(remote_path),
                path,
                download=download,
            )
            df = pd.read_parquet(buff)
            # The item_id channel is a string with the series domain (e.g.
            # 'Health'), series name (e.g. 'MotorImagery'), series index, and
            # channel index, all concatenated by '_'. We begin by breaking them
            # apart into separate columns.
            df[['series_name', 'i_series', 'i_channel']] = \
                df.pop('item_id').str.rsplit('_', n=2, expand=True)
            # And remove the domain from the series name.
            df[['domain', 'series_name']] = df['series_name'].str.split(
                '_', n=1, expand=True
            )
            # We then mask out rows that are not part of the desired task.
            df = df[df['series_name'] == task]
            # And convert the indices to integers.
            df['i_series'] = df['i_series'].astype(int)
            df['i_channel'] = df['i_channel'].astype(int)
            dfs.append(df)
        dfs = pd.concat(dfs)

        # Now that we have our data in memory, the next step is to convert it
        # to a torch.Tensor. However, each series will generally be of varying
        # length. So, first, we break it up by series index. This should now
        # also be sorted by series.
        dfs = [df for _, df in dfs.groupby('i_series')]
        # I don't know why, but this can end up with duplicate rows.
        dfs = [df.drop_duplicates('i_channel') for df in dfs]
        # And sort the individual series by channel index.
        dfs = [df.sort_values('i_channel') for df in dfs]
        tensors = [
            torch.from_numpy(np.stack(df['target'].values.tolist(), axis=0))
            for df in dfs
        ]

        super().__init__(
            tensors,
            transform=transform,
            return_length=return_length,
        )


@lru_cache
def _get_utsd_manifest():
    '''
    This retrieves the JSON containing the index of which Parquet files contain
    which series. It is broken out as a separate function so that we can
    lru_cache the output.
    '''
    path = os.path.join(os.path.dirname(__file__), 'utsd_manifest.json')
    with open(path, 'r') as f:
        return json.load(f)
