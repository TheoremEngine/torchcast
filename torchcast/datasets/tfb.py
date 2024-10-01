import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_from_google_drive_and_extract

__all__ = ['TFBDataset']


GOOGLE_ID = '1vgpOmAygokoUt235piWKUjfwao6KwLv7'


DATE_FORMATS = [
    '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M', '%Y-%m-%d',
    '%Y-%m-%d %H:%M:%S-01:00'
]

# Most tasks in TFB use a 7-1-2 split, but some use a 6-2-2 split. See
# SPLIT_DICT in ts_benchmark.evaluation.strategy.rolling_forecast, or Table 5
# in the paper.
BIG_VAL_TASKS = {
    'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08',
    'Solar', 'AQShunyi', 'AQWan',
}


class TFBDataset(TensorSeriesDataset):
    '''
    This dataset provides the `Qiu et al. 2024
    <https://arxiv.org/abs/2403.20150>`__ TFB Time Series Forecasting Benchmark
    collection, obtained from:

        https://github.com/decisionintelligence/TFB

    This collection includes the LTSF collection as a subset, but the data is
    pre-processed and split in different ways.
    '''
    _tasks: Optional[List[str]] = None

    def __init__(self, task: str, split: str = 'all',
                 path: Optional[str] = None, download: bool = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            task (str): Which dataset to retrieve.
            split (str): What split of the data to return. The splits are taken
                from Qiu et al. Choices: 'all', 'train', 'val', 'test'.
            path (optional, str): Path to find the dataset at.
            download (bool or str): Whether to download the dataset if it is
                not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if task not in self.tasks:
            raise ValueError(f'Did not recognize {task}')

        buff = _download_from_google_drive_and_extract(
            GOOGLE_ID, file_name=f'forecasting/{task}.csv',
            remote_name='forecasting.zip', local_path=path,
        )
        # When downloaded, the csv has three columns: 'date', 'data', and
        # 'cols'. For example, for a multivariate time series with variables
        # 'A' and 'B', there would be two rows for each date:
        #
        # '2000-1-1 00:00:00',1.2,'A'
        # '2000-1-1 00:00:00',2.3,'B'
        # '2000-1-2 00:00:00',1.4,'A'
        # ...
        #
        # Note that the date is initially a string, not a datetime.
        df = pd.read_csv(buff)

        # First, convert the date into a datetime, to make sure we preserve the
        # correct ordering in our next operation. Unfortunately, the datetimes
        # are not in a consistent format. In some cases, they're not even
        # dates. We can catch those by checking the type, since pandas will
        # auto-convert them to floats or ints. However, note that this requires
        # ==, not is, because it's a pandas datatype, not a numpy datatype.
        if (df['date'].dtype != np.float64) and (df['date'].dtype != np.int64):
            for fmt in DATE_FORMATS:
                try:
                    df['date'] = pd.to_datetime(df['date'], format=fmt)
                except ValueError:
                    continue
                else:
                    break
            else:
                raise ValueError(
                    f"Could not parse datetime column: {df['date'][0]}"
                )

        # Next, pivot the dataframe.
        df = df.pivot(index='date', columns='cols', values='data')

        # And extract values.
        data = torch.from_numpy(np.array(df.astype(np.float32)))
        data = data.T.unsqueeze(0)
        meta = Metadata(channel_names=df.columns.tolist())

        if (df.index.dtype == np.float64) or (df.index.dtype == np.int64):
            data, meta = (data,), [meta]
        else:
            t = torch.from_numpy(np.array(df.index.astype(np.int64)))
            t = t.view(1, 1, -1)
            data = (t, data)
            meta = [Metadata(name='Datetime'), meta]

        train_percent = 0.75 if (task in BIG_VAL_TASKS) else 0.875
        data = _split_tfb(split, train_percent, *data)

        super().__init__(
            *data, return_length=return_length, transform=transform,
            metadata=meta,
        )

    @property
    def tasks(self) -> List[str]:
        if self._tasks is None:
            path = os.path.join(os.path.dirname(__file__), 'tfb.txt')
            with open(path, 'r') as tasks_file:
                self._tasks = [row[:-1] for row in tasks_file.readlines()]
        return self._tasks


def _split_tfb(split: str, train_percent: float, *tensors: torch.Tensor):
    '''
    Replicates the train-val-test split used in Qiu et al. The split is either
    7-1-2 or 6-2-2, depending on the dataset, with the choices shown in Table
    5.

    Args:
        split (str): Split to return. Choices: 'all', 'train', 'val', 'test'.
        train_percent (float): Percent of the non-test data used in training.
    '''
    ts = {x.shape[2] for x in tensors if x.shape[2] > 1}
    if len(ts) > 1:
        raise ValueError(f'Received conflicting time lengths: {ts}')

    if split in {'train', 'val', 'test'}:
        num_all = max(x.shape[2] for x in tensors)
        num_test = num_all - int(0.8 * num_all)
        num_train = int((num_all - num_test) * train_percent)
        num_val = num_all - (num_train + num_test)
        if split == 'train':
            t_0, t_1 = 0, num_train
        elif split == 'val':
            t_0, t_1 = num_train, num_train + num_val
        else:
            t_0, t_1 = num_train + num_val, num_all
        tensors = tuple(x[:, :, t_0:t_1] for x in tensors)
        return tensors

    elif split == 'all':
        return tensors

    else:
        raise ValueError(split)
