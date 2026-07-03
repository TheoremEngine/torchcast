import os
from typing import Callable, Optional, Union

import pandas as pd
import torch

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_and_extract

__all__ = ['TailedTSDataset']

TAILEDTS_URL = 'https://zenodo.org/records/19562496/files/'
TAILEDTS_FILE_NAME = 'data-2024{n:02}.parquet'


class TailedTSDataset(TensorSeriesDataset):
    def __init__(self, path: Optional[str] = None,
                 download: Union[bool, str] = False,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (optional, str): Path to find the dataset at. This should be
                a path to a directory containing the parquet files.
            download (bool or str): Whether to download the dataset if it is
                not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        df = []
        for n in range(1, 13):
            name = TAILEDTS_FILE_NAME.format(n=n)
            file_path = None if (path is None) else os.path.join(path, name)
            buff = _download_and_extract(
                os.path.join(TAILEDTS_URL, name),
                name,
                file_path,
                download=download,
            )
            df.append(pd.read_parquet(buff))
        df = pd.concat(df)

        value_columns = sorted(c for c in df.columns if c.startswith('count_'))
        values = df[value_columns].values
        values = torch.from_numpy(values).float().unsqueeze(1)

        metadata = [Metadata(series_names=df['page_title'].tolist())]

        super().__init__(
            values,
            metadata=metadata,
            return_length=return_length,
            transform=transform,
        )
