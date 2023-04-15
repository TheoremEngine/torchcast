import os
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import torch

from ..data import TensorSeriesDataset
from .utils import _download_and_extract

__all__ = ['NEONAquaticDataset', 'NEONTerrestrialDataset']

NEON_AQUATIC_URL = 'https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-targets.csv.gz'  # noqa
NEON_AQUATIC_FILE_NAME = 'aquatics-targets.csv'
NEON_AQUATIC_KEYS = ['temperature', 'chla', 'oxygen']

NEON_TERRA_URL = 'https://data.ecoforecast.org/neon4cast-targets/terrestrial_daily/terrestrial_daily-targets.csv.gz'  # noqa
NEON_TERRA_FILE_NAME = 'terrestrial_daily-targets.csv'
NEON_TERRA_KEYS = ['le', 'nee']


class NEONDataset(TensorSeriesDataset):
    '''
    This is a base class for NEON ecoforecast datasets.
    '''
    def __init__(self, path: str, url: str, file_name: str,
                 download: bool = False,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        # Make sure data is in place.
        if os.path.isdir(path):
            path = os.path.join(path, file_name)

        if not os.path.exists(path) or (download == 'force'):
            if download:
                path = _download_and_extract(url, path, file_name=file_name)
            else:
                raise FileNotFoundError(
                    f'NEON dataset not found at: {path}'
                )

        df = pd.read_csv(path)
        # Remove NaN values
        df = df.dropna()
        # Convert dates to integers
        df['datetime'] = pd.to_datetime(df.pop('datetime'), format='%Y-%m-%d')
        # Dates is stored in nanoseconds in pandas, and we want it in days.
        df['datetime'] = df['datetime'].astype(np.int64)
        df['datetime'] //= (24 * 60 * 60 * 1_000_000_000)

        # Construct dictionary mapping site_ID to index, and variable to index,
        # and apply to the dataframe.
        sites = list(df['site_id'].unique())
        site_ids = {site: i for i, site in enumerate(sites)}
        df['site_id'] = df['site_id'].replace(site_ids)
        channel_names = list(df['variable'].unique())
        var_ids = {k: i for i, k in enumerate(channel_names)}
        df['variable'] = df['variable'].replace(var_ids)

        # Build buffer
        site_date_min = df.groupby('site_id')['datetime'].min()
        site_date_max = df.groupby('site_id')['datetime'].max()
        n_t = (site_date_max - site_date_min).max() + 1
        buff = torch.full(
            (len(sites), len(channel_names), n_t), float('nan'),
            dtype=torch.float32
        )

        # Add data to buffer. We iterate by index instead of by row to preserve
        # the dtype. TODO: There has to be a better way to do this.
        for row in df.index:
            site, col = df['site_id'][row], df['variable'][row]
            t = df['datetime'][row] - site_date_min[site]
            buff[site, col, t] = df['observation'][row]

        # Build dates and coerce to NCT arrangement.
        dates = [
            torch.arange(site_min, site_min + n_t)
            for site_min in site_date_min
        ]
        dates = torch.stack(dates, dim=0).unsqueeze(1)

        super().__init__(
            dates, buff,
            return_length=return_length,
            transform=transform,
            channel_names=channel_names,
            series_names=sites,
        )


class NEONAquaticDataset(NEONDataset):
    def __init__(self, path: str, download: Union[bool, str] = False,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (str): Path to find the dataset at.
            download (bool, str): Whether to download the dataset if it is not
            already available. Since the NEON datasets are updated daily, this
            can also be set to the string "force", which will redownload the
            data even if the data is already present.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        super().__init__(
            path,
            url=NEON_AQUATIC_URL,
            file_name=NEON_AQUATIC_FILE_NAME,
            return_length=return_length,
            transform=transform,
            download=download,
        )


class NEONTerrestrialDataset(NEONDataset):
    def __init__(self, path: str, download: Union[bool, str] = False,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (str): Path to find the dataset at.
            download (bool, str): Whether to download the dataset if it is not
            already available. Since the NEON datasets are updated daily, this
            can also be set to the string "force", which will redownload the
            data even if the data is already present.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        super().__init__(
            path,
            url=NEON_TERRA_URL,
            file_name=NEON_TERRA_FILE_NAME,
            return_length=return_length,
            transform=transform,
            download=download,
        )
