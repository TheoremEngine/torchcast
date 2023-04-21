import os
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from ..data import TensorSeriesDataset
from .utils import (
    _add_missing_values,
    _download_and_extract,
    _stack_mismatched_tensors,
)

__all__ = [
    'get_neon_weather_data', 'NEONAquaticDataset', 'NEONTerrestrialDataset'
]

NEON_AQUATIC_URL = 'https://data.ecoforecast.org/neon4cast-targets/aquatics/aquatics-targets.csv.gz'  # noqa
NEON_AQUATIC_FILE_NAME = 'aquatics-targets.csv'
NEON_AQUATIC_KEYS = ['temperature', 'chla', 'oxygen']

NEON_TERRA_URL = 'https://data.ecoforecast.org/neon4cast-targets/terrestrial_daily/terrestrial_daily-targets.csv.gz'  # noqa
NEON_TERRA_FILE_NAME = 'terrestrial_daily-targets.csv'
NEON_TERRA_KEYS = ['le', 'nee']

GEFS_STAGE_3_URL = 'https://data.ecoforecast.org/neon4cast-drivers/noaa/gefs-v12/stage3/parquet/{site}/part-0.parquet'  # noqa


class NEONDataset(TensorSeriesDataset):
    '''
    This is a base class for NEON ecoforecast datasets.
    '''
    def __init__(self, path: str, url: str, file_name: str,
                 download: Union[bool, str] = False,
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

        dates, observations = [], []

        # Iterating over df.groupby returns pairs of the groupby-ed variable
        # and the dataframe restricted to that variable value. We sort it to
        # ensure consistent ordering of the site_ids.
        for site_id, _df in sorted(df.groupby('site_id')):
            del _df['site_id']
            date_min, date_max = _df['datetime'].min(), _df['datetime'].max()
            _df = _add_missing_values(
                _df,
                variable=var_ids.values(),
                datetime=np.arange(date_min, date_max + 1),
            )
            _df.sort_values(['datetime', 'variable'])
            obvs = np.array(_df['observation']).reshape(len(channel_names), -1)
            observations.append(torch.from_numpy(obvs))
            dates.append((date_min, date_max))

        # Now convert to torch tensors
        n_t = max(d_2 - d_1 for d_1, d_2 in dates)
        dates = [torch.arange(d_1, d_1 + n_t + 1) for d_1, _ in dates]
        dates = torch.stack(dates, dim=0).view(len(sites), 1, -1)
        observations = [observations[site_id] for site_id in range(len(sites))]
        observations = _stack_mismatched_tensors(observations)
        observations = observations.reshape(len(sites), len(channel_names), -1)

        super().__init__(
            dates, observations,
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


def get_neon_weather_data(path: str, site_ids: Union[str, List[str]],
                          stage: int = 3, download: Union[bool, str] = False) \
        -> torch.Tensor:
    '''
    Fetches weather forecast data for NEON and returns as a
    :class:`torch.Tensor`, arranged by (site index, date, hour, variable,
    ensemble member).

    Args:
        path (str): Path to find the dataset at.
        site_ids (str or list of str): List of four-letter site IDs to fetch.
        stage (int): The weather data is provided at three levels of
        processing: 1, 2, and 3.
        download (bool, str): Whether to download the dataset if it is not
        already available. Since the NEON datasets are updated daily, this
        can also be set to the string "force", which will redownload the
        data even if the data is already present.
    '''
    if os.path.exists(path) and not os.path.isdir(path):
        raise RuntimeError(f'{path} should be a directory')
    if download:
        os.makedirs(path, exist_ok=True)
    if stage in {1, 2}:
        raise NotImplementedError(stage)
    elif stage != 3:
        raise ValueError(stage)
    if isinstance(site_ids, str):
        site_ids = [site_ids]

    weather, dates = [], []

    for site_id in site_ids:
        site_path = os.path.join(path, f'{site_id}.parquet')
        # Make sure data is in place.
        if not os.path.exists(site_path) or (download == 'force'):
            if download:
                _download_and_extract(
                    GEFS_STAGE_3_URL.format(site=site_id),
                    site_path,
                    file_name=os.path.basename(site_path),
                )
            else:
                raise FileNotFoundError(
                    f'NEON weather dataset not found at: {path}'
                )
        df = pd.read_parquet(site_path)
        # Drop unneeded columns
        del df['longitude'], df['latitude'], df['family'], df['site_id']
        # Combine height and variable columns, then convert to index
        df['variable'] = df.pop('variable') + ' at ' + df.pop('height')
        channel_names = list(df['variable'].unique())
        var_ids = {k: i for i, k in enumerate(channel_names)}
        df['variable'] = df['variable'].replace(var_ids)
        # Drop NaNs
        df = df.dropna()
        # Convert datetime. We want both the date (to match with the NEON
        # forecast data) and the hour (since the forecasts are provided
        # every two hours).
        df['date'] = df['datetime'].astype(np.int64)
        df['date'] //= (24 * 60 * 60 * 1_000_000_000)
        df['hour'] = df.pop('datetime').astype(np.int64)
        df['hour'] %= (24 * 60 * 60 * 1_000_000_000)
        df['hour'] //= (60 * 60 * 1_000_000_000)

        # Fill in missing values
        df = _add_missing_values(
            df,
            variable=var_ids.values(),
            date=np.arange(df['date'].min(), df['date'].max() + 1),
            parameter=np.arange(1, 63),
            hour=np.arange(0, 24),
        )

        # Sort
        df = df.sort_values(['date', 'hour', 'variable', 'parameter'])
        preds = torch.from_numpy(np.array(df['prediction']))
        weather.append(preds.view(-1, 24, len(channel_names), 62))
        t = np.arange(df['date'].min(), df['date'].max() + 1)
        dates.append(torch.from_numpy(t))

    weather = torch.stack(weather, dim=0)
    dates = torch.stack(dates, dim=0)
    return weather, dates
