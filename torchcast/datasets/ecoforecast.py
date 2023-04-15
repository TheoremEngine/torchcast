from datetime import datetime
import os
from typing import Callable, List, Optional, Union

import torch

from ..data import TensorSeriesDataset
from .utils import (
    _download_and_extract,
    _load_csv_file,
    _stack_mismatched_tensors
)

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
                 channel_names: List[str], download: bool = False,
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

        # This will return a dictionary mapping keys to lists
        data = _load_csv_file(path)

        # Remove NaN values
        data = {
            k: [x for x, o in zip(vs, data['observation']) if o != 'NA']
            for k, vs in data.items()
        }

        # Convert dates to integers
        data['datetime'] = [
            (datetime.strptime(d, '%Y-%m-%d') - datetime.min).days
            for d in data['datetime']
        ]

        # Construct dictionary mapping site_ID to index, and variable to index.
        sites = sorted(set(data['site_id']))
        site_ids = {site: i for i, site in enumerate(sites)}
        var_ids = {k: i for i, k in enumerate(channel_names)}

        # Determine date ranges for each site.
        site_date_ranges = {
            site: (min(r for r, s in zip(data['datetime'], data['site_id'])
                       if s == site),
                   max(r for r, s in zip(data['datetime'], data['site_id'])
                       if s == site) + 1)
            for site in site_ids.keys()
        }

        # Build buffer
        n_t = max(v[1] - v[0] for v in site_date_ranges.values())
        buff = torch.full(
            (len(site_ids), len(channel_names), n_t), float('nan'),
            dtype=torch.float32
        )

        # Add data to buffer
        for i, site in enumerate(data['site_id']):
            idx_v = var_ids[data['variable'][i]]
            idx_t = data['datetime'][i] - site_date_ranges[site][0]
            buff[site_ids[site], idx_v, idx_t] = float(data['observation'][i])

        # Build dates
        ordered_sites = sorted(site_ids.keys(), key=lambda s: site_ids[s])
        dates = _stack_mismatched_tensors(
            [torch.arange(*site_date_ranges[site]).float().unsqueeze(0)
             for site in ordered_sites],
        )

        super().__init__(
            dates, buff,
            return_length=return_length,
            transform=transform,
            channel_names=channel_names,
            series_names=ordered_sites,
        )

        return dates, buff, ordered_sites


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
            channel_names=NEON_AQUATIC_KEYS,
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
            channel_names=NEON_TERRA_KEYS,
            return_length=return_length,
            transform=transform,
            download=download,
        )
