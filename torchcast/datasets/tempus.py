from datetime import datetime
import json
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch

from ..data import Metadata, TensorSeriesDataset
from .utils import (
    _create_time_array,
    _decode,
    _download_and_extract,
    _timestamp_to_int
)


__all__ = ['TempusDataset']


COVARIATE_TASKS = [
    'advertising_sales_covariate', 'building_manufacturing_covariate',
    'california_energy_covariate', 'california_hourly_covariate',
    'currency_dense_covariate', 'currency_monthly_covariate',
    'cybersecurity_count_covariate', 'cybersecurity_software_covariate',
    'gdp_noisy_covariate', 'gdp_years_covariate',
    'mobility_transport_covariate', 'nifty_longest_covariate',
    'nifty_minutes_covariate', 'nifty_nonstationary_covariate',
    'nyc_covid_healthcare_covariate', 'solar_100_covariate',
    'solar_500_covariate', 'solar_nature_covariate',
    'stocks_continuous_covariate', 'stocks_daily_covariate',
    'stocks_economics_covariate', 'stocks_real_covariate',
    'weather_climate_covariate', 'weather_cyclical_covariate'
]

MULTIVARIATE_TASKS = [
    'baggage_100_multivariate', 'baggage_months_multivariate',
    'baggage_sales_multivariate', 'batadal_software_multivariate',
    'gold_india_continuous_multivariate', 'gold_india_dense_multivariate',
    'gold_india_economics_multivariate', 'gold_india_real_multivariate',
    'india_gold_days_multivariate', 'lt_stock_longest_multivariate',
    'lt_stock_minutes_multivariate', 'madrid_count_multivariate',
    'madrid_cyclical_multivariate', 'madrid_hours_multivariate',
    'madrid_noisy_multivariate', 'madrid_transport_multivariate',
    'nyc_covid_healthcare_multivariate', 'soil_500_multivariate',
    'soil_nature_multivariate', 'split_smart_energy_multivariate',
    'utah_manufacturing_multivariate'
]

UNIVARIATE_TASKS = [
    'absent_binary_univariate', 'chickenpox_dense_univariate',
    'coinbase_days_univariate', 'coinbase_economics_univariate',
    'delhi_climate_univariate', 'electricity_energy_univariate',
    'employees_healthcare_univariate', 'federal_funds_weeks_univariate',
    'german_houses_sales_univariate', 'german_quarterly_univariate',
    'german_quaterly_univariate', 'inventories_manufacturing_univariate',
    'inventories_months_univariate', 'madrid_transport_univariate',
    'occupancy_count_univariate', 'patient_sparse_univariate',
    'power_consumption_years_univariate', 'retail_categorical_univariate',
    'software_nonstationary_univariate', 'soil_nature_univariate',
    'sw_job_postings_software_univariate', 'synthetic_additive2_univariate',
    'synthetic_cyclic_univariate', 'synthetic_multiplicative_univariate',
    'synthetic_nonstationary_univariate', 'web_traffic_univariate',
]

DATA_URL = 'https://github.com/Smlcrm/TempusBench/raw/refs/heads/prod/tempus_bench/tasks/{task_type}/{task}/{task}.csv'  # noqa


class TempusDataset(TensorSeriesDataset):
    '''
    This is the Tempus dataset for benchmarking time series foundation models,
    found `here<https://github.com/Smlcrm/TempusBench>`__ and  documented by
    `Goktas et al. 2026 <https://arxiv.org/abs/2604.11529>`__.
    '''
    tasks: List[str] = COVARIATE_TASKS + MULTIVARIATE_TASKS + UNIVARIATE_TASKS

    def __init__(self, task: str, path: Optional[str] = None,
                 download: bool = True, transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            task (str): Which dataset to retrieve.
            path (optional, str): Path to find the dataset at.
            download (bool or str): Whether to download the dataset if it is
                not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if task in COVARIATE_TASKS:
            task_type = 'covariate'
        elif task in MULTIVARIATE_TASKS:
            task_type = 'multivariate'
        elif task in UNIVARIATE_TASKS:
            task_type = 'univariate'
        else:
            raise ValueError(f'Did not recognize {task}')

        buff = _download_and_extract(
            DATA_URL.format(task_type=task_type, task=task),
            f'{task}.csv',
            path,
            download=download,
        )
        buff = _decode(buff)
        df = pd.read_csv(buff)

        # Tempus has two different formats for the csv files, and we handle
        # them separately.
        if 'variable_type' in df:
            covars, targets = {}, {}
            for _, row in df.iterrows():
                values = _parse_array(row['values'])
                if row['variable_type'] == 'covariate':
                    covars[row['variable_name']] = values
                else:
                    targets[row['variable_name']] = values

            t = pd.to_datetime(pd.Series(json.loads(row['timestamps'])))
            t = torch.from_numpy(_timestamp_to_int(t)).view(1, 1, -1)

            target_channels = list(targets.keys())
            metadata = [
                Metadata(name='Datetime'),
                Metadata(name='Target', channel_names=target_channels)
            ]
            targets = np.stack([targets[k] for k in target_channels], axis=0)
            targets = torch.from_numpy(targets).float().unsqueeze(0)

            if covars:
                cov_channels = list(covars.keys())
                metadata += [
                    Metadata(name='Covariates', channel_names=cov_channels)
                ]
                covars = np.stack([covars[k] for k in cov_channels], axis=0)
                covars = torch.from_numpy(covars).float().unsqueeze(0)
                tensors = (t, targets, covars)
            else:
                tensors = (t, targets)

        else:
            targets = _parse_array(df['target'].item())
            targets = torch.from_numpy(targets).float()
            if targets.ndim == 1:
                targets = targets.reshape(1, 1, -1)
            else:
                targets = targets.reshape(1, *targets.shape)

            freq = df['freq'].item()
            if freq != 'unknown':
                start_t = df['start'].item()
                if 'T' in start_t:
                    fmt = '%Y-%m-%dT%H:%M'
                elif freq.endswith('T'):
                    fmt = '%Y-%m-%d %H:%M'
                else:
                    fmt = '%Y-%m-%d'
                start_t = datetime.strptime(start_t, fmt)

                t = _create_time_array(start_t, freq, targets.shape[-1])
                t = torch.from_numpy(t).reshape(1, 1, -1)
                tensors = (t, targets)
                metadata = [Metadata(name='Datetime'), Metadata(name='Target')]
            else:
                tensors = (targets,)
                metadata = [Metadata(name='Target')]

        super().__init__(
            *tensors,
            return_length=return_length,
            transform=transform,
            metadata=metadata,
        )


def _parse_array(x: pd.Series) -> np.ndarray:
    x = x.replace('null', 'NaN').replace('None', 'NaN')
    return np.array(json.loads(x))
