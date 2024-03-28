from datetime import datetime
import os
from typing import Callable, List, Optional, Union
import warnings

import numpy as np
import pandas as pd
import torch

from ..data import Metadata, TensorSeriesDataset
from ._file_readers import parse_tsf
from .utils import _download_and_extract

__all__ = ['MonashArchiveDataset']


# These should all be prepended by 'https://zenodo.org/records/'.
MONASH_ARCHIVE_URLS = {
    'AustralianElectricityDemand': '4659727/files/australian_electricity_demand_dataset.zip',  # noqa
    'Bitcoin': '5121965/files/bitcoin_dataset_with_missing_values.zip',
    'CarParts': '4656022/files/car_parts_dataset_with_missing_values.zip',
    'CIF2016': '4656042/files/cif_2016_dataset.zip',
    'COVID19Deaths': '4656009/files/covid_deaths_dataset.zip',
    'Electricity-Hourly': '4656140/files/electricity_hourly_dataset.zip',
    'Electricity-Weekly': '4656141/files/electricity_weekly_dataset.zip',
    'KaggleWikipediaWebTraffic': '4656080/files/kaggle_web_traffic_dataset_with_missing_values.zip',  # noqa
    'Dominick': '4654802/files/dominick_dataset.zip',
    'FREDMD': '4654833/files/fred_md_dataset.zip',
    'Hospital': '4656014/files/hospital_dataset.zip',
    'KDDCup2018': '4656719/files/kdd_cup_2018_dataset_with_missing_values.zip',
    'LondonSmartMeters': '4656072/files/london_smart_meters_dataset_with_missing_values.zip',  # noqa
    'M1-Monthly': '4656159/files/m1_monthly_dataset.zip',
    'M1-Quarterly': '4656154/files/m1_quarterly_dataset.zip',
    'M1-Yearly': '4656193/files/m1_yearly_dataset.zip',
    'M3-Monthly': '4656298/files/m3_monthly_dataset.zip',
    'M3-Other': '4656335/files/m3_other_dataset.zip',
    'M3-Quarterly': '4656262/files/m3_quarterly_dataset.zip',
    'M3-Yearly': '4656222/files/m3_yearly_dataset.zip',
    'M4-Daily': '4656548/files/m4_daily_dataset.zip',
    'M4-Hourly': '4656589/files/m4_hourly_dataset.zip',
    'M4-Monthly': '4656480/files/m4_monthly_dataset.zip',
    'M4-Quarterly': '4656410/files/m4_quarterly_dataset.zip',
    'M4-Weekly': '4656522/files/m4_weekly_dataset.zip',
    'M4-Yearly': '4656379/files/m4_yearly_dataset.zip',
    'MelbournePedestrianCounts': '4656626/files/pedestrian_counts_dataset.zip',
    'NN5-Daily': '4656110/files/nn5_daily_dataset_with_missing_values.zip',
    'NN5-Weekly': '4656125/files/nn5_weekly_dataset.zip',
    'Rideshare': '5122114/files/rideshare_dataset_with_missing_values.zip',
    'SaugeenRiverFlow': '4656058/files/saugeenday_dataset.zip',
    'Solar-10Minutes': '4656144/files/solar_10_minutes_dataset.zip',
    'Solar-Weekly': '4656151/files/solar_weekly_dataset.zip',
    'SolarPower': '4656027/files/solar_4_seconds_dataset.zip',
    'Sunspot': '4654773/files/sunspot_dataset_with_missing_values.zip',
    'TemperatureRain': '5129073/files/temperature_rain_dataset_with_missing_values.zip',  # noqa
    'Tourism-Monthly': '4656096/files/tourism_monthly_dataset.zip',
    'Tourism-Quarterly': '4656093/files/tourism_quarterly_dataset.zip',
    'Tourism-Yearly': '4656103/files/tourism_yearly_dataset.zip',
    'Traffic-Hourly': '4656132/files/traffic_hourly_dataset.zip',
    'Traffic-Weekly': '4656135/files/traffic_weekly_dataset.zip',
    'USBirths': '4656049/files/us_births_dataset.zip',
    'VehicleTrips': '5122535/files/vehicle_trips_dataset_with_missing_values.zip',  # noqa
    'Weather': '4654822/files/weather_dataset.zip',
    'WindFarms': '4654909/files/wind_farms_minutely_dataset_with_missing_values.zip',  # noqa
    'WindPower': '4656032/files/wind_4_seconds_dataset.zip',
}

# Most Monash datasets include the forecasting horizon as part of the tsf
# header, but some do not. The following forecast horizons are taken from the
# paper, page 7. Note that Solar-Weekly is an exception handled in the __init__
# method.
HORIZON = {
    'monthly': 12,
    'weekly': 8,
    'daily': 30,
    'hourly': 7 * 24,
    'half_hourly': 7 * 24 * 2,
    '10_minutes': 7 * 24 * 6,
    '4_seconds': 7 * 24 * 60 * 15,
}

# This keeps track of which attribute specifies the variable in a multivariate
# series. If not listed here, the series is univariate.
MULTIVARIATE_VARIABLE_NAMES = {
    'Bitcoin': 'series_name',
    'KDDCup2018': 'air_quality_measurement',
    'Rideshare': 'type',
    'TemperatureRain': 'obs_or_fcst',
    'VehicleTrips': 'type',
    'Weather': 'series_type',
}
# This keeps track of which attribute(s) specify the series in a multivariate
# series.
MULTIVARIATE_SERIES_NAMES = {
    'KDDCup2018': ('city', 'station'),
    'Rideshare': ('source_location', 'provider_name', 'provider_service'),
    'TemperatureRain': ('station_id',),
    'VehicleTrips': ('base_number', 'base_name'),
}


class MonashArchiveDataset(TensorSeriesDataset):
    '''
    This provides access to all `Monash forecasting archive
    <https://forecastingdata.org>`__ datasets, as discussed in `Godahewa et al.
    2021 <https://arxiv.org/abs/2105.06643>`__.
    '''
    def __init__(self, task: str, split: str = 'train',
                 path: Optional[str] = None,
                 download: Union[str, bool] = True,
                 transform: Optional[Callable] = None,
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
        if split not in {'all', 'train', 'test'}:
            raise ValueError(f'Did not recognize split {split}')
        if task not in MONASH_ARCHIVE_URLS:
            raise ValueError(f'Did not recognize task {task}')

        url = f'https://zenodo.org/records/{MONASH_ARCHIVE_URLS[task]}'
        buff = _download_and_extract(
            url,
            os.path.basename(url).replace('zip', 'tsf'),
            path,
            download=download,
            # For the encoding, see the convert_tsf_to_dataframe function in
            # the utils/data_loader.py file in the repo:
            # https://github.com/rakshitha123/TSForecasting
            encoding='cp1252',
        )

        # data may be a np.ndarray OR a List[np.ndarray] at this stage,
        # depending on the header options in the tsf file.
        data, attrs = parse_tsf(buff.read())
        t = None

        # Parse data. After this if-elif-else block, data should be a
        # List[np.ndarray], and t should be either None or a List[np.ndarray].
        # The entries in data should all be in CT arrangement, although T may
        # vary. The entries in t should all be in T arrangement, and should
        # match the corresponding entries in data.
        if task == 'Bitcoin':
            channel_names = attrs[MULTIVARIATE_VARIABLE_NAMES[task]]
            series_names = None
            data = [np.concatenate(data, axis=0)]
            t = [
                _create_time_array(attrs['start_timestamp'][0],
                                   attrs['frequency'], data[0].shape[-1])
            ]

        elif task in MULTIVARIATE_SERIES_NAMES:
            series_columns = MULTIVARIATE_SERIES_NAMES[task]
            if len(series_columns) == 1:
                series_idx = attrs[series_columns[0]]
            else:
                series_idx = [
                    tuple(row)
                    for row in zip(*(attrs[col] for col in series_columns))
                ]
            series_names = sorted(set(series_idx))
            series_idx = [series_names.index(idx) for idx in series_idx]

            var_idx = attrs[MULTIVARIATE_VARIABLE_NAMES[task]]
            channel_names = sorted(set(var_idx))
            var_idx = [channel_names.index(idx) for idx in var_idx]

            data = _reindex_data(data, series_idx, var_idx)

            if 'start_timestamp' in attrs:
                t = [
                    _create_time_array(
                        attrs['start_timestamp'][series_idx.index(i_ser)],
                        attrs['frequency'], data[i_ser].shape[-1]
                    ) for i_ser in range(len(series_names))
                ]

        else:
            if task == 'Weather':
                warnings.warn(
                    'weather dataset is multivariate, but does not include a '
                    'column specifying which measurements correspond to the '
                    'same series, and there are different numbers of series '
                    'for each variable type. We therefore treat it as a '
                    'collection of univariate series.'
                )

            if 'start_timestamp' in attrs:
                t = [
                    _create_time_array(
                        start, attrs['frequency'], data[i].shape[-1]
                    ) for i, start in enumerate(attrs['start_timestamp'])
                ]

            channel_names = series_names = None

        # As discussed in the Monash Archive paper, Solar-Weekly is handled
        # differently and has a unique horizon. For the other datasets, the
        # frequency is either part of the tsf header, or a default value based
        # on the frequency is used.
        if task == 'Solar-Weekly':
            horizon = 5
        else:
            horizon = attrs.get('horizon', HORIZON[attrs['frequency']])

        data = _split(data, split=split, horizon=horizon)

        # Construct metadata
        metadata = [
            Metadata(name='Data', channel_names=channel_names,
                     series_names=series_names)
        ]
        if t is not None:
            metadata = [Metadata(name='Datetime')] + metadata
            t = [_t.reshape(1, -1) for _t in t]
            t = _split(t, split=split, horizon=horizon)

        super().__init__(
            *((data,) if (t is None) else (t, data)),
            return_length=return_length,
            transform=transform,
            metadata=metadata,
        )


MINUTE_NS = 60 * 1_000_000_000
HOUR_NS = 60 * MINUTE_NS
DAY_NS = 24 * HOUR_NS


def _create_time_array(start: datetime, frequency: str, n: int) \
        -> np.ndarray:
    # Yearly, quarterly, and monthly need to be parsed as lists initially
    # because their duration in nanoseconds can vary.
    if frequency == 'yearly':
        out = [
            datetime(year=(start.year + t), month=start.month, day=start.day,
                     hour=start.hour, minute=start.minute, second=start.second)
            for t in range(n)
        ]
        return np.array(pd.Series(out).astype(np.int64))
    elif frequency == 'quarterly':
        out = [
            datetime(year=(start.year + (start.month + t) // 12),
                     month=(((start.month + t - 1) % 12) + 1), day=start.day,
                     hour=start.hour, minute=start.minute, second=start.second)
            for t in range(0, 3 * n, 3)
        ]
        return np.array(pd.Series(out).astype(np.int64))
    elif frequency == 'monthly':
        out = [
            datetime(year=(start.year + (start.month + t) // 12),
                     month=(((start.month + t - 1) % 12) + 1), day=start.day,
                     hour=start.hour, minute=start.minute, second=start.second)
            for t in range(n)
        ]
        return np.array(pd.Series(out).astype(np.int64))
    elif frequency == 'weekly':
        start = pd.Timestamp(start).value
        return np.arange(
            start, start + n * (7 * DAY_NS), 7 * DAY_NS, dtype=np.int64,
        )
    elif frequency == 'daily':
        start = pd.Timestamp(start).value
        return np.arange(
            start, start + n * DAY_NS, DAY_NS, dtype=np.int64,
        )
    elif frequency == 'hourly':
        start = pd.Timestamp(start).value
        return np.arange(
            start, start + n * HOUR_NS, HOUR_NS, dtype=np.int64,
        )
    elif frequency == 'half_hourly':
        start = pd.Timestamp(start).value
        return np.arange(
            start, start + n * 30 * MINUTE_NS, 30 * MINUTE_NS, dtype=np.int64,
        )
    elif frequency == '10_minutes':
        start = pd.Timestamp(start).value
        return np.arange(
            start, start + n * 10 * MINUTE_NS, 10 * MINUTE_NS, dtype=np.int64,
        )
    elif frequency == '4_seconds':
        start = pd.Timestamp(start).value
        return np.arange(
            start, start + n * 4_000_000_000, 4_000_000_000, dtype=np.int64,
        )
    else:
        raise ValueError(f'Did not recognize frequency {frequency}')


def _split(tensors: List[torch.Tensor], split: str,
           horizon: Union[int, np.ndarray]) -> List[torch.Tensor]:
    '''
    Given one or more tensors, a choice of train/test split, and a horizon,
    applies the split to the tensors.
    '''
    if split == 'all':
        return tensors
    elif split == 'train':
        # CIF2016 has a different horizon for each series.
        if isinstance(horizon, int):
            return [x[:, :-horizon] for x in tensors]
        else:
            return [x[:, :-h] for x, h in zip(tensors, horizon)]
    elif split == 'test':
        # CIF2016 has a different horizon for each series.
        if isinstance(horizon, int):
            return [x[:, -horizon:] for x in tensors]
        else:
            return [x[:, -h:] for x, h in zip(tensors, horizon)]
    else:
        raise ValueError(split)


def _reindex_data(data: Union[np.ndarray, List[np.ndarray]],
                  series_idx: List[int], var_idx: List[int]) \
        -> List[np.ndarray]:
    '''
    Given a dataset in arrangement (NC)1T, and a list of integers indexing the
    channels and series for each row in the dataset, reindexs it to be in
    arrangement NCT.
    '''
    all_idxs = list(zip(var_idx, series_idx))
    max_var_idx, max_series_idx = max(var_idx), max(series_idx)
    rtn = []

    for i_ser in range(max_series_idx + 1):
        rtn.append([])
        for i_var in range(max_var_idx + 1):
            if (i_var, i_ser) in all_idxs:
                rtn[-1].append(data[all_idxs.index((i_var, i_ser))])
            else:
                rtn[-1].append(None)
        if all(x is None for x in rtn):
            rtn[-1] = np.full((1, 1), float('nan'), dtype=np.float32)
        else:
            n_t = next(x.shape[-1] for x in rtn[-1] if x is not None)
            rtn[-1] = [
                np.full((1, n_t), float('nan'), dtype=np.float32)
                if (x is None) else x for x in rtn[-1]
            ]
            rtn[-1] = np.concatenate(rtn[-1], axis=0)

    return rtn
