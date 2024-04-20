from itertools import product
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_and_extract, _split_ltsf

__all__ = ['GermanWeatherDataset']

WEATHER_URL = 'https://www.bgc-jena.mpg.de/wetter/mpi_{site}_{year}{part}.zip'
WEATHER_FILE_NAME = 'mpi_{site}_{year}{part}.csv'

WEATHER_SITES = {
    'beutenberg': 'roof',
    'saaleaue': 'saale',
    'versuchsbeete': 'Soil',
}


class GermanWeatherDataset(TensorSeriesDataset):
    '''
    This is a dataset of weather data from Germany, obtained from:

        https://www.bgc-jena.mpg.de/wetter/weather_data.html

    This dataset was first used in time series forecasting in `Zeng et al. 2022
    <https://arxiv.org/abs/2205.13504>`__, which used only the data from
    Beutenberg in 2020.
    '''
    def __init__(self, path: Optional[str] = None,
                 year: Union[int, Iterable[int]] = 2020,
                 site: Union[str, Iterable[str]] = 'beutenberg',
                 split: str = 'all', download: Union[bool, str] = True,
                 scale: bool = True, columns_as_channels: bool = True,
                 transform: Optional[Callable] = None,
                 input_margin: Optional[int] = 336,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (optional, str): Path to find the dataset at. This should be a
                directory, as the dataset consists of at least two files.
            year (int or iterable of int): The year or years of data to
                download. Choices: 2003 to present.
            site: (str or iterable of str): The site or sites of data to
                retrieve. Choices: 'beutenberg', 'saaleaue', 'versuchsbeete'.
            split (str): What split of the data to return. The splits are taken
                from Zeng et al. Choices: 'all', 'train', 'val', 'test'.
            download (bool or str): Whether to download the dataset if it is
                not already available. Choices: True, False, 'force'.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            scale (bool): Whether to normalize the data, as in the benchmark.
            columns_as_channels (bool): If true, each column is treated as a
                separate channel. If false, each column is treated as a
                separate series.
            input_margin (optional, int): The amount of margin to include on
                the left-hand side of the dataset, as it is used as an input to
                the model.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if isinstance(year, int):
            year = [year]
        if isinstance(site, str):
            site = [site]

        data = []

        for s in site:
            dfs = []
            s = WEATHER_SITES[s]

            for y, p in product(year, ['a', 'b']):
                url = WEATHER_URL.format(site=s, year=y, part=p)
                name = WEATHER_FILE_NAME.format(site=s, year=y, part=p)
                buff = _download_and_extract(
                    url, name, path, download=download, encoding='ISO-8859-1'
                )
                dfs.append(pd.read_csv(buff))

            df = pd.concat(dfs)
            dates = pd.to_datetime(
                df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S'
            )
            channel_names = list(df.columns)
            data.append(np.array(df, dtype=np.float32).T)

        dates = np.array(dates.astype(np.int64))
        dates = dates.reshape(1, 1, dates.shape[0])
        date_meta = Metadata(name='Datetime')

        if len(data) > 1:
            data = np.stack(data, axis=0)
        else:
            data = data[0].reshape(1, *data[0].shape)

        if scale:
            train_data = _split_ltsf('train', input_margin, data)
            mean, std = train_data.mean((0, 2)), train_data.std((0, 2))
            data = (data - mean.reshape(1, -1, 1)) / std.reshape(1, -1, 1)

        data_meta = Metadata(name='Data', channel_names=channel_names)

        dates, data = _split_ltsf(split, input_margin, dates, data)

        if not columns_as_channels:
            data = data.permute(1, 0, 2)

        super().__init__(
            dates, data,
            transform=transform,
            return_length=return_length,
            metadata=[date_meta, data_meta]
        )
