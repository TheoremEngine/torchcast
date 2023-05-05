from itertools import product
import os
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_and_extract

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

    This is provided because it was used in the paper:

        https://arxiv.org/abs/2205.13504

    Which used only the data from Beutenberg in 2020.
    '''
    def __init__(self, path: str, year: Union[int, Iterable[int]] = 2020,
                 site: Union[str, Iterable[str]] = 'beutenberg',
                 download: Union[bool, str] = False,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            path (str): Path to find the dataset at. This should be a
            directory, as the dataset consists of at least two files.
            year (int or iterable of int): The year or years of data to
            download. Choices: 2003 to present.
            site: (str or iterable of str): The site or sites of data to
            download. Choices: 'beutenberg', 'saaleaue', 'versuchsbeete'.
            download (bool or str): Whether to download the dataset if it is
            not already available. Choices: True, False, 'force'.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
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
                file_path = os.path.join(path, name)

                if (
                    (download == 'force') or
                    (download and (not os.path.exists(file_path)))
                ):
                    _download_and_extract(url, file_path, file_name=name)

                if os.path.exists(file_path):
                    dfs.append(pd.read_csv(file_path, encoding='ISO-8859-1'))
                else:
                    dfs.append(pd.read_csv(url, encoding='ISO-8859-1'))

            df = pd.concat(dfs)
            dates = pd.to_datetime(
                df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S'
            )
            channel_names = list(df.columns)
            data.append(np.array(df, dtype=np.float32).T)

        # This converts time to nanoseconds, and we want seconds.
        dates = np.array(dates.astype(np.int64)) // 1_000_000_000
        dates = dates.reshape(1, 1, dates.shape[0])
        if len(data) > 1:
            data = np.stack(data, axis=0)
        else:
            data = data[0].reshape(1, *data[0].shape)
        meta = Metadata(channel_names=channel_names)

        super().__init__(
            dates, data,
            transform=transform,
            return_length=return_length,
            metadata=[None, meta]
        )
