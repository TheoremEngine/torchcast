from datetime import datetime
from functools import lru_cache
import gzip
from io import BytesIO, StringIO
import lzma
import os
import re
import tarfile
from typing import Optional, Union
import zipfile

import numpy as np
import pandas as pd
import requests
import torch

from ..data import Metadata
from ._file_readers import load_ts_file, load_tsf_file

try:
    import pyarrow
except ImportError:
    has_pyarrow = False
else:
    has_pyarrow = True

__all__ = ['load_ts_file', 'load_tsf_file']


_GOOGLE_URL = 'https://drive.google.com/uc?export=download'


def _add_missing_values(df: pd.DataFrame, **variables) -> pd.DataFrame:
    '''
    This creates rows in a :class:`pandas.DataFrame` to ensure that all values
    for a set of columns have rows.
    '''
    # Based on the second answer at:
    # https://stackoverflow.com/questions/31786881/adding-values-for-missing-data-combinations-in-pandas  # noqa
    var_names, var_values = zip(*variables.items())
    mind = pd.MultiIndex.from_product(var_values, names=var_names)
    # The zip returns var_names as a tuple. set_index interprets a tuple as a
    # single column name, so we need to cast to a list so it will be
    # interpreted as a collection of column names.
    df = df.set_index(list(var_names))
    df = df.reindex(mind, fill_value=float('nan'))
    return df.reset_index()


MINUTE_US = 60 * 1_000_000
HOUR_US = 60 * MINUTE_US
DAY_US = 24 * HOUR_US


def _create_time_array(start: datetime, frequency: str, n: int) \
        -> np.ndarray:
    # Yearly, quarterly, and monthly need to be parsed as lists initially
    # because their duration in nanoseconds can vary.
    if frequency in {'yearly', '1Y'}:
        out = [
            datetime(year=(start.year + t), month=start.month, day=start.day,
                     hour=start.hour, minute=start.minute, second=start.second)
            for t in range(n)
        ]
        return _timestamp_to_int(pd.Series(out))
    elif frequency in {'quarterly', '1Q', 'Q'}:
        out = [
            datetime(year=(start.year + (start.month + t) // 12),
                     month=(((start.month + t - 1) % 12) + 1), day=start.day,
                     hour=start.hour, minute=start.minute, second=start.second)
            for t in range(0, 3 * n, 3)
        ]
        return _timestamp_to_int(pd.Series(out))
    elif frequency in {'monthly', '1M', 'M'}:
        out = [
            datetime(year=(start.year + (start.month + t) // 12),
                     month=(((start.month + t - 1) % 12) + 1), day=start.day,
                     hour=start.hour, minute=start.minute, second=start.second)
            for t in range(n)
        ]
        return _timestamp_to_int(pd.Series(out))
    elif frequency in {'weekly', '1W', 'W'}:
        start = _timestamp_to_int(pd.Timestamp(start))
        return np.arange(
            start, start + n * (7 * DAY_US), 7 * DAY_US, dtype=np.int64,
        )
    elif frequency in {'1B', 'B'}:
        out = pd.bdate_range(start, periods=n)
        return _timestamp_to_int(out)
    elif frequency in {'daily', '1D', 'D'}:
        start = _timestamp_to_int(pd.Timestamp(start))
        return np.arange(
            start, start + n * DAY_US, DAY_US, dtype=np.int64,
        )
    elif frequency in {'hourly', '1H', 'H'}:
        start = _timestamp_to_int(pd.Timestamp(start))
        return np.arange(
            start, start + n * HOUR_US, HOUR_US, dtype=np.int64,
        )
    elif frequency in {'half_hourly', '30T'}:
        start = _timestamp_to_int(pd.Timestamp(start))
        return np.arange(
            start, start + n * 30 * MINUTE_US, 30 * MINUTE_US, dtype=np.int64,
        )
    elif frequency.endswith('_minutes'):
        start = _timestamp_to_int(pd.Timestamp(start))
        frequency = int(frequency.removesuffix('_minutes'))
        return np.arange(
            start, start + n * frequency * MINUTE_US, frequency * MINUTE_US,
            dtype=np.int64,
        )
    elif frequency.endswith('T'):
        if frequency == 'T':
            frequency = '1T'
        start = _timestamp_to_int(pd.Timestamp(start))
        frequency = int(frequency.removesuffix('T'))
        return np.arange(
            start, start + n * frequency * MINUTE_US, frequency * MINUTE_US,
            dtype=np.int64,
        )
    elif frequency == '4_seconds':
        start = _timestamp_to_int(pd.Timestamp(start))
        return np.arange(
            start, start + n * 4_000_000, 4_000_000, dtype=np.int64,
        )
    elif frequency.endswith('S'):
        if frequency == 'S':
            frequency = '1S'
        start = _timestamp_to_int(pd.Timestamp(start))
        frequency = int(frequency.removesuffix('S'))
        return np.arange(
            start, start + n * 1_000_000, 1_000_000, dtype=np.int64,
        )
    else:
        raise ValueError(f'Did not recognize frequency {frequency}')


def _decode(buff: BytesIO, encoding: str = 'utf-8') -> StringIO:
    return StringIO(buff.read().decode(encoding))


def _download_and_extract(url: str, file_name: str, local_path: Optional[str],
                          download: Union[bool, str] = True) -> BytesIO:
    '''
    Convenience function to wrangle fetching a remote file. This will return
    the file as a :class:`io.BytesIO` object, fetching it if necessary. This
    function is designed to work with files small enough to be held in memory.

    Args:
        url (str): The URL to download the data from.
        file_name (str): Name of the file. This should be the desired file, not
            the name of the zipped file, if the file is compressed. If the file
            has a relative path inside an archive file, this name should
            include that path.
        local_path (optional, str): The local path to find the file, and put
            the extracted data in if it needs to be fetched.
        download (bool or str): Whether to download the file if it is not
            present. Can be true, false, or 'force', in which case the file
            will be redownloaded even if it is present locally.
    '''
    # The name of the file we're fetching may be different from the name of the
    # file we want, e.g. if it's zipped.
    fetched_name = os.path.basename(url)

    if local_path is not None:
        # First, infer whether local_path is a directory or the path of the
        # file. Make sure target location exists.
        if os.path.basename(local_path) in {file_name, fetched_name}:
            local_path = os.path.dirname(local_path)
        os.makedirs(local_path, exist_ok=True)
        fetched_path = os.path.join(local_path, fetched_name)
        file_path = os.path.join(local_path, os.path.basename(file_name))

        # If file is already present, open it and return.
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return BytesIO(f.read())
        elif os.path.exists(fetched_path):
            with open(fetched_path, 'rb') as f:
                buff = BytesIO(f.read())
        elif not download:
            raise FileNotFoundError(local_path)

    elif not download:
        # If download == False, but no local_path is provided, what are we to
        # do?
        raise ValueError('If download is false, local_path must be provided')

    # We break this out as a separate function so we can wrap it in an
    # lru_cache.
    buff = _fetch_from_remote(url)
    # Need to include buff.seek, because _fetch_from_remote is lru_cached, and
    # it returns a BytesIO. When first returned, it will be at index 0, but
    # when it's processed below, it won't be rewound. Then, if it's fetched
    # again, it will error out because the index is at the end and it looks
    # empty.
    buff.seek(0)

    # If local_path is provided, write to disk
    if (local_path is not None) and (not os.path.exists(fetched_path)):
        with open(fetched_path, 'wb') as out_file:
            out_file.write(buff.read())
        buff.seek(0)

    return _extract_file_from_buffer(buff, fetched_name, file_name)


def _download_from_google_drive_and_extract(doc_id: str, file_name: str,
                                            remote_name: str,
                                            local_path: Optional[str],
                                            download: Union[bool, str] = True)\
        -> BytesIO:
    '''
    Convenience function to wrangle fetching a remote file from Google drive.
    This will return the file as a :class:`io.StringIO` object, fetching it if
    necessary. This function is designed to work with files small enough to be
    held in memory.

    Args:
        doc_id (str): The document ID on Google drive.
        file_name (str): Name of the file. This should be the desired file, not
            the name of the zipped file, if the file is compressed. If the file
            has a relative path inside an archive file, this name should
            include that path.
        remote_name (str): Name of the file we're fetching. This may not be
            the same as the file_name, e.g. if we're fetching a zip archive.
        local_path (optional, str): The local path to find the file, and put
            the extracted data in if it needs to be fetched.
        download (bool or str): Whether to download the file if it is not
            present. Can be true, false, or 'force', in which case the file
            will be redownloaded even if it is present locally.
    '''
    if local_path is not None:
        # First, infer whether local_path is a directory or the path of the
        # file. Make sure target location exists.
        if os.path.basename(local_path) in {file_name, remote_name}:
            local_path = os.path.dirname(local_path)
        os.makedirs(local_path, exist_ok=True)
        fetched_path = os.path.join(local_path, remote_name)
        file_path = os.path.join(local_path, os.path.basename(file_name))

        # If file is already present, open it and return.
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return BytesIO(f.read())
        elif os.path.exists(fetched_path):
            with open(fetched_path, 'rb') as f:
                buff = BytesIO(f.read())
        elif not download:
            raise FileNotFoundError(local_path)

    elif not download:
        # If download == False, but no local_path is provided, what are we to
        # do?
        raise ValueError('If download is false, local_path must be provided')

    # We break this out as a separate function so we can wrap it in an
    # lru_cache.
    buff = _fetch_from_google(doc_id)
    # Need to include buff.seek, because _fetch_from_google is lru_cached, and
    # it returns a BytesIO. When first returned, it will be at index 0, but
    # when it's processed below, it won't be rewound. Then, if it's fetched
    # again, it will error out because the index is at the end and it looks
    # empty.
    buff.seek(0)

    # If local_path is provided, write to disk
    if (local_path is not None) and (not os.path.exists(fetched_path)):
        with open(fetched_path, 'wb') as out_file:
            out_file.write(buff.read())
        buff.seek(0)

    return _extract_file_from_buffer(buff, remote_name, file_name)


def _extract_file_from_buffer(buff: BytesIO, buff_name: str,
                              file_name: str) -> BytesIO:
    '''
    Extracts a desired file from an arbitrary buffer. The buffer might be the
    file we want, or it might be some archive format.

    Args:
        buff (:class:`BytesIO`): The buffer to extract the file from.
        buff_name (str): Name of the buffer. We use this to infer whether it's
            an archive and how to open it up.
        file_name (str): Name of the file we want.
    '''
    # Extract, if needed.
    while True:
        buff_name, ext = os.path.splitext(buff_name.lower())

        if ext == '.gz':
            buff = BytesIO(gzip.decompress(buff.read()))
            continue
        elif ext == '.xz':
            buff = BytesIO(lzma.decompress(buff.read()))
            continue
        elif ext == '.tar':
            with tarfile.TarFile(fileobj=buff, mode='r') as tar_file:
                buff = BytesIO(tar_file.extractfile(file_name).read())
        elif ext == '.zip':
            with zipfile.ZipFile(buff) as zip_file:
                buff = BytesIO(zip_file.read(file_name))
        else:
            break

    return buff


@lru_cache
def _fetch_from_google(doc_id: str) -> BytesIO:
    '''
    Retrieves a file from a URL and downloads it, returning it as an in-memory
    buffer. We break this out as a separate function so we can wrap it in
    :func:`lru_cache`.

    Args:
        doc_id (str): Document ID of the file to retrieve.
    '''
    url, params = _GOOGLE_URL, {'id': doc_id}
    session = requests.session()

    # Based on gdown package implementation
    while True:
        r = session.get(url, params=params, stream=True, verify=True)
        r.raise_for_status()

        if 'Content-Disposition' in r.headers:
            break

        search = re.search('id="download-form" action="(.+?)"', r.text)
        url = search.groups()[0].replace('&amp;', '&')
        params = dict(re.findall(
            '<input type="hidden" name="(.+?)" value="(.+?)">', r.text
        ))
        r.close()

    buff = BytesIO()

    for chunk in r.iter_content(chunk_size=8192):
        buff.write(chunk)

    r.close()
    session.close()
    buff.seek(0)

    return buff


@lru_cache
def _fetch_from_remote(url: str) -> BytesIO:
    '''
    Retrieves a file from a URL and downloads it, returning it as an in-memory
    buffer. We break this out as a separate function so we can wrap it in
    :func:`lru_cache`.

    Args:
        url (str): URL to download.
    '''
    buff = BytesIO()

    # Syntax is taken from:
    # https://stackoverflow.com/questions/16694907/download-large-file-in- \
    #     python-with-requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            buff.write(chunk)

    buff.seek(0)

    return buff


def _read_hf_arrow_buffer(buff, strptime_format: Optional[str] = None):
    '''
    Reads the Arrow file used by HuggingFace for storing time series, returning
    both the tensors and metadata.
    '''
    if not has_pyarrow:
        raise ImportError('pyarrow not installed')

    in_memory_stream = pyarrow.input_stream(buff)
    opened_stream = pyarrow.ipc.open_stream(in_memory_stream)
    df = opened_stream.read_all().to_pandas()

    data, ts, series_names = [], [], []

    for _, row in df.iterrows():
        x = row['target']
        if x.dtype is np.dtype('O'):
            x = np.stack([_x for _x in x], axis=0)
        elif x.ndim == 1:
            # Need to clone this here because the numpy array has a read-
            # only flag, which torch does not support.
            x = x.reshape(1, -1).copy()
        data.append(torch.from_numpy(x).float())
        if strptime_format is None:
            start_date = row['start'].to_pydatetime()
        else:
            start_date = datetime.strptime(row['start'], strptime_format)
        t = _create_time_array(start_date, row['freq'], data[-1].shape[-1])
        ts.append(torch.from_numpy(t.reshape(1, -1)))
        series_names.append(row['item_id'])

    if 'variate_names' in df.columns:
        channel_names = row['variate_names'].tolist()
    else:
        channel_names = None

    metadata = [
        Metadata(name='Datetime', series_names=series_names),
        Metadata(name='Target', channel_names=channel_names,
                 series_names=series_names),
    ]

    return (ts, data), metadata


def _split_ltsf(split: str, input_length: Optional[int],
                *tensors: torch.Tensor):
    '''
    Replicates the train-val-test split used in Zeng et al. in most of their
    datasets. The first 70% will be allocated to the training set, the last 20%
    will be the test set, and the remainder will be the validation set.

    Args:
        split (str): Split to return. Choices: 'all', 'train', 'val', 'test'.
        input_length (optional, int): The length of sequences used in
        prediction. If provided, then the val and test sets will include this
        much margin on the left-hand side.
    '''
    ts = {x.shape[2] for x in tensors if x.shape[2] > 1}
    if len(ts) > 1:
        raise ValueError(f'Received conflicting time lengths: {ts}')

    if split in {'train', 'val', 'test'}:
        input_length = input_length or 0
        num_all = max(x.shape[2] for x in tensors)
        num_train, num_test = int(0.7 * num_all), int(0.2 * num_all)
        num_val = num_all - (num_train + num_test)
        if split == 'train':
            t_0, t_1 = 0, num_train
        elif split == 'val':
            t_0, t_1 = num_train - input_length, num_train + num_val
        else:
            t_0, t_1 = num_train + num_val - input_length, num_all
        tensors = tuple(x[:, :, t_0:t_1] for x in tensors)
        return tensors if (len(tensors) > 1) else tensors[0]

    elif split == 'all':
        return tensors if (len(tensors) > 1) else tensors[0]

    else:
        raise ValueError(split)


def _timestamp_to_int(x: Union[pd.DatetimeIndex, pd.Series, pd.Timestamp]) \
        -> Union[np.array, int]:
    '''
    This utility function is necessary because pandas changed its internal
    representation of Timestamps between version 2 and 3. Previously,
    pd.Timestamp.value returned the number of microseconds since the epoch
    start, now it returns the number of nanoseconds. Since we want to support
    both old and new pandas, we need to handle this more carefully than
    before.
    '''
    if isinstance(x, pd.DatetimeIndex):
        x = x.to_series()

    example = x.iloc[0] if isinstance(x, pd.Series) else x
    if example.tz is None:
        epoch_start = pd.Timestamp(1970, 1, 1)
    else:
        epoch_start = pd.Timestamp(1970, 1, 1, tz='UTC')

    out = (x - epoch_start) // pd.Timedelta(1, unit='us')
    if isinstance(out, pd.Series):
        # This copy is needed because a pd.Series is write-protected, and that
        # will be inherited by out.values, but a torch.Tensor cannot be write-
        # protected.
        out = out.values.copy()
    return out
