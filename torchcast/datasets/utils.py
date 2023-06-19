import gzip
from io import BytesIO, StringIO
import lzma
import os
import re
import tarfile
from typing import Callable, List, Optional, Union
import zipfile

import pandas as pd
import requests
import torch

from ._file_readers import load_ts_file, load_tsf_file

__all__ = ['load_ts_file', 'load_tsf_file']


_GOOGLE_URL = 'https://drive.google.com/uc?id={doc_id}'


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


def _download_and_extract(url: str, file_name: str, local_path: Optional[str],
                          download: Union[bool, str] = True,
                          fetcher: Optional[Callable] = None,
                          encoding: str = 'utf-8') -> StringIO:
    '''
    Convenience function to wrangle fetching a remote file. This will return
    the file as a :class:`io.StringIO` object, fetching it if necessary. This
    function is designed to work with files small enough to be held in memory.

    Args:
        url (str): The URL to download the data from.
        file_name (str): Name of the file. This should be the desired file, not
        the name of the zipped file, if the file is compressed.
        local_path (optional, str): The local path to find the file, and put
        the extracted data in if it needs to be fetched.
        download (bool or str): Whether to download the file if it is not
        present. Can be true, false, or 'force', in which case the file will be
        redownloaded even if it is present locally.
        encoding (str): Encoding of bytes object.
    '''
    if local_path is not None:
        # First, infer whether local_path is a directory or the path of the
        # file. Make sure target location exists.
        if os.path.basename(local_path) == file_name:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
        elif os.path.isdir(local_path):
            local_path = os.path.join(local_path, file_name)
        elif not os.path.exists(local_path):
            os.makedirs(local_path, exist_ok=True)
            local_path = os.path.join(local_path, file_name)

        # If file is already present, open it and return.
        if os.path.exists(local_path):
            with open(local_path, 'r') as f:
                return StringIO(f.read())
        elif not download:
            raise FileNotFoundError(local_path)

    elif not download:
        # If download == False, but no local_path is provided, what are we to
        # do?
        raise ValueError('If download is false, local_path must be provided')

    # If we reach here, the file is not available locally. So let's get that
    # file!
    if fetcher is None:
        fetcher = _fetch_from_remote
    buff = fetcher(url)

    # Extract, if needed.
    while True:
        url, ext = os.path.splitext(url.lower())

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

        break

    # Convert from bytes to string
    buff.seek(0)
    buff = StringIO(buff.read().decode(encoding))

    # If local_path is provided, write to disk
    if local_path is not None:
        with open(local_path, 'w') as out_file:
            out_file.write(buff.read())
        buff.seek(0)

    return buff


def _fetch_from_google_drive(doc_id: str) -> BytesIO:
    '''
    Downloads a file from Google drive and returns as an in-memory buffer.

    Args:
        doc_id (str): Identifier for the file on Google drive.
    '''
    # Check if it's already downloaded.
    url = _GOOGLE_URL.format(doc_id=doc_id)
    session = requests.session()

    # Based on gdown package implementation
    while True:
        r = session.get(url, stream=True, verify=True)
        r.raise_for_status()

        if 'Content-Disposition' in r.headers:
            break

        _, response = r.text.splitlines()
        search = re.search('id="downloadForm" action="(.+?)"', response)
        url = search.groups()[0].replace('&amp;', '&')
        r.close()

    buff = BytesIO()

    for chunk in r.iter_content(chunk_size=8192):
        buff.write(chunk)

    r.close()
    buff.seek(0)

    return buff


def _fetch_from_remote(url: str) -> BytesIO:
    '''
    Retrieves a file from a URL and downloads it, returning it as an in-memory
    buffer.

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


def _split_7_1_2(split: str, input_length: Optional[int],
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


def _stack_mismatched_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    '''
    Stacks a collection of :class:`torch.Tensor`s along the 0th dimension,
    where not every entry has the same shape. Each entry will be padded with
    NaNs to the same size. We currently only support stacking 2-dimensional
    :class:`torch.Tensor`s.
    '''
    tensors = [torch.as_tensor(x) for x in tensors]

    # Create the output holder
    shape = (len(tensors), *(max(s) for s in zip(*(t.shape for t in tensors))))
    dtype, device = tensors[0].dtype, tensors[0].device
    out = torch.full(shape, float('nan'), dtype=dtype, device=device)

    # Fill it in.
    for i, tensor in enumerate(tensors):
        out[i, :tensor.shape[0], :tensor.shape[1]] = tensor

    return out
