import gzip
from io import BytesIO
import lzma
import os
import re
import tarfile
from typing import List, Optional
import zipfile

import numpy as np
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


def _download_and_extract(url: str, local_path: str,
                          file_name: Optional[str] = None) -> str:
    '''
    Convenience function to download a remote file, decompress it if necessary,
    and write it to disk.

    Args:
        url (str): The URL to download the data from.
        local_path (str): The local path to put the extracted data in.
        file_name (optional, str): If provided, only extract this file from the
        downloaded archive.

    Returns:
        The path to the file written to disk.
    '''
    # First, infer whether local_path is a directory or the path of the file.
    # Make sure target location exists.
    if file_name is not None:
        if local_path.endswith(file_name):
            local_root = os.path.dirname(local_path)
        else:
            local_root = local_path
            local_path = os.path.join(local_path, file_name)
    else:
        local_root = local_path

    os.makedirs(local_root, exist_ok=True)

    # Let's get that file!
    buff = _fetch_from_remote(url)

    # Extract, if needed.
    while True:
        url, ext = os.path.splitext(url.lower())

        if ext == '.gz':
            buff = BytesIO(gzip.decompress(buff.read()))
        elif ext == '.xz':
            buff = BytesIO(lzma.decompress(buff.read()))
        elif ext == '.tar':
            with tarfile.TarFile(fileobj=buff, mode='r') as tar_file:
                if file_name is None:
                    tar_file.extractall(local_root)
                else:
                    tar_file.extract(file_name, path=local_root)
                return local_path
        elif ext == '.zip':
            with zipfile.ZipFile(buff) as zip_file:
                if file_name is None:
                    zip_file.extractall(local_root)
                else:
                    zip_file.extract(file_name, path=local_root)
                return local_path
        else:
            if file_name is None:
                file_name = os.path.basename(url + ext)
            if not local_path.endswith(file_name):
                local_path = os.path.join(local_root, file_name)
            with open(local_path, 'wb') as out_file:
                out_file.write(buff.read())
            return local_path


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
