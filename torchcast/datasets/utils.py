import csv
import gzip
from io import BytesIO
import lzma
import os
import re
import tarfile
from typing import Dict, List, Optional
import zipfile

import numpy as np
import requests
import torch

__all__ = ['load_ts_file']


_GOOGLE_URL = 'https://drive.google.com/uc?id={doc_id}'


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


def _load_csv_file(path: str, delimiter: str = ',') -> Dict[str, List]:
    '''
    Convenience function for loading a csv file. The file is returned as a
    dictionary, whose keys are given by the header and whose values are lists
    giving the contents of the columns.

    Args:
        path (str): Path to file to load.
    '''
    with open(path, 'r') as record_file:
        records = list(zip(*csv.reader(record_file, delimiter=delimiter)))

    return {r[0]: r[1:] for r in records}


def _stack_mismatched_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    '''
    Stacks a collection of :class:`torch.Tensor`s along the 0th dimension,
    where not every entry has the same shape. Each entry will be padded with
    NaNs to the same size. We currently only support stacking 2-dimensional
    :class:`torch.Tensor`s.
    '''
    # Create the output holder
    shape = (len(tensors), *(max(s) for s in zip(*(t.shape for t in tensors))))
    dtype, device = tensors[0].dtype, tensors[0].device
    out = torch.full(shape, float('nan'), dtype=dtype, device=device)

    # Fill it in.
    for i, tensor in enumerate(tensors):
        out[i, :tensor.shape[0], :tensor.shape[1]] = tensor

    return out


def load_ts_file(series_file: str) -> torch.Tensor:
    '''
    Parses a .ts file from the scikit-time package. This is documented at:

        https://github.com/alan-turing-institute/sktime/blob/main/examples/loading_data.ipynb

    This returns a batch of series, in the form of a 3-dimensional
    :class:`torch.Tensor` in NCT arrangement.
    '''
    if isinstance(series_file, str):
        with open(series_file, 'r') as series_file:
            return load_ts_file(series_file)

    # Valid keys in the ts header:
    #
    # problemName (str)
    # timeStamps (bool)
    # missing (bool)
    # univariate (bool)
    # dimensions (int)
    # equalLength (bool)
    # seriesLength (int)
    # classLabel (bool, followed by Sequence[str])

    in_header, header, values, labels = True, {}, [], []

    for line in series_file.readlines():
        # Remove '\n' from line
        line = line.strip('\n')
        if in_header:
            # ts files allow comments
            if line.startswith('#'):
                continue

            elif line.startswith('@data'):
                is_sparse = (header.get('@timeStamps', 'false') == 'true')
                n_t = int(header.get('@seriesLength', '0'))
                if header.get('@univariate', 'false') == 'true':
                    n_dim = 1
                else:
                    n_dim = int(header['@dimensions'])

                categories = header.get('@classLabel', 'false')
                use_labels, *category_names = categories.split()
                use_labels = (use_labels == 'true')
                in_header = False

            elif line.startswith('@'):
                key, value = line.split(' ', 1)
                header[key] = value

            else:
                raise ValueError(f'Cannot parse: {line}')

        else:
            if use_labels:
                # When labels are present, they are appended to the end of the
                # line following a ':'.
                line, label = line.rsplit(':', 1)
                labels.append(category_names.index(label))

            if is_sparse:
                value = [
                    np.full((n_t,), np.nan, dtype=np.float32)
                    for _ in range(n_dim)
                ]
                for dim, row in enumerate(line.split(':')):
                    for entry in row.strip('()').split('),('):
                        t, v = entry.split(',')
                        value[dim][int(t)] = float(v)

            else:
                value = [
                    [np.array(x, dtype=np.float32) for x in row.split(',')]
                    for row in line.replace('?', 'nan').split(':')
                ]

            values.append(value)

    if not n_t:
        n_t = max(max(len(row) for row in value) for value in values)

    series = np.full((len(values), n_dim, n_t), np.nan, dtype=np.float32)
    for i, line in enumerate(values):
        for j, row in enumerate(line):
            series[i, j, :len(row)] = row

    series = torch.from_numpy(series)

    if use_labels:
        labels = torch.as_tensor(labels)
        return series, labels
    else:
        return series
