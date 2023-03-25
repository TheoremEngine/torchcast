import csv
import lzma
import os
import re
import tarfile
from typing import Dict, List, Optional
import urllib
import zipfile

import numpy as np
import requests
import torch

__all__ = ['load_ts_file']


_GOOGLE_URL = 'https://drive.google.com/uc?id={doc_id}'


def _fetch_from_google_drive(root_path: Optional[str], doc_id: str):
    '''
    Downloads a file from Google drive.

    Args:
        root_path (optional, str): The path of the directory to download the
        file to. If this is None, then use the environment variable
        TORCHCAST_DATASETS. If that's not set either, default to './data'.
        doc_id (str): Identifier for the file on Google drive.
    '''
    # Check if it's already downloaded.
    root_path = root_path or os.getenv('TORCHCAST_DATASETS', './data')
    os.makedirs(root_path, exist_ok=True)
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

    content_disp = urllib.parse.unquote(r.headers['Content-Disposition'])
    m = re.search(r"filename\*=UTF-8''(.*)", content_disp).groups()[0]
    name = m.replace(os.path.sep, '_')

    # Check if it's already downloaded
    if os.path.exists(root_path) and (os.path.basename(root_path) == name):
        return root_path
    local_path = os.path.join(root_path, name)
    if os.path.exists(local_path):
        return local_path

    with open(local_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    r.close()


def _fetch_to_local(path: str, url: str) -> str:
    '''
    Retrieves a file from a URL and downloads it to a local path, returning the
    local path.

    Args:
        path (str): The path to download the file to.
        url (str): URL to download.

    Returns:
        The path to the now-downloaded file.
    '''
    name = os.path.basename(url)

    # Coerce path to point to the target FILE, not the target DIRECTORY.
    if os.path.basename(path) != name:
        path = os.path.join(path, name)
    # If it's already present, we're done.
    if os.path.exists(path):
        return path
    # Make sure the target directory exists.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Syntax is taken from:
    # https://stackoverflow.com/questions/16694907/download-large-file-in- \
    #     python-with-requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    return path


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


def _unzip_archive(path: str, out_path: Optional[str] = None,
                   file_name: Optional[str] = None):
    '''
    Unzips an archive file.

    Args:
        path (str): Path to file to extract.
        out_path (optional, str): Directory to place extracted files; defaults
        to the directory containing the archive file.
        file_name (optional, str): If provided, only extract this file, not all
        files.
    '''
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if out_path is None:
        out_path = os.path.dirname(path)
    elif (file_name is not None) and (os.path.basename(out_path) == file_name):
        out_path = os.path.dirname(out_path)

    os.makedirs(out_path, exist_ok=True)

    def _extract(extractor):
        if file_name is None:
            extractor.extractall(out_path)
        else:
            # path needs to be a directory
            extractor.extract(file_name, path=out_path)

    if path.lower().endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zip_file:
            _extract(zip_file)
    elif path.lower().endswith('.tar'):
        with tarfile.open(path, 'r') as tar_file:
            _extract(tar_file)
    elif path.lower().endswith('.tar.gz'):
        with tarfile.open(path, 'r:gz') as tar_file:
            _extract(tar_file)
    elif path.lower().endswith('.tar.xz'):
        with lzma.open(path, 'rb') as xz_file:
            with tarfile.TarFile(fileobj=xz_file, mode='r') as tar_file:
                _extract(tar_file)
    else:
        raise ValueError(f'Do not understand how to extract {path}')


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
