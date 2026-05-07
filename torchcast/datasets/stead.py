from glob import glob
import os
from typing import Callable, Iterable, Optional, Tuple, Union

import h5py
import torch

from ..data import ListOfArrayLike, Metadata, SeriesDataset
from ..data.h5_dataset import H5View


class STEADDataset(SeriesDataset):
    def __init__(self, paths: Union[str, Iterable[str]],
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            paths (str or iterable of str): Path to find the dataset at. This
                can be either a directory, a path to a single HDF5 file, or an
                iterable of paths to HDF5 files.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if isinstance(paths, str):
            if os.path.isdir(paths):
                paths = glob(os.path.join(paths, '*.hdf5'))
            else:
                paths = [paths]

        label_dict = {'noise': 0, 'earthquake_local': 1}
        self.h5_files, series, labels, series_names = [], [], [], []
        for path in paths:
            h5_file = h5py.File(path, 'r')
            self.h5_files.append(h5_file)
            for k in h5_file['data'].keys():
                series.append(h5_file['data'][k])
                labels.append(label_dict[series[-1].attrs['trace_category']])
                series_names.append(k)

        series = ListOfArrayLike(series)
        labels = torch.tensor(labels).reshape(-1, 1, 1)

        metadata = [
            Metadata(name='Data', series_names=series_names),
            Metadata(name='Label', series_names=series_names),
        ]

        super().__init__(
            series, labels,
            return_length=return_length,
            transform=transform,
        )

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # Underlying data is stored as (T, C) so we need to transpose it.
        x, y = super().__getitem__(idx)
        return x.T, y
