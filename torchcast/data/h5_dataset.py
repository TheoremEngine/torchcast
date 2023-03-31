from typing import Callable, Iterator, List, Optional, Tuple, Union

import h5py
import torch

from .series_dataset import SeriesDataset

__all__ = ['H5SeriesDataset']


class H5SeriesDataset(SeriesDataset):
    '''
    This encapsulates a :class:`h5py.File` containing a series stored on disk.
    '''
    def __init__(self, path: str, keys: Union[List[str], str],
                 return_length: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 channel_names: Optional[Iterator[str]] = None,
                 series_names: Optional[Iterator[str]] = None):
        '''
        Args:
            path (str): Path to the HDF5 file.
            keys (list of str): The keys in the file to return.
            return_length (optional, int): Length of the sequence to be
            returned when the dataset is sampled.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            channel_names (optional, iterator of str): If provided, the names
            of the channels.
            series_names (optional, iterator of str): If provided, the names of
            the series.
        '''
        self.h5_file = h5py.File(path, 'r')

        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key not in self.h5_file:
                raise ValueError(f'{key} not found in {path}')

        super().__init__(
            *(self.h5_file[k] for k in keys),
            return_length=return_length,
            transform=transform,
            channel_names=channel_names,
            series_names=series_names,
        )

    @staticmethod
    def _coerce_inputs(*series):
        if series[0].ndim != 3:
            raise ValueError(f'Received {series[0].ndim}-dimensional series')
        else:
            n = max(s.shape[0] for s in series)
            # Index by -1 instead of 2 in case an entry in series is not 3-
            # dimensional, so we can raise the proper exception below...
            t = max(s.shape[-1] for s in series)

        for x in series:
            if x.ndim != 3:
                raise ValueError(f'Received {x.ndim}-dimensional series')
            if (x.shape[0] not in {n, 1}) or (x.shape[2] not in {t, 1}):
                raise ValueError(
                    f'Mismatch in shapes: {[s.shape for s in series]}'
                )

        return series

    def _get_storage_shape(self) -> Tuple[int, int]:
        n = max(x.shape[0] for x in self.series)
        t = max(x.shape[2] for x in self.series)
        return n, t

    def _retrieve_from_series(self, idx: int, t_0: int, t_1: int):
        out = []
        for series in self.series:
            if (series.shape[0] != 1) and (series.shape[2] != 1):
                series = series[idx, :, t_0:t_1]
            elif (series.shape[0] == 1) and (series.shape[2] != 1):
                series = series[0, :, t_0:t_1]
            elif (series.shape[0] != 1) and (series.shape[2] == 1):
                series = series[idx, :, :]
            else:
                series = series[...]
            out.append(torch.tensor(series))
        return tuple(out)
