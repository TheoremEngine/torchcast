from copy import copy
from typing import Callable, List, Optional, Tuple, Union

import torch

__all__ = ['SeriesDataset']


class SeriesDataset(torch.utils.data.Dataset):
    def __init__(self, *series, transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            series: The objects storing the underlying time series. The type
            will depend on the subclass.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        self.series = self._coerce_inputs(*series)
        self.transform = transform
        self.return_length = return_length

        n, t = self._get_storage_shape()
        self._time_range = (0, t)
        self._index_range = (0, n)

    @staticmethod
    def _coerce_inputs(self, *series):
        '''
        Coerces inputs to the appropriate form. We break this out as a separate
        method so it can be overridden by subclasses, where the data is not
        stored as an in-memory :class:`torch.Tensor`.
        '''
        raise NotImplementedError()

    def _get_storage_shape(self) -> Tuple[int, int]:
        '''
        Provides the shape of the underlying storage object.
        '''
        raise NotImplementedError()

    def _retrieve_from_series(self, idx: int, t_0: int, t_1: int) \
            -> List[torch.Tensor]:
        '''
        Retrieves values from the series. We break this out as a separate
        method so it can be overridden by subclasses, where the data is not
        stored as an in-memory :class:`torch.Tensor`.

        Args:
            idx (int): Index of the series to retrieve from.
            t_0 (int): Time to begin retrieving from.
            t_1 (int): Time to end retrieving from.
        '''
        raise NotImplementedError()

    def __getitem__(self, idx: int):
        if idx < 0:
            raise NotImplementedError('Negative indices are not supported')

        i_start, i_end = self._index_range
        t_start, t_end = self._time_range

        if self.return_length is None:
            if idx + i_start >= i_end:
                raise IndexError(idx)
            idx, t_0, t_1 = idx + i_start, t_start, t_end
        else:
            n_per_row = t_end - t_start - self.return_length + 1
            if idx >= n_per_row * (i_end - i_start):
                raise IndexError(idx, n_per_row, *self._time_range)
            idx, t_0 = i_start + idx // n_per_row, t_start + (idx % n_per_row)
            t_1 = t_0 + self.return_length

        series = self._retrieve_from_series(idx, t_0, t_1)

        if self.transform is not None:
            series = self.transform(*series)

        return series[0] if (len(series) == 1) else series

    def __len__(self) -> int:
        n_rows = self._index_range[1] - self._index_range[0]

        if self.return_length is None:
            return n_rows
        else:
            n_t = self._time_range[1] - self._time_range[0]
            return n_rows * (n_t - self.return_length + 1)

    def split_by_time(self, t: Union[int, float]):
        t_start, t_end = self._time_range
        if isinstance(t, float):
            t = int((t_end - t_start) * t)
        if (t <= 0) or (t >= t_end - t_start):
            raise ValueError(f'Split {t} out of range')

        # Shallow copy is done here deliberately to ensure backend storage is
        # not copied. It may be necessary to override __copy__ in some
        # subclasses.
        ds_1, ds_2 = copy(self), copy(self)

        ds_1._time_range = (t_start, t_start + t)
        ds_2._time_range = (t_start + t, t_end)

        return ds_1, ds_2
