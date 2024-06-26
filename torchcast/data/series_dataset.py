from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch

from .utils import ArrayLike

__all__ = ['Metadata', 'SeriesDataset']


@dataclass
class Metadata:
    '''
    :class:`Metadata` encapsulates metadata about a multiseries. In a
    :class:`torchcast.data.SeriesDataset`, each multiseries will have a
    corresponding :class:`Metadata` object. All fields of :class:`Metadata` are
    optional. The fields that may be available are:
    
     * name: Name of the series.
     * channel_names: A list of the names of each channel.
     * series_names: A list of the names of each series.
    '''
    name: Optional[str] = None
    channel_names: Optional[List[str]] = None
    series_names: Optional[List[str]] = None

    def check_consistency(self, multiseries: ArrayLike):
        '''
        Checks if an array-like object is compatible with the metadata.
        '''
        if self.channel_names is not None:
            if len(self.channel_names) != multiseries.shape[1]:
                raise ValueError(
                    f'Number of channels in array ({multiseries.shape[1]}) '
                    f'does not match number of channel names ('
                    f'{len(self.channel_names)})'
                )
        if self.series_names is not None:
            if len(self.series_names) != multiseries.shape[0]:
                raise ValueError(
                    f'Number of series in array ({multiseries.shape[0]}) does '
                    f'not match number of series names '
                    f'({len(self.series_names)})'
                )


class SeriesDataset(torch.utils.data.Dataset):
    '''
    This is a base class for time series datasets. It is expected to only be
    used in a subclass, such as :class:`torchcast.data.TensorSeriesDataset`.

    Data held by a :class:`SeriesDataset` is always returned in shape
    (channels, time steps), so that it can be stacked to form a batch of series
    in shape (series, channels, time steps).
    '''
    def __init__(self, *data: ArrayLike, return_length: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 metadata: Optional[Union[Metadata, List[Metadata]]] = None):
        '''
        Args:
            data: The objects storing the underlying multiseries. The type will
                depend on the subclass, but should be array-like.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
            metadata (optional, list of :class:`Metadata`): If provided, should
                contain metadata about the series such as sequence names,
                channel names, etc. Should be a list of :class:`Metadata`
                objects of the same length as the number of multiseries.
        '''
        self.data = self._coerce_inputs(*data)
        self.transform = transform
        self.return_length = return_length
        if metadata is not None:
            if isinstance(metadata, Metadata):
                metadata = [metadata]
            if len(metadata) != len(data):
                raise ValueError(
                    f'Length of metadata {len(metadata)} and number of '
                    f'multiseries {len(data)} do not match'
                )
            for md, ms in zip(metadata, self.data):
                if md is not None:
                    md.check_consistency(ms)
        self.metadata = metadata

    def __getitem__(self, idx: int):
        if isinstance(idx, torch.Tensor):
            if idx.numel() == 1:
                idx = idx.item()
            else:
                raise TypeError('Index must be a single integer value')
        if (not isinstance(idx, int)) or (idx < 0):
            raise IndexError(idx)

        if self.return_length is None:
            out = [x[idx if (x.shape[0] > 1) else 0] for x in self.data]
        else:
            i, t = self._find_i_t(idx)
            out = []
            for x in self.data:
                j = i if (x.shape[0] > 1) else 0
                if x.shape[2] != 1:
                    out.append(x[j, :, t:t + self.return_length])
                else:
                    out.append(x[j, :, :])

        # TODO: This syntax is awkward...
        out = [torch.from_numpy(x.__array__()) for x in out]

        if self.transform is not None:
            out = self.transform(*out)

        return out[0] if (len(out) == 1) else out

    def __len__(self) -> int:
        if self.return_length is None:
            return self.shape[0]
        else:
            return sum(
                max(t_r + 1 - self.return_length, 0)
                for t_r in self._time_ranges
            )

    @staticmethod
    def _coerce_inputs(*data: ArrayLike):
        '''
        Coerces inputs to the appropriate form and checks that the shapes are
        correct. We break this out as a separate method so it can be overridden
        by subclasses.
        '''
        # Multiseries are allowed to have varying numbers of channels, but not
        # number of series or amount of time.
        if len({x.shape[0] for x in data} - {1}) > 1:
            raise ValueError(
                f'Conflicting number of series: {[x.shape for x in data]}'
            )
        if len({x.shape[2] for x in data} - {1}) > 1:
            raise ValueError(
                f'Conflicting time length: {[x.shape for x in data]}'
            )
        return data

    def _find_i_t(self, idx: int) -> Tuple[int, int]:
        '''
        Convenience function to convert a flat index to the appropriate indexes
        of the sequence and time.
        '''
        t = idx
        for i, max_t in enumerate(self._time_ranges):
            if t <= (max_t - self.return_length):
                break
            t -= max(max_t + 1 - self.return_length, 0)
        else:
            raise IndexError(idx)
        return i, t

    @property
    def _time_ranges(self) -> List[int]:
        return [
            max(x[i if x.shape[0] > 1 else 0].shape[1] for x in self.data)
            for i in range(self.shape[0])
        ]

    @property
    def shape(self) -> Tuple[int]:
        return (
            max(x.shape[0] for x in self.data),
            -1,
            max(x.shape[2] for x in self.data)
        )

    def split_by_time(self, t: Union[int, float]) \
            -> Tuple['SeriesDataset', 'SeriesDataset']:
        '''
        Splits the dataset by time.

        Args:
            t (int or float): If this is an integer, then perform the split at
                this time. If it is a float, perform the split at this
                percentage of the time.
        '''
        def _split_by_time(ms: ArrayLike, t: Union[int, float]):
            if ms.shape[2] == 1:
                return ms, ms
            else:
                if isinstance(t, float):
                    t = int(ms.shape[2] * t)
                return ms[:, :, :t], ms[:, :, t:]

        data_a, data_b = zip(*(_split_by_time(ms, t) for ms in self.data))
        ds_a = SeriesDataset(
            *data_a, return_length=self.return_length, transform=self.transform,
            metadata=self.metadata,
        )
        ds_b = SeriesDataset(
            *data_b, return_length=self.return_length, transform=self.transform,
            metadata=self.metadata,
        )
        return ds_a, ds_b
