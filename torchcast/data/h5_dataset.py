from typing import Callable, List, Optional, Tuple, Union

import h5py
import numpy as np

from .series_dataset import Metadata, SeriesDataset

__all__ = ['H5SeriesDataset']

View = List[Union[int, slice]]


class H5View:
    '''
    :class:`H5View` wraps an :class:`h5py.Dataset` to allow us to index it as a
    view, without calling the view into memory. It supports a limited subset of
    indexing operations; it can be passed positive integers or slices. It does
    NOT support non-standard steps, negative integers, or other more
    complicated indexing schemes.
    '''
    def __init__(self, h5_data: h5py.Dataset, view: Optional[View] = None):
        if view is None:
            view = []
        if len(view) < h5_data.ndim:
            view += [slice(0, s) for s in h5_data.shape[len(view):]]
        # TODO: Should coerce 1- and 2-dimensional datasets instead of throwing
        # an error.
        if h5_data.ndim != 3:
            raise ValueError('H5 dataset is not 3-dimensional')
        self.view = view
        self.h5_data = h5_data

    def __array__(self) -> np.ndarray:
        return self.h5_data.__getitem__(tuple(self.view))

    def __getitem__(self, idx) -> 'H5View':
        if not isinstance(idx, tuple):
            idx = (idx,)
        # j will index idx
        new_view, j = [], 0

        for i in self.view:
            if j >= len(idx):
                new_view.append(i)
            elif isinstance(i, int):
                new_view.append(i)
            elif isinstance(i, slice) and isinstance(idx[j], int):
                if idx[j] < 0:
                    raise IndexError('H5View does not negative indices')
                new_view.append((i.start or 0) + idx[j])
                j += 1
            elif isinstance(i, slice) and isinstance(idx[j], slice):
                # We can safely assume i.start, i.stop are ints. We cannot
                # assume that of idx[j].start, idx[j].stop
                if idx[j].step is not None:
                    raise IndexError('H5View does not support step')
                j_start = idx[j].start or 0
                j_stop = idx[j].stop or (i.stop - i.start)
                if (j_start < 0) or (j_stop < 0):
                    raise IndexError('H5View does not negative indices')
                if j_stop + i.start > i.stop:
                    raise IndexError('Index out of range')
                s = slice(i.start + j_start, i.start + j_stop)
                new_view.append(s)
                j += 1
            else:
                raise TypeError(idx[j])

        if j < len(idx):
            raise IndexError(idx)

        return H5View(self.h5_data, new_view)

    def __len__(self) -> int:
        try:
            return self.shape[0]
        except IndexError:
            return 1

    @property
    def ndim(self) -> int:
        return sum(isinstance(s, slice) for s in self.view)

    @property
    def shape(self) -> Tuple[int]:
        return tuple(
            s.stop - s.start for s in self.view if isinstance(s, slice)
        )


class H5SeriesDataset(SeriesDataset):
    '''
    This encapsulates a :class:`h5py.File` containing a series stored on disk.
    '''
    def __init__(self, path: str, keys: Union[List[str], str],
                 return_length: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 metadata: Optional[Union[Metadata, List[Metadata]]] = None):
        '''
        Args:
            path (str): Path to the HDF5 file.
            keys (list of str): The keys in the file to return.
            return_length (optional, int): Length of the sequence to be
                returned when the dataset is sampled.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            metadata (optional, list of :class:`Metadata`): If provided, should
                contain metadata about the series such as sequence names,
                channel names, etc. Should be a list of :class:`Metadata`
                objects of the same length as the number of multiseries. If not
                provided, the metadata will attempt to be extracted from the
                HDF5 file.
        '''
        self.h5_file = h5py.File(path, 'r')

        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            if key not in self.h5_file:
                raise ValueError(f'{key} not found in {path}')

        # If metadata is not provided, try to extract from the file.
        if metadata is None:
            metadata = [
                _extract_metadata_from_attrs(self.h5_file[k].attrs)
                for k in keys
            ]
            if all(x is None for x in metadata):
                metadata = None

        # H5View is used so we can index the dataset without calling the whole
        # thing into memory. Its __init__ method also handles issuing warnings
        # about coercing shapes.
        super().__init__(
            *(H5View(self.h5_file[k]) for k in keys),
            return_length=return_length,
            transform=transform,
            metadata=metadata,
        )


def _extract_metadata_from_attrs(attrs: h5py.AttributeManager) \
        -> Optional[Metadata]:
    '''
    Extracts metadata for an :class:`h5py.Dataset` from the
    :class:`h5py.AttributeManager`. Returns None if it is not available.
    '''
    name = attrs.get('name')
    if 'channel_names' in attrs:
        channel_names = attrs.get('channel_names').tolist()
    else:
        channel_names = None
    if 'series_names' in attrs:
        series_names = attrs.get('series_names').tolist()
    else:
        series_names = None

    if all(x is None for x in (name, channel_names, series_names)):
        return None

    return Metadata(
        name=name, channel_names=channel_names, series_names=series_names,
    )
