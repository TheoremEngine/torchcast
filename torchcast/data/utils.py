from typing import List, Protocol, Tuple, Union
import warnings

import numpy as np
import torch


class ArrayLike(Protocol):
    '''
    :class:`ArrayLike` is used in type hinting to represent a class that has an
    API suitable for indexing similar to :class:`torch.Tensor`,
    :class:`numpy.ndarray`, or :class:`h5py.Dataset`.
    '''
    shape: Tuple[int]
    ndim: int


class ListOfTensors:
    '''
    This class encapsulates a list of :class:`torch.Tensor`. This is used so
    that we can have a single multiseries whose constituent series are varying
    lengths without wasting memory.
    '''
    def __init__(self, tensors: List[ArrayLike]):
        self.tensors = [_coerce_to_series(x) for x in tensors]
        if len({x.shape[0] for x in self.tensors}) != 1:
            raise ValueError(
                f'Mismatch in tensor shapes: {[x.shape for x in self.tensors]}'
            )
        if any(x.ndim != 2 for x in self.tensors):
            raise ValueError(
                'All tensors in a ListOfTensors must be dimension 2'
            )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.tensors[idx]
        elif isinstance(idx, slice):
            return type(self)(self.tensors[idx])
        elif isinstance(idx, tuple):
            if isinstance(idx[0], slice):
                out = [
                    tensor.__getitem__(idx[1:])
                    for tensor in self[idx[0]]
                ]
                return type(self)(out)
            else:
                return self.tensors[idx[0]].__getitem__(idx[1:])
        else:
            raise IndexError(idx)

    @property
    def ndim(self) -> int:
        return 3

    @property
    def shape(self) -> Tuple[int, int, int]:
        t = max(x.shape[1] for x in self.tensors)
        return (len(self.tensors), self.tensors[0].shape[0], t)


def _coerce_to_multiseries(x: ArrayLike) -> Union[torch.Tensor, ListOfTensors]:
    '''
    Convenience function to coerce an array-like object to a
    :class:`torch.Tensor` of 3 dimensions.
    '''
    # TODO: There should be a better way to handle this...
    if (
        isinstance(x, (list, tuple))
        and isinstance(x[0], (np.ndarray, torch.Tensor))
        and len({_x.shape[-1] for _x in x}) > 1
    ):
        return ListOfTensors(x)

    x = torch.as_tensor(x)
    if x.ndim == 1:
        warnings.warn(
            f'Received tensor of shape {x.shape}, assuming it is a single '
            f'univariate series.'
        )
        return x.view(1, 1, -1)
    elif x.ndim == 2:
        warnings.warn(
            f'Received tensor of shape {x.shape}, assuming it is a single '
            f'multivariate series.'
        )
        return x.unsqueeze(0)
    elif x.ndim == 3:
        return x
    else:
        raise ValueError(f'Received tensor of shape {x.shape}')


def _coerce_to_series(x: ArrayLike) -> torch.Tensor:
    '''
    Convenience function to coerce an array-like object to a
    :class:`torch.Tensor` of 2 dimensions.
    '''
    x = torch.as_tensor(x)
    if x.ndim == 1:
        warnings.warn(
            f'Received tensor of shape {x.shape}, assuming it is a single '
            f'univariate series.'
        )
        return x.view(1, -1)
    elif x.ndim == 2:
        return x
    else:
        raise ValueError(f'Received tensor of shape {x.shape}')
