from math import prod
from typing import Callable, Iterable, Optional, Union

import torch


OneOrMoreDims = Union[int, Iterable[int]]


def _ensure_nct(series: torch.Tensor, time_dim: int = -1,
                batch_dim: Optional[OneOrMoreDims] = None,
                keepdim: bool = False) -> (torch.Tensor, Callable):
    '''
    Given an input series, permutes and rearranges it to ensure it is in NCT
    arrangement. Returns the series as a :class:`torch.Tensor`, along with a
    function that will reshape it back to its original form after the batch
    dimensions have been contracted:

    .. code-block::

        tensor, restore_shape = _ensure_nct(series, ...)
        tensor = do_stuff(tensor)
        tensor = restore_shape(tensor)

    Args:
        series (:class:`torchcast.Series` or :class:`torch.Tensor`): The series
        to permute.
        time_dim (int): The dimension to use as the time dimension.
        batch_dim (optional, int or iterator of int): The dimension(s) to use
        as the batch dimension(s). If not set, assume the series has no batch
        dimension.
        keepdim (bool): Modify the function so that the restore_shape function
        preserves the batch dimensions as 1s.
    '''
    if not isinstance(series, torch.Tensor):
        raise TypeError(series)

    # Ensure dim > 0 for convenience.
    time_dim = time_dim if (time_dim >= 0) else (time_dim + series.ndim)
    if batch_dim is None:
        batch_dim = []
    elif isinstance(batch_dim, int):
        batch_dim = [batch_dim]
    # Check dim != channel_dim
    if time_dim in batch_dim:
        raise ValueError('The time dimension cannot be a batch dimension.')
    channel_dims = [
        d for d in range(series.ndim) if d not in {*batch_dim, time_dim}
    ]

    N = prod((series.shape[d] for d in batch_dim))
    C = prod((series.shape[d] for d in channel_dims))
    T = series.shape[time_dim]

    idx_permute = (*batch_dim, *channel_dims, time_dim)

    rtn_shape = tuple(
        1 if d in batch_dim else series.shape[d]
        for d in idx_permute
    )
    rtn_permute = tuple(idx_permute.index(d) for d in range(series.ndim))

    series = series.permute(*idx_permute).reshape(N, C, T)

    def restore_shape(x):
        x = x.reshape(*rtn_shape).permute(*rtn_permute)
        if not keepdim:
            x = x.squeeze(batch_dim)
        return x

    return series, restore_shape


def _sliding_window_view(x: torch.Tensor, window_size: int, dim: int = -1) \
        -> torch.Tensor:
    '''
    Given an input tensor, uses stride tricks to create a tensor where there is
    a new dimension that acts as a sliding window along when of the original
    dimensions. For example:

    .. code::
        >>> x = torch.arange(9).view(3, 3)
        >>> print(x)
        tensor([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])
        >>> x = tc.utils._shaping._sliding_window_view(x, 2, dim=0)
        >>> print(x.shape)
        torch.Size([2, 2, 3])
        >>> print(x[0])
        tensor([[0, 1, 2],
                [3, 4, 5]])
        >>> print(x[1])
        tensor([[3, 4, 5],
                [6, 7, 8]])

    Args:
        x (:class:`torch.Tensor`): Tensor to cut into sliding windows.
        window_size (int): Width of window.
        dim (int): Dimension to form windows on.
    '''
    if (x.ndim < dim) or (x.shape[dim] < window_size):
        raise ValueError(
            f'Tensor shape {x.shape} does not support window of size '
            f'{window_size} on dimension {dim}'
        )
    dim = (x.ndim + dim) if (dim < 0) else dim

    shape = (
        *x.shape[:dim], x.shape[dim] + 1 - window_size, window_size,
        *x.shape[dim + 1:]
    )
    stride = x.stride()
    stride = (*stride[:dim], stride[dim], stride[dim], *stride[dim + 1:])
    return torch.as_strided(x, size=shape, stride=stride)
