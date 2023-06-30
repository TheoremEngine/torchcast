from math import prod
from typing import Callable, Iterator, Optional, Union

import torch


def _ensure_nct(series: torch.Tensor, time_dim: int = -1,
                batch_dim: Optional[Union[int, Iterator[int]]] = None,
                keepdim: bool = False) \
        -> (torch.Tensor, Callable):
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
