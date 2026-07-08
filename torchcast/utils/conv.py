from typing import Iterable, Optional, Union

import torch
import torch.nn.functional as f

from ._shaping import _ensure_nct


__all__ = ['moving_average']

Ints = Union[int, Iterable[int]]


def moving_average(x: torch.Tensor, window: Ints, dim: int = -1,
                   batch_dim: Optional[Ints] = None) -> torch.Tensor:
    '''
    Applies a moving average to a time series.

    Args:

        x (:class:`torch.Tensor`): Time series to calculate the moving average
            of.
        window (int or iterable of int): The window or windows to apply. If an
            iterable of integers is provided, these are treated as nested
            windows - for example, if window is (4, 2), two moving averages
            will be applied, the first of size 4 and the second of size 2.
    '''
    if not x.is_floating_point():
        x = x.float()
    x, restore_shape = _ensure_nct(x, dim, batch_dim, allow_time_changes=True)

    if isinstance(window, int):
        window = [window]

    kernels = [
        torch.full((1, 1, w), 1. / w, dtype=x.dtype, device=x.device)
        for w in window
    ]
    kernel = kernels[0]
    for k in kernels[1:]:
        kernel = f.conv_transpose1d(kernel, k)

    # Since we're applying the same kernel to each channel, we can do this more
    # efficiently by moving channels into the batch dimension.
    x = x.view(x.shape[0] * x.shape[1], 1, x.shape[2])
    x = f.conv1d(x, kernel)

    return restore_shape(x)
