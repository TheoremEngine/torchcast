from typing import Tuple
import warnings

import torch

from .series_dataset import SeriesDataset

__all__ = ['TensorSeriesDataset']


class TensorSeriesDataset(SeriesDataset):
    '''
    This encapsulates one or more :class:`torch.Tensor` containing a series as
    a dataset, for use in a :class:`torch.utils.data.DataLoader`.
    '''
    @staticmethod
    def _coerce_inputs(*series):
        # Coerce to torch.Tensor
        series = [torch.as_tensor(x) for x in series]
        # Track original shape so it can be properly reported in error messages
        # if necessary
        shapes = [x.shape for x in series]
        # Calculate n, t values. We assume that a 1-dimensional series is a
        # single univariate series, while a 2-dimensional series is a single
        # multivariate series.
        n = max((s[0] if len(s) == 3 else 1) for s in shapes)
        t = max(s[-1] for s in shapes)

        # In this loop, we loop over the index i so that we can change the
        # entry in the series list itself. We will coerce everything to NCT
        # arrangement, and check for mismatches in shapes.
        for i in range(len(series)):
            if series[i].ndim == 1:
                warnings.warn(
                    f'Received tensor of shape {series[i].shape}, assuming it '
                    f'is a single univariate series.'
                )
                series[i] = series[i].view(1, 1, -1)
            elif series[i].ndim == 2:
                warnings.warn(
                    f'Received tensor of shape {series[i].shape}, assuming it '
                    f'is a single multivariate series.'
                )
                series[i] = series[i].unsqueeze(0)
            elif series[i].ndim > 3:
                raise ValueError(f'Received series of shape {series[i].shape}')

            if series[i].shape[0] not in {1, n}:
                raise ValueError(f'Mismatch in shapes: {shapes}')
            if series[i].shape[2] not in {1, t}:
                raise ValueError(f'Mismatch in shapes: {shapes}')

        return series

    def _get_storage_shape(self) -> Tuple[int, int]:
        n = max(x.shape[0] for x in self.series)
        t = max(x.shape[2] for x in self.series)
        return n, t

    def _retrieve_from_series(self, idx: int, t_0: int, t_1: int):
        # Indexing tensors generates views, so this is a no-op.
        out = tuple(x[0] if (x.shape[0] == 1) else x[idx] for x in self.series)
        return tuple(x if (x.shape[1] == 1) else x[:, t_0:t_1] for x in out)
