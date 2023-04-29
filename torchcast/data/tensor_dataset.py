from .series_dataset import SeriesDataset
from .utils import ArrayLike, _coerce_to_multiseries

__all__ = ['TensorSeriesDataset']


class TensorSeriesDataset(SeriesDataset):
    '''
    This encapsulates one or more :class:`torch.Tensor` containing a
    multiseries as a dataset, for use in a
    :class:`torch.utils.data.DataLoader`. The underlying data can be stored
    either as a :class:`torch.Tensor` or as a :class:`ListOfTensors`.
    '''
    @staticmethod
    def _coerce_inputs(*data: ArrayLike):
        data = [_coerce_to_multiseries(x) for x in data]
        return SeriesDataset._coerce_inputs(*data)
