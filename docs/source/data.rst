data Submodule
==============

This submodule contains tools for working with time series datasets. For clarity, we need some definitions:

 * An *array* is a Python object that has a similar signature to a numpy array or PyTorch tensor for indexing. In particular, it has shape and ndim properties and has a length.

 * A *series* is a 2-dimensional array, where the 0th dimension indexs channel and the 1st dimension indexs time. If a 1-dimensional array is passed to a function expecting a series, it will be interpreted as a univariate series and coerced to 2 dimensions.

 * A *multiseries* is a 3-dimensional array consisting of a collection of series, where the 0th dimension indexs the series, the 1st dimension indexs the channel, and the 2nd dimension indexs the time. If a 2-dimensional array is passed to a function expecting a multiseries, it will be interpreted as a single multivariate series and coerced to 3 dimensions.

 * A *dataset* is a collection of one or more multiseries. The multiseries in the dataset must all have broadcastable shapes, except for the number of channels, which is allowed to vary. That is, for each multiseries in the dataset, the 0th (series) and 2nd (time) dimensions must either be equal or one.

.. autoclass:: torchcast.data.SeriesDataset
  :members:

.. autoclass:: torchcast.data.TensorSeriesDataset
  :members:

.. autoclass:: torchcast.data.H5SeriesDataset
  :members:

Utility Classes
---------------

.. autoclass:: torchcast.data.Metadata
  :members:

.. autoclass:: torchcast.data.ListOfTensors
  :members:
