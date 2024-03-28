datasets Submodule
==================

LTSF Datasets
-------------

`LTSF <https://github.com/cure-lab/LTSF-Linear>`_ is a collection of time series forecasting datasets that are commonly used in benchmarking forecasting algorithms. Typically, the performance is reported as the mean squared error and mean absolute error over multiple forecasting horizons: 96, 192, 336, and 720 time steps.

.. autoclass:: torchcast.datasets.ElectricityLoadDataset

.. autoclass:: torchcast.datasets.ElectricityTransformerDataset

.. autoclass:: torchcast.datasets.ExchangeRateDataset

.. autoclass:: torchcast.datasets.GermanWeatherDataset

.. autoclass:: torchcast.datasets.ILIDataset

.. autoclass:: torchcast.datasets.SanFranciscoTrafficDataset

Monash Archive Datasets
-----------------------

The `Monash archive <https://forecastingdata.org>`_ is a collection of time series forecasting datasets in a standard format.

.. autoclass:: torchcast.datasets.MonashArchiveDataset

UCR/UEA Datasets
----------------

The `UCR/UEA archive <https://timeseriesclassification.com>`_ is a collection of time series classification datasets in a standard format. The UCR archive provides univariate time series, while the UEA archive provides multivariate time series.

.. autoclass:: torchcast.datasets.UCRDataset

.. autoclass:: torchcast.datasets.UEADataset

Other Datasets
--------------

.. autoclass:: torchcast.datasets.AirQualityDataset

