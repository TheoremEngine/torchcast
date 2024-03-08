torchcast: A PyTorch Library for Time Series Analysis
=====================================================

torchcast is a simple module for time series analysis using deep learning. It can be downloaded from:

`https://github.com/TheoremEngine/torchcast <https://github.com/TheoremEngine/torchcast>`_

Or installed by:

.. code-block:: bash

    pip install torchcast

**torchcast is still under heavy construction.** The intent for torchcast, at least in the near term, is to focus on data fetching and preprocessing. torchcast currently supports automatic ingestion of time series classification datasets from the `UCR/UEA Archive <https://timeseriesclassification.com>`_, time series forecasting datasets from the `Monash Archive <https://forecastingdata.org>`_, and the `LSTF benchmark <https://github.com/cure-lab/LTSF-Linear>`_. Most of these datasets are small enough to hold in memory, where we standardize on a 3-dimensional (Batch, Channel, Time) arrangement. We intend to add additional datasets in the future.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   data
   datasets
   nn
   utils
