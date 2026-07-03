import os
from typing import Callable, List, Optional, Union

from ..data import TensorSeriesDataset
from .utils import _download_and_extract, _read_hf_arrow_buffer, has_pyarrow

__all__ = ['TIMEDataset']


TIME_TASKS = {
    'Australia_Solar': ['H'], 'Auto_Production_SF': ['M'],
    'CPHL': ['15T', '30T', 'H'], 'Coastal_T_S': ['15T', '20T', '5T', 'H'],
    'Commodity_Import': ['M'], 'Commodity_Production': ['M'], 'Crypto': ['D'],
    'ECDC_COVID': ['D', 'W'], 'EWELD_Load': ['15T'],
    'Finland_Traffic': ['15T'], 'Global_Influenza': ['W'],
    'Global_Price': ['Q'], 'Housing_Inventory': ['M'], 'JOLTS': ['M'],
    'Job_Claims': ['W'], 'MetroPT-3': ['5T'], 'NE_China_Wind': ['H'],
    'Oil_Price': ['B'], 'Online_Retail_2_UCI': ['D'],
    'OpenElectricity_NEM': ['5T'], 'Port_Activity': ['D', 'W'],
    'SG_Carpark': ['15T'], 'SG_PM25': ['H'], 'SG_Weather': ['D'],
    'Smart_Manufacturing': ['H'], 'Supply_Chain_Customer': ['D'],
    'Supply_Chain_Location': ['D'], 'US_Labor': ['M'],
    'US_Term_Structure': ['B'], 'Uncertainty_1M': ['M'],
    'Vehicle_Sales': ['M'], 'Vehicle_Supply': ['M'], 'WUI_Global': ['Q'],
    'Water_Quality_Darwin': ['15T'], 'azure2019_D': ['5T'],
    'azure2019_I': ['5T'], 'azure2019_U': ['5T'],
    'current_velocity': ['10T', '15T', '20T', '5T', 'H'],
    'epf_electricity_price': ['H']
}
TIME_URL = 'https://huggingface.co/datasets/Real-TSF/TIME/resolve/main/{task}/{freq}/data-00000-of-00001.arrow'  # noqa


class TIMEDataset(TensorSeriesDataset):
    '''
    This is TIME time series forecasting dataset, first published in
    `Qiao et al. 2026 <https://arxiv.org/abs/2602.12147>`__. It is intended for
    evaluating time series foundation models, and so has no train split.

    The dataset contains two series, the first being a timestamp and the second
    the multivariable series to be forecast.
    '''
    tasks: List[str] = list(TIME_TASKS.keys())

    def __init__(self, task: str, freq: Optional[str] = None,
                 path: Optional[str] = None, download: Union[bool, str] = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            task (str): Which dataset to retrieve.
            freq (optional, str): Frequency to retrieve. Most datasets have
                only one frequency, in which case this does not need to be
                specified.
            path (optional, str): Path to find the dataset at.
            download (bool or str): Whether to download the dataset if it is
                not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if not has_pyarrow:
            raise ImportError(
                'The TIMEDataset requires pyarrow to be installed.'
            )
        if task not in TIME_TASKS:
            raise ValueError(
                f'Did not recognize task {task}; choose from {TIME_TASKS}'
            )
        if freq is None:
            if len(TIME_TASKS[task]) == 1:
                freq, = TIME_TASKS[task]
            else:
                raise ValueError(
                    f'Task {task} has multiple frequencies, so frequency must '
                    f'be specified; choose from {TIME_TASKS[task]}'
                )

        url = TIME_URL.format(task=task, freq=freq)
        buff = _download_and_extract(
            url,
            os.path.basename(url),
            path,
            download=download,
        )
        tensors, metadata = _read_hf_arrow_buffer(buff)

        super().__init__(
            *tensors,
            return_length=return_length,
            transform=transform,
            metadata=metadata,
        )
