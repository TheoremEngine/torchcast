import json
import os
from typing import Callable, Dict, List, Optional, Union

from ..data import TensorSeriesDataset
from .utils import _download_and_extract, _read_hf_arrow_buffer, has_pyarrow


__all__ = ['BOOMDataset']


BOOM_URL = 'https://huggingface.co/datasets/Datadog/BOOM/resolve/main/ds-{task}-{freq}/data-00000-of-00001.arrow'  # noqa


def _load_boom_manifest() -> Dict[int, str]:
    path = os.path.join(os.path.dirname(__file__), 'boom.json')
    with open(path, 'r') as tasks_file:
        return json.load(tasks_file)


class BOOMDataset(TensorSeriesDataset):
    '''
    This is the Benchmark Of Observability Metrics (BOOM) dataset, a dataset of
    time series collected by DataDog from monitoring of pre-production
    environments. More information can be found at `Cohen et al. 2025
    <https://arxiv.org/abs/2505.14766>`__.

    The dataset contains two series, the first being a timestamp and the second
    the multivariable series to be forecast.
    '''
    _manifest = _load_boom_manifest()

    def __init__(self, task: Union[int, str], path: Optional[str] = None,
                 download: Union[bool, str] = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            task (int or str): Which dataset to retrieve.
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
                'The BOOMDataset requires pyarrow to be installed.'
            )
        if isinstance(task, int):
            task = str(task)
        if task not in self._manifest:
            raise ValueError(f'Did not recognize {task}')

        url = BOOM_URL.format(task=task, freq=self._manifest[task])
        buff = _download_and_extract(
            url,
            os.path.basename(url),
            path,
            download=download,
        )
        tensors, metadata = _read_hf_arrow_buffer(buff, '%Y-%m-%dT%H:%M:%S')

        super().__init__(
            *tensors,
            metadata=metadata,
            return_length=return_length,
            transform=transform,
        )

    @property
    def tasks(self) -> List[str]:
        return list(self._manifest.keys())
