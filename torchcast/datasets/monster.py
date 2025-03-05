import os
from typing import Callable, Optional, Union

import numpy as np
import torch

from ..data import Metadata, TensorSeriesDataset
from .utils import _download_and_extract

__all__ = ['MonsterDataset']


MONSTER_TASKS = [
    'AudioMNIST', 'AudioMNIST-DS', 'CornellWhaleChallenge', 'CrowdSourced',
    'DREAMERA', 'DREAMERV', 'FordChallenge', 'FruitFlies', 'InsectSound',
    'LakeIce', 'LenDB', 'MosquitoSound', 'Opportunity', 'PAMAP2', 'Pedestrian',
    'S2Agri-10pc-17', 'S2Agri-10pc-34', 'S2Agri-17', 'S2Agri-34', 'Skoda',
    'STEW', 'TimeSen2Crop', 'Tiselac', 'Traffic', 'UCIActivity', 'USCActivity',
    'WhaleSounds', 'WISDM', 'WISDM2',
]
MONSTER_URL = 'https://huggingface.co/datasets/monster-monash/{task}/resolve/main/'  # noqa
MONSTER_DATA_FILE_NAME = '{task}_X.npy'
MONSTER_LABEL_FILE_NAME = '{task}_y.npy'
MONSTER_INDEX_FILE_NAME = 'test_indices_fold_{i}.txt'


class MonsterDataset(TensorSeriesDataset):
    '''
    This is Monster time series classification dataset, first published in
    `Dempster et al. 2025 <https://arxiv.org/abs/2502.15122>`__.
    '''
    def __init__(self, task: str, split: str = 'train',
                 fold: int = 0, path: Optional[str] = None,
                 download: Union[bool, str] = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            task (str): Which dataset to retrieve.
            split (str): What split of the data to return. Choices: 'all',
                'train', 'test'.
            fold (int): Which train-test fold to return. Choices are 0, 1, 2,
                3, 4.
            path (optional, str): Path to find the dataset at.
            download (bool or str): Whether to download the dataset if it is
                not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
                before returning.
            return_length (optional, int): If provided, the length of the
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if split not in {'all', 'train', 'test'}:
            raise ValueError(
                f'Did not recognize split {split}; choose from '
                f'{{"all", "train", "test"}}'
            )
        if task not in MONSTER_TASKS:
            raise ValueError(
                f'Did not recognize task {task}; choose from {MONSTER_TASKS}'
            )
        if (split != 'all') and (fold not in {0, 1, 2, 3, 4}):
            raise ValueError(
                f'Did not recognize fold {fold}; choose from {{0, 1, 2, 3, 4}}'
            )

        x_name = MONSTER_DATA_FILE_NAME.format(task=task)
        buff_x = _download_and_extract(
            os.path.join(MONSTER_URL.format(task=task), x_name),
            x_name,
            path,
            download=download,
        )
        array_x = torch.from_numpy(np.load(buff_x))

        y_name = MONSTER_LABEL_FILE_NAME.format(task=task)
        buff_y = _download_and_extract(
            os.path.join(MONSTER_URL.format(task=task), y_name),
            y_name,
            path,
            download=download,
        )
        array_y = torch.from_numpy(np.load(buff_y)).view(-1, 1, 1).long()

        if split != 'all':
            idxs_name = MONSTER_INDEX_FILE_NAME.format(i=fold)
            buff_idxs = _download_and_extract(
                os.path.join(MONSTER_URL.format(task=task), idxs_name),
                idxs_name,
                path,
                download=download,
            )
            test_idxs = torch.from_numpy(np.loadtxt(buff_idxs, dtype=np.int64))
            if split == 'test':
                array_x, array_y = array_x[test_idxs], array_y[test_idxs]
            else:
                mask = torch.ones(array_y.shape[0:1], dtype=torch.bool)
                mask[test_idxs] = False
                array_x, array_y = array_x[mask, ...], array_y[mask, ...]

        super().__init__(
            array_x, array_y,
            metadata=[Metadata(name='Data'), Metadata(name='Labels')],
            transform=transform,
            return_length=return_length,
        )
