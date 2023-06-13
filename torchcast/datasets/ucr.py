from typing import Callable, Optional, Union

import torch

from ..data import Metadata, TensorSeriesDataset
from ._file_readers import parse_ts
from .utils import _download_and_extract

__all__ = ['UCRDataset', 'UEADataset']

ROOT_URL = 'http://www.timeseriesclassification.com/ClassificationDownloads/{name}.zip'  # noqa: E501

UCR_DATASETS = (
    'ACSF1', 'Adiac', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
    'AllGestureWiimoteZ', 'ArrowHead', 'Beef', 'BeetleFly', 'BirdChicken',
    'BME', 'Car', 'CBF', 'Chinatown', 'ChlorineConcentration', 'CinCECGTorso',
    'Coffee', 'Computers', 'CricketX', 'CricketY', 'CricketZ', 'Crop',
    'DiatomSizeReduction', 'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'DodgerLoopDay',
    'DodgerLoopGame', 'DodgerLoopWeekend', 'Earthquakes', 'ECG200', 'ECG5000',
    'ECGFiveDays', 'ElectricDevices', 'EOGHorizontalSignal',
    'EOGVerticalSignal', 'EthanolLevel', 'FaceAll', 'FaceFour', 'FacesUCR',
    'FiftyWords', 'Fish', 'FordA', 'FordB', 'FreezerRegularTrain',
    'FreezerSmallTrain', 'Fungi', 'GestureMidAirD1', 'GestureMidAirD2',
    'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2', 'GunPoint',
    'GunPointAgeSpan', 'GunPointMaleVersusFemale', 'GunPointOldVersusYoung',
    'Ham', 'HandOutlines', 'Haptics', 'Herring', 'HouseTwenty', 'InlineSkate',
    'InsectEPGRegularTrain', 'InsectEPGSmallTrain', 'InsectWingbeatSound',
    'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2', 'Lightning7',
    'Mallat', 'Meat', 'MedicalImages', 'MelbournePedestrian',
    'MiddlePhalanxOutlineAgeGroup', 'MiddlePhalanxOutlineCorrect',
    'MiddlePhalanxTW', 'MixedShapesRegularTrain', 'MixedShapesSmallTrain',
    'MoteStrain', 'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2',
    'OliveOil', 'OSULeaf', 'PhalangesOutlinesCorrect', 'Phoneme',
    'PickupGestureWiimoteZ', 'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
    'PLAID', 'Plane', 'PowerCons', 'ProximalPhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
    'RefrigerationDevices', 'Rock', 'ScreenType', 'SemgHandGenderCh2',
    'SemgHandMovementCh2', 'SemgHandSubjectCh2', 'ShakeGestureWiimoteZ',
    'ShapeletSim', 'ShapesAll', 'SmallKitchenAppliances', 'SmoothSubspace',
    'SonyAIBORobotSurface1', 'SonyAIBORobotSurface2', 'StarLightCurves',
    'Strawberry', 'SwedishLeaf', 'Symbols', 'SyntheticControl',
    'ToeSegmentation1', 'ToeSegmentation2', 'Trace', 'TwoLeadECG',
    'TwoPatterns', 'UMD', 'UWaveGestureLibraryAll', 'UWaveGestureLibraryX',
    'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'Wafer', 'Wine',
    'WordSynonyms', 'Worms', 'WormsTwoClass', 'Yoga'
)

UEA_DATASETS = (
    'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
    'CharacterTrajectories', 'Cricket', 'DuckDuckGeese', 'EigenWorms',
    'Epilepsy', 'ERing', 'EthanolConcentration', 'FaceDetection',
    'FingerMovements', 'HandMovementDirection', 'Handwriting', 'Heartbeat',
    'InsectWingbeat', 'JapaneseVowels', 'Libras', 'LSST', 'MotorImagery',
    'NATOPS', 'PEMS-SF', 'PenDigits', 'PhonemeSpectra', 'RacketSports',
    'SelfRegulationSCP1', 'SelfRegulationSCP2', 'SpokenArabicDigits',
    'StandWalkJump', 'UWaveGestureLibrary',
)


class UCRDataset(TensorSeriesDataset):
    '''
    This is the UCR dataset for univariate time series classification, found
    at:

        https://www.timeseriesclassification.com/

    '''
    def __init__(self, task: str, split: str = 'train',
                 path: Optional[str] = None,
                 download: Union[bool, str] = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            task (str): Which dataset to retrieve.
            split (str): Which split to retrieve; choose from 'train', 'test'.
            path (optional, str): Path to find the dataset at.
            download (bool or str): Whether to download the dataset if it is
            not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        if task not in UCR_DATASETS:
            raise ValueError(
                f'Did not recognize {task}; choose from {UCR_DATASETS}'
            )
        if split not in {'train', 'test'}:
            raise ValueError(f"Split should be 'train' or 'test', got {split}")

        buff = _download_and_extract(
            ROOT_URL.format(name=task),
            f'{task}_{split.upper()}.ts',
            path,
            download=download,
        )
        data, attrs = parse_ts(buff.read())
        data = torch.from_numpy(data)
        labels = torch.from_numpy(attrs['labels']).view(-1, 1, 1)

        meta = [Metadata(name='Data'), Metadata(name='Labels')]

        super().__init__(
            data, labels,
            return_length=return_length,
            transform=transform,
            metadata=meta,
        )


class UEADataset(TensorSeriesDataset):
    '''
    This is the UEA dataset for multivariate time series classification, found
    at:

        https://www.timeseriesclassification.com/

    '''
    def __init__(self, task: str, split: str = 'train',
                 path: Optional[str] = None,
                 download: Union[bool, str] = True,
                 transform: Optional[Callable] = None,
                 return_length: Optional[int] = None):
        '''
        Args:
            task (str): Which dataset to retrieve.
            split (str): Which split to retrieve; choose from 'train', 'test'.
            path (optional, str): Path to find the dataset at.
            download (bool or str): Whether to download the dataset if it is
            not already available. Can be true, false, or 'force'.
            transform (optional, callable): Pre-processing functions to apply
            before returning.
            return_length (optional, int): If provided, the length of the
            sequence to return. If not provided, returns an entire sequence.
        '''
        if task not in UEA_DATASETS:
            raise ValueError(
                f'Did not recognize {task}; choose from {UEA_DATASETS}'
            )
        if split not in {'train', 'test'}:
            raise ValueError(f"Split should be 'train' or 'test', got {split}")

        buff = _download_and_extract(
            ROOT_URL.format(name=task),
            f'{task}_{split.upper()}.ts',
            path,
            download=download,
        )
        data, attrs = parse_ts(buff.read())
        data = torch.from_numpy(data)
        labels = torch.from_numpy(attrs['labels']).view(-1, 1, 1)

        meta = [Metadata(name='Data'), Metadata(name='Labels')]

        super().__init__(
            data, labels,
            return_length=return_length,
            transform=transform,
            metadata=meta,
        )
