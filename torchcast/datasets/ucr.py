from typing import Callable, List, Optional, Union

import torch

from ..data import Metadata, TensorSeriesDataset
from ._file_readers import parse_ts
from .utils import _decode, _download_and_extract

__all__ = ['UCRDataset', 'UEADataset']

ZENODO_URL = 'https://zenodo.org/record/{zenodo_id}/files/{name}'

UCR_DATASETS = {
    'ACSF1': 11184893, 'Adiac': 11179788, 'AllGestureWiimoteX': 11185036,
    'AllGestureWiimoteY': 11185107, 'AllGestureWiimoteZ': 11185136,
    'ArrowHead': 11185163, 'Beef': 11185190, 'BeetleFly': 11185218,
    'BirdChicken': 11185259, 'BME': 11185291, 'Car': 11185322, 'CBF': 11186181,
    'Chinatown': 11186207, 'ChlorineConcentration': 11186229,
    'CinCECGTorso': 11186247, 'Coffee': 11186266, 'Computers': 11186293,
    'CricketX': 11186304, 'CricketY': 11186320, 'CricketZ': 11186333,
    'Crop': 11186344, 'DiatomSizeReduction': 11186365,
    'DistalPhalanxOutlineAgeGroup': 11186386,
    'DistalPhalanxOutlineCorrect': 11186597, 'DistalPhalanxTW': 11186610,
    'DodgerLoopDay': 11186618, 'DodgerLoopGame': 11186628,
    'DodgerLoopWeekend': 11186647, 'Earthquakes': 11186659, 'ECG200': 11186675,
    'ECG5000': 11186692, 'ECGFiveDays': 11186702, 'ElectricDevices': 11190880,
    'EOGHorizontalSignal': 11190930, 'EOGVerticalSignal': 11190951,
    'EthanolLevel': 11190985, 'FaceAll': 11191011, 'FaceFour': 11191042,
    'FacesUCR': 11191065, 'FiftyWords': 11191097, 'Fish': 11191141,
    'FordA': 11191164, 'FordB': 11191172, 'FreezerRegularTrain': 11191184,
    'FreezerSmallTrain': 11191211, 'Fungi': 11191230,
    'GestureMidAirD1': 11197478, 'GestureMidAirD2': 11197490,
    'GestureMidAirD3': 11197504, 'GesturePebbleZ1': 11197515,
    'GesturePebbleZ2': 11197520, 'GunPoint': 11191244,
    'GunPointAgeSpan': 11194425, 'GunPointMaleVersusFemale': 11194429,
    'GunPointOldVersusYoung': 11194437, 'Ham': 11197526,
    'HandOutlines': 11197528, 'Haptics': 11197538, 'Herring': 11197540,
    'HouseTwenty': 11197555, 'InlineSkate': 11197575,
    'InsectEPGRegularTrain': 11197587, 'InsectEPGSmallTrain': 11197608,
    'InsectWingbeatSound': 11197635, 'ItalyPowerDemand': 11197656,
    'LargeKitchenAppliances': 11197689, 'Lightning2': 11197697,
    'Lightning7': 11197706, 'Mallat': 11197731, 'Meat': 11197742,
    'MedicalImages': 11197752, 'MelbournePedestrian': 11197762,
    'MiddlePhalanxOutlineAgeGroup': 11197771,
    'MiddlePhalanxOutlineCorrect': 11197782, 'MiddlePhalanxTW': 11197799,
    'MixedShapesRegularTrain': 11197803, 'MixedShapesSmallTrain': 11197811,
    'MoteStrain': 11197817, 'NonInvasiveFetalECGThorax1': 11197817,
    'NonInvasiveFetalECGThorax2': 11197831, 'OliveOil': 11197843,
    'OSULeaf': 11197848, 'PhalangesOutlinesCorrect': 11197875,
    'Phoneme': 11197891, 'PickupGestureWiimoteZ': 11197898,
    'PigAirwayPressure': 11197911, 'PigArtPressure': 11197920,
    'PigCVP': 11197924, 'PLAID': 11197936, 'Plane': 11197940,
    'PowerCons': 11197948, 'ProximalPhalanxOutlineAgeGroup': 11197960,
    'ProximalPhalanxOutlineCorrect': 11197968, 'ProximalPhalanxTW': 11197973,
    'RefrigerationDevices': 11197996, 'Rock': 11198001, 'ScreenType': 11198182,
    'SemgHandGenderCh2': 11198193, 'SemgHandMovementCh2': 11198197,
    'SemgHandSubjectCh2': 11198203, 'ShakeGestureWiimoteZ': 11198219,
    'ShapeletSim': 11198235, 'ShapesAll': 11198237,
    'SmallKitchenAppliances': 11198251, 'SmoothSubspace': 11198271,
    'SonyAIBORobotSurface1': 11198277, 'SonyAIBORobotSurface2': 11198290,
    'StarLightCurves': 11198308, 'Strawberry': 11198313,
    'SwedishLeaf': 11198315, 'Symbols': 11198322, 'SyntheticControl': 11198330,
    'ToeSegmentation1': 11198338, 'ToeSegmentation2': 11198342,
    'Trace': 11198344, 'TwoLeadECG': 11198352, 'TwoPatterns': 11198356,
    'UMD': 11198362, 'UWaveGestureLibraryAll': 11198366,
    'UWaveGestureLibraryX': 11198374, 'UWaveGestureLibraryY': 11198382,
    'UWaveGestureLibraryZ': 11198384, 'Wafer': 11198387, 'Wine': 11198391,
    'WordSynonyms': 11198396, 'Worms': 11198402, 'WormsTwoClass': 11198406,
    'Yoga': 11198408,
}

UEA_DATASETS = {
    'ArticularyWordRecognition': 11204924, 'AtrialFibrillation': 11206175,
    'BasicMotions': 11206179, 'CharacterTrajectories': 11206183,
    'Cricket': 11206185, 'DuckDuckGeese': 11206189, 'EigenWorms': 11206196,
    'Epilepsy': 11206204, 'EthanolConcentration': 11206212, 'ERing': 11206210,
    'FaceDetection': 11206216, 'FingerMovements': 11206220,
    'HandMovementDirection': 11206224, 'Handwriting': 11206227,
    'Heartbeat': 11206229, 'InsectWingbeat': 11206234,
    'JapaneseVowels': 11206237, 'Libras': 11206239, 'LSST': 11206243,
    'MotorImagery': 11206246, 'NATOPS': 11206248, 'PenDigits': 11206259,
    'PEMS-SF': 11206252, 'PhonemeSpectra': 11206261, 'RacketSports': 11206263,
    'SelfRegulationSCP1': 11206265, 'SelfRegulationSCP2': 11206269,
    'SpokenArabicDigits': 11206274, 'StandWalkJump': 11206278,
    'UWaveGestureLibrary': 11206282,
}


class UCRDataset(TensorSeriesDataset):
    '''
    This is the UCR dataset for univariate time series classification, found
    at:

        https://www.timeseriesclassification.com/

    '''
    tasks: List[str] = list(UCR_DATASETS)

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
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if task not in UCR_DATASETS.keys():
            raise ValueError(
                f'Did not recognize {task}; choose from '
                f'{tuple(UCR_DATASETS.keys())}'
            )
        if split not in {'train', 'test'}:
            raise ValueError(f"Split should be 'train' or 'test', got {split}")

        file_name = f'{task}_{split.upper()}.ts'
        buff = _download_and_extract(
            ZENODO_URL.format(zenodo_id=UCR_DATASETS[task], name=file_name),
            file_name,
            path,
            download=download,
        )
        buff = _decode(buff)
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
    tasks: List[str] = list(UEA_DATASETS)

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
                sequence to return. If not provided, returns an entire
                sequence.
        '''
        if task not in UEA_DATASETS.keys():
            raise ValueError(
                f'Did not recognize {task}; choose from '
                f'{tuple(UEA_DATASETS.keys())}'
            )
        if split not in {'train', 'test'}:
            raise ValueError(f"Split should be 'train' or 'test', got {split}")

        file_name = f'{task}_{split.upper()}.ts'
        buff = _download_and_extract(
            ZENODO_URL.format(zenodo_id=UEA_DATASETS[task], name=file_name),
            file_name,
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
