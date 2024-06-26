'''
Contains copies of code from:

    https://github.com/cure-lab/LTSF-Linear

Used to verify that our implementations are isomorphic.
'''
from typing import List

import numpy as np
import pandas as pd


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class SecondOfMinute(TimeFeature):
    '''
    Minute of hour encoded as value between [-0.5, 0.5].
    '''
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    '''
    Minute of hour encoded as value between [-0.5, 0.5].
    '''
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    '''
    Hour of day encoded as value between [-0.5, 0.5]
    '''
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    '''
    Hour of day encoded as value between [-0.5, 0.5]
    '''
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    '''
    Day of month encoded as value between [-0.5, 0.5]
    '''
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    '''
    Day of year encoded as value between [-0.5, 0.5]
    '''
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    '''
    Month of year encoded as value between [-0.5, 0.5]
    '''
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    '''
    Week of year encoded as value between [-0.5, 0.5]
    '''
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    '''
    Returns a list of time features that will be appropriate for the given
    frequency string.

    Args:
        freq_str (str): Frequency string of the form [multiple][granularity]
        such as "12H", "5min", "1D" etc.
    '''
    features_by_offsets = {
        pd.tseries.offsets.YearEnd: [],
        pd.tseries.offsets.QuarterEnd: [MonthOfYear],
        pd.tseries.offsets.MonthEnd: [MonthOfYear],
        pd.tseries.offsets.Week: [DayOfMonth, WeekOfYear],
        pd.tseries.offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        pd.tseries.offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        pd.tseries.offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        pd.tseries.offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        pd.tseries.offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    offset = pd.tseries.frequencies.to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    return np.vstack(
        [feat(dates) for feat in time_features_from_frequency_str(freq)]
    )
