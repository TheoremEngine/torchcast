import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import torch
import torchcast as tc

from ltsf.custom import Dataset_Custom
from ltsf.ett import Dataset_ETT_hour, Dataset_ETT_minute

LTSF_DATA_ROOT = os.path.abspath(os.path.join(__file__, '../ltsf/data/'))

# The purpose of these tests is to verify that our dataset objects return the
# same values as the corresponding datasets from the original github repo:
#
#     https://github.com/cure-lab/LTSF-Linear
#
# We have copied some of the code into the subdirectory ltsf to assist.
#
# All LTSF datasets return a 4-tuple consisting of x, y, the x datetime, and
# the y datetime. If we retrieve index t, then x will consist of both
# predictors and targets for times [t, t + 336). The y will consist of both
# predictors and targets for times [t + 288, t + 432). Both are returned as
# numpy.ndarrays, not torch.Tensors, and are returned in TC arrangement, not
# CT. The datetimes are returned as T4 numpy.ndarrays, corresponding to hour,
# day of week, day of month, day of year. Note they do not encode the year.


class ElectricityLoadDataset(unittest.TestCase):
    def test_full_up(self):
        ds = tc.datasets.ElectricityLoadDataset(
            download=True, return_length=432, split='train',
        )

        ltsf_ds = Dataset_Custom(
            LTSF_DATA_ROOT,
            # size=(seq_len, label_len, pred_len).
            size=[336, 48, 96],
            # 'M': multivariate predicting multivariate
            features='M',
            target='OT',
            # timeenc=0 does not work.
            timeenc=1,
            # 'h': Hourly
            freq='h',
            flag='train',
            # This is a change from the defaults. We do our scaling using a
            # transform, so we want to verify that our values are
            # consistent without the scaling.
            scale=False,
            data_path='electricity.csv',
        )

        self.assertEqual(len(ds), len(ltsf_ds))

        for t in [0, 1, len(ds) - 1]:
            my_series = ds[t]
            x, y, _, _ = ltsf_ds[t]
            # Check values
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            ltsf_series = torch.cat((x, y[48:, :]), dim=0).T.float()
            self.assertTrue(torch.isclose(my_series, ltsf_series).all())

        self.assertEqual(len(ds.data), 1)
        self.assertEqual(ds.data[0].dtype, torch.float32)


class ETTDataset(unittest.TestCase):
    def test_full_up_hourly(self):
        if not os.path.exists(LTSF_DATA_ROOT):
            warnings.warn(
                'Cannot run ETTDataset.test_full_up_hourly; please download '
                'dataset from https://github.com/cure-lab/LTSF-Linear and '
                'place in tests/ltsf/data/'
            )
            return

        # The purpose of this test is to verify that our
        # ElectricityTransformerDataset returns the same values as the
        # Dataset_ETT_hour from the original github repo:
        #
        #     https://github.com/cure-lab/LTSF-Linear
        #
        # We have copied some of the code into the subdirectory ltsf to assist.

        for i in [1, 2]:
            ds = tc.datasets.ElectricityTransformerDataset(
                task=f'hourly-{i}', download=True, split='train',
                return_length=432
            )

            # Parameters are copied from argparser in run_longExp.py and
            # script in scripts. pred_len varies from 96 to 729.
            ltsf_ds = Dataset_ETT_hour(
                LTSF_DATA_ROOT,
                # size=(seq_len, label_len, pred_len).
                size=[336, 48, 96],
                # 'M': multivariate predicting multivariate
                features='M',
                target='OT',
                # timeenc=0 does not work.
                timeenc=1,
                # 'h': Hourly
                freq='h',
                flag='train',
                # This is a change from the defaults. We do our scaling using a
                # transform, so we want to verify that our values are
                # consistent without the scaling.
                scale=False,
                data_path=f'ETTh{i}.csv',
            )

            self.assertEqual(len(ds), len(ltsf_ds))

            for t in [0, 1, len(ds) - 1]:
                my_dates, my_predictor, my_target = ds[t]
                my_series = torch.cat((my_predictor, my_target), dim=0)
                x, y, x_date, y_date = ltsf_ds[t]
                # Check values
                x, y = torch.from_numpy(x), torch.from_numpy(y)
                ltsf_series = torch.cat((x, y[48:, :]), dim=0).T.float()
                self.assertTrue(torch.isclose(my_series, ltsf_series).all())
                # Check dates. Note we cannot check year.
                ltsf_dates = np.concatenate((x_date, y_date[48:, :]), axis=0)
                ltsf_dates = [
                    [round(365 * (d[3] + 0.5) + 1), round(23 * (d[0] + 0.5))]
                    for d in ltsf_dates
                ]
                my_dates = pd.to_datetime(my_dates.squeeze().numpy())
                my_dates = np.stack(
                    (my_dates.day_of_year.values, my_dates.hour.values),
                    axis=1
                )
                my_dates = my_dates.tolist()
                self.assertEqual(ltsf_dates, my_dates)

        self.assertEqual(len(ds.data), 3)
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].dtype, torch.float32)
        self.assertEqual(ds.data[2].dtype, torch.float32)

        # Check metadata
        self.assertTrue(isinstance(ds.metadata, list))
        self.assertEqual(len(ds.metadata), 3)
        self.assertEqual(ds.metadata[0].name, 'Datetime')
        self.assertEqual(ds.metadata[0].channel_names, None)
        self.assertEqual(ds.metadata[0].series_names, None)
        self.assertEqual(ds.metadata[1].name, 'Predictors')
        self.assertEqual(ds.metadata[1].channel_names[0], 'High Useful Load')
        self.assertEqual(ds.metadata[1].series_names, None)
        self.assertEqual(ds.metadata[2].name, 'Target')
        self.assertEqual(ds.metadata[2].channel_names[0], 'Oil Temperature')
        self.assertEqual(ds.metadata[2].series_names, None)

    def test_full_up_15min(self):
        if not os.path.exists(LTSF_DATA_ROOT):
            warnings.warn(
                'Cannot run ETTDataset.test_full_up_15min; please download '
                'dataset from https://github.com/cure-lab/LTSF-Linear and '
                'place in tests/ltsf/data/'
            )
            return

        for i in [1, 2]:
            ds = tc.datasets.ElectricityTransformerDataset(
                task=f'15min-{i}', download=True,
                split='train', return_length=432,
            )

            # Parameters are copied from argparser in run_longExp.py and
            # script in scripts. pred_len varies from 96 to 729.
            ltsf_ds = Dataset_ETT_minute(
                LTSF_DATA_ROOT,
                # size=(seq_len, label_len, pred_len).
                size=[336, 48, 96],
                # 'M': multivariate predicting multivariate
                features='M',
                target='OT',
                # timeenc=0 does not work.
                timeenc=1,
                # 'h': Hourly
                freq='h',
                flag='train',
                # This is a change from the defaults. We do our scaling using a
                # transform, so we want to verify that our values are
                # consistent without the scaling.
                scale=False,
                data_path=f'ETTm{i}.csv',
            )

            self.assertEqual(len(ds), len(ltsf_ds))

            for t in [0, 1, len(ds) - 1]:
                my_dates, my_predictor, my_target = ds[t]
                my_series = torch.cat((my_predictor, my_target), dim=0)
                x, y, x_date, y_date = ltsf_ds[t]
                # Check values
                x, y = torch.from_numpy(x), torch.from_numpy(y)
                ltsf_series = torch.cat((x, y[48:, :]), dim=0).T.float()
                self.assertTrue(torch.isclose(my_series, ltsf_series).all())
                # Check dates. Note we cannot check year.
                ltsf_dates = np.concatenate((x_date, y_date[48:, :]), axis=0)
                ltsf_dates = [
                    [round(365 * (d[3] + 0.5) + 1), round(23 * (d[0] + 0.5))]
                    for d in ltsf_dates
                ]
                my_dates = pd.to_datetime(my_dates.squeeze().numpy())
                my_dates = np.stack(
                    (my_dates.day_of_year.values, my_dates.hour.values),
                    axis=1
                )
                my_dates = my_dates.tolist()
                self.assertEqual(ltsf_dates, my_dates)

        # Check metadata
        self.assertTrue(isinstance(ds.metadata, list))
        self.assertEqual(len(ds.metadata), 3)
        self.assertEqual(ds.metadata[0].name, 'Datetime')
        self.assertEqual(ds.metadata[0].channel_names, None)
        self.assertEqual(ds.metadata[0].series_names, None)
        self.assertEqual(ds.metadata[1].name, 'Predictors')
        self.assertEqual(ds.metadata[1].channel_names[0], 'High Useful Load')
        self.assertEqual(ds.metadata[1].series_names, None)
        self.assertEqual(ds.metadata[2].name, 'Target')
        self.assertEqual(ds.metadata[2].channel_names[0], 'Oil Temperature')
        self.assertEqual(ds.metadata[2].series_names, None)


class ExchangeRateDataset(unittest.TestCase):
    def test_full_up(self):
        if not os.path.exists(LTSF_DATA_ROOT):
            warnings.warn(
                'Cannot run ExchangeRateDataset.test_full_up; please download '
                'dataset from https://github.com/cure-lab/LTSF-Linear and '
                'place in tests/ltsf/data/'
            )
            return

        ds = tc.datasets.ExchangeRateDataset(
            download=True, split='train', return_length=432,
        )

        ltsf_ds = Dataset_Custom(
            LTSF_DATA_ROOT,
            # size=(seq_len, label_len, pred_len).
            size=[336, 48, 96],
            # 'M': multivariate predicting multivariate
            features='M',
            target='OT',
            # timeenc=0 does not work.
            timeenc=1,
            # 'h': Hourly
            freq='h',
            flag='train',
            # This is a change from the defaults. We do our scaling using a
            # transform, so we want to verify that our values are
            # consistent without the scaling.
            scale=False,
            data_path='exchange_rate.csv',
        )

        self.assertEqual(len(ds), len(ltsf_ds))

        # In this case, the dates in Dataset_Custom are dummies.

        for t in [0, 1, len(ds) - 1]:
            my_series = ds[t]
            x, y, _, _ = ltsf_ds[t]
            # Check values
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            ltsf_series = torch.cat((x, y[48:, :]), dim=0).T.float()
            self.assertTrue(torch.isclose(my_series, ltsf_series).all())

        self.assertEqual(len(ds.data), 1)
        self.assertEqual(ds.data[0].dtype, torch.float32)


class GermanWeatherDataset(unittest.TestCase):
    def test_full_up(self):
        if not os.path.exists(LTSF_DATA_ROOT):
            warnings.warn(
                'Cannot run GermanWeatherDataset.test_full_up; please download'
                ' dataset from https://github.com/cure-lab/LTSF-Linear and '
                'place in tests/ltsf/data/'
            )
            return

        ds = tc.datasets.GermanWeatherDataset(
            download=True, split='train', return_length=432,
        )

        ltsf_ds = Dataset_Custom(
            LTSF_DATA_ROOT,
            # size=(seq_len, label_len, pred_len).
            size=[336, 48, 96],
            # 'M': multivariate predicting multivariate
            features='M',
            target='OT',
            # timeenc=0 does not work.
            timeenc=1,
            # 'h': Hourly
            freq='h',
            flag='train',
            # This is a change from the defaults. We do our scaling using a
            # transform, so we want to verify that our values are
            # consistent without the scaling.
            scale=False,
            data_path='weather.csv',
        )

        self.assertEqual(len(ds), len(ltsf_ds))

        for t in [0, 1, len(ds) - 1]:
            my_dates, my_series = ds[t]
            x, y, x_date, y_date = ltsf_ds[t]
            # Check values
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            ltsf_series = torch.cat((x, y[48:, :]), dim=0).T.float()
            self.assertTrue(torch.isclose(my_series, ltsf_series).all())
            # Check dates. Note we cannot check year.
            ltsf_dates = np.concatenate((x_date, y_date[48:, :]), axis=0)
            ltsf_dates = [
                [round(365 * (d[3] + 0.5) + 1), round(23 * (d[0] + 0.5))]
                for d in ltsf_dates
            ]
            my_dates = pd.to_datetime(my_dates.squeeze().numpy())
            my_dates = np.stack(
                (my_dates.day_of_year.values, my_dates.hour.values),
                axis=1
            )
            my_dates = my_dates.tolist()
            self.assertEqual(ltsf_dates, my_dates)

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].dtype, torch.float32)

        # Check metadata
        self.assertTrue(isinstance(ds.metadata, list))
        self.assertEqual(len(ds.metadata), 2)
        self.assertEqual(ds.metadata[0].name, 'Datetime')
        self.assertEqual(ds.metadata[0].channel_names, None)
        self.assertEqual(ds.metadata[0].series_names, None)
        self.assertEqual(ds.metadata[1].name, 'Data')
        self.assertEqual(ds.metadata[1].channel_names[0], 'p (mbar)')
        self.assertEqual(ds.metadata[1].series_names, None)


"""
class ILIDataset(unittest.TestCase):
    def test_full_up(self):
        if not os.path.exists(LTSF_DATA_ROOT):
            warnings.warn(
                'Cannot run ILIDataset.test_full_up; please download dataset '
                'from https://github.com/cure-lab/LTSF-Linear and place in '
                'tests/ltsf/data/'
            )
            return

        if not os.path.exists(os.getenv('ILI_PATH', '')):
            warnings.warn(
                "Cannot run ILIDataset.test_full_up; please download dataset "
                "from https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html"
                " and set the environment variable 'ILI_PATH' to point to it"
            )
            return

        ds = tc.datasets.ILIDataset(
            os.environ['ILI_PATH'], split='train', return_length=432
        )

        ltsf_ds = Dataset_Custom(
            LTSF_DATA_ROOT,
            # size=(seq_len, label_len, pred_len).
            size=[336, 48, 96],
            # 'M': multivariate predicting multivariate
            features='M',
            target='OT',
            # timeenc=0 does not work.
            timeenc=1,
            # 'h': Hourly
            freq='h',
            flag='train',
            # This is a change from the defaults. We do our scaling using a
            # transform, so we want to verify that our values are
            # consistent without the scaling.
            scale=False,
            data_path='traffic.csv',
        )

        self.assertEqual(len(ds), len(ltsf_ds))

        # We do not store the datetimes here.

        for t in [0, 1, len(ds) - 1]:
            my_series = ds[t]
            x, y, _, _ = ltsf_ds[t]
            # Check values
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            ltsf_series = torch.cat((x, y[48:, :]), dim=0).T.float()
            self.assertTrue(torch.isclose(my_series, ltsf_series).all())

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].dtype, torch.float32)
"""


class SanFranciscoTrafficDataset(unittest.TestCase):
    def test_full_up(self):
        if not os.path.exists(LTSF_DATA_ROOT):
            warnings.warn(
                'Cannot run SanFranciscoTrafficDataset.test_full_up; please '
                'download dataset from https://github.com/cure-lab/LTSF-Linear'
                ' and place in tests/ltsf/data/'
            )
            return

        ds = tc.datasets.SanFranciscoTrafficDataset(
            download=True, split='train', return_length=432
        )

        ltsf_ds = Dataset_Custom(
            LTSF_DATA_ROOT,
            # size=(seq_len, label_len, pred_len).
            size=[336, 48, 96],
            # 'M': multivariate predicting multivariate
            features='M',
            target='OT',
            # timeenc=0 does not work.
            timeenc=1,
            # 'h': Hourly
            freq='h',
            flag='train',
            # This is a change from the defaults. We do our scaling using a
            # transform, so we want to verify that our values are
            # consistent without the scaling.
            scale=False,
            data_path='traffic.csv',
        )

        self.assertEqual(len(ds), len(ltsf_ds))

        # We do not store the datetimes here.

        for t in [0, 1, len(ds) - 1]:
            my_series = ds[t]
            x, y, _, _ = ltsf_ds[t]
            # Check values
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            ltsf_series = torch.cat((x, y[48:, :]), dim=0).T.float()
            self.assertTrue(torch.isclose(my_series, ltsf_series).all())

        self.assertEqual(len(ds.data), 1)
        self.assertEqual(ds.data[0].dtype, torch.float32)


if __name__ == '__main__':
    unittest.main()
