from datetime import datetime
from math import isclose
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torchcast as tc


class AirQualityTest(unittest.TestCase):
    def test_full_up(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with self.assertRaises(FileNotFoundError):
                ds = tc.datasets.AirQualityDataset(temp_root, download=False)
            ds = tc.datasets.AirQualityDataset(temp_root, download=True)
            ds = tc.datasets.AirQualityDataset(temp_root, download=False)

            self.assertEqual(len(ds.data), 2)
            self.assertEqual(ds.data[0].shape, (1, 1, 9357))
            self.assertEqual(ds.data[0].dtype, torch.int64)
            self.assertEqual(ds.data[1].shape, (1, 13, 9357))
            self.assertEqual(ds.data[1].dtype, torch.float32)

            row_0 = torch.tensor([
                2.6, 1360, 150, 11.9, 1046, 166, 1056, 113, 1692, 1268, 13.6,
                48.9, 0.7578
            ])
            self.assertTrue((ds.data[1][0, :, 0] == row_0).all())
            row_last = torch.tensor([
                2.2, 1071, float('nan'), 11.9, 1047, 265, 654, 168, 1129, 816,
                28.5, 13.1, 0.5028
            ])
            self.assertTrue(torch.isnan(ds.data[1][0, 2, -1]).item())
            i_real = [i for i in range(13) if i != 2]
            self.assertTrue(
                (ds.data[1][0, :, -1][i_real] == row_last[i_real]).all(),
            )

            self.assertTrue(isinstance(ds.metadata, list))
            self.assertEqual(len(ds.metadata), 2)
            self.assertEqual(ds.metadata[0], None)
            self.assertEqual(ds.metadata[1].channel_names[0], 'CO(GT)')
            self.assertEqual(ds.metadata[1].series_names, None)


class ElectricityLoadDataset(unittest.TestCase):
    def test_full_up(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with self.assertRaises(FileNotFoundError):
                ds = tc.datasets.ElectricityLoadDataset(
                    temp_root, download=False
                )
            ds = tc.datasets.ElectricityLoadDataset(
                temp_root, download=True
            )

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].shape, (1, 1, 140256))
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].shape, (1, 370, 140256))
        self.assertEqual(ds.data[1].dtype, torch.float32)

        self.assertTrue((ds.data[1][0, :123, 0] == 0).all())
        self.assertTrue(isclose(ds.data[1][0, 123, 0].item(), 71.77033233643))


class ETTDataset(unittest.TestCase):
    def test_full_up(self):
        with tempfile.TemporaryDirectory() as temp_root:
            ds = tc.datasets.ElectricityTransformerDataset(
                temp_root, task='hourly', download=True
            )
            for i in [1, 2]:
                path = os.path.join(temp_root, f'ETTh{i}.csv')
                self.assertTrue(os.path.exists(path))
        ds = tc.datasets.ElectricityTransformerDataset(
            temp_root, task='15min', download=False
        )

        self.assertEqual(len(ds.data), 3)
        self.assertEqual(ds.data[0].shape, (1, 1, 69680))
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].shape, (2, 6, 69680))
        self.assertEqual(ds.data[1].dtype, torch.float32)
        self.assertEqual(ds.data[2].shape, (2, 1, 69680))
        self.assertEqual(ds.data[2].dtype, torch.float32)

        row_0 = torch.tensor([
            5.827000141143799, 2.009000062942505, 1.5989999771118164,
            0.4620000123977661, 4.203000068664552, 1.3400000333786009
        ])
        self.assertTrue((ds.data[1][0, :, 0] == row_0).all())

        self.assertEqual(ds.data[2][0, 0, 0].item(), 30.5310001373291)

        self.assertTrue(isinstance(ds.metadata, list))
        self.assertEqual(len(ds.metadata), 3)
        self.assertEqual(ds.metadata[0], None)
        self.assertEqual(ds.metadata[1].channel_names[0], 'High Useful Load')
        self.assertEqual(ds.metadata[1].series_names, None)
        self.assertEqual(ds.metadata[2].channel_names[0], 'Oil Temperature')
        self.assertEqual(ds.metadata[2].series_names, None)


class ExchangeRateDataset(unittest.TestCase):
    def test_full_up(self):
        with tempfile.TemporaryDirectory() as temp_root:
            ds = tc.datasets.ExchangeRateDataset(temp_root, download=True)

        self.assertEqual(len(ds.data), 1)
        self.assertEqual(ds.data[0].shape, (1, 8, 7587))
        self.assertEqual(ds.data[0].dtype, torch.float32)

        row = np.array([
            0.7817999720573425, 1.6100000143051147, 0.8611040115356445,
            0.6335129737854004, 0.21124200522899628, 0.006862999871373177,
            0.593999981880188, 0.5239719748497009
        ])

        for c in range(8):
            self.assertTrue(isclose(ds.data[0][0, c, 0], row[c]))


class GermanWeatherDataset(unittest.TestCase):
    def test_full_up(self):
        with tempfile.TemporaryDirectory() as temp_root:
            ds = tc.datasets.GermanWeatherDataset(temp_root, download=True)
            for part in ['a', 'b']:
                path = os.path.join(temp_root, f'mpi_roof_2020{part}.csv')
                self.assertTrue(os.path.exists(path))

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].shape, (1, 1, 52696))
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].shape, (1, 21, 52696))
        self.assertEqual(ds.data[1].dtype, torch.float32)

        self.assertEqual(ds.data[0][0, 0, 0], 1577837400)

        self.assertTrue(abs(ds.data[1][0, 0, 0].item() - 1008.89) < 0.01)
        self.assertTrue(abs(ds.data[1][0, 1, 0].item() - 0.71) < 0.01)
        self.assertTrue(abs(ds.data[1][0, 2, 0].item() - 273.18) < 0.01)

        self.assertTrue(isinstance(ds.metadata, list))
        self.assertEqual(len(ds.metadata), 2)
        self.assertEqual(ds.metadata[0], None)
        self.assertEqual(ds.metadata[1].channel_names[0], 'p (mbar)')
        self.assertEqual(ds.metadata[1].series_names, None)


class SanFranciscoTrafficDataset(unittest.TestCase):
    def test_full_up(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with self.assertRaises(FileNotFoundError):
                ds = tc.datasets.SanFranciscoTrafficDataset(
                    temp_root, download=False
                )
            ds = tc.datasets.SanFranciscoTrafficDataset(
                temp_root, download=True
            )

        self.assertEqual(len(ds.data), 1)
        self.assertEqual(ds.data[0].shape, (1, 862, 17544))
        self.assertEqual(ds.data[0].dtype, torch.float32)

        self.assertTrue(isclose(ds.data[0][0, 0, 0].item(), 0.00480000022799))
        self.assertTrue(isclose(ds.data[0][0, 1, 0].item(), 0.01460000034422))
        self.assertTrue(isclose(ds.data[0][0, 2, 0].item(), 0.02889999933541))


class UtilsTests(unittest.TestCase):
    def test_stack_mismatched_tensors(self):
        tensors = [
            torch.tensor([0., 3., 6.]).reshape(1, 3),
            torch.tensor([0., 2.]).reshape(1, 2),
            torch.tensor([[1., 1.], [1., 1.]])
        ]
        tensor = tc.datasets.utils._stack_mismatched_tensors(tensors)

        self.assertEqual(tensor.shape, (3, 2, 3))
        self.assertTrue((tensor[0, 0, :] == tensors[0]).all())
        self.assertTrue(torch.isnan(tensor[0, 1, :]).all())
        self.assertTrue((tensor[1, 0, :2] == tensors[1]).all())
        self.assertTrue(torch.isnan(tensor[1, 0, 2]))
        self.assertTrue(torch.isnan(tensor[1, 1, :]).all())
        self.assertTrue((tensor[2, :, :2] == 1).all())
        self.assertTrue(torch.isnan(tensor[2, :, 2]).all())

    def test_add_missing_values(self):
        df = pd.DataFrame({'a': [0, 3], 'b': [4, 5], 'c': [0, 0]})
        df = tc.datasets.utils._add_missing_values(
            df, a=[0, 3], b=[4, 5]
        )
        self.assertEqual(len(df), 4)

    def test_load_tsf_file(self):
        path = os.path.join(os.path.dirname(__file__), 'data/example.tsf')
        series, attrs = tc.load_tsf_file(path)

        self.assertEqual(series.shape, (2, 4))
        self.assertEqual(len(attrs), 3)

        should_be = torch.tensor([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.]
        ])
        self.assertTrue((series == should_be).all(), series)

        self.assertEqual(attrs.keys(), {'str', 'num', 'dat'})
        self.assertEqual(attrs['str'], ['a', 'b'])
        self.assertEqual(attrs['num'], [1, 2])
        self.assertEqual(
            attrs['dat'],
            [datetime(2010, 1, 1, 0, 0, 0), datetime(2010, 1, 2, 0, 0, 0)]
        )


if __name__ == '__main__':
    unittest.main()
