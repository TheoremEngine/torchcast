from datetime import datetime
from math import isclose, isnan
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
            self.assertEqual(ds.metadata[0].name, 'Datetime')
            self.assertEqual(ds.metadata[0].channel_names, None)
            self.assertEqual(ds.metadata[0].series_names, None)
            self.assertEqual(ds.metadata[1].name, 'Data')
            self.assertEqual(ds.metadata[1].channel_names[0], 'CO(GT)')
            self.assertEqual(ds.metadata[1].series_names, None)


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

    def test_load_tsf_file_1(self):
        path = os.path.join(os.path.dirname(__file__), 'data/example_1.tsf')
        series, attrs = tc.load_tsf_file(path)

        self.assertEqual(series.shape, (2, 4))
        self.assertEqual(len(attrs), 4)

        should_be = np.array([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.]
        ])
        self.assertTrue((series == should_be).all(), series)

        self.assertEqual(attrs.keys(), {'str', 'num', 'dat', 'horizon'})
        self.assertEqual(attrs['str'], ['a', 'b'])
        self.assertTrue(isinstance(attrs['num'], np.ndarray))
        self.assertTrue((attrs['num'] == np.array([1, 2])).all())
        self.assertEqual(
            attrs['dat'],
            [datetime(2010, 1, 1, 0, 0, 0), datetime(2010, 1, 2, 0, 0, 0)]
        )

    def test_load_tsf_file_2(self):
        path = os.path.join(os.path.dirname(__file__), 'data/example_2.tsf')
        series, attrs = tc.load_tsf_file(path)

        self.assertEqual(len(series), 2)
        self.assertEqual(len(attrs), 4)

        self.assertEqual(series[0].shape, (4,))
        self.assertTrue(isnan(series[0][0]))
        self.assertTrue((series[0][1:] == np.array([2., 3., 4.])).all())
        self.assertEqual(series[1].shape, (3,))
        self.assertTrue((series[1] == np.array([5., 6., 7.])).all())

        self.assertEqual(attrs.keys(), {'str', 'num', 'dat', 'horizon'})
        self.assertEqual(attrs['str'], ['a', 'b'])
        self.assertTrue(isinstance(attrs['num'], np.ndarray))
        self.assertTrue((attrs['num'] == np.array([1, 2])).all())
        self.assertEqual(
            attrs['dat'],
            [datetime(2010, 1, 1, 0, 0, 0), datetime(2010, 1, 2, 0, 0, 0)]
        )


if __name__ == '__main__':
    unittest.main()
