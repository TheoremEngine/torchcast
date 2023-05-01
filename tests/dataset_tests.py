import os
import tempfile
import unittest

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


if __name__ == '__main__':
    unittest.main()
