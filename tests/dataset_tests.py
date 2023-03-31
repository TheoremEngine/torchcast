import csv
from math import isnan
import os
import tempfile
import unittest

import torch
import torchcast as tc


class AirQualityTest(unittest.TestCase):
    def test_full_up(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with self.assertRaises(FileNotFoundError):
                ds = tc.datasets.AirQualityDataset(temp_root, download=False)
            ds = tc.datasets.AirQualityDataset(temp_root, download=True)
            ds = tc.datasets.AirQualityDataset(temp_root, download=False)

            self.assertEqual(len(ds.series), 1)

            self.assertEqual(ds.series[0].shape, (1, 14, 9357))
            row_0 = torch.tensor([
                2.6, 1360, 150, 11.9, 1046, 166, 1056, 113, 1692, 1268, 13.6,
                48.9, 0.7578
            ])
            self.assertTrue((ds.series[0][0, 1:, 0] == row_0).all())
            row_last = torch.tensor([
                2.2, 1071, float('nan'), 11.9, 1047, 265, 654, 168, 1129, 816,
                28.5, 13.1, 0.5028
            ])
            self.assertTrue(torch.isnan(ds.series[0][0, 3, -1]).item())
            i_real = [i for i in range(13) if i != 2]
            self.assertTrue(
                (ds.series[0][0, 1:, -1][i_real] == row_last[i_real]).all()
            )

            self.assertEqual(ds.channel_names[0], 'CO(GT)')
            self.assertEqual(ds.series_names, None)


class NEONAquaticTest(unittest.TestCase):
    def test_full_up(self):
        with tempfile.TemporaryDirectory() as temp_root:
            with self.assertRaises(FileNotFoundError):
                ds = tc.datasets.NEONAquaticDataset(temp_root, download=False)
            ds = tc.datasets.NEONAquaticDataset(temp_root, download=True)
            ds = tc.datasets.NEONAquaticDataset(temp_root, download=False)

            self.assertEqual(len(ds.series), 2)

            self.assertEqual(ds.series[0].shape[:2], (34, 1))
            dates = ds.series[0]
            self.assertTrue(
                torch.isclose(dates[..., 1:8] + 1, dates[..., 2:9]).all()
            )

            self.assertEqual(ds.series[1].shape[:2], (34, 3))
            self.assertTrue(isnan(ds.series[1][0, 0, 0]))
            self.assertTrue(isnan(ds.series[1][0, 1, 0]))
            self.assertTrue(abs(ds.series[1][0, 2, 0] - 3.40215339) < 1e-5)

            self.assertEqual(
                ds.channel_names, ['temperature', 'chla', 'oxygen']
            )
            self.assertEqual(len(ds.series_names), 34)
            self.assertEqual(ds.series_names[0], 'ARIK')


class UtilsTests(unittest.TestCase):
    def test_load_csv_file(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_path = os.path.join(temp_root, 'test.csv')
            with open(temp_path, 'w') as test_file:
                writer = csv.writer(test_file)
                writer.writerow(['A', 'B', 'C'])
                writer.writerow([1, 2, 3])
                writer.writerow([8, 9, 10])

            records = tc.datasets.utils._load_csv_file(temp_path)

            self.assertEqual(records.keys(), {'A', 'B', 'C'})
            self.assertEqual(records['A'], ('1', '8'))
            self.assertEqual(records['B'], ('2', '9'))
            self.assertEqual(records['C'], ('3', '10'))

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


if __name__ == '__main__':
    unittest.main()
