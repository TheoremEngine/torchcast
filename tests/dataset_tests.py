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

if __name__ == '__main__':
    unittest.main()
