from datetime import datetime
from math import isclose, isnan
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import torchcast as tc


'''
class AirQualityTest(unittest.TestCase):
    def test_full_up(self):
        ds = tc.datasets.AirQualityDataset()

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


class MonashTests(unittest.TestCase):
    def test_bitcoin(self):
        # Bitcoin is one of the special cases, since it is multivariate but is
        # a single series.
        ds = tc.datasets.MonashArchiveDataset('Bitcoin', 'all')

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].shape, (1, 1, 4581))
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].shape, (1, 18, 4581))
        self.assertEqual(ds.data[1].dtype, torch.float32)

        self.assertTrue(isclose(ds.data[1][0, -1, -1].item(), 30593))
        delta_t = ds.data[0][0, 0, 1:] - ds.data[0][0, 0, :-1]
        self.assertTrue((delta_t == 24 * 60 * 60 * 1_000_000_000).all())

    def test_cif_2016(self):
        # CIF2016 is another special case, since it has variable horizon per
        # series.
        ds = tc.datasets.MonashArchiveDataset('CIF2016', 'test')

        self.assertEqual(len(ds.data), 1)
        self.assertEqual(ds.data[0].shape, (72, 1, 12))
        self.assertEqual(ds.data[0].dtype, torch.float32)

        self.assertEqual(ds.data[0][0, 0, 0], _to_float32(1657.01533755845))
        self.assertEqual(ds.data[0][0, 0, -1], _to_float32(1683.41802153606))
        self.assertEqual(ds.data[0][0].shape, (1, 12))
        self.assertEqual(ds.data[0][-1, 0, 5], _to_float32(13058445.73))
        self.assertEqual(ds.data[0][-1].shape, (1, 6))

    def test_univariate(self):
        ds = tc.datasets.MonashArchiveDataset('CarParts', 'train')

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].shape, (2674, 1, 51 - 12))
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].shape, (2674, 1, 51 - 12))
        self.assertEqual(ds.data[1].dtype, torch.float32)

        self.assertTrue(isclose(ds.data[1][0, 0, 0], 0))
        self.assertTrue(isnan(ds.data[1][0, 0, -1]))
        self.assertTrue(isclose(ds.data[1][-1, 0, 0], 0))
        self.assertTrue(isclose(ds.data[1][-1, 0, -1], 0))

    def test_multivariate(self):
        ds = tc.datasets.MonashArchiveDataset('Rideshare', 'all')

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].shape, (156, 1, 541))
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[1].shape, (156, 15, 541))
        self.assertEqual(ds.data[1].dtype, torch.float32)

        self.assertTrue(isclose(ds.data[1][0, 0, 0], 9.0))
        self.assertTrue(isclose(ds.data[1][0, 0, -1], 14.0))

    def test_create_time_array(self):
        start = datetime(1999, 1, 1)

        cases = {
            'yearly': datetime(2000, 1, 1),
            'quarterly': datetime(1999, 4, 1),
            'monthly': datetime(1999, 2, 1),
            'weekly': datetime(1999, 1, 8),
            'daily': datetime(1999, 1, 2),
            'hourly': datetime(1999, 1, 1, hour=1),
            'half_hourly': datetime(1999, 1, 1, minute=30),
            '10_minutes': datetime(1999, 1, 1, minute=10),
            '4_seconds': datetime(1999, 1, 1, second=4),
        }

        for freq, end in cases.items():
            t = tc.datasets.monash._create_time_array(start, freq, 2)
            self.assertTrue(isinstance(t, np.ndarray))
            self.assertEqual(t.shape, (2,))
            self.assertEqual(t.dtype, np.int64)
            self.assertEqual(t[1], pd.Timestamp(end).value)


class TFBTests(unittest.TestCase):
    def test_multivariate(self):
        ds = tc.datasets.TFBDataset('Wind')

        self.assertEqual(len(ds.data), 2)

        self.assertEqual(ds.data[0].shape, (1, 1, 48673))
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[0][0, 0, 0], pd.Timestamp(2020, 1, 1).value)
        self.assertEqual(ds.data[0][0, 0, -1], pd.Timestamp(2021, 5, 22).value)
        self.assertEqual(ds.metadata[0].name, 'Datetime')
        self.assertEqual(ds.metadata[0].channel_names, None)
        self.assertEqual(ds.metadata[0].series_names, None)

        self.assertEqual(ds.data[1].shape, (1, 7, 48673))
        self.assertEqual(ds.data[1].dtype, torch.float32)
        self.assertEqual(ds.metadata[1].name, None)
        self.assertEqual(
            set(ds.metadata[1].channel_names),
            {'pred_w_speed', 'pred_w_dir', 'pred_temp', 'pred_pressure',
             'pred_humidity', 'ture_w_speed', 'target'}
        )
        self.assertEqual(ds.metadata[1].series_names, None)

    def test_multivariate_with_nonstandard_datetime(self):
        ds = tc.datasets.TFBDataset('Solar')

        self.assertEqual(len(ds.data), 2)

        self.assertEqual(ds.data[0].shape, (1, 1, 52560))
        self.assertEqual(ds.data[0].dtype, torch.int64)
        self.assertEqual(ds.data[0][0, 0, 0], pd.Timestamp(2006, 1, 1).value)
        self.assertEqual(
            ds.data[0][0, 0, -1], pd.Timestamp(2006, 12, 31, 23, 50).value
        )
        self.assertEqual(ds.metadata[0].name, 'Datetime')
        self.assertEqual(ds.metadata[0].channel_names, None)
        self.assertEqual(ds.metadata[0].series_names, None)

        self.assertEqual(ds.data[1].shape, (1, 137, 52560))
        self.assertEqual(ds.data[1].dtype, torch.float32)
        self.assertEqual(ds.metadata[1].name, None)
        self.assertEqual(
            set(ds.metadata[1].channel_names),
            {f'channel_{c}' for c in range(1, 138)}
        )
        self.assertEqual(ds.metadata[1].series_names, None)

    def test_univariate(self):
        ds = tc.datasets.TFBDataset('human_1')

        self.assertEqual(len(ds.data), 1)

        self.assertEqual(ds.data[0].shape, (1, 1, 169))
        self.assertEqual(ds.data[0].dtype, torch.float32)
        self.assertEqual(ds.metadata[0].name, None)
        self.assertEqual(ds.metadata[0].channel_names[0], 'channel_1')
        self.assertEqual(ds.metadata[0].series_names, None)


class UCRTests(unittest.TestCase):
    def test_ucr_full_up(self):
        ds = tc.datasets.UCRDataset('Crop', 'train')

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].shape, (7200, 1, 46))
        self.assertEqual(ds.data[0].dtype, torch.float32)
        self.assertEqual(ds.data[1].shape, (7200, 1, 1))
        self.assertEqual(ds.data[1].dtype, torch.int64)

        row_0 = torch.tensor([
            0.24, 0.257, 0.274, 0.257, 0.277, 0.297, 0.317, 0.29325, 0.2695,
            0.24575, 0.222, 0.19, 0.229, 0.246, 0.28, 0.334, 0.388, 0.458,
            0.528, 0.615, 0.717, 0.727, 0.753, 0.725, 0.743, 0.761, 0.77233,
            0.78367, 0.795, 0.771, 0.749, 0.705, 0.702, 0.642, 0.5415, 0.441,
            0.41475, 0.3885, 0.36225, 0.336, 0.324, 0.312, 0.331, 0.35, 0.333,
            0.316
        ])

        self.assertTrue(torch.isclose(ds.data[0][0, 0, :], row_0).all())
        self.assertEqual(ds.data[1][0, 0, 0].item(), 0)

        self.assertTrue(isinstance(ds.metadata, list))
        self.assertEqual(len(ds.metadata), 2)
        self.assertEqual(ds.metadata[0].name, 'Data')
        self.assertEqual(ds.metadata[0].channel_names, None)
        self.assertEqual(ds.metadata[0].series_names, None)
        self.assertEqual(ds.metadata[1].name, 'Labels')
        self.assertEqual(ds.metadata[1].channel_names, None)
        self.assertEqual(ds.metadata[1].series_names, None)

    def test_uea_full_up(self):
        ds = tc.datasets.UEADataset(
            'ArticularyWordRecognition', 'train',
        )

        self.assertEqual(len(ds.data), 2)
        self.assertEqual(ds.data[0].shape, (275, 9, 144))
        self.assertEqual(ds.data[0].dtype, torch.float32)
        self.assertEqual(ds.data[1].shape, (275, 1, 1))
        self.assertEqual(ds.data[1].dtype, torch.int64)

        row_0 = torch.tensor([
            0.87159, 0.88042, 0.88042, 0.81962, 0.81962, 1.1109, 1.1109,
            0.82844, 0.82844, 1.0403, 0.86767, 0.86767, 1.0001, 1.0001, 1.0265,
            1.0265, 0.8814, 0.8814, 1.0363, 1.0089, 1.0089, 0.95789, 0.95789,
            0.9726, 0.9726, 0.94318, 0.94318, 1.0334, 0.99516, 0.99516,
            0.95005, 0.95005, 0.9628, 0.9628, 1.0216, 1.0216, 0.78627, 0.79608,
            0.79608, 0.93632, 0.93632, 1.1119, 1.1119, 0.88532, 0.88532,
            0.88336, 0.81569, 0.81569, 0.80981, 0.80981, 0.77254, 0.77254,
            0.91376, 0.91376, 0.78137, 0.72939, 0.72939, 0.90788, 0.90788,
            0.80883, 0.80883, 0.65388, 0.65388, 0.70292, 0.75783, 0.75783,
            0.69017, 0.69017, 0.74214, 0.74214, 0.55189, 0.55189, 0.69703,
            0.71468, 0.71468, 0.48619, 0.48619, 0.27828, 0.27828, 0.26259,
            0.26259, 0.16648, -0.078686, -0.078686, -0.30032, -0.30032,
            -0.28561, -0.28561, -0.43565, -0.43565, -0.74064, -0.87696,
            -0.87696, -0.94463, -0.94463, -1.129, -1.129, -1.1368, -1.1368,
            -1.2173, -1.1643, -1.1643, -1.1162, -1.1162, -1.2388, -1.2388,
            -1.5732, -1.5732, -1.5154, -1.4261, -1.4261, -1.5164, -1.5164,
            -1.8586, -1.8586, -1.5262, -1.5262, -1.2987, -1.2271, -1.2271,
            -1.3546, -1.3546, -1.4281, -1.4281, -1.4114, -1.4114, -1.2006,
            -1.2065, -1.2065, -1.2094, -1.2094, -1.2663, -1.2663, -1.0888,
            -1.0888, -0.96914, -0.98778, -0.98778, -0.99562, -0.99562,
            -0.95541, -0.95541, -0.64258, -0.64258
        ])

        self.assertTrue(torch.isclose(ds.data[0][0, 0, :], row_0).all())
        self.assertEqual(ds.data[1][0, 0, 0].item(), 0)

        self.assertTrue(isinstance(ds.metadata, list))
        self.assertEqual(len(ds.metadata), 2)
        self.assertEqual(ds.metadata[0].name, 'Data')
        self.assertEqual(ds.metadata[0].channel_names, None)
        self.assertEqual(ds.metadata[0].series_names, None)
        self.assertEqual(ds.metadata[1].name, 'Labels')
        self.assertEqual(ds.metadata[1].channel_names, None)
        self.assertEqual(ds.metadata[1].series_names, None)


class UtilsTests(unittest.TestCase):
    def test_download_and_extract(self):
        # Use air_quality for this, as it's a zip file.
        with tempfile.TemporaryDirectory() as temp_root:
            with self.assertRaises(FileNotFoundError):
                tc.datasets.utils._download_and_extract(
                    tc.datasets.air_quality.AIR_QUALITY_URL,
                    tc.datasets.air_quality.AIR_QUALITY_FILE_NAME,
                    temp_root,
                    download=False
                )
            tc.datasets.utils._download_and_extract(
                tc.datasets.air_quality.AIR_QUALITY_URL,
                tc.datasets.air_quality.AIR_QUALITY_FILE_NAME,
                temp_root,
                download=True
            )
            tc.datasets.utils._download_and_extract(
                tc.datasets.air_quality.AIR_QUALITY_URL,
                tc.datasets.air_quality.AIR_QUALITY_FILE_NAME,
                temp_root,
                download=False
            )
        # Now do it again, to make sure that the caching isn't messed up.
        tc.datasets.utils._download_and_extract(
            tc.datasets.air_quality.AIR_QUALITY_URL,
            tc.datasets.air_quality.AIR_QUALITY_FILE_NAME,
            None,
            download=True
        )

    def test_add_missing_values(self):
        df = pd.DataFrame({'a': [0, 3], 'b': [4, 5], 'c': [0, 0]})
        df = tc.datasets.utils._add_missing_values(
            df, a=[0, 3], b=[4, 5]
        )
        self.assertEqual(len(df), 4)

    def test_load_ts_file_1(self):
        path = os.path.join(os.path.dirname(__file__), 'data/example_1.ts')
        series, attrs = tc.load_ts_file(path)

        self.assertTrue(isinstance(series, list))
        self.assertEqual(len(attrs), 4)

        should_be = [
            np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]]),
            np.array([[9., 10., 11., 12.], [13., 14., 15., 16.]])
        ]
        for i in [0, 1]:
            self.assertEqual(series[i].shape, (2, 4))
            self.assertTrue((series[i] == should_be[i]).all())

        self.assertEqual(
            attrs.keys(),
            {'problemName', 'univariate', 'classLabel', 'labels'},
        )
        self.assertEqual(attrs['problemName'], 'Wowza')
        self.assertEqual(attrs['univariate'], False)
        self.assertEqual(attrs['classLabel'], ['a', 'b', 'c'])
        self.assertTrue(isinstance(attrs['labels'], np.ndarray))
        self.assertTrue((attrs['labels'] == np.array([0, 1])).all())

    def test_load_ts_file_2(self):
        path = os.path.join(os.path.dirname(__file__), 'data/example_2.ts')
        series, attrs = tc.load_ts_file(path)

        self.assertEqual(series.shape, (2, 2, 4))
        self.assertEqual(len(attrs), 4)

        self.assertTrue(isnan(series[0, 0, 0]))
        self.assertTrue(isnan(series[0, 0, 1]))
        self.assertEqual(series[0, 0, 2], 5.)
        self.assertEqual(series[0, 0, 3], 1.)
        self.assertEqual(series[0, 1, 0], 0.)
        self.assertEqual(series[0, 1, 1], 1.)
        self.assertTrue(isnan(series[0, 1, 2]))
        self.assertTrue(isnan(series[0, 1, 3]))
        self.assertTrue(isnan(series[1, 0, 0]))
        self.assertEqual(series[1, 0, 1], 2.)
        self.assertTrue(np.isnan(series[1, 0, 2:]).all())
        self.assertTrue(np.isnan(series[1, 1, :]).all())

        self.assertEqual(
            attrs.keys(),
            {'problemName', 'seriesLength', 'classLabel', 'timeStamps'},
        )
        self.assertEqual(attrs['problemName'], 'Wowza')
        self.assertEqual(attrs['classLabel'], False)
        self.assertEqual(attrs['seriesLength'], 4)

    def test_load_tsf_file_1(self):
        path = os.path.join(os.path.dirname(__file__), 'data/example_1.tsf')
        series, attrs = tc.load_tsf_file(path)

        self.assertEqual(series.shape, (2, 1, 4))
        self.assertEqual(len(attrs), 7)

        should_be = np.array([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.]
        ])
        should_be = should_be.reshape(2, 1, 4)
        self.assertTrue((series == should_be).all(), series)

        self.assertEqual(
            attrs.keys(),
            {'str', 'num', 'dat', 'horizon', 'frequency', 'missing',
             'equallength'}
        )
        self.assertEqual(attrs['horizon'], 4)
        self.assertEqual(attrs['missing'], True)
        self.assertEqual(attrs['equallength'], True)
        self.assertEqual(attrs['frequency'], '12')
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
        self.assertEqual(len(attrs), 7)

        self.assertEqual(series[0].shape, (1, 4,))
        self.assertTrue(isnan(series[0][0, 0]))
        self.assertTrue((series[0][0, 1:] == np.array([2., 3., 4.])).all())
        self.assertEqual(series[1].shape, (1, 3,))
        self.assertTrue((series[1][0, :] == np.array([5., 6., 7.])).all())

        self.assertEqual(
            attrs.keys(),
            {'str', 'num', 'dat', 'horizon', 'frequency', 'missing',
             'equallength'}
        )
        self.assertEqual(attrs['horizon'], 4)
        self.assertEqual(attrs['missing'], True)
        self.assertEqual(attrs['equallength'], False)
        self.assertEqual(attrs['frequency'], '12')
        self.assertEqual(attrs['str'], ['a', 'b'])
        self.assertTrue(isinstance(attrs['num'], np.ndarray))
        self.assertTrue((attrs['num'] == np.array([1, 2])).all())
        self.assertEqual(
            attrs['dat'],
            [datetime(2010, 1, 1, 0, 0, 0), datetime(2010, 1, 2, 0, 0, 0)]
        )


'''
class UTSDTests(unittest.TestCase):
    def test_fetch(self):
        ds = tc.datasets.UTSDDataset('baian', return_length=128)
        should_be = torch.tensor([
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.1636962890625, 1540.1636962890625, 1540.1636962890625,
            1540.1636962890625, 1540.163818359375, 1540.163818359375,
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.1636962890625, 1540.1636962890625, 1540.1636962890625,
            1540.1636962890625, 1540.1636962890625, 1540.1636962890625,
            1540.1636962890625, 1540.163818359375, 1540.163818359375,
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.1639404296875, 1540.1639404296875, 1540.1639404296875,
            1540.1639404296875, 1540.1639404296875, 1540.1639404296875,
            1540.163818359375, 1540.1639404296875, 1540.1639404296875,
            1540.1639404296875, 1540.1640625, 1540.1640625, 1540.1640625,
            1540.1640625, 1540.1639404296875, 1540.1639404296875,
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.163818359375, 1540.163818359375, 1540.163818359375,
            1540.1639404296875, 1540.1639404296875, 1540.1639404296875,
            1540.1639404296875, 1540.1639404296875, 1540.163818359375,
            1540.163818359375, 1540.1636962890625, 1540.1636962890625,
            1540.1636962890625, 1540.16357421875, 1540.16357421875,
            1540.1634521484375, 1540.1634521484375, 1540.1634521484375,
            1540.163330078125, 1540.163330078125, 1540.1634521484375,
            1540.1634521484375, 1540.1634521484375, 1540.1634521484375,
            1540.1634521484375, 1540.1634521484375, 1540.1634521484375,
            1540.1634521484375, 1540.1634521484375, 1540.16357421875,
            1540.16357421875, 1540.16357421875, 1540.16357421875,
            1540.16357421875, 1540.16357421875, 1540.16357421875,
            1540.1634521484375, 1540.1634521484375, 1540.1634521484375,
            1540.1634521484375, 1540.163330078125, 1540.163330078125,
            1540.163330078125, 1540.1632080078125, 1540.1632080078125,
            1540.1632080078125, 1540.163330078125, 1540.163330078125,
            1540.163330078125, 1540.163330078125, 1540.163330078125,
            1540.163330078125, 1540.163330078125, 1540.163330078125,
            1540.163330078125, 1540.163330078125, 1540.1634521484375,
            1540.1634521484375, 1540.1634521484375, 1540.1634521484375,
            1540.1634521484375, 1540.163330078125, 1540.163330078125,
            1540.1632080078125, 1540.1632080078125, 1540.1632080078125,
            1540.1632080078125, 1540.1632080078125, 1540.1632080078125,
            1540.1632080078125, 1540.1630859375, 1540.1630859375,
            1540.1630859375
        ])

        self.assertEqual(len(ds.data), 1)
        self.assertTrue(isinstance(ds.data[0], tc.data.ListOfTensors))
        self.assertEqual(ds.data[0].shape, (51, 18, 180142))
        self.assertEqual(ds.data[0].dtype, torch.float32)

        self.assertTrue(torch.isclose(ds[0][0, :], should_be).all())


def _to_float32(x: float) -> float:
    return np.array([x], dtype=np.float32).item()


if __name__ == '__main__':
    unittest.main()
