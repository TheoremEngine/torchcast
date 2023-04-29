from contextlib import contextmanager
import os
import tempfile
from typing import Dict, Sequence
import unittest

import h5py
import numpy as np
import torch
import torchcast as tc


class SeriesTest(unittest.TestCase):
    def test_coerce_inputs(self):
        tc.data.SeriesDataset._coerce_inputs(
            torch.empty((2, 3, 5)),
            torch.empty((2, 7, 1)),
            torch.empty((1, 9, 5))
        )
        with self.assertRaises(ValueError):
            tc.data.SeriesDataset._coerce_inputs(
                torch.empty((3, 3, 5)),
                torch.empty((2, 7, 1)),
                torch.empty((1, 9, 5))
            )
        with self.assertRaises(ValueError):
            tc.data.SeriesDataset._coerce_inputs(
                torch.empty((2, 3, 7)),
                torch.empty((2, 7, 1)),
                torch.empty((1, 9, 5))
            )

    def test_find_i_t(self):
        ds = tc.data.SeriesDataset(return_length=4)
        ds.data = [tc.data.utils.ListOfSeries([
            torch.empty((3, 6)),
            torch.empty((3, 4)),
            torch.empty((3, 5)),
        ])]
        self.assertEqual(ds._find_i_t(0), (0, 0))
        self.assertEqual(ds._find_i_t(1), (0, 1))
        self.assertEqual(ds._find_i_t(2), (0, 2))
        self.assertEqual(ds._find_i_t(3), (1, 0))
        self.assertEqual(ds._find_i_t(4), (2, 0))
        self.assertEqual(ds._find_i_t(5), (2, 1))
        with self.assertRaises(IndexError):
            ds._find_i_t(6)

    def test_time_ranges(self):
        ds = tc.data.SeriesDataset()
        ds.data = [
            tc.data.utils.ListOfSeries([
                torch.empty((2, 4)), torch.empty((2, 5))
            ]),
            torch.empty((2, 3, 1))
        ]
        self.assertEqual(ds._time_ranges, [4, 5])
        ds.data = [
            torch.empty((2, 2, 5)),
            torch.empty((2, 3, 1))
        ]
        self.assertEqual(ds._time_ranges, [5, 5])

    def test_shape(self):
        ds = tc.data.SeriesDataset()
        ds.data = [
            tc.data.utils.ListOfSeries([
                torch.empty((2, 4)), torch.empty((2, 5))
            ]),
            torch.empty((2, 3, 1))
        ]
        self.assertEqual(ds.shape, (2, -1, 5))
        ds.data = [
            torch.empty((2, 2, 5)),
            torch.empty((2, 3, 1))
        ]
        self.assertEqual(ds.shape, (2, -1, 5))


class TensorSeriesTest(unittest.TestCase):
    def test_full_up_with_return_length(self):
        data = {
            'a': np.random.normal(0, 1, (2, 5, 3)),
            'c': np.random.normal(0, 1, (2, 4, 1)),
        }
        ds = tc.data.TensorSeriesDataset(
            data['a'], data['c'], return_length=2
        )
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds._time_ranges, [3, 3])
        self.assertEqual(ds.shape, (2, -1, 3))

        a, c = ds[0]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue((a.numpy() == data['a'][0, :, :2]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][0, :, :]).all())

        a, c = ds[1]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue((a.numpy() == data['a'][0, :, 1:3]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][0, :, :]).all())

        a, c = ds[2]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue((a.numpy() == data['a'][1, :, :2]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][1, :, :]).all())

        a, c = ds[3]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue((a.numpy() == data['a'][1, :, 1:3]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][1, :, :]).all())

        with self.assertRaises(IndexError):
            ds[4]

    def test_full_up_with_return_length_with_list_of_series(self):
        data = {
            'a': [np.random.normal(0, 1, (5, 3)),
                  np.random.normal(0, 1, (5, 2))],
            'c': np.random.normal(0, 1, (2, 4, 1)),
        }
        ds = tc.data.TensorSeriesDataset(
            data['a'], data['c'], return_length=2
        )
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds._time_ranges, [3, 2])
        self.assertEqual(ds.shape, (2, -1, 3))

        a, c = ds[0]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue((a.numpy() == data['a'][0][:, :2]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][0, :, :]).all())

        a, c = ds[1]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue((a.numpy() == data['a'][0][:, 1:3]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][0, :, :]).all())

        a, c = ds[2]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue((a.numpy() == data['a'][1][:, :2]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][1, :, :]).all())

        with self.assertRaises(IndexError):
            ds[3]

    def test_full_up_without_return_length(self):
        data = {
            'a': np.random.normal(0, 1, (2, 5, 3)),
            'c': np.random.normal(0, 1, (2, 4, 1)),
        }
        ds = tc.data.TensorSeriesDataset(data['a'], data['c'])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds._time_ranges, [3, 3])
        self.assertEqual(ds.shape, (2, -1, 3))

        a, c = ds[0]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 3))
        self.assertTrue((a.numpy() == data['a'][0, :, :]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][0, :, :]).all())

        a, c = ds[1]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 3))
        self.assertTrue((a.numpy() == data['a'][1, :, :]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][1, :, :]).all())

        with self.assertRaises(IndexError):
            ds[2]

    def test_full_up_without_return_length_with_list_of_series(self):
        data = {
            'a': [np.random.normal(0, 1, (5, 3)),
                  np.random.normal(0, 1, (5, 2))],
            'c': np.random.normal(0, 1, (2, 4, 1)),
        }
        ds = tc.data.TensorSeriesDataset(data['a'], data['c'])
        self.assertEqual(len(ds), 2)
        self.assertEqual(ds._time_ranges, [3, 2])
        self.assertEqual(ds.shape, (2, -1, 3))

        a, c = ds[0]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 3))
        self.assertTrue((a.numpy() == data['a'][0]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][0, :, :]).all())

        a, c = ds[1]
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 2))
        self.assertTrue((a.numpy() == data['a'][1]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (4, 1))
        self.assertTrue((c.numpy() == data['c'][1, :, :]).all())

        with self.assertRaises(IndexError):
            ds[2]


class H5SeriesTest(unittest.TestCase):
    @staticmethod
    @contextmanager
    def _create_h5_file(shapes: Dict[str, Sequence[int]]):
        with tempfile.TemporaryDirectory() as temp_root:
            path = os.path.join(temp_root, 'test.h5')
            with h5py.File(path, 'w') as h5_file:
                for k, v in shapes.items():
                    if isinstance(v, np.ndarray):
                        h5_file[k] = v
                    else:
                        h5_file[k] = np.random.normal(0, 1, v)

            yield path

    def test_full_up_with_return_length(self):
        data = {
            'a': np.random.normal(0, 1, (2, 5, 1)),
            'b': np.random.normal(0, 1, (1, 4, 3)),
            'c': np.random.normal(0, 1, (2, 3, 3)),
        }

        with self._create_h5_file(data) as path:
            ds = tc.data.H5SeriesDataset(path, ('c', 'a'), return_length=2)
            self.assertEqual(len(ds), 4)
            self.assertEqual(ds._time_ranges, [3, 3])
            self.assertEqual(ds.shape, (2, -1, 3))

            c, a = ds[0]
            self.assertTrue(isinstance(a, torch.Tensor))
            self.assertEqual(a.shape, (5, 1))
            self.assertTrue((a.numpy() == data['a'][0, :, :]).all())
            self.assertTrue(isinstance(c, torch.Tensor))
            self.assertEqual(c.shape, (3, 2))
            self.assertTrue((c.numpy() == data['c'][0, :, :2]).all())

            c, a = ds[1]
            self.assertTrue(isinstance(a, torch.Tensor))
            self.assertEqual(a.shape, (5, 1))
            self.assertTrue((a.numpy() == data['a'][0, :, :]).all())
            self.assertTrue(isinstance(c, torch.Tensor))
            self.assertEqual(c.shape, (3, 2))
            self.assertTrue((c.numpy() == data['c'][0, :, 1:3]).all())

            c, a = ds[2]
            self.assertTrue(isinstance(a, torch.Tensor))
            self.assertEqual(a.shape, (5, 1))
            self.assertTrue((a.numpy() == data['a'][1, :, :]).all())
            self.assertTrue(isinstance(c, torch.Tensor))
            self.assertEqual(c.shape, (3, 2))
            self.assertTrue((c.numpy() == data['c'][1, :, :2]).all())

            c, a = ds[3]
            self.assertTrue(isinstance(a, torch.Tensor))
            self.assertEqual(a.shape, (5, 1))
            self.assertTrue((a.numpy() == data['a'][1, :, :]).all())
            self.assertTrue(isinstance(c, torch.Tensor))
            self.assertEqual(c.shape, (3, 2))
            self.assertTrue((c.numpy() == data['c'][1, :, 1:3]).all())

            with self.assertRaises(IndexError):
                ds[4]

    def test_full_up_with_return_length(self):
        data = {
            'a': np.random.normal(0, 1, (2, 5, 1)),
            'b': np.random.normal(0, 1, (1, 4, 3)),
            'c': np.random.normal(0, 1, (2, 3, 3)),
        }

        with self._create_h5_file(data) as path:
            ds = tc.data.H5SeriesDataset(path, ('c', 'a'))
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds._time_ranges, [3, 3])
            self.assertEqual(ds.shape, (2, -1, 3))

            c, a = ds[0]
            self.assertTrue(isinstance(a, torch.Tensor))
            self.assertEqual(a.shape, (5, 1))
            self.assertTrue((a.numpy() == data['a'][0, :, :]).all())
            self.assertTrue(isinstance(c, torch.Tensor))
            self.assertEqual(c.shape, (3, 3))
            self.assertTrue((c.numpy() == data['c'][0, :, :]).all())

            c, a = ds[1]
            self.assertTrue(isinstance(a, torch.Tensor))
            self.assertEqual(a.shape, (5, 1))
            self.assertTrue((a.numpy() == data['a'][1, :, :]).all())
            self.assertTrue(isinstance(c, torch.Tensor))
            self.assertEqual(c.shape, (3, 3))
            self.assertTrue((c.numpy() == data['c'][1, :, :]).all())

            with self.assertRaises(IndexError):
                ds[2]

    def test_h5_view(self):
        data = {
            'a': np.random.normal(0, 1, (2, 5, 1)),
            'b': np.random.normal(0, 1, (1, 4, 3)),
            'c': np.random.normal(0, 1, (2, 3, 3)),
        }

        with self._create_h5_file(data) as path:
            with h5py.File(path, 'r') as h5_file:
                h5_view = tc.data.h5_dataset.H5View(h5_file['a'])
                self.assertEqual(
                    h5_view.view, [slice(0, 2), slice(0, 5), slice(0, 1)]
                )
                self.assertEqual(h5_view.ndim, 3)
                self.assertEqual(h5_view.shape, (2, 5, 1))

                self.assertEqual(
                    h5_view[1].view, [1, slice(0, 5), slice(0, 1)]
                )
                self.assertEqual(h5_view[1].ndim, 2)
                self.assertEqual(h5_view[1].shape, (5, 1))

                self.assertEqual(
                    h5_view[1:].view, [slice(1, 2), slice(0, 5), slice(0, 1)]
                )
                self.assertEqual(h5_view[1:].ndim, 3)
                self.assertEqual(h5_view[1:].shape, (1, 5, 1))

                self.assertEqual(
                    h5_view[0][1].view, [0, 1, slice(0, 1)]
                )
                self.assertEqual(h5_view[0][1].ndim, 1)
                self.assertEqual(h5_view[0][1].shape, (1,))


class UtilsTest(unittest.TestCase):
    def test_coerce_to_multiseries(self):
        a = torch.randn((5, 3))
        b = torch.randn((5, 4))
        c = torch.randn((2, 5, 4))
        d = torch.randn((5,))

        out = tc.data.utils._coerce_to_multiseries([a, b])
        self.assertTrue(isinstance(out, tc.data.utils.ListOfSeries))

        out = tc.data.utils._coerce_to_multiseries(c)
        self.assertIs(out, c)

        out = tc.data.utils._coerce_to_multiseries(c.numpy())
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (2, 5, 4))
        self.assertTrue((out == c).all())

        with self.assertWarns(UserWarning):
            out = tc.data.utils._coerce_to_multiseries(a)
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (1, 5, 3))
        self.assertTrue((out.squeeze() == a).all())

        with self.assertWarns(UserWarning):
            out = tc.data.utils._coerce_to_multiseries(d)
        self.assertTrue(isinstance(out, torch.Tensor))
        self.assertEqual(out.shape, (1, 1, 5))
        self.assertTrue((out.squeeze() == d).all())

        with self.assertRaises(ValueError):
            tc.data.utils._coerce_to_multiseries(torch.randn(5, 5, 5, 5))

    def test_coerce_to_series(self):
        a = torch.randn((5, 3))
        b = torch.randn((5,))

        out = tc.data.utils._coerce_to_series(a)
        self.assertIs(out, a)

        out = tc.data.utils._coerce_to_series(a.numpy())
        self.assertEqual(out.shape, (5, 3))
        self.assertTrue((out == a).all())

        with self.assertWarns(UserWarning):
            out = tc.data.utils._coerce_to_series(b)
        self.assertEqual(out.shape, (1, 5))
        self.assertTrue((out.squeeze() == b).all())

        with self.assertRaises(ValueError):
            tc.data.utils._coerce_to_series(torch.randn(5, 5, 5))

    def test_list_of_series(self):
        a = torch.randn((5, 3))
        b = torch.randn((5, 4))
        ls = tc.data.utils.ListOfSeries([a, b])

        self.assertEqual(ls.ndim, 3)
        self.assertEqual(ls.shape, (2, 5, 4))

        self.assertEqual(ls[0].shape, (5, 3))
        self.assertTrue((ls[0] == a).all())
        self.assertTrue((ls[0, 1] == a[1]).all())

        self.assertEqual(ls[1].shape, (5, 4))
        self.assertTrue((ls[1] == b).all())
        self.assertTrue((ls[1, 1:3] == b[1:3]).all())


class SamplerTest(unittest.TestCase):
    def test_infinite_sampler(self):
        # Smoke test
        sampler = tc.data.samplers.InfiniteSampler(4)
        sample_iter = iter(sampler)
        out = [next(sample_iter) for _ in range(8)]


if __name__ == '__main__':
    unittest.main()
