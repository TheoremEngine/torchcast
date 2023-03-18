from contextlib import contextmanager
import os
import tempfile
from typing import Dict, Sequence
import unittest

import h5py
import numpy as np
import torch
import torchcast as tc


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

    def test_coerce_inputs(self):
        # Base case
        shapes = {'a': (3, 5, 7), 'b': (3, 4, 7), 'c': (1, 2, 1)}
        with self._create_h5_file(shapes) as path:
            with h5py.File(path, 'r') as h5_file:
                a, b, c = tc.data.H5SeriesDataset._coerce_inputs(
                    h5_file['a'], h5_file['b'], h5_file['c']
                )
                self.assertTrue((a[...] == h5_file['a'][...]).all())
                self.assertTrue((b[...] == h5_file['b'][...]).all())
                self.assertTrue((c[...] == h5_file['c'][...]).all())

        # Test wrong number of dimensions
        with self._create_h5_file({'a': (3, 5), 'b': (3, 5, 7)}) as path:
            with h5py.File(path, 'r') as h5_file:
                with self.assertRaises(ValueError):
                    tc.data.H5SeriesDataset._coerce_inputs(
                        h5_file['a'], h5_file['b']
                    )
        with self._create_h5_file({'a': (3, 5, 7), 'b': (3, 5)}) as path:
            with h5py.File(path, 'r') as h5_file:
                with self.assertRaises(ValueError):
                    tc.data.H5SeriesDataset._coerce_inputs(
                        h5_file['a'], h5_file['b']
                    )

        # Test mismatch in shapes
        with self._create_h5_file({'a': (3, 5, 7), 'b': (3, 5, 6)}) as path:
            with h5py.File(path, 'r') as h5_file:
                with self.assertRaises(ValueError):
                    tc.data.H5SeriesDataset._coerce_inputs(
                        h5_file['a'], h5_file['b']
                    )

    def test_full_up(self):
        data = {
            'a': np.random.normal(0, 1, (2, 5, 1)),
            'b': np.random.normal(0, 1, (1, 4, 3)),
            'c': np.random.normal(0, 1, (2, 3, 3)),
        }

        # Test with return_length
        with self._create_h5_file(data) as path:
            ds = tc.data.H5SeriesDataset(path, ('c', 'a'), return_length=2)
            self.assertEqual(len(ds), 4)
            self.assertEqual(ds._index_range, (0, 2))
            self.assertEqual(ds._time_range, (0, 3))

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

        # Test without return_length
        with self._create_h5_file(data) as path:
            ds = tc.data.H5SeriesDataset(path, ('c', 'a'))
            self.assertEqual(ds._index_range, (0, 2))
            self.assertEqual(ds._time_range, (0, 3))

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

    def test_get_storage_shape(self):
        with self._create_h5_file({'a': (3, 5, 7), 'b': (3, 4, 1)}) as path:
            ds = tc.data.H5SeriesDataset(path, ('a', 'b'))
            self.assertEqual(ds._get_storage_shape(), (3, 7))

    def test_len(self):
        with self._create_h5_file({'a': (3, 5, 7), 'b': (3, 4, 1)}) as path:
            # Test with return_length
            ds = tc.data.H5SeriesDataset(path, ('a', 'b'), return_length=4)
            self.assertEqual(len(ds), 12)

            # Test without return_length
            ds = tc.data.H5SeriesDataset(path, ('a', 'b'))
            self.assertEqual(len(ds), 3)

    def test_retrieve_from_series(self):
        data = {
            'a': np.random.normal(0, 1, (3, 5, 7)),
            'b': np.random.normal(0, 1, (1, 4, 7)),
            'c': np.random.normal(0, 1, (3, 3, 1)),
        }

        with self._create_h5_file(data) as path:
            ds = tc.data.H5SeriesDataset(path, ('a', 'b', 'c'))
            a, b, c = ds._retrieve_from_series(1, 3, 6)

        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 3))
        self.assertTrue((a.numpy() == data['a'][1, :, 3:6]).all())
        self.assertTrue(isinstance(b, torch.Tensor))
        self.assertEqual(b.shape, (4, 3))
        self.assertTrue((b.numpy() == data['b'][0, :, 3:6]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (3, 1))
        self.assertTrue((c.numpy() == data['c'][1, :, :]).all())


class SamplerTest(unittest.TestCase):
    def test_infinite_sampler(self):
        # Smoke test
        sampler = tc.data.samplers.InfiniteSampler(4)
        sample_iter = iter(sampler)
        out = [next(sample_iter) for _ in range(8)]


class SeriesTest(unittest.TestCase):
    def test_get_item(self):
        class TestDataset(tc.data.SeriesDataset):
            @staticmethod
            def _coerce_inputs(*series):
                return None

            def _get_storage_shape(self):
                return 1, 1

            def _retrieve_from_series(self, *x):
                self.x = x
                return [1]

        # Test with return_length
        ds = TestDataset(return_length=2)
        ds._time_range = (2, 5)
        ds._index_range = (1, 3)

        self.assertEqual(len(ds), 4)

        ds[0]
        self.assertEqual(ds.x, (1, 2, 4))
        ds[1]
        self.assertEqual(ds.x, (1, 3, 5))
        ds[2]
        self.assertEqual(ds.x, (2, 2, 4))
        ds[3]
        self.assertEqual(ds.x, (2, 3, 5))

        with self.assertRaises(IndexError):
            ds[4]

        # Test without return_length
        ds = TestDataset()
        ds._time_range = (2, 5)
        ds._index_range = (1, 3)

        self.assertEqual(len(ds), 2)

        ds[0]
        self.assertEqual(ds.x, (1, 2, 5))
        ds[1]
        self.assertEqual(ds.x, (2, 2, 5))

        with self.assertRaises(IndexError):
            ds[2]


class TensorSeriesTest(unittest.TestCase):
    def test_coerce_inputs(self):
        # Base case
        data = {
            'a': np.random.normal(0, 1, (3, 5, 7)),
            'b': np.random.normal(0, 1, (1, 4, 7)),
            'c': np.random.normal(0, 1, (3, 3, 1)),
        }
        a, b, c = tc.data.TensorSeriesDataset._coerce_inputs(
            data['a'], data['b'], data['c']
        )
        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (3, 5, 7))
        self.assertTrue((a.numpy() == data['a']).all())
        self.assertTrue(isinstance(b, torch.Tensor))
        self.assertEqual(b.shape, (1, 4, 7))
        self.assertTrue((b.numpy() == data['b']).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (3, 3, 1))
        self.assertTrue((c.numpy() == data['c']).all())

        # Test wrong number of dimensions
        with self.assertWarns(UserWarning):
            x, = tc.data.TensorSeriesDataset._coerce_inputs(
                np.random.normal(0, 1, (9)),
            )
        self.assertTrue(isinstance(x, torch.Tensor))
        self.assertEqual(x.shape, (1, 1, 9))
        with self.assertWarns(UserWarning):
            x, = tc.data.TensorSeriesDataset._coerce_inputs(
                np.random.normal(0, 1, (2, 9)),
            )
        self.assertTrue(isinstance(x, torch.Tensor))
        self.assertEqual(x.shape, (1, 2, 9))

        with self.assertRaises(ValueError):
            tc.data.TensorSeriesDataset._coerce_inputs(
                np.random.normal(0, 1, (4, 2, 3, 9)),
            )

        # Test mismatch in shapes
        data = {
            'a': np.random.normal(0, 1, (3, 5, 7)),
            'b': np.random.normal(0, 1, (3, 5, 6)),
        }
        with self.assertRaises(ValueError):
            tc.data.H5SeriesDataset._coerce_inputs(data['a'], data['b'])

    def test_full_up(self):
        data = {
            'a': np.random.normal(0, 1, (2, 5, 3)),
            'c': np.random.normal(0, 1, (2, 4, 1)),
        }

        # Test with return_length
        ds = tc.data.TensorSeriesDataset(
            data['a'], data['c'], return_length=2
        )
        self.assertEqual(ds._index_range, (0, 2))
        self.assertEqual(ds._time_range, (0, 3))

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

        # Test without return_length
        ds = tc.data.TensorSeriesDataset(data['a'], data['c'])
        self.assertEqual(ds._index_range, (0, 2))
        self.assertEqual(ds._time_range, (0, 3))

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

    def test_get_storage_shape(self):
        data = {
            'a': np.random.normal(0, 1, (3, 5, 7)),
            'b': np.random.normal(0, 1, (1, 4, 7)),
            'c': np.random.normal(0, 1, (3, 4, 1)),
        }
        ds = tc.data.TensorSeriesDataset(data['a'], data['b'], data['c'])
        self.assertEqual(ds._get_storage_shape(), (3, 7))

    def test_len(self):
        # Test with return_length
        x = np.random.normal(0, 1, (3, 4, 7))
        ds = tc.data.TensorSeriesDataset(x, return_length=4)
        self.assertEqual(len(ds), 12)

        # Test without return_length
        ds = tc.data.TensorSeriesDataset(x)
        self.assertEqual(len(ds), 3)

    def test_retrieve_from_series(self):
        # First, without NaNs
        data = {
            'a': np.random.normal(0, 1, (3, 5, 7)),
            'b': np.random.normal(0, 1, (1, 4, 7)),
            'c': np.random.normal(0, 1, (3, 3, 1)),
        }

        ds = tc.data.TensorSeriesDataset(data['a'], data['b'], data['c'])
        a, b, c = ds._retrieve_from_series(1, 3, 6)

        self.assertTrue(isinstance(a, torch.Tensor))
        self.assertEqual(a.shape, (5, 3))
        self.assertTrue((a.numpy() == data['a'][1, :, 3:6]).all())
        self.assertTrue(isinstance(b, torch.Tensor))
        self.assertEqual(b.shape, (4, 3))
        self.assertTrue((b.numpy() == data['b'][0, :, 3:6]).all())
        self.assertTrue(isinstance(c, torch.Tensor))
        self.assertEqual(c.shape, (3, 1))
        self.assertTrue((c.numpy() == data['c'][1, :, :]).all())


if __name__ == '__main__':
    unittest.main()
