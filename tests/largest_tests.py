import argparse
import os
import unittest

import numpy as np
import torch
import torchcast as tc

from largest.dataloader import load_dataset


class _DummyLogger:
    def info(self, message):
        return


class LargeSTTest(unittest.TestCase):
    def test_ca(self):
        dummy_logger = _DummyLogger()
        data_path = os.path.join(os.path.dirname(__file__), 'largest/data')
        args = argparse.Namespace(
            years='test', input_dim=99999, seq_len=12, horizon=12, bs=2,
        )
        largest_ds, scaler = load_dataset(data_path, args, dummy_logger)

        for k in ['train', 'val', 'test']:
            largest_iter = largest_ds[f'{k}_loader'].get_iterator()
            largest_x, largest_y = next(largest_iter)
            largest_z = np.concatenate(
                (largest_x[..., 0], largest_y[..., 0]), axis=1
            )
            largest_z = torch.from_numpy(largest_z).float().permute(0, 2, 1)

            tc_ds = tc.datasets.LargeSTDataset(
                data_path, ['test'], split=k, return_length=24
            )
            tc_dl = torch.utils.data.DataLoader(tc_ds, batch_size=2)
            _, tc_z = next(iter(tc_dl))

            self.assertTrue(
                torch.isclose(largest_z, tc_z, atol=1e-4, rtol=1e-4).all()
            )

    def test_other(self):
        data_path = os.path.join(os.path.dirname(__file__), 'largest/data')
        # Smoke test only, to avoid having to distribute large data files
        for task in ['gla', 'gba', 'sd']:
            tc_ds = tc.datasets.LargeSTDataset(
                data_path, ['test'], split='all', task=task, return_length=24
            )
            tc_dl = torch.utils.data.DataLoader(tc_ds, batch_size=2)
            _, tc_z = next(iter(tc_dl))


if __name__ == '__main__':
    unittest.main()
