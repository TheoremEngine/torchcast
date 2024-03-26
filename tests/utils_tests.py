import unittest

import numpy as np
import scipy.fft
import statsmodels.api as sm
import torch
import torchcast as tc


class AutoCovTests(unittest.TestCase):
    def test_lagged_sum_dot(self):
        '''
        Tests the _lagged_prod_sum_dot function. This calculates the sum of the
        lagged dot products explicitly rather than using the FFT. As an
        internal function, the input MUST be in NCT arrangement.
        '''

        x = torch.arange(12.).view(2, 2, -1)
        sum_prod = tc.utils.autocov._lag_prod_sum_dot(x, 2)
        self.assertEqual(sum_prod.shape, (2, 3))
        should_be = torch.tensor([
            [154., 100., 48.],
            [352., 232., 114.]
        ])
        self.assertTrue(torch.isclose(sum_prod, should_be).all(), sum_prod)

    def test_lagged_sum_fft(self):
        '''
        Tests the _lagged_prod_sum_fft function. This calculates the sum of the
        lagged dot products using the FFT. As an internal function, the input
        MUST be in NCT arrangement.
        '''
        x = torch.arange(12.).view(2, 2, -1)
        sum_prod = tc.utils.autocov._lag_prod_sum_fft(x, 2)
        self.assertEqual(sum_prod.shape, (2, 3))
        should_be = torch.tensor([
            [154., 100., 48.],
            [352., 232., 114.]
        ])
        self.assertTrue(torch.isclose(sum_prod, should_be).all(), sum_prod)

    def test_autocorrelation(self):
        '''
        Basic test of the autocorrelation function.
        '''
        # Since this is a simple wrapper around autocovariance, we only test it
        # once.
        x = torch.arange(9.)
        tc_autocor = tc.utils.autocorrelation(x, adjusted=False)
        tc_autocor = tc_autocor.flatten().numpy()
        sm_autocor = sm.tsa.acf(x.numpy(), adjusted=False, nlags=8, fft=False)
        self.assertEqual(tc_autocor.shape, sm_autocor.shape)
        self.assertTrue(
            np.isclose(tc_autocor, sm_autocor).all(),
            (tc_autocor, sm_autocor)
        )

    def test_autocovariance_n1_unmasked_denom(self):
        '''
        Tests the autocovariance method, when there's no NaN values. The
        test_lagged_sum_* methods establish that we can properly calculate the
        sum of the lagged products, so this only tests the calculation of the
        denominator.
        '''
        x = torch.arange(9.)
        for adjusted in [False, True]:
            tc_autocov = tc.utils.autocovariance(x, adjusted=adjusted)
            tc_autocov = tc_autocov.flatten().numpy()
            sm_autocov = sm.tsa.acovf(x.numpy(), adjusted=adjusted, fft=False)
            self.assertEqual(tc_autocov.shape, sm_autocov.shape)
            self.assertTrue(
                np.isclose(tc_autocov, sm_autocov).all(),
                (tc_autocov, sm_autocov)
            )

    def test_autocovariance_n1_masked_denom(self):
        '''
        Tests the autocovariance method, when there are NaN values in the
        series. The test_lagged_sum_* methods establish that we can properly
        calculate the sum of the lagged products, so this only tests the
        calculation of the denominator.
        '''
        x = torch.arange(9.)
        x[4] = float('nan')
        for adjusted in [False, True]:
            tc_autocov = tc.utils.autocovariance(x, adjusted=adjusted)
            tc_autocov = tc_autocov.flatten().numpy()
            sm_autocov = sm.tsa.acovf(
                x.numpy(), adjusted=adjusted, fft=False, missing='conservative'
            )
            self.assertEqual(tc_autocov.shape, sm_autocov.shape)
            self.assertTrue(
                np.isclose(tc_autocov, sm_autocov).all(),
                (tc_autocov, sm_autocov, adjusted)
            )

    def test_autocovariance_n4_unmasked_denom(self):
        '''
        Extends the test of the autocovariance method to a case with multiple
        samples of the series.
        '''
        x = torch.arange(8.).view(4, 1, 2)
        # Did this by hand.
        should_be = {
            True: torch.tensor([[5.25, 4.75]]),
            False: torch.tensor([[5.25, 2.375]]),
        }

        for adjusted in [False, True]:
            ac = tc.utils.autocovariance(x, batch_dim=0, adjusted=adjusted)
            self.assertEqual(ac.shape, (1, 2))
            self.assertTrue(
                torch.isclose(ac, should_be[adjusted]).all(),
                (ac, adjusted, should_be[adjusted])
            )

    def test_autocovariance_multichannel(self):
        '''
        Extends the test of the autocovariance method to a case with multiple
        channels.
        '''
        # x will be in CT arrangement and have two channels.
        x = torch.arange(12.).view(2, 6)
        for adjusted in [False, True]:
            tc_autocov = tc.utils.autocovariance(x, adjusted=adjusted).numpy()
            sm_autocov = [
                sm.tsa.acovf(_x, adjusted=adjusted, fft=False)
                for _x in x.numpy()
            ]
            sm_autocov = np.stack(sm_autocov, axis=0)
            self.assertEqual(tc_autocov.shape, sm_autocov.shape)
            self.assertTrue(
                np.isclose(tc_autocov, sm_autocov).all(),
                (tc_autocov, sm_autocov)
            )

    def test_next_fast_len(self):
        for i in range(100):
            self.assertEqual(
                tc.utils.autocov.next_fast_len(i),
                scipy.fft.next_fast_len(i, real=True),
            )

    def test_partial_autocorrelation(self):
        '''
        Tests partial_autocorrelation function.
        '''
        # Need to have plenty of elements and a clear autoregressive
        # relationship because statsmodels uses different math.
        x = torch.randn(1000,)
        x = x[1:] + (0.5 * x[:-1])
        should_be = sm.tsa.pacf(x.numpy(), nlags=5, method='ywadjusted')
        pac = tc.utils.partial_autocorrelation(
            x, n_lags=5, adjusted=True,
        )

        self.assertTrue(
            (np.abs(should_be - pac.numpy()) <= 1e-4).all(),
            (should_be, pac.numpy())
        )

    def test_partial_autocorrelation_multichannel(self):
        '''
        Tests partial_autocorrelation function with multiple channels.
        '''
        # Need to have plenty of elements and a clear autoregressive
        # relationship because statsmodels uses different math.
        x = torch.randn((2, 1000,))
        x = x[:, 1:] + (0.5 * x[:, :-1])
        should_be = [
            sm.tsa.pacf(_x, nlags=5, method='ywadjusted')
            for _x in x.numpy()
        ]
        should_be = np.stack(should_be, axis=0)

        pac = tc.utils.partial_autocorrelation(
            x, n_lags=5, adjusted=True,
        )

        self.assertTrue(
            (np.abs(should_be - pac.numpy()) <= 1e-4).all(),
            (should_be, pac.numpy())
        )


class ShapingTests(unittest.TestCase):
    def test_ensure_nct_series(self):
        orig_x = torch.arange(6.).view(1, 2, 3)
        x, reshape = tc.utils._shaping._ensure_nct(orig_x, -1)
        self.assertEqual(x.shape, (1, 2, 3))
        rtn_x = reshape(x)
        self.assertEqual(rtn_x.shape, (1, 2, 3))
        self.assertTrue((rtn_x == orig_x.squeeze(0)).all())

        # Input tensor has arrangement (C_1, T, C_2)
        orig_x = torch.randn((3, 5, 2))
        x, reshape = tc.utils._shaping._ensure_nct(orig_x, 1)
        self.assertEqual(x.shape, (1, 6, 5))
        rtn_x = reshape(x[0])
        self.assertEqual(rtn_x.shape, (3, 5, 2))
        self.assertTrue((orig_x == rtn_x).all())

        # Input tensor has the arrangement (C_1, N_1, T, N_2, C_2).
        C_1, N_1, T, N_2, C_2 = 2, 3, 11, 5, 7
        orig_x = torch.randn((C_1, N_1, T, N_2, C_2))
        x, reshape = tc.utils._shaping._ensure_nct(orig_x, 2, batch_dim=(1, 3))
        self.assertEqual(x.shape, (N_1 * N_2, C_1 * C_2, T))
        rtn_x = reshape(x.mean(0))
        self.assertEqual(rtn_x.shape, (C_1, T, C_2))
        self.assertTrue(
            torch.isclose(orig_x.mean((1, 3)), rtn_x, atol=1e-6).all(),
            (orig_x.mean((1, 3)), rtn_x)
        )

    def test_sliding_window_view(self):
        x = torch.arange(60).view(5, 4, 3)
        win_x = tc.utils._shaping._sliding_window_view(x, 3, 1)

        self.assertEqual(win_x.shape, (5, 2, 3, 3))
        self.assertTrue((win_x[:, 0, :, :] == x[:, 0:3, :]).all())
        self.assertTrue((win_x[:, 1, :, :] == x[:, 1:4, :]).all())

        win2_x = tc.utils._shaping._sliding_window_view(win_x, 2, 2)
        self.assertEqual(win2_x.shape, (5, 2, 2, 2, 3))
        self.assertTrue((win2_x[:, :, 0, :, :] == win_x[:, :, 0:2, :]).all())
        self.assertTrue((win2_x[:, :, 1, :, :] == win_x[:, :, 1:3, :]).all())


if __name__ == '__main__':
    unittest.main()
