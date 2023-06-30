from functools import lru_cache
from typing import Iterator, Optional, Union

import torch

from ._shaping import _ensure_nct

__all__ = [
    'autocorrelation', 'autocovariance', 'partial_autocorrelation'
]

Ints = Union[int, Iterator[int]]


def autocorrelation(series: torch.Tensor, n_lags: Optional[int] = None,
                    adjusted: bool = True, dim: int = -1,
                    batch_dim: Optional[Ints] = None,
                    use_fft: Optional[bool] = None) -> torch.Tensor:
    '''
    Calculates the autocorrelation of a series. The autocorrelation is the
    ratio of the autocovariance to the variance:

    .. math::

        \\mbox{Autocorr}(x)_k = \\mbox{Autocov}(x)_k / \\Var(x)

    For a multivariate series, this returns only the autocorrelations of each
    channel to itself. The autocorrelation is returned as a 2-dimensional
    :class:`torch.Tensor`, with the 0th dimension indexing the channel and the
    1st dimension indexing the lag.

    Args:
        series (:class:`torch.Tensor`): The series to find the autocorrelation
        of.
        n_lags (optional, int): Number of lags. If not set, return all lags.
        adjusted (bool): If set, use (n - k) in the denominator instead of n.
        The adjusted estimator has lower bias but typically has worse mean
        squared error.
        dim (int): Dimension indexing time in the series.
        batch_dim (optional, int or iterator of int): Dimensions indexing the
        batches. If provided, we assume that each row along these dimensions is
        a different sample from the same underlying distribution.
        use_fft (optional, bool): Allows user to specify explicitly whether to
        use the FFT in the calculation. FFT is generally faster for longer time
        series, and slower for shorter ones. If not set, then we infer from the
        size of the data.
    '''
    # autocov will be in shape CT coming out of autocovariance.
    autocov = autocovariance(
        series, n_lags=n_lags, adjusted=adjusted, dim=dim,
        batch_dim=batch_dim, use_fft=use_fft,
    )
    # Do this in-place to save a little memory. We need the clone to avoid a
    # RuntimeError caused by modifying the dividor at the same time.
    autocov /= autocov[:, 0:1].clone()

    return autocov


def autocovariance(series: torch.Tensor, n_lags: Optional[int] = None,
                   adjusted: bool = True, dim: int = -1,
                   batch_dim: Optional[Ints] = None,
                   use_fft: Optional[bool] = None) -> torch.Tensor:
    '''
    Calculates the autocovariance of a series:

    .. math::

        \\mbox{Autocov}(x)_k = \\mathbb{E}(x_tx_{t + k}) - (\\mathbb{E}x)^2

    Where the expectation is taken over time. Note that the 0th autocovariance
    is equal to the variance of the series.

    For a multivariate series, this returns only the autocovariances of each
    channel to itself; to get the full matrix use the cross-covariance
    function. The autocovariance is returned as a 2-dimensional
    :class:`torch.Tensor`, with the 0th dimension indexing the channel and the
    1st dimension indexing the lag.

    Args:
        series (:class:`torch.Tensor`): The series to find the autocovariance
        of.
        n_lags (optional, int): Number of lags. If not set, return all lags.
        adjusted (bool): If set, use (n - k) in the denominator instead of n.
        The adjusted estimator has lower bias but typically has worse mean
        squared error.
        dim (int): Dimension indexing time in the series.
        batch_dim (optional, int or iterator of int): Dimensions indexing the
        batches. If provided, we assume that each row along these dimensions is
        a different sample from the same underlying distribution.
        use_fft (optional, bool): Allows user to specify explicitly whether to
        use the FFT in the calculation. FFT is generally faster for longer time
        series, and slower for shorter ones. If not set, then we infer from the
        size of the data.
    '''
    # Implementation is based closely on statsmodels.tsa.stattools.acovf.

    # We begin by normalizing to NCT form. We calculate the autocovariance over
    # each channel separately, but we assume that each entry in the batch
    # corresponds to the same process but separate instantiations - so if our
    # series has shape (2, 3, 5), then autocovariance will be of shape (3, 5),
    # and autocovariance[i, j] will be the covariance of z_it, z_i(t-j),
    # calculated jointly over batch 0 and 1.
    #
    # We use rtn_shape and rtn_permute to keep track of how we reshaped the
    # tensor so we can undo it. Specifically, at the bottom of this function,
    # we'll end up with a tensor containing the autocovariance in CL form,
    # where L indexs the lag. We then reshape:
    #
    # autocov = autocov.view(*rtn_shape).permute(*rtn_permute)
    n_lags = n_lags or (series.shape[dim] - 1)
    series, _ = _ensure_nct(series, dim, batch_dim)
    N, C, T = series.shape

    if T < n_lags:
        raise ValueError('Series must have length at least equal to n_lags.')

    # Construct a mask of missing values, and set those missing values to 0.
    # deal_with_masked will track whether there were any NaNs, as we can
    # default to faster paths if there aren't any.
    is_nan = torch.isnan(series)
    deal_with_masked = is_nan.any().item()

    # Normalize by mean. Note that we make this out-of-place deliberately, so
    # that the function does not have a side effect on the inputs. Note that,
    # after this point, all NaNs will be replaced by zeros.
    if deal_with_masked:
        is_notnan = ~is_nan
        series = series.clone()
        # Set NaNs to 0 so that they will not effect the calculations.
        series[is_nan] = 0
        # Normalize by the mean, calculating the denominator by excluding NaNs.
        series -= (series.sum((0, 2)) / is_notnan.sum((0, 2))).view(1, -1, 1)
        # Reset NaN entries back to 0 again.
        series[is_nan] = 0
    else:
        series = series - series.mean((0, 2), keepdim=True)

    # The FFT offers a faster option for computing the autocovariance for long
    # time series. I have not done extensive benchmarking of when the FFT
    # option is faster yet.
    use_fft = (n_lags > 4) if (use_fft is None) else use_fft

    _lag_prod_sum = _lag_prod_sum_fft if use_fft else _lag_prod_sum_dot

    # Calculate the sum of the products of the lags. This will be a CL-shaped
    # tensor, where L is the number of lags. Entry (i, j) will be the sum:
    #
    # sum_k sum_t series[k, i, t] * series[k, i, t + j]
    autocov = _lag_prod_sum(series, n_lags)

    # Calculate the denominator.
    if deal_with_masked and adjusted:
        # We will be using is_notnan to calculate denominators, and so it
        # needs to be a float. We need to convert at this stage because
        # torch.bmm is not implemented for the bool data type.
        is_notnan = is_notnan.float()
        # We calculate the count by the same procedure as we calculated the
        # sum of the products.
        denom = _lag_prod_sum(is_notnan, n_lags)
        denom[denom == 0] = 1
    elif deal_with_masked:
        denom = is_notnan.sum((0, 2)).float()
    elif adjusted:
        denom = torch.arange(
            N * T, N * T - ((n_lags + 1) * N), -N,
            device=series.device
        )
    else:
        denom = N * T

    autocov /= denom

    # Okay, however we did it, we have now calculated the autocovariance.

    return autocov


def partial_autocorrelation(series: torch.Tensor, n_lags: Optional[int] = None,
                            adjusted: bool = True, dim: int = -1,
                            batch_dim: Optional[Ints] = None,
                            use_fft: Optional[bool] = None) -> torch.Tensor:
    '''
    Calculates the partial autocorrelation of a series, using the recursive
    Yule-Walker method.

    This currently returns only the autocorrelations of each channel to itself,
    it does not support returning the full matrix. The autocorrelation is
    returned as a 2-dimensional :class:`torch.Tensor`, with the 0th dimension
    indexing the channel and the 1st dimension indexing the lag.

    Args:
        series (:class:`torch.Tensor`): The series to find the partial
        autocorrelation of.
        n_lags (optional, int): Number of lags. If not set, return all lags.
        adjusted (bool): If set, use (n - k) in the denominator instead of n.
        The adjusted estimator has lower bias but typically has worse mean
        squared error.
        dim (int): Dimension indexing time in the series.
        batch_dim (optional, int or iterator of int): Dimensions indexing the
        batches. If provided, we assume that each row along these dimensions is
        a different sample from the same underlying distribution.
        use_fft (optional, bool): Allows user to specify explicitly whether to
        use the FFT in the calculation. FFT is generally faster for longer time
        series, and slower for shorter ones. If not set, then we infer from the
        size of the data.
    '''
    # We apply the Yule-Walker method, see p. 84 of Box, Jenkins, et al, "Time
    # Series Analysis", 5th edition. Specifically, equations A3.2.7 and A3.2.8
    # on p. 86.

    # Coerce to NCT arrangement.
    n_lags = n_lags or (series.shape[dim] - 1)
    series, _ = _ensure_nct(series, dim, batch_dim)

    # Calculate the autocorrelation. Remember that autocorr will be in CT
    # arrangement.
    autocorr = autocorrelation(
        series, n_lags=n_lags, adjusted=adjusted, dim=2, batch_dim=0,
        use_fft=use_fft
    )

    # Build holder for partial autocorrelation.
    pac = torch.empty_like(autocorr)
    # Flipped value of the autocorrelation, for convenience.
    ac_flip = torch.flip(autocorr, dims=(1,))

    # The first partial autocorrelation equals the first autocorrelation. Note
    # that phi should be in CT arrangement, even when T=1.
    phi, pac[:, 0:2] = autocorr[:, 1:2], autocorr[:, 0:2]
    C, _ = autocorr.shape

    # Iterate through remaining values.
    for p in range(2, n_lags + 1):
        # Calculate the new partial autocorrelation. First, get the numerator
        # of A3.2.8:
        #
        # r_{c,p+1} - sum_{j=1}^p phi^p_{c,j} r_{c,p+1-j}
        #
        # Where r is the autocorrelation. Since we want to reverse the
        # indexing, we use ac_flip. We implement this through torch.baddbmm,
        # since sadly there's no batched dot product as of PyTorch 1.8.1. The
        # numerator will then be in C11 arrangement.
        a328_num = torch.baddbmm(
            autocorr[:, p].view(C, 1, 1),       # input : C11
            phi.unsqueeze(1),                   # batch1: C1T
            ac_flip[:, -p:-1].unsqueeze(2),     # batch2: CT1
            alpha=-1,
        )

        # Next, get the denominator:
        #
        # 1 - sum_{j=1}^p phi^p_{c,j} r_{c,j}
        #
        # Again, this will come out in C11 arrangement.
        a328_den = 1 - torch.bmm(
            phi.unsqueeze(1),                   # input : C1T
            autocorr[:, 1:p].unsqueeze(2),      # mat2  : CT1
        )

        # Okay, now we have the new partial autocorrelation, which is also
        # phi^{p+1}_{c,p+1}:
        pac[:, p] = a328_num.view(C) / a328_den.view(C)

        # Now we calculate the remaining pieces of the new phi, for use in the
        # next iteration, as in A3.2.7:
        #
        # phi^{p+1}_{c,j} = phi^p_{c,j} - phi^{p+1}_{c,p+1} phi^p_{p+1-j}
        phi = phi - pac[:, p:(p + 1)] * torch.flip(phi, dims=(1,))
        # And stick phi^{p+1}_{c,p+1} on there as well:
        phi = torch.cat((phi, pac[:, p:p + 1]), dim=1)

    return pac


def _lag_prod_sum_dot(series: torch.Tensor, n_lags: int) -> torch.Tensor:
    '''
    Calculates the sum of the lagged products using the explicit dot product
    calculation instead of the FFT. This is equal to the autocoviarance without
    dividing by the denominator.

    Args:
        series (:class:`torch.Tensor`): The series to find the sum of the
        lagged products for. This is expected to be in NCT arrangement, where
        the 0th dimension indexs the batch, the 1st dimension indexs the
        channel, and the 2nd dimension indexs the time. We assume that each
        batch is a separate instantiation of the same process, and so the
        sum is found jointly over all the batches. We assume that the series
        has already been de-meaned and, if the series contains any invalid/NaN
        entries, they have been replaced with 0s.
        n_lags (int): The number of lags to calculate the autocovariance for.
    '''
    N, C, T = series.shape

    # This construction allows us to avoid an unnecessary copy if N = 1, which
    # will be a common occurrence.
    if N > 1:
        def _sum_on_batch(x):
            return x.view(N, C).sum(0)
    else:
        def _sum_on_batch(x):
            return x.view(C)

    # There ought to be a better way to do this, and there's a long-standing
    # PyTorch issue for a batched dot product:
    #
    # https://github.com/pytorch/pytorch/issues/18027
    #
    # But this appears to currently be the best option. Now, the bmm expects
    # the matrices to be in the form NUV and NVW, and will produce output of
    # shape NUW. Our series is currently in the form NCT, and we therefore want
    # to reshape it into form (NC)1T and (NC)T1. This will produce output of
    # the form (NC)11, which we can then reshape to NC. This will give us the
    # sum of the products:
    #
    # sum_t z_nct * z_nc(t+j)
    #
    # And we then need to further sum over n in order to aggregate those
    # values. Note that, since we set our NaNs to zeros, they will have no
    # impact on the actual results of these matrix multiplies. Note also that,
    # since we already de-meaned our series, we only need to divide by the
    # denominator to obtain the autocovariance, we do not need to subtract the
    # squared mean.
    sum_prod = torch.bmm(series.view(-1, 1, T), series.view(-1, T, 1))
    sum_prods = [_sum_on_batch(sum_prod)]

    for lag in range(1, n_lags + 1):
        sum_prod = torch.bmm(
            series.view(-1, T)[:, lag:].unsqueeze(1),
            series.view(-1, T)[:, :-lag].unsqueeze(2)
        )
        sum_prods.append(_sum_on_batch(sum_prod))

    return torch.stack(sum_prods, dim=-1)


def _lag_prod_sum_fft(series: torch.Tensor, n_lags: int) -> torch.Tensor:
    '''
    Calculates the sum of the lagged products using the FFT. This is equal to
    the autocoviarance without dividing by the denominator.

    Args:
        series (:class:`torch.Tensor`): The series to find the sum of the
        lagged products for. This is expected to be in NCT arrangement, where
        the 0th dimension indexs the batch, the 1st dimension indexs the
        channel, and the 2nd dimension indexs the time. We assume that each
        batch is a separate instantiation of the same process, and so the
        sum is found jointly over all the batches. We assume that the series
        has already been de-meaned and, if the series contains any invalid/NaN
        entries, they have been replaced with 0s.
        n_lags (int): The number of lags to calculate the autocovariance for.
    '''
    # The FFT series will be:
    #
    # fft_series[k]
    #     = sum_n x[n] e^{-2 * pi * i * k * n / N}
    #     = sum_n x[n] (cos(2 * pi * k * n / N) +
    #                   i sin (2 * pi * k * n / N))
    #
    # We then multiply by the complex conjugate, which is equal to getting
    # the square of the absolute value. The equations get a little messy,
    # but we start with:
    #
    #     (sum_n x[n] cos(2 * pi * k * n / N))^2
    #             + (sum_n x[n] sin(2 * pi * k * n / N))^2
    #
    # Then we expand those out. We get two sets of terms, one where a term
    # is multiplied by itself:
    #
    #     sum_n (x[n] cos(2 * pi * k * n / N))^2
    #             + sum_n (x[n] sin(2 * pi * k * n / N))^2
    #
    # And one where unlike terms are multiplied together:
    #
    #     2 sum_{n_1 != n_2} (x[n_1] * x[n_2] * cos(2 * pi * k * n_1 / N)
    #             * cos(2 * pi * k * n_2 / N)) + ...
    #
    # In the first set of terms, we apply the Pythagorean identity to
    # combine cosine and sine terms. In the second set of terms, we use the
    # product-to-sum identities. Together, we obtain that the squared
    # magnitude of the discrete FFT is:
    #
    # = sum_n (x[n])^2 +
    #     2 sum_{n_1 != n_2} (x[n_1] * x[n_2] *
    #                         cos (2 * pi * k * (n_1 - n_2) / N)
    #
    # Then, we take the inverse discrete FFT of this. We'll use l as the
    # variable for the inverse FFT. If l == 0, then we're left with only
    # the constant term:
    #
    # sum_n (x[n]^2)
    #
    # If l != 0, then the constant term falls out. We apply the product-to-
    # sum formula gain, obtaining:
    #
    #     (1 / N) sum_k sum_{n_1 != n_2} x[n_1] * x[n_2] * (
    #             cos(2 * pi * k * (n_1 - n_2 + l) / N) +
    #             cos(2 * pi * k * (n_1 - n_2 - l) / N) )
    #
    # The summed terms are 0 unless l = +/- (n_1 - n_2). So once we sum on
    # the different series in the batch and divide by the appropriate
    # denominator, we get the autocovariance.

    # Padding the entry to the next regular number significantly improves
    # performance - by a factor of about x2 in my benchmarks.
    _, _, T = series.shape
    n = next_fast_len(2 * T + 1)
    fft_series = torch.fft.fft(series, n=n, dim=-1)
    sum_prods = torch.fft.ifft(fft_series.abs().pow_(2))
    # Truncate to remove excess entries.
    sum_prods = sum_prods[:, :, :n_lags + 1].real
    # Sum on the different batches.
    return sum_prods.sum(0)


@lru_cache
def next_fast_len(n: int):
    '''
    Calculates the optimal size for performing the FFT, which will be the
    smallest composite of 2, 3, and 5 that is larger than the argument. This is
    based on the :func:`good_size_real` function from PocketFFT.
    '''
    if n <= 6:
        return n

    best_fac, f_5 = 2 * n, 1
    while f_5 < best_fac:
        x = f_5
        while x < n:
            x *= 2
        while True:
            if x < n:
                x *= 3
            elif x > n:
                if x < best_fac:
                    best_fac = x
                if x % 2 == 1:
                    break
                x //= 2
            else:
                return n
        f_5 *= 5
    return best_fac
