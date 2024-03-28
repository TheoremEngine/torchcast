from typing import Iterable, Optional, Tuple, Union

import torch

from ._shaping import _ensure_nct, _sliding_window_view

__all__ = ['cross_spectral_density', 'power_spectral_density']

Ints = Union[int, Iterable[int]]


def cross_spectral_density(x: torch.Tensor, y: torch.Tensor,
                           sampling_frequency: float = 1.0,
                           window_length: Optional[int] = None,
                           overlap: int = 0, dim: int = -1,
                           batch_dim: Optional[Ints] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Args:
        x (:class:`torch.Tensor`): First tensor to calculate the cross spectral
            density of.
        y (:class:`torch.Tensor`): Second tensor to calculate the cross
            spectral density of.
        sampling_frequency (float): Frequency of samples in the tensor.
        window_length (optional, int): Length of window to use in Welch method.
            If not specified, defaults to the minimum of 256 and the length of
            the tensors.
        overlap (int): Length of window overlap to use in Welch method.
        dim (int): Dimension of time axis.
        batch_dim (optional, int or iterable of ints): Dimension(s) of batch
            axes.
    '''
    if x.shape[dim] != y.shape[dim]:
        raise ValueError(
            f'Time dimension ({dim}) does not match in x ({x.shape}), y '
            f'({y.shape})'
        )

    x, _ = _ensure_nct(x, dim, batch_dim)
    y, _ = _ensure_nct(y, dim, batch_dim)
    # TODO: Check x, y dtypes and lengths

    if window_length is None:
        window_length = min(256, x.shape[-1])
    window = torch.ones(window_length, device=x.device)

    if not isinstance(overlap, int):
        raise TypeError(overlap)

    is_real = (not torch.is_complex(x))
    freqs = (torch.fft.fftfreq, torch.fft.rfftfreq)[is_real](
        window_length, 1. / sampling_frequency, device=x.device
    )

    x = _apply_windowed_fft(x, window, overlap)
    y = _apply_windowed_fft(y, window, overlap)
    scale = 1.0 / (sampling_frequency * (window ** 2).sum())
    p_xy = scale * (x.conj() * y)

    if is_real:
        if window_length % 2:
            p_xy[..., 1:] *= 2
        else:
            p_xy[..., 1:-1] *= 2

    return freqs, p_xy.mean(dim=(0, 2))


def power_spectral_density(x: torch.Tensor, sampling_frequency: float = 1.0,
                           window_length: Optional[int] = None,
                           overlap: int = 0, dim: int = -1,
                           batch_dim: Optional[Ints] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Estimates the power spectral density, also called the periodogram.

    The calculation uses the Welch method from:

        Welch, 1967. ``The use of Fast Fourier Transform for the Estimation of
        Power Spectra: A Method Based on Time Averaging over Short, Modified
        Periodograms.'' *IEEE Transactions on Audio and Electroacoustic*, Vol.
        AU-15, No. 2, pp. 70-73.

    Args:
        x (:class:`torch.Tensor`): Tensor to calculate the power spectral
            density of.
        sampling_frequency (float): Frequency of samples in the tensor.
        window_length (optional, int): Length of window to use in Welch method.
            If not specified, defaults to the minimum of 256 and the length of
            the tensors.
        overlap (int): Length of window overlap to use in Welch method.
        dim (int): Dimension of time axis.
        batch_dim (optional, int or iterable of ints): Dimension(s) of batch
            axes.
    '''
    x, _ = _ensure_nct(x, dim, batch_dim)

    if window_length is None:
        window_length = min(256, x.shape[-1])
    window = torch.ones(window_length, device=x.device)

    if not isinstance(overlap, int):
        raise TypeError(overlap)

    is_real = (not torch.is_complex(x))
    freqs = (torch.fft.fftfreq, torch.fft.rfftfreq)[is_real](
        window_length, 1. / sampling_frequency, device=x.device
    )

    x = _apply_windowed_fft(x, window, overlap)
    scale = 1.0 / (sampling_frequency * (window ** 2).sum())
    p_xx = scale * (x.abs() ** 2)

    if is_real:
        if window_length % 2:
            p_xx[..., 1:] *= 2
        else:
            p_xx[..., 1:-1] *= 2

    return freqs, p_xx.mean(dim=(0, 2))


def _apply_windowed_fft(x: torch.Tensor, window: torch.Tensor, overlap: int) \
        -> torch.Tensor:
    x = _sliding_window_view(x, len(window))[:, :, 0::len(window) - overlap, :]
    x = x * window.view(1, 1, 1, -1)
    fft = torch.fft.fft if torch.is_complex(x) else torch.fft.rfft
    return fft(x, n=len(window))
