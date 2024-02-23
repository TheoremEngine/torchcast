from math import prod
from typing import Iterator

import torch
import torch.nn.functional as f

__all__ = [
    'l1_loss', 'L1Loss', 'mse_loss', 'MSELoss', 'smooth_l1_loss',
    'SmoothL1Loss',
]


# The torch.nansum and torch.nanmean do not accurately backprop through the
# non-NaN values. So we implement torch.autograd.Functions.


class NaNSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.is_nan, ctx.shape = torch.isnan(tensor), tensor.shape
        ctx.is_nan_all = ctx.is_nan.all()

        if ctx.is_nan_all:
            return torch.zeros((), device=tensor.device)
        else:
            return torch.nansum(tensor)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        if ctx.is_nan_all:
            return torch.zeros(ctx.shape, dtype=grad.dtype, device=grad.device)
        else:
            grad = torch.full(
                ctx.shape, grad.item(), dtype=grad.dtype, device=grad.device,
            )
            grad[ctx.is_nan] = 0
            return grad


class NaNMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.is_nan, ctx.shape = torch.isnan(tensor), tensor.shape
        ctx.is_nan_all = ctx.is_nan.all()

        if ctx.is_nan_all:
            return torch.zeros((), device=tensor.device)
        else:
            return torch.nanmean(tensor)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Iterator[torch.Tensor]:
        if ctx.is_nan_all:
            return torch.zeros(ctx.shape, dtype=grad.dtype, device=grad.device)
        else:
            n_real = prod(ctx.shape) - ctx.is_nan.sum().item()
            grad = torch.full(
                ctx.shape, grad.item() / n_real, dtype=grad.dtype,
                device=grad.device,
            )
            grad[ctx.is_nan] = 0
            return grad


def l1_loss(pred: torch.Tensor, target: torch.Tensor,
            reduction: str = 'mean') -> torch.Tensor:
    '''
    This is an L1 loss that ignores NaN values.

    Args:
        pred (:class:`torch.Tensor`): Predictions.
        target (:class:`torch.Tensor`): Targets for the predictions. The
        predictions and targets must be broadcastable.
        reduction (str): Form of reduction to apply. Choices: 'mean', 'sum'.
    '''
    loss = f.l1_loss(pred, target, reduction='none')

    if reduction == 'mean':
        return NaNMean.apply(loss)
    elif reduction == 'sum':
        return NaNSum.apply(loss)
    else:
        raise ValueError(reduction)


class L1Loss(torch.nn.Module):
    '''
    This is an L1 loss that ignores NaN values.
    '''
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        return l1_loss(pred, target, reduction=self.reduction)


def mse_loss(pred: torch.Tensor, target: torch.Tensor,
             reduction: str = 'mean') -> torch.Tensor:
    '''
    This is a mean-squared error loss that ignores NaN values.

    Args:
        pred (:class:`torch.Tensor`): Predictions.
        target (:class:`torch.Tensor`): Targets for the predictions. The
        predictions and targets must be broadcastable.
        reduction (str): Form of reduction to apply. Choices: 'mean', 'sum'.
    '''
    loss = f.mse_loss(pred, target, reduction='none')

    if reduction == 'mean':
        return NaNMean.apply(loss)
    elif reduction == 'sum':
        return NaNSum.apply(loss)
    else:
        raise ValueError(reduction)


class MSELoss(torch.nn.Module):
    '''
    This is a mean-squared error loss that ignores NaN values.
    '''
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        return mse_loss(pred, target, reduction=self.reduction)


def smooth_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                   reduction: str = 'mean', beta: float = 1.0) -> torch.Tensor:
    '''
    This is a smooth L1 loss that ignores NaN values.

    Args:
        pred (:class:`torch.Tensor`): Predictions.
        target (:class:`torch.Tensor`): Targets for the predictions. The
        predictions and targets must be broadcastable.
        reduction (str): Form of reduction to apply. Choices: 'mean', 'sum'.
        beta (float): Boundary between L1 and L2 components.
    '''
    loss = f.smooth_l1_loss(pred, target, reduction='none', beta=beta)

    if reduction == 'mean':
        return NaNMean.apply(loss)
    elif reduction == 'sum':
        return NaNSum.apply(loss)
    else:
        raise ValueError(reduction)


class SmoothL1Loss(torch.nn.Module):
    '''
    This is a smooth L1 loss that ignores NaN values.
    '''
    def __init__(self, reduction: str = 'mean', beta: float = 1.0):
        super().__init__()
        self.reduction = reduction
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        return smooth_l1_loss(
            pred, target, reduction=self.reduction, beta=self.beta
        )
