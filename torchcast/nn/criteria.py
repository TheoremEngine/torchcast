from typing import Iterator, Tuple

import torch
import torch.nn.functional as f

__all__ = [
    'l1_loss', 'L1Loss', 'mse_loss', 'MSELoss', 'smooth_l1_loss',
    'SmoothL1Loss',
]


class ZeroGrad(torch.autograd.Function):
    '''
    Returns a zero value. Backpropping through this function generates zeros
    for all gradients. The purpose of this function is to handle cases where
    you are masking out all values in a tensor but want to keep the tensor
    connected to the gradient graph.
    '''
    @staticmethod
    def forward(ctx, *tensors: torch.Tensor) -> torch.Tensor:
        ctx.shapes = [x.shape for x in tensors]
        return torch.zeros((), device=tensors[0].device)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Iterator[torch.Tensor]:
        return tuple(torch.zeros(s, device=grad.device) for s in ctx.shapes)


class NaNMask(torch.autograd.Function):
    '''
    Changes nothing in the forward pass, replaces all NaNs in the backward pass
    with zeros.
    '''
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        grad[torch.isnan(grad)] = 0.
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
    pred, target = NaNMask.apply(pred), NaNMask.apply(target)
    is_real = torch.isfinite(pred) & torch.isfinite(target)

    if is_real.all():
        return f.l1_loss(pred, target, reduction=reduction)
    elif not is_real.any():
        return ZeroGrad.apply(pred, target)
    elif reduction == 'mean':
        return torch.nanmean(f.l1_loss(pred, target, reduction='none'))
    elif reduction == 'sum':
        return torch.nansum(f.l1_loss(pred, target, reduction='none'))
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
    pred, target = NaNMask.apply(pred), NaNMask.apply(target)
    is_real = torch.isfinite(pred) & torch.isfinite(target)

    if is_real.all():
        return f.mse_loss(pred, target, reduction=reduction)
    elif not is_real.any():
        return ZeroGrad.apply(pred, target)
    elif reduction == 'mean':
        return torch.nanmean(f.mse_loss(pred, target, reduction='none'))
    elif reduction == 'sum':
        return torch.nansum(f.mse_loss(pred, target, reduction='none'))
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
    pred, target = NaNMask.apply(pred), NaNMask.apply(target)
    is_real = torch.isfinite(pred) & torch.isfinite(target)

    if is_real.all():
        return f.smooth_l1_loss(pred, target, reduction=reduction)
    elif not is_real.any():
        return ZeroGrad.apply(pred, target)
    elif reduction == 'mean':
        return torch.nanmean(f.smooth_l1_loss(pred, target, reduction='none'))
    elif reduction == 'sum':
        return torch.nansum(f.smooth_l1_loss(pred, target, reduction='none'))
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
