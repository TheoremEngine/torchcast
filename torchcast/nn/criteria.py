from typing import Iterator, Tuple

import torch
import torch.nn.functional as f

__all__ = [
    'l1_loss', 'L1Loss', 'mse_loss', 'MSELoss', 'soft_l1_loss', 'SoftL1Loss',
    'soft_mse_loss', 'SoftMSELoss'
]


class ZeroGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *tensors: torch.Tensor) -> torch.Tensor:
        ctx.shapes = [x.shape for x in tensors]
        return torch.zeros((), device=tensors[0].device)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Iterator[torch.Tensor]:
        return tuple(torch.zeros(s, device=grad.device) for s in ctx.shapes)


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
    is_real = ~(torch.isnan(pred) | torch.isnan(target))

    if is_real.all():
        return f.l1_loss(pred, target, reduction=reduction)
    elif not is_real.any():
        return ZeroGrad.apply(pred, target)
    elif reduction == 'mean':
        return torch.nanmean((pred - target).abs())
    elif reduction == 'sum':
        return torch.nansum((pred - target).abs())
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
    is_real = ~(torch.isnan(pred) | torch.isnan(target))

    if is_real.all():
        return f.mse_loss(pred, target, reduction=reduction)
    elif not is_real.any():
        return ZeroGrad.apply(pred, target)
    elif reduction == 'mean':
        return torch.nanmean((pred - target)**2)
    elif reduction == 'sum':
        return torch.nansum((pred - target)**2)
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


def soft_l1_loss(pred: torch.Tensor, target: torch.Tensor,
                 target_range: Tuple[float, float] = (0., 1.)):
    '''
    This is an L1 loss that ignores when inputs go out of range in the same
    direction as the target. For example, if the range of inputs is [0, 1], and
    the target is 1, and the prediction is 2, this will produce a loss of 0.
    However, if the target was 0, this would produce a loss of 2.
    '''
    target_min, target_max = target_range
    target = torch.where(
        torch.minimum(target, pred) >= target_max,
        pred,
        target
    )
    target = torch.where(
        torch.maximum(target, pred) <= target_min,
        pred,
        target
    )
    return l1_loss(pred, target)


class SoftL1Loss(torch.nn.Module):
    '''
    This is an L1 loss that ignores when inputs go out of range in the same
    direction as the target. For example, if the range of inputs is [0, 1], and
    the target is 1, and the prediction is 2, this will produce a loss of 0.
    However, if the target was 0, this would produce a loss of 2.
    '''
    def __init__(self, target_range: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.target_range = target_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        return soft_l1_loss(pred, target, self.target_range)


def soft_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                  target_range: Tuple[float, float] = (0., 1.)):
    '''
    This is an MSE loss that ignores when inputs go out of range in the same
    direction as the target. For example, if the range of inputs is [0, 1], and
    the target is 1, and the prediction is 2, this will produce a loss of 0.
    However, if the target was 0, this would produce a loss of 4.
    '''
    target_min, target_max = target_range
    target = torch.where(
        torch.minimum(target, pred) >= target_max,
        pred,
        target
    )
    target = torch.where(
        torch.maximum(target, pred) <= target_min,
        pred,
        target
    )
    return mse_loss(pred, target)


class SoftMSELoss(torch.nn.Module):
    '''
    This is a mean-squared error loss that ignores when inputs go out of range
    in the same direction as the target. For example, if the range of inputs is
    [0, 1], and the target is 1, and the prediction is 2, this will produce a
    loss of 0. However, if the target was 0, this would produce a loss of 4.
    '''
    def __init__(self, target_range: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.target_range = target_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        return soft_mse_loss(pred, target, self.target_range)
