from typing import Tuple

import torch
import torch.nn.functional as f

__all__ = [
    'l1_loss', 'L1Loss', 'mse_loss', 'MSELoss', 'soft_l1_loss', 'SoftL1Loss',
    'soft_mse_loss', 'SoftMSELoss'
]


def _broadcast_mask_to_shape(mask: torch.Tensor, shape: torch.Size) \
        -> torch.Tensor:
    '''
    This is a convenience function for broadcasting a mask to a specified
    shape.

    Args:
        mask (:class:`torch.Tensor`): The mask to broadcast.
        shape (:class:`torch.Size`): The shape to be broadcast to.
    '''
    if mask.ndim != len(shape):
        raise ValueError(f'Shapes do not match: {mask.shape} vs {shape}')

    for s, (d_m, d_s) in enumerate(zip(mask.shape, shape)):
        if d_m == d_s:
            continue
        elif d_s == 1:
            mask = mask.any(s, keepdim=True)
        else:
            raise ValueError(f'Shapes do not match: {mask.shape} vs {shape}')

    return mask


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    This is an L1 loss that ignores NaN values.
    '''
    is_real = ~(torch.isnan(pred) | torch.isnan(target))

    if is_real.all():
        return f.l1_loss(pred, target, reduction='mean')

    pred_mask = _broadcast_mask_to_shape(is_real, pred.shape)
    target_mask = _broadcast_mask_to_shape(is_real, target.shape)

    return f.l1_loss(pred[pred_mask], target[target_mask], reduction='mean')


class L1Loss(torch.nn.Module):
    '''
    This is an L1 loss that ignores NaN values.
    '''
    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        return l1_loss(pred, target)


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    '''
    This is an MSE loss that ignores NaN values.
    '''
    is_real = ~(torch.isnan(pred) | torch.isnan(target))

    if is_real.all():
        return f.mse_loss(pred, target, reduction='mean')

    pred_mask = _broadcast_mask_to_shape(is_real, pred.shape)
    target_mask = _broadcast_mask_to_shape(is_real, target.shape)

    return f.mse_loss(pred[pred_mask], target[target_mask], reduction='mean')


class MSELoss(torch.nn.Module):
    '''
    This is a mean-squared error loss that ignores NaN values.
    '''
    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        return mse_loss(pred, target)


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
