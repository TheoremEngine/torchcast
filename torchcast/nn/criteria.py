from typing import Tuple

import torch
import torch.nn.functional as f

__all__ = ['L1Loss', 'MSELoss', 'SoftL1Loss', 'SoftMSELoss']


class L1Loss(torch.nn.Module):
    '''
    This is an L1 loss that ignores NaN values.
    '''
    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        is_real = ~(torch.isnan(pred) | torch.isnan(target))
        # We apply the mask AFTER taking the loss so that predictions and
        # targets can broadcast.
        if is_real.all():
            return f.l1_loss(pred, target, reduction='mean')
        else:
            return f.l1_loss(pred, target, reduction='none')[is_real].mean()


class MSELoss(torch.nn.Module):
    '''
    This is a mean-squared error loss that ignores NaN values.
    '''
    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        is_real = ~(torch.isnan(pred) | torch.isnan(target))
        # We apply the mask AFTER taking the loss so that predictions and
        # targets can broadcast.
        if is_real.all():
            return f.mse_loss(pred, target, reduction='mean')
        else:
            return f.mse_loss(pred, target, reduction='none')[is_real].mean()


class SoftL1Loss(torch.nn.Module):
    '''
    This is an L1 loss that ignores when inputs go out of range in the same
    direction as the target. For example, if the range of inputs is [0, 1], and
    the target is 1, and the prediction is 2, this will produce a loss of 0.
    However, if the target was 0, this would produce a loss of 2.
    '''
    def __init__(self, target_range: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.target_min, self.target_max = target_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        target = torch.where(target < self.target_max, target, pred)
        target = torch.where(target > self.target_min, target, pred)
        is_real = ~(torch.isnan(pred) | torch.isnan(target))
        # We apply the mask AFTER taking the loss so that predictions and
        # targets can broadcast.
        if is_real.all():
            return f.l1_loss(pred, target, reduction='mean')
        else:
            return f.l1_loss(pred, target, reduction='none')[is_real].mean()


class SoftMSELoss(torch.nn.Module):
    '''
    This is a mean-squared error loss that ignores when inputs go out of range
    in the same direction as the target. For example, if the range of inputs is
    [0, 1], and the target is 1, and the prediction is 2, this will produce a
    loss of 0. However, if the target was 0, this would produce a loss of 4.
    '''
    def __init__(self, target_range: Tuple[float, float] = (0, 1)):
        super().__init__()
        self.target_min, self.target_max = target_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) \
            -> torch.Tensor:
        target = torch.where(target < self.target_max, target, pred)
        target = torch.where(target > self.target_min, target, pred)
        is_real = ~(torch.isnan(pred) | torch.isnan(target))
        # We apply the mask AFTER taking the loss so that predictions and
        # targets can broadcast.
        if is_real.all():
            return f.mse_loss(pred, target, reduction='mean')
        else:
            return f.mse_loss(pred, target, reduction='none')[is_real].mean()
