import torch
import torch.nn.functional as f

__all__ = ['L1Loss', 'MSELoss']


class L1Loss(torch.nn.Module):
    '''
    This is an L1 loss that ignores NaN values.
    '''
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        is_real = ~(torch.isnan(prediction) | torch.isnan(target))
        # We apply the mask AFTER taking the loss so that predictions and
        # targets can broadcast.
        return f.l1_loss(prediction, target, reduction='none')[is_real].mean()


class MSELoss(torch.nn.Module):
    '''
    This is a mean-squared error loss that ignores NaN values.
    '''
    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        is_real = ~(torch.isnan(prediction) | torch.isnan(target))
        # We apply the mask AFTER taking the loss so that predictions and
        # targets can broadcast.
        return f.mse_loss(prediction, target, reduction='none')[is_real].mean()
