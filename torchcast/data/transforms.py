import torch

__all__ = ['Normalize', 'Transform']


class Transform:
    def __call__(self, *series):
        raise NotImplementedError()


class Normalize(Transform):
    def __init__(self, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds

    def __call__(self, *series):
        return tuple(
            x if (m is None) else ((x.to(torch.float32) - m) / s)
            for x, m, s in zip(series, self.means, self.stds)
        )
