import torch

__all__ = ['Normalize', 'Transform']


class Transform:
    '''
    Transforms are data transformations applied to data series during
    dataloading, analogous to the members of :class:`torchvision.transforms`.
    Transforms are expected to take multiple :class:`torch.Tensor` as inputs,
    and may return an arbitrary object.
    '''
    def __call__(self, *series):
        raise NotImplementedError()


class Normalize(Transform):
    '''
    Normalizes input tensors by subtracting the mean and dividing by the
    standard deviation. The transform expects to receive as many values for
    mean and standard deviation as there are series to be transformed. If
    multiple series are being transformed, and one or more of them should not
    be normalized - for example, if they are class labels - then substitute
    None for the mean in that case.
    '''
    def __init__(self, means, stds):
        super().__init__()
        self.means = means
        self.stds = stds

    def __call__(self, *series):
        return tuple(
            x if (m is None) else ((x.to(torch.float32) - m) / s)
            for x, m, s in zip(series, self.means, self.stds)
        )
