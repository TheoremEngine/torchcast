import torch

__all__ = ['InfiniteSampler']


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size: int):
        self.size = size

    def __iter__(self):
        while True:
            yield torch.randint(self.size, ()).item()
