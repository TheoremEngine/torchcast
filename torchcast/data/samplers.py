import torch

__all__ = ['InfiniteSampler']


class InfiniteSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size: int):
        self.size = size

    def __iter__(self):
        while True:
            # Usually I would do yield from torch.randperm, but that won't work
            # for the cerebral dataset because the permuted indices would be
            # too big to fit in memory!
            yield torch.randint(self.size, ()).item()
