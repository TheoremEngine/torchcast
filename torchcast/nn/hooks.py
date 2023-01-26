from typing import Any

import torch

__all__ = ['max_norm_constraint']


class MaxNormConstraint:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, module: torch.nn.Module, x: Any):
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name: str):
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, MaxNormConstraint) and (hook.name == name):
                raise RuntimeError(
                    f'Cannot register two max_norm_constraint hooks on the '
                    f'same parameter {name}'
                )

        fn = MaxNormConstraint(name)
        weight = getattr(module, name)
        # Remove weight from parameter list
        del module._parameters[name]

        module.register_parameter(
            f'{name}_nc',
            torch.nn.Parameter(weight.data)
        )
        setattr(module, name, fn.compute_weight(module))

        module.register_forward_pre_hook(fn)

        return fn

    def compute_weight(self, module: torch.nn.Module) -> torch.Tensor:
        w = getattr(module, f'{self.name}_nc')
        norm = w.flatten().norm(p=2).clamp_(min=1.)
        return w / norm

    def remove(self, module: torch.nn.Module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[f'{self.name}_nc']
        setattr(module, self.name, torch.nn.Parameter(weight.data))


def max_norm_constraint(module: torch.nn.Module, name: str = 'weight') \
        -> torch.nn.Module:
    MaxNormConstraint.apply(module, name)
    return module
