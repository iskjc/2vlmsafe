from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class HookState:
    last_input: Optional[tuple[torch.Tensor, ...]] = None
    last_output: Optional[torch.Tensor] = None


class ModuleHook:
    """Utility hook to capture a module's latest input/output tensors."""

    def __init__(self, module: nn.Module, detach: bool = True):
        self.module = module
        self.detach = detach
        self.state = HookState()
        self._handle: Optional[torch.utils.hooks.RemovableHandle] = None

    def _hook_fn(self, _module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if self.detach:
            self.state.last_input = tuple(x.detach() if torch.is_tensor(x) else x for x in inputs)
            self.state.last_output = output.detach() if torch.is_tensor(output) else output
        else:
            self.state.last_input = inputs
            self.state.last_output = output

    def attach(self) -> None:
        if self._handle is not None:
            raise RuntimeError("Hook already attached")
        self._handle = self.module.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self) -> None:
        self.state = HookState()

    def __enter__(self) -> "ModuleHook":
        self.attach()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()
