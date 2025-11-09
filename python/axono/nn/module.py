# 基类定义
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np

from ..core import Tensor


class Module(ABC):
    def __init__(self):
        self._parameters = {}
        self._is_training = True

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def train(self, mode: bool = True):
        self._is_training = mode
        return self

    def eval(self):
        return self.train(False)

    @property
    def is_training(self) -> bool:
        return self._is_training

    def parameters(self) -> List[Tensor]:
        return list(self._parameters.values())

    def to(self, device: str) -> 'Module':
        for name, param in self._parameters.items():
            self._parameters[name] = param.to(device)
        return self