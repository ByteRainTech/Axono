# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod
from typing import List

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

    def to(self, device: str) -> "Module":
        for name, param in self._parameters.items():
            self._parameters[name] = param.to(device)
        return self
