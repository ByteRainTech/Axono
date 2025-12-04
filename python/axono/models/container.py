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
from collections import OrderedDict
from typing import Dict, List, Optional

from ..core import Tensor
from ..nn import Module


class Sequential(Module):
    def __init__(self, layers: List[Module]):
        super().__init__()
        self.layers = layers

        # Register layers as submodules
        for i, layer in enumerate(layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def add_module(self, name: str, module: Optional[Module]):
        if module is not None:
            self._modules[name] = module


class ModuleList(Module):
    def __init__(self, modules: List[Module] = None):
        super().__init__()
        self._modules = OrderedDict()
        if modules is not None:
            for i, module in enumerate(modules):
                self.add_module(str(i), module)

    def append(self, module: Module):
        self.add_module(str(len(self)), module)
        return self

    def extend(self, modules: List[Module]):
        for module in modules:
            self.append(module)
        return self

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]


class ModuleDict(Module):
    def __init__(self, modules: Dict[str, Module] = None):
        super().__init__()
        self._modules = OrderedDict()
        if modules is not None:
            for key, module in modules.items():
                self.add_module(key, module)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module):
        self.add_module(key, module)

    def __delitem__(self, key: str):
        del self._modules[key]

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()
