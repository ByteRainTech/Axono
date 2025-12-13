# python/axono/nn/module.py
from typing import Dict, List
from libaxono import Module as _Module
from libaxono import Tensor as _Tensor
from ..core import Tensor

class Module:
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._cpp_module = _Module()
        self._is_training = True
        self._name = self.__class__.__name__

    def add_weight(self, name: str, tensor: Tensor) -> None:
        self._parameters[name] = tensor
        self._cpp_module.add_weight(name, tensor._tensor)

    def parameters(self) -> Dict[str, Tensor]:
        for k, v in self._parameters.items():
            if type(v) == _Tensor:
                self._parameters[k] = Tensor.from_raw(v)
        return dict(self._parameters)

    def train(self, mode: bool = True) -> "Module":
        self._is_training = mode
        return self
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        init_args = []
        if hasattr(self, '_init_args'):
            init_args = [f"{k}={v}" for k, v in self._init_args.items()]

        if not hasattr(self, '_modules') or not self._modules:
            if init_args:
                return f"{cls_name}({', '.join(init_args)})"
            else:
                return f"{cls_name}()"

        lines = [f"{cls_name}("]
        indent = "  "
        if init_args:
            lines.append(f"{indent}{', '.join(init_args)},")

        for name, module in self._modules.items():
            submodule_repr = repr(module).replace("\n", f"\n{indent}")
            lines.append(f"{indent}({name}): {submodule_repr}")
        
        lines.append(")")
        return "\n".join(lines)
