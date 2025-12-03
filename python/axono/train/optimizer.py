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
from typing import Any, Dict, List, Tuple

from ..core import Tensor


class Optimizer:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr
        self._state: Dict[str, Any] = {}

    def step(self):
        """Update parameters using gradients"""
        raise NotImplementedError

    def zero_grad(self):
        """Zero out parameter gradients"""
        for param in self.params:
            if param.grad is not None:
                param.grad.fill_zero()


class SGD(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay

        if momentum > 0:
            self._state["momentum_buffer"] = [Tensor.zeros_like(p) for p in params]

    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param

            if self.momentum > 0:
                buf = self._state["momentum_buffer"][i]
                buf = buf * self.momentum + grad
                self._state["momentum_buffer"][i] = buf
                grad = buf

            param -= self.lr * grad


class Adam(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self._state["step"] = 0
        self._state["exp_avg"] = [Tensor.zeros_like(p) for p in params]
        self._state["exp_avg_sq"] = [Tensor.zeros_like(p) for p in params]

    def step(self):
        self._state["step"] += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param

            beta1, beta2 = self.betas
            exp_avg = self._state["exp_avg"][i]
            exp_avg_sq = self._state["exp_avg_sq"][i]

            # Update biased first moment estimate
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad

            # Update biased second raw moment estimate
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

            # Store updated moments
            self._state["exp_avg"][i] = exp_avg
            self._state["exp_avg_sq"][i] = exp_avg_sq

            # Bias correction
            bias_correction1 = 1 - beta1 ** self._state["step"]
            bias_correction2 = 1 - beta2 ** self._state["step"]

            # Compute bias-corrected moments
            exp_avg_corrected = exp_avg / bias_correction1
            exp_avg_sq_corrected = exp_avg_sq / bias_correction2

            # Update parameters
            param -= (
                self.lr * exp_avg_corrected / (exp_avg_sq_corrected.sqrt() + self.eps)
            )
