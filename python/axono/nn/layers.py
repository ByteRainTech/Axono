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
from typing import Optional, Tuple, Union

import numpy as np

from ..core import Tensor
from .module import Module


class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化权重
        scale = np.sqrt(2.0 / (in_channels * kernel_size[0] * kernel_size[1]))
        weight_data = np.random.normal(
            0, scale, (out_channels, in_channels, kernel_size[0], kernel_size[1])
        )

        self._parameters["weight"] = Tensor.from_numpy(weight_data).to(device)

        if bias:
            bias_data = np.zeros(out_channels)
            self._parameters["bias"] = Tensor.from_numpy(bias_data).to(device)
        else:
            self._parameters["bias"] = None

    def forward(self, x: Tensor) -> Tensor:
        # 使用CUDA kernel或优化的CPU实现
        from ..core.ops import conv2d

        return conv2d(
            x,
            self._parameters["weight"],
            self._parameters.get("bias"),
            self.stride,
            self.padding,
        )


class Linear(Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        # 初始化权重
        scale = np.sqrt(2.0 / in_features)
        weight_data = np.random.normal(0, scale, (out_features, in_features))

        self._parameters["weight"] = Tensor.from_numpy(weight_data).to(device)

        if bias:
            bias_data = np.zeros(out_features)
            self._parameters["bias"] = Tensor.from_numpy(bias_data).to(device)
        else:
            self._parameters["bias"] = None

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self._parameters["weight"].T
        if self._parameters["bias"] is not None:
            output = output + self._parameters["bias"]
        return output


class BatchNorm2d(Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self._parameters["weight"] = Tensor.from_numpy(np.ones(num_features)).to(device)
        self._parameters["bias"] = Tensor.from_numpy(np.zeros(num_features)).to(device)

        # 运行时统计量
        self.register_buffer(
            "running_mean", Tensor.from_numpy(np.zeros(num_features)).to(device)
        )
        self.register_buffer(
            "running_var", Tensor.from_numpy(np.ones(num_features)).to(device)
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.fill_zero()
        self.running_var.fill(1)
        self._parameters["weight"].fill(1)
        self._parameters["bias"].fill_zero()

    def forward(self, x: Tensor) -> Tensor:
        if self.is_training:
            # 计算批次统计量
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)

            # 更新运行时统计量
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # 标准化
        x_normalized = (x - mean[None, :, None, None]) / (
            np.sqrt(var[None, :, None, None] + self.eps)
        )

        # 缩放和平移
        return (
            self._parameters["weight"][None, :, None, None] * x_normalized
            + self._parameters["bias"][None, :, None, None]
        )


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        from ..core.ops import relu

        return relu(x)


class MaxPool2d(Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        from ..core.ops import max_pool2d

        return max_pool2d(x, self.kernel_size, self.stride, self.padding)


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.is_training:
            mask = Tensor.from_numpy(
                (np.random.rand(*x.shape) > self.p).astype(np.float32)
            ).to(x.device)
            return x * mask / (1 - self.p)
        return x
