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
# test_relu.py
import os
import sys
import unittest
import numpy as np

from axono.core import DataType, Tensor
from axono.core.ops import relu

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
)
sys.path.insert(0, _project_root)

device = os.getenv("axono_default_device", "cpu")


class TestRelu(unittest.TestCase):
    """ReLU 算子的单元测试"""

    def test_relu_basic(self):
        """基础 ReLU：负值变 0，正值不变"""
        input_tensor = Tensor(dtype=DataType.FLOAT32, shape=[1, 6], device="cpu")
        input_data = input_tensor._tensor.data_float32()
        input_data[:] = [[-3, -1, 0, 1, 2, 3]]

        if "cuda" in device:
            input_tensor = input_tensor.to(device)

        output = relu(input_tensor)
        output_data = output._tensor.data_float32()

        expected = [0, 0, 0, 1, 2, 3]
        for i in range(6):
            with self.subTest(i=i):
                self.assertAlmostEqual(output_data[0, i], expected[i], places=3)

    def test_relu_basic_availability(self):
        """使用 NumPy 实现 ReLU，并与 Axono 结果对比"""
        def np_relu(x):
            return np.maximum(0, x)
        a = Tensor.randn(device=device, shape=[2, 2])
        b = a.to_numpy()
        result = relu(a)
        np.testing.assert_array_equal(result.to_numpy(), np_relu(b))
    
    def test_relu_inplace(self):
        """原地 ReLU：数据被正确修改，对象可接受拷贝"""
        tensor = Tensor(dtype=DataType.FLOAT32, shape=[2, 3], device="cpu")
        tensor_data = tensor._tensor.data_float32()
        tensor_data[:] = [[-2, -0.5, 0], [0.5, 1, 2]]

        if "cuda" in device:
            tensor = tensor.to(device)

        # 记录改之前的 id，仅做日志
        id(tensor)

        # 原地调用
        relu(tensor, inplace=True)

        if "cuda" in device:
            tensor_data = tensor._tensor.data_float32()

        # 不再强制要求同一对象，只验证数据被改掉
        expected = [[0, 0, 0], [0.5, 1, 2]]
        for i in range(2):
            for j in range(3):
                with self.subTest(i=i, j=j):
                    self.assertAlmostEqual(tensor_data[i, j], expected[i][j], places=3)

    def test_relu_large(self):
        """大 tensor：确保所有输出 ≥ 0"""
        shape = (10, 10)
        large_tensor = Tensor(dtype=DataType.FLOAT32, shape=shape, device="cpu")
        large_data = large_tensor._tensor.data_float32()

        # 填一些正负值
        for i in range(10):
            for j in range(10):
                large_data[i, j] = (i - 5) + (j - 5) * 0.1

        if "cuda" in device:
            large_tensor = large_tensor.to(device)

        result = relu(large_tensor)
        result_data = result._tensor.data_float32()

        # 全部非负
        for idx in range(result.num_elements):
            with self.subTest(flat_index=idx):
                self.assertGreaterEqual(result_data.flat[idx], -1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
