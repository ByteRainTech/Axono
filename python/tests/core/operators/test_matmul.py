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
import os
import sys
import unittest

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from axono.core import DataType, Tensor
from axono.core.operators import matmul

device = os.getenv("axono_default_device", "cpu")

class TestMatmul(unittest.TestCase):
    """矩阵乘法单元测试"""

    def test_matmul_basic(self):
        """测试基础矩阵乘法"""
        a = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = Tensor.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))

        result = a @ b

        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        actual = result.to_numpy()

        self.assertTrue(np.allclose(expected, actual))

    def test_matmul_different_sizes(self):
        """测试不同大小的矩阵乘法：3x2 * 2x4 = 3x4"""
        a = Tensor.from_numpy(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
        b = Tensor.from_numpy(np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32))

        result = matmul(a, b)

        expected = np.array(
            [[11, 14, 17, 20], [23, 30, 37, 44], [35, 46, 57, 68]],
            dtype=np.float32,
        )
        actual = result.to_numpy()

        self.assertTrue(np.allclose(expected, actual))
        self.assertEqual(result.shape, [3, 4])

    def test_matmul_large(self):
        """测试大矩阵乘法"""
        a = Tensor(DataType.FLOAT32, [10, 20], device="cpu")
        b = Tensor(DataType.FLOAT32, [20, 15], device="cpu")

        # 填充随机数据
        a_data = a._tensor.data_float32()
        b_data = b._tensor.data_float32()
        a_data[:] = np.random.rand(10, 20).astype(np.float32)
        b_data[:] = np.random.rand(20, 15).astype(np.float32)

        if "cuda" in device:
            a = a.to("cuda")
            b = b.to("cuda")

        result = matmul(a, b)

        self.assertEqual(result.shape, [10, 15])

    # def test_matmul_validation(self):
    #     """测试矩阵乘法形状验证：不兼容形状应抛出异常"""
    #     a = Tensor(DataType.FLOAT32, [2, 3])
    #     b = Tensor(DataType.FLOAT32, [4, 5])  # 不兼容的形状

    #     with self.assertRaises(Exception):
    #         matmul(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
