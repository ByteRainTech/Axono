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
from axono.core.operators import add

device = os.getenv("axono_default_device", "cpu")

class TestAdd(unittest.TestCase):
    """逐元素加法单元测试"""

    def test_add_basic(self):
        """测试基础逐元素加法"""
        a = Tensor.from_numpy(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b = Tensor.from_numpy(np.array([[5, 6], [7, 8]], dtype=np.float32))

        result = a + b

        expected = np.array([[6, 8], [10, 12]], dtype=np.float32)
        actual = result.to_numpy()

        self.assertTrue(np.allclose(expected, actual))

    
    def test_add_large(self):
        """测试大 tensor 逐元素加法"""
        shape = [100, 200]
        a = Tensor(DataType.FLOAT32, shape, device="cpu")
        b = Tensor(DataType.FLOAT32, shape, device="cpu")

        # 填充随机数据
        a_data = a._tensor.data_float32()
        b_data = b._tensor.data_float32()
        a_data[:] = np.random.rand(*shape).astype(np.float32)
        b_data[:] = np.random.rand(*shape).astype(np.float32)

        if "cuda" in device:
            a = a.to(device)
            b = b.to(device)

        result = add(a, b)

        if "cuda" in device:
            a_data = a._tensor.data_float32()
            b_data = b._tensor.data_float32()

        self.assertEqual(result.shape, shape)

        # 抽查前 10 个元素
        expected_flat = a_data + b_data
        actual_flat = result._tensor.data_float32()
        self.assertTrue(np.allclose(expected_flat[:10], actual_flat[:10]))

    def test_add_validation(self):
        """测试不可广播形状应抛出异常"""
        a = Tensor(DataType.FLOAT32, [2, 3])
        b = Tensor(DataType.FLOAT32, [4, 5])  # 无法广播到同一形状

        with self.assertRaises(Exception):
            add(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
