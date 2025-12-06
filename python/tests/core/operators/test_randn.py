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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from axono.core import DataType, Tensor

device = os.getenv("axono_default_device", "cpu")


class TestTensorRandn(unittest.TestCase):
    """测试 Tensor.randn 正态分布随机数生成"""

    def test_randn_basic_shape_dtype(self):
        """测试基本形状和数据类型生成"""
        # 测试不同形状
        shapes = [[3], [2, 4], [1, 5, 3]]
        dtypes = [DataType.FLOAT32, DataType.FLOAT64]

        for shape in shapes:
            for dtype in dtypes:
                tensor = Tensor.randn(shape=shape, dtype=dtype, device=device)
                self.assertEqual(
                    tensor.shape,
                    shape,
                    f"形状不匹配: 预期 {shape}, 实际 {tensor.shape}",
                )
                self.assertEqual(
                    tensor.dtype,
                    dtype,
                    f"数据类型不匹配: 预期 {dtype}, 实际 {tensor.dtype}",
                )
                self.assertEqual(
                    tensor.device,
                    device,
                    f"设备不匹配: 预期 {device}, 实际 {tensor.device}",
                )
                self.assertEqual(
                    tensor.num_elements,
                    np.prod(shape),
                    f"元素数量错误: 预期 {np.prod(shape)}, 实际 {tensor.num_elements}",
                )

    def test_randn_distribution_properties(self):
        """测试正态分布的统计特性（均值和标准差）"""
        shape = [10000, 10000]
        mean = 2.5
        stddev = 1.3

        tensor = Tensor.randn(shape=shape, mean=mean, stddev=stddev, device=device)
        data = tensor.to_numpy()

        # 计算实际均值和标准差（允许一定误差范围）
        actual_mean = np.mean(data)
        actual_std = np.std(data)

        # 均值误差允许在 0.05 以内，标准差误差允许在 0.05 以内
        self.assertAlmostEqual(
            actual_mean,
            mean,
            delta=0.05,
            msg=f"均值不符合预期: 预期 {mean}, 实际 {actual_mean}",
        )
        self.assertAlmostEqual(
            actual_std,
            stddev,
            delta=0.05,
            msg=f"标准差不符合预期: 预期 {stddev}, 实际 {actual_std}",
        )

    def test_randn_device_compatibility(self):
        """测试不同设备（CPU/CUDA）上的生成功能"""
        cpu_tensor = Tensor.randn(shape=[3, 3], device="cpu")
        self.assertEqual(cpu_tensor.device, "cpu")
        cpu_data = cpu_tensor.to_numpy()  # 验证CPU数据可访问

        if "cuda" in device:
            cuda_tensor = Tensor.randn(shape=[5, 5], device=device)
            self.assertTrue(cuda_tensor.is_cuda())
            cuda_data = cuda_tensor.to_numpy()  # 验证CUDA数据可拷贝到CPU
            self.assertEqual(cuda_data.shape, (5, 5))

    def test_randn_randomness(self):
        """测试每次调用生成不同的随机数"""
        tensor1 = Tensor.randn(shape=[100, 100], device=device)
        tensor2 = Tensor.randn(shape=[100, 100], device=device)

        data1 = tensor1.to_numpy()
        data2 = tensor2.to_numpy()

        self.assertFalse(np.array_equal(data1, data2), "两次生成的随机数完全相同")

    def test_randn_edge_cases(self):
        """测试边界情况（空张量、单元素）"""
        # 空张量
        empty_tensor = Tensor.randn(shape=[], device=device)
        self.assertEqual(empty_tensor.num_elements, 0)

        # 单元素张量
        single_tensor = Tensor.randn(shape=[1], mean=0, stddev=1, device=device)
        single_data = single_tensor.to_numpy()
        self.assertEqual(single_data.shape, (1,))

        # 大形状张量（内存分配验证）
        large_tensor = Tensor.randn(shape=[1024, 1024, 16], device=device)
        self.assertEqual(large_tensor.num_elements, 1024 * 1024 * 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
