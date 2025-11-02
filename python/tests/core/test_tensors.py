import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import unittest

import numpy as np
from axono.core import DataType, Tensor


class TestTensor(unittest.TestCase):
    def test_tensor_creation(self):
        """测试 Tensor 创建"""

        shapes = [[1], [2, 3], [1, 1, 1], [2, 2, 2, 2]]
        for shape in shapes:
            tensor = Tensor(shape=shape)
            self.assertEqual(tensor.shape, shape)

    def test_tensor_data_types(self):
        """测试不同数据类型"""
        dtypes = [
            DataType.FLOAT32,
            DataType.FLOAT64,
            DataType.INT32,
            DataType.INT64,
            DataType.INT16,
            DataType.INT8,
            DataType.BOOLEAN,
        ]

        for dtype in dtypes:
            tensor = Tensor(dtype=dtype, shape=[2, 2])
            self.assertEqual(tensor.dtype, dtype)

    def test_tensor_fill(self):
        """测试 Tensor 填充"""
        tensor_float = Tensor(dtype=DataType.FLOAT32, shape=[1, 3])
        tensor_float.fill(3.14)
        data_float = tensor_float.to_numpy()
        self.assertTrue(np.allclose(data_float, 3.14))

        # 测试 INT32 填充
        tensor_int = Tensor(dtype=DataType.INT32, shape=[2, 2])
        tensor_int.fill(42)
        data_int = tensor_int.to_numpy()
        self.assertTrue(np.all(data_int == 42))

    def test_tensor_fill_zero(self):
        """测试零填充"""

        tensor = Tensor(dtype=DataType.FLOAT32, shape=[3, 3])
        tensor.fill_zero()
        data = tensor.to_numpy()
        self.assertTrue(np.all(data == 0))

    def test_tensor_copy(self):
        """测试 Tensor 拷贝"""

        src_tensor = Tensor(dtype=DataType.FLOAT32, shape=[2, 2])
        src_data = src_tensor._tensor.data_float32()
        src_data[:] = [[1.1, 2.2], [3.3, 4.4]]

        dst_tensor = Tensor(dtype=DataType.FLOAT32, shape=[2, 2])
        dst_tensor.copy_from(src_tensor)

        src_numpy = src_tensor.to_numpy()
        dst_numpy = dst_tensor.to_numpy()
        self.assertTrue(np.allclose(src_numpy, dst_numpy))

    def test_tensor_reshape(self):
        """测试 Tensor 重塑"""

        tensor = Tensor(dtype=DataType.FLOAT32, shape=[2, 3])
        tensor.fill(1.0)
        tensor.reshape([3, 2])
        self.assertEqual(tensor.shape, [3, 2])
        self.assertEqual(tensor.num_elements, 6)

    def test_tensor_properties(self):
        """测试 Tensor 属性"""
        tensor = Tensor(dtype=DataType.INT32, shape=[4, 5])
        self.assertEqual(tensor.shape, [4, 5])
        self.assertEqual(tensor.dtype, DataType.INT32)
        self.assertEqual(tensor.ndim, 2)
        self.assertEqual(tensor.num_elements, 20)
        self.assertEqual(tensor.num_bytes, 20 * 4)

    def test_tensor_large_array(self):
        tensor = Tensor(dtype=DataType.FLOAT32, shape=[100, 100])  # 10,000 元素
        tensor.fill(2.5)

        data = tensor.to_numpy()
        self.assertTrue(np.allclose(data, 2.5))
        self.assertEqual(data.shape, (100, 100))

    def test_tensor_edge_cases(self):
        """测试边界情况"""
        empty_tensor = Tensor(dtype=DataType.FLOAT32, shape=[])
        self.assertEqual(empty_tensor.num_elements, 0)

        single_tensor = Tensor(dtype=DataType.INT32, shape=[1, 1, 1])
        single_tensor.fill(99)
        single_data = single_tensor.to_numpy()
        self.assertEqual(single_data.flatten()[0], 99)


class TestTensorFactoryMethods(unittest.TestCase):
    """测试 Tensor 工厂方法"""

    def test_create_like(self):
        """测试 create_like 方法"""
        original = Tensor(dtype=DataType.FLOAT64, shape=[3, 4])
        original.fill(7.7)

        copy = Tensor.create_like(original)
        self.assertEqual(copy.shape, original.shape)
        self.assertEqual(copy.dtype, original.dtype)
        self.assertEqual(copy.num_elements, original.num_elements)

    def test_zeros_ones_full(self):
        """测试便捷创建函数"""

        # 测试 zeros
        zeros_tensor = Tensor.zeros(shape=[2, 3], dtype=DataType.FLOAT32)
        zeros_data = zeros_tensor.to_numpy()
        self.assertTrue(np.all(zeros_data == 0))

        ones_tensor = Tensor.ones(shape=[1, 4], dtype=DataType.INT32)
        ones_data = ones_tensor.to_numpy()
        self.assertTrue(np.all(ones_data == 1))

        full_tensor = Tensor.full(shape=[2, 2], value=3.14, dtype=DataType.FLOAT32)
        full_data = full_tensor.to_numpy()
        self.assertTrue(np.allclose(full_data, 3.14))


if __name__ == "__main__":
    unittest.main()
