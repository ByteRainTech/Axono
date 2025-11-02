#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include <cstddef>

namespace axono {
namespace compute {
namespace cpu {
namespace kernel {

// 基础的矩阵乘法内核
template <typename T>
AXONO_FORCE_INLINE void MatMulKernel(const T *a, const T *b, T *result,
                                     size_t m, size_t n, size_t k) {
  // C = A * B
  // A: m x k, B: k x n, C: m x n
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      T sum = 0;
      for (size_t l = 0; l < k; ++l) {
        sum += a[i * k + l] * b[l * n + j];
      }
      result[i * n + j] = sum;
    }
  }
}

// 优化的矩阵乘法内核（循环重排以改善缓存局部性）
template <typename T>
AXONO_FORCE_INLINE void MatMulOptimizedKernel(const T *a, const T *b, T *result,
                                              size_t m, size_t n, size_t k) {
  // 初始化结果为0
  for (size_t i = 0; i < m * n; ++i) {
    result[i] = 0;
  }

  // 优化版本：循环重排以改善缓存性能
  for (size_t l = 0; l < k; ++l) {
    for (size_t i = 0; i < m; ++i) {
      T a_val = a[i * k + l];
      for (size_t j = 0; j < n; ++j) {
        result[i * n + j] += a_val * b[l * n + j];
      }
    }
  }
}

// 类型分派的矩阵乘法
AXONO_FORCE_INLINE Status DispatchMatMul(const Tensor &a, const Tensor &b,
                                         Tensor &result) {
  auto a_shape = a.shape();
  auto b_shape = b.shape();

  size_t m = a_shape[0];
  size_t k = a_shape[1];
  size_t n = b_shape[1];

  // 检查形状兼容性
  if (a_shape[1] != b_shape[0]) {
    return Status::SHAPE_MISMATCH;
  }

  // 根据数据类型选择内核
  switch (a.dtype()) {
  case DataType::FLOAT32:
    MatMulOptimizedKernel(a.data<float>(), b.data<float>(),
                          result.data<float>(), m, n, k);
    break;
  case DataType::FLOAT64:
    MatMulOptimizedKernel(a.data<double>(), b.data<double>(),
                          result.data<double>(), m, n, k);
    break;
  case DataType::INT32:
    MatMulOptimizedKernel(a.data<int32_t>(), b.data<int32_t>(),
                          result.data<int32_t>(), m, n, k);
    break;
  default:
    return Status::UNSUPPORTED_TYPE;
  }

  return Status::OK;
}

} // namespace kernel
} // namespace cpu
} // namespace compute
} // namespace axono
