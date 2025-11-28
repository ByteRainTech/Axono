#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include <cstddef>
#include <cstring>

namespace axono {
namespace compute {
namespace cpu {
namespace operators {

// 广播加法
template <typename T>
AXONO_FORCE_INLINE void AddBroadcastKernel(const T* a, const T* b, T* out,
                        size_t M, size_t K) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            out[m * K + k] = a[m * K + k] + b[k];   // b 被广播
        }
    }
}

// 逐元素加法内核
template <typename T>
AXONO_FORCE_INLINE void AddKernel(const T *a, const T *b, T *result,
                                  size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = a[i] + b[i];
  }
}

// 标量加法内核
template <typename T>
AXONO_FORCE_INLINE void AddScalarKernel(const T *a, T scalar, T *result,
                                        size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    result[i] = a[i] + scalar;
  }
}

// 类型分派的加法
AXONO_FORCE_INLINE core::Status DispatchAdd(const core::Tensor &a, const core::Tensor &b,
                                      core::Tensor &result) {
  // 1. dtype 必须一致
  if (a.dtype() != b.dtype() || a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  auto a_shape = a.shape();
  auto b_shape = b.shape();

  /* 情形 1：完全同形状 → 逐元素 */
  if (a.IsSameShape(b) && a.IsSameShape(result)) {
    const size_t num = a.num_elements();
    switch (a.dtype()) {
    case core::DataType::FLOAT32:
      AddKernel(a.data<float>(), b.data<float>(), result.data<float>(), num);
      return core::Status::OK;
    case core::DataType::FLOAT64:
      AddKernel(a.data<double>(), b.data<double>(), result.data<double>(), num);
      return core::Status::OK;
    case core::DataType::INT32:
      AddKernel(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), num);
      return core::Status::OK;
    case core::DataType::INT64:
      AddKernel(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), num);
      return core::Status::OK;
    default:
      return core::Status::UNSUPPORTED_TYPE;
    }
  }

  /* 情形 2：[M,K] + [K] 广播 */
  if (a_shape.size() == 2 && b_shape.size() == 1 && result.shape() == a_shape &&
      a_shape[1] == b_shape[0]) {
    const size_t M = a_shape[0];
    const size_t K = a_shape[1];
    switch (a.dtype()) {
    case core::DataType::FLOAT32:
      AddBroadcastKernel(a.data<float>(), b.data<float>(), result.data<float>(), M, K);
      return core::Status::OK;
    case core::DataType::FLOAT64:
      AddBroadcastKernel(a.data<double>(), b.data<double>(), result.data<double>(), M, K);
      return core::Status::OK;
    case core::DataType::INT32:
      AddBroadcastKernel(a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), M, K);
      return core::Status::OK;
    case core::DataType::INT64:
      AddBroadcastKernel(a.data<int64_t>(), b.data<int64_t>(), result.data<int64_t>(), M, K);
      return core::Status::OK;
    default:
      return core::Status::UNSUPPORTED_TYPE;
    }
  }

  return core::Status::SHAPE_MISMATCH;
}

// 类型分派的标量加法
AXONO_FORCE_INLINE core::Status DispatchAddScalar(const core::Tensor &a, void *scalar,
                                            size_t scalar_size,
                                            core::Tensor &result) {
  auto num_elements = a.num_elements();

  // 检查形状一致性
  if (!a.IsSameShape(result)) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (a.dtype() != result.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 根据数据类型选择内核
  switch (a.dtype()) {
  case core::DataType::FLOAT32: {
    float scalar_value = 0.0f;
    if (scalar_size >= sizeof(float)) {
      memcpy(&scalar_value, scalar, sizeof(float));
    }
    AddScalarKernel(a.data<float>(), scalar_value, result.data<float>(),
                    num_elements);
    break;
  }
  case core::DataType::FLOAT64: {
    double scalar_value = 0.0;
    if (scalar_size >= sizeof(double)) {
      memcpy(&scalar_value, scalar, sizeof(double));
    }
    AddScalarKernel(a.data<double>(), scalar_value, result.data<double>(),
                    num_elements);
    break;
  }
  case core::DataType::INT32: {
    int32_t scalar_value = 0;
    if (scalar_size >= sizeof(int32_t)) {
      memcpy(&scalar_value, scalar, sizeof(int32_t));
    }
    AddScalarKernel(a.data<int32_t>(), scalar_value, result.data<int32_t>(),
                    num_elements);
    break;
  }
  default:
    return core::Status::UNSUPPORTED_TYPE;
  }

  return core::Status::OK;
}

} // namespace kernel
} // namespace cpu
} // namespace compute
} // namespace axono
