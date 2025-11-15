#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include <algorithm> // for std::max
#include <cstddef>

namespace axono {
namespace compute {
namespace cpu {
namespace ops {

// ReLU 激活函数内核：f(x) = max(0, x)
template <typename T>
AXONO_FORCE_INLINE void ReluKernel(const T *input, T *output,
                                   size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    output[i] = std::max(static_cast<T>(0), input[i]);
  }
}

// 原地 ReLU 激活函数
template <typename T>
AXONO_FORCE_INLINE void ReluInplaceKernel(T *data, size_t num_elements) {
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = std::max(static_cast<T>(0), data[i]);
  }
}

// 类型分派的 ReLU
AXONO_FORCE_INLINE core::Status DispatchRelu(const core::Tensor &input, core::Tensor &output) {
  auto num_elements = input.num_elements();

  // 检查形状一致性
  if (!input.IsSameShape(output)) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (input.dtype() != output.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 根据数据类型选择内核
  switch (input.dtype()) {
  case core::DataType::FLOAT32:
    ReluKernel(input.data<float>(), output.data<float>(), num_elements);
    break;
  case core::DataType::FLOAT64:
    ReluKernel(input.data<double>(), output.data<double>(), num_elements);
    break;
  case core::DataType::INT32:
    ReluKernel(input.data<int32_t>(), output.data<int32_t>(), num_elements);
    break;
  default:
    return core::Status::UNSUPPORTED_TYPE;
  }

  return core::Status::OK;
}

// 类型分派的原地 ReLU
AXONO_FORCE_INLINE core::Status DispatchReluInplace(core::Tensor &tensor) {
  auto num_elements = tensor.num_elements();

  // 根据数据类型选择内核
  switch (tensor.dtype()) {
  case core::DataType::FLOAT32:
    ReluInplaceKernel(tensor.data<float>(), num_elements);
    break;
  case core::DataType::FLOAT64:
    ReluInplaceKernel(tensor.data<double>(), num_elements);
    break;
  case core::DataType::INT32:
    ReluInplaceKernel(tensor.data<int32_t>(), num_elements);
    break;
  default:
    return core::Status::UNSUPPORTED_TYPE;
  }

  return core::Status::OK;
}

} // namespace kernel
} // namespace cpu
} // namespace compute
} // namespace axono
