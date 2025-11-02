#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include <cstring>

namespace axono {
namespace compute {
namespace cpu {
namespace kernel {

// Tensor 数据填充内核
template <typename T>
AXONO_FORCE_INLINE void TensorFillKernel(T *data, size_t num_elements,
                                         T value) {
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = value;
  }
}

// Tensor 零填充内核
template <typename T>
AXONO_FORCE_INLINE void TensorZeroKernel(T *data, size_t num_elements) {
  std::memset(data, 0, num_elements * sizeof(T));
}

// Tensor 拷贝内核
AXONO_FORCE_INLINE void TensorCopyKernel(void *dst, const void *src,
                                         size_t num_bytes) {
  std::memcpy(dst, src, num_bytes);
}

// 类型分派的填充函数
// 类型分派的填充函数
AXONO_FORCE_INLINE Status DispatchFill(Tensor &tensor, void *value,
                                       size_t value_size) {
  auto num_elements = tensor.num_elements();
  auto dtype = tensor.dtype();

  switch (dtype) {
  case DataType::INT8: {
    int8_t fill_value = 0;
    if (value_size >= sizeof(int8_t)) {
      std::memcpy(&fill_value, value, sizeof(int8_t));
    }
    TensorFillKernel(tensor.data<int8_t>(), num_elements, fill_value);
    break;
  }
  case DataType::INT16: {
    int16_t fill_value = 0;
    if (value_size >= sizeof(int16_t)) {
      std::memcpy(&fill_value, value, sizeof(int16_t));
    }
    TensorFillKernel(tensor.data<int16_t>(), num_elements, fill_value);
    break;
  }
  case DataType::INT32: {
    int32_t fill_value = 0;
    if (value_size >= sizeof(int32_t)) {
      std::memcpy(&fill_value, value, sizeof(int32_t));
    }
    TensorFillKernel(tensor.data<int32_t>(), num_elements, fill_value);
    break;
  }
  case DataType::INT64: {
    int64_t fill_value = 0;
    if (value_size >= sizeof(int64_t)) {
      std::memcpy(&fill_value, value, sizeof(int64_t));
    }
    TensorFillKernel(tensor.data<int64_t>(), num_elements, fill_value);
    break;
  }
  case DataType::FLOAT32: {
    float fill_value = 0.0f;
    if (value_size >= sizeof(float)) {
      std::memcpy(&fill_value, value, sizeof(float));
    }
    TensorFillKernel(tensor.data<float>(), num_elements, fill_value);
    break;
  }
  case DataType::FLOAT64: {
    double fill_value = 0.0;
    if (value_size >= sizeof(double)) {
      std::memcpy(&fill_value, value, sizeof(double));
    }
    TensorFillKernel(tensor.data<double>(), num_elements, fill_value);
    break;
  }
  case DataType::BOOLEAN: {
    bool fill_value = false;
    if (value_size >= sizeof(bool)) {
      std::memcpy(&fill_value, value, sizeof(bool));
    }
    TensorFillKernel(tensor.data<bool>(), num_elements, fill_value);
    break;
  }
  default:
    return Status::UNSUPPORTED_TYPE;
  }

  return Status::OK;
}

} // namespace kernel
} // namespace cpu
} // namespace compute
} // namespace axono
