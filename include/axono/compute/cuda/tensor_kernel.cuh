#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include <cstring>
#include "axono/core/types.h"
#include <cuda_runtime.h>

namespace axono {
namespace compute {
namespace cuda {
namespace kernel {

// CUDA Tensor 数据填充内核
template <typename T>
__global__ void TensorFillKernel(T *data, size_t num_elements, T value) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = value;
  }
}

// CUDA Tensor 零填充内核
template <typename T>
__global__ void TensorZeroKernel(T *data, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = static_cast<T>(0);
  }
}

// Tensor 拷贝内核
AXONO_FORCE_INLINE void TensorCopyKernel(void *dst, const void *src,
                                         size_t num_bytes) {
    cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToDevice);
}

// 启动配置计算
inline dim3 CalculateLaunchConfig(size_t num_elements) {
  const size_t block_size = 256;
  const size_t grid_size = (num_elements + block_size - 1) / block_size;
  return dim3(grid_size);
}

// 类型分派的填充函数
AXONO_FORCE_INLINE Status DispatchFill(Tensor &tensor, void *value, size_t value_size) {
  auto num_elements = tensor.num_elements();
  auto dtype = tensor.dtype();
  auto launch_config = CalculateLaunchConfig(num_elements);

  switch (dtype) {
  case DataType::INT8: {
    int8_t fill_value = 0;
    if (value_size >= sizeof(int8_t)) {
      cudaMemcpyToSymbol(&fill_value, value, sizeof(int8_t));
    }
    TensorFillKernel<int8_t><<<launch_config, 256>>>(
        tensor.data<int8_t>(), num_elements, fill_value);
    break;
  }
  case DataType::INT16: {
    int16_t fill_value = 0;
    if (value_size >= sizeof(int16_t)) {
      cudaMemcpyToSymbol(&fill_value, value, sizeof(int16_t));
    }
    TensorFillKernel<int16_t><<<launch_config, 256>>>(
        tensor.data<int16_t>(), num_elements, fill_value);
    break;
  }
  case DataType::INT32: {
    int32_t fill_value = 0;
    if (value_size >= sizeof(int32_t)) {
      cudaMemcpyToSymbol(&fill_value, value, sizeof(int32_t));
    }
    TensorFillKernel<int32_t><<<launch_config, 256>>>(
        tensor.data<int32_t>(), num_elements, fill_value);
    break;
  }
  case DataType::INT64: {
    int64_t fill_value = 0;
    if (value_size >= sizeof(int64_t)) {
      cudaMemcpyToSymbol(&fill_value, value, sizeof(int64_t));
    }
    TensorFillKernel<int64_t><<<launch_config, 256>>>(
        tensor.data<int64_t>(), num_elements, fill_value);
    break;
  }
  case DataType::FLOAT32: {
    float fill_value = 0.0f;
    if (value_size >= sizeof(float)) {
      cudaMemcpyToSymbol(&fill_value, value, sizeof(float));
    }
    TensorFillKernel<float><<<launch_config, 256>>>(
        tensor.data<float>(), num_elements, fill_value);
    break;
  }
  case DataType::FLOAT64: {
    double fill_value = 0.0;
    if (value_size >= sizeof(double)) {
      cudaMemcpyToSymbol(&fill_value, value, sizeof(double));
    }
    TensorFillKernel<double><<<launch_config, 256>>>(
        tensor.data<double>(), num_elements, fill_value);
    break;
  }
  case DataType::BOOLEAN: {
    bool fill_value = false;
    if (value_size >= sizeof(bool)) {
      cudaMemcpyToSymbol(&fill_value, value, sizeof(bool));
    }
    TensorFillKernel<bool><<<launch_config, 256>>>(
        tensor.data<bool>(), num_elements, fill_value);
    break;
  }
  default:
    return Status::UNSUPPORTED_TYPE;
  }

  cudaDeviceSynchronize();
  return Status::OK;
}

// 零填充函数
AXONO_FORCE_INLINE Status DispatchZero(Tensor &tensor) {
  auto num_elements = tensor.num_elements();
  auto dtype = tensor.dtype();
  auto launch_config = CalculateLaunchConfig(num_elements);

  switch (dtype) {
  case DataType::INT8:
    TensorZeroKernel<int8_t><<<launch_config, 256>>>(tensor.data<int8_t>(), num_elements);
    break;
  case DataType::INT16:
    TensorZeroKernel<int16_t><<<launch_config, 256>>>(tensor.data<int16_t>(), num_elements);
    break;
  case DataType::INT32:
    TensorZeroKernel<int32_t><<<launch_config, 256>>>(tensor.data<int32_t>(), num_elements);
    break;
  case DataType::INT64:
    TensorZeroKernel<int64_t><<<launch_config, 256>>>(tensor.data<int64_t>(), num_elements);
    break;
  case DataType::FLOAT32:
    TensorZeroKernel<float><<<launch_config, 256>>>(tensor.data<float>(), num_elements);
    break;
  case DataType::FLOAT64:
    TensorZeroKernel<double><<<launch_config, 256>>>(tensor.data<double>(), num_elements);
    break;
  case DataType::BOOLEAN:
    TensorZeroKernel<bool><<<launch_config, 256>>>(tensor.data<bool>(), num_elements);
    break;
  default:
    return Status::UNSUPPORTED_TYPE;
  }

  cudaDeviceSynchronize();
  return Status::OK;
}

} // namespace kernel
} // namespace cuda
} // namespace compute
} // namespace axono
