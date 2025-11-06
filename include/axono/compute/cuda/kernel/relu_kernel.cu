#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <cstddef>

namespace axono {
namespace compute {
namespace cuda {
namespace kernel {

// CUDA ReLU 激活函数内核
template <typename T>
__global__ void ReluKernel(const T *input, T *output, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    output[idx] = max(static_cast<T>(0), input[idx]);
  }
}

// CUDA 原地 ReLU 激活函数
template <typename T>
__global__ void ReluInplaceKernel(T *data, size_t num_elements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    data[idx] = max(static_cast<T>(0), data[idx]);
  }
}

// 启动配置计算
inline dim3 CalculateLaunchConfig(size_t num_elements) {
  const size_t block_size = 256;
  const size_t grid_size = (num_elements + block_size - 1) / block_size;
  return dim3(grid_size);
}

// 类型分派的 ReLU
AXONO_FORCE_INLINE Status DispatchRelu(const Tensor &input, Tensor &output) {
  auto num_elements = input.num_elements();

  // 检查形状一致性
  if (!input.IsSameShape(output)) {
    return Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (input.dtype() != output.dtype()) {
    return Status::UNSUPPORTED_TYPE;
  }

  auto launch_config = CalculateLaunchConfig(num_elements);
  
  // 根据数据类型选择内核
  switch (input.dtype()) {
  case DataType::FLOAT32:
    ReluKernel<float><<<launch_config, 256>>>(
        input.data<float>(), output.data<float>(), num_elements);
    break;
  case DataType::FLOAT64:
    ReluKernel<double><<<launch_config, 256>>>(
        input.data<double>(), output.data<double>(), num_elements);
    break;
  case DataType::INT32:
    ReluKernel<int32_t><<<launch_config, 256>>>(
        input.data<int32_t>(), output.data<int32_t>(), num_elements);
    break;
  default:
    return Status::UNSUPPORTED_TYPE;
  }

  cudaDeviceSynchronize();
  return Status::OK;
}

// 类型分派的原地 ReLU
AXONO_FORCE_INLINE Status DispatchReluInplace(Tensor &tensor) {
  auto num_elements = tensor.num_elements();
  auto launch_config = CalculateLaunchConfig(num_elements);
  
  // 根据数据类型选择内核
  switch (tensor.dtype()) {
  case DataType::FLOAT32:
    ReluInplaceKernel<float><<<launch_config, 256>>>(
        tensor.data<float>(), num_elements);
    break;
  case DataType::FLOAT64:
    ReluInplaceKernel<double><<<launch_config, 256>>>(
        tensor.data<double>(), num_elements);
    break;
  case DataType::INT32:
    ReluInplaceKernel<int32_t><<<launch_config, 256>>>(
        tensor.data<int32_t>(), num_elements);
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
