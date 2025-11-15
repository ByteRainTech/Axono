#include <algorithm>
#include <cuda_runtime.h>
#include <cstddef>

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace compute {
namespace cuda {
namespace ops {

// CUDA ReLU 激活函数内核
template <typename T>
__global__ void ReluKernel(const T* __restrict__ input, 
                          T* __restrict__ output, 
                          int num_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 边界检查
  if (idx < num_elements) {
    T value = input[idx];
    output[idx] = (value > static_cast<T>(0)) ? value : static_cast<T>(0);
    
    // 调试输出 - 只在前几个线程打印
    if (idx < 6) {
      printf("tid=%d in=%.2f out=%.2f\n", idx, (float)value, (float)output[idx]);
    }
  }
}

// CUDA 原地 ReLU 激活函数
template <typename T>
__global__ void ReluInplaceKernel(T* data, int num_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < num_elements) {
    T value = data[idx];
    data[idx] = (value > static_cast<T>(0)) ? value : static_cast<T>(0);
  }
}

// 启动配置计算
inline dim3 CalculateLaunchConfig(int num_elements) {
  const int block_size = 256;
  
  if (num_elements <= 0) {
    return dim3(1);
  }
  
  int grid_size = (num_elements + block_size - 1) / block_size;
  
  // CUDA 硬件限制
  const int max_grid_size = 65535;
  if (grid_size > max_grid_size) {
    grid_size = max_grid_size;
  }
  
  // 确保至少启动1个block
  if (grid_size == 0) {
    grid_size = 1;
  }
  
  return dim3(grid_size);
}

// 类型分派的 ReLU
AXONO_FORCE_INLINE core::Status DispatchRelu(const core::Tensor& input, core::Tensor& output) {
  // 获取元素数量并转换为int
  size_t num_elements_size = input.num_elements();
  if (num_elements_size > INT_MAX) {
    printf("Error: Too many elements (%zu), exceeds INT_MAX\n", num_elements_size);
    return core::Status::INVALID_ARGUMENT;
  }
  int num_elements = static_cast<int>(num_elements_size);
  
  printf("[DispatchRelu] num_elements=%d\n", num_elements);
  printf("input ptr=%p output ptr=%p\n", 
         (void*)input.data<float>(), (void*)output.data<float>());
  
  // 基本检查
  if (num_elements <= 0) {
    printf("Error: Invalid number of elements: %d\n", num_elements);
    return core::Status::INVALID_ARGUMENT;
  }
  
  if (input.data<void>() == nullptr) {
    printf("Error: Input pointer is null\n");
    return core::Status::INVALID_PTR;
  }
  
  if (output.data<void>() == nullptr) {
    printf("Error: Output pointer is null\n");
    return core::Status::INVALID_PTR;
  }
  
  if (!input.IsSameShape(output)) {
    printf("Error: Shape mismatch between input and output\n");
    return core::Status::SHAPE_MISMATCH;
  }
  
  if (input.dtype() != output.dtype()) {
    printf("Error: Data type mismatch between input and output\n");
    return core::Status::UNSUPPORTED_TYPE;
  }
  
  // 检查设备一致性 - 使用辅助函数
  if (input.is_cuda() != output.is_cuda()) {
    printf("Error: Tensors not on CUDA device\n");
    return core::Status::DEVICE_MISMATCH;
  }
  
  // 计算启动配置
  dim3 grid_size = CalculateLaunchConfig(num_elements);
  const int block_size = 256;
  
  printf("Launch config: grid=%u, block=%d\n", grid_size.x, block_size);
  
  // 清除之前的CUDA错误
  cudaGetLastError();
  
  // 根据数据类型分派内核
  cudaError_t cuda_err = cudaSuccess;
  switch (input.dtype()) {
    case core::DataType::FLOAT32: {
      printf("Launching float kernel...\n");
      ReluKernel<float><<<grid_size, block_size>>>(
          input.data<float>(), 
          output.data<float>(), 
          num_elements);
      cuda_err = cudaGetLastError();
      break;
    }
    case core::DataType::FLOAT64: {
      printf("Launching double kernel...\n");
      ReluKernel<double><<<grid_size, block_size>>>(
          input.data<double>(), 
          output.data<double>(), 
          num_elements);
      cuda_err = cudaGetLastError();
      break;
    }
    case core::DataType::INT32: {
      printf("Launching int32 kernel...\n");
      ReluKernel<int32_t><<<grid_size, block_size>>>(
          input.data<int32_t>(), 
          output.data<int32_t>(), 
          num_elements);
      cuda_err = cudaGetLastError();
      break;
    }
    default: {
      printf("Error: Unsupported data type: %d\n", static_cast<int>(input.dtype()));
      return core::Status::UNSUPPORTED_TYPE;
    }
  }
  
  if (cuda_err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(cuda_err));
    return core::Status::DEVICE_ERROR;
  }
  
  printf("Kernel launched successfully, synchronizing...\n");
  
  // 同步设备
  cuda_err = cudaDeviceSynchronize();
  if (cuda_err != cudaSuccess) {
    printf("Device synchronization error: %s\n", cudaGetErrorString(cuda_err));
    return core::Status::DEVICE_ERROR;
  }
  
  printf("ReLU operation completed successfully\n");
  return core::Status::OK;
}

// 类型分派的原地 ReLU
AXONO_FORCE_INLINE core::Status DispatchReluInplace(core::Tensor& tensor) {
  // 获取元素数量并转换为int
  size_t num_elements_size = tensor.num_elements();
  if (num_elements_size > INT_MAX) {
    printf("Error: Too many elements (%zu), exceeds INT_MAX\n", num_elements_size);
    return core::Status::INVALID_ARGUMENT;
  }
  int num_elements = static_cast<int>(num_elements_size);
  
  printf("[DispatchReluInplace] num_elements=%d\n", num_elements);
  
  // 基本检查
  if (num_elements <= 0) {
    return core::Status::INVALID_ARGUMENT;
  }
  
  if (tensor.data<void>() == nullptr) {
    return core::Status::INVALID_PTR;
  }
  
  // 检查设备 - 使用辅助函数
  if (!tensor.is_cuda()) {
    return core::Status::DEVICE_MISMATCH;
  }
  
  // 计算启动配置
  dim3 grid_size = CalculateLaunchConfig(num_elements);
  const int block_size = 256;
  
  // 清除之前的CUDA错误
  cudaGetLastError();
  
  // 根据数据类型分派内核
  cudaError_t cuda_err = cudaSuccess;
  switch (tensor.dtype()) {
    case core::DataType::FLOAT32:
      ReluInplaceKernel<float><<<grid_size, block_size>>>(
          tensor.data<float>(), num_elements);
      cuda_err = cudaGetLastError();
      break;
    case core::DataType::FLOAT64:
      ReluInplaceKernel<double><<<grid_size, block_size>>>(
          tensor.data<double>(), num_elements);
      cuda_err = cudaGetLastError();
      break;
    case core::DataType::INT32:
      ReluInplaceKernel<int32_t><<<grid_size, block_size>>>(
          tensor.data<int32_t>(), num_elements);
      cuda_err = cudaGetLastError();
      break;
    default:
      return core::Status::UNSUPPORTED_TYPE;
  }
  
  if (cuda_err != cudaSuccess) {
    printf("Inplace kernel launch error: %s\n", cudaGetErrorString(cuda_err));
    return core::Status::DEVICE_ERROR;
  }
  
  cuda_err = cudaDeviceSynchronize();
  if (cuda_err != cudaSuccess) {
    printf("Inplace device synchronization error: %s\n", cudaGetErrorString(cuda_err));
    return core::Status::DEVICE_ERROR;
  }
  
  return core::Status::OK;
}

}
}
}
}
