#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include <cuda_runtime.h>
#include "axono/core/types.h"
#include <cstddef>

namespace axono {
namespace compute {
namespace cuda {
namespace kernel {

// CUDA 矩阵乘法内核 - 使用共享内存优化
template <typename T>
__global__ void MatMulKernel(const T *a, const T *b, T *result,
                             size_t m, size_t n, size_t k) {
  // 使用二维网格和块
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    T sum = 0;
    for (size_t l = 0; l < k; ++l) {
      sum += a[row * k + l] * b[l * n + col];
    }
    result[row * n + col] = sum;
  }
}

// 优化的矩阵乘法内核（使用共享内存）
template <typename T, size_t BLOCK_SIZE>
__global__ void MatMulOptimizedKernel(const T *a, const T *b, T *result,
                                      size_t m, size_t n, size_t k) {
  __shared__ T a_tile[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ T b_tile[BLOCK_SIZE][BLOCK_SIZE];

  size_t row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  size_t col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  T sum = 0;

  for (size_t tile_idx = 0; tile_idx < (k + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile_idx) {
    // 加载A的瓦片到共享内存
    if (row < m && (tile_idx * BLOCK_SIZE + threadIdx.x) < k) {
      a_tile[threadIdx.y][threadIdx.x] = a[row * k + tile_idx * BLOCK_SIZE + threadIdx.x];
    } else {
      a_tile[threadIdx.y][threadIdx.x] = 0;
    }

    // 加载B的瓦片到共享内存
    if (col < n && (tile_idx * BLOCK_SIZE + threadIdx.y) < k) {
      b_tile[threadIdx.y][threadIdx.x] = b[(tile_idx * BLOCK_SIZE + threadIdx.y) * n + col];
    } else {
      b_tile[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    // 计算部分和
    for (size_t l = 0; l < BLOCK_SIZE; ++l) {
      sum += a_tile[threadIdx.y][l] * b_tile[l][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < m && col < n) {
    result[row * n + col] = sum;
  }
}

// 类型分派的矩阵乘法
AXONO_FORCE_INLINE Status DispatchMatMul(const Tensor &a, const Tensor &b, Tensor &result) {
  auto a_shape = a.shape();
  auto b_shape = b.shape();

  size_t m = a_shape[0];
  size_t k = a_shape[1];
  size_t n = b_shape[1];

  // 检查形状兼容性
  if (a_shape[1] != b_shape[0]) {
    return Status::SHAPE_MISMATCH;
  }

  // 配置CUDA内核启动参数
  const size_t BLOCK_SIZE = 16;
  dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_dim((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // 根据数据类型选择内核
  switch (a.dtype()) {
  case DataType::FLOAT32:
    MatMulOptimizedKernel<float, BLOCK_SIZE><<<grid_dim, block_dim>>>(
        a.data<float>(), b.data<float>(), result.data<float>(), m, n, k);
    break;
  case DataType::FLOAT64:
    MatMulOptimizedKernel<double, BLOCK_SIZE><<<grid_dim, block_dim>>>(
        a.data<double>(), b.data<double>(), result.data<double>(), m, n, k);
    break;
  case DataType::INT32:
    MatMulOptimizedKernel<int32_t, BLOCK_SIZE><<<grid_dim, block_dim>>>(
        a.data<int32_t>(), b.data<int32_t>(), result.data<int32_t>(), m, n, k);
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
