#include "axono/core/macros.h"
#include <cstddef>
#include <cstdint>
#include "axono/core/types.h"
#include <cuda_runtime.h>
#include <cstring>

namespace axono {
namespace compute {
namespace cuda {
namespace operators {

// CUDA 内存拷贝内核
__global__ void MemoryCopyKernel(void *dst, const void *src, size_t num_bytes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t byte_idx = idx * sizeof(uint64_t);
  
  if (byte_idx + sizeof(uint64_t) <= num_bytes) {
    // 按64位字拷贝
    uint64_t *dst_word = reinterpret_cast<uint64_t*>(static_cast<char*>(dst) + byte_idx);
    const uint64_t *src_word = reinterpret_cast<const uint64_t*>(static_cast<const char*>(src) + byte_idx);
    *dst_word = *src_word;
  } else {
    // 处理剩余字节
    char *dst_byte = static_cast<char*>(dst) + idx;
    const char *src_byte = static_cast<const char*>(src) + idx;
    if (idx < num_bytes) {
      *dst_byte = *src_byte;
    }
  }
}

// 优化的设备到设备内存拷贝
AXONO_FORCE_INLINE core::Status DispatchMemoryCopy(void *dst, const void *src, size_t num_bytes) {
  if (num_bytes == 0) {
    return core::Status::OK;
  }

  // 对于小数据量，使用单个块
  if (num_bytes < 1024) {
    const size_t block_size = 256;
    const size_t grid_size = (num_bytes + block_size - 1) / block_size;
    MemoryCopyKernel<<<grid_size, block_size>>>(dst, src, num_bytes);
  } else {
    // 对于大数据量，使用cudaMemcpy（已经高度优化）
    cudaError_t err = cudaMemcpy(dst, src, num_bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
      return core::Status::DEVICE_ERROR;
    }
  }

  cudaDeviceSynchronize();
  return core::Status::OK;
}

} // namespace kernel
} // namespace cuda
} // namespace compute
} // namespace axono
