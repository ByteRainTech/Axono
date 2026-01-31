#include <cuda_runtime.h>

#include "axono/core/tensor.h"
#include "axono/core/macros.h"
#include "axono/core/types.h"

namespace axono {
namespace core {
namespace cuda {
namespace tensor {

namespace {
template <typename T>
__global__ void Transpose2DKernel(const T* src, T* dst, 
                                  size_t dim0_size, size_t dim1_size,
                                  size_t other_size, size_t src_stride0, size_t src_stride1) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= other_size * dim0_size * dim1_size) return;

    const size_t total_dim1_dim0 = dim1_size * dim0_size;
    const size_t other_idx = idx / total_dim1_dim0;
    const size_t rem = idx % total_dim1_dim0;
    const size_t dim1_idx = rem / dim0_size;
    const size_t dim0_idx = rem % dim0_size;

    const size_t src_idx = other_idx * src_stride0 * src_stride1 
                           + dim0_idx * src_stride1 
                           + dim1_idx;
    const size_t dst_idx = other_idx * src_stride0 * src_stride1 
                           + dim1_idx * src_stride0 
                           + dim0_idx;
    dst[dst_idx] = src[src_idx];
}

template <typename T>
Status LaunchTransposeKernel(const Tensor& src, Tensor& dst, int dim0, int dim1) {
    dim0 = (dim0 < 0) ? static_cast<int>(src.shape().size()) + dim0 : dim0;
    dim1 = (dim1 < 0) ? static_cast<int>(src.shape().size()) + dim1 : dim1;

    const size_t dim0_size = src.shape()[dim0];
    const size_t dim1_size = src.shape()[dim1];
    size_t other_size = 1;
    for (int i = 0; i < static_cast<int>(src.shape().size()); ++i) {
        if (i != dim0 && i != dim1) {
            other_size *= src.shape()[i];
        }
    }

    const size_t total_elements = other_size * dim0_size * dim1_size;
    const int block_size = 256;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    Transpose2DKernel<T><<<grid_size, block_size>>>(
        src.data<T>(), dst.data<T>(),
        dim0_size, dim1_size, other_size,
        dim0_size, dim1_size
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::DEVICE_ERROR;
    }
    return Status::OK;
}
}  // anonymous namespace

Status TransposeKernel(const Tensor& src, Tensor& dst, int dim0, int dim1) {
    if (!src.is_cuda() || !dst.is_cuda()) {
        return Status::DEVICE_MISMATCH;
    }
    if (src.dtype() != dst.dtype()) {
        return Status::UNSUPPORTED_TYPE;
    }

    const DataType dtype = src.dtype();
    switch (dtype) {
        case DataType::INT8:
            return LaunchTransposeKernel<int8_t>(src, dst, dim0, dim1);
        case DataType::INT16:
            return LaunchTransposeKernel<int16_t>(src, dst, dim0, dim1);
        case DataType::INT32:
            return LaunchTransposeKernel<int32_t>(src, dst, dim0, dim1);
        case DataType::INT64:
            return LaunchTransposeKernel<int64_t>(src, dst, dim0, dim1);
        case DataType::FLOAT32:
            return LaunchTransposeKernel<float>(src, dst, dim0, dim1);
        case DataType::FLOAT64:
            return LaunchTransposeKernel<double>(src, dst, dim0, dim1);
        case DataType::BOOLEAN:
            return LaunchTransposeKernel<bool>(src, dst, dim0, dim1);
        default:
            return Status::UNSUPPORTED_TYPE;
    }
}

}  // namespace tensor
}  // namespace cuda
}  // namespace core
}  // namespace axono
