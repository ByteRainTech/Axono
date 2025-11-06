#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace compute {
namespace cuda {
// Tensor 操作
AXONO_EXPORT Status TensorCopy(const Context &ctx, Tensor &dst,
                               const Tensor &src);
AXONO_EXPORT Status TensorFill(const Context &ctx, Tensor &tensor, void *value,
                               size_t value_size);
AXONO_EXPORT Status TensorFillZero(const Context &ctx, Tensor &tensor);
AXONO_EXPORT Status TensorCreateLike(const Context &ctx, const Tensor &src,
                                     Tensor &dst);

AXONO_EXPORT Status Relu(const Context &ctx, const Tensor &input,
                         Tensor &output);

AXONO_EXPORT Status ReluInplace(const Context &ctx, Tensor &tensor);

// 读取内核需要声明为模板函数
template <typename T>
AXONO_FORCE_INLINE Status TensorReadKernel(const T* device_data, T* host_data, size_t num_elements)
{
    (void)device_data;
    (void)host_data;
    (void)num_elements;
    // TODO: cudaMemcpyDeviceToHost
    return Status::OK;
}
} // namespace cuda
} // namespace compute
} // namespace axono
