#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace compute {
namespace cpu {

// 基础内存操作
AXONO_EXPORT Status MemoryCopy(const Context &ctx, void *dst, const void *src,
                               size_t num_bytes);

AXONO_EXPORT Status MemorySet(const Context &ctx, void *dst, int value,
                              size_t num_bytes);

// Tensor 操作
AXONO_EXPORT Status TensorCopy(const Context &ctx, Tensor &dst,
                               const Tensor &src);
AXONO_EXPORT Status TensorFill(const Context &ctx, Tensor &tensor, void *value,
                               size_t value_size);
AXONO_EXPORT Status TensorFillZero(const Context &ctx, Tensor &tensor);
AXONO_EXPORT Status TensorCreateLike(const Context &ctx, const Tensor &src,
                                     Tensor &dst);

// 矩阵喵~
AXONO_EXPORT Status MatMul(const Context &ctx, const Tensor &a, const Tensor &b,
                           Tensor &result);

AXONO_EXPORT Status Add(const Context& ctx, 
                       const Tensor& a, 
                       const Tensor& b, 
                       Tensor& result);

AXONO_EXPORT Status AddScalar(const Context& ctx,
                             const Tensor& a,
                             void* scalar,
                             size_t scalar_size,
                             Tensor& result);

} // namespace cpu
} // namespace compute
} // namespace axono
