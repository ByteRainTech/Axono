#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace compute {
namespace cpu {
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
} // namespace cpu
} // namespace compute
} // namespace axono
