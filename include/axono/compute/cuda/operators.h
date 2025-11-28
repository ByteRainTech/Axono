#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace compute {
namespace cuda {
namespace operators {
// 基础内存操作
AXONO_EXPORT core::Status MemoryCopy(const core::Context &ctx, void *dst, const void *src,
                               size_t num_bytes);

AXONO_EXPORT core::Status MemorySet(const core::Context &ctx, void *dst, int value,
                              size_t num_bytes);
// 矩阵喵~
AXONO_EXPORT core::Status MatMul(const core::Context &ctx, const core::Tensor &a, const core::Tensor &b,
                           core::Tensor &result);

AXONO_EXPORT core::Status Add(const core::Context &ctx, const core::Tensor &a, const core::Tensor &b,
                        core::Tensor &result);

AXONO_EXPORT core::Status AddScalar(const core::Context &ctx, const core::Tensor &a, void *scalar,
                              size_t scalar_size, core::Tensor &result);
}
}
}
}
