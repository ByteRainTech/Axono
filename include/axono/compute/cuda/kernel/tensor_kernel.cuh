#pragma once
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include <cstddef>

namespace axono {
namespace compute {
namespace cuda {
namespace kernel {

// 填充函数
AXONO_FORCE_INLINE Status DispatchFill(Tensor &tensor, void *value, size_t value_size)
{
    (void)tensor;
    (void)value;
    (void)value_size;
    return Status::OK;
}

// 零填充函数
AXONO_FORCE_INLINE Status DispatchZero(Tensor &tensor)
{
    (void)tensor;
    return Status::OK;
}

} // namespace kernel
} // namespace cuda
} // namespace compute
} // namespace axono
