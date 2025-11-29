#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include <cstddef>

namespace axono {
namespace compute {
namespace cpu {
namespace operators {

core::Status MatMul(const core::Context &ctx, const core::Tensor &a,
                    const core::Tensor &b, core::Tensor &result);

} // namespace operators
} // namespace cpu
} // namespace compute
} // namespace axono
