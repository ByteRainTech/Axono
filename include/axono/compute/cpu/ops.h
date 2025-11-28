#pragma once

#include "axono/core/macros.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono {
namespace compute {
namespace cpu {
namespace ops {
AXONO_EXPORT core::Status Relu(const core::Context &ctx,
                               const core::Tensor &input, core::Tensor &output);

AXONO_EXPORT core::Status ReluInplace(const core::Context &ctx,
                                      core::Tensor &tensor);
} // namespace ops
} // namespace cpu
} // namespace compute
} // namespace axono
