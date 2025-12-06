#pragma once
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace axono::compute::cuda::operators {

core::Status Randn(const core::Context& ctx, core::Tensor& out,
                   float mean = 0.0f, float stddev = 1.0f);

}  // namespace axono::compute::cuda::operators
