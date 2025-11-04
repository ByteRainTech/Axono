#include "axono/compute/cuda/kernel/relu_kernel.cuh"
#include "axono/compute/cuda/ops.cuh"
#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cuda {

Status Relu(const Context &ctx, const Tensor &input, Tensor &output) {
  (void)ctx; // 暂时未使用

  // 设置结果张量的形状
  Status status = output.Resize(input.shape());
  if (status != Status::OK) {
    return status;
  }

  // 设置结果的数据类型
  if (output.dtype() != input.dtype()) {
    return Status::UNSUPPORTED_TYPE;
  }

  // 调用内核执行 ReLU
  return kernel::DispatchRelu(input, output);
}

Status ReluInplace(const Context &ctx, Tensor &tensor) {
  (void)ctx; // 暂时未使用

  // 直接原地执行 ReLU
  return kernel::DispatchReluInplace(tensor);
}

} // namespace cuda
} // namespace compute
} // namespace axono
