
#include "axono/compute/cuda/operators.cuh"
#include "axono/compute/cuda/kernel/tensor_kernel.cuh"
#include "axono/core/macros.h"
#include "axono/core/tensor.h"

namespace axono {
namespace compute {
namespace cuda {

Status TensorCopy(const Context &ctx, Tensor &dst, const Tensor &src) {
  (void)ctx; // 暂时未使用

  if (!dst.IsSameShape(src)) {
    return Status::SHAPE_MISMATCH;
  }

  if (dst.dtype() != src.dtype()) {
    return Status::UNSUPPORTED_TYPE;
  }

  if (!src.raw_data() || !dst.raw_data()) {
    return Status::INVALID_ARGUMENT;
  }

  kernel::TensorCopyKernel(dst.raw_data(), src.raw_data(), src.num_bytes());
  return Status::OK;
}

Status TensorFill(const Context &ctx, Tensor &tensor, void *value,
                  size_t value_size) {
  (void)ctx;
  return tensor.Fill(value, value_size);
}

Status TensorFillZero(const Context &ctx, Tensor &tensor) {
  (void)ctx;
  return tensor.FillZero();
}

Status TensorCreateLike(const Context &ctx, const Tensor &src, Tensor &dst) {
  (void)ctx;
  dst = Tensor::CreateLike(src);
  return Status::OK;
}

} // namespace cpu
} // namespace compute
} // namespace axono
