#include "axono/compute/cuda/kernel/add_kernel.cuh"
#include "axono/compute/cuda/operators.cuh"
#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cuda {

Status Add(const Context &ctx, const Tensor &a, const Tensor &b,
           Tensor &result) {
  (void)ctx; // 暂时未使用

  // 基本参数检查
  if (a.ndim() != b.ndim()) {
    return Status::SHAPE_MISMATCH;
  }

  // 检查形状一致性
  if (!a.IsSameShape(b)) {
    return Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (a.dtype() != b.dtype()) {
    return Status::UNSUPPORTED_TYPE;
  }

  // 设置结果张量的形状
  Status status = result.Resize(a.shape());
  if (status != Status::OK) {
    return status;
  }

  // 设置结果的数据类型
  if (result.dtype() != a.dtype()) {
    return Status::UNSUPPORTED_TYPE;
  }

  // 调用内核执行加法
  return kernel::DispatchAdd(a, b, result);
}

Status AddScalar(const Context &ctx, const Tensor &a, void *scalar,
                 size_t scalar_size, Tensor &result) {
  (void)ctx; // 暂时未使用

  // 设置结果张量的形状
  Status status = result.Resize(a.shape());
  if (status != Status::OK) {
    return status;
  }

  // 设置结果的数据类型
  if (result.dtype() != a.dtype()) {
    return Status::UNSUPPORTED_TYPE;
  }

  // 调用内核执行标量加法
  return kernel::DispatchAddScalar(a, scalar, scalar_size, result);
}

} // namespace cuda
} // namespace compute
} // namespace axono
