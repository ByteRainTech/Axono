// Axono/src/compute/cpu/operators/add.cpp
#include "axono/compute/cpu/operators/add.h"

#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cpu {
namespace operators {

core::Status Add(const core::Context &ctx, const core::Tensor &a,
                 const core::Tensor &b, core::Tensor &result) {
  (void)ctx;  // 暂时未使用

  // 检查数据类型一致性
  if (a.dtype() != b.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 设置结果张量的形状
  core::Status status = result.Resize(a.shape());
  if (status != core::Status::OK) {
    return status;
  }

  // 设置结果的数据类型
  if (result.dtype() != a.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 调用内核执行加法
  return DispatchAdd(a, b, result);
}

core::Status AddScalar(const core::Context &ctx, const core::Tensor &a,
                       void *scalar, size_t scalar_size, core::Tensor &result) {
  (void)ctx;  // 暂时未使用

  // 设置结果张量的形状
  core::Status status = result.Resize(a.shape());
  if (status != core::Status::OK) {
    return status;
  }

  // 设置结果的数据类型
  if (result.dtype() != a.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 调用内核执行标量加法
  return DispatchAddScalar(a, scalar, scalar_size, result);
}

}  // namespace operators
}  // namespace cpu
}  // namespace compute
}  // namespace axono
