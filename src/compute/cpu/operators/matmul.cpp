#include "axono/compute/cpu/operators/matmul.h"
#include "axono/compute/cpu/operators.h"
#include "axono/core/macros.h"

namespace axono::compute::cpu::operators {
    template void MatMulOptimizedKernel<float>(
        const float*, const float*, float*, size_t, size_t, size_t);

    template void MatMulOptimizedKernel<double>(
        const double*, const double*, double*, size_t, size_t, size_t);

    template void MatMulOptimizedKernel<int32_t>(
        const int32_t*, const int32_t*, int32_t*, size_t, size_t, size_t);
}

namespace axono {
namespace compute {
namespace cpu {
namespace operators {

core::Status MatMul(const core::Context &ctx, const core::Tensor &a, const core::Tensor &b,
              core::Tensor &result) {
  (void)ctx; // 暂时未使用

  // 基本参数检查
  if (a.ndim() != 2 || b.ndim() != 2) {
    return core::Status::INVALID_ARGUMENT;
  }

  auto a_shape = a.shape();
  auto b_shape = b.shape();

  // 检查矩阵乘法形状兼容性
  if (a_shape[1] != b_shape[0]) {
    return core::Status::SHAPE_MISMATCH;
  }

  // 检查数据类型一致性
  if (a.dtype() != b.dtype()) {
    return core::Status::UNSUPPORTED_TYPE;
  }

  // 设置结果张量的形状
  std::vector<size_t> result_shape = {a_shape[0], b_shape[1]};
  core::Status status = result.Resize(result_shape);
  if (status != core::Status::OK) {
    return status;
  }

  // 设置结果的数据类型
  if (result.dtype() != a.dtype()) {
    result = core::Tensor(a.dtype(), result_shape);
    if (!result.data()) return core::Status::OUT_OF_MEMORY;
}

  // 调用内核执行矩阵乘法
  return DispatchMatMul(a, b, result);
}

}
}
}
}
