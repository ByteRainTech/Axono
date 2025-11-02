#include "axono/compute/cpu/operators.h"
#include "axono/compute/cpu/kernel/matmul_kernel.h"
#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cpu {

Status MatMul(const Context& ctx, const Tensor& a, const Tensor& b, Tensor& result) {
    (void)ctx; // 暂时未使用
    
    // 基本参数检查
    if (a.ndim() != 2 || b.ndim() != 2) {
        return Status::INVALID_ARGUMENT;
    }
    
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    
    // 检查矩阵乘法形状兼容性
    if (a_shape[1] != b_shape[0]) {
        return Status::SHAPE_MISMATCH;
    }
    
    // 检查数据类型一致性
    if (a.dtype() != b.dtype()) {
        return Status::UNSUPPORTED_TYPE;
    }
    
    // 设置结果张量的形状
    std::vector<size_t> result_shape = {a_shape[0], b_shape[1]};
    Status status = result.Resize(result_shape);
    if (status != Status::OK) {
        return status;
    }
    
    // 设置结果的数据类型
    if (result.dtype() != a.dtype()) {
        // 如果需要，可以在这里重新创建结果张量
        return Status::UNSUPPORTED_TYPE;
    }
    
    // 调用内核执行矩阵乘法
    return kernel::DispatchMatMul(a, b, result);
}

} // namespace cpu
} // namespace compute
} // namespace axono
