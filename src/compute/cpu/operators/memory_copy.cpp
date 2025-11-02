#include "axono/compute/cpu/operators.h"
#include "axono/compute/cpu/kernel/memory_copy_kernel.h"
#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cpu {

Status MemoryCopy(const Context& ctx,
                  void* dst,
                  const void* src,
                  size_t num_bytes) {
    // 使用上下文参数避免警告
    (void)ctx;  // 标记为已使用
    
    // 参数检查
    if (AXONO_UNLIKELY(dst == nullptr || src == nullptr)) {
        return Status::INVALID_ARGUMENT;
    }
    
    if (AXONO_UNLIKELY(num_bytes == 0)) {
        return Status::OK;
    }

    // 检查自拷贝
    if (AXONO_UNLIKELY(dst == src)) {
        return Status::OK;
    }

    // 调用底层内核执行拷贝
    kernel::MemoryCopyKernel(dst, src, num_bytes);

    return Status::OK;
}

Status MemorySet(const Context& ctx,
                 void* dst,
                 int value,
                 size_t num_bytes) {
    // 使用所有参数避免警告
    (void)ctx;
    (void)dst;
    (void)value;
    (void)num_bytes;
    
    // 暂时返回未实现，未来可以在这里实现
    return Status::UNSUPPORTED_TYPE;
}

} // namespace cpu
} // namespace compute
} // namespace axono
