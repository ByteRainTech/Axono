#include "axono/compute/cpu/operators/memory_copy.h"

#include "axono/compute/cpu/operators.h"
#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cpu {
namespace operators {

core::Status MemoryCopy(const core::Context &ctx, void *dst, const void *src,
                        size_t num_bytes) {
  // 使用上下文参数避免警告
  (void)ctx;  // 标记为已使用

  // 参数检查
  if (AXONO_UNLIKELY(dst == nullptr || src == nullptr)) {
    return core::Status::INVALID_ARGUMENT;
  }

  if (AXONO_UNLIKELY(num_bytes == 0)) {
    return core::Status::OK;
  }

  // 检查自拷贝
  if (AXONO_UNLIKELY(dst == src)) {
    return core::Status::OK;
  }

  // 调用底层内核执行拷贝
  MemoryCopyKernel(dst, src, num_bytes);

  return core::Status::OK;
}

core::Status MemorySet(const core::Context &ctx, void *dst, int value,
                       size_t num_bytes) {
  // 使用所有参数避免警告
  (void)ctx;
  (void)dst;
  (void)value;
  (void)num_bytes;

  // 暂时返回未实现，未来可以在这里实现
  return core::Status::UNSUPPORTED_TYPE;
}

}  // namespace operators
}  // namespace cpu
}  // namespace compute
}  // namespace axono
