#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "axono/core/macros.h"

namespace axono {
namespace compute {
namespace cpu {
namespace operators {

// 高度优化的内存拷贝内核
// 在头文件中实现以确保内联优化
AXONO_FORCE_INLINE void MemoryCopyKernel(void *__restrict dst,
                                         const void *__restrict src,
                                         size_t num_bytes) {
  if (AXONO_UNLIKELY(num_bytes == 0)) {
    return;
  }

  char *__restrict dst_ptr = static_cast<char *>(dst);
  const char *__restrict src_ptr = static_cast<const char *>(src);

  // 对于小数据量，使用逐字节拷贝
  if (AXONO_UNLIKELY(num_bytes < 16)) {
    for (size_t i = 0; i < num_bytes; ++i) {
      dst_ptr[i] = src_ptr[i];
    }
    return;
  }

  // 对于中等数据量，使用字长拷贝
  if (num_bytes < 1024) {
    size_t num_words = num_bytes / sizeof(uint64_t);
    size_t remainder = num_bytes % sizeof(uint64_t);

    uint64_t *__restrict dst_word = reinterpret_cast<uint64_t *>(dst_ptr);
    const uint64_t *__restrict src_word =
        reinterpret_cast<const uint64_t *>(src_ptr);

    for (size_t i = 0; i < num_words; ++i) {
      dst_word[i] = src_word[i];
    }

    size_t offset = num_words * sizeof(uint64_t);
    for (size_t i = 0; i < remainder; ++i) {
      dst_ptr[offset + i] = src_ptr[offset + i];
    }
    return;
  }

  std::memcpy(dst, src, num_bytes);
}

}  // namespace operators
}  // namespace cpu
}  // namespace compute
}  // namespace axono
