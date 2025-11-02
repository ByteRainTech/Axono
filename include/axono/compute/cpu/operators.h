#pragma once

#include "axono/core/types.h"
#include "axono/core/macros.h"  // 必须包含这个来定义 AXONO_EXPORT

namespace axono {
namespace compute {
namespace cpu {

/**
 * @brief 内存拷贝算子
 * 
 * @param ctx 计算上下文
 * @param dst 目标内存地址
 * @param src 源内存地址  
 * @param num_bytes 拷贝的字节数
 * @return Status 操作状态
 */
AXONO_EXPORT Status MemoryCopy(const Context& ctx,
                               void* dst,
                               const void* src,
                               size_t num_bytes);

/**
 * @brief 内存设置算子（未来实现）
 */
AXONO_EXPORT Status MemorySet(const Context& ctx,
                              void* dst,
                              int value,
                              size_t num_bytes);

} // namespace cpu
} // namespace compute
} // namespace axono
