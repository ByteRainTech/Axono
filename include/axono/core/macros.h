#pragma once

#include <cstdint>
#include <cstddef>

namespace axono {

// 基础数据类型枚举
enum class DataType {
    INT8,
    INT16,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    BOOLEAN
};

// 状态返回码
enum class Status {
    OK,
    INVALID_ARGUMENT,
    OUT_OF_MEMORY,
    UNSUPPORTED_TYPE,
    INTERNAL_ERROR
};

// 计算上下文
struct Context {
    int device_id = 0;  // 未来可用于多设备
};

} // namespace axono
