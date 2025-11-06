#pragma once
#include "axono/core/tensor.h"
#include "axono/core/types.h"
#include <cstddef>

namespace axono::compute::cuda::kernel {

// 只声明，不定义
AXONO_EXPORT Status DispatchFill(Tensor &tensor, void *value, size_t value_size);
AXONO_EXPORT Status DispatchZero(Tensor &tensor);
AXONO_EXPORT void TensorCopyKernel(void *dst, const void *src, size_t num_bytes);

// 模板函数声明
template <typename T>
AXONO_EXPORT Status TensorReadKernel(const T* device_data, T* host_data, size_t num_elements);

}
