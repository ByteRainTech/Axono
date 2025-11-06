// Axono/src/core/cuda/detail.cuh
#pragma once

#include <memory>
#include <string>

namespace axono {
namespace detail {

// 分配 CUDA 设备内存，返回带 cudaFree 的 shared_ptr
std::shared_ptr<void> CudaAllocateStorage(size_t bytes,
                                          const std::string& device = "cuda:0");

} // namespace detail
} // namespace axono
