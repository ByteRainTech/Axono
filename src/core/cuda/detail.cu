// Axono/src/core/cuda/detail.cu
#include "axono/core/tensor.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace axono {

namespace detail {

// CUDA设备内存分配实现
std::shared_ptr<void> CudaAllocateStorage(size_t bytes, const std::string& device) {
    // 解析设备字符串
    int device_id = 0;
    if (device.size() > 5) {
        try {
            device_id = std::stoi(device.substr(5));
        } catch (...) {
            throw std::invalid_argument("Invalid CUDA device format: " + device);
        }
    }

    // 设备检测
    int cuda_device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&cuda_device_count);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA device detection failed: " + 
                               std::string(cudaGetErrorString(err)));
    }
    if (device_id < 0 || device_id >= cuda_device_count) {
        throw std::out_of_range("CUDA device " + std::to_string(device_id) + 
                              " out of range (0-" + std::to_string(cuda_device_count-1) + ")");
    }

    // 设置当前设备
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " + 
                               std::string(cudaGetErrorString(err)));
    }

    // 分配CUDA内存
    void* dev_ptr = nullptr;
    err = cudaMalloc(&dev_ptr, bytes);
    if (err != cudaSuccess) {
        throw std::bad_alloc();
    }

    // 初始化为0
    err = cudaMemset(dev_ptr, 0, bytes);
    if (err != cudaSuccess) {
        cudaFree(dev_ptr); // 清理已分配的内存
        throw std::runtime_error("CUDA memset failed: " + 
                               std::string(cudaGetErrorString(err)));
    }

    // 返回带CUDA释放器的智能指针
    return std::shared_ptr<void>(dev_ptr, [](void* ptr) {
        cudaFree(ptr);
    });
}

} // namespace detail

} // namespace axono
