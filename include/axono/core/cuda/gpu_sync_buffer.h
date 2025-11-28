#ifndef GPU_SYNC_BUFFER_H
#define GPU_SYNC_BUFFER_H

#include <vector>
#include <memory>
#include "axono/core/tensor.h"

namespace pybind11 {
    class array;
    class capsule;
}

class GPUSyncBuffer {
private:
    float* host_data_;
    axono::core::Tensor gpu_tensor_;
    bool modified_;
    size_t num_elems_;

public:
    // 构造函数：初始化并同步数据从GPU到CPU
    GPUSyncBuffer(const axono::core::Tensor& tensor);
    
    // 析构函数：自动同步修改回GPU
    ~GPUSyncBuffer();
    
    // 禁止拷贝构造和赋值
    GPUSyncBuffer(const GPUSyncBuffer&) = delete;
    GPUSyncBuffer& operator=(const GPUSyncBuffer&) = delete;
    
    // 移动构造和赋值
    GPUSyncBuffer(GPUSyncBuffer&& other) noexcept;
    GPUSyncBuffer& operator=(GPUSyncBuffer&& other) noexcept;
    
    // 获取主机数据指针
    float* data() { return host_data_; }
    const float* data() const { return host_data_; }
    
    // 获取元素数量
    size_t size() const { return num_elems_; }
    
    // 标记数据已被修改
    void mark_modified() { modified_ = true; }
    
    // 手动触发同步到GPU
    void sync_to_gpu();
    
    // 从GPU同步到CPU
    void sync_from_gpu();
    
    // 检查是否有未同步的修改
    bool is_modified() const { return modified_; }
    
    // 获取原始GPU张量
    const axono::core::Tensor& gpu_tensor() const { return gpu_tensor_; }
};

// 工具函数：将GPU张量转换为带同步的numpy数组
// Attention：这里只声明，实现在.cu文件中
pybind11::array tensor_to_sync_numpy(const axono::core::Tensor& tensor);

#endif // GPU_SYNC_BUFFER_H
