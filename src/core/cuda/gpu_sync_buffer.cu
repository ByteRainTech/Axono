#include "axono/core/cuda/gpu_sync_buffer.h"
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "axono/core/tensor.h"

// 只在当前文件中使用py命名空间
namespace py = pybind11;

namespace {

inline cudaError_t raw_copy_d2h(const float* device_ptr, float* host_ptr, size_t n) {
    if (!device_ptr || !host_ptr || n == 0) return cudaErrorInvalidValue;
    return cudaMemcpy(host_ptr, device_ptr, n * sizeof(float), cudaMemcpyDefault);
}

inline cudaError_t raw_copy_h2d(const float* host_ptr, float* device_ptr, size_t n) {
    if (!host_ptr || !device_ptr || n == 0) return cudaErrorInvalidValue;
    // 用异步 + 默认流，驱动对齐检查更宽松
    cudaError_t err = cudaMemcpyAsync(device_ptr, host_ptr, n * sizeof(float),
                                      cudaMemcpyHostToDevice, nullptr);
    if (err == cudaSuccess) cudaStreamSynchronize(nullptr);
    return err;
}

inline cudaError_t raw_write_d2h(const float* host_ptr, float* device_ptr, size_t n) {
    if (!host_ptr || !device_ptr || n == 0) return cudaErrorInvalidValue;
    for (size_t i = 0; i < n; ++i) {
        cudaError_t err = cudaMemcpyAsync(device_ptr + i, host_ptr + i, sizeof(float),
                                          cudaMemcpyHostToDevice, nullptr);
        if (err != cudaSuccess) return err;
    }
    return cudaStreamSynchronize(nullptr);
}

}  // namespace

GPUSyncBuffer::GPUSyncBuffer(const axono::Tensor& tensor)
    : gpu_tensor_(tensor), modified_(false), num_elems_(tensor.num_elements()) {
    
    

    std::cerr << "[GPUSyncBuffer] is_cuda=" << tensor.is_cuda()
              << "  num_elems=" << num_elems_
              << "  gpu_data_ptr=" << tensor.data<float>() << std::endl;

    if (!tensor.is_cuda()) {
        throw std::runtime_error("喵！GPUSyncBuffer 只能接受 CUDA tensor！");
    }
    if (num_elems_ == 0) {
        throw std::runtime_error("喵！GPUSyncBuffer 收到空 tensor~");
    }
    if (tensor.data<float>() == nullptr) {
        throw std::runtime_error("喵！GPUSyncBuffer 收到 nullptr 数据指针~");
    }

    cudaError_t err = cudaMallocHost(&host_data_, num_elems_ * sizeof(float));
    if (err != cudaSuccess || host_data_ == nullptr) {
        throw std::runtime_error("喵！cudaMallocHost 失败: " +
                                std::string(cudaGetErrorString(err)));
    }
    // sync_from_gpu();
    modified_ = true;
}

GPUSyncBuffer::~GPUSyncBuffer() {
    if (modified_ && gpu_tensor_.is_cuda()) {
        sync_to_gpu();
    }
    
    if (host_data_) {
        cudaFreeHost(host_data_);
        host_data_ = nullptr;
    }
}

GPUSyncBuffer::GPUSyncBuffer(GPUSyncBuffer&& other) noexcept
    : host_data_(other.host_data_),
      gpu_tensor_(std::move(other.gpu_tensor_)),
      modified_(other.modified_),
      num_elems_(other.num_elems_) {
    
    other.host_data_ = nullptr;
    other.modified_ = false;
    other.num_elems_ = 0;
}

GPUSyncBuffer& GPUSyncBuffer::operator=(GPUSyncBuffer&& other) noexcept {
    if (this != &other) {
        if (host_data_) { cudaFreeHost(host_data_); }

        host_data_ = other.host_data_;
        gpu_tensor_ = std::move(other.gpu_tensor_);
        modified_ = other.modified_;
        num_elems_ = other.num_elems_;

        other.host_data_ = nullptr;
        other.modified_ = false;
        other.num_elems_ = 0;
    }
    return *this;
}

void GPUSyncBuffer::sync_from_gpu() {
    if (!gpu_tensor_.is_cuda() || !host_data_) return;

    const float* dev_ptr = gpu_tensor_.data<float>();

    cudaError_t err = raw_copy_d2h(dev_ptr, host_data_, num_elems_);
    if (err != cudaSuccess) {
        throw std::runtime_error("喵！从GPU同步失败: " + std::string(cudaGetErrorString(err)));
    }
    modified_ = false;
}

void GPUSyncBuffer::sync_to_gpu() {
    if (!gpu_tensor_.is_cuda() || !host_data_) return;
    cudaError_t err = raw_write_d2h(host_data_, gpu_tensor_.data<float>(), num_elems_);
    if (err != cudaSuccess)
        throw std::runtime_error("喵！sync_to_gpu 失败: " + std::string(cudaGetErrorString(err)));
    modified_ = false;
}

std::vector<py::ssize_t> calculate_strides(const std::vector<int64_t>& shape, py::ssize_t item_size) {
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride = item_size;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

std::vector<int64_t> shape_to_vector(const axono::Shape& shape) {
    return std::vector<int64_t>(shape.begin(), shape.end());
}

py::array tensor_to_sync_numpy(const axono::Tensor& tensor) {
    if (!tensor.is_cuda()) {
        throw std::runtime_error("喵！张量不在GPU上，无需同步缓冲QwQ");
    }
    
    auto* buffer = new GPUSyncBuffer(tensor);

    py::capsule buffer_capsule(buffer, [](void* ptr) {
        delete static_cast<GPUSyncBuffer*>(ptr);
    });
    
    auto shape_vec = shape_to_vector(tensor.shape());
    auto strides = calculate_strides(shape_vec, sizeof(float));
    
    auto array = py::array_t<float>(
        py::buffer_info(
            buffer->data(),
            sizeof(float),
            py::format_descriptor<float>::format(),
            shape_vec.size(),
            shape_vec,
            strides
        ),
        buffer_capsule
    );
    
    return array;
}
