#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept> // std::runtime_error

#include "./cuda/detail.h"

#include "axono/core/cpu/tensor/kernel.h"
#include "axono/core/cuda/tensor/kernel.h"
#include "axono/core/tensor.h"
#include "axono/core/types.h"

namespace {
// 自定义删除器，用于 shared_ptr
struct FreeDeleter {
  void operator()(void *ptr) const { std::free(ptr); }
};
} // namespace
namespace axono {
namespace core {

Tensor::Tensor() : dtype_(DataType::FLOAT32), num_elements_(0) {}

Tensor::Tensor(DataType dtype) : dtype_(dtype), num_elements_(0) {}

Tensor::Tensor(DataType dtype, const Shape &shape)
    : dtype_(dtype), shape_(shape) {
  num_elements_ = CalculateNumElements(shape_);
  InitializeStorage();
}

Tensor::Tensor(DataType dtype, const Shape &shape,
               const std::string &device) // 设备在这里喵
    : dtype_(dtype), shape_(shape), device_(device) {
  num_elements_ = CalculateNumElements(shape_);
  InitializeStorage();
}

Tensor::Tensor(DataType dtype, const Shape &shape, void *data)
    : dtype_(dtype), shape_(shape) {
  num_elements_ = CalculateNumElements(shape_);
  data_ = std::shared_ptr<void>(data, FreeDeleter());
}

Tensor::Tensor(const Tensor &other)
    : dtype_(other.dtype_), shape_(other.shape_),
      num_elements_(other.num_elements_) {
  if (other.data_) {
    InitializeStorage();
    std::memcpy(data_.get(), other.data_.get(), num_bytes());
  }
}

Tensor &Tensor::operator=(const Tensor &other) {
  if (this != &other) {
    dtype_ = other.dtype_;
    shape_ = other.shape_;
    num_elements_ = other.num_elements_;

    if (other.data_) {
      InitializeStorage();
      std::memcpy(data_.get(), other.data_.get(), num_bytes());
    } else {
      data_.reset();
    }
  }
  return *this;
}

Tensor::Tensor(Tensor &&other) noexcept
    : dtype_(other.dtype_), shape_(std::move(other.shape_)),
      num_elements_(other.num_elements_), data_(std::move(other.data_)) {
  other.num_elements_ = 0;
  other.shape_.clear();
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    dtype_ = other.dtype_;
    shape_ = std::move(other.shape_);
    num_elements_ = other.num_elements_;
    data_ = std::move(other.data_);

    other.num_elements_ = 0;
    other.shape_.clear();
  }
  return *this;
}

Tensor::~Tensor() = default;

Tensor Tensor::Create(DataType dtype, const Shape &shape) {
  return Tensor(dtype, shape);
}

Tensor Tensor::CreateLike(const Tensor &other) {
  return Tensor(other.dtype_, other.shape_);
}

Tensor Tensor::FromData(DataType dtype, const Shape &shape, void *data) {
  return Tensor(dtype, shape, data);
}

void Tensor::InitializeStorage() {
  if (num_elements_ == 0)
    return;
  size_t bytes = num_bytes();
  if (bytes == 0)
    return;

  if (device_.substr(0, 4) == "cuda") {
#ifdef COMPILED_WITH_CUDA
    data_ = cuda::detail::CudaAllocateStorage(bytes, device_);
#endif
  } else {
    // CPU HERE~
    void *ptr = std::malloc(bytes);
    if (ptr) {
      data_ = std::shared_ptr<void>(ptr, FreeDeleter());
      std::memset(ptr, 0, bytes);
    }
  }
}

Status Tensor::Reshape(const Shape &new_shape) {
  size_t new_num_elements = CalculateNumElements(new_shape);
  if (new_num_elements != num_elements_) {
    throw std::runtime_error("喵！Reshape要求形状相同的喵！");
  }
  shape_ = new_shape;
  return Status::OK;
}

Status Tensor::Resize(const Shape &new_shape) {
  size_t new_num_elements = CalculateNumElements(new_shape);
  if (new_num_elements != num_elements_) {
    shape_ = new_shape;
    num_elements_ = new_num_elements;
    InitializeStorage();
  } else {
    shape_ = new_shape;
  }
  return Status::OK;
}

Status Tensor::FillZero() {
  if (!data_)
    return Status::INVALID_ARGUMENT;

  switch (dtype_) {
  case DataType::INT8:
    if (this->is_cuda()) {
#ifdef COMPILED_WITH_CUDA
      cuda::tensor::DispatchZero(*this);
      break;
#endif
    }
    cpu::tensor::TensorZeroKernel(data<int8_t>(), num_elements_);
    break;
  case DataType::INT16:
    if (this->is_cuda()) {
#ifdef COMPILED_WITH_CUDA
      cuda::tensor::DispatchZero(*this);
      break;
#endif
    }
    cpu::tensor::TensorZeroKernel(data<int16_t>(), num_elements_);
    break;
  case DataType::INT32:
    if (this->is_cuda()) {
#ifdef COMPILED_WITH_CUDA
      cuda::tensor::DispatchZero(*this);
      break;
#endif
    }
    cpu::tensor::TensorZeroKernel(data<int32_t>(), num_elements_);
    break;
  case DataType::INT64:
    if (this->is_cuda()) {
#ifdef COMPILED_WITH_CUDA
      cuda::tensor::DispatchZero(*this);
      break;
#endif
    }
    cpu::tensor::TensorZeroKernel(data<int64_t>(), num_elements_);
    break;
  case DataType::FLOAT32:
    if (this->is_cuda()) {
#ifdef COMPILED_WITH_CUDA
      cuda::tensor::DispatchZero(*this);
      break;
#endif
    }
    cpu::tensor::TensorZeroKernel(data<float>(), num_elements_);
    break;
  case DataType::FLOAT64:
    if (this->is_cuda()) {
#ifdef COMPILED_WITH_CUDA
      cuda::tensor::DispatchZero(*this);
      break;
#endif
    }
    cpu::tensor::TensorZeroKernel(data<double>(), num_elements_);
    break;
  case DataType::BOOLEAN:
    if (this->is_cuda()) {
#ifdef COMPILED_WITH_CUDA
      cuda::tensor::DispatchZero(*this);
      break;
#endif
    }
    cpu::tensor::TensorZeroKernel(data<bool>(), num_elements_);
    break;
  default:
    return Status::UNSUPPORTED_TYPE;
  }
  return Status::OK;
}

Status Tensor::Fill(void *value, size_t value_size) {
  if (!data_)
    return Status::INVALID_ARGUMENT;
  if (this->is_cuda()) {
#ifdef COMPILED_WITH_CUDA
    return cuda::tensor::DispatchFill(*this, value, value_size);
#endif
  }
  return cpu::tensor::DispatchFill(*this, value, value_size);
}

bool Tensor::IsSameShape(const Tensor &other) const {
  return shape_ == other.shape_;
}

std::string Tensor::ToString() const {
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << shape_[i];
  }
  oss << "], dtype=";

  switch (dtype_) {
  case DataType::INT8:
    oss << "int8";
    break;
  case DataType::INT16:
    oss << "int16";
    break;
  case DataType::INT32:
    oss << "int32";
    break;
  case DataType::INT64:
    oss << "int64";
    break;
  case DataType::FLOAT32:
    oss << "float32";
    break;
  case DataType::FLOAT64:
    oss << "float64";
    break;
  case DataType::BOOLEAN:
    oss << "bool";
    break;
  default:
    oss << "unknown";
    break;
  }
  oss << ", device=" << device_ << ")";
  return oss.str();
}

} // namespace core
} // namespace axono
