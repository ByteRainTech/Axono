#pragma once

#include "macros.h"
#include "types.h"
#include <iostream>
#include <memory>
#include <vector>

namespace axono {

class Tensor {
public:
  // 构造函数
  Tensor();
  explicit Tensor(DataType dtype);
  Tensor(DataType dtype, const Shape &shape);
  Tensor(DataType dtype, const Shape& shape, const std::string& device);
  Tensor(DataType dtype, const Shape &shape, void *data);

  // 拷贝构造函数和赋值操作符
  Tensor(const Tensor &other);
  Tensor &operator=(const Tensor &other);

  // 移动构造函数和赋值操作符
  Tensor(Tensor &&other) noexcept;
  Tensor &operator=(Tensor &&other) noexcept;

  // 析构函数
  ~Tensor();

  // 工厂函数
  static Tensor Create(DataType dtype, const Shape &shape);
  static Tensor CreateLike(const Tensor &other);
  static Tensor FromData(DataType dtype, const Shape &shape, void *data);

  // 基本信息
  const std::string& device() const { return device_; }
  bool is_cuda() const { return device_.substr(0, 4) == "cuda"; }
  DataType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
  size_t ndim() const { return shape_.size(); }
  size_t num_elements() const { return num_elements_; }
  size_t num_bytes() const { return num_elements_ * GetDataTypeSize(dtype_); }
  bool is_contiguous() const { return true; } // TODO

  // 数据访问
template <typename T>
T *data() {
  if (is_cuda()) [[likely]] {
    // 确保返回的是设备指针
    return static_cast<T*>(data_.get());
  }
  return static_cast<T*>(data_.get());
}

template <typename T>
const T *data() const {
  if (is_cuda()) [[likely]]
    return reinterpret_cast<const T *>(data_.get());
  return reinterpret_cast<const T *>(data_.get());
}

  void *raw_data() { return data_.get(); }
  const void *raw_data() const { return data_.get(); }

  // 形状操作
  Status Reshape(const Shape &new_shape);
  Status Resize(const Shape &new_shape);

  // 填充操作
  Status FillZero();
  Status Fill(void *value, size_t value_size);

  // 工具函数
  bool IsSameShape(const Tensor &other) const;
  std::string ToString() const;

  void* data() { return raw_data(); }
  const void* data() const { return raw_data(); }

private:
  DataType dtype_ = DataType::FLOAT32;
  Shape shape_;
  std::string device_; // 设备看这里喵，后续会出一个文档方便你们理解底层运行逻辑~希望越来越多人PR哦~
  size_t num_elements_ = 0;
  std::shared_ptr<void> data_;

  // 初始化数据存储
  void InitializeStorage();
};

} // namespace axono
