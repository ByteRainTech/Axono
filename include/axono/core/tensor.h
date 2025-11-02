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
  DataType dtype() const { return dtype_; }
  const Shape &shape() const { return shape_; }
  size_t ndim() const { return shape_.size(); }
  size_t num_elements() const { return num_elements_; }
  size_t num_bytes() const { return num_elements_ * GetDataTypeSize(dtype_); }
  bool is_contiguous() const { return true; } // 简化版本

  // 数据访问
  template <typename T> T *data() { return reinterpret_cast<T *>(data_.get()); }

  template <typename T> const T *data() const {
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

private:
  DataType dtype_ = DataType::FLOAT32;
  Shape shape_;
  size_t num_elements_ = 0;
  std::shared_ptr<void> data_;

  // 初始化数据存储
  void InitializeStorage();
};

// Tensor 操作函数
namespace compute {
namespace cpu {

// Tensor 内存拷贝
AXONO_EXPORT Status TensorCopy(const Context &ctx, Tensor &dst,
                               const Tensor &src);

// Tensor 填充
AXONO_EXPORT Status TensorFill(const Context &ctx, Tensor &tensor, void *value,
                               size_t value_size);
AXONO_EXPORT Status TensorFillZero(const Context &ctx, Tensor &tensor);

// Tensor 创建
AXONO_EXPORT Status TensorCreateLike(const Context &ctx, const Tensor &src,
                                     Tensor &dst);

} // namespace cpu
} // namespace compute

} // namespace axono
