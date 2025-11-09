#pragma once

#include <memory>
#include <vector>
#include "axono/core/tensor.h"

namespace axono {
namespace core {

class TensorView {
public:
    TensorView(Tensor& tensor, const std::vector<int>& shape);
    TensorView(Tensor& tensor, const std::vector<int>& shape,
               const std::vector<int>& strides);

    // 基本操作
    void copy_to(Tensor& dest) const;
    void copy_from(const Tensor& src);
    
    // 访问原始数据
    template<typename T>
    T* data() { return tensor_.data<T>(); }
    
    template<typename T>
    const T* data() const { return tensor_.data<T>(); }

    // 属性访问
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<int>& strides() const { return strides_; }
    int ndim() const { return shape_.size(); }
    int size() const;

    // 索引操作
    template<typename T>
    T& at(const std::vector<int>& indices);
    
    template<typename T>
    const T& at(const std::vector<int>& indices) const;

private:
    Tensor& tensor_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    
    int calculate_offset(const std::vector<int>& indices) const;
    void validate_indices(const std::vector<int>& indices) const;
};

// 创建连续视图
TensorView create_contiguous_view(Tensor& tensor, const std::vector<int>& shape);

// 创建转置视图
TensorView create_transpose_view(Tensor& tensor, int dim0, int dim1);

// 创建重塑视图
TensorView create_reshape_view(Tensor& tensor, const std::vector<int>& shape);

}  // namespace core
}  // namespace axono