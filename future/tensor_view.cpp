#include "axono/core/tensor_view.h"

#include <numeric>
#include <stdexcept>

namespace axono {
namespace core {

TensorView::TensorView(Tensor &tensor, const std::vector<int> &shape)
    : tensor_(tensor), shape_(shape) {
  // Calculate default strides (row-major)
  strides_.resize(shape.size());
  int stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= shape[i];
  }
}

TensorView::TensorView(Tensor &tensor, const std::vector<int> &shape,
                       const std::vector<int> &strides)
    : tensor_(tensor), shape_(shape), strides_(strides) {
  if (shape.size() != strides.size()) {
    throw std::invalid_argument(
        "Shape and strides must have same dimensionality");
  }
}

void TensorView::validate_indices(const std::vector<int> &indices) const {
  if (indices.size() != shape_.size()) {
    throw std::out_of_range("Invalid number of indices");
  }

  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] < 0 || indices[i] >= shape_[i]) {
      throw std::out_of_range("Index out of bounds");
    }
  }
}

int TensorView::calculate_offset(const std::vector<int> &indices) const {
  int offset = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    offset += indices[i] * strides_[i];
  }
  return offset;
}

int TensorView::size() const {
  return std::accumulate(shape_.begin(), shape_.end(), 1,
                         std::multiplies<int>());
}

template <typename T>
T &TensorView::at(const std::vector<int> &indices) {
  validate_indices(indices);
  int offset = calculate_offset(indices);
  return tensor_.data<T>()[offset];
}

template <typename T>
const T &TensorView::at(const std::vector<int> &indices) const {
  validate_indices(indices);
  int offset = calculate_offset(indices);
  return tensor_.data<T>()[offset];
}

void TensorView::copy_to(Tensor &dest) const {
  if (dest.shape() != shape_) {
    throw std::invalid_argument("Destination tensor has incorrect shape");
  }

  // 根据数据类型选择适当的拷贝方式
  switch (tensor_.dtype()) {
    case DataType::FLOAT32: {
      float *src_data = tensor_.data<float>();
      float *dst_data = dest.data<float>();
      for (int i = 0; i < size(); ++i) {
        std::vector<int> indices(shape_.size());
        int temp = i;
        for (int j = shape_.size() - 1; j >= 0; --j) {
          indices[j] = temp % shape_[j];
          temp /= shape_[j];
        }
        dst_data[i] = at<float>(indices);
      }
      break;
    }
    // 添加其他数据类型的支持...
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

void TensorView::copy_from(const Tensor &src) {
  if (src.shape() != shape_) {
    throw std::invalid_argument("Source tensor has incorrect shape");
  }

  // 类似copy_to的实现...
}

TensorView create_contiguous_view(Tensor &tensor,
                                  const std::vector<int> &shape) {
  return TensorView(tensor, shape);
}

TensorView create_transpose_view(Tensor &tensor, int dim0, int dim1) {
  std::vector<int> new_shape = tensor.shape();
  std::swap(new_shape[dim0], new_shape[dim1]);

  std::vector<int> new_strides(tensor.ndim());
  int stride = 1;
  for (int i = tensor.ndim() - 1; i >= 0; --i) {
    new_strides[i] = stride;
    stride *= new_shape[i];
  }
  std::swap(new_strides[dim0], new_strides[dim1]);

  return TensorView(tensor, new_shape, new_strides);
}

TensorView create_reshape_view(Tensor &tensor, const std::vector<int> &shape) {
  int total_size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  if (total_size != tensor.num_elements()) {
    throw std::invalid_argument(
        "New shape must have same total size as original tensor");
  }
  return TensorView(tensor, shape);
}

}  // namespace core
}  // namespace axono
