#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "axono/compute/cpu/operators.h"
#include "axono/compute/cpu/ops.h"
#include "axono/core/tensor.h"

namespace py = pybind11;

std::vector<size_t> calculate_strides(const std::vector<size_t> &shape,
                                      size_t element_size) {
  std::vector<size_t> strides(shape.size());
  size_t stride = element_size;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// 基础内存操作
void memory_copy_wrapper(py::bytes dst, py::bytes src) {
  char *dst_ptr;
  Py_ssize_t dst_len;
  char *src_ptr;
  Py_ssize_t src_len;

  PyBytes_AsStringAndSize(dst.ptr(), &dst_ptr, &dst_len);
  PyBytes_AsStringAndSize(src.ptr(), &src_ptr, &src_len);

  if (dst_len != src_len) {
    throw std::runtime_error(
        "Source and destination must have the same length");
  }

  axono::Context ctx;
  auto status = axono::compute::cpu::MemoryCopy(ctx, dst_ptr, src_ptr, dst_len);

  if (status != axono::Status::OK) {
    throw std::runtime_error("Memory copy failed");
  }
}

// Tensor Python 绑定
void init_tensor(py::module &m) {
  py::class_<axono::Tensor>(m, "Tensor")
      .def(py::init<>())
      .def(py::init<axono::DataType>())
      .def(py::init<axono::DataType, const std::vector<size_t> &>())
      .def_static("create", &axono::Tensor::Create)
      .def_static("create_like", &axono::Tensor::CreateLike)
      .def("reshape", &axono::Tensor::Reshape)
      .def("resize", &axono::Tensor::Resize)
      .def("fill_zero", &axono::Tensor::FillZero)
      .def("fill", &axono::Tensor::Fill)
      .def("copy_from",
           [](axono::Tensor &self, const axono::Tensor &other) {
             axono::Context ctx;
             return axono::compute::cpu::TensorCopy(ctx, self, other);
           })
      .def("is_same_shape", &axono::Tensor::IsSameShape)
      .def("__repr__", &axono::Tensor::ToString)
      .def("__str__", &axono::Tensor::ToString)
      .def_property_readonly("dtype", &axono::Tensor::dtype)
      .def_property_readonly("shape", &axono::Tensor::shape)
      .def_property_readonly("ndim", &axono::Tensor::ndim)
      .def_property_readonly("num_elements", &axono::Tensor::num_elements)
      .def_property_readonly("num_bytes", &axono::Tensor::num_bytes)
      .def(
          "data_int8",
          [](axono::Tensor &self) {
            auto strides = calculate_strides(self.shape(), sizeof(int8_t));
            return py::array_t<int8_t>(
                self.shape(), strides, self.data<int8_t>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as int8 numpy array (shared memory)")
      .def(
          "data_int16",
          [](axono::Tensor &self) {
            auto strides = calculate_strides(self.shape(), sizeof(int16_t));
            return py::array_t<int16_t>(
                self.shape(), strides, self.data<int16_t>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as int16 numpy array (shared memory)")
      .def(
          "data_int32",
          [](axono::Tensor &self) {
            auto strides = calculate_strides(self.shape(), sizeof(int32_t));
            return py::array_t<int32_t>(
                self.shape(), strides, self.data<int32_t>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as int32 numpy array (shared memory)")
      .def(
          "data_int64",
          [](axono::Tensor &self) {
            auto strides = calculate_strides(self.shape(), sizeof(int64_t));
            return py::array_t<int64_t>(
                self.shape(), strides, self.data<int64_t>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as int64 numpy array (shared memory)")
      .def(
          "data_float32",
          [](axono::Tensor &self) {
            auto strides = calculate_strides(self.shape(), sizeof(float));
            return py::array_t<float>(
                self.shape(), strides, self.data<float>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as float32 numpy array (shared memory)")
      .def(
          "data_float64",
          [](axono::Tensor &self) {
            auto strides = calculate_strides(self.shape(), sizeof(double));
            return py::array_t<double>(
                self.shape(), strides, self.data<double>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as float64 numpy array (shared memory)")
      .def(
          "data_bool",
          [](axono::Tensor &self) {
            auto strides = calculate_strides(self.shape(), sizeof(bool));
            return py::array_t<bool>(
                self.shape(), strides, self.data<bool>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as bool numpy array (shared memory)");
}

void init_matmul_operations(py::module &m) {
  m.def(
      "matmul",
      [](const axono::Tensor &a, const axono::Tensor &b) {
        axono::Context ctx;
        axono::Tensor result;

        auto status = axono::compute::cpu::MatMul(ctx, a, b, result);
        if (status != axono::Status::OK) {
          throw std::runtime_error("Matrix multiplication failed");
        }

        return result;
      },
      "Matrix multiplication of two tensors", py::arg("a"), py::arg("b"));
}

// 加法操作绑定
void init_add_operations(py::module &m) {
  m.def(
      "add",
      [](const axono::Tensor &a, const axono::Tensor &b) {
        axono::Context ctx;
        axono::Tensor result;

        auto status = axono::compute::cpu::Add(ctx, a, b, result);
        if (status != axono::Status::OK) {
          throw std::runtime_error("Add operation failed with status: " +
                                   std::to_string(static_cast<int>(status)));
        }

        return result;
      },
      "Element-wise addition of two tensors", py::arg("a"), py::arg("b"));

  m.def(
      "add_scalar",
      [](const axono::Tensor &a, py::object scalar) {
        axono::Context ctx;
        axono::Tensor result;

        // 将 Python 标量转换为 C++ 数据
        if (a.dtype() == axono::DataType::FLOAT32) {
          float value = scalar.cast<float>();
          auto status = axono::compute::cpu::AddScalar(ctx, a, &value,
                                                       sizeof(float), result);
          if (status != axono::Status::OK) {
            throw std::runtime_error("Add scalar operation failed");
          }
        } else if (a.dtype() == axono::DataType::INT32) {
          int32_t value = scalar.cast<int32_t>();
          auto status = axono::compute::cpu::AddScalar(ctx, a, &value,
                                                       sizeof(int32_t), result);
          if (status != axono::Status::OK) {
            throw std::runtime_error("Add scalar operation failed");
          }
        } else {
          throw std::runtime_error("Unsupported data type for scalar addition");
        }

        return result;
      },
      "Add scalar to tensor", py::arg("a"), py::arg("scalar"));
}

// ReLU 操作绑定
void init_activation_operations(py::module &m) {
  m.def(
      "relu",
      [](const axono::Tensor &input) {
        axono::Context ctx;
        axono::Tensor output;

        auto status = axono::compute::cpu::Relu(ctx, input, output);
        if (status != axono::Status::OK) {
          throw std::runtime_error("ReLU operation failed with status: " +
                                   std::to_string(static_cast<int>(status)));
        }

        return output;
      },
      "ReLU activation function", py::arg("input"));

  m.def(
      "relu_",
      [](axono::Tensor &tensor) {
        axono::Context ctx;

        auto status = axono::compute::cpu::ReluInplace(ctx, tensor);
        if (status != axono::Status::OK) {
          throw std::runtime_error("Inplace ReLU operation failed");
        }

        return tensor;
      },
      "Inplace ReLU activation function", py::arg("tensor"));
}

PYBIND11_MODULE(core, m) {
  m.doc() = "Axono Core Library";

  // 基础内存操作
  m.def("memory_copy", &memory_copy_wrapper);

  // 数据类型枚举
  py::enum_<axono::DataType>(m, "DataType")
      .value("INT8", axono::DataType::INT8)
      .value("INT16", axono::DataType::INT16)
      .value("INT32", axono::DataType::INT32)
      .value("INT64", axono::DataType::INT64)
      .value("FLOAT32", axono::DataType::FLOAT32)
      .value("FLOAT64", axono::DataType::FLOAT64)
      .value("BOOLEAN", axono::DataType::BOOLEAN)
      .export_values();

  // 状态枚举
  py::enum_<axono::Status>(m, "Status")
      .value("OK", axono::Status::OK)
      .value("INVALID_ARGUMENT", axono::Status::INVALID_ARGUMENT)
      .value("OUT_OF_MEMORY", axono::Status::OUT_OF_MEMORY)
      .value("UNSUPPORTED_TYPE", axono::Status::UNSUPPORTED_TYPE)
      .value("SHAPE_MISMATCH", axono::Status::SHAPE_MISMATCH)
      .value("INTERNAL_ERROR", axono::Status::INTERNAL_ERROR)
      .export_values();

  // 初始化 Tensor
  init_tensor(m);
  init_matmul_operations(m);
  init_add_operations(m);
  init_activation_operations(m);
}
