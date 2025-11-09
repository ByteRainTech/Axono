#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "axono/compute/cpu/operators.h"
#include "axono/compute/cpu/ops.h"

#include "axono/compute/cuda/operators.cuh"
#include "../python/src/autosync/cuda_autosync.h"
#include "axono/compute/cuda/ops.cuh"
#include "axono/compute/cuda/kernel/tensor_kernel.cuh"

#include "axono/core/tensor.h"
#include "axono/core/cuda/gpu_sync_buffer.h"

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
    //   .def(py::init<>())
    //   .def(py::init<axono::DataType>())
    //   .def(py::init<axono::DataType, const std::vector<size_t> &>())
    //   .def(py::init<axono::DataType, const std::vector<size_t> &>(),
    //          py::arg("dtype"), py::arg("shape"))
      .def(py::init<axono::DataType, const std::vector<size_t> &, const std::string &>(),
             py::arg("dtype"), py::arg("shape"), py::arg("device"))
      .def_static("create", &axono::Tensor::Create)
      .def_static("create_like", &axono::Tensor::CreateLike)
      .def("reshape", &axono::Tensor::Reshape)
      .def("resize", &axono::Tensor::Resize)
      .def("fill_zero", &axono::Tensor::FillZero)
      .def("fill", &axono::Tensor::Fill)
      .def("copy_from",
           [](axono::Tensor &self, const axono::Tensor &other) {
             axono::Context ctx;
             if (self.is_cuda()){
                return axono::compute::cuda::TensorCopy(ctx, self, other);
             }
             return axono::compute::cpu::TensorCopy(ctx, self, other);
           })
      .def("is_same_shape", &axono::Tensor::IsSameShape)
      .def("__repr__", &axono::Tensor::ToString)
      .def("__str__", &axono::Tensor::ToString)
      .def("device", &axono::Tensor::device)
      .def("is_cuda", &axono::Tensor::is_cuda)
      .def_property_readonly("dtype", &axono::Tensor::dtype)
      .def_property_readonly("shape", &axono::Tensor::shape)
      .def_property_readonly("ndim", &axono::Tensor::ndim)
      .def_property_readonly("num_elements", &axono::Tensor::num_elements)
      .def_property_readonly("num_bytes", &axono::Tensor::num_bytes)
      .def(
          "data_int8",
          [](axono::Tensor &self) {
            if (self.is_cuda()){
                size_t num_elems = self.num_elements();
                auto host_data = std::make_unique<int8_t[]>(num_elems);
                auto status = axono::compute::cuda::kernel::TensorReadKernel(
                    self.data<int8_t>(), host_data.get(), num_elems);
                if (status != axono::Status::OK) {
                throw std::runtime_error("本喵建议您检查下精度，CUDA在读取 int8 数据时候出现问题，状态代码：" + std::to_string(static_cast<int>(status)));
                }
                auto strides = calculate_strides(self.shape(), sizeof(int8_t));
                int8_t* data_ptr = host_data.release();
                py::capsule free_when_done(data_ptr, [](void* ptr) {
                    delete[] static_cast<int8_t*>(ptr);
                });
                return py::array_t<int8_t>(
                    self.shape(),
                    strides,
                    data_ptr,
                    free_when_done
                );
            }
            auto strides = calculate_strides(self.shape(), sizeof(int8_t));
            return py::array_t<int8_t>(
                self.shape(), strides, self.data<int8_t>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as int8 numpy array")
      .def(
          "data_int16",
          [](axono::Tensor &self) {
            if (self.is_cuda()){
                size_t num_elems = self.num_elements();
                auto host_data = std::make_unique<int16_t[]>(num_elems);
                auto status = axono::compute::cuda::kernel::TensorReadKernel(
                    self.data<int16_t>(), host_data.get(), num_elems);
                if (status != axono::Status::OK) {
                throw std::runtime_error("本喵建议您检查下精度，CUDA在读取 int16 数据时候出现问题，状态代码：" + std::to_string(static_cast<int>(status)));
                }
                auto strides = calculate_strides(self.shape(), sizeof(int16_t));
                int16_t* data_ptr = host_data.release();
                py::capsule free_when_done(data_ptr, [](void* ptr) {
                    delete[] static_cast<int16_t*>(ptr);
                });
                return py::array_t<int16_t>(
                    self.shape(),
                    strides,
                    data_ptr,
                    free_when_done
                );
            }
            auto strides = calculate_strides(self.shape(), sizeof(int16_t));
            return py::array_t<int16_t>(
                self.shape(), strides, self.data<int16_t>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as int16 numpy array (shared memory)")
      .def(
          "data_int32",
          [](axono::Tensor &self) {
            if (self.is_cuda()){
                size_t num_elems = self.num_elements();
                auto host_data = std::make_unique<int32_t[]>(num_elems);
                auto status = axono::compute::cuda::kernel::TensorReadKernel(
                    self.data<int32_t>(), host_data.get(), num_elems);
                if (status != axono::Status::OK) {
                throw std::runtime_error("本喵建议您检查下精度，CUDA在读取 int32 数据时候出现问题，状态代码：" + std::to_string(static_cast<int>(status)));
                }
                auto strides = calculate_strides(self.shape(), sizeof(int32_t));
                int32_t* data_ptr = host_data.release();
                py::capsule free_when_done(data_ptr, [](void* ptr) {
                    delete[] static_cast<int32_t*>(ptr);
                });
                return py::array_t<int32_t>(
                    self.shape(),
                    strides,
                    data_ptr,
                    free_when_done
                );
            }
            auto strides = calculate_strides(self.shape(), sizeof(int32_t));
            return py::array_t<int32_t>(
                self.shape(), strides, self.data<int32_t>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as int32 numpy array (shared memory)")
      .def(
          "data_int64",
          [](axono::Tensor &self) {
            if (self.is_cuda()){
                size_t num_elems = self.num_elements();
                auto host_data = std::make_unique<int64_t[]>(num_elems);
                auto status = axono::compute::cuda::kernel::TensorReadKernel(
                    self.data<int64_t>(), host_data.get(), num_elems);
                if (status != axono::Status::OK) {
                throw std::runtime_error("本喵建议您检查下精度，CUDA在读取 int64 数据时候出现问题，状态代码：" + std::to_string(static_cast<int>(status)));
                }
                auto strides = calculate_strides(self.shape(), sizeof(int64_t));
                int64_t* data_ptr = host_data.release();
                py::capsule free_when_done(data_ptr, [](void* ptr) {
                    delete[] static_cast<int64_t*>(ptr);
                });
                return py::array_t<int64_t>(
                    self.shape(),
                    strides,
                    data_ptr,
                    free_when_done
                );
            }
            auto strides = calculate_strides(self.shape(), sizeof(int64_t));
            return py::array_t<int64_t>(
                self.shape(), strides, self.data<int64_t>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as int64 numpy array")
.def("data_float32",
     [](axono::Tensor &self){
         
         if (self.is_cuda()) {
            size_t num_elems = self.num_elements();
            auto host_data = std::make_unique<float[]>(num_elems);
            auto status = axono::compute::cuda::kernel::TensorReadKernel(
                self.data<float>(), host_data.get(), num_elems);
            if (status != axono::Status::OK) {
            throw std::runtime_error("本喵建议您检查下精度，CUDA在读取 float32 数据时候出现问题，状态代码：" + std::to_string(static_cast<int>(status)));
            }
            auto strides = calculate_strides(self.shape(), sizeof(float));
            float* data_ptr = host_data.release();
            py::capsule free_when_done(data_ptr, [](void* ptr) {
                delete[] static_cast<float*>(ptr);
            });
            return py::array_t<float>(
                self.shape(),
                strides,
                data_ptr,
                free_when_done
            );
            // return tensor_to_sync_numpy(self);
         }
         
         // CPU 路径
         auto strides = calculate_strides(self.shape(), sizeof(float));
         return py::array_t<float>(self.shape(), strides, self.data<float>(),
                                 py::capsule(self.data<void*>(), [](void*) {}));
     },
     "Get data as float32 numpy array")
      .def(
          "data_float64",
          [](axono::Tensor &self) {
            if (self.is_cuda()){
                size_t num_elems = self.num_elements();
                auto host_data = std::make_unique<double[]>(num_elems);
                auto status = axono::compute::cuda::kernel::TensorReadKernel(
                    self.data<double>(), host_data.get(), num_elems);
                if (status != axono::Status::OK) {
                throw std::runtime_error("本喵建议您检查下精度，CUDA在读取 float64 数据时候出现问题，状态代码：" + std::to_string(static_cast<int>(status)));
                }
                auto strides = calculate_strides(self.shape(), sizeof(double));
                double* data_ptr = host_data.release();
                py::capsule free_when_done(data_ptr, [](void* ptr) {
                    delete[] static_cast<double*>(ptr);
                });
                return py::array_t<double>(
                    self.shape(),
                    strides,
                    data_ptr,
                    free_when_done
                );
            }
            auto strides = calculate_strides(self.shape(), sizeof(double));
            return py::array_t<double>(
                self.shape(), strides, self.data<double>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as float64 numpy array")
      .def(
          "data_bool",
          [](axono::Tensor &self) {
            if (self.is_cuda()){
                size_t num_elems = self.num_elements();
                auto host_data = std::make_unique<bool[]>(num_elems);
                auto status = axono::compute::cuda::kernel::TensorReadKernel(
                    self.data<bool>(), host_data.get(), num_elems);
                if (status != axono::Status::OK) {
                throw std::runtime_error("本喵建议您检查下精度，CUDA在读取 float64 数据时候出现问题，状态代码：" + std::to_string(static_cast<int>(status)));
                }
                auto strides = calculate_strides(self.shape(), sizeof(bool));
                bool* data_ptr = host_data.release();
                py::capsule free_when_done(data_ptr, [](void* ptr) {
                    delete[] static_cast<bool*>(ptr);
                });
                return py::array_t<bool>(
                    self.shape(),
                    strides,
                    data_ptr,
                    free_when_done
                );
            }
            auto strides = calculate_strides(self.shape(), sizeof(bool));
            return py::array_t<bool>(
                self.shape(), strides, self.data<bool>(),
                py::capsule(self.data<void *>(), [](void *) {}));
          },
          "Get data as bool numpy array")
          .def("__copy__",  [](const axono::Tensor& self){
             return axono::Tensor(self.dtype(), self.shape(), self.device()); 
             })
            .def("__deepcopy__", [](const axono::Tensor& self, py::dict){
                return axono::Tensor(self.dtype(), self.shape(), self.device());
            });
}

void init_matmul_operations(py::module &m) {
  m.def(
      "matmul",
      [](const axono::Tensor &a, const axono::Tensor &b) {
        axono::Context ctx;
        axono::Tensor result;

        axono::Status status;

        if (a.is_cuda()){
            status = axono::compute::cuda::MatMul(ctx, a, b, result);
        } else{
            status = axono::compute::cpu::MatMul(ctx, a, b, result);
        }
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
        // axono::Tensor result;
        axono::Tensor result = axono::Tensor(a.dtype(), a.shape(), a.device());

        axono::Status status;
        if (a.is_cuda()){
            status = axono::compute::cuda::Add(ctx, a, b, result);
        }
        else{
            status = axono::compute::cpu::Add(ctx, a, b, result);
        }
        if (status != axono::Status::OK) {
          throw std::runtime_error("喵！计算矩阵加法的时候出现问题啦，错误代码：" +
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
        axono::Status status;

        // 将 Python 标量转换为 C++ 数据
        if (a.dtype() == axono::DataType::FLOAT32) {
          float value = scalar.cast<float>();
          if (a.is_cuda()){
            status = axono::compute::cuda::AddScalar(ctx, a, &value,sizeof(float), result);
          }
          else{
            status = axono::compute::cpu::AddScalar(ctx, a, &value, sizeof(float), result);
          }
        }
        if (status != axono::Status::OK) {
          throw std::runtime_error("Add scalar operation failed");
        } else if (a.dtype() == axono::DataType::INT32) {
          int32_t value = scalar.cast<int32_t>();
          if (a.is_cuda()){
            status = axono::compute::cuda::AddScalar(ctx, a, &value, sizeof(int32_t), result);
          } else {
            status = axono::compute::cpu::AddScalar(ctx, a, &value, sizeof(int32_t), result);
          }
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
        // axono::Tensor output = axono::Tensor(input.dtype(), input.shape(), input.device());
        axono::Tensor output = axono::Tensor::CreateLike(input);

        axono::Status status;
        if (input.is_cuda()){
            status = axono::compute::cuda::Relu(ctx, input, output);
        }
        else{
            status = axono::compute::cpu::Relu(ctx, input, output);
        }
        if (status != axono::Status::OK) {
          throw std::runtime_error("喵！ReLU计算时发生错误，错误代码: " +
                                   std::to_string(static_cast<int>(status)));
        }
        return output;
      },
      "ReLU activation function", py::arg("input"),
      py::return_value_policy::move
      ),

  m.def(
      "relu_",
      [](axono::Tensor &tensor) {
        axono::Context ctx;
        axono::Status status;
        if (tensor.is_cuda()){
            status = axono::compute::cuda::ReluInplace(ctx, tensor);
        } else {
            status = axono::compute::cpu::ReluInplace(ctx, tensor);
        }
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
