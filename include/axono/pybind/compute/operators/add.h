#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifdef COMPILED_WITH_CUDA
#include "axono/compute/cuda/operators/add.h"
#endif
#include "axono/compute/cpu/operators/add.h"

void init_add_operations(py::module &m) {
  m.def(
      "add",
      [](const axono::core::Tensor &a, const axono::core::Tensor &b) {
        axono::core::Context ctx;
        axono::core::Tensor result =
            axono::core::Tensor(a.dtype(), a.shape(), a.device());

        axono::core::Status status;
        if (a.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
          status = axono::compute::cuda::operators::Add(ctx, a, b, result);
#endif
        } else {
          status = axono::compute::cpu::operators::Add(ctx, a, b, result);
        }
        if (status != axono::core::Status::OK) {
          throw std::runtime_error(
              "喵！计算矩阵加法的时候出现问题啦，错误代码：" +
              std::to_string(static_cast<int>(status)));
        }

        return result;
      },
      "Element-wise addition of two tensors", py::arg("a"), py::arg("b"));

  m.def(
      "add_scalar",
      [](const axono::core::Tensor &a, py::object scalar) {
        axono::core::Context ctx;
        axono::core::Tensor result;
        axono::core::Status status;

        // 将 Python 标量转换为 C++ 数据
        if (a.dtype() == axono::core::DataType::FLOAT32) {
          float value = scalar.cast<float>();
          if (a.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
            status = axono::compute::cuda::operators::AddScalar(
                ctx, a, &value, sizeof(float), result);
#endif
          } else {
            status = axono::compute::cpu::operators::AddScalar(
                ctx, a, &value, sizeof(float), result);
          }
        }
        if (status != axono::core::Status::OK) {
          throw std::runtime_error("Add scalar operation failed");
        } else if (a.dtype() == axono::core::DataType::INT32) {
          int32_t value = scalar.cast<int32_t>();
          if (a.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
            status = axono::compute::cuda::operators::AddScalar(
                ctx, a, &value, sizeof(int32_t), result);
#endif
          } else {
            status = axono::compute::cpu::operators::AddScalar(
                ctx, a, &value, sizeof(int32_t), result);
          }

          if (status != axono::core::Status::OK) {
            throw std::runtime_error("喵！Add 操作出现了一些问题~");
          }
        } else {
          throw std::runtime_error("喵！当前类型不支持执行Add操作喵~");
        }

        return result;
      },
      "Add scalar to tensor", py::arg("a"), py::arg("scalar"));
}
