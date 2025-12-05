#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "axono/core/tensor.h"

#ifdef COMPILED_WITH_CUDA
#include "axono/compute/cuda/ops/relu.h"
#endif
#include "axono/compute/cpu/ops/relu.h"

void init_relu_operations(py::module &m) {
  m.def(
      "relu",
      [](const axono::core::Tensor &input) {
        axono::core::Context ctx;
        axono::core::Tensor output =
            axono::core::Tensor(input.dtype(), input.shape(), input.device());

        axono::core::Status status;
        if (input.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
          status = axono::compute::cuda::ops::Relu(ctx, input, output);
#endif
        } else {
          status = axono::compute::cpu::ops::Relu(ctx, input, output);
        }

        if (status != axono::core::Status::OK) {
          throw std::runtime_error("喵！ReLU计算时发生错误，错误代码: " +
                                   std::to_string(static_cast<int>(status)));
        }
        return output;
      },
      "ReLU activation function", py::arg("input"),
      py::return_value_policy::move),

      m.def(
          "relu_",
          [](axono::core::Tensor &tensor) {
            axono::core::Context ctx;
            axono::core::Status status;
            if (tensor.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
              status = axono::compute::cuda::ops::ReluInplace(ctx, tensor);
#endif
            } else {
              status = axono::compute::cpu::ops::ReluInplace(ctx, tensor);
            }
            if (status != axono::core::Status::OK) {
              throw std::runtime_error("喵！InplaceReLU 出现错误！");
            }

            return tensor;
          },
          "Inplace ReLU activation function", py::arg("tensor"));
}
