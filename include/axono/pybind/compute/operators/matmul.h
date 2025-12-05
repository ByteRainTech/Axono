#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifdef COMPILED_WITH_CUDA
#include "axono/compute/cuda/operators/matmul.h"
#endif
#include "axono/compute/cpu/operators/matmul.h"

void init_matmul_operations(py::module &m) {
  m.def(
      "matmul",
      [](const axono::core::Tensor &a, const axono::core::Tensor &b) {
        axono::core::Context ctx;
        axono::core::Tensor result;
        axono::core::Status status;

        if (a.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
          size_t m = a.shape()[0];
          size_t n = b.shape()[1];
          auto result = axono::core::Tensor(
              a.dtype(), std::vector<size_t>{m, n}, a.device());
          status = axono::compute::cuda::operators::MatMul(ctx, a, b, result);
          return result;
#endif
        } else {
          status = axono::compute::cpu::operators::MatMul(ctx, a, b, result);
        }
        if (status != axono::core::Status::OK) {
          throw std::runtime_error("喵！Matmul 操作 出现错误！");
        }

        return result;
      },
      "Matrix multiplication of two tensors", py::arg("a"), py::arg("b"));
}
