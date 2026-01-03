#include <pybind11/pybind11.h>

namespace py = pybind11;

#ifdef COMPILED_WITH_CUDA
#include "axono/compute/cuda/operators/matmul.h"
#endif
#include "axono/compute/cpu/operators/matmul.h"

namespace axono {
namespace compute {
namespace operators {

py::object op_impl_matmul(const py::args& args);

REGISTER_OP(matmul) {
    if (args.size() != 2) {
        throw std::runtime_error("执行 add 需要传入 2 个 Tensor 喵~");
    }
    auto& a = pybind11::cast<core::Tensor&>(args[0]);
    auto& b = pybind11::cast<core::Tensor&>(args[1]);
    core::Context ctx;
    core::Tensor result;
    core::Status status;

    if (a.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
        size_t m = a.shape()[0];
        size_t n = b.shape()[1];
        auto result = core::Tensor(a.dtype(), std::vector<size_t>{m, n}, a.device());
        status = cuda::operators::MatMul(ctx, a, b, result);
#endif
    } else {
        status = compute::cpu::operators::MatMul(ctx, a, b, result);
    }
    if (status != core::Status::OK)
        throw std::runtime_error("执行 Matmul 时出现问题，错误代码：" + std::to_string(static_cast<int>(status)));

    return pybind11::cast(result);
}

}
}
}
