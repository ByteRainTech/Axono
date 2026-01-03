#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "axono/core/tensor.h"
#include "axono/core/ops.h"

#ifdef COMPILED_WITH_CUDA
#include "axono/compute/cuda/ops/relu.h"
#endif
#include "axono/compute/cpu/ops/relu.h"

namespace axono {
namespace compute {
namespace ops {

py::object op_impl_relu(const py::args& args);
py::object op_impl_relu_(const py::args& args);

REGISTER_OP(relu) {
    core::Context ctx;
    core::Tensor result;
    core::Status status;
    if (args.size() != 1) {
        throw std::runtime_error("执行 add 需要传入 1 个 Tensor 喵~");
    }
    
    auto& input = pybind11::cast<core::Tensor&>(args[0]);
    core::Tensor output(input.dtype(), input.shape(), input.device());
    
    if (input.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
        status = cuda::ops::Relu(ctx, input, output);
#endif
    } else {
        status = cpu::ops::Relu(ctx, input, output);
    }
    if (status != core::Status::OK)
        throw std::runtime_error("执行 ReLU 时出现问题，错误代码：" + std::to_string(static_cast<int>(status)));

    return pybind11::cast(output);
}
REGISTER_OP(relu_) {
    core::Context ctx;
    core::Tensor result;
    core::Status status;
    if (args.size() != 1) {
        throw std::runtime_error("执行 add 需要传入 1 个 Tensor 喵~");
    }
    
    auto& tensor = pybind11::cast<core::Tensor&>(args[0]);
    
    if (tensor.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
        status = cuda::ops::ReluInplace(ctx, tensor);
#endif
    } else {
        status = cpu::ops::ReluInplace(ctx, tensor);
    }
    if (status != core::Status::OK)
        throw std::runtime_error("执行 ReLU 时出现问题，错误代码：" + std::to_string(static_cast<int>(status)));

    return pybind11::cast(tensor);
}
}
}
}
