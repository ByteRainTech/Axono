#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "axono/core/ops.h"

#ifdef COMPILED_WITH_CUDA
#include "axono/compute/cuda/operators/add.h"
#endif
#include "axono/compute/cpu/operators/add.h"

namespace axono {
namespace compute {
namespace operators {

py::object op_impl_add(const py::args& args);
py::object op_impl_add_scalar(const py::args& args);

REGISTER_OP(add) {
    if (args.size() != 2) {
        throw std::runtime_error("执行 add 需要传入 2 个 Tensor 喵~");
    }
    auto& a = pybind11::cast<core::Tensor&>(args[0]);
    auto& b = pybind11::cast<core::Tensor&>(args[1]);
    core::Context ctx;
    core::Tensor result = core::Tensor(a.dtype(), a.shape(), a.device());
    core::Status status;
    if (a.is_cuda())
#ifdef COMPILED_WITH_CUDA
        status = cuda::operators::Add(ctx, a, b, result);
#endif
    else 
        status = cpu::operators::Add(ctx, a, b, result);
    if (status != core::Status::OK)
        throw std::runtime_error("执行 add 时出现问题，错误代码：" + std::to_string(static_cast<int>(status)));

    return pybind11::cast(result);
}

REGISTER_OP(add_scalar) {
    if (args.size() != 2) {
        throw std::runtime_error("执行 add 需要传入 1 个 Tensor, 1 个 Scalar 喵~");
    }
    auto& a = pybind11::cast<core::Tensor&>(args[0]);
    py::object scalar = pybind11::cast<py::object>(args[1]);
    core::Context ctx;
    core::Tensor result;
    core::Status status;
    if (a.dtype() == core::DataType::FLOAT32) {
        float value = scalar.cast<float>();
        if (a.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
            status = cuda::operators::AddScalar(ctx, a, &value, sizeof(float), result);
#endif
        }
        else {
            status = cpu::operators::AddScalar(ctx, a, &value, sizeof(float), result);
        }
    }
    if (status != core::Status::OK) {
        throw std::runtime_error("执行 add_scalar 的时候出现问题，错误代码：" + std::to_string(static_cast<int>(status)));
    } else if (a.dtype() == core::DataType::INT32) {
        int32_t value = scalar.cast<int32_t>();
        if (a.is_cuda()) {
#ifdef COMPILED_WITH_CUDA
            status = cuda::operators::AddScalar(ctx, a, &value, sizeof(int32_t), result);
#endif
        }
        else {
            status = cpu::operators::AddScalar(ctx, a, &value, sizeof(int32_t), result);
        }

        if (status != core::Status::OK)
            throw std::runtime_error("执行 add_scalar 的时候出现问题，错误代码：" + std::to_string(static_cast<int>(status)));
    } else {
        throw std::runtime_error("当前类型不支持执行 add_scalar 操作喵~");
    }

    return pybind11::cast(result);
}

}
}
}
