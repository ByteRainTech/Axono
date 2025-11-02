#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "axono/compute/cpu/operators.h"
#include "axono/core/tensor.h"

namespace py = pybind11;

// 基础内存操作
void memory_copy_wrapper(py::bytes dst, py::bytes src) {
    char* dst_ptr;
    Py_ssize_t dst_len;
    char* src_ptr;  
    Py_ssize_t src_len;
    
    PyBytes_AsStringAndSize(dst.ptr(), &dst_ptr, &dst_len);
    PyBytes_AsStringAndSize(src.ptr(), &src_ptr, &src_len);
    
    if (dst_len != src_len) {
        throw std::runtime_error("Source and destination must have the same length");
    }
    
    axono::Context ctx;
    auto status = axono::compute::cpu::MemoryCopy(ctx, dst_ptr, src_ptr, dst_len);
    
    if (status != axono::Status::OK) {
        throw std::runtime_error("Memory copy failed");
    }
}

// Tensor Python 绑定
void init_tensor(py::module& m) {
    py::class_<axono::Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<axono::DataType>())
        .def(py::init<axono::DataType, const std::vector<size_t>&>())
        .def_static("create", &axono::Tensor::Create)
        .def_static("create_like", &axono::Tensor::CreateLike)
        .def("reshape", &axono::Tensor::Reshape)
        .def("resize", &axono::Tensor::Resize)
        .def("fill_zero", &axono::Tensor::FillZero)
        .def("fill", &axono::Tensor::Fill)
        .def("copy_from", [](axono::Tensor& self, const axono::Tensor& other) {
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
        // 数据访问方法
        .def("data_int8", [](axono::Tensor& self) {
            return py::array_t<int8_t>(self.shape(), self.data<int8_t>());
        }, "Get data as int8 numpy array")
        .def("data_int16", [](axono::Tensor& self) {
            return py::array_t<int16_t>(self.shape(), self.data<int16_t>());
        }, "Get data as int16 numpy array")
        .def("data_int32", [](axono::Tensor& self) {
            return py::array_t<int32_t>(self.shape(), self.data<int32_t>());
        }, "Get data as int32 numpy array")
        .def("data_int64", [](axono::Tensor& self) {
            return py::array_t<int64_t>(self.shape(), self.data<int64_t>());
        }, "Get data as int64 numpy array")
        .def("data_float32", [](axono::Tensor& self) {
            return py::array_t<float>(self.shape(), self.data<float>());
        }, "Get data as float32 numpy array")
        .def("data_float64", [](axono::Tensor& self) {
            return py::array_t<double>(self.shape(), self.data<double>());
        }, "Get data as float64 numpy array")
        .def("data_bool", [](axono::Tensor& self) {
            return py::array_t<bool>(self.shape(), self.data<bool>());
        }, "Get data as bool numpy array");
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
}
