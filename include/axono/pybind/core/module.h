#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "axono/core/module.h"

namespace py = pybind11;

void init_module(py::module &m) {
    py::class_<axono::core::Module>(m, "Module")
        .def(py::init<>(), "创建一个空的 Module 实例")
        .def("add_weight", 
             &axono::core::Module::add_weight, 
             py::arg("name"), py::arg("weight"), 
             "向模块添加权重张量")
        .def("get_weight", 
             &axono::core::Module::get_weight, 
             py::arg("name"), 
             py::return_value_policy::reference_internal,
             "获取指定名称的权重张量")
        .def("weights", 
             &axono::core::Module::weights, 
             py::return_value_policy::reference_internal, 
             "返回模块中所有权重的映射");
}
