#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "axono/pybind/compute/operators/add.h"
#include "axono/pybind/compute/operators/matmul.h"
#include "axono/pybind/compute/ops/relu.h"
#include "axono/pybind/core/tensor.h"
#include "axono/pybind/core/module.h"

namespace py = pybind11;

PYBIND11_MODULE(libaxono, m) {
  m.doc() = "Axono Library";

  // 数据类型枚举
  py::enum_<axono::core::DataType>(m, "DataType")
      .value("INT8", axono::core::DataType::INT8)
      .value("INT16", axono::core::DataType::INT16)
      .value("INT32", axono::core::DataType::INT32)
      .value("INT64", axono::core::DataType::INT64)
      .value("FLOAT32", axono::core::DataType::FLOAT32)
      .value("FLOAT64", axono::core::DataType::FLOAT64)
      .value("BOOLEAN", axono::core::DataType::BOOLEAN)
      .export_values();

  // 状态枚举
  py::enum_<axono::core::Status>(m, "Status")
      .value("OK", axono::core::Status::OK)
      .value("INVALID_ARGUMENT", axono::core::Status::INVALID_ARGUMENT)
      .value("OUT_OF_MEMORY", axono::core::Status::OUT_OF_MEMORY)
      .value("UNSUPPORTED_TYPE", axono::core::Status::UNSUPPORTED_TYPE)
      .value("SHAPE_MISMATCH", axono::core::Status::SHAPE_MISMATCH)
      .value("INTERNAL_ERROR", axono::core::Status::INTERNAL_ERROR)
      .export_values();

  // 初始化 Tensor
  init_tensor(m);
  init_module(m);
  init_matmul_operations(m);
  init_add_operations(m);
  init_relu_operations(m);
}
