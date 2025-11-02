#include "axono/compute/cpu/operators.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace axono {
namespace python {

void memory_copy_python(py::array dst, py::array src) {
  py::object dst_flags = dst.attr("flags");
  py::object src_flags = src.attr("flags");

  if (!dst_flags.attr("c_contiguous").cast<bool>()) {
    throw std::runtime_error("Destination array must be C-contiguous");
  }
  if (!src_flags.attr("c_contiguous").cast<bool>()) {
    throw std::runtime_error("Source array must be C-contiguous");
  }

  auto dst_buf = dst.request();
  auto src_buf = src.request();

  if (dst_buf.size != src_buf.size) {
    throw std::runtime_error(
        "Source and destination arrays must have the same size");
  }

  axono::Context ctx;
  auto status = axono::compute::cpu::MemoryCopy(
      ctx, dst_buf.ptr, src_buf.ptr, dst_buf.size * dst_buf.itemsize);

  if (status != axono::Status::OK) {
    throw std::runtime_error("Memory copy failed");
  }
}

template <typename T>
void typed_memory_copy(py::array_t<T> dst, py::array_t<T> src) {
  py::object dst_flags = dst.attr("flags");
  py::object src_flags = src.attr("flags");

  if (!dst_flags.attr("c_contiguous").cast<bool>()) {
    throw std::runtime_error("Destination array must be C-contiguous");
  }
  if (!src_flags.attr("c_contiguous").cast<bool>()) {
    throw std::runtime_error("Source array must be C-contiguous");
  }

  auto dst_buf = dst.request();
  auto src_buf = src.request();

  if (dst_buf.size != src_buf.size) {
    throw std::runtime_error(
        "Source and destination arrays must have the same size");
  }

  axono::Context ctx;
  auto status = axono::compute::cpu::MemoryCopy(ctx, dst_buf.ptr, src_buf.ptr,
                                                dst_buf.size * sizeof(T));

  if (status != axono::Status::OK) {
    throw std::runtime_error("Memory copy failed");
  }
}

PYBIND11_MODULE(core, m) {
  m.doc() = "Axono Core Library - High Performance Computing Library";

  m.def("memory_copy", &memory_copy_python,
        "Copy memory from source to destination array", py::arg("dst"),
        py::arg("src"));

  m.def("memory_copy_int8", &typed_memory_copy<int8_t>, "Copy int8 arrays",
        py::arg("dst"), py::arg("src"));
  m.def("memory_copy_int16", &typed_memory_copy<int16_t>, "Copy int16 arrays",
        py::arg("dst"), py::arg("src"));
  m.def("memory_copy_int32", &typed_memory_copy<int32_t>, "Copy int32 arrays",
        py::arg("dst"), py::arg("src"));
  m.def("memory_copy_int64", &typed_memory_copy<int64_t>, "Copy int64 arrays",
        py::arg("dst"), py::arg("src"));
  m.def("memory_copy_float32", &typed_memory_copy<float>, "Copy float32 arrays",
        py::arg("dst"), py::arg("src"));
  m.def("memory_copy_float64", &typed_memory_copy<double>,
        "Copy float64 arrays", py::arg("dst"), py::arg("src"));

  py::enum_<axono::Status>(m, "Status")
      .value("OK", axono::Status::OK)
      .value("INVALID_ARGUMENT", axono::Status::INVALID_ARGUMENT)
      .value("OUT_OF_MEMORY", axono::Status::OUT_OF_MEMORY)
      .value("UNSUPPORTED_TYPE", axono::Status::UNSUPPORTED_TYPE)
      .value("INTERNAL_ERROR", axono::Status::INTERNAL_ERROR)
      .export_values();

  py::enum_<axono::DataType>(m, "DataType")
      .value("INT8", axono::DataType::INT8)
      .value("INT16", axono::DataType::INT16)
      .value("INT32", axono::DataType::INT32)
      .value("INT64", axono::DataType::INT64)
      .value("FLOAT32", axono::DataType::FLOAT32)
      .value("FLOAT64", axono::DataType::FLOAT64)
      .value("BOOLEAN", axono::DataType::BOOLEAN)
      .export_values();
}

} // namespace python
} // namespace axono
