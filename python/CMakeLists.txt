find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

pybind11_add_module(core
    src/pybind11_module.cpp
)

target_link_libraries(core PRIVATE axono_core)

target_include_directories(core PRIVATE ${CMAKE_SOURCE_DIR}/include)

set_target_properties(core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/axono/library/
    PREFIX ""
    SUFFIX ".so"
)
