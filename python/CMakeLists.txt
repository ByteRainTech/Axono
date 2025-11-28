find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

option(WITH_CUDA "Enable CUDA" OFF)

if(WITH_CUDA)
    enable_language(CUDA)
    file(GLOB_RECURSE CORE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu
    )
    target_compile_definitions(axono PRIVATE COMPILED_WITH_CUDA)
else()
    file(GLOB_RECURSE CORE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
    )
endif()

list(FILTER CORE_SOURCES EXCLUDE REGEX "\\.ipynb_checkpoints/")

pybind11_add_module(axonolib
    src/pybind11_module.cpp
    ${CORE_SOURCES}
)

target_include_directories(axonolib PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(axonolib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set output properties
set_target_properties(axonolib PROPERTIES
    OUTPUT_NAME "axonolib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/library"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/library"
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
