find_package(pybind11 REQUIRED)

option(WITH_CUDA "Enable CUDA" OFF)

if(WITH_CUDA)
    enable_language(CUDA)
    file(GLOB_RECURSE CORE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cu
    )
else()
    file(GLOB_RECURSE CORE_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/*.cpp
    )
endif()

list(FILTER CORE_SOURCES EXCLUDE REGEX "\\.ipynb_checkpoints/")

pybind11_add_module(libaxono
    src/pybind11_module.cpp
    ${CORE_SOURCES}
)

if(WITH_CUDA)
target_compile_definitions(libaxono PRIVATE COMPILED_WITH_CUDA)
endif()

target_include_directories(libaxono PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(libaxono PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set output properties
set_target_properties(libaxono PROPERTIES
    OUTPUT_NAME "libaxono"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/axono/library"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/axono/library"
    PREFIX ""
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
