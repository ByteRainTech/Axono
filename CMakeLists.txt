cmake_minimum_required(VERSION 3.15)
project(axono LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/library)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/library)

foreach(cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${cfg} CFG_UC)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG_UC} ${CMAKE_SOURCE_DIR}/python/library)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG_UC} ${CMAKE_SOURCE_DIR}/python/library)
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native -fPIC)
endif()

include_directories(include)

add_library(axono_core SHARED
    src/core/tensor.cpp
    src/compute/cpu/operators/memory_copy.cpp
    src/compute/cpu/operators/matmul.cpp
    src/compute/cpu/operators/add.cpp
    src/compute/cpu/ops/tensor_ops.cpp
    src/compute/cpu/ops/relu.cpp
)

target_compile_definitions(axono_core PRIVATE AXONO_CORE_EXPORTS)

set_target_properties(axono_core PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/library"  # .dll
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/library"  # .lib
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/library"  # .so/.dylib
)

set_target_properties(axono_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

set_target_properties(axono_core PROPERTIES
    OUTPUT_NAME "core"
    PREFIX ""
    SOVERSION 1
)

add_subdirectory(python)
