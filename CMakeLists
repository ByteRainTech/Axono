cmake_minimum_required(VERSION 3.15)
project(axono LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

include_directories(include)

add_library(axono_cpu_core
    src/compute/cpu/operators/memory_copy.cpp
)

add_subdirectory(python)
