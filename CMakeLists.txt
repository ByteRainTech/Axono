cmake_minimum_required(VERSION 3.15)
project(axono LANGUAGES CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/axono/library)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/axono/library)

foreach(cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${cfg} CFG_UC)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CFG_UC} ${CMAKE_SOURCE_DIR}/python/axono/library)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CFG_UC} ${CMAKE_SOURCE_DIR}/python/axono/library)
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native -fPIC)
endif()

# Find Python with the correct components
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)
find_package(pybind11 REQUIRED)

include_directories(include)

# Add Python subdirectory
add_subdirectory(python)
