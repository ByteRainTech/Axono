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

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native -fPIC -ffast-math -fopenmp -DNDEBUG)
    # 添加链接选项
    add_link_options(-fopenmp)
endif()

# 检测CPU架构并自动定义宏
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i386|i686" OR 
   CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64" OR
   CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64|i386|i686")
    message(STATUS "Detected x86 architecture, enabling AVX optimizations")
    add_compile_definitions(AXONO_USE_X86_INTRINSICS=1)
    
    # 检查AVX2支持
    include(CheckCXXCompilerFlag)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        if(COMPILER_SUPPORTS_AVX2)
            add_compile_options(-mavx2 -mfma)
            add_link_options(-mavx2 -mfma)
        endif()
    endif()
else()
    message(STATUS "Non-x86 architecture, using generic implementation")
    add_compile_definitions(AXONO_USE_X86_INTRINSICS=0)
endif()

# Find Python with the correct components
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)
find_package(pybind11 REQUIRED)

include_directories(include)

# Add Python subdirectory
add_subdirectory(python)
