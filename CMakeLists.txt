# 项目名称设定
project(axono CXX CUDA)

# where python3
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# POLICY
if(POLICY CMP0007)
    cmake_policy(SET CMP0007 NEW)
endif()

# 最低限度要求
cmake_minimum_required(VERSION 3.18)

# INFO 展示
function(showInfo text)
    if(CMAKE_HOST_UNIX)
        message("[32m-- ${text}[0m")
    else()
        message(STATUS "-- ${text}")
    endif()
endfunction()

# CMake 基本设定档
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/wizard/"
)

include(cmake/wizard/FindNanoBind.cmake)

# 对于 Nvidia 用户
option(WITH_CUDA "编译 Axono 时附带 CUDA 支持" OFF)
option(WITH_CINN "编译 Axono 时附带 CINN 支持" OFF)
option(WITH_NCCL "编译 Axono 时附带 NCCL 支持" OFF)

if(WITH_CUDA)
  add_definitions(-DWITH_CUDA)
  showInfo("CUDA Support - ON")
endif()
if(WITH_CINN)
  add_definitions(-DWITH_CINN)
  showInfo("CINN Support - ON")
endif()
if(WITH_NCCL)
  add_definitions(-DWITH_NCCL)
  showInfo("NCCL Support - ON")
endif()

# 第三方 THIRD PARTY
include(third_party)

# 核心编译
add_subdirectory(axono)
