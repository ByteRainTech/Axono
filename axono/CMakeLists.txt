# 寻找所有 Source Codes
file(GLOB_RECURSE AXONO_SRCS 
    "*.cc" "*.cpp" "*.cu" "*.cuh")

# 获取总文件数量
list(LENGTH AXONO_SRCS AXONO_SRCS_COUNT)

showInfo("For Axono Library, ${AXONO_SRCS_COUNT} files are required to be built.")

# 添加至 Axono 的 Library
add_library(axono SHARED ${AXONO_SRCS})
