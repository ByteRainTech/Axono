# 帮助寻找 NanoBind 库
execute_process(
   COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
   OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

showInfo("Found NanoBind: ${nanobind_ROOT}")

find_package(nanobind CONFIG REQUIRED)
