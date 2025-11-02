#!/bin/bash
set -e
mkdir -p build
cd build

cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make -j$(nproc)
