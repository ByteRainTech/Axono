#!/bin/bash
set -e
mkdir -p build
cd build

cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  cmake --build build --config Release
else
  make -j$(nproc)
fi
