#!/bin/bash
set -e
mkdir -p build
cd build

cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  make -j$(nproc)
else
  cmake --build build --config Release
fi
