#!/usr/bin/env bash
set -e

cmake -B build -DCMAKE_BUILD_TYPE=Release

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  cmake --build build --config Release
else
  make -C build -j$(nproc)
fi
