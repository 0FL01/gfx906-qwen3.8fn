#!/bin/sh
set -eu
# Run inside the existing cmake-4.4.3 container, source at /core/src.
cmake -S /core/src -B /core/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCORE_REVISION="${CORE_REVISION:-unknown}" -DCORE_DIRTY="${CORE_DIRTY:-ON}"
cmake --build /core/build -j 4
ctest --test-dir /core/build --output-on-failure
