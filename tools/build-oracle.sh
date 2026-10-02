#!/bin/sh
# Compile only the separate oracle, using already-built production shared libs.
# No llama/common/core build, HIP compilation, GPU access or model load here.
set -eu

revision=dcd685463d597d31f5ca759d32c94592a2740fa4
if [ "$#" -ne 3 ]; then
    printf '%s\n' 'usage: ORACLE_LIBRARY_REVISION=<image revision label> sh tools/build-oracle.sh SOURCE_ROOT LIBRARY_DIR OUTPUT_PATH' >&2
    exit 2
fi
source_root=$1
library_dir=$2
output=$3
tool_dir=$(CDPATH= cd "$(dirname "$0")" && pwd -P)

if [ "${ORACLE_LIBRARY_REVISION:-}" != "$revision" ]; then
    printf '%s\n' "Set ORACLE_LIBRARY_REVISION to the verified origin of the libraries; expected $revision." >&2
    printf '%s\n' 'The production image revision label is suitable evidence. llama_version() is not a commit identifier.' >&2
    exit 1
fi
if [ ! -d "$source_root" ] || [ ! -d "$library_dir" ] || [ ! -d "$(dirname "$output")" ]; then
    printf '%s\n' 'Source, library and output parent directories must already exist.' >&2
    exit 1
fi
source_root=$(CDPATH= cd "$source_root" && pwd -P)
library_dir=$(CDPATH= cd "$library_dir" && pwd -P)
# This path is embedded as a C string and an ELF runtime search path.
case "$library_dir" in
    *'"'*|*'\'*|*':'*) printf '%s\n' 'LIBRARY_DIR cannot contain a quote, backslash or colon.' >&2; exit 1 ;;
esac
actual_revision=$(git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD)
if [ "$actual_revision" != "$revision" ]; then
    printf '%s\n' "Source revision mismatch: $actual_revision (expected $revision)." >&2
    exit 1
fi
if ! git -c "safe.directory=$source_root" -C "$source_root" diff --quiet HEAD -- include ggml src common cmake CMakeLists.txt; then
    printf '%s\n' 'Pinned mx source/header/build files have tracked modifications.' >&2
    exit 1
fi
for header in include/llama.h ggml/include/ggml.h ggml/include/ggml-cpu.h \
              ggml/include/ggml-backend.h ggml/include/ggml-opt.h ggml/include/gguf.h; do
    if [ ! -f "$source_root/$header" ]; then
        printf '%s\n' "Missing exact mx header: $header" >&2
        exit 1
    fi
done
resolve_library() {
    if [ -f "$library_dir/$1.so" ]; then
        printf '%s\n' "$library_dir/$1.so"
    elif [ -f "$library_dir/$1.so.0" ]; then
        printf '%s\n' "$library_dir/$1.so.0"
    else
        printf '%s\n' "Missing production shared library: $1.so (or .so.0)" >&2
        return 1
    fi
}
llama_lib=$(resolve_library libllama)
ggml_lib=$(resolve_library libggml)
base_lib=$(resolve_library libggml-base)
if [ -n "${CXX:-}" ]; then
    compiler=$CXX
elif [ -x /opt/rocm/llvm/bin/clang++ ]; then
    compiler=/opt/rocm/llvm/bin/clang++
else
    compiler=c++
fi

# Publish only a successfully linked executable. A failed build leaves the old
# output untouched and exits nonzero; no pipeline/tee can hide linker status.
temporary="${output}.tmp.$$"
trap 'rm -f "$temporary"' EXIT HUP INT TERM
"$compiler" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Wpedantic -Werror \
    -DLLAMA_SHARED -DGGML_SHARED \
    "-DCORE_ORACLE_LIBRARY_REVISION=\"$ORACLE_LIBRARY_REVISION\"" \
    "-DCORE_ORACLE_LIBRARY_DIR=\"$library_dir\"" \
    -I "$source_root/include" -I "$source_root/ggml/include" \
    "$tool_dir/oracle.cpp" "$llama_lib" "$ggml_lib" "$base_lib" \
    -pthread -Wl,--enable-new-dtags "-Wl,-rpath,$library_dir" \
    '-Wl,-rpath,$ORIGIN' "-Wl,-rpath-link,$library_dir" -o "$temporary"
mv -f "$temporary" "$output"
trap - EXIT HUP INT TERM
printf '%s\n' "Built separate diagnostic oracle: $output" "Pinned source and attested library revision: $revision"
