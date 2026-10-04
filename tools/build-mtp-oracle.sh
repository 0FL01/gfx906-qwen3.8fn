#!/bin/sh
# Only this standalone C++ diagnostic, linked to existing production libraries.
# No library/core rebuild, HIP compilation, GPU access or model load.
set -eu

revision=dcd685463d597d31f5ca759d32c94592a2740fa4
if [ "$#" -ne 3 ]; then
    printf '%s\n' 'usage: ORACLE_LIBRARY_REVISION=<verified production image revision label> sh tools/build-mtp-oracle.sh SOURCE_ROOT LIBRARY_DIR OUTPUT_PATH' >&2
    exit 2
fi
source_root=$1
library_dir=$2
output=$3
tool_dir=$(CDPATH= cd "$(dirname "$0")" && pwd -P)
if [ "${ORACLE_LIBRARY_REVISION:-}" != "$revision" ]; then
    printf '%s\n' "Expected build-attested production library revision $revision." >&2
    printf '%s\n' 'Use the verified production image revision label, not llama_version().' >&2
    exit 1
fi
if [ ! -d "$source_root" ] || [ ! -d "$library_dir" ] || [ ! -d "$(dirname "$output")" ]; then
    printf '%s\n' 'Source, library and output parent directories must already exist.' >&2
    exit 1
fi
source_root=$(CDPATH= cd "$source_root" && pwd -P)
library_dir=$(CDPATH= cd "$library_dir" && pwd -P)
case "$library_dir" in
    *'"'*|*'\'*|*':'*) printf '%s\n' 'LIBRARY_DIR cannot contain a quote, backslash or colon.' >&2; exit 1 ;;
esac
actual_revision=$(git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD)
if [ "$actual_revision" != "$revision" ]; then
    printf '%s\n' "Source revision mismatch: $actual_revision (expected $revision)." >&2
    exit 1
fi
if ! git -c "safe.directory=$source_root" -C "$source_root" diff --quiet HEAD -- include ggml src common cmake CMakeLists.txt vendor/nlohmann; then
    printf '%s\n' 'Pinned source, headers, JSON dependency or build files have tracked modifications.' >&2
    exit 1
fi
for header in include/llama.h src/llama-ext.h src/llama-model.h \
              ggml/include/ggml.h ggml/include/ggml-backend.h vendor/nlohmann/json.hpp; do
    if [ ! -f "$source_root/$header" ]; then
        printf '%s\n' "Missing exact donor header: $header" >&2
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
# Restrict publication to a new diagnostic name; never replace a hot binary.
if [ "$(basename "$output")" != mtp-teacher-oracle ] || [ -e "$output" ] || [ -L "$output" ]; then
    printf '%s\n' 'OUTPUT_PATH must be a fresh executable named mtp-teacher-oracle.' >&2
    exit 1
fi
# Absolute sibling path also makes the self-test executable with a bare relative OUTPUT_PATH.
temporary="$(CDPATH= cd "$(dirname "$output")" && pwd -P)/$(basename "$output").tmp.$$"
trap 'rm -f "$temporary"' EXIT HUP INT TERM
"$compiler" -std=c++20 -O2 -DNDEBUG -Wall -Wextra -Wpedantic -Werror -ffp-contract=off \
    -DLLAMA_SHARED -DGGML_SHARED \
    "-DCORE_ORACLE_LIBRARY_REVISION=\"$ORACLE_LIBRARY_REVISION\"" \
    "-DCORE_ORACLE_LIBRARY_DIR=\"$library_dir\"" \
    -I "$source_root/include" -I "$source_root/src" -I "$source_root/ggml/include" \
    -isystem "$source_root/vendor/nlohmann" \
    "$tool_dir/mtp_teacher_oracle.cpp" "$llama_lib" "$ggml_lib" "$base_lib" \
    -pthread -Wl,--enable-new-dtags "-Wl,-rpath,$library_dir" \
    '-Wl,-rpath,$ORIGIN' "-Wl,-rpath-link,$library_dir" -o "$temporary"
# Mandatory model-free CPU regression against these actual production GGML libs.
# set -e and the trap prevent publishing any executable that fails its self-test.
"$temporary" --self-test
# Hard-link publication is atomic and fails if somebody claimed the output name.
ln "$temporary" "$output"
rm -f "$temporary"
trap - EXIT HUP INT TERM
printf '%s\n' "Built standalone MTP teacher oracle: $output" "Model-free CPU self-test: passed before publication" \
    "Pinned source and attested library revision: $revision"
