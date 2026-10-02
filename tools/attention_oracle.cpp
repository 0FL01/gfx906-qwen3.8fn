// Standalone atomic R3b diagnostic. Link libggml and libggml-base ONLY;
// dynamically load the unchanged, parent-attested production libggml-hip.so.
// Public APIs/source checked at mx dcd685463d597d31f5ca759d32c94592a2740fa4.
// No model, own runtime/kernels, automatic quantizer, CPU attention or timing.
// Usage: core-attention-oracle NEW_OUTPUT.f32 [ABSOLUTE_libggml-hip.so]
// GCC: -std=c++20 -O2 -Wall -Wextra -Wpedantic -Werror -DGGML_SHARED,
// pinned ggml/include, original libggml.so + libggml-base.so, their rpath,
// -pthread. CORE_REVISION/CORE_DIRTY/CORE_ORACLE_LIBRARY_REVISION are optional
// build provenance; an available library attestation must match the pin.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Attention oracle requires finite checks and no fast-math"
#endif

namespace {
constexpr int dimension = 256, queries = 1, query_heads = 24, kv_heads = 2;
constexpr int kv_positions = 256, gqa = query_heads / kv_heads;
// This pin has NO public GGML_KQ_MASK_PAD or 64-query-row requirement:
// ggml.c:5516-5523, fattn-common.cuh:993-1000, llama-graph.cpp:1002.
// The selected cols_per_block=1 vector kernel reads only query row zero.
constexpr int mask_rows = queries;
constexpr int q4_block_elements = 32;
constexpr std::size_t q4_block_bytes = 18, q4_row_bytes = 144;
constexpr std::size_t output_elements = dimension * query_heads;
constexpr std::size_t f32_bytes = output_elements * sizeof(float);
constexpr std::size_t kv_bytes = q4_row_bytes * kv_positions * kv_heads;
constexpr std::size_t mask_bytes = kv_positions * mask_rows * sizeof(std::uint16_t);
constexpr float attention_scale = 0.0625f;
// SOURCE evidence, not a value introspected from the dynamically loaded binary.
constexpr float source_max_offset = 3.0f * 0.6931f;
constexpr const char* production_revision = "dcd685463d597d31f5ca759d32c94592a2740fa4";
constexpr const char* production_image = "llama.cpp-gfx906:pp-stream-dcd685463d";
constexpr const char* production_source = "/home/radneon/src/worktrees/qwen38-pp-trace-75";
constexpr const char* default_backend = "/core/build/oracle-production-libs/libggml-hip.so";
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(ggml_fp16_t) == 2 && sizeof(std::uint16_t) == 2);
static_assert(std::endian::native == std::endian::little && GGML_MAX_DIMS == 4);
static_assert(GGML_TYPE_Q4_0 == 2 && GGML_PREC_F32 == 10 && gqa == 12);
static_assert(f32_bytes == 24576 && kv_bytes == 73728 && mask_bytes == 512);

void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
std::string json_string(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 32 || c >= 127) {
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += static_cast<char>(c);
    }
    return result + '"';
}
std::uint32_t bits(float value) { return std::bit_cast<std::uint32_t>(value); }
void build_provenance(std::ostream& out) {
    out << ",\"compiler\":" << json_string(__VERSION__) << ",\"cplusplus\":" << __cplusplus
        << ",\"fast_math\":false,\"revision\":";
#ifdef CORE_REVISION
    out << json_string(CORE_REVISION);
#else
    out << "null";
#endif
    out << ",\"dirty\":";
#ifdef CORE_DIRTY
    out << (CORE_DIRTY ? "true" : "false");
#else
    out << "null";
#endif
    out << ",\"compiled_library_revision_attested\":";
#ifdef CORE_ORACLE_LIBRARY_REVISION
    out << json_string(CORE_ORACLE_LIBRARY_REVISION);
#else
    out << "null";
#endif
}

struct File {
    int fd = -1;
    explicit File(const char* path, int flags = O_RDONLY | O_CLOEXEC | O_NONBLOCK)
        : fd(::open(path, flags, 0644)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), std::string("open ") + path);
    }
    ~File() { if (fd >= 0 && ::close(fd) != 0) std::cerr << "attention oracle: file cleanup failed\n"; }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    struct stat status() const {
        struct stat s{};
        if (::fstat(fd, &s) != 0) throw std::system_error(errno, std::generic_category(), "fstat");
        require(S_ISREG(s.st_mode) && s.st_size >= 0, "expected a regular file");
        return s;
    }
    void close_checked() {
        const int descriptor = std::exchange(fd, -1);
        if (::close(descriptor) != 0) throw std::system_error(errno, std::generic_category(), "close");
    }
};
bool same_file_metadata(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
void check_backend_file(File& file, const char* path, const struct stat& original) {
    struct stat current_path{};
    if (::stat(path, &current_path) != 0)
        throw std::system_error(errno, std::generic_category(), "stat explicit backend path");
    require(same_file_metadata(original, file.status()) && same_file_metadata(original, current_path),
            "backend path/file metadata changed during the diagnostic");
}
void require_new_output(const char* path) {
    require(path && *path, "output path must not be empty");
    struct stat s{};
    if (::lstat(path, &s) == 0) throw std::runtime_error("output already exists (including symlinks)");
    if (errno != ENOENT) throw std::system_error(errno, std::generic_category(), "lstat new output");
    const auto parent = std::filesystem::path(path).parent_path();
    require(std::filesystem::is_directory(parent.empty() ? "." : parent), "output parent must already exist");
}
void write_output(const char* path, std::span<const float> output) {
    require(output.size() == output_elements, "wrong output element count");
    File file(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW);
    try {
        const auto bytes = std::as_bytes(output);
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto n = ::write(file.fd, bytes.data() + offset, bytes.size() - offset);
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::system_error(errno, std::generic_category(), "write new attention output");
            }
            require(n > 0, "zero-byte write of attention output");
            offset += static_cast<std::size_t>(n);
        }
        require(file.status().st_size == static_cast<off_t>(f32_bytes), "new output must be exactly 24576 bytes");
        file.close_checked();
    } catch (...) {
        // Only this exclusively created output is removed on a write failure.
        if (::unlink(path) != 0) std::cerr << "attention oracle: partial output cleanup failed\n";
        throw;
    }
}

struct Fixture {
    std::vector<float> q;
    std::vector<std::uint8_t> k, v;
    std::vector<std::uint16_t> mask;
    Fixture() : q(output_elements, 0.0f), k(kv_bytes), v(kv_bytes), mask(kv_positions * mask_rows, 0xfc00U) {
        // Canonical Q4_0 bytes: LE half scale 1 (00 3c), 16 packed codes.
        // Zero is code 8 in BOTH nibbles. Avoid d=0 and any uninitialized padding.
        for (std::size_t offset = 0; offset < kv_bytes; offset += q4_block_bytes) {
            k[offset] = v[offset] = 0x00;
            k[offset + 1] = v[offset + 1] = 0x3c;
            std::fill_n(k.begin() + static_cast<std::ptrdiff_t>(offset + 2), 16, 0x88);
            std::fill_n(v.begin() + static_cast<std::ptrdiff_t>(offset + 2), 16, 0x88);
        }
        for (int head = 0; head < kv_heads; ++head) {
            const std::size_t row = static_cast<std::size_t>(head) * kv_positions * q4_row_bytes;
            for (int block = 0; block < dimension / q4_block_elements; ++block) {
                const auto offset = row + static_cast<std::size_t>(block) * q4_block_bytes;
                std::fill_n(v.begin() + static_cast<std::ptrdiff_t>(offset + 2), 16, head == 0 ? 0x99 : 0xaa);
            }
        }
        // Only physical position zero is visible, including any unused padded
        // query rows if mask_rows is ever increased. A valid row avoids 0/0.
        for (int row = 0; row < mask_rows; ++row) mask[row * kv_positions] = 0x0000U;
        for (float value : q) require(bits(value) == 0, "Q must be canonical +0 F32");
        for (int head = 0; head < kv_heads; ++head) {
            for (int position = 0; position < kv_positions; ++position) {
                const std::size_t row = (static_cast<std::size_t>(head) * kv_positions + position) * q4_row_bytes;
                for (int block = 0; block < dimension / q4_block_elements; ++block) {
                    const auto offset = row + static_cast<std::size_t>(block) * q4_block_bytes;
                    require(offset <= kv_bytes && q4_block_bytes <= kv_bytes - offset,
                            "canonical Q4 block exceeds storage");
                    require(k[offset] == 0 && k[offset + 1] == 0x3c && v[offset] == 0 && v[offset + 1] == 0x3c,
                            "Q4 scale must be exactly F16 1");
                    const auto expected_v = position == 0 ? (head == 0 ? 0x99 : 0xaa) : 0x88;
                    for (std::size_t byte = 2; byte < q4_block_bytes; ++byte)
                        require(k[offset + byte] == 0x88 && v[offset + byte] == expected_v,
                                "Q4 codes differ from canonical recipe");
                }
            }
        }
        for (std::size_t i = 0; i < mask.size(); ++i)
            require(mask[i] == (i % kv_positions == 0 ? 0x0000U : 0xfc00U), "incorrect single-visible-key mask");
    }
};

struct TensorSpec {
    const char* name;
    ggml_type type;
    std::array<std::int64_t, 4> ne;
    std::array<std::size_t, 4> nb;
    std::size_t bytes, alignment;
};
constexpr std::array<TensorSpec, 5> specs{{
    {"Q", GGML_TYPE_F32, {256, 1, 24, 1}, {4, 1024, 1024, 24576}, f32_bytes, alignof(float)},
    {"K", GGML_TYPE_Q4_0, {256, 256, 2, 1}, {18, 144, 36864, 73728}, kv_bytes, alignof(std::uint16_t)},
    {"V", GGML_TYPE_Q4_0, {256, 256, 2, 1}, {18, 144, 36864, 73728}, kv_bytes, alignof(std::uint16_t)},
    {"mask", GGML_TYPE_F16, {256, mask_rows, 1, 1}, {2, 512, mask_bytes, mask_bytes}, mask_bytes, alignof(std::uint16_t)},
    {"output", GGML_TYPE_F32, {256, 24, 1, 1}, {4, 1024, 24576, 24576}, f32_bytes, alignof(float)}
}};
void validate_tensors(const std::array<ggml_tensor*, 5>& tensors, ggml_backend_buffer_t buffer) {
    const auto size = ggml_backend_buffer_get_size(buffer);
    const auto base = reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(buffer));
    require(base != 0 && size <= (1U << 20), "unexpected GGML device allocation span");
    std::array<std::uintptr_t, 5> addresses{};
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto* t = tensors[i];
        const auto& s = specs[i];
        require(t && t->type == s.type && !t->view_src && t->view_offs == 0 && t->buffer == buffer,
                std::string(s.name) + ": unexpected type, view or buffer");
        for (std::size_t axis = 0; axis < 4; ++axis)
            require(t->ne[axis] == s.ne[axis] && t->nb[axis] == s.nb[axis],
                    std::string(s.name) + ": incorrect shape or strides");
        require(ggml_is_contiguous(t) && ggml_nbytes(t) == s.bytes &&
                ggml_nelements(t) == s.ne[0] * s.ne[1] * s.ne[2] * s.ne[3],
                std::string(s.name) + ": incorrect element count or byte span");
        addresses[i] = reinterpret_cast<std::uintptr_t>(t->data);
        require(addresses[i] >= base && addresses[i] - base <= size &&
                s.bytes <= size - (addresses[i] - base) &&
                s.bytes <= std::numeric_limits<std::uintptr_t>::max() - addresses[i] &&
                addresses[i] % s.alignment == 0, std::string(s.name) + ": invalid tensor storage bounds/alignment");
        for (std::size_t j = 0; j < i; ++j)
            require(addresses[i] + s.bytes <= addresses[j] || addresses[j] + specs[j].bytes <= addresses[i],
                    "GGML input/output storage aliases");
    }
}

struct Registry {
    ggml_backend_reg_t reg = nullptr;
    explicit Registry(const char* path) : reg(ggml_backend_load(path)) {
        require(reg != nullptr, "loading explicit production HIP backend failed (see stderr)");
    }
    ~Registry() { if (reg) ggml_backend_unload(reg); }
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
};
struct DeviceProperties {
    std::string name, description, id;
    std::size_t memory_free = 0, memory_total = 0;
    ggml_backend_dev_caps caps{};
};
struct OracleResult {
    std::vector<float> output;
    std::string registry, backend, buffer;
    std::array<DeviceProperties, 2> devices;
    std::vector<std::pair<std::string, std::string>> features;
    bool features_available = false;
    std::size_t allocation_bytes = 0;
    std::array<std::int32_t, 4> op_params{};
};
OracleResult compute(const char* backend_path, const Fixture& fixture) {
    Registry registry(backend_path); // No auto-discovery, scheduler or CPU fallback.
    OracleResult result;
    const char* reg_name = ggml_backend_reg_name(registry.reg);
    require(reg_name && std::string_view(reg_name) == "ROCm", "explicit backend registry must be ROCm");
    result.registry = reg_name;
    require(ggml_backend_reg_dev_count(registry.reg) == 2, "production registry must expose exactly two GPUs");
    for (std::size_t i = 0; i < result.devices.size(); ++i) {
        auto* dev = ggml_backend_reg_dev_get(registry.reg, i);
        require(dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU, "registry contains a non-GPU device");
        ggml_backend_dev_props p{};
        ggml_backend_dev_get_props(dev, &p);
        require(p.type == GGML_BACKEND_DEVICE_TYPE_GPU && p.memory_total > 0, "invalid production GPU properties");
        result.devices[i] = {p.name ? p.name : "", p.description ? p.description : "", p.device_id ? p.device_id : "",
                             p.memory_free, p.memory_total, p.caps};
    }
    if (auto* address = ggml_backend_reg_get_proc_address(registry.reg, "ggml_backend_get_features")) {
        const auto get_features = reinterpret_cast<ggml_backend_get_features_t>(address);
        const auto* features = get_features(registry.reg);
        if (features) {
            result.features_available = true;
            std::size_t i = 0;
            for (; i < 64 && features[i].name; ++i)
                result.features.emplace_back(features[i].name, features[i].value ? features[i].value : "");
            require(i < 64, "unexpected unterminated backend feature list");
        }
    }
    auto* device = ggml_backend_reg_dev_get(registry.reg, 0);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    require(backend != nullptr && ggml_backend_get_device(backend.get()) == device, "GPU0 backend initialization failed");
    const char* backend_name = ggml_backend_name(backend.get());
    require(backend_name != nullptr, "missing backend name");
    result.backend = backend_name;
    require(ggml_blck_size(GGML_TYPE_Q4_0) == q4_block_elements && ggml_type_size(GGML_TYPE_Q4_0) == q4_block_bytes &&
            ggml_blck_size(GGML_TYPE_F32) == 1 && ggml_type_size(GGML_TYPE_F32) == sizeof(float) &&
            ggml_blck_size(GGML_TYPE_F16) == 1 && ggml_type_size(GGML_TYPE_F16) == sizeof(std::uint16_t),
            "loaded GGML type sizes/block sizes differ from the pinned recipe");
    constexpr std::size_t graph_capacity = 8;
    const auto context_bytes = specs.size() * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_capacity, false);
    require(context_bytes > 0 && context_bytes <= (1U << 20), "unexpected GGML context overhead");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init({context_bytes, nullptr, true}), ggml_free);
    require(context != nullptr, "small GGML context allocation failed");
    std::array<ggml_tensor*, 5> tensors{};
    for (std::size_t i = 0; i < 4; ++i) {
        const auto& s = specs[i];
        tensors[i] = ggml_new_tensor_4d(context.get(), s.type, s.ne[0], s.ne[1], s.ne[2], s.ne[3]);
        require(tensors[i] != nullptr, "GGML input tensor creation failed");
        ggml_set_name(tensors[i], s.name);
        ggml_set_input(tensors[i]);
    }
    auto* output = ggml_flash_attn_ext(context.get(), tensors[0], tensors[1], tensors[2], tensors[3],
                                       attention_scale, 0.0f, 0.0f);
    require(output != nullptr, "GGML flash attention tensor creation failed");
    tensors[4] = output;
    ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    ggml_set_name(output, "attention_oracle_pre_inverse_h64");
    ggml_set_output(output);
    auto* graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    require(graph != nullptr, "GGML graph creation failed");
    ggml_build_forward_expand(graph, output);
    require(ggml_graph_n_nodes(graph) == 1 && ggml_graph_node(graph, 0) == output &&
            output->op == GGML_OP_FLASH_ATTN_EXT && ggml_flash_attn_ext_get_prec(output) == GGML_PREC_F32,
            "expected a single ORIGINAL GGML_FLASH_ATTN_EXT node with requested F32 precision");
    for (std::size_t i = 0; i < GGML_MAX_SRC; ++i)
        require(output->src[i] == (i < 4 ? tensors[i] : nullptr), "unexpected FA source (including sinks)");
    std::memcpy(result.op_params.data(), output->op_params, sizeof(result.op_params));
    require(std::bit_cast<float>(result.op_params[0]) == attention_scale &&
            std::bit_cast<float>(result.op_params[1]) == 0.0f && std::bit_cast<float>(result.op_params[2]) == 0.0f &&
            result.op_params[3] == GGML_PREC_F32, "incorrect FA call metadata");
    require(ggml_backend_dev_supports_op(device, output), "production GPU0 backend does not support this FA node");
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    require(buffer != nullptr && !ggml_backend_buffer_is_host(buffer.get()) &&
            ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer.get())) == device, "expected a GPU0 device buffer");
    const char* buffer_name = ggml_backend_buffer_name(buffer.get());
    require(buffer_name != nullptr, "missing device buffer name");
    result.buffer = buffer_name;
    result.allocation_bytes = ggml_backend_buffer_get_size(buffer.get());
    require(result.allocation_bytes >= 2 * f32_bytes + 2 * kv_bytes + mask_bytes && result.allocation_bytes <= (1U << 20),
            "unexpected GGML tensor allocation size");
    validate_tensors(tensors, buffer.get()); // Before ANY backend tensor set/get.
    const std::array<std::span<const std::byte>, 4> inputs{
        std::as_bytes(std::span(fixture.q)), std::as_bytes(std::span(fixture.k)),
        std::as_bytes(std::span(fixture.v)), std::as_bytes(std::span(fixture.mask))};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        require(inputs[i].size() == specs[i].bytes, "incorrect host input span");
        ggml_backend_tensor_set(tensors[i], inputs[i].data(), 0, inputs[i].size());
    }
    std::vector<std::byte> scratch(kv_bytes);
    const auto check_inputs = [&] {
        validate_tensors(tensors, buffer.get());
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            require(inputs[i].size() <= scratch.size(), "input readback exceeds host scratch");
            ggml_backend_tensor_get(tensors[i], scratch.data(), 0, inputs[i].size());
            require(std::memcmp(scratch.data(), inputs[i].data(), inputs[i].size()) == 0,
                    std::string(specs[i].name) + ": input upload/readback or immutability check failed");
        }
    };
    check_inputs();
    // An unwritten output cannot accidentally satisfy the finite-execution gate.
    result.output.assign(output_elements, std::bit_cast<float>(std::uint32_t{0x7fc00000U}));
    ggml_backend_tensor_set(output, result.output.data(), 0, f32_bytes);
    const auto status = ggml_backend_graph_compute(backend.get(), graph);
    ggml_backend_synchronize(backend.get()); // Also drain before cleanup on a returned failure.
    require(status == GGML_STATUS_SUCCESS, "production ggml_backend_graph_compute failed, status=" + std::to_string(status));
    validate_tensors(tensors, buffer.get());
    ggml_backend_tensor_get(output, result.output.data(), 0, f32_bytes);
    for (std::size_t i = 0; i < result.output.size(); ++i)
        require(std::isfinite(result.output[i]), "nonfinite production output at index " + std::to_string(i) +
                ", F32 bits=" + std::to_string(bits(result.output[i])));
    check_inputs();
    // Destruction before returning: buffer, graph/context, backend (streams and
    // backend-owned scratch), registry. Public cleanup APIs are void.
    return result;
}

struct Errors {
    std::size_t count = 0, bit_mismatches = 0, worst_index = 0;
    double squared_error = 0, signed_error = 0, max_abs = 0, max_relative = 0;
    double value_sum = 0, factor_sum = 0, factor_min = std::numeric_limits<double>::infinity();
    double factor_max = -std::numeric_limits<double>::infinity();
    void add(float observed, float expected, std::size_t index) {
        const double error = static_cast<double>(observed) - expected;
        const double abs_error = std::abs(error), factor = static_cast<double>(observed) / expected;
        if (count == 0 || abs_error > max_abs) { max_abs = abs_error; worst_index = index; }
        max_relative = std::max(max_relative, abs_error / expected);
        squared_error += error * error;
        signed_error += error;
        value_sum += observed;
        factor_sum += factor;
        factor_min = std::min(factor_min, factor);
        factor_max = std::max(factor_max, factor);
        bit_mismatches += bits(observed) != bits(expected);
        ++count;
    }
    void json(std::ostream& out) const {
        require(count > 0, "empty error summary");
        out << "\"count\":" << count << ",\"ideal_bit_mismatches\":" << bit_mismatches
            << ",\"max_abs_error\":" << max_abs << ",\"max_relative_error\":" << max_relative
            << ",\"rms_error\":" << std::sqrt(squared_error / static_cast<double>(count))
            << ",\"mean_signed_error\":" << signed_error / static_cast<double>(count)
            << ",\"mean_value\":" << value_sum / static_cast<double>(count)
            << ",\"inferred_factor_mean\":" << factor_sum / static_cast<double>(count)
            << ",\"inferred_factor_min\":" << factor_min << ",\"inferred_factor_max\":" << factor_max;
    }
};
float expected_value(std::size_t index) { return index / dimension / gqa == 0 ? 1.0f : 2.0f; }
void example_json(std::ostream& out, const std::vector<float>& values, std::size_t index) {
    const float observed = values.at(index), expected = expected_value(index);
    out << "{\"index\":" << index << ",\"head\":" << index / dimension << ",\"lane\":" << index % dimension
        << ",\"expected\":" << expected << ",\"expected_f32_bits\":" << bits(expected)
        << ",\"observed\":" << observed << ",\"observed_f32_bits\":" << bits(observed)
        << ",\"signed_error\":" << static_cast<double>(observed) - expected
        << ",\"inferred_factor\":" << static_cast<double>(observed) / expected << '}';
}
void report(std::ostream& out, const OracleResult& r, const char* backend_path, const struct stat& library,
            const char* output_path) {
    Errors global;
    std::array<Errors, query_heads> heads{};
    for (std::size_t i = 0; i < r.output.size(); ++i) {
        const float expected = expected_value(i);
        global.add(r.output[i], expected, i);
        heads[i / dimension].add(r.output[i], expected, i);
    }
    out << "{\"kind\":\"attention_oracle\",\"protocol\":1,\"diagnostic_only\":true,\"passed\":true"
        << ",\"pass_criterion\":\"checked_setup_successful_finite_execution_not_ideal_match\""
        << ",\"scope\":\"synthetic_single_visible_key_production_q4_fa_pre_inverse_h64\""
        << ",\"control_scope\":\"canonical_contiguous_synthetic_KV_not_model_forward_or_logits_gate\""
        << ",\"model_loaded\":false,\"own_runtime_used\":false,\"cpu_attention_emulation\":false";
    build_provenance(out);
    out << ",\"production_revision\":" << json_string(production_revision)
        << ",\"production_source\":" << json_string(production_source)
        << ",\"source_pin_and_cleanliness_attested\":true,\"source_pin_runtime_verified\":false"
        << ",\"production_image_attested\":" << json_string(production_image)
        << ",\"library_revision_attested\":" << json_string(production_revision)
        << ",\"library_attestation_source\":\"parent_verified_production_image_export_not_rebuilt_by_tool\""
        << ",\"library_revision_runtime_verified\":false,\"backend_file_metadata_unchanged\":true"
        << ",\"backend_path\":" << json_string(backend_path)
        << ",\"backend_file_bytes\":" << library.st_size
        << ",\"backend_file_device\":" << static_cast<std::uint64_t>(library.st_dev)
        << ",\"backend_file_inode\":" << static_cast<std::uint64_t>(library.st_ino)
        << ",\"backend_file_mtime_sec\":" << library.st_mtim.tv_sec << ",\"backend_file_mtime_nsec\":" << library.st_mtim.tv_nsec
        << ",\"registry\":" << json_string(r.registry) << ",\"backend\":" << json_string(r.backend)
        << ",\"registry_device_count\":2,\"selected_device\":0,\"devices\":[";
    for (std::size_t i = 0; i < r.devices.size(); ++i) {
        const auto& d = r.devices[i];
        if (i) out << ',';
        out << "{\"index\":" << i << ",\"type\":\"GPU\",\"name\":" << json_string(d.name)
            << ",\"description\":" << json_string(d.description) << ",\"device_id\":" << json_string(d.id)
            << ",\"memory_free\":" << d.memory_free << ",\"memory_total\":" << d.memory_total
            << ",\"caps\":{\"async\":" << d.caps.async << ",\"host_buffer\":" << d.caps.host_buffer
            << ",\"buffer_from_host_ptr\":" << d.caps.buffer_from_host_ptr << ",\"events\":" << d.caps.events
            << ",\"mmap_support\":" << d.caps.mmap_support << "}}";
    }
    out << "],\"backend_features\":";
    if (!r.features_available) out << "null";
    else {
        out << '[';
        for (std::size_t i = 0; i < r.features.size(); ++i) {
            if (i) out << ',';
            out << "{\"name\":" << json_string(r.features[i].first) << ",\"value\":" << json_string(r.features[i].second) << '}';
        }
        out << ']';
    }
    out << ",\"buffer\":" << json_string(r.buffer) << ",\"allocation_bytes\":" << r.allocation_bytes
        << ",\"graph\":[\"GGML_FLASH_ATTN_EXT\"],\"graph_nodes\":1,\"graph_compute_calls\":1"
        << ",\"q_heads\":24,\"kv_heads\":2,\"gqa\":12,\"head_mapping\":\"kv_head=query_head/12\""
        << ",\"query_count\":1,\"visible_keys\":1,\"physical_kv_positions\":256,\"mask_query_padding_rows\":0"
        << ",\"scale\":" << attention_scale << ",\"scale_f32_bits\":" << bits(attention_scale)
        << ",\"max_bias\":0,\"logit_softcap\":0,\"precision_requested\":\"GGML_PREC_F32\""
        << ",\"precision_enum\":" << r.op_params[3] << ",\"f32_precision_guarantees_f32_v_accumulation\":false"
        << ",\"op_params_i32\":[" << r.op_params[0] << ',' << r.op_params[1] << ',' << r.op_params[2] << ',' << r.op_params[3] << ']'
        << ",\"tensors\":[";
    for (std::size_t i = 0; i < specs.size(); ++i) {
        const auto& s = specs[i];
        if (i) out << ',';
        out << "{\"name\":" << json_string(s.name) << ",\"type\":" << json_string(ggml_type_name(s.type)) << ",\"ne\":[";
        for (std::size_t axis = 0; axis < 4; ++axis) { if (axis) out << ','; out << s.ne[axis]; }
        out << "],\"nb\":[";
        for (std::size_t axis = 0; axis < 4; ++axis) { if (axis) out << ','; out << s.nb[axis]; }
        out << "],\"bytes\":" << s.bytes << '}';
    }
    out << "]"
        << ",\"input_recipe\":{\"byte_order\":\"little\",\"q_f32_bits\":0,\"q4_block_elements\":32,\"q4_block_bytes\":18"
        << ",\"q4_scale_f16_bits\":15360,\"q4_scale_le_hex\":\"003c\",\"q4_scale_value\":1"
        << ",\"q4_code_rule\":\"scale*(nibble-8); low nibble lanes0..15, high lanes16..31\""
        << ",\"q4_block_offset_rule\":\"(head*256+position)*144+block*18; block=lane/32\""
        << ",\"k_all_blocks\":\"003c followed by sixteen 88 bytes\""
        << ",\"v_position0_head0\":\"eight blocks: 003c followed by sixteen 99 bytes; exactly 1\""
        << ",\"v_position0_head1\":\"eight blocks: 003c followed by sixteen aa bytes; exactly 2\""
        << ",\"v_positions1_to255_all_heads\":\"003c followed by sixteen 88 bytes; finite zero\""
        << ",\"mask_position0_f16_bits\":0,\"mask_positions1_to255_f16_bits\":64512,\"mask_other_query_rows\":\"none\""
        << ",\"automatic_quantizer_used\":false,\"visible_v_half_rounding_error\":0}"
        << ",\"source_expected_dispatch\":\"flash_attn_ext_vec<256,1,GGML_TYPE_Q4_0,GGML_TYPE_Q4_0,false> on gfx906\""
        << ",\"kernel_dispatch_runtime_verified\":false"
        << ",\"source_max_offset_expression\":\"3.0f*0.6931f\",\"source_max_offset_f32\":" << source_max_offset
        << ",\"source_max_offset_f32_bits\":" << bits(source_max_offset) << ",\"max_offset_runtime_introspected\":false"
        << ",\"source_arithmetic\":\"half V dequantization, half unnormalized weights and V accumulators, float softmax denominator\""
        << ",\"source_references\":["
        << "{\"file\":\"ggml/include/ggml.h\",\"lines\":\"2450-2476\",\"function\":\"ggml_flash_attn_ext / set_prec / get_prec\"},"
        << "{\"file\":\"ggml/src/ggml.c\",\"lines\":\"5501-5562\",\"function\":\"ggml_flash_attn_ext\"},"
        << "{\"file\":\"ggml/src/ggml-cuda/fattn.cu\",\"lines\":\"358-533\",\"function\":\"ggml_cuda_get_best_fattn_kernel\"},"
        << "{\"file\":\"ggml/src/ggml-cuda/fattn-common.cuh\",\"lines\":\"19;409-431;975-1259\",\"function\":\"FATTN_KQ_MAX_OFFSET / dequantize_V_q4_0 / launch_fattn\"},"
        << "{\"file\":\"ggml/src/ggml-cuda/fattn-vec.cuh\",\"lines\":\"128-137;288-307;332-357;496-505;537-567\",\"function\":\"flash_attn_ext_vec / ggml_cuda_flash_attn_ext_vec_case\"},"
        << "{\"file\":\"src/llama-graph.cpp\",\"lines\":\"1002;2752-2768\",\"function\":\"mask allocation / build_attn_mha\"}]"
        << ",\"error_reference\":\"mathematical probability1; heads0..11=1, heads12..23=2; informational only\""
        << ",\"all_errors_from\":\"all_6144_observed_F32_output_elements\",\"errors\":{";
    global.json(out);
    out << ",\"worst_example\":";
    example_json(out, r.output, global.worst_index);
    out << "},\"per_head\":[";
    for (std::size_t head = 0; head < heads.size(); ++head) {
        if (head) out << ',';
        const auto begin = head * dimension;
        const float expected = expected_value(begin);
        bool uniform = true;
        for (std::size_t lane = 1; lane < dimension; ++lane)
            uniform = uniform && bits(r.output[begin + lane]) == bits(r.output[begin]);
        out << "{\"head\":" << head << ",\"kv_head\":" << head / gqa << ",\"expected_value\":" << expected
            << ",\"expected_f32_bits\":" << bits(expected) << ",\"uniform_f32_bits\":" << uniform << ',';
        heads[head].json(out);
        out << ",\"example\":";
        example_json(out, r.output, begin);
        out << ",\"worst_example\":";
        example_json(out, r.output, heads[head].worst_index);
        out << '}';
    }
    out << "],\"ideal_bit_equal\":" << (global.bit_mismatches == 0)
        << ",\"output\":" << json_string(output_path) << ",\"output_bytes\":24576,\"output_format\":\"little_endian_F32_lane_fast_then_head\""
        << ",\"output_exclusive_create\":true,\"output_sentinel_overwritten\":true,\"all_finite\":true"
        << ",\"inputs_checked_before_compute\":true,\"inputs_immutable_after_compute\":true"
        << ",\"shape_stride_bounds_checked\":true,\"cleanup_complete\":true,\"logits_gate_waived\":false}\n"
        << "{\"kind\":\"attention_oracle_complete\",\"protocol\":1,\"diagnostic_only\":true,\"cleanup_complete\":true,\"passed\":true}\n";
}
} // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic());
    try {
        require(argc == 2 || argc == 3, "usage: core-attention-oracle NEW_OUTPUT.f32 [ABSOLUTE_libggml-hip.so]");
#ifdef CORE_ORACLE_LIBRARY_REVISION
        require(std::string_view(CORE_ORACLE_LIBRARY_REVISION) == production_revision, "compiled library attestation differs from production pin");
#endif
        const char* backend_path = argc == 3 ? argv[2] : default_backend;
        require(std::filesystem::path(backend_path).is_absolute(), "backend path must be explicit and absolute");
        require(std::filesystem::path(backend_path).filename() == "libggml-hip.so", "backend must be the explicit production libggml-hip.so");
        require_new_output(argv[1]); // Validate NEW path before initializing any GPU.
        File backend_file(backend_path);
        const auto backend_stat = backend_file.status();
        require(backend_stat.st_size > 0, "empty backend library");
        check_backend_file(backend_file, backend_path, backend_stat);
        const Fixture fixture;
        const auto result = compute(backend_path, fixture); // All GGML resources freed on return.
        check_backend_file(backend_file, backend_path, backend_stat);
        backend_file.close_checked();
        // Format first so an unexpected formatting failure cannot publish a
        // partial JSON record. stdout is strict JSONL; raw F32 is the ONLY file.
        std::ostringstream json;
        json.imbue(std::locale::classic());
        json << std::boolalpha << std::setprecision(17);
        report(json, result, backend_path, backend_stat, argv[1]);
        require(static_cast<bool>(json), "JSON formatting failed");
        write_output(argv[1], result.output);
        std::cout << json.str();
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "{\"kind\":\"attention_oracle_error\",\"protocol\":1,\"diagnostic_only\":true,\"passed\":false,\"error\":"
                  << json_string(error.what()) << "}\n";
        std::cout.flush();
        return 1;
    }
}
