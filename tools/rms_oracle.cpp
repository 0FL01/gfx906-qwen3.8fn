// Standalone R3b diagnostic; link model.cpp, libggml and libggml-base ONLY.
// Public APIs checked against production mx dcd685463d597d31f5ca759d32c94592a2740fa4.
// qwen4exp.cpp::build_hc_mix: per-branch RMS, then unchanged full-width gamma.
// [2560,4] MUL is the same elementwise math as its reshape to [10240,1].
// Execution uses the ORIGINAL production HIP library, never own blocks.hip,
// llama_decode, a CPU RMS emulation or a model runtime. Session1-e layer1
// ple_residual position0 is supplied explicitly; no manifest parsing or timing.
#include "model.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
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
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "RMS oracle requires finite checks and no fast-math"
#endif

namespace {
constexpr int hidden = 2560, branches = 4, wide = hidden * branches;
constexpr std::size_t f32_bytes = wide * sizeof(float);
constexpr float required_epsilon = 1.0e-6f;
constexpr const char* gamma_name = "blk.1.hc_attn_norm.weight";
constexpr const char* production_revision = "dcd685463d597d31f5ca759d32c94592a2740fa4";
constexpr const char* parent_source_pin = "e9f1dfe";
constexpr const char* default_backend = "/core/build/oracle-production-libs/libggml-hip.so";
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little && GGML_MAX_DIMS == 4);

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
void build_provenance() {
    std::cout << ",\"parent_source_pin\":" << json_string(parent_source_pin) << ",\"revision\":";
#ifdef CORE_REVISION
    std::cout << json_string(CORE_REVISION);
#else
    std::cout << "null";
#endif
    std::cout << ",\"dirty\":";
#ifdef CORE_DIRTY
    std::cout << (CORE_DIRTY ? "true" : "false");
#else
    std::cout << "null";
#endif
}

struct File {
    int fd = -1;
    explicit File(const char* path, int flags = O_RDONLY | O_CLOEXEC | O_NONBLOCK)
        : fd(::open(path, flags, 0644)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), std::string("open ") + path);
    }
    ~File() {
        if (fd >= 0 && ::close(fd) != 0) std::cerr << "rms oracle: file cleanup failed\n";
    }
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
bool aliases(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino;
}
void finite_values(std::span<const float> values, std::string_view what) {
    for (std::size_t i = 0; i < values.size(); ++i)
        require(std::isfinite(values[i]), std::string(what) + ": nonfinite element " + std::to_string(i));
}
std::vector<float> read_f32(File& file) {
    require(file.status().st_size == static_cast<off_t>(f32_bytes), "F32 input must be exactly 40960 bytes");
    std::vector<float> result(wide);
    auto bytes = std::as_writable_bytes(std::span(result));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto n = ::pread(file.fd, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "pread F32 input");
        }
        require(n > 0, "F32 input truncated during read");
        offset += static_cast<std::size_t>(n);
    }
    require(file.status().st_size == static_cast<off_t>(f32_bytes), "F32 input changed size during read");
    finite_values(result, "input");
    return result;
}

struct Parameters {
    float epsilon;
    qwen::TensorView view;
    std::vector<float> gamma;
    explicit Parameters(const qwen::Model& model) : epsilon(0), view(model.tensor(gamma_name)), gamma(wide) {
        const auto& arch = model.metadata_value("general.architecture");
        require(arch.type == qwen::MetadataType::STRING && arch.get<std::string>() == "qwen4exp",
                "expected qwen4exp GGUF");
        const auto number = [&](const char* name) -> std::uint64_t {
            const auto& value = model.metadata_value(name);
            if (value.type == qwen::MetadataType::UINT32) return value.get<std::uint32_t>();
            if (value.type == qwen::MetadataType::UINT64) return value.get<std::uint64_t>();
            throw std::runtime_error(std::string("invalid geometry metadata: ") + name);
        };
        require(number("qwen4exp.embedding_length") == hidden &&
                number("qwen4exp.hyper_connection.count") == branches && number("qwen4exp.block_count") == 48,
                "expected hidden2560, HC4 and 48 layers");
        const auto& eps = model.metadata_value("qwen4exp.attention.layer_norm_rms_epsilon");
        require(eps.type == qwen::MetadataType::FLOAT32, "RMS epsilon must have FLOAT32 metadata type");
        epsilon = eps.get<float>();
        require(std::isfinite(epsilon) && epsilon == required_epsilon, "expected RMS epsilon 1e-6");
        require(view.type == qwen::TensorType::F32 && view.rank == 1 && view.dimensions[0] == wide &&
                view.elements == wide && view.byte_size == f32_bytes && view.strides[0] == sizeof(float),
                "gamma must be contiguous rank1 F32[10240]");
        require(view.relative_offset % model.alignment() == 0 && view.file_offset >= model.data_offset() &&
                view.file_offset - model.data_offset() == view.relative_offset &&
                view.file_offset <= model.file_size() && view.byte_size <= model.file_size() - view.file_offset,
                "gamma offsets or access span invalid");
        model.read_tensor(gamma_name, std::as_writable_bytes(std::span(gamma)));
        finite_values(gamma, "gamma");
    }
};

struct Registry {
    ggml_backend_reg_t reg = nullptr;
    explicit Registry(const char* path) : reg(ggml_backend_load(path)) {
        require(reg != nullptr, "loading explicit production HIP backend failed (see stderr)");
    }
    ~Registry() { if (reg) ggml_backend_unload(reg); }
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;
};
struct OracleResult {
    std::vector<float> output;
    std::string backend, device, description, device_id;
    std::size_t allocation_bytes = 0;
};
OracleResult compute(const char* backend_path, const Parameters& p, std::span<const float> input) {
    Registry registry(backend_path); // No backend auto-discovery or CPU fallback.
    const std::string reg_name = ggml_backend_reg_name(registry.reg);
    require(reg_name == "ROCm" || reg_name == "HIP", "explicit backend is not HIP/ROCm");
    require(ggml_backend_reg_dev_count(registry.reg) == 2, "production backend must expose exactly two GPUs");
    for (std::size_t i = 0; i < 2; ++i)
        require(ggml_backend_dev_type(ggml_backend_reg_dev_get(registry.reg, i)) == GGML_BACKEND_DEVICE_TYPE_GPU,
                "production registry contains a non-GPU device");
    auto* device = ggml_backend_reg_dev_get(registry.reg, 0);
    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(device, &props);
    OracleResult result;
    result.backend = reg_name;
    result.device = props.name ? props.name : "";
    result.description = props.description ? props.description : "";
    result.device_id = props.device_id ? props.device_id : "";
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    require(backend != nullptr, "production HIP backend initialization failed");
    constexpr std::size_t graph_capacity = 8;
    const std::size_t context_bytes = 4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_capacity, false);
    require(context_bytes > 0 && context_bytes <= (1U << 20), "unexpected GGML context overhead");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
        ggml_init({context_bytes, nullptr, true}), ggml_free);
    require(context != nullptr, "small GGML context allocation failed");
    auto* x = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, hidden, branches);
    auto* gamma = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, hidden, branches);
    auto* rms = ggml_rms_norm(context.get(), x, p.epsilon);
    auto* output = ggml_mul(context.get(), rms, gamma);
    ggml_set_input(x);
    ggml_set_input(gamma);
    ggml_set_output(output);
    auto* graph = ggml_new_graph_custom(context.get(), graph_capacity, false);
    ggml_build_forward_expand(graph, output);
    require(ggml_graph_n_nodes(graph) == 2 && ggml_graph_node(graph, 0) == rms &&
            ggml_graph_node(graph, 1) == output && rms->op == GGML_OP_RMS_NORM && output->op == GGML_OP_MUL,
            "expected RMS_NORM -> MUL production graph");
    require(ggml_backend_dev_supports_op(device, rms) && ggml_backend_dev_supports_op(device, output),
            "production HIP backend does not support the RMS/MUL graph");
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(context.get(), backend.get()), ggml_backend_buffer_free);
    require(buffer != nullptr && !ggml_backend_buffer_is_host(buffer.get()), "expected small device buffer");
    result.allocation_bytes = ggml_backend_buffer_get_size(buffer.get());
    require(result.allocation_bytes >= 4 * f32_bytes && result.allocation_bytes <= (1U << 20),
            "unexpected GGML tensor allocation size");
    const auto base = reinterpret_cast<std::uintptr_t>(ggml_backend_buffer_get_base(buffer.get()));
    std::array<std::uintptr_t, 4> addresses{};
    const std::array<ggml_tensor*, 4> tensors{x, gamma, rms, output};
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        auto* t = tensors[i];
        addresses[i] = reinterpret_cast<std::uintptr_t>(t->data);
        require(t->type == GGML_TYPE_F32 && t->ne[0] == hidden && t->ne[1] == branches &&
                t->ne[2] == 1 && t->ne[3] == 1 && ggml_is_contiguous(t) && ggml_nbytes(t) == f32_bytes &&
                t->buffer == buffer.get() && !t->view_src && base != 0 && addresses[i] >= base &&
                addresses[i] - base <= result.allocation_bytes &&
                f32_bytes <= result.allocation_bytes - (addresses[i] - base) &&
                f32_bytes <= std::numeric_limits<std::uintptr_t>::max() - addresses[i] &&
                addresses[i] % alignof(float) == 0, "GGML F32 shape, strides or storage span invalid");
        for (std::size_t j = 0; j < i; ++j)
            require(addresses[i] + f32_bytes <= addresses[j] || addresses[j] + f32_bytes <= addresses[i],
                    "GGML writable/readable tensor storage aliases");
    }
    ggml_backend_tensor_set(x, input.data(), 0, f32_bytes);
    ggml_backend_tensor_set(gamma, p.gamma.data(), 0, f32_bytes);
    require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS,
            "production ggml_backend_graph_compute failed");
    ggml_backend_synchronize(backend.get());
    result.output.resize(wide);
    ggml_backend_tensor_get(output, result.output.data(), 0, f32_bytes);
    finite_values(result.output, "production output");
    std::vector<float> unchanged(wide);
    ggml_backend_tensor_get(x, unchanged.data(), 0, f32_bytes);
    require(std::memcmp(unchanged.data(), input.data(), f32_bytes) == 0, "production modified input");
    ggml_backend_tensor_get(gamma, unchanged.data(), 0, f32_bytes);
    require(std::memcmp(unchanged.data(), p.gamma.data(), f32_bytes) == 0, "production modified gamma");
    // Destruction: buffer, context, backend, registry. Public free APIs are void.
    return result;
}

void require_new_output(const char* path) {
    struct stat s{};
    if (::lstat(path, &s) == 0) throw std::runtime_error("output already exists (including symlinks)");
    if (errno != ENOENT) throw std::system_error(errno, std::generic_category(), "lstat new output");
    const auto parent = std::filesystem::path(path).parent_path();
    require(std::filesystem::is_directory(parent.empty() ? "." : parent), "output parent must already exist");
}
void write_output(const char* path, std::span<const float> output) {
    File file(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW);
    const auto bytes = std::as_bytes(output);
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto n = ::write(file.fd, bytes.data() + offset, bytes.size() - offset);
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "write new oracle output");
        }
        require(n > 0, "zero-byte write of oracle output");
        offset += static_cast<std::size_t>(n);
    }
    require(file.status().st_size == static_cast<off_t>(f32_bytes), "new output has wrong size");
    file.close_checked();
}
} // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic());
    std::cout << std::boolalpha << std::setprecision(17);
    try {
        require(argc == 4 || argc == 5,
                "usage: core-rms-oracle MODEL INPUT.f32 NEW_OUTPUT.f32 [ABSOLUTE_libggml-hip.so]");
        const char* backend_path = argc == 5 ? argv[4] : default_backend;
        require(std::filesystem::path(backend_path).is_absolute(), "backend path must be explicit and absolute");
        require_new_output(argv[3]);
        File model_file(argv[1]), input_file(argv[2]), backend_file(backend_path);
        const auto model_stat = model_file.status(), input_stat = input_file.status(), backend_stat = backend_file.status();
        require(!aliases(model_stat, input_stat) && !aliases(model_stat, backend_stat) &&
                !aliases(input_stat, backend_stat), "model/input/backend files must not alias");
        OracleResult result;
        qwen::TensorView gamma_view;
        float epsilon = 0;
        {
            const qwen::Model model(argv[1]); // Header/inventory only, no mmap/model payload load.
            const Parameters parameters(model);
            require(model.file_size() == static_cast<std::uint64_t>(model_stat.st_size), "model file size changed");
            const auto input = read_f32(input_file);
            gamma_view = parameters.view;
            epsilon = parameters.epsilon;
            result = compute(backend_path, parameters, input);
        } // GGML resources and Model cleaned up before output/footer.
        model_file.close_checked();
        input_file.close_checked();
        backend_file.close_checked();
        write_output(argv[3], result.output);
        std::cout << "{\"kind\":\"rms_oracle\",\"protocol\":1,\"diagnostic_only\":true"
                  << ",\"scope\":\"same_captured_input_layer1_attention_group_rms\"";
        build_provenance();
        std::cout << ",\"production_revision\":" << json_string(production_revision)
                  << ",\"library_provenance\":\"parent_exported_production_libraries_not_rebuilt_by_tool\""
                  << ",\"backend_path\":" << json_string(backend_path)
                  << ",\"backend\":" << json_string(result.backend) << ",\"device\":0,\"device_name\":" << json_string(result.device)
                  << ",\"description\":" << json_string(result.description) << ",\"device_id\":" << json_string(result.device_id)
                  << ",\"model\":" << json_string(argv[1]) << ",\"input\":" << json_string(argv[2])
                  << ",\"output\":" << json_string(argv[3])
                  << ",\"input_file_device\":" << static_cast<std::uint64_t>(input_stat.st_dev)
                  << ",\"input_file_inode\":" << static_cast<std::uint64_t>(input_stat.st_ino)
                  << ",\"input_bytes\":" << f32_bytes << ",\"output_bytes\":" << f32_bytes
                  << ",\"model_bytes\":" << model_stat.st_size << ",\"tensor_payload_bytes_read\":" << f32_bytes
                  << ",\"gamma\":" << json_string(gamma_name) << ",\"gamma_file_offset\":" << gamma_view.file_offset
                  << ",\"gamma_byte_size\":" << gamma_view.byte_size << ",\"epsilon\":" << epsilon
                  << ",\"epsilon_f32_bits\":" << std::bit_cast<std::uint32_t>(epsilon)
                  << ",\"shape\":[2560,4],\"graph\":[\"GGML_RMS_NORM\",\"GGML_MUL\"]"
                  << ",\"allocation_bytes\":" << result.allocation_bytes
                  << ",\"all_finite\":true,\"input_immutable\":true,\"gamma_immutable\":true,\"cleanup_complete\":true}\n"
                  << "{\"kind\":\"rms_oracle_complete\",\"protocol\":1,\"diagnostic_only\":true,\"cleanup_complete\":true,\"passed\":true}\n";
        std::cout.flush();
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "{\"kind\":\"rms_oracle_error\",\"protocol\":1,\"diagnostic_only\":true,\"passed\":false,\"error\":"
                  << json_string(error.what()) << "}\n";
        return 1;
    }
}
