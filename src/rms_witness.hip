// R3b SAME-captured-input diagnostic, not a runtime normalization selector or
// whole-logits waiver. Parent source pin e9f1dfe + current Session working tree.
// Reference file comes from tools/rms_oracle.cpp using original production mx
// dcd685463d597d31f5ca759d32c94592a2740fa4 HIP libraries, not our CPU RMS.
// Link qwen-blocks-gpu/qwen-linear-gpu and their qwen-core dependency. No new
// kernels, projections, model payload mapping or timing/speed claims here.
// Compile all linked own modules without fast-math and with -ffp-contract=off.
// stdout: exactly one JSON/device (0,1), then a complete footer AFTER cleanup.
// Only full F32 frozen-gate parity + integrity + zero sticky flags gates pass.
// Q8 byte parity is attempted and reported, not required for that primary pass.
#include "model.hpp"
#include "quant.hpp"
#include "hip/blocks.cuh"
#include "hip/linear.cuh"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
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
#error "RMS witness requires finite checks and no fast-math"
#endif
#ifdef __clang__
#pragma clang fp contract(off)
#endif

namespace {
constexpr int hidden = 2560, branches = 4, wide = hidden * branches;
constexpr std::size_t f32_bytes = wide * sizeof(float), q8_blocks = wide / 32;
constexpr std::size_t q8_bytes = q8_blocks * sizeof(qwen::Q8_1), guard_bytes = 64;
constexpr unsigned char canary = 0xa7;
constexpr float required_epsilon = 1.0e-6f;
// Frozen before GPU execution; reference magnitude is always the bound source.
constexpr double absolute_gate = 2.0e-4, relative_gate = 2.0e-4;
constexpr const char* gamma_name = "blk.1.hc_attn_norm.weight";
constexpr const char* production_revision = "dcd685463d597d31f5ca759d32c94592a2740fa4";
constexpr const char* parent_source_pin = "e9f1dfe";
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little && sizeof(qwen::Q8_1) == 36);
static_assert(q8_blocks == 320 && q8_bytes == 11520);

void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void check(hipError_t status, std::string_view operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
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
    explicit File(const char* path) : fd(::open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), std::string("open ") + path);
    }
    ~File() {
        if (fd >= 0 && ::close(fd) != 0) std::cerr << "rms witness: file cleanup failed\n";
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
std::vector<float> read_f32(File& file, std::string_view label) {
    require(file.status().st_size == static_cast<off_t>(f32_bytes), "F32 files must be exactly 40960 bytes");
    std::vector<float> result(wide);
    auto bytes = std::as_writable_bytes(std::span(result));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto n = ::pread(file.fd, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "pread F32");
        }
        require(n > 0, "F32 file truncated during read");
        offset += static_cast<std::size_t>(n);
    }
    require(file.status().st_size == static_cast<off_t>(f32_bytes), "F32 file changed size during read");
    finite_values(result, label);
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

struct Metrics {
    std::size_t count = 0, exceed_count = 0, bit_mismatches = 0;
    double max_abs = 0, rms = 0, max_bound_ratio = 0;
};
Metrics compare(std::span<const float> actual, std::span<const float> reference) {
    require(actual.size() == reference.size() && !actual.empty(), "RMS comparison shape mismatch");
    finite_values(actual, "own RMS output");
    Metrics result;
    result.count = actual.size();
    double squared = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(double(actual[i]) - double(reference[i]));
        const double bound = absolute_gate + relative_gate * std::abs(double(reference[i]));
        result.max_abs = std::max(result.max_abs, error);
        result.max_bound_ratio = std::max(result.max_bound_ratio, error / bound);
        result.exceed_count += error > bound;
        result.bit_mismatches += std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(reference[i]);
        squared += error * error;
    }
    result.rms = std::sqrt(squared / double(result.count));
    return result;
}
void print_metrics(const Metrics& m) {
    std::cout << "{\"count\":" << m.count << ",\"all_finite\":true,\"max_abs\":" << m.max_abs
              << ",\"rms\":" << m.rms << ",\"max_bound_ratio\":" << m.max_bound_ratio
              << ",\"exceed_count\":" << m.exceed_count << ",\"float_bit_mismatches\":" << m.bit_mismatches
              << ",\"byte_exact\":" << (m.bit_mismatches == 0)
              << ",\"within_frozen_gate\":" << (m.exceed_count == 0) << '}';
}

// Five small allocations with two independent canaries each; every payload is
// separately allocated, 4-byte aligned, capacity checked, and non-overlapping.
struct Allocation {
    void* raw = nullptr;
    std::size_t bytes = 0;
    void* data() const { return static_cast<std::byte*>(raw) + guard_bytes; }
};
struct Resources {
    int device;
    bool& cleanup_complete;
    bool closed = false;
    hipStream_t stream = nullptr;
    std::array<hipEvent_t, 2> events{};
    std::array<Allocation, 5> buffers{};
    Resources(int d, bool& complete) : device(d), cleanup_complete(complete) {}
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() { if (!closed) release(); }
    void initialize() {
        check(hipSetDevice(device), "hipSetDevice");
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
        for (auto& event : events)
            check(hipEventCreateWithFlags(&event, hipEventDisableTiming), "hipEventCreateWithFlags");
    }
    void allocate(std::size_t index, std::size_t bytes) {
        require(index < buffers.size() && !buffers[index].raw && bytes > 0 && bytes % 4 == 0,
                "invalid witness allocation");
        auto& b = buffers[index];
        b.bytes = bytes;
        check(hipMalloc(&b.raw, bytes + 2 * guard_bytes), "hipMalloc guarded buffer");
        const auto begin = reinterpret_cast<std::uintptr_t>(b.raw);
        require(begin != 0 && begin % 4 == 0 && bytes + 2 * guard_bytes <=
                std::numeric_limits<std::uintptr_t>::max() - begin, "invalid guarded allocation span");
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            if (i == index || !buffers[i].raw) continue;
            const auto other = reinterpret_cast<std::uintptr_t>(buffers[i].raw);
            require(begin + bytes + 2 * guard_bytes <= other ||
                    other + buffers[i].bytes + 2 * guard_bytes <= begin, "device allocations alias");
        }
        check(hipMemsetAsync(b.raw, canary, bytes + 2 * guard_bytes, stream), "initialize canaries");
        check(hipMemsetAsync(b.data(), 0xff, bytes, stream), "poison provisional payload");
    }
    void complete(std::size_t event) {
        check(hipEventRecord(events[event], stream), "hipEventRecord completion");
        check(hipEventSynchronize(events[event]), "hipEventSynchronize completion");
        check(hipStreamSynchronize(stream), "hipStreamSynchronize completion");
    }
    std::vector<std::byte> snapshot(std::size_t index) {
        const auto& b = buffers[index];
        std::vector<std::byte> result(b.bytes + 2 * guard_bytes);
        check(hipMemcpyAsync(result.data(), b.raw, result.size(), hipMemcpyDeviceToHost, stream), "D2H guarded snapshot");
        check(hipStreamSynchronize(stream), "complete guarded snapshot");
        for (std::size_t i = 0; i < guard_bytes; ++i)
            require(result[i] == std::byte(canary) && result[guard_bytes + b.bytes + i] == std::byte(canary),
                    "device buffer canary changed");
        return result;
    }
    void release() noexcept {
        bool ok = true;
        auto cleanup = [&](hipError_t status, const char* operation) {
            if (status != hipSuccess) {
                ok = false;
                std::cerr << "rms witness cleanup " << operation << ": " << hipGetErrorString(status) << '\n';
            }
        };
        cleanup(hipSetDevice(device), "set device");
        if (stream) cleanup(hipStreamSynchronize(stream), "synchronize stream");
        for (auto& b : buffers) if (b.raw) {
            cleanup(hipFree(b.raw), "free buffer");
            b.raw = nullptr;
        }
        for (auto& event : events) if (event) {
            cleanup(hipEventDestroy(event), "destroy event");
            event = nullptr;
        }
        if (stream) cleanup(hipStreamDestroy(stream), "destroy stream");
        stream = nullptr;
        closed = true;
        cleanup_complete = ok;
    }
    void close_checked() {
        release();
        require(cleanup_complete, "HIP cleanup failed");
    }
};

struct DeviceResult {
    Metrics rms;
    std::array<Metrics, branches> groups;
    std::vector<qwen::Q8_1> q8;
    int rms_error = -1, quant_error = -1;
    bool cleanup_complete = false, integrity_complete = false, comparison_complete = false;
    std::string error;
};

void run_device(int device, const Parameters& p, std::span<const float> input,
                std::span<const float> reference, DeviceResult& result) {
    Resources resources(device, result.cleanup_complete);
    resources.initialize();
    resources.allocate(0, f32_bytes);
    resources.allocate(1, f32_bytes);
    resources.allocate(2, f32_bytes);
    resources.allocate(3, q8_bytes);
    resources.allocate(4, sizeof(int));
    auto* x = static_cast<float*>(resources.buffers[0].data());
    auto* gamma = static_cast<float*>(resources.buffers[1].data());
    auto* output = static_cast<float*>(resources.buffers[2].data());
    auto* quantized = static_cast<qwen::Q8_1*>(resources.buffers[3].data());
    auto* error = static_cast<int*>(resources.buffers[4].data());
    check(hipMemcpyAsync(x, input.data(), f32_bytes, hipMemcpyHostToDevice, resources.stream), "upload captured input");
    check(hipMemcpyAsync(gamma, p.gamma.data(), f32_bytes, hipMemcpyHostToDevice, resources.stream), "upload actual gamma");
    check(hipMemsetAsync(error, 0, sizeof(int), resources.stream), "reset RMS sticky flag");
    check(qwen::launch_group_rms_norm(x, gamma, 1, wide, hidden, p.epsilon, false,
                                     output, error, resources.stream), "launch_group_rms_norm");
    std::vector<float> actual(wide);
    check(hipMemcpyAsync(actual.data(), output, f32_bytes, hipMemcpyDeviceToHost, resources.stream), "read own RMS output");
    check(hipMemcpyAsync(&result.rms_error, error, sizeof(int), hipMemcpyDeviceToHost, resources.stream), "read RMS sticky flag");
    resources.complete(0);
    require(result.rms_error == 0, "RMS sticky flag must be zero before consuming output");
    result.rms = compare(actual, reference);
    for (std::size_t group = 0; group < branches; ++group)
        result.groups[group] = compare(std::span(actual).subspan(group * hidden, hidden), reference.subspan(group * hidden, hidden));
    result.comparison_complete = true;
    // Even a bounded-FP gate failure still attempts quantized-byte diagnostics.
    result.q8.resize(q8_blocks);
    check(hipMemsetAsync(error, 0, sizeof(int), resources.stream), "reset Q8 sticky flag after RMS check");
    check(qwen::launch_quantize_q8_1(output, quantized, wide, 1, error, resources.stream), "launch_quantize_q8_1");
    check(hipMemcpyAsync(result.q8.data(), quantized, q8_bytes, hipMemcpyDeviceToHost, resources.stream), "read own Q8 blocks");
    check(hipMemcpyAsync(&result.quant_error, error, sizeof(int), hipMemcpyDeviceToHost, resources.stream), "read Q8 sticky flag");
    resources.complete(1);
    require(result.quant_error == 0, "Q8 sticky flag must be zero");
    const auto unchanged_input = resources.snapshot(0), unchanged_gamma = resources.snapshot(1);
    require(std::memcmp(unchanged_input.data() + guard_bytes, input.data(), f32_bytes) == 0, "captured input mutated");
    require(std::memcmp(unchanged_gamma.data() + guard_bytes, p.gamma.data(), f32_bytes) == 0, "actual gamma mutated");
    const auto unchanged_output = resources.snapshot(2), checked_q8 = resources.snapshot(3), checked_error = resources.snapshot(4);
    require(std::memcmp(unchanged_output.data() + guard_bytes, actual.data(), f32_bytes) == 0, "Q8 producer mutated RMS input");
    require(std::memcmp(checked_q8.data() + guard_bytes, result.q8.data(), q8_bytes) == 0, "Q8 readback changed");
    require(std::memcmp(checked_error.data() + guard_bytes, &result.quant_error, sizeof(int)) == 0, "sticky flag changed");
    result.integrity_complete = true;
    resources.close_checked(); // Required before a successful device record.
}

void print_q8(std::span<const qwen::Q8_1> actual, std::span<const qwen::Q8_1> reference) {
    require(actual.size() == q8_blocks && reference.size() == q8_blocks, "Q8 diagnostic shape mismatch");
    std::size_t differing_bytes = 0, differing_blocks = 0, scale_blocks = 0, raw_sum_blocks = 0, code_blocks = 0, code_count = 0;
    const auto actual_bytes = std::as_bytes(actual), reference_bytes = std::as_bytes(reference);
    for (std::size_t i = 0; i < q8_bytes; ++i) differing_bytes += actual_bytes[i] != reference_bytes[i];
    for (std::size_t b = 0; b < q8_blocks; ++b) {
        differing_blocks += std::memcmp(&actual[b], &reference[b], sizeof(qwen::Q8_1)) != 0;
        scale_blocks += actual[b].d != reference[b].d;
        raw_sum_blocks += actual[b].s != reference[b].s;
        std::size_t codes = 0;
        for (std::size_t j = 0; j < 32; ++j) codes += actual[b].qs[j] != reference[b].qs[j];
        code_blocks += codes != 0;
        code_count += codes;
    }
    std::cout << "{\"attempted\":true,\"completed\":true,\"required_for_primary_pass\":false,\"reference\":\"CPU_source_quantize_q8_oracle_F32\""
              << ",\"actual\":\"GPU_launch_quantize_q8_1_own_RMS_F32\",\"block_count\":" << q8_blocks
              << ",\"block_bytes\":36,\"byte_exact\":" << (differing_bytes == 0)
              << ",\"differing_bytes\":" << differing_bytes << ",\"differing_blocks\":" << differing_blocks
              << ",\"scale_differing_blocks\":" << scale_blocks << ",\"raw_sum_differing_blocks\":" << raw_sum_blocks
              << ",\"code_differing_blocks\":" << code_blocks << ",\"code_mismatches\":" << code_count << ",\"differences\":[";
    bool first = true;
    for (std::size_t b = 0; b < q8_blocks; ++b) {
        if (std::memcmp(&actual[b], &reference[b], sizeof(qwen::Q8_1)) == 0) continue;
        std::size_t codes = 0;
        int max_delta = 0;
        for (std::size_t j = 0; j < 32; ++j) {
            codes += actual[b].qs[j] != reference[b].qs[j];
            max_delta = std::max(max_delta, std::abs(int(actual[b].qs[j]) - int(reference[b].qs[j])));
        }
        std::cout << (first ? "" : ",") << "{\"block\":" << b
                  << ",\"scale_actual_bits\":" << actual[b].d << ",\"scale_reference_bits\":" << reference[b].d
                  << ",\"raw_sum_actual_bits\":" << actual[b].s << ",\"raw_sum_reference_bits\":" << reference[b].s
                  << ",\"code_mismatches\":" << codes << ",\"max_code_delta\":" << max_delta << '}';
        first = false;
    }
    std::cout << "]}";
}

bool passed(const DeviceResult& r) {
    return r.error.empty() && r.comparison_complete && r.rms.exceed_count == 0 &&
           r.rms.count == wide && r.rms_error == 0 && r.quant_error == 0 && r.integrity_complete && r.cleanup_complete;
}

bool execute(int argc, char** argv) {
    require(argc == 4, "usage: core-rms-witness MODEL INPUT.f32 ORACLE_OUTPUT.f32");
    File model_file(argv[1]), input_file(argv[2]), oracle_file(argv[3]);
    const auto model_stat = model_file.status(), input_stat = input_file.status(), oracle_stat = oracle_file.status();
    require(!aliases(model_stat, input_stat) && !aliases(model_stat, oracle_stat) && !aliases(input_stat, oracle_stat),
            "model/input/oracle files must not alias (including hard links)");
    const auto parameters = [&] {
        const qwen::Model model(argv[1]); // Header/inventory and ONE 40-KiB gamma only.
        require(model.file_size() == static_cast<std::uint64_t>(model_stat.st_size), "model file size changed");
        return Parameters(model);
    }(); // Model descriptor already closed before any HIP work.
    const auto input = read_f32(input_file, "captured input");
    const auto reference = read_f32(oracle_file, "production oracle output");
    model_file.close_checked();
    input_file.close_checked();
    oracle_file.close_checked();
    std::vector<qwen::Q8_1> reference_q8(q8_blocks);
    qwen::quantize_q8(reference, reference_q8); // Actual CPU source implementation, not reimplemented math.
    for (const auto& block : reference_q8)
        require(std::isfinite(qwen::half_to_float(block.d)) && std::isfinite(qwen::half_to_float(block.s)),
                "CPU reference Q8 contains nonfinite half fields");

    int count = 0;
    check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    require(count == 2, "RMS witness requires exactly two gfx906 GPUs");
    std::array<hipDeviceProp_t, 2> properties{};
    for (int device = 0; device < count; ++device) {
        check(hipGetDeviceProperties(&properties[device], device), "hipGetDeviceProperties");
        const std::string_view arch(properties[device].gcnArchName);
        require((arch == "gfx906" || arch.starts_with("gfx906:")) && properties[device].warpSize == 64 &&
                properties[device].maxThreadsPerBlock >= 1024, "expected gfx906, wave64 and support for RMS1024 topology");
    }
    bool all_passed = true;
    for (int device = 0; device < count; ++device) {
        DeviceResult result;
        try {
            run_device(device, parameters, input, reference, result);
        } catch (const std::exception& error) {
            // run_device's RAII cleanup has completed before this record, even
            // on launch/copy/check failure. An actual cleanup failure stays false.
            result.error = error.what();
        }
        const bool device_passed = passed(result);
        all_passed = all_passed && device_passed;
        std::cout << "{\"kind\":\"rms_witness\",\"protocol\":1,\"diagnostic_only\":true"
                  << ",\"scope\":\"same_captured_input_layer1_attention_group_rms\",\"device\":" << device;
        build_provenance();
        std::cout << ",\"oracle_production_revision\":" << json_string(production_revision)
                  << ",\"oracle_provenance\":\"external_core_rms_oracle_output_required\""
                  << ",\"device_name\":" << json_string(properties[device].name)
                  << ",\"gcn_arch\":" << json_string(properties[device].gcnArchName)
                  << ",\"wave_size\":" << properties[device].warpSize
                  << ",\"vram_bytes\":" << properties[device].totalGlobalMem
                  << ",\"model\":" << json_string(argv[1]) << ",\"input\":" << json_string(argv[2])
                  << ",\"oracle_output\":" << json_string(argv[3])
                  << ",\"input_file_device\":" << static_cast<std::uint64_t>(input_stat.st_dev)
                  << ",\"input_file_inode\":" << static_cast<std::uint64_t>(input_stat.st_ino)
                  << ",\"oracle_file_device\":" << static_cast<std::uint64_t>(oracle_stat.st_dev)
                  << ",\"oracle_file_inode\":" << static_cast<std::uint64_t>(oracle_stat.st_ino)
                  << ",\"input_bytes\":" << f32_bytes << ",\"oracle_bytes\":" << f32_bytes
                  << ",\"model_bytes\":" << model_stat.st_size << ",\"tensor_payload_bytes_read\":" << f32_bytes
                  << ",\"gamma\":" << json_string(gamma_name) << ",\"gamma_file_offset\":" << parameters.view.file_offset
                  << ",\"gamma_byte_size\":" << parameters.view.byte_size << ",\"epsilon\":" << parameters.epsilon
                  << ",\"epsilon_f32_bits\":" << std::bit_cast<std::uint32_t>(parameters.epsilon)
                  << ",\"tokens\":1,\"width\":10240,\"group_size\":2560,\"shared_gamma\":false"
                  << ",\"absolute_gate\":" << absolute_gate << ",\"relative_gate\":" << relative_gate
                  << ",\"rms\":";
        if (result.comparison_complete) print_metrics(result.rms);
        else std::cout << "null";
        std::cout << ",\"groups\":[";
        if (result.comparison_complete)
            for (std::size_t group = 0; group < branches; ++group) {
                if (group) std::cout << ',';
                print_metrics(result.groups[group]);
            }
        std::cout << "],\"q8\":";
        if (result.quant_error == 0 && result.q8.size() == q8_blocks) print_q8(result.q8, reference_q8);
        else std::cout << "{\"attempted\":" << (!result.q8.empty()) << ",\"completed\":false,\"required_for_primary_pass\":false}";
        std::cout << ",\"rms_error_flag\":" << result.rms_error << ",\"quant_error_flag\":" << result.quant_error
                  << ",\"sticky_flags_reset_and_checked\":" << (result.rms_error == 0 && result.quant_error == 0)
                  << ",\"input_immutable\":" << result.integrity_complete << ",\"gamma_immutable\":" << result.integrity_complete
                  << ",\"rms_output_immutable_through_q8\":" << result.integrity_complete
                  << ",\"canaries_intact\":" << result.integrity_complete
                  << ",\"canary_checks\":" << (result.integrity_complete ? 10 : 0)
                  << ",\"allocation_aliases_checked\":" << result.integrity_complete
                  << ",\"cleanup_complete\":" << result.cleanup_complete
                  << ",\"error\":" << (result.error.empty() ? "null" : json_string(result.error))
                  << ",\"passed\":" << device_passed << "}\n";
    }
    return all_passed;
} // All file/model/host/GPU owners die before the footer emitted by main.
} // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic());
    std::cout << std::boolalpha << std::setprecision(17);
    try {
        const bool success = execute(argc, argv);
        std::cout << "{\"kind\":\"rms_witness_complete\",\"protocol\":1,\"diagnostic_only\":true,\"device_records\":2"
                  << ",\"primary_gate\":\"all_10240_F32_frozen_bound_plus_integrity_and_zero_error_flags\""
                  << ",\"q8_byte_exact_required\":false,\"passed\":" << success << "}\n";
        std::cout.flush();
        return success && std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cout << "{\"kind\":\"rms_witness_error\",\"protocol\":1,\"diagnostic_only\":true,\"passed\":false,\"error\":"
                  << json_string(error.what()) << "}\n";
        return 1;
    }
}
