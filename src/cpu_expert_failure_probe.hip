// Diagnostic replay only; no Session/CPU/GPU production arithmetic changes.
// Parent target: core-cpu-expert-failure-probe MODEL LAYER EXPERT Q8FILE.
// C++20 strict FP (-ffp-contract=off, no fast-math), gfx906; link qwen-core
// (model/quant/cpu_expert), qwen-linear-gpu, Session session_ops and Threads.
// Define actual CORE_REVISION/CORE_DIRTY. KEEP current canonical GPU launches;
// ADAPT guarded ownership/capture style from cpu_expert_shadow.hip. The optional
// CPU down replay below copies cpu_expert.cpp's fragment/FMA/XOR order, validates
// itself bitwise on the normal CPU middle Q8, and is NOT a production backend.
// No model copies/read_tensor/requantization of the caller's original Q8 row.
#include "cpu_expert.hpp"
#include "hip/linear.cuh"
#include "hip/session_ops.cuh"
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
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
#include <vector>

#if !defined(CORE_REVISION) || !defined(CORE_DIRTY)
#error "failure probe requires actual compiled CORE_REVISION and CORE_DIRTY"
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "failure probe requires strict floating point"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

namespace {
using namespace qwen;
constexpr int H = 2560, M = 640;
constexpr std::size_t guard = 64, absent = std::numeric_limits<std::size_t>::max();
constexpr std::byte poison{0xa7};
constexpr std::string_view revision = CORE_REVISION;
constexpr bool valid_revision() {
    if (revision.size() != 40) return false;
    for (char c : revision)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    return true;
}
static_assert(valid_revision() && (CORE_DIRTY == 0 || CORE_DIRTY == 1));
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 && std::endian::native == std::endian::little);
static_assert(sizeof(Q8_1) == 36 && offsetof(Q8_1, qs) == 4);
void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void check(hipError_t status, const char* operation) {
    if (status != hipSuccess) throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
std::string quote(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c < 32 || c >= 127) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += char(c);
    }
    return result + '"';
}
void provenance() {
    std::cout << ",\"revision\":" << quote(revision) << ",\"dirty\":" << (CORE_DIRTY ? "true" : "false");
}
int integer(std::string_view text, int maximum) {
    int value = -1;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    require(!text.empty() && result.ec == std::errc{} && result.ptr == text.data() + text.size() && value >= 0 && value <= maximum,
        "invalid integer layer/expert range");
    return value;
}
void finite(std::span<const float> values) {
    for (float v : values) require(std::isfinite(v), "nonfinite FP32 capture");
}
void finite_q8(std::span<const Q8_1> values) {
    for (const auto& b : values)
        require((b.d & 0x7c00U) != 0x7c00U && (b.s & 0x7c00U) != 0x7c00U && !(b.d & 0x8000U), "nonfinite/negative Q8 header");
    // Valid stored half(d)=0 never implies zero original FP32 d/codes/raw sum.
}
std::array<Q8_1, 80> read_input(const char* path) {
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "open original Q8");
    struct Close { int fd; ~Close() { if (fd >= 0) ::close(fd); } } close{fd};
    const auto size = [&] {
        struct stat s{};
        if (::fstat(fd, &s) != 0) throw std::system_error(errno, std::generic_category(), "fstat original Q8");
        require(S_ISREG(s.st_mode) && s.st_size == 2880, "Q8 input must be regular and exactly 2880 bytes");
    };
    size();
    alignas(64) std::array<Q8_1, 80> result{};
    auto bytes = std::as_writable_bytes(std::span(result));
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto n = ::pread(fd, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) throw std::system_error(errno, std::generic_category(), "read original Q8");
        require(n > 0, "truncated original Q8 input"); offset += static_cast<std::size_t>(n);
    }
    size(); finite_q8(result);
    close.fd = -1;
    if (::close(fd) != 0) throw std::system_error(errno, std::generic_category(), "close original Q8");
    return result;
}
struct Matrix {
    TensorView tensor;
    int input, output;
    std::vector<std::byte> payload;
    std::vector<std::byte> original_payload;
    Matrix(const Model& model, int layer, int expert, const char* part, int k, int m, TensorType type)
        : tensor(model.tensor("blk." + std::to_string(layer) + ".ffn_" + part + "_exps.weight")), input(k), output(m) {
        const auto layout = type_layout(type), actual = type_layout(tensor.type);
        const auto row = static_cast<std::uint64_t>(k / 32) * layout.block_bytes;
        const auto slice = row * static_cast<std::uint64_t>(m);
        require(tensor.type == type && actual.block_elements == 32 && tensor.rank == 3 &&
            tensor.dimensions[0] == static_cast<std::uint64_t>(k) && tensor.dimensions[1] == static_cast<std::uint64_t>(m) &&
            tensor.dimensions[2] == 512 && tensor.strides[0] == layout.block_bytes && tensor.strides[1] == row &&
            tensor.strides[2] == slice && tensor.byte_size == slice * 512 && tensor.file_offset >= model.data_offset() &&
            tensor.file_offset <= model.file_size() && tensor.byte_size <= model.file_size() - tensor.file_offset,
            "canonical expert geometry/type/offset mismatch");
        payload.resize(static_cast<std::size_t>(slice));
        model.read_expert(tensor.name, static_cast<std::uint64_t>(expert), payload);
        original_payload = payload; // Bounded one-expert diagnostic readonly snapshot, not a model copy.
    }
    QMatrix view() const { return {tensor.type, input, output, payload}; }
    void unchanged() const { require(payload == original_payload, "canonical host weight payload changed"); }
};
struct Expert {
    Matrix gate, up, down;
    CpuExpert cpu;
    Expert(const Model& model, int layer, int expert)
        : gate(model, layer, expert, "gate", H, M, TensorType::Q4_0),
          up(model, layer, expert, "up", H, M, TensorType::Q4_0),
          down(model, layer, expert, "down", M, H, layer < 6 ? TensorType::Q4_1 : TensorType::Q4_0),
          cpu(gate.view(), up.view(), down.view()) {}
    void unchanged() const { gate.unchanged(); up.unchanged(); down.unchanged(); }
};
struct Context {
    const char* stage = "arguments";
    int device = -1, layer = -1, expert = -1;
    int completed_devices = 0;
    bool parity = true;
    hipError_t cleanup = hipSuccess;
    const char* cleanup_operation = "none";
    void record(hipError_t result, const char* operation) noexcept {
        if (result != hipSuccess && cleanup == hipSuccess) { cleanup = result; cleanup_operation = operation; }
    }
};
struct Stream {
    Context& context;
    hipStream_t value = nullptr;
    explicit Stream(Context& c) : context(c) { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking), "create stream"); }
    ~Stream() {
        if (value) { context.record(hipStreamSynchronize(value), "cleanup synchronize"); context.record(hipStreamDestroy(value), "cleanup stream"); }
    }
    void wait() const { check(hipStreamSynchronize(value), "stream completion"); }
};
struct Buffer {
    Stream& stream;
    void* base = nullptr;
    std::size_t bytes;
    Buffer(Stream& s, std::size_t capacity) : stream(s), bytes(capacity) { check(hipMalloc(&base, bytes + 2 * guard), "allocate guarded buffer"); }
    ~Buffer() {
        if (base) { stream.context.record(hipStreamSynchronize(stream.value), "cleanup buffer synchronize"); stream.context.record(hipFree(base), "cleanup buffer free"); }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() const { return reinterpret_cast<T*>(static_cast<std::byte*>(base) + guard); }
    void clear() const { check(hipMemsetAsync(base, 0xa7, bytes + 2 * guard, stream.value), "buffer canaries"); }
    template<class T> void upload(std::span<const T> values) const {
        require(values.size_bytes() == bytes, "upload exact bounded capacity"); clear();
        check(hipMemcpyAsync(as<T>(), values.data(), bytes, hipMemcpyHostToDevice, stream.value), "upload payload"); stream.wait();
    }
    template<class T> void download(std::span<T> values) const {
        require(values.size_bytes() == bytes, "download exact bounded capacity");
        std::vector<std::byte> copy(bytes + 2 * guard);
        check(hipMemcpyAsync(copy.data(), base, copy.size(), hipMemcpyDeviceToHost, stream.value), "download guarded payload"); stream.wait();
        for (std::size_t i = 0; i < guard; ++i) require(copy[i] == poison && copy[guard + bytes + i] == poison, "GPU canary changed");
        std::memcpy(values.data(), copy.data() + guard, bytes);
    }
    template<class T> void unchanged(std::span<const T> values) const {
        std::vector<T> copy(values.size()); download<T>(copy);
        require(std::memcmp(copy.data(), values.data(), bytes) == 0, "GPU read-only payload changed");
    }
};
struct Capture {
    std::array<float, M> gate{}, up{}, middle{};
    alignas(64) std::array<Q8_1, 20> q8{};
    std::array<float, H> down{};
};
struct Gpu {
    Stream& stream;
    Buffer gate_weights, up_weights, down_weights, input, gate, up, middle, q8, down, error;
    Gpu(Stream& s, const Expert& e, std::span<const Q8_1> original)
        : stream(s), gate_weights(s, e.gate.payload.size()), up_weights(s, e.up.payload.size()), down_weights(s, e.down.payload.size()),
          input(s, 2880), gate(s, M * 4), up(s, M * 4), middle(s, M * 4), q8(s, 20 * sizeof(Q8_1)), down(s, H * 4), error(s, sizeof(int)) {
        gate_weights.upload<std::byte>(e.gate.payload); up_weights.upload<std::byte>(e.up.payload); down_weights.upload<std::byte>(e.down.payload);
        input.upload<Q8_1>(original);
    }
    QuantizedDeviceMatrix matrix(const Matrix& m, const Buffer& b) const { return {m.tensor.type, b.as<std::byte>(), m.input, m.output}; }
    void flag_zero() const { const int zero = 0; error.upload<int>({&zero, 1}); }
    void checked() const { int flag = -1; error.download<int>({&flag, 1}); require(flag == 0, "GPU sticky error"); }
    void tail(const Expert& e, Capture& result) {
        middle.clear(); q8.clear(); down.clear();
        check(launch_silu_up(gate.as<float>(), up.as<float>(), M, middle.as<float>(), error.as<int>(), stream.value), "canonical SiLU/up");
        check(launch_quantize_q8_1(middle.as<float>(), q8.as<Q8_1>(), M, 1, error.as<int>(), stream.value), "canonical middle Q8");
        check(launch_quantized_linear(matrix(e.down, down_weights), q8.as<Q8_1>(), 1, down.as<float>(), stream.value), "canonical down");
        checked();
        middle.download<float>(result.middle); q8.download<Q8_1>(result.q8); down.download<float>(result.down);
        finite(result.middle); finite_q8(result.q8); finite(result.down);
    }
    Capture run(const Expert& e) {
        Capture result;
        flag_zero(); gate.clear(); up.clear();
        check(launch_quantized_linear(matrix(e.gate, gate_weights), input.as<Q8_1>(), 1, gate.as<float>(), stream.value), "canonical gate");
        check(launch_quantized_linear(matrix(e.up, up_weights), input.as<Q8_1>(), 1, up.as<float>(), stream.value), "canonical up");
        gate.download<float>(result.gate); up.download<float>(result.up); finite(result.gate); finite(result.up);
        tail(e, result); return result;
    }
    Capture cpu_gate_up(const Expert& e, const CpuExpertScratch& cpu) {
        Capture result;
        std::copy_n(cpu.gate.begin(), M, result.gate.begin()); std::copy_n(cpu.up.begin(), M, result.up.begin());
        gate.upload<float>(result.gate); up.upload<float>(result.up); flag_zero(); tail(e, result); return result;
    }
    std::array<Q8_1, 20> quantize_same(std::span<const float> floats) {
        middle.upload<float>(floats); q8.clear(); flag_zero();
        check(launch_quantize_q8_1(middle.as<float>(), q8.as<Q8_1>(), M, 1, error.as<int>(), stream.value), "same-float canonical GPU Q8");
        checked(); std::array<Q8_1, 20> result{}; q8.download<Q8_1>(result); finite_q8(result); return result;
    }
    std::array<float, H> common_cpu_q8(const Expert& e, std::span<const Q8_1> bytes) {
        q8.upload<Q8_1>(bytes); down.clear(); flag_zero();
        check(launch_quantized_linear(matrix(e.down, down_weights), q8.as<Q8_1>(), 1, down.as<float>(), stream.value), "common CPU Q8 GPU down");
        checked(); std::array<float, H> result{}; down.download<float>(result); finite(result); return result;
    }
    void readonly(const Expert& e, std::span<const Q8_1> original) const {
        input.unchanged<Q8_1>(original); gate_weights.unchanged<std::byte>(e.gate.payload);
        up_weights.unchanged<std::byte>(e.up.payload); down_weights.unchanged<std::byte>(e.down.payload);
    }
};

// Diagnostic scalar integer fragments + EXACT production CPU floating order.
// No generic GPU oracle. Validate normal CPU Q8 replay bitwise before trusting
// the optional GPU-middle-Q8 counterfactual; unchanged run_cpu_expert is primary.
std::array<float, H> cpu_down_replay(QMatrix m, std::span<const Q8_1> q) {
    require(m.input == M && m.output == H && q.size() == 20, "counterfactual down shape");
    const std::size_t stride = m.type == TensorType::Q4_0 ? 18 : 20, offset = stride - 16;
    const auto half = [](const std::byte* p) { std::uint16_t h; std::memcpy(&h, p, 2); return half_to_float(h); };
    std::array<float, H> result{};
    for (int row = 0; row < H; ++row) {
        std::array<float, 128> partial{};
        for (std::size_t k = 0; k < 20; ++k) {
            const auto* b = m.weights.data() + (static_cast<std::size_t>(row) * 20 + k) * stride;
            const float d4 = half(b), d8 = half_to_float(q[k].d), s8 = half_to_float(q[k].s);
            float dd = 0, ms = 0;
            if (m.type == TensorType::Q4_1) {
                dd = half_to_float(float_to_half(d4 * d8)); ms = half_to_float(float_to_half(half(b + 2) * s8));
            }
            int dot[2]{};
            // CPU AVX2 combines low[j]+high[j] before half-fragment reduction:
            // f0 indices0..7 and16..23, f1 indices8..15 and24..31.
            for (std::size_t j = 0; j < 16; ++j) {
                const auto code = std::to_integer<unsigned>(b[offset + j]);
                dot[j / 8] += static_cast<int>(code & 15U) * q[k].qs[j] + static_cast<int>(code >> 4) * q[k].qs[j + 16];
            }
            for (std::size_t f = 0; f < 2; ++f) {
                auto& acc = partial[2 * (k % 64) + f];
                if (m.type == TensorType::Q4_0) acc = std::fma(d4, std::fma(float(dot[f]), d8, -4.0f * s8), acc);
                else acc = acc + std::fma(float(dot[f]), dd, ms * 0.5f);
            }
        }
        for (std::size_t lane = 0; lane < 64; ++lane) partial[lane] = partial[lane] + partial[lane + 64];
        for (std::size_t mask = 1; mask < 64; mask *= 2)
            for (std::size_t lane = 0; lane < 64; ++lane) if (!(lane & mask)) {
                const float sum = partial[lane] + partial[lane ^ mask]; partial[lane] = sum; partial[lane ^ mask] = sum;
            }
        result[static_cast<std::size_t>(row)] = partial[0];
    }
    finite(result); return result;
}
void float_bits(const char* label, float value) {
    std::cout << quote(label) << ":{\"value\":" << value << ",\"bits\":" << std::bit_cast<std::uint32_t>(value) << '}';
}
struct Metric {
    std::size_t elements = 0, bits = 0, violations = 0, first_bit = absent, first_gate = absent;
    double max_abs = 0, max_ratio = 0;
    std::span<const float> cpu, gpu;
    Metric(std::span<const float> a, std::span<const float> r, double absolute, double relative) : cpu(a), gpu(r) {
        require(a.size() == r.size() && !a.empty(), "metric shape"); finite(a); finite(r); elements = a.size();
        for (std::size_t i = 0; i < elements; ++i) {
            if (std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(r[i]) && bits++ == 0) first_bit = i;
            const auto delta = std::abs(double(a[i]) - r[i]);
            const auto ratio = delta / (absolute + relative * std::abs(double(r[i])));
            max_abs = std::max(max_abs, delta); max_ratio = std::max(max_ratio, ratio);
            if (ratio > 1 && violations++ == 0) first_gate = i;
        }
    }
    void coordinate(std::size_t i) const {
        if (i == absent) { std::cout << "null"; return; }
        std::cout << "{\"row\":" << i << ','; float_bits("actual", cpu[i]); std::cout << ','; float_bits("reference", gpu[i]); std::cout << '}';
    }
    void emit(const char* name) const {
        std::cout << ',' << quote(name) << ":{\"elements\":" << elements << ",\"nonfinite\":0,\"bit_mismatches\":" << bits
            << ",\"violations\":" << violations << ",\"max_abs\":" << max_abs << ",\"max_bound_ratio\":" << max_ratio << ",\"first_bit\":";
        coordinate(first_bit); std::cout << ",\"first_gate\":"; coordinate(first_gate); std::cout << '}';
    }
};
struct QuantDifference {
    std::size_t bytes = 0, scales = 0, sums = 0, codes = 0, first_header = absent, first_code = absent, first_byte = absent;
    QuantDifference(std::span<const Q8_1> a, std::span<const Q8_1> b) {
        require(a.size() == b.size(), "Q8 difference shape"); finite_q8(a); finite_q8(b);
        const auto x = std::as_bytes(a), y = std::as_bytes(b);
        for (std::size_t i = 0; i < x.size(); ++i) if (x[i] != y[i] && bytes++ == 0) first_byte = i;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if ((a[i].d != b[i].d || a[i].s != b[i].s) && first_header == absent) first_header = i;
            scales += a[i].d != b[i].d; sums += a[i].s != b[i].s;
            for (std::size_t j = 0; j < 32; ++j) if (a[i].qs[j] != b[i].qs[j] && codes++ == 0) first_code = i * 32 + j;
        }
    }
    void emit(const char* name) const {
        const auto nullable = [](std::size_t i) { if (i == absent) std::cout << "null"; else std::cout << i; };
        std::cout << ',' << quote(name) << ":{\"bytes\":720,\"byte_mismatches\":" << bytes << ",\"scale_mismatches\":" << scales
            << ",\"raw_sum_mismatches\":" << sums << ",\"code_mismatches\":" << codes << ",\"first_byte\":"; nullable(first_byte);
        std::cout << ",\"first_header_block\":"; nullable(first_header); std::cout << ",\"first_code_row\":"; nullable(first_code); std::cout << '}';
    }
};
void quantizer_source(std::span<const float> values, const Q8_1& encoded) {
    require(values.size() == 32, "quantizer source block");
    float amax = 0;
    std::size_t amax_row = 0;
    std::array<float, 32> sums{};
    for (std::size_t i = 0; i < 32; ++i) {
        if (std::fabs(values[i]) > amax) { amax = std::fabs(values[i]); amax_row = i; }
        sums[i] = values[i];
    }
    for (std::size_t mask = 1; mask < 32; mask *= 2)
        for (std::size_t i = 0; i < 32; ++i) if (!(i & mask)) { const float s = sums[i] + sums[i ^ mask]; sums[i] = s; sums[i ^ mask] = s; }
    std::cout << '{'; float_bits("original_FP32_amax", amax); std::cout << ','; float_bits("original_FP32_d", amax / 127.0f);
    std::cout << ','; float_bits("original_FP32_raw_sum_XOR", sums[0]);
    std::cout << ",\"amax_lane\":" << amax_row << ",\"stored_half_d_bits\":" << encoded.d << ",\"stored_half_raw_sum_bits\":" << encoded.s
        << ",\"reconstructed_half_d_bits\":" << float_to_half(amax / 127.0f)
        << ",\"reconstructed_half_raw_sum_bits\":" << float_to_half(sums[0]) << '}';
}
void quant_block(const char* name, std::size_t block, const CpuExpertScratch& cpu, const Capture& gpu) {
    std::cout << ',' << quote(name) << ':';
    if (block == absent) { std::cout << "null"; return; }
    require(block < 20, "quant divergence block range");
    std::cout << "{\"block\":" << block << ",\"cpu\":";
    quantizer_source(std::span(cpu.middle).subspan(block * 32, 32), cpu.middle_q8[block]);
    std::cout << ",\"gpu\":"; quantizer_source(std::span(gpu.middle).subspan(block * 32, 32), gpu.q8[block]);
    std::cout << ",\"rows\":[";
    for (std::size_t lane = 0; lane < 32; ++lane) {
        if (lane) std::cout << ',';
        const auto row = block * 32 + lane;
        std::cout << "{\"row\":" << row << ','; float_bits("cpu_gate", cpu.gate[row]); std::cout << ','; float_bits("gpu_gate", gpu.gate[row]);
        std::cout << ','; float_bits("cpu_up", cpu.up[row]); std::cout << ','; float_bits("gpu_up", gpu.up[row]);
        std::cout << ','; float_bits("cpu_middle", cpu.middle[row]); std::cout << ','; float_bits("gpu_middle", gpu.middle[row]);
        std::cout << ",\"cpu_code\":" << int(cpu.middle_q8[block].qs[lane]) << ",\"gpu_code\":" << int(gpu.q8[block].qs[lane]) << '}';
    }
    std::cout << "]}";
}
void device_case(Context& context, const Expert& e, std::span<const Q8_1> original, const CpuExpertScratch& cpu,
                 std::span<const float> cpu_output) {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, context.device), "device properties");
    require(std::string_view(properties.gcnArchName).substr(0, 6) == "gfx906" && properties.warpSize == 64, "expected gfx906 wave64");
    Stream stream(context); Gpu device(stream, e, original);
    context.stage = "canonical_GPU_replay";
    const auto gpu = device.run(e);
    const Metric gate({cpu.gate.data(), M}, gpu.gate, .0002, .00002), up({cpu.up.data(), M}, gpu.up, .0002, .00002);
    const Metric middle({cpu.middle.data(), M}, gpu.middle, .002, .002), down(cpu_output, gpu.down, .002, .002);
    const QuantDifference cross({cpu.middle_q8.data(), 20}, gpu.q8);
    context.stage = "same_float_quantizer_counterfactuals";
    std::array<Q8_1, 20> cpu_quant_gpu_middle{};
    quantize_q8(gpu.middle, cpu_quant_gpu_middle);
    const auto gpu_quant_cpu_middle = device.quantize_same({cpu.middle.data(), M});
    const QuantDifference same_gpu_float(cpu_quant_gpu_middle, gpu.q8), same_cpu_float({cpu.middle_q8.data(), 20}, gpu_quant_cpu_middle);
    context.stage = "common_Q8_down_counterfactuals";
    const auto cpu_own_replay = cpu_down_replay(e.down.view(), {cpu.middle_q8.data(), 20});
    require(std::memcmp(cpu_own_replay.data(), cpu_output.data(), H * sizeof(float)) == 0,
        "diagnostic CPU down replay must match unchanged production CPU arithmetic bitwise");
    const auto cpu_down_gpu_q8 = cpu_down_replay(e.down.view(), gpu.q8);
    const auto gpu_down_cpu_q8 = device.common_cpu_q8(e, {cpu.middle_q8.data(), 20});
    const Metric common_gpu_q8(cpu_down_gpu_q8, gpu.down, .002, .002), common_cpu_q8(cpu_output, gpu_down_cpu_q8, .002, .002);
    context.stage = "CPU_gate_up_GPU_nonlinearity_counterfactual";
    const auto counter = device.cpu_gate_up(e, cpu);
    const Metric same_cpu_gate_middle({cpu.middle.data(), M}, counter.middle, .002, .002), same_cpu_gate_down(cpu_output, counter.down, .002, .002);
    const QuantDifference same_cpu_gate_q8({cpu.middle_q8.data(), 20}, counter.q8);
    // Identical GPU g/u on the host isolates libm without changing either runtime.
    std::array<float, M> host_gpu_gate_middle{};
    for (std::size_t i = 0; i < M; ++i) host_gpu_gate_middle[i] = (gpu.gate[i] / (1.0f + std::exp(-gpu.gate[i]))) * gpu.up[i];
    finite(host_gpu_gate_middle);
    const Metric same_gpu_gate_middle(host_gpu_gate_middle, gpu.middle, .002, .002);
    device.readonly(e, original); e.unchanged(); stream.wait();
    const bool parity = gate.violations == 0 && up.violations == 0 && middle.violations == 0 && down.violations == 0 &&
        same_cpu_gate_middle.violations == 0 && same_cpu_gate_down.violations == 0 && same_gpu_gate_middle.violations == 0 &&
        common_gpu_q8.violations == 0 && common_cpu_q8.violations == 0 && same_gpu_float.bytes == 0 && same_cpu_float.bytes == 0;
    context.parity = context.parity && parity;
    std::cout << "{\"kind\":\"cpu_expert_failure_probe_device\",\"protocol\":1,\"device\":" << context.device
        << ",\"arch\":" << quote(properties.gcnArchName) << ",\"wave_size\":" << properties.warpSize
        << ",\"layer\":" << context.layer << ",\"expert\":" << context.expert << ",\"columns\":1,\"original_Q8_bytes\":2880"
        << ",\"linear_gate\":{\"absolute\":0.0002,\"relative\":0.00002},\"middle_down_gate\":{\"absolute\":0.002,\"relative\":0.002}";
    gate.emit("gate"); up.emit("up"); middle.emit("middle"); down.emit("down"); cross.emit("middle_Q8");
    same_gpu_float.emit("same_GPU_middle_float_quantization"); same_cpu_float.emit("same_CPU_middle_float_quantization");
    common_gpu_q8.emit("CPU_down_with_GPU_middle_Q8"); common_cpu_q8.emit("GPU_down_with_CPU_middle_Q8");
    same_cpu_gate_middle.emit("CPU_gate_up_GPU_SiLU_middle"); same_cpu_gate_down.emit("CPU_gate_up_GPU_SiLU_down");
    same_cpu_gate_q8.emit("CPU_gate_up_GPU_SiLU_Q8"); same_gpu_gate_middle.emit("GPU_gate_up_host_libm_middle");
    quant_block("first_header_divergence", cross.first_header, cpu, gpu);
    quant_block("first_code_divergence", cross.first_code == absent ? absent : cross.first_code / 32, cpu, gpu);
    std::cout << ",\"original_input_opaque_readonly\":true,\"canonical_weight_readonly\":true,\"GPU_canaries\":true"
        << ",\"CPU_down_replay_selfcheck_bitexact\":true,\"all_results_finite\":true,\"numerical_parity_passed\":" << (parity ? "true" : "false")
        << ",\"quantizer_FP32_statistics_scope\":\"host_reconstruction_from_actual_captured_middle_not_internal_device_register_capture\""
        << ",\"diagnostic_complete\":true,\"qualification_claim\":false}\n";
    ++context.completed_devices;
}
void run(Context& context, int argc, char** argv) {
    require(argc == 5, "usage: core-cpu-expert-failure-probe MODEL LAYER EXPERT Q8FILE");
    context.layer = integer(argv[2], 47); context.expert = integer(argv[3], 511);
    context.stage = "read_exact_original_Q8";
    alignas(64) const auto original = read_input(argv[4]); const auto original_copy = original;
    context.stage = "bounded_canonical_expert_read";
    Model model(argv[1]);
    const auto& architecture = model.metadata_value("general.architecture");
    require(architecture.type == MetadataType::STRING && architecture.get<std::string>() == "qwen4exp", "expected qwen4exp GGUF");
    Expert e(model, context.layer, context.expert);
    require(cpu_has_avx2(), "actual AVX2/FMA/F16C CPU required");
    context.stage = "unchanged_CPU_expert";
    CpuExpertScratch cpu{}; std::array<float, H> output{};
    CpuExpertJob job; job.expert = &e.cpu; job.columns = 1; job.input[0] = original; job.output = output;
    const auto status = run_cpu_expert(job, cpu);
    require(status == CpuExpertStatus::ok, std::string("CPU expert: ") + std::string(cpu_expert_status_name(status)));
    require(std::memcmp(cpu.gathered_x.data(), original.data(), 2880) == 0, "CPU original input gather changed bytes");
    finite({cpu.gate.data(), M}); finite({cpu.up.data(), M}); finite({cpu.middle.data(), M}); finite_q8({cpu.middle_q8.data(), 20}); finite(output);
    e.unchanged();
    std::cout << "{\"kind\":\"cpu_expert_failure_probe_source\",\"protocol\":1"; provenance();
    std::cout << ",\"model_path\":" << quote(argv[1]) << ",\"model_file_bytes\":" << model.file_size() << ",\"layer\":" << context.layer
        << ",\"expert\":" << context.expert << ",\"Q8_path\":" << quote(argv[4])
        << ",\"input_scope\":\"caller_supplied_exact_original_GPU_Q8_no_capture_provenance_attested\",\"input_bytes\":2880"
        << ",\"input_requantized\":false,\"weight_read_api\":\"Model.read_expert\",\"weight_reads\":3,\"weight_slices\":[";
    bool first = true; std::uint64_t payload = 0;
    for (const Matrix* m : {&e.gate, &e.up, &e.down}) {
        if (!first) std::cout << ',';
        first = false; payload += m->payload.size();
        std::cout << "{\"tensor\":" << quote(m->tensor.name) << ",\"type\":" << quote(type_name(m->tensor.type))
            << ",\"input\":" << m->input << ",\"output\":" << m->output << ",\"bytes\":" << m->payload.size()
            << ",\"file_offset\":" << m->tensor.file_offset + m->tensor.strides[2] * static_cast<std::uint64_t>(context.expert) << '}';
    }
    std::cout << "],\"weight_payload_bytes_read\":" << payload << ",\"GPU_path\":\"current_canonical_linear_then_Session_SiLU_then_Q8_then_down\""
        << ",\"CPU_down_counterfactual\":\"diagnostic_fragment_FMA_XOR_order_selfchecked_against_CpuExpert\""
        << ",\"HF_claim\":false,\"performance_claim\":false,\"full_logits_qualified\":false,\"R5_complete_claim\":false}\n";
    int devices = 0; check(hipGetDeviceCount(&devices), "device count"); require(devices == 2, "requires devices 0 and 1");
    for (int device : {0, 1}) {
        context.device = device; context.stage = "device_setup"; check(hipSetDevice(device), "set device");
        device_case(context, e, original, cpu, output);
        check(context.cleanup, context.cleanup_operation);
        require(std::memcmp(original.data(), original_copy.data(), 2880) == 0, "host original Q8 mutated");
    }
    require(context.completed_devices == 2, "both GPU diagnostic count");
}
} // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic()); std::cout << std::setprecision(17);
    Context context;
    try {
        run(context, argc, argv); // Expert, streams and buffers destroyed BEFORE footer.
        check(context.cleanup, context.cleanup_operation);
        require(static_cast<bool>(std::cout), "diagnostic JSONL write failure");
        std::cout << "{\"kind\":\"cpu_expert_failure_probe_complete\",\"protocol\":1"; provenance();
        std::cout << ",\"devices\":2,\"columns\":1,\"weight_reads\":3,\"diagnostic_complete\":true,\"numerical_parity_passed\":"
            << (context.parity ? "true" : "false") << ",\"qualification_claim\":false,\"full_logits_qualified\":false,\"R5_complete_claim\":false}\n";
        require(static_cast<bool>(std::cout), "diagnostic footer write failure"); return 0;
    } catch (const std::exception& e) {
        std::cout << "{\"kind\":\"cpu_expert_failure_probe_error\",\"protocol\":1"; provenance();
        std::cout << ",\"stage\":" << quote(context.stage) << ",\"device\":" << context.device << ",\"layer\":" << context.layer
            << ",\"expert\":" << context.expert << ",\"error\":" << quote(e.what()) << ",\"diagnostic_complete\":false,\"numerical_parity_passed\":false}\n";
        return 1;
    }
}
