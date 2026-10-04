// Correctness-only prerequisite for R5. No Session/cache/scheduler publication.
// Build core-cpu-expert-shadow from this file; add cpu_expert.cpp to qwen-core,
// link qwen-core + Threads::Threads + qwen-linear-gpu + qwen-session (which owns
// session_ops.hip), and define CORE_REVISION="<actual 40 hex>" / CORE_DIRTY=0|1.
// All linked arithmetic needs strict FP, no fast-math, -ffp-contract=off; HIP
// target gfx906, C++20. No R1 GpuExpert, MMQ, repack or replacement GPU kernels.
// Usage: core-cpu-expert-shadow MODEL F32_ROW0 F32_ROW1 F32_ROW2
// Each file is exactly 10240 little-endian bytes, contiguous FP32[2560]. These
// are caller-supplied captured hc_ffn_mix rows: paths do NOT attest HF/capture
// provenance. Only expert0 slices of layers0/6 are read, without changed weights.
// A successful run emits 1 source, 2 device, 4 contracts, 12 case records, then
// 1 complete AFTER all RAII cleanup. Failures emit context + a failed footer and
// exit1. Gates are frozen here, never CLI options. No timings/allocation claims.
#include "cpu_expert.hpp"
#include "hip/linear.cuh"
#include "hip/session_ops.cuh"
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
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

#if !defined(CORE_REVISION) || !defined(CORE_DIRTY)
#error "cpu expert shadow requires compiled CORE_REVISION and CORE_DIRTY"
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "cpu expert shadow requires strict floating point"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

namespace {
using namespace qwen;
constexpr int hidden = 2560, middle = 640, max_n = 3;
constexpr std::size_t guard_bytes = 64, row_bytes = 10240;
constexpr std::byte poison{0xa7};
constexpr double linear_abs = 2e-4, linear_rel = 2e-5;
constexpr double intermediate_abs = .002, intermediate_rel = .002;
constexpr std::string_view revision = CORE_REVISION;
constexpr bool valid_revision() {
    if (revision.size() != 40) return false;
    for (char c : revision)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    return true;
}
static_assert(valid_revision(), "use the actual full compiled Git revision");
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little);
static_assert(sizeof(Q8_1) == 36 && offsetof(Q8_1, qs) == 4);

struct Context {
    const char* stage = "arguments";
    int device = -1, layer = -1, columns = 0;
    std::size_t devices = 0, contracts = 0, cases = 0, comparisons = 0;
    std::size_t same_float_bytes = 0, pool_jobs = 0;
    hipError_t cleanup_status = hipSuccess;
    const char* cleanup_operation = "none";
    void cleanup(hipError_t status, const char* operation) noexcept {
        if (status != hipSuccess && cleanup_status == hipSuccess) {
            cleanup_status = status; cleanup_operation = operation;
        }
    }
};
void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
void cpu_status(CpuExpertStatus actual, CpuExpertStatus expected, const char* operation) {
    if (actual != expected)
        throw std::runtime_error(std::string(operation) + ": " + std::string(cpu_expert_status_name(actual)));
}
std::string quote(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c < 32 || c >= 127) {
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += char(c);
    }
    return result + '"';
}
void provenance() {
    std::cout << ",\"revision\":" << quote(revision) << ",\"dirty\":" << (CORE_DIRTY ? "true" : "false");
}
void number(double value) {
    if (std::isfinite(value)) std::cout << value;
    else std::cout << "null";
}
bool bytes_equal(const void* a, const void* b, std::size_t bytes) {
    return bytes == 0 || std::memcmp(a, b, bytes) == 0;
}
void poisoned(std::span<const std::byte> bytes, const char* message) {
    require(std::all_of(bytes.begin(), bytes.end(), [](auto b) { return b == poison; }), message);
}
void finite(std::span<const float> values) {
    for (float x : values) require(std::isfinite(x), "nonfinite float input/result");
}
void finite_q8(std::span<const Q8_1> values) {
    for (const auto& b : values) {
        require((b.d & 0x7c00U) != 0x7c00U && (b.s & 0x7c00U) != 0x7c00U &&
                (b.d & 0x8000U) == 0, "nonfinite/negative Q8 scale or raw sum");
        // Stored half(d)=0 does NOT imply original FP32 d=0: raw sum and
        // codes remain independent, including valid tiny captured/middle rows.
    }
}

template<class T, std::size_t Capacity> struct alignas(64) HostBuffer {
    std::array<std::byte, guard_bytes> before;
    alignas(64) std::array<T, Capacity> data;
    std::array<std::byte, guard_bytes> after;
    HostBuffer() { clear(); }
    void clear() {
        before.fill(poison); after.fill(poison);
        std::memset(data.data(), std::to_integer<int>(poison), sizeof(data));
    }
    std::span<T> live(std::size_t count) {
        require(count <= Capacity, "host live span exceeds capacity");
        return {data.data(), count};
    }
    std::span<const T> live(std::size_t count) const {
        require(count <= Capacity, "host const span exceeds capacity");
        return {data.data(), count};
    }
    void guards(std::size_t count) const {
        require(count <= Capacity, "host guard span exceeds capacity");
        poisoned(before, "host prefix canary"); poisoned(after, "host suffix canary");
        poisoned(std::as_bytes(std::span(data).subspan(count)), "host inactive tail canary");
    }
};
struct alignas(64) Workspace {
    std::array<std::byte, guard_bytes> before;
    CpuExpertScratch scratch;
    std::array<std::byte, guard_bytes> after;
    Workspace() { clear(); }
    void clear() {
        before.fill(poison); after.fill(poison);
        std::memset(&scratch, std::to_integer<int>(poison), sizeof(scratch));
    }
    void guards(int n) const {
        poisoned(before, "CPU scratch prefix"); poisoned(after, "CPU scratch suffix");
        const auto tail = [](const auto& a, std::size_t start) {
            poisoned(std::as_bytes(std::span(a).subspan(start)), "CPU scratch inactive tail");
        };
        tail(scratch.gathered_x, std::size_t(n) * 80); tail(scratch.middle_q8, std::size_t(n) * 20);
        tail(scratch.gate, std::size_t(n) * middle); tail(scratch.up, std::size_t(n) * middle);
        tail(scratch.middle, std::size_t(n) * middle); tail(scratch.down, std::size_t(n) * hidden);
    }
};

struct File {
    int fd = -1;
    explicit File(const char* path) : fd(::open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "open input row");
    }
    ~File() { if (fd >= 0) ::close(fd); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    void size() const {
        struct stat s{};
        if (::fstat(fd, &s) != 0) throw std::system_error(errno, std::generic_category(), "fstat input row");
        require(S_ISREG(s.st_mode) && s.st_size == off_t(row_bytes), "row must be a regular, exact 10240-byte file");
    }
    void read(std::span<float> destination) {
        require(destination.size_bytes() == row_bytes, "row destination size"); size();
        auto bytes = std::as_writable_bytes(destination);
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const auto n = ::pread(fd, bytes.data() + offset, bytes.size() - offset, off_t(offset));
            if (n < 0) {
                if (errno == EINTR) continue;
                throw std::system_error(errno, std::generic_category(), "pread input row");
            }
            require(n > 0, "input row truncated during read"); offset += std::size_t(n);
        }
        size(); finite(destination);
        const int descriptor = std::exchange(fd, -1);
        if (::close(descriptor) != 0) throw std::system_error(errno, std::generic_category(), "close input row");
    }
};
struct Matrix {
    TensorView tensor;
    int input, output;
    std::vector<std::byte> storage, original;
    Matrix(const Model& model, int layer, const char* part, int k, int m, TensorType type)
        : tensor(model.tensor("blk." + std::to_string(layer) + ".ffn_" + part + "_exps.weight")), input(k), output(m) {
        const auto layout = type_layout(type);
        const auto row = std::uint64_t(k / 32) * layout.block_bytes;
        const auto slice = row * std::uint64_t(m);
        require(tensor.type == type && tensor.rank == 3 && tensor.dimensions[0] == std::uint64_t(k) &&
                tensor.dimensions[1] == std::uint64_t(m) && tensor.dimensions[2] == 512 &&
                tensor.strides[0] == layout.block_bytes && tensor.strides[1] == row && tensor.strides[2] == slice &&
                tensor.byte_size == slice * 512 && tensor.file_offset >= model.data_offset() &&
                tensor.file_offset <= model.file_size() && slice <= model.file_size() - tensor.file_offset,
                "expected unchanged canonical expert geometry/type/offsets");
        storage.assign(std::size_t(slice) + 2 * guard_bytes, poison);
        model.read_expert(tensor.name, 0, writable()); // Exactly ONE expert, never read_tensor.
        original = storage;
    }
    std::span<std::byte> writable() { return {storage.data() + guard_bytes, storage.size() - 2 * guard_bytes}; }
    std::span<const std::byte> bytes() const { return {storage.data() + guard_bytes, storage.size() - 2 * guard_bytes}; }
    QMatrix view() const { return {tensor.type, input, output, bytes()}; }
    void unchanged() const { require(storage == original, "borrowed host weight bytes/canaries mutated"); }
};
struct Expert {
    Matrix gate, up, down;
    CpuExpert cpu;
    Expert(const Model& model, int layer)
        : gate(model, layer, "gate", hidden, middle, TensorType::Q4_0),
          up(model, layer, "up", hidden, middle, TensorType::Q4_0),
          down(model, layer, "down", middle, hidden, layer == 0 ? TensorType::Q4_1 : TensorType::Q4_0),
          cpu(gate.view(), up.view(), down.view()) {}
    void unchanged() const { gate.unchanged(); up.unchanged(); down.unchanged(); }
};

struct Stream {
    Context& context;
    hipStream_t value = nullptr;
    explicit Stream(Context& c) : context(c) { hip_check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking), "stream create"); }
    ~Stream() {
        if (value) {
            context.cleanup(hipStreamSynchronize(value), "stream synchronize cleanup");
            context.cleanup(hipStreamDestroy(value), "stream destroy");
        }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    void wait() const { hip_check(hipStreamSynchronize(value), "stream complete"); }
};
struct DeviceBuffer {
    Stream& stream;
    void* base = nullptr;
    std::size_t capacity;
    DeviceBuffer(Stream& s, std::size_t bytes) : stream(s), capacity(bytes) {
        hip_check(hipMalloc(&base, capacity + 2 * guard_bytes), "buffer allocate");
    }
    ~DeviceBuffer() {
        if (base) {
            stream.context.cleanup(hipStreamSynchronize(stream.value), "buffer synchronize cleanup");
            stream.context.cleanup(hipFree(base), "buffer free");
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    template<class T> T* as() const { return reinterpret_cast<T*>(static_cast<std::byte*>(base) + guard_bytes); }
    void clear() const {
        hip_check(hipMemsetAsync(base, std::to_integer<int>(poison), capacity + 2 * guard_bytes, stream.value), "buffer poison");
        stream.wait();
    }
    template<class T> void upload(std::span<const T> values) const {
        require(values.size_bytes() <= capacity, "device upload capacity"); clear();
        hip_check(hipMemcpyAsync(as<std::byte>(), values.data(), values.size_bytes(), hipMemcpyHostToDevice, stream.value), "buffer upload");
        stream.wait(); // Keep even a stack-local flag value alive until completion.
    }
    std::vector<std::byte> snapshot(std::size_t live) const {
        require(live <= capacity, "device snapshot capacity"); stream.wait();
        std::vector<std::byte> result(capacity + 2 * guard_bytes);
        hip_check(hipMemcpyAsync(result.data(), base, result.size(), hipMemcpyDeviceToHost, stream.value), "buffer readback");
        stream.wait();
        const auto bytes = std::span<const std::byte>(result);
        poisoned(bytes.first(guard_bytes), "device prefix canary");
        poisoned(bytes.subspan(guard_bytes + live), "device suffix/inactive canary");
        return result;
    }
    template<class T> void download(std::span<T> values) const {
        const auto bytes = snapshot(values.size_bytes());
        std::memcpy(values.data(), bytes.data() + guard_bytes, values.size_bytes());
    }
    template<class T> void unchanged(std::span<const T> values) const {
        const auto bytes = snapshot(values.size_bytes());
        require(bytes_equal(bytes.data() + guard_bytes, values.data(), values.size_bytes()), "device read-only payload changed");
    }
};
struct Capture {
    HostBuffer<float, 1920> gate, up, mid;
    HostBuffer<float, 7680> down;
    HostBuffer<Q8_1, 60> q8;
    void guards(int n) const {
        gate.guards(std::size_t(n) * middle); up.guards(std::size_t(n) * middle);
        mid.guards(std::size_t(n) * middle); down.guards(std::size_t(n) * hidden); q8.guards(std::size_t(n) * 20);
    }
};
struct Gpu {
    Stream& stream;
    DeviceBuffer gate_weights, up_weights, down_weights;
    DeviceBuffer gate, up, mid, q8, down, flag;
    DeviceBuffer same_float, same_q8, common_down;
    Gpu(Stream& s, const Expert& e)
        : stream(s), gate_weights(s, e.gate.bytes().size()), up_weights(s, e.up.bytes().size()),
          down_weights(s, e.down.bytes().size()), gate(s, 1920 * 4), up(s, 1920 * 4), mid(s, 1920 * 4),
          q8(s, 60 * sizeof(Q8_1)), down(s, 7680 * 4), flag(s, sizeof(int)),
          same_float(s, 1920 * 4), same_q8(s, 60 * sizeof(Q8_1)), common_down(s, 7680 * 4) {
        gate_weights.upload(e.gate.bytes()); up_weights.upload(e.up.bytes()); down_weights.upload(e.down.bytes());
    }
    QuantizedDeviceMatrix matrix(const Matrix& m, const DeviceBuffer& weights) const {
        return {m.tensor.type, weights.as<std::byte>(), m.input, m.output};
    }
    void set_flag(int value = 0) { flag.upload(std::span<const int>(&value, 1)); }
    int get_flag() const {
        int result = 0; flag.download(std::span<int>(&result, 1)); return result;
    }
    void zero_flag() const { require(get_flag() == 0, "nonzero sticky error in provisional GPU chain"); }
    void run(const Expert& e, const DeviceBuffer& input, int n, Capture& out) {
        const auto h = std::size_t(n) * middle, k = std::size_t(n) * hidden;
        gate.clear(); up.clear(); mid.clear(); q8.clear(); down.clear(); set_flag();
        hip_check(launch_quantized_linear(matrix(e.gate, gate_weights), input.as<Q8_1>(), n, gate.as<float>(), stream.value), "canonical gate");
        hip_check(launch_quantized_linear(matrix(e.up, up_weights), input.as<Q8_1>(), n, up.as<float>(), stream.value), "canonical up");
        gate.download(out.gate.live(h)); up.download(out.up.live(h)); zero_flag();
        finite(out.gate.live(h)); finite(out.up.live(h));
        hip_check(launch_silu_up(gate.as<float>(), up.as<float>(), int(h), mid.as<float>(), flag.as<int>(), stream.value), "Session SiLU up");
        zero_flag(); mid.download(out.mid.live(h)); finite(out.mid.live(h));
        hip_check(launch_quantize_q8_1(mid.as<float>(), q8.as<Q8_1>(), middle, n, flag.as<int>(), stream.value), "Session middle Q8");
        zero_flag(); q8.download(out.q8.live(h / 32)); finite_q8(out.q8.live(h / 32));
        hip_check(launch_quantized_linear(matrix(e.down, down_weights), q8.as<Q8_1>(), n, down.as<float>(), stream.value), "canonical down");
        zero_flag(); down.download(out.down.live(k)); finite(out.down.live(k)); out.guards(n);
        // Gate/up are read-only to SiLU, and middle/Q8 read-only to consumers.
        gate.unchanged<float>(out.gate.live(h)); up.unchanged<float>(out.up.live(h));
        mid.unchanged<float>(out.mid.live(h)); q8.unchanged<Q8_1>(out.q8.live(h / 32));
    }
    void unchanged(const Expert& e) const {
        gate_weights.unchanged(e.gate.bytes()); up_weights.unchanged(e.up.bytes()); down_weights.unchanged(e.down.bytes()); e.unchanged();
    }
};

struct Metric {
    std::size_t elements = 0, finite_pairs = 0, nonfinite = 0, violations = 0;
    double max_abs = 0, square_sum = 0, max_ratio = 0;
    std::size_t first_bad = 0;
    double bad_actual = 0, bad_reference = 0;
    bool passed() const { return elements != 0 && nonfinite == 0 && violations == 0; }
    void compare(std::span<const float> actual, std::span<const float> reference, double absolute, double relative) {
        require(actual.size() == reference.size(), "metric shape"); elements = actual.size();
        for (std::size_t i = 0; i < elements; ++i) {
            const double a = actual[i], r = reference[i];
            bool bad = !std::isfinite(a) || !std::isfinite(r);
            if (bad) ++nonfinite;
            else {
                ++finite_pairs;
                const double difference = std::abs(a - r), ratio = difference / (absolute + relative * std::abs(r));
                max_abs = std::max(max_abs, difference); max_ratio = std::max(max_ratio, ratio);
                square_sum += difference * difference; bad = ratio > 1;
            }
            if (bad) {
                if (violations == 0) { first_bad = i; bad_actual = a; bad_reference = r; }
                ++violations;
            }
        }
    }
    void emit(const char* name) const {
        std::cout << ',' << quote(name) << ":{\"elements\":" << elements << ",\"nonfinite\":" << nonfinite
                  << ",\"violations\":" << violations << ",\"max_abs\":";
        number(finite_pairs ? max_abs : std::numeric_limits<double>::quiet_NaN());
        std::cout << ",\"rms\":"; number(finite_pairs ? std::sqrt(square_sum / double(finite_pairs)) : std::numeric_limits<double>::quiet_NaN());
        std::cout << ",\"max_bound_ratio\":"; number(finite_pairs ? max_ratio : std::numeric_limits<double>::quiet_NaN());
        std::cout << ",\"first_bad\":";
        if (violations) {
            std::cout << "{\"index\":" << first_bad << ",\"actual\":"; number(bad_actual);
            std::cout << ",\"reference\":"; number(bad_reference); std::cout << '}';
        } else std::cout << "null";
        std::cout << '}';
    }
};
struct QuantDifference {
    std::size_t bytes = 0, scales = 0, sums = 0, codes = 0;
    int max_code_delta = 0;
    void compare(std::span<const Q8_1> a, std::span<const Q8_1> b) {
        require(a.size() == b.size(), "quant diagnostic shape");
        for (std::size_t i = 0; i < a.size(); ++i) {
            const auto x = std::as_bytes(a.subspan(i, 1)), y = std::as_bytes(b.subspan(i, 1));
            for (std::size_t j = 0; j < x.size(); ++j) bytes += std::size_t(x[j] != y[j]);
            scales += std::size_t(a[i].d != b[i].d); sums += std::size_t(a[i].s != b[i].s);
            for (std::size_t j = 0; j < 32; ++j) {
                codes += std::size_t(a[i].qs[j] != b[i].qs[j]);
                max_code_delta = std::max(max_code_delta, std::abs(int(a[i].qs[j]) - int(b[i].qs[j])));
            }
        }
    }
};
CpuExpertJob job(const CpuExpert& expert, std::span<const Q8_1> x, int n, std::span<float> out) {
    CpuExpertJob result; result.expert = &expert; result.columns = n; result.output = out;
    for (int c = 0; c < n; ++c) result.input[std::size_t(c)] = x.subspan(std::size_t(c) * 80, 80);
    return result;
}
void cpu_guards(const Workspace& work, const CpuExpertJob& j, const HostBuffer<float, 7680>& out) {
    const auto n = std::size_t(j.columns), h = n * middle, k = n * hidden;
    work.guards(j.columns); out.guards(k);
    const auto& s = work.scratch;
    for (std::size_t c = 0; c < n; ++c)
        require(bytes_equal(s.gathered_x.data() + c * 80, j.input[c].data(), 80 * sizeof(Q8_1)), "CPU changed borrowed Q8 gather bytes");
    finite({s.gate.data(), h}); finite({s.up.data(), h}); finite({s.middle.data(), h});
    finite({s.down.data(), k}); finite(out.live(k)); finite_q8({s.middle_q8.data(), h / 32});
    require(bytes_equal(s.down.data(), out.data.data(), k * sizeof(float)), "CPU success output copy");
}
// SEPARATE synthetic error-contract fixture. No captured row or model payload
// participates; actual-model parity cases below retain their original weights.
struct SyntheticNumeric {
    static Q4_0 block(bool large) {
        Q4_0 result{};
        result.d = large ? 0x7b53 : 0x3c00; // Exactly 60,000 or 1 in binary16.
        for (std::size_t j = 0; j < 16; ++j)
            result.qs[j] = large ? (j % 2 == 0 ? 0xff : 0) : 0x88;
        return result;
    }
    const std::vector<Q4_0> gate, up, down;
    CpuExpert cpu;
    HostBuffer<float, 7680> original;
    HostBuffer<Q8_1, 240> input;
    SyntheticNumeric()
        : gate(51200, block(true)), up(51200, block(true)), down(51200, block(false)),
          cpu({TensorType::Q4_0, hidden, middle, std::as_bytes(std::span(gate))},
              {TensorType::Q4_0, hidden, middle, std::as_bytes(std::span(up))},
              {TensorType::Q4_0, middle, hidden, std::as_bytes(std::span(down))}) {
        require(half_to_float(0x7b53) == 60000.0f, "synthetic exact finite half scale");
        for (std::size_t i = 0; i < original.data.size(); ++i)
            original.data[i] = i % 2 == 0 ? 7620000.0f : -7620000.0f;
        quantize_q8(original.live(7680), input.live(240));
        // Independent analytic full-block dot: 16 positive positions/block,
        // code15 * input127, raw-sum correction zero. exp(-g) rounds to zero.
        const double expected_gate = (hidden / 2) * 15.0 * 127.0 * 60000.0 * 60000.0;
        const double expected_middle = expected_gate * expected_gate;
        require(std::isfinite(expected_gate) && std::isfinite(expected_middle) &&
                expected_middle < double(std::numeric_limits<float>::max()) &&
                expected_middle / 127.0 >= 65520.0 && expected_middle * 32.0 >= 65520.0,
                "synthetic analytic finite FP32 middle must overflow half headers");
        unchanged();
    }
    void unchanged() const {
        original.guards(7680); input.guards(240); finite(original.live(7680)); finite_q8(input.live(240));
        for (std::size_t i = 0; i < original.data.size(); ++i)
            require(original.data[i] == (i % 2 == 0 ? 7620000.0f : -7620000.0f), "synthetic original input changed");
        for (const auto& b : input.live(240)) {
            require(b.d == 0x7b53 && b.s == 0, "synthetic canonical exact Q8 headers");
            for (std::size_t i = 0; i < 32; ++i)
                require(b.qs[i] == (i % 2 == 0 ? 127 : -127), "synthetic canonical exact Q8 codes");
        }
        for (const auto* weights : {&gate, &up, &down}) {
            const auto expected = block(weights != &down);
            for (const auto& b : *weights)
                require(bytes_equal(&b, &expected, sizeof(b)), "synthetic borrowed weight bytes changed");
        }
    }
    void rejected_middle(const Workspace& work) const {
        const auto& s = work.scratch;
        require(bytes_equal(s.gathered_x.data(), input.data.data(), 240 * sizeof(Q8_1)), "synthetic gather changed Q8 bytes");
        finite(s.gate); finite(s.up); finite(s.middle);
        for (std::size_t i = 0; i < s.middle.size(); ++i)
            require(s.gate[i] > 0 && s.up[i] > 0 && s.middle[i] / 127.0f >= 65520.0f &&
                    s.middle[i] * 32.0f >= 65520.0f, "synthetic rejection did not reach finite middle/half overflow");
        unchanged();
    }
};
struct Rounding {
    int saved = std::fegetround();
    Rounding() { require(saved != -1 && std::fesetround(FE_DOWNWARD) == 0, "set numeric failure FP environment"); }
    ~Rounding() { if (std::fesetround(saved) != 0) std::terminate(); }
};

void contracts(Context& context, Expert& e, Gpu& gpu, std::span<const Q8_1> input, CpuExpertPool& pool) {
    context.stage = "contracts";
    Workspace work;
    HostBuffer<float, 7680> output;
    const auto good = job(e.cpu, input, 3, output.live(7680));
    std::array<std::byte, sizeof(CpuExpertScratch)> before{};
    std::memcpy(before.data(), &work.scratch, before.size());
    std::size_t invalid = 0, gpu_invalid = 0;
    const auto reject = [&](const CpuExpertJob& bad) {
        cpu_status(run_cpu_expert(bad, work.scratch), CpuExpertStatus::invalid_argument, "CPU invalid descriptor");
        require(bytes_equal(before.data(), &work.scratch, before.size()), "invalid argument modified scratch");
        work.guards(0); output.guards(0); ++invalid;
    };
    auto bad = good; bad.expert = nullptr; reject(bad);
    for (int n : {0, 4}) { bad = good; bad.columns = n; reject(bad); }
    bad = good; bad.input[2] = bad.input[2].first(79); reject(bad);
    bad = good; bad.output = bad.output.first(7679); reject(bad);
    bad = good; bad.columns = 1; bad.output = output.live(2560); reject(bad); // Nonempty unused inputs.
    bad = good; bad.output = work.scratch.down; reject(bad);
    bad = good; bad.input[0] = {work.scratch.gathered_x.data(), 80}; reject(bad);
    bad = good; bad.output = {reinterpret_cast<float*>(const_cast<Q8_1*>(input.data())), 7680}; reject(bad);
    HostBuffer<Q8_1, 240> malformed;
    std::copy(input.begin(), input.end(), malformed.data.begin());
    malformed.data.back().s = 0x7c00;
    reject(job(e.cpu, malformed.live(240), 3, output.live(7680))); malformed.guards(240);
    {
        Rounding rounding;
        cpu_status(run_cpu_expert(good, work.scratch), CpuExpertStatus::numeric_error, "CPU FP environment numeric rejection");
    }
    // numeric_error allows unspecified scratch; output MUST remain untouched.
    output.guards(0); work.guards(0);
    cpu_status(run_cpu_expert(good, work.scratch), CpuExpertStatus::ok, "CPU recovery after rejection");
    cpu_guards(work, good, output);
    std::size_t ticket = 999;
    bad = good; bad.input[2] = bad.input[2].first(79);
    cpu_status(pool.submit(bad, ticket), CpuExpertStatus::invalid_argument, "pool invalid descriptor");
    require(ticket == 999, "rejected pool submit changed ticket");
    cpu_guards(work, good, output);
    cpu_status(pool.drain(), CpuExpertStatus::ok, "empty pool drain after rejection");
    cpu_status(pool.reset(), CpuExpertStatus::ok, "pool reset after rejection");
    // Deterministic SEPARATE synthetic weights + quantizer-produced valid Q8.
    // This rejection remains mandatory even if all three caller rows are zero.
    SyntheticNumeric synthetic;
    output.clear(); work.clear();
    const auto numeric = job(synthetic.cpu, synthetic.input.live(240), 3, output.live(7680));
    cpu_status(run_cpu_expert(numeric, work.scratch), CpuExpertStatus::numeric_error, "CPU separate synthetic arithmetic rejection");
    output.guards(0); work.guards(3); synthetic.rejected_middle(work);
    cpu_status(run_cpu_expert(good, work.scratch), CpuExpertStatus::ok, "CPU arithmetic error recovery");
    cpu_guards(work, good, output);
    HostBuffer<float, 7680> failed_pool_output, recovered_pool_output;
    ticket = 999;
    cpu_status(pool.submit(job(synthetic.cpu, synthetic.input.live(240), 3, failed_pool_output.live(7680)), ticket), CpuExpertStatus::ok, "pool accepted separate synthetic numeric job");
    require(ticket == 0, "pool numeric ticket");
    cpu_status(pool.drain(), CpuExpertStatus::numeric_error, "pool numeric batch rejection");
    const auto failed = pool.result(ticket);
    require(failed.state == CpuExpertJobState::complete && failed.status == CpuExpertStatus::numeric_error, "pool numeric result");
    failed_pool_output.guards(0); synthetic.unchanged();
    cpu_status(pool.submit(job(e.cpu, input, 3, recovered_pool_output.live(7680)), ticket), CpuExpertStatus::ok, "pool healthy job after failure");
    require(ticket == 1, "pool failed ticket overwritten");
    cpu_status(pool.drain(), CpuExpertStatus::numeric_error, "pool drain must retain first batch error");
    const auto recovered = pool.result(ticket);
    require(recovered.state == CpuExpertJobState::complete && recovered.status == CpuExpertStatus::ok, "pool healthy subsequent status");
    failed_pool_output.guards(0); recovered_pool_output.guards(7680); finite(recovered_pool_output.live(7680));
    require(bytes_equal(recovered_pool_output.data.data(), output.data.data(), 7680 * sizeof(float)), "pool recovery/direct mismatch");
    synthetic.unchanged();
    cpu_status(pool.reset(), CpuExpertStatus::ok, "failed pool batch reset");
    context.pool_jobs += 2;
    const auto constructor_reject = [&](QMatrix gate, QMatrix down) {
        bool rejected = false;
        try { CpuExpert invalid_expert(gate, e.up.view(), down); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "CPU invalid borrowed weight descriptor accepted"); ++invalid;
    };
    auto gate = e.gate.view(); gate.type = TensorType::Q4_1; constructor_reject(gate, e.down.view());
    auto down = e.down.view(); down.weights = down.weights.first(down.weights.size() - 1); constructor_reject(e.gate.view(), down);

    gpu.same_float.clear(); gpu.same_q8.clear(); gpu.common_down.clear(); gpu.set_flag();
    const auto invalid_gpu = [&](hipError_t status) {
        require(status == hipErrorInvalidValue, "GPU pre-enqueue rejection status"); ++gpu_invalid;
    };
    invalid_gpu(launch_quantized_linear(gpu.matrix(e.gate, gpu.gate_weights), gpu.same_q8.as<Q8_1>(), 4, gpu.common_down.as<float>(), gpu.stream.value));
    invalid_gpu(launch_quantize_q8_1(gpu.same_float.as<float>(), gpu.same_q8.as<Q8_1>(), 31, 1, gpu.flag.as<int>(), gpu.stream.value));
    invalid_gpu(launch_silu_up(gpu.same_float.as<float>(), gpu.same_float.as<float>(), 0, gpu.common_down.as<float>(), gpu.flag.as<int>(), gpu.stream.value));
    invalid_gpu(launch_silu_up(gpu.same_float.as<float>(), gpu.same_float.as<float>(), middle, gpu.same_float.as<float>(), gpu.flag.as<int>(), gpu.stream.value));
    gpu.same_float.snapshot(0); gpu.same_q8.snapshot(0); gpu.common_down.snapshot(0); gpu.zero_flag();
    HostBuffer<float, 1920> probe;
    HostBuffer<Q8_1, 60> packed;
    std::fill_n(probe.data.begin(), middle, 0.0f);
    probe.data[middle - 1] = std::numeric_limits<float>::quiet_NaN();
    gpu.same_float.upload<float>(probe.live(middle)); gpu.same_q8.clear(); gpu.set_flag(2);
    hip_check(launch_quantize_q8_1(gpu.same_float.as<float>(), gpu.same_q8.as<Q8_1>(), middle, 1, gpu.flag.as<int>(), gpu.stream.value), "quantizer sticky error probe");
    require(gpu.get_flag() == 3, "quantizer failed to preserve/OR sticky bits");
    gpu.same_q8.download(packed.live(20));
    const Q8_1 zero{};
    for (const auto& b : packed.live(20)) require(bytes_equal(&b, &zero, sizeof(b)), "failed/zero quantizer block not zero");
    gpu.same_float.unchanged<float>(probe.live(middle)); probe.guards(middle); packed.guards(20);
    probe.data[middle - 1] = std::numeric_limits<float>::denorm_min();
    gpu.same_float.upload<float>(probe.live(middle)); gpu.same_q8.clear(); gpu.set_flag();
    hip_check(launch_quantize_q8_1(gpu.same_float.as<float>(), gpu.same_q8.as<Q8_1>(), middle, 1, gpu.flag.as<int>(), gpu.stream.value), "quantizer scale numeric probe");
    require(gpu.get_flag() == 2, "quantizer scale numeric error flag");
    gpu.same_q8.download(packed.live(20));
    for (const auto& b : packed.live(20)) require(bytes_equal(&b, &zero, sizeof(b)), "numeric-failed quantizer block not zero");
    gpu.same_float.unchanged<float>(probe.live(middle));
    probe.data[middle - 1] = 0;
    gpu.same_float.upload<float>(probe.live(middle)); gpu.same_q8.clear(); // Keep bit2 sticky on healthy work.
    hip_check(launch_quantize_q8_1(gpu.same_float.as<float>(), gpu.same_q8.as<Q8_1>(), middle, 1, gpu.flag.as<int>(), gpu.stream.value), "healthy sticky probe");
    require(gpu.get_flag() == 2, "healthy quantization cleared sticky error");
    gpu.same_q8.download(packed.live(20)); finite_q8(packed.live(20));
    gpu.set_flag(); gpu.same_q8.clear();
    hip_check(launch_quantize_q8_1(gpu.same_float.as<float>(), gpu.same_q8.as<Q8_1>(), middle, 1, gpu.flag.as<int>(), gpu.stream.value), "quantizer numeric recovery");
    gpu.zero_flag(); gpu.same_q8.download(packed.live(20));
    for (const auto& b : packed.live(20)) require(bytes_equal(&b, &zero, sizeof(b)), "healthy zero quantization recovery");
    gpu.same_float.unchanged<float>(probe.live(middle));
    // Exercise Session's pointwise nonfinite input and arithmetic sanitization.
    HostBuffer<float, 1920> operand, result;
    std::fill_n(operand.data.begin(), middle, 1.0f);
    for (int test : {0, 1, 2}) {
        std::fill_n(probe.data.begin(), middle, test == 2 ? -1000.0f : 0.0f);
        probe.data[middle - 1] = test == 0 ? std::numeric_limits<float>::infinity() :
                                      test == 1 ? std::numeric_limits<float>::max() : -1000.0f;
        operand.data[middle - 1] = test == 1 ? std::numeric_limits<float>::max() : 1.0f;
        gpu.same_float.upload<float>(probe.live(middle)); gpu.up.upload<float>(operand.live(middle));
        gpu.mid.clear(); gpu.set_flag(test == 0 ? 2 : 0);
        hip_check(launch_silu_up(gpu.same_float.as<float>(), gpu.up.as<float>(), middle, gpu.mid.as<float>(), gpu.flag.as<int>(), gpu.stream.value), "Session SiLU error/recovery probe");
        require(gpu.get_flag() == (test == 0 ? 3 : test == 1 ? 2 : 0), "Session SiLU sticky/numeric flag");
        gpu.mid.download(result.live(middle)); finite(result.live(middle));
        require(result.data[middle - 1] == 0, "Session SiLU sanitization/negative exp overflow");
        gpu.same_float.unchanged<float>(probe.live(middle)); gpu.up.unchanged<float>(operand.live(middle));
        probe.guards(middle); operand.guards(middle); result.guards(middle);
    }
    gpu.unchanged(e);
    std::cout << "{\"kind\":\"cpu_expert_shadow_contracts\",\"protocol\":1,\"device\":" << context.device
              << ",\"layer\":" << context.layer << ",\"expert\":0,\"cpu_invalid_rejects\":" << invalid
              << ",\"cpu_numeric_rejects\":2,\"cpu_numeric_probes\":[\"non_nearest_fp_environment\",\"separate_synthetic_Q4_Q8_middle_half_overflow\"]"
                 ",\"arithmetic_rejection_model_or_capture_dependent\":false"
                 ",\"cpu_output_preserved\":true,\"pool_invalid_ticket_preserved\":true,\"pool_failed_output_and_first_error_preserved\":true,\"gpu_invalid_rejects\":" << gpu_invalid
              << ",\"gpu_quantizer_numeric_probes\":2,\"gpu_silu_numeric_probes\":2,\"sticky_and_recovery\":true,\"passed\":true}\n";
    ++context.contracts;
}

void run_case(Context& context, Expert& e, Gpu& gpu, const DeviceBuffer& device_input,
              std::span<const Q8_1> input, Workspace& work, CpuExpertPool& pool, int n) {
    context.stage = "expert_case"; context.columns = n;
    const auto h = std::size_t(n) * middle, k = std::size_t(n) * hidden, q = h / 32;
    Capture first_gpu, repeated_gpu;
    HostBuffer<float, 7680> cpu_output, first_cpu_output, common, pool_a, pool_b;
    HostBuffer<float, 1920> same_gate_middle;
    HostBuffer<Q8_1, 60> cpu_quantized_gpu_middle, gpu_quantized_cpu_middle;
    work.clear();
    const auto j = job(e.cpu, input, n, cpu_output.live(k));
    gpu.run(e, device_input, n, first_gpu);
    cpu_status(run_cpu_expert(j, work.scratch), CpuExpertStatus::ok, "borrowed CPU expert");
    cpu_guards(work, j, cpu_output);
    // Snapshot object bytes explicitly, including padding, rather than depending
    // on padding preservation by a memberwise C++ copy constructor.
    std::array<std::byte, sizeof(CpuExpertScratch)> first_scratch{};
    std::memcpy(first_scratch.data(), &work.scratch, first_scratch.size());
    std::copy(cpu_output.live(k).begin(), cpu_output.live(k).end(), first_cpu_output.data.begin());
    Metric gate, up, mid, same_gate_mid, down, common_down;
    gate.compare({work.scratch.gate.data(), h}, first_gpu.gate.live(h), linear_abs, linear_rel);
    up.compare({work.scratch.up.data(), h}, first_gpu.up.live(h), linear_abs, linear_rel);
    mid.compare({work.scratch.middle.data(), h}, first_gpu.mid.live(h), intermediate_abs, intermediate_rel);
    down.compare(cpu_output.live(k), first_gpu.down.live(k), intermediate_abs, intermediate_rel);
    // Isolate host/device expf on identical GPU gate/up FP32, independently of
    // linear differences. expf itself may overflow for a finite negative gate;
    // the divided activation and final middle must nevertheless be finite.
    for (std::size_t i = 0; i < h; ++i) {
        const float activated = first_gpu.gate.data[i] / (1.0f + std::exp(-first_gpu.gate.data[i]));
        same_gate_middle.data[i] = activated * first_gpu.up.data[i];
    }
    same_gate_mid.compare(same_gate_middle.live(h), first_gpu.mid.live(h), intermediate_abs, intermediate_rel);
    quantize_q8(first_gpu.mid.live(h), cpu_quantized_gpu_middle.live(q));
    const bool same_gpu_float = bytes_equal(cpu_quantized_gpu_middle.data.data(), first_gpu.q8.data.data(), q * sizeof(Q8_1));
    // The other direction uses EXACT CPU middle floats, not the device expf
    // result. These two same-float tests are mandatory byte equality gates.
    gpu.same_float.upload<float>({work.scratch.middle.data(), h}); gpu.same_q8.clear(); gpu.set_flag();
    hip_check(launch_quantize_q8_1(gpu.same_float.as<float>(), gpu.same_q8.as<Q8_1>(), middle, n, gpu.flag.as<int>(), gpu.stream.value), "same CPU middle GPU quantization");
    gpu.zero_flag(); gpu.same_q8.download(gpu_quantized_cpu_middle.live(q));
    gpu.same_float.unchanged<float>({work.scratch.middle.data(), h});
    const bool same_cpu_float = bytes_equal(gpu_quantized_cpu_middle.data.data(), work.scratch.middle_q8.data(), q * sizeof(Q8_1));
    cpu_quantized_gpu_middle.guards(q); gpu_quantized_cpu_middle.guards(q);
    finite_q8(cpu_quantized_gpu_middle.live(q)); finite_q8(gpu_quantized_cpu_middle.live(q));
    gpu.same_q8.upload<Q8_1>({work.scratch.middle_q8.data(), q}); gpu.common_down.clear();
    hip_check(launch_quantized_linear(gpu.matrix(e.down, gpu.down_weights), gpu.same_q8.as<Q8_1>(), n, gpu.common_down.as<float>(), gpu.stream.value), "common CPU-middle-Q8 down");
    gpu.zero_flag(); gpu.common_down.download(common.live(k)); finite(common.live(k)); common.guards(k);
    gpu.same_q8.unchanged<Q8_1>({work.scratch.middle_q8.data(), q});
    common_down.compare(cpu_output.live(k), common.live(k), intermediate_abs, intermediate_rel);
    QuantDifference diagnostic;
    diagnostic.compare({work.scratch.middle_q8.data(), q}, first_gpu.q8.live(q));
    // Reuse the same scratch/device buffers and original borrowed Q8, with all
    // output bounds re-poisoned. Repeat equality is a same-path stability check,
    // never a CPU/device expf or end-to-end bit-identity requirement.
    cpu_output.clear(); gpu.run(e, device_input, n, repeated_gpu);
    cpu_status(run_cpu_expert(j, work.scratch), CpuExpertStatus::ok, "CPU repeated reuse");
    cpu_guards(work, j, cpu_output);
    require(bytes_equal(first_scratch.data(), &work.scratch, first_scratch.size()) &&
            bytes_equal(cpu_output.data.data(), first_cpu_output.data.data(), k * 4), "CPU repeat instability");
    for (auto pair : {std::pair{&first_gpu.gate, &repeated_gpu.gate}, std::pair{&first_gpu.up, &repeated_gpu.up}, std::pair{&first_gpu.mid, &repeated_gpu.mid}})
        require(bytes_equal(pair.first->data.data(), pair.second->data.data(), h * 4), "GPU gate/up/middle repeat instability");
    require(bytes_equal(first_gpu.down.data.data(), repeated_gpu.down.data.data(), k * 4) &&
            bytes_equal(first_gpu.q8.data.data(), repeated_gpu.q8.data.data(), q * sizeof(Q8_1)), "GPU down/Q8 repeat instability");
    // Persistent pool consumes the same exact borrowed GPU bytes. No affinity,
    // concurrency/DDR speed, or worker-private scratch-bound claim is made here.
    std::size_t a = 999, b = 999;
    cpu_status(pool.submit(job(e.cpu, input, n, pool_a.live(k)), a), CpuExpertStatus::ok, "pool submit A");
    cpu_status(pool.submit(job(e.cpu, input, n, pool_b.live(k)), b), CpuExpertStatus::ok, "pool submit B");
    cpu_status(pool.drain(), CpuExpertStatus::ok, "pool publication gate");
    require(a == 0 && b == 1, "pool tickets after reset");
    for (auto t : {a, b}) {
        const auto r = pool.result(t);
        require(r.state == CpuExpertJobState::complete && r.status == CpuExpertStatus::ok, "pool completed status");
    }
    for (const auto* out : {&pool_a, &pool_b}) {
        out->guards(k); finite(out->live(k));
        require(bytes_equal(out->data.data(), cpu_output.data.data(), k * 4), "pool/direct same-path reuse mismatch");
    }
    cpu_status(pool.reset(), CpuExpertStatus::ok, "pool reuse reset"); context.pool_jobs += 2;
    first_cpu_output.guards(k); same_gate_middle.guards(h); first_gpu.guards(n); repeated_gpu.guards(n);
    gpu.unchanged(e);
    const bool passed = gate.passed() && up.passed() && mid.passed() && same_gate_mid.passed() &&
                        down.passed() && common_down.passed() && same_gpu_float && same_cpu_float;
    context.comparisons += 4 * h + 2 * k; context.same_float_bytes += 2 * q * sizeof(Q8_1);
    std::cout << "{\"kind\":\"cpu_expert_shadow_case\",\"protocol\":1,\"device\":" << context.device
              << ",\"layer\":" << context.layer << ",\"expert\":0,\"columns\":" << n
              << ",\"down_type\":" << quote(type_name(e.down.tensor.type)) << ",\"repeats\":2"
                 ",\"same_gpu_middle_float_quant_bytes_equal\":" << (same_gpu_float ? "true" : "false")
              << ",\"same_cpu_middle_float_quant_bytes_equal\":" << (same_cpu_float ? "true" : "false")
              << ",\"cross_middle_q8_diagnostic_only\":{\"byte_mismatches\":" << diagnostic.bytes
              << ",\"scale_mismatches\":" << diagnostic.scales << ",\"raw_sum_mismatches\":" << diagnostic.sums
              << ",\"code_mismatches\":" << diagnostic.codes << ",\"max_code_delta\":" << diagnostic.max_code_delta << '}';
    gate.emit("gate"); up.emit("up"); mid.emit("middle"); same_gate_mid.emit("middle_same_gpu_gate_up");
    down.emit("down"); common_down.emit("down_common_cpu_middle_q8");
    std::cout << ",\"readonly_and_canaries\":true,\"repeated_reuse\":true,\"pool_jobs\":2,\"passed\":" << (passed ? "true" : "false") << "}\n";
    require(passed, "frozen numerical or same-float quantizer gate failed"); ++context.cases;
}

void source(const Model& model, const Expert& a, const Expert& b, char** argv) {
    std::cout << "{\"kind\":\"cpu_expert_shadow_source\",\"protocol\":1"; provenance();
    std::cout << ",\"model_path\":" << quote(argv[1]) << ",\"model_file_bytes\":" << model.file_size()
              << ",\"input_scope\":\"caller_supplied_captured_hc_ffn_mix_FP32_rows\",\"capture_attested\":false"
                 ",\"hf_source_attested\":false,\"input_dtype\":\"little_endian_FP32\",\"input_elements_each\":2560"
                 ",\"input_bytes_each\":10240,\"inputs\":[";
    for (int i = 2; i < 5; ++i) { if (i != 2) std::cout << ','; std::cout << quote(argv[i]); }
    std::cout << "],\"weight_read_api\":\"Model.read_expert\",\"weight_slices\":[";
    bool first = true;
    for (const Expert* e : {&a, &b}) for (const Matrix* m : {&e->gate, &e->up, &e->down}) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"tensor\":" << quote(m->tensor.name) << ",\"expert\":0,\"type\":" << quote(type_name(m->tensor.type))
                  << ",\"input\":" << m->input << ",\"output\":" << m->output << ",\"bytes\":" << m->bytes().size()
                  << ",\"file_offset\":" << m->tensor.file_offset << '}';
    }
    std::cout << "],\"weight_payload_bytes_read\":5632000,\"gpu_path\":\"current_canonical_linear_max3_then_Session_silu_up_then_Q8_then_canonical_down\""
                 ",\"linear_gate\":{\"absolute\":0.0002,\"relative\":0.00002},\"middle_down_gate\":{\"absolute\":0.002,\"relative\":0.002}"
                  ",\"comparison_reference\":\"actual_current_GPU_path\",\"pool_workers\":2,\"pool_pinned\":false"
                  ",\"cpu_arithmetic_rejection_probe\":{\"scope\":\"separate_synthetic_not_model_or_captured_rows\",\"weight_type\":\"Q4_0\",\"gate_up_shape\":[2560,640],\"down_shape\":[640,2560],\"gate_up_half_scale\":60000,\"input_values\":[7620000,-7620000],\"input_producer\":\"quantize_q8_exact_FP32\",\"input_half_scale\":60000,\"input_raw_sum\":0,\"input_codes\":[127,-127],\"expected_failure\":\"finite_middle_half_scale_and_raw_sum_overflow\"}"
                 ",\"hot_allocations_measured\":false,\"performance_measured\":false,\"device_expf_bitparity_claimed\":false"
                 ",\"full_logits_qualified\":false,\"R5_closed\":false}\n";
}
void run(Context& context, int argc, char** argv) {
    require(argc == 5, "usage: core-cpu-expert-shadow MODEL F32_ROW0 F32_ROW1 F32_ROW2");
    context.stage = "read_input_rows";
    HostBuffer<float, 7680> rows;
    for (int i = 0; i < max_n; ++i) { File file(argv[i + 2]); file.read(rows.live(7680).subspan(std::size_t(i) * hidden, hidden)); }
    const auto original_rows = rows;
    rows.guards(7680); finite(rows.live(7680));
    context.stage = "checked_model_and_bounded_slices";
    Model model(argv[1]);
    const auto& architecture = model.metadata_value("general.architecture");
    require(architecture.type == MetadataType::STRING && architecture.get<std::string>() == "qwen4exp", "expected qwen4exp GGUF");
    Expert layer0(model, 0), layer6(model, 6);
    source(model, layer0, layer6, argv);
    require(cpu_has_avx2(), "CpuExpert requires actual AVX2/FMA/F16C CPU support");
    HostBuffer<Q8_1, 240> cpu_original_q8;
    quantize_q8(rows.live(7680), cpu_original_q8.live(240)); cpu_original_q8.guards(240); finite_q8(cpu_original_q8.live(240));
    context.stage = "devices";
    int count = 0; hip_check(hipGetDeviceCount(&count), "device count"); require(count >= 2, "both GPU devices 0 and 1 required");
    for (int device : {0, 1}) {
        context.device = device; context.layer = -1; context.columns = 0; context.stage = "device_setup";
        hip_check(hipSetDevice(device), "set device");
        hipDeviceProp_t properties{}; hip_check(hipGetDeviceProperties(&properties, device), "device properties");
        require(std::string_view(properties.gcnArchName).substr(0, 6) == "gfx906" && properties.warpSize == 64, "expected gfx906 wave64");
        Stream stream(context);
        DeviceBuffer raw(stream, 7680 * 4), original_q8(stream, 240 * sizeof(Q8_1)), flag(stream, sizeof(int));
        raw.upload<float>(rows.live(7680)); original_q8.clear();
        const int zero = 0; flag.upload(std::span<const int>(&zero, 1));
        // Quantize ALL THREE ORIGINAL rows exactly once on this device. Every
        // N1/N2/N3 job consumes prefixes of these GPU-produced bytes unchanged.
        hip_check(launch_quantize_q8_1(raw.as<float>(), original_q8.as<Q8_1>(), hidden, max_n, flag.as<int>(), stream.value), "original FP32 GPU Q8 once");
        int error = -1; flag.download(std::span<int>(&error, 1)); require(error == 0, "original quantizer sticky error");
        HostBuffer<Q8_1, 240> borrowed;
        original_q8.download(borrowed.live(240)); finite_q8(borrowed.live(240)); borrowed.guards(240);
        require(bytes_equal(borrowed.data.data(), cpu_original_q8.data.data(), 240 * sizeof(Q8_1)), "original FP32 GPU/CPU quantizer byte parity");
        context.same_float_bytes += 240 * sizeof(Q8_1);
        const auto borrowed_before = borrowed;
        std::cout << "{\"kind\":\"cpu_expert_shadow_device\",\"protocol\":1,\"device\":" << device
                  << ",\"name\":" << quote(properties.name) << ",\"arch\":" << quote(properties.gcnArchName)
                  << ",\"wave_size\":" << properties.warpSize << ",\"total_vram_bytes\":" << properties.totalGlobalMem
                  << ",\"original_quantizer_launches\":1,\"original_Q8_bytes\":8640,\"original_same_float_bytes_equal\":true}\n";
        ++context.devices;
        for (auto pair : {std::pair{0, &layer0}, std::pair{6, &layer6}}) {
            context.layer = pair.first; context.columns = 0;
            Expert& expert = *pair.second;
            Gpu gpu(stream, expert);
            CpuExpertPool pool({2, false, {}, cpu_expert_max_workers});
            contracts(context, expert, gpu, borrowed.live(240), pool);
            Workspace workspace;
            for (int n : {1, 2, 3}) {
                run_case(context, expert, gpu, original_q8, borrowed.live(240), workspace, pool, n);
                original_q8.unchanged<Q8_1>(borrowed_before.live(240)); raw.unchanged<float>(original_rows.live(7680));
                require(bytes_equal(borrowed.data.data(), borrowed_before.data.data(), 240 * sizeof(Q8_1)), "CPU modified borrowed original GPU Q8");
                borrowed.guards(240); rows.guards(7680); cpu_original_q8.guards(240);
                require(bytes_equal(rows.data.data(), original_rows.data.data(), 7680 * 4), "host original FP32 changed");
            }
        }
        stream.wait();
    }
    layer0.unchanged(); layer6.unchanged();
    context.stage = "expected_counts";
    require(context.devices == 2 && context.contracts == 4 && context.cases == 12 && context.pool_jobs == 32 &&
            context.comparisons == 184320 && context.same_float_bytes == 51840, "fixture expected counts");
}
} // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic()); std::cout << std::setprecision(12);
    Context context;
    bool passed = false;
    try {
        run(context, argc, argv); // Every model/pool/stream/buffer is gone on return.
        hip_check(context.cleanup_status, context.cleanup_operation);
        passed = true;
    } catch (const std::exception& error) {
        std::cout << "{\"kind\":\"cpu_expert_shadow_failure\",\"protocol\":1"; provenance();
        std::cout << ",\"stage\":" << quote(context.stage) << ",\"device\":" << context.device
                  << ",\"layer\":" << context.layer << ",\"columns\":" << context.columns
                  << ",\"message\":" << quote(error.what()) << ",\"cleanup_status\":" << int(context.cleanup_status)
                  << ",\"passed\":false}\n";
        std::cerr << "core-cpu-expert-shadow: " << error.what() << '\n';
    }
    std::cout << "{\"kind\":\"cpu_expert_shadow_complete\",\"protocol\":1"; provenance();
    std::cout << ",\"devices\":" << context.devices << ",\"contracts\":" << context.contracts
              << ",\"cases\":" << context.cases << ",\"comparison_elements\":" << context.comparisons
              << ",\"same_float_quant_bytes_compared\":" << context.same_float_bytes << ",\"pool_jobs\":" << context.pool_jobs
              << ",\"raii_scope_exited\":true,\"cleanup_complete\":" << (context.cleanup_status == hipSuccess ? "true" : "false")
              << ",\"cleanup_status\":" << int(context.cleanup_status)
              << ",\"cleanup_operation\":" << quote(context.cleanup_operation) << ",\"passed\":" << (passed ? "true" : "false") << "}\n";
    return passed ? 0 : 1;
}
