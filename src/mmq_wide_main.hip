#include "hip/mmq_wide.cuh"
#include "linear_reference.hpp"
#include "model.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cfenv>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// core-mmq-wide MODEL.gguf. Link hip/mmq_wide.hip, qwen-mmq-gpu,
// qwen-linear-gpu and qwen-core; C++20, gfx906, no fast-math, -ffp-contract=off.
// KEEP the existing launchers/36B Q8 producer/144B DS4 packer, not another kernel.
// The specialized donor is mx dcd685463d597d31f5ca759d32c94592a2740fa4,
// MIT (third_party/mx-LICENSE); adaptation/provenance lives in mmq_wide.hip.
// This separate protocol is NOT the frozen Q4 collector's protocol. Numeric
// head coverage is sampled ONLY; all head outputs are computed/finite-checked.
// Counts below are case/validation counts, not counts of HIP kernel launches.
namespace {
constexpr double absolute_gate = 2e-4, relative_gate = 2e-5;
constexpr int repeats = 20;
constexpr std::string_view protocol = "qwen.mmq-wide.v1";
constexpr std::array widths32{32, 160, 640, 2560, 16384};
constexpr std::array widths256{256, 768, 2560, 16384};
constexpr std::array row_counts{1, 63, 64, 65, 127, 128, 129};
constexpr std::array column_counts{1, 3, 4, 17, 65, 128};
constexpr std::array benchmark_columns{1, 3, 8, 32, 128};
constexpr std::array types{qwen::TensorType::Q4_0, qwen::TensorType::Q4_1,
    qwen::TensorType::Q5_0, qwen::TensorType::Q8_0, qwen::TensorType::Q6_K};
constexpr std::array<std::string_view, 3> phases{"A1", "B", "A2"};
// EXACT row IDs from head_main.hip, including the last physical row.
constexpr std::array head_rows{0,1,7,31,63,127,255,511,1023,4095,8191,16383,
    32767,65535,131071,200003,248043,248044,248045,248046,248047,248127,248318,248319};
constexpr std::array<std::int8_t, 16> subscales{
    -128,-97,-73,-41,-19,-7,-3,-1,1,5,13,29,47,71,103,127};
static_assert(sizeof(qwen::Q8_1) == 36 && offsetof(qwen::Q8_1, qs) == 4);
static_assert(sizeof(qwen::Q8MmqBlock) == 144 && alignof(qwen::Q8MmqBlock) == 16 &&
    offsetof(qwen::Q8MmqBlock, qs) == 16);
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little);
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "mmq_wide_main.hip requires strict floating-point arithmetic"
#endif

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) { std::cerr << "wide cleanup: " << hipGetErrorString(status) << '\n'; std::abort(); }
}
struct Owners {
    std::size_t buffers = 0, streams = 0, events = 0;
    void empty() const { require(!buffers && !streams && !events, "live device owners after cleanup"); }
} owners;
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); ++owners.streams; }
    ~Stream() { cleanup(hipStreamSynchronize(value)); cleanup(hipStreamDestroy(value)); --owners.streams; }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Event {
    hipStream_t stream;
    hipEvent_t value = nullptr;
    explicit Event(hipStream_t s) : stream(s) { check(hipEventCreate(&value)); ++owners.events; }
    ~Event() { cleanup(hipStreamSynchronize(stream)); cleanup(hipEventDestroy(value)); --owners.events; }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct CopyCompletion {
    hipStream_t stream;
    bool finished = false;
    ~CopyCompletion() { if (!finished) cleanup(hipStreamSynchronize(stream)); }
    void wait() { check(hipStreamSynchronize(stream)); finished = true; }
};

// Exact payload capacity, explicit owned qNaN redzones. Weight expectation is a
// BORROW of the one original host payload, not a backup of the 521472000B head.
// Readonly weights are compared in bounded 4MiB readback chunks EVERY validation.
// All async copies drain before host storage changes; all frees drain the stream,
// including constructor failure/exception unwinding. Stream outlives its users.
struct Buffer {
    hipStream_t stream;
    std::size_t bytes, prefix, suffix;
    void* allocation = nullptr;
    std::vector<std::byte> expected;
    std::span<const std::byte> borrowed;
    std::array<std::byte, 16> guard{};
    Buffer(hipStream_t s, std::size_t n, std::size_t front, std::size_t back,
           std::span<const std::byte> original = {})
        : stream(s), bytes(n), prefix(front), suffix(back), borrowed(original) {
        require(n > 0 && front <= guard.size() && back <= guard.size(), "buffer geometry");
        for (std::size_t i = 0; i < guard.size(); ++i) guard[i] = i % 2 ? std::byte{0x7e} : std::byte{0};
        if (borrowed.empty()) {
            expected.assign(prefix + bytes + suffix, std::byte{0xa5});
            std::copy_n(guard.begin(), prefix, expected.begin());
            std::copy_n(guard.begin(), suffix, expected.begin() + prefix + bytes);
        } else require(borrowed.size() == bytes, "borrowed weight exact size");
        check(hipMalloc(&allocation, prefix + bytes + suffix)); ++owners.buffers;
        try {
            if (borrowed.empty()) upload();
            else {
                check(hipMemcpyAsync(allocation, guard.data(), prefix, hipMemcpyHostToDevice, stream));
                check(hipMemcpyAsync(as<void>(), borrowed.data(), bytes, hipMemcpyHostToDevice, stream));
                check(hipMemcpyAsync(as<std::byte>(bytes), guard.data(), suffix, hipMemcpyHostToDevice, stream));
                check(hipStreamSynchronize(stream));
            }
        } catch (...) {
            cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation));
            allocation = nullptr; --owners.buffers; throw;
        }
    }
    ~Buffer() {
        if (allocation) { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation)); --owners.buffers; }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as(std::size_t offset = 0) const {
        return reinterpret_cast<T*>(static_cast<std::byte*>(allocation) + prefix + offset);
    }
    void copy(void* destination, const void* source, std::size_t count, hipMemcpyKind kind) const {
        // This guard unwinds BEFORE the caller's temporary host storage. A copy
        // or synchronization error must not leave a DMA targeting freed memory.
        CopyCompletion completion{stream};
        check(hipMemcpyAsync(destination, source, count, kind, stream)); completion.wait();
    }
    void upload() {
        require(borrowed.empty(), "cannot overwrite borrowed weights");
        copy(allocation, expected.data(), expected.size(), hipMemcpyHostToDevice);
    }
    void store(std::span<const std::byte> source, std::size_t offset = 0) {
        require(borrowed.empty() && offset <= bytes && source.size() <= bytes - offset, "buffer store bounds");
        std::copy(source.begin(), source.end(), expected.begin() + prefix + offset);
        copy(as<std::byte>(offset), source.data(), source.size(), hipMemcpyHostToDevice);
    }
    void expect(std::span<const std::byte> source) {
        require(borrowed.empty() && source.size() == bytes, "buffer expectation bounds");
        std::copy(source.begin(), source.end(), expected.begin() + prefix);
    }
    std::vector<std::byte> read() const {
        require(borrowed.empty(), "full weight backup forbidden");
        std::vector<std::byte> got(expected.size());
        copy(got.data(), allocation, got.size(), hipMemcpyDeviceToHost); return got;
    }
    void canaries(std::span<const std::byte> got) const {
        require(got.size() == prefix + bytes + suffix &&
            std::equal(got.begin(), got.begin() + prefix, guard.begin()) &&
            std::equal(got.begin() + prefix + bytes, got.end(), guard.begin()), "output redzone changed");
    }
    void readonly() const {
        if (borrowed.empty()) { require(read() == expected, "readonly payload/redzone changed"); return; }
        std::array<std::byte, 16> front{}, back{};
        copy(front.data(), allocation, prefix, hipMemcpyDeviceToHost);
        copy(back.data(), as<std::byte>(bytes), suffix, hipMemcpyDeviceToHost);
        require(std::equal(front.begin(), front.begin() + prefix, guard.begin()) &&
            std::equal(back.begin(), back.begin() + suffix, guard.begin()), "weight redzone changed");
        std::vector<std::byte> chunk(std::min(bytes, std::size_t{4} << 20));
        for (std::size_t at = 0; at < bytes; at += chunk.size()) {
            const auto count = std::min(chunk.size(), bytes - at);
            copy(chunk.data(), as<std::byte>(at), count, hipMemcpyDeviceToHost);
            require(std::memcmp(chunk.data(), borrowed.data() + at, count) == 0, "original weight bytes changed");
        }
    }
};

void json_string(std::string_view value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32) {
            constexpr char hex[] = "0123456789abcdef";
            std::cout << "\\u00" << hex[c / 16] << hex[c % 16];
        } else std::cout << char(c);
    }
    std::cout << '"';
}
template<class T, std::size_t N> void json_array(const std::array<T, N>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < N; ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
void record(std::string_view kind) {
    std::cout << "{\"kind\":"; json_string(kind); std::cout << ",\"protocol\":"; json_string(protocol);
}
std::uint16_t half_bits(const std::byte* p) {
    return std::uint16_t(std::to_integer<unsigned>(p[0]) | (std::to_integer<unsigned>(p[1]) << 8));
}
void half_store(std::byte* p, std::uint16_t h) { p[0] = std::byte(h & 255); p[1] = std::byte(h >> 8); }
struct Error {
    double max_abs = 0, max_bound_ratio = 0;
    std::size_t compared = 0, finite = 0, readonly = 0;
    void compare(float got, float ref) {
        require(std::isfinite(got) && std::isfinite(ref), "nonfinite common-Q8 comparison");
        const double e = std::abs(double(got) - double(ref));
        const double ratio = e / (absolute_gate + relative_gate * std::abs(double(ref)));
        max_abs = std::max(max_abs, e); max_bound_ratio = std::max(max_bound_ratio, ratio);
        if (ratio > 1) std::cerr << "wide got=" << got << " CPU=" << ref << " bound_ratio=" << ratio << '\n';
        require(ratio <= 1, "frozen common-Q8 gate failed");
        ++compared;
    }
    void merge(const Error& e) {
        max_abs = std::max(max_abs, e.max_abs); max_bound_ratio = std::max(max_bound_ratio, e.max_bound_ratio);
        compared += e.compared; finite += e.finite; readonly += e.readonly;
    }
    void emit() const { std::cout << "{\"max_abs\":" << max_abs << ",\"max_bound_ratio\":" << max_bound_ratio << '}'; }
};
#define WIDE_COUNTERS(X) \
    X(grid_cases) X(q4_forward_cases) X(limit_cases) X(manual_cases) X(reuse_cases) X(actual_cases) \
    X(matrices) X(quantization) X(packing) X(poison_cases) X(poison_subblocks) \
    X(finite_elements) X(reference_elements) X(reference_slices) X(readonly_validations) \
    X(quantizer_rejects) X(packer_rejects) X(mmq_rejects) X(int8_min_cases) X(int8_max_cases) \
    X(raw_sum_difference_cases) X(signed_scale_cases) X(subnormal_scale_cases) X(extreme_scale_cases) \
    X(q6_subscale_cases) X(q6_activation_scale_cases) X(q6_integer_grouping_cases)
struct Counters {
#define DECLARE(name) std::size_t name = 0;
    WIDE_COUNTERS(DECLARE)
#undef DECLARE
    Error error;
    void verify(const Counters& e) const {
#define VERIFY(name) require(name == e.name, "wide coverage mismatch: " #name);
        WIDE_COUNTERS(VERIFY)
#undef VERIFY
        require(std::isfinite(error.max_abs) && std::isfinite(error.max_bound_ratio) &&
            error.max_bound_ratio <= 1, "nonfinite/error counters");
        require(error.compared == reference_elements && error.finite == finite_elements &&
            error.readonly == readonly_validations, "observed correctness validation counters");
    }
    void emit() const {
        std::cout << '{'; bool first = true;
#define EMIT(name) if (!first) std::cout << ','; first = false; std::cout << '"' << #name << "\":" << name;
        WIDE_COUNTERS(EMIT)
#undef EMIT
        std::cout << '}';
    }
};
struct Matrix {
    qwen::TensorType type = qwen::TensorType::Q5_0;
    int width = 0, rows = 0;
    std::string name;
    std::vector<std::byte> bytes;
    std::size_t payload_reads = 0, finite_halfscales = 0;
    bool head() const { return type == qwen::TensorType::Q6_K && width == 2560 && rows == 248320; }
    std::size_t row_bytes() const {
        const auto l = qwen::type_layout(type); return std::size_t(width / l.block_elements) * l.block_bytes;
    }
    qwen::QMatrix view() const { return {type, width, rows, bytes}; }
};
std::size_t finite_weights(const Matrix& m) {
    const auto l = qwen::type_layout(m.type);
    std::size_t checked = 0;
    require(m.width > 0 && m.width % l.block_elements == 0 && m.rows > 0 &&
        m.bytes.size() == m.row_bytes() * m.rows, "canonical weight byte size");
    for (std::size_t at = 0; at < m.bytes.size(); at += l.block_bytes) {
        require(std::isfinite(qwen::half_to_float(half_bits(m.bytes.data() + at +
            (m.type == qwen::TensorType::Q6_K ? 208 : 0)))), "nonfinite weight halfscale");
        ++checked;
        if (m.type == qwen::TensorType::Q4_1)
            require(std::isfinite(qwen::half_to_float(half_bits(m.bytes.data() + at + 2))), "nonfinite Q4 offset");
    }
    return checked;
}
Matrix synthetic(qwen::TensorType type, int width, int rows, bool corner = false) {
    Matrix m{type, width, rows, corner ? "manual" : "synthetic", {}};
    const auto l = qwen::type_layout(type); m.bytes.resize(m.row_bytes() * rows);
    std::uint32_t rng = 0x90612345;
    for (auto& b : m.bytes) { rng = rng * 1664525 + 1013904223; b = std::byte(rng >> 24); }
    const int blocks = width / l.block_elements;
    for (int r = 0; r < rows; ++r) for (int k = 0; k < blocks; ++k) {
        auto* p = m.bytes.data() + (std::size_t(r) * blocks + k) * l.block_bytes;
        std::uint16_t scale = qwen::float_to_half(std::ldexp(float(k % 4 + 1), -13));
        if ((r + k) % 2) scale |= 0x8000;
        if ((r + k) % 11 == 0) scale = (r % 2 ? 0x8001 : 0x0001);
        if (corner) {
            constexpr std::array<std::uint16_t, 5> extreme{0x7bff, 0xfbff, 0x0001, 0x8001, 0x8400};
            scale = extreme[std::size_t(k) % extreme.size()];
        }
        half_store(p + (type == qwen::TensorType::Q6_K ? 208 : 0), scale);
        if (type == qwen::TensorType::Q4_1) half_store(p + 2, qwen::float_to_half(float(r % 7 - 3) / 4096));
        if (type == qwen::TensorType::Q5_0) {
            std::fill_n(p + 2, 20, std::byte{0});
            for (int j = 0; j < 32; ++j) {
                const unsigned code = j == 0 ? 0 : j == 31 ? 31 : unsigned((j + 7 * k + r) % 32);
                p[6 + j % 16] |= std::byte((code & 15) << (4 * (j / 16)));
                p[2 + j / 8] |= std::byte((code >> 4) << (j % 8));
            }
        }
        if (type == qwen::TensorType::Q8_0) { p[2] = std::byte{0x80}; p[3] = std::byte{0x7f}; }
        if (type == qwen::TensorType::Q6_K)
            for (int s = 0; s < 16; ++s) p[192 + s] = std::byte(std::uint8_t(subscales[s]));
    }
    finite_weights(m); return m;
}

// Validate ALL bindings/geometry/ranges BEFORE reading ANY payload. Read each
// complete actual matrix ONCE, including the complete actual head, never a model
// copy/hash/requantization. CPU sampled rows borrow the original unchanged bytes.
std::array<Matrix, 3> actual_matrices(const qwen::Model& model) {
    constexpr std::array<std::string_view, 3> names{
        "blk.6.ffn_down_shexp.weight", "blk.0.ffn_down_shexp.weight", "output.weight"};
    constexpr std::array type{qwen::TensorType::Q5_0, qwen::TensorType::Q8_0, qwen::TensorType::Q6_K};
    constexpr std::array k{640, 640, 2560}, rows{2560, 2560, 248320};
    constexpr std::array<std::size_t, 3> sizes{1126400, 1740800, 521472000};
    std::array<Matrix, 3> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto& t = model.tensor(names[i]); const auto l = qwen::type_layout(type[i]);
        const std::uint64_t rb = std::uint64_t(k[i] / l.block_elements) * l.block_bytes;
        require(t.type == type[i] && t.rank == 2 && t.dimensions[0] == std::uint64_t(k[i]) &&
            t.dimensions[1] == std::uint64_t(rows[i]) && t.dimensions[2] == 0 && t.dimensions[3] == 0 &&
            t.strides[0] == l.block_bytes && t.strides[1] == rb && t.strides[2] == 0 && t.strides[3] == 0 &&
            t.elements == std::uint64_t(k[i]) * rows[i] && t.byte_size == sizes[i] &&
            t.byte_size == rb * rows[i], "actual wide rank/type/geometry/strides/elements/bytes");
        require(t.relative_offset <= std::numeric_limits<std::uint64_t>::max() - model.data_offset() &&
            t.file_offset == model.data_offset() + t.relative_offset && t.file_offset <= model.file_size() &&
            t.byte_size <= model.file_size() - t.file_offset && t.file_offset % model.alignment() == 0,
            "actual wide file range/alignment");
        result[i] = {type[i], k[i], rows[i], std::string(names[i]), {}};
    }
    for (auto& m : result) {
        m.bytes.resize(m.row_bytes() * m.rows); model.read_tensor(m.name, m.bytes); ++m.payload_reads;
        m.finite_halfscales = finite_weights(m);
    }
    return result;
}
std::vector<float> float_input(int k, int n, int edge = -1) {
    std::vector<float> x(std::size_t(k) * n);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (edge == 0) x[j] = 0;
        else if (edge == 1) x[j] = -0.0f;
        else if (edge == 2) x[j] = j % 32 == 0 ? 15.875f : float(int(j % 13) - 6) * .0625f;
        else if (edge == 3) x[j] = std::ldexp(j % 32 == 0 ? 127.0f : float(int(j % 63) - 31), -24);
        else if (edge == 4) x[j] = j % 2 ? -1.0003f : 1.0003f;
        else if (edge == 5) x[j] = std::sin(float(j)) * .013f;
        else if (edge == 6) x[j] = j % 32 == 0 ? 127.0381f : float(int(j % 11) - 5) * 1.0003f;
        else if (edge == 7) x[j] = j % 2 ? -500.0f : 500.0f;
        else x[j] = .5f * std::sin(float(j) * .137f) + .125f * std::cos(float(j) * .071f);
    }
    return x;
}
std::vector<qwen::Q8MmqBlock> expected_pack(std::span<const qwen::Q8_1> q, int k, int n) {
    require(q.size() == std::size_t(k / 32) * n, "Q8 exact span");
    std::vector<qwen::Q8MmqBlock> ds(std::size_t((k + 127) / 128) * n);
    for (int b = 0; b < k / 32; ++b) for (int c = 0; c < n; ++c) {
        const auto& x = q[std::size_t(c) * (k / 32) + b]; auto& y = ds[std::size_t(b / 4) * n + c];
        y.ds[2 * (b % 4)] = x.d; y.ds[2 * (b % 4) + 1] = x.s;
        std::copy_n(x.qs, 32, y.qs + 32 * (b % 4));
    }
    return ds; // Value initialization proves absent halves/codes are all zero.
}
std::vector<qwen::Q8_1> manual_input(const Matrix& m, int n) {
    std::vector<qwen::Q8_1> q(std::size_t(m.width / 32) * n);
    for (std::size_t b = 0; b < q.size(); ++b) {
        const int kb = int(b % (m.width / 32)); const int group = m.type == qwen::TensorType::Q6_K ? kb / 8 : kb;
        const int sub = m.type == qwen::TensorType::Q6_K ? kb % 8 : 0;
        const auto wd = half_bits(m.bytes.data() + std::size_t(group) * qwen::type_layout(m.type).block_bytes +
            (m.type == qwen::TensorType::Q6_K ? 208 : 0));
        // Inverse extreme operand sizes keep all arithmetic finite. Eight Q6
        // activation block scales are distinct, signed, including subnormals.
        q[b].d = std::uint16_t((wd & 0x7fff) == 0x7bff ? 1 + sub :
            (m.type != qwen::TensorType::Q6_K || sub == 7) ? 0x7bff : 0x7800 + 64 * sub);
        if (kb % 2) q[b].d |= 0x8000;
        q[b].s = std::uint16_t(kb % 2 ? 0x8001 : 0x0002);
        for (int j = 0; j < 32; ++j) q[b].qs[j] = std::int8_t(int((b * 37 + j * 17) % 256) - 128);
        q[b].qs[0] = -128; q[b].qs[31] = 127;
    }
    q.front().s = 0x8000; // DS4 must preserve negative-zero raw-sum bits too.
    return q;
}
void manual_proofs(const Matrix& m, std::span<const qwen::Q8_1> q, int n, Counters& c) {
    bool lo = false, hi = false, raw = false, negative = false, tiny = false, extreme = false;
    for (const auto& x : q) {
        const float d = qwen::half_to_float(x.d), s = qwen::half_to_float(x.s);
        require(std::isfinite(d) && std::isfinite(s), "manual finite Q8 halves");
        int sum = 0; for (auto code : x.qs) { lo |= code == -128; hi |= code == 127; sum += code; }
        raw |= s != d * float(sum); negative |= d < 0; tiny |= (x.d & 0x7fff) > 0 && (x.d & 0x7fff) < 0x400;
    }
    const auto l = qwen::type_layout(m.type);
    for (std::size_t at = 0; at < m.bytes.size(); at += l.block_bytes) {
        const auto h = half_bits(m.bytes.data() + at + (m.type == qwen::TensorType::Q6_K ? 208 : 0));
        extreme |= (h & 0x7fff) == 0x7bff;
    }
    require(lo && hi && raw && negative && tiny && extreme, "manual extrema/raw-sum/scale coverage");
    ++c.int8_min_cases; ++c.int8_max_cases; ++c.raw_sum_difference_cases;
    ++c.signed_scale_cases; ++c.subnormal_scale_cases; ++c.extreme_scale_cases;
    if (m.type != qwen::TensorType::Q6_K) return;
    for (std::size_t at = 0; at < m.bytes.size(); at += 210)
        for (int s = 0; s < 16; ++s)
            require(m.bytes[at + 192 + s] == std::byte(std::uint8_t(subscales[s])), "Q6 distinctive sixteen subscales");
    for (std::size_t b = 0; b < q.size(); b += 8)
        for (int i = 0; i < 8; ++i) for (int j = i + 1; j < 8; ++j)
            require(q[b + i].d != q[b + j].d, "Q6 eight distinct activation block scales");
    // Independent logical element decoder verifies nontrivial signed integer
    // dot*subscale products before exact FP32 conversion (the reference ABI).
    bool nontrivial = false;
    for (int col = 0; col < n; ++col) for (int b = 0; b < m.width / 256; ++b) {
        const auto* p = m.bytes.data() + std::size_t(b) * 210;
        for (int j = 0; j < 256; j += 4) {
            int dot = 0;
            for (int t = 0; t < 4; ++t) {
                const int pos = j + t, half = pos / 128, quarter = (pos % 128) / 32, lane = pos % 32;
                const unsigned low = (std::to_integer<unsigned>(p[half * 64 + (quarter % 2) * 32 + lane]) >> (4 * (quarter / 2))) & 15;
                const unsigned high = (std::to_integer<unsigned>(p[128 + half * 32 + lane]) >> (2 * quarter)) & 3;
                dot += (int(low + 16 * high) - 32) * q[std::size_t(col) * (m.width / 32) + b * 8 + pos / 32].qs[pos % 32];
            }
            const int product = dot * subscales[j / 16];
            require(std::abs(product) <= (1 << 21) && int(float(product)) == product, "Q6 exact integer product grouping");
            nontrivial |= dot != 0 && std::abs(subscales[j / 16]) > 1;
        }
    }
    require(nontrivial, "Q6 integer grouping witness");
    ++c.q6_subscale_cases; ++c.q6_activation_scale_cases; ++c.q6_integer_grouping_cases;
}

struct Inputs {
    hipStream_t stream;
    int width, n;
    std::vector<float> x;
    std::vector<qwen::Q8_1> q;
    std::vector<qwen::Q8MmqBlock> packed;
    Buffer raw, canonical, ds4, flag;
    Inputs(hipStream_t s, int k, int columns, Counters* c,
           std::span<const qwen::Q8_1> manual = {}, int edge = -1)
        : stream(s), width(k), n(columns), x(float_input(k, columns, edge)), q(std::size_t(k / 32) * columns),
          raw(s, x.size() * 4, 4, 4), canonical(s, q.size() * 36, 4, 4),
          ds4(s, std::size_t((k + 127) / 128) * columns * 144, 16, 16), flag(s, sizeof(int), 4, 4) {
        raw.store(std::as_bytes(std::span<const float>(x)));
        const int zero = 0; flag.store(std::as_bytes(std::span(&zero, 1)));
        if (manual.empty()) {
            qwen::quantize_q8(x, q);
            check(qwen::launch_quantize_q8_1(raw.as<float>(), canonical.as<qwen::Q8_1>(), width, n, flag.as<int>(), stream));
            canonical.expect(std::as_bytes(std::span<const qwen::Q8_1>(q)));
            canonical.readonly(); flag.readonly(); raw.readonly(); if (c) ++c->quantization;
        } else {
            require(manual.size() == q.size(), "manual Q8 span"); std::copy(manual.begin(), manual.end(), q.begin());
            canonical.store(std::as_bytes(std::span<const qwen::Q8_1>(q)));
        }
        packed = expected_pack(q, width, n);
        check(qwen::launch_q8_1_to_mmq(canonical.as<qwen::Q8_1>(), width, n, ds4.as<qwen::Q8MmqBlock>(), stream));
        ds4.expect(std::as_bytes(std::span<const qwen::Q8MmqBlock>(packed))); readonly(); if (c) ++c->packing;
    }
    void readonly() const { raw.readonly(); canonical.readonly(); ds4.readonly(); flag.readonly(); }
    void poison(Counters& c) {
        const int used = (width / 32) % 4; if (!used) return;
        for (int col = 0; col < n; ++col) {
            const std::size_t b = std::size_t((width / 32) / 4) * n + col;
            for (int sub = used; sub < 4; ++sub) {
                auto& p = packed[b]; p.ds[2 * sub] = 0x7e11; p.ds[2 * sub + 1] = 0xfe55;
                for (int j = 0; j < 32; ++j) p.qs[32 * sub + j] = j % 2 ? 127 : -128;
                ds4.store(std::as_bytes(std::span<const std::uint16_t>(p.ds + 2 * sub, 2)), b * 144 + 4 * sub);
                ds4.store(std::as_bytes(std::span<const std::int8_t>(p.qs + 32 * sub, 32)), b * 144 + 16 + 32 * sub);
                ++c.poison_subblocks;
            }
        }
        // Prove every active half/code is unchanged against ORIGINAL Q8 bytes.
        for (int b = 0; b < width / 32; ++b) for (int col = 0; col < n; ++col) {
            const auto& p = packed[std::size_t(b / 4) * n + col]; const auto& a = q[std::size_t(col) * (width / 32) + b];
            require(p.ds[2 * (b % 4)] == a.d && p.ds[2 * (b % 4) + 1] == a.s &&
                std::memcmp(p.qs + 32 * (b % 4), a.qs, 32) == 0, "poison changed active Q8 subblock");
        }
        readonly(); ++c.poison_cases;
    }
};
struct Case {
    const Matrix& matrix;
    const Buffer& weights;
    hipStream_t stream;
    int n, reference_rows;
    Inputs input;
    Buffer output;
    std::vector<float> reference;
    Case(const Matrix& m, const Buffer& w, hipStream_t s, int columns, Counters* c,
         std::span<const qwen::Q8_1> manual = {})
        : matrix(m), weights(w), stream(s), n(columns), reference_rows(m.head() ? int(head_rows.size()) : m.rows),
          input(s, m.width, columns, c, manual), output(s, std::size_t(m.rows) * columns * 4, 4, 4),
          reference(std::size_t(reference_rows) * columns) {
        require(reinterpret_cast<std::uintptr_t>(weights.as<void>()) % 4 == 2 &&
            reinterpret_cast<std::uintptr_t>(input.ds4.as<void>()) % 16 == 0, "alignment2 weights/alignment16 DS4");
        for (int col = 0; col < n; col += 3) {
            const int count = std::min(3, n - col);
            const auto q = std::span<const qwen::Q8_1>(input.q).subspan(std::size_t(col) * (m.width / 32), std::size_t(count) * (m.width / 32));
            if (!m.head()) {
                qwen::matmul_q8_reference(m.view(), q, count, std::span<float>(reference).subspan(std::size_t(col) * m.rows, std::size_t(count) * m.rows));
                if (c) ++c->reference_slices;
            } else for (std::size_t r = 0; r < head_rows.size(); ++r) {
                const qwen::QMatrix row{m.type, m.width, 1, std::span<const std::byte>(m.bytes).subspan(std::size_t(head_rows[r]) * m.row_bytes(), m.row_bytes())};
                std::array<float, 3> result{}; qwen::matmul_q8_reference(row, q, count, std::span<float>(result).first(count));
                for (int j = 0; j < count; ++j) reference[std::size_t(col + j) * reference_rows + r] = result[j];
                if (c) ++c->reference_slices;
            }
        }
    }
    qwen::QuantizedDeviceMatrix device_matrix() const { return {matrix.type, weights.as<void>(), matrix.width, matrix.rows}; }
    void reset() {
        constexpr std::uint32_t nan = 0x7fc12345;
        for (std::size_t at = 0; at < output.bytes; at += 4) std::memcpy(output.expected.data() + output.prefix + at, &nan, 4);
        output.upload();
    }
    void launch(bool wide) const {
        if (wide) check(qwen::launch_mmq_linear(device_matrix(), input.ds4.as<qwen::Q8MmqBlock>(), n, output.as<float>(), stream));
        else for (int col = 0; col < n; col += 3)
            check(qwen::launch_quantized_linear(device_matrix(), input.canonical.as<qwen::Q8_1>() + std::size_t(col) * (matrix.width / 32),
                std::min(3, n - col), output.as<float>() + std::size_t(col) * matrix.rows, stream));
    }
    Error validate() const {
        const auto bytes = output.read(); output.canaries(bytes); Error e;
        auto value = [&](std::size_t i) { float x; std::memcpy(&x, bytes.data() + output.prefix + i * 4, 4); return x; };
        for (std::size_t i = 0; i < std::size_t(matrix.rows) * n; ++i) {
            const float x = value(i); require(std::isfinite(x), "nonfinite/unwritten full GPU output");
            ++e.finite;
            if (!matrix.head()) e.compare(x, reference[i]);
        }
        if (matrix.head()) for (int col = 0; col < n; ++col) for (std::size_t r = 0; r < head_rows.size(); ++r)
            e.compare(value(std::size_t(col) * matrix.rows + head_rows[r]), reference[std::size_t(col) * reference_rows + r]);
        input.readonly(); weights.readonly(); ++e.readonly;
        require(e.finite == std::size_t(matrix.rows) * n && e.compared == reference.size(), "live validation element coverage");
        return e;
    }
    void qualify(Counters& c) {
        input.poison(c); reset(); launch(true); const auto e = validate(); c.error.merge(e);
        ++c.matrices; c.readonly_validations += e.readonly; c.finite_elements += e.finite; c.reference_elements += e.compared;
    }
};

// Invalid numeric addresses are used ONLY with pre-enqueue rejection paths;
// never copied/dereferenced or passed to a valid call. Valid buffers are drained
// and compared byte-for-byte after rejects, then the SAME allocations are reused.
void rejects_and_reuse(hipStream_t s, Counters& c) {
    const auto top = std::numeric_limits<std::uintptr_t>::max();
    for (auto type : types) {
        const int k = type == qwen::TensorType::Q6_K ? 256 : 160;
        const auto m = synthetic(type, k, 65); Buffer w(s, m.bytes.size(), 2, 2, m.bytes); Case t(m, w, s, 3, &c);
        t.reset(); check(hipGetLastError()); const auto good = t.device_matrix();
        auto* x = t.input.raw.as<float>(); auto* q = t.input.canonical.as<qwen::Q8_1>();
        auto* p = t.input.ds4.as<qwen::Q8MmqBlock>(); auto* y = t.output.as<float>(); auto* flag = t.input.flag.as<int>();
        auto reject = [&](hipError_t got, std::size_t& count, hipError_t expected = hipErrorInvalidValue) {
            require(got == expected, "pre-enqueue reject status"); check(hipGetLastError()); ++count;
        };
        if (type == types.front()) {
            auto quant = [&](const float* a, qwen::Q8_1* b, int width, int n, int* f) {
                reject(qwen::launch_quantize_q8_1(a, b, width, n, f, s), c.quantizer_rejects);
            };
            for (int width : {0,-32,31,16416,std::numeric_limits<int>::max()}) quant(x,q,width,3,flag);
            for (int n : {0,-1,16385,std::numeric_limits<int>::max()}) quant(x,q,k,n,flag);
            quant(nullptr,q,k,3,flag); quant(x,nullptr,k,3,flag); quant(x,q,k,3,nullptr);
            quant(t.input.raw.as<float>(2),q,k,3,flag); quant(x,t.input.canonical.as<qwen::Q8_1>(2),k,3,flag); quant(x,q,k,3,t.input.flag.as<int>(2));
            quant(reinterpret_cast<float*>(top-3),q,k,3,flag); quant(x,reinterpret_cast<qwen::Q8_1*>(top-3),k,3,flag); quant(x,q,k,3,reinterpret_cast<int*>(top-3));
            quant(x,reinterpret_cast<qwen::Q8_1*>(x),k,3,flag); quant(x,q,k,3,reinterpret_cast<int*>(x)); quant(x,q,k,3,reinterpret_cast<int*>(q));
            quant(x,reinterpret_cast<qwen::Q8_1*>(x+479),k,3,flag); quant(x,q,k,3,reinterpret_cast<int*>(x+479)); quant(x,q,k,3,reinterpret_cast<int*>(q+14));
            quant(reinterpret_cast<float*>(top-1),q,k,3,flag); quant(x,reinterpret_cast<qwen::Q8_1*>(top-1),k,3,flag); quant(x,q,k,3,reinterpret_cast<int*>(top-1));
            quant(x,q,-1,-1,flag); quant(nullptr,nullptr,k,3,nullptr); quant(x,q,32,0,flag);
            auto pack = [&](const qwen::Q8_1* a, int width, int n, qwen::Q8MmqBlock* b) {
                reject(qwen::launch_q8_1_to_mmq(a,width,n,b,s),c.packer_rejects);
            };
            for (int width : {0,-32,31,16416,std::numeric_limits<int>::max()}) pack(q,width,3,p);
            for (int n : {0,-1,129,std::numeric_limits<int>::max()}) pack(q,k,n,p);
            pack(nullptr,k,3,p); pack(q,k,3,nullptr); pack(t.input.canonical.as<qwen::Q8_1>(2),k,3,p); pack(q,k,3,t.input.ds4.as<qwen::Q8MmqBlock>(8));
            pack(reinterpret_cast<qwen::Q8_1*>(top-3),k,3,p); pack(q,k,3,reinterpret_cast<qwen::Q8MmqBlock*>(top-15));
            pack(q,k,3,reinterpret_cast<qwen::Q8MmqBlock*>(q)); pack(reinterpret_cast<qwen::Q8_1*>(p),k,3,p); pack(reinterpret_cast<qwen::Q8_1*>(p)+1,k,3,p);
            pack(reinterpret_cast<qwen::Q8_1*>(top-1),k,3,p); pack(q,k,3,reinterpret_cast<qwen::Q8MmqBlock*>(top-7));
            pack(nullptr,0,0,nullptr); pack(q,32,129,p); pack(q,33,1,p);
        }
        auto mmq = [&](qwen::QuantizedDeviceMatrix a, const qwen::Q8MmqBlock* b, int n, float* out, hipError_t expected = hipErrorInvalidValue) {
            reject(qwen::launch_mmq_linear(a,b,n,out,s),c.mmq_rejects,expected);
        };
        for (int width : {0,-k,k==256?255:31,k==256?16640:16416,std::numeric_limits<int>::max()}) {
            auto a=good; a.input=width; mmq(a,p,3,y);
        }
        for (int rows : {0,-1,16385,std::numeric_limits<int>::max()}) { auto a=good; a.output=rows; mmq(a,p,3,y); }
        for (int n : {0,-1,129,std::numeric_limits<int>::max()}) mmq(good,p,n,y);
        auto a=good; a.weights=nullptr; mmq(a,p,3,y); a=good; a.weights=w.as<std::byte>(1); mmq(a,p,3,y);
        a=good; a.weights=reinterpret_cast<void*>(top-1); mmq(a,p,3,y);
        mmq(good,nullptr,3,y); mmq(good,p,3,nullptr); mmq(good,t.input.ds4.as<qwen::Q8MmqBlock>(8),3,y); mmq(good,p,3,t.output.as<float>(2));
        mmq(good,reinterpret_cast<qwen::Q8MmqBlock*>(top-15),3,y); mmq(good,p,3,reinterpret_cast<float*>(top-3));
        mmq(good,p,3,reinterpret_cast<float*>(p)); mmq(good,p,3,reinterpret_cast<float*>(w.as<std::byte>(2)));
        mmq(good,p,3,reinterpret_cast<float*>(t.input.ds4.as<std::byte>(t.input.ds4.bytes-4)));
        const auto weight_end = reinterpret_cast<std::uintptr_t>(w.as<std::byte>(w.bytes));
        mmq(good,p,3,reinterpret_cast<float*>((weight_end-4)&~std::uintptr_t{3}));
        for (int rows : {248319,248321}) { a=good; a.input=2560; a.output=rows; mmq(a,p,3,y); }
        a=good; a.input=2560; a.output=248320;
        if (type == qwen::TensorType::Q6_K) {
            for (int width : {2304,2816}) { a.input=width; mmq(a,p,3,y); }
            for (int width : {32,160,640}) { a=good; a.input=width; mmq(a,p,3,y); }
        } else mmq(a,p,3,y);
        if (type == types.front()) for (auto unsupported : {qwen::TensorType::F32,qwen::TensorType::F16,qwen::TensorType::BF16,static_cast<qwen::TensorType>(999)}) {
            a=good; a.type=unsupported; mmq(a,p,3,y,hipErrorNotSupported);
        }
        check(hipStreamSynchronize(s)); t.input.readonly(); w.readonly(); t.output.readonly();
        check(qwen::launch_quantize_q8_1(x,q,k,3,flag,s)); t.input.readonly();
        check(qwen::launch_q8_1_to_mmq(q,k,3,p,s)); t.input.readonly();
        t.qualify(c); ++c.reuse_cases;
    }
}

// Independent combinatorial expectations. These are emitted as EXPECTATIONS,
// compared with live counters; correctness/footer always emit actual counters.
Counters expected_counters() {
    Counters c;
    auto add = [&](int k,int m,int n,bool manual=false,bool head=false) {
        ++c.matrices; ++c.readonly_validations; ++c.packing; if (!manual) ++c.quantization;
        c.finite_elements += std::size_t(m)*n; c.reference_elements += std::size_t(head?24:m)*n;
        c.reference_slices += std::size_t((n+2)/3)*(head?24:1);
        if ((k/32)%4) { ++c.poison_cases; c.poison_subblocks += std::size_t(4-(k/32)%4)*n; }
    };
    for (int format=0;format<3;++format) {
        const auto widths = format==2 ? std::span<const int>(widths256) : std::span<const int>(widths32);
        for (int k:widths) for (int m:row_counts) for (int n:column_counts) { add(k,m,n); ++c.grid_cases; }
    }
    for (int format=0;format<2;++format) for (int k:{160,2560}) for (int m:{65,129}) for (int n:column_counts) {
        add(k,m,n); ++c.q4_forward_cases;
    }
    for (auto type:types) { const int k=type==qwen::TensorType::Q6_K?256:160;
        add(k,65,3); ++c.reuse_cases; add(type==qwen::TensorType::Q6_K?256:32,16384,3); ++c.limit_cases;
    }
    for (int k:{160,160,768}) { add(k,65,3,true); ++c.manual_cases; }
    for (int matrix=0;matrix<3;++matrix) for (int n:benchmark_columns) {
        add(matrix==2?2560:640,matrix==2?248320:2560,n,false,matrix==2); ++c.actual_cases;
    }
    c.quantization+=8; c.packing+=8; c.quantizer_rejects=30; c.packer_rejects=23; c.mmq_rejects=153;
    c.int8_min_cases=c.int8_max_cases=c.raw_sum_difference_cases=c.signed_scale_cases=c.subnormal_scale_cases=c.extreme_scale_cases=3;
    c.q6_subscale_cases=c.q6_activation_scale_cases=c.q6_integer_grouping_cases=1;
    c.error.compared=c.reference_elements; c.error.finite=c.finite_elements; c.error.readonly=c.readonly_validations;
    return c;
}
void qualification(int device, const std::array<Matrix,3>& actual, Counters& c) {
    check(hipSetDevice(device)); Stream s; rejects_and_reuse(s.value,c);
    for (int edge=0;edge<8;++edge) { Inputs input(s.value,256,1,&c,{},edge); }
    for (auto type:{qwen::TensorType::Q5_0,qwen::TensorType::Q8_0,qwen::TensorType::Q6_K}) {
        const auto widths = type==qwen::TensorType::Q6_K ? std::span<const int>(widths256) : std::span<const int>(widths32);
        for (int k:widths) for (int rows:row_counts) {
            const auto m=synthetic(type,k,rows); Buffer w(s.value,m.bytes.size(),2,2,m.bytes);
            for (int n:column_counts) { Case t(m,w,s.value,n,&c); t.qualify(c); ++c.grid_cases; }
        }
    }
    for (auto type:{qwen::TensorType::Q4_0,qwen::TensorType::Q4_1}) for (int k:{160,2560}) for (int rows:{65,129}) {
        const auto m=synthetic(type,k,rows); Buffer w(s.value,m.bytes.size(),2,2,m.bytes);
        for (int n:column_counts) { Case t(m,w,s.value,n,&c); t.qualify(c); ++c.q4_forward_cases; }
    }
    for (auto type:types) {
        const auto m=synthetic(type,type==qwen::TensorType::Q6_K?256:32,16384); Buffer w(s.value,m.bytes.size(),2,2,m.bytes);
        Case t(m,w,s.value,3,&c); t.qualify(c); ++c.limit_cases;
    }
    for (auto type:{qwen::TensorType::Q5_0,qwen::TensorType::Q8_0,qwen::TensorType::Q6_K}) {
        const auto m=synthetic(type,type==qwen::TensorType::Q6_K?768:160,65,true); const auto q=manual_input(m,3);
        manual_proofs(m,q,3,c); Buffer w(s.value,m.bytes.size(),2,2,m.bytes); Case t(m,w,s.value,3,&c,q);
        t.qualify(c); ++c.manual_cases;
    }
    for (const auto& m:actual) {
        Buffer w(s.value,m.bytes.size(),2,2,m.bytes);
        for (int n:benchmark_columns) { Case t(m,w,s.value,n,&c); t.qualify(c); ++c.actual_cases; }
    }
    c.verify(expected_counters()); check(hipStreamSynchronize(s.value));
}

struct Metric {
    std::size_t sequence=0, tensor=0, column=0, phase=0, validated=0, finite_elements=0, reference_elements=0;
    int device=0;
    std::array<double,repeats> ms{};
    double total=0;
    Error warmup,error;
};
struct Totals {
    std::size_t source=0, correctness=0, metrics=0, intervals=0, finite_elements=0, reference_elements=0;
    void verify() const {
        require(source==1 && correctness==2 && metrics==90 && intervals==1800, "wide live protocol counts");
        const std::size_t sum_n=1+3+8+32+128;
        require(finite_elements==std::size_t(2)*3*repeats*sum_n*(2560+2560+248320) &&
            reference_elements==std::size_t(2)*3*repeats*sum_n*(2560+2560+24), "wide live performance validation counts");
        owners.empty();
    }
};
Metric measurement(Case& t,int device,std::size_t tensor,std::size_t column,std::size_t phase,std::size_t sequence,Event& begin,Event& end) {
    Metric r; r.device=device; r.tensor=tensor; r.column=column; r.phase=phase; r.sequence=sequence;
    const bool wide=phase==1; t.reset(); t.launch(wide); r.warmup=t.validate();
    for (int rep=0;rep<repeats;++rep) {
        t.reset(); check(hipEventRecord(begin.value,t.stream)); t.launch(wide);
        check(hipEventRecord(end.value,t.stream)); check(hipEventSynchronize(end.value));
        float ms=0; check(hipEventElapsedTime(&ms,begin.value,end.value));
        require(std::isfinite(ms) && ms>0, "completed individual HIP-event interval"); r.ms[rep]=ms; r.total+=ms;
        const auto e = t.validate(); r.error.merge(e); ++r.validated; r.finite_elements+=e.finite; r.reference_elements+=e.compared;
    }
    require(r.validated==repeats && std::isfinite(r.total) && r.error.compared==r.reference_elements &&
        r.error.finite==r.finite_elements && r.error.readonly==r.validated, "wide live measurement counters"); return r;
}
std::vector<Metric> performance(int device,const std::array<Matrix,3>& actual,std::size_t sequence) {
    check(hipSetDevice(device)); Stream s; Event begin(s.value),end(s.value); std::vector<Metric> results;
    for (std::size_t matrix=0;matrix<actual.size();++matrix) {
        const auto& m=actual[matrix]; Buffer w(s.value,m.bytes.size(),2,2,m.bytes);
        for (std::size_t col=0;col<benchmark_columns.size();++col) {
            Case t(m,w,s.value,benchmark_columns[col],nullptr);
            for (std::size_t phase=0;phase<phases.size();++phase)
                results.push_back(measurement(t,device,matrix,col,phase,sequence++,begin,end));
        }
    }
    require(results.size()==45, "wide per-device metric count"); check(hipStreamSynchronize(s.value)); return results;
}
void source(const char* path,const qwen::Model& model,const std::array<Matrix,3>& actual,const std::array<hipDeviceProp_t,2>& devices) {
    for (const auto& m : actual)
        require(m.payload_reads == 1 && m.finite_halfscales == m.bytes.size() / qwen::type_layout(m.type).block_bytes,
            "live original payload reads/all-halfscale checks");
    record("mmq_wide_source"); std::cout << ",\"revision\":"; json_string(CORE_REVISION);
    std::cout << ",\"dirty\":" << CORE_DIRTY << ",\"model\":"; json_string(path);
    std::cout << ",\"model_variant\":\"qwen38-keep1-Q4_0\",\"architecture\":\"qwen4exp\",\"donor_revision\":\"dcd685463d597d31f5ca759d32c94592a2740fa4\",\"donor_license\":\"MIT\""
        ",\"absolute_gate\":0.0002,\"relative_gate\":0.00002,\"scope\":\"component qualification; no PP-production/full-model/inference speed claim\""
        ",\"oracle\":\"unchanged matmul_q8_reference in ordered <=3-column slices; full synthetic/shared-expert outputs, 24 original head rows ONLY\""
        ",\"baseline\":\"diagnostic_sliced_linear\",\"timing\":{\"clock\":\"individually completed HIP events\",\"repeats\":20,\"phases\":[\"A1\",\"B\",\"A2\"],\"columns\":";
    json_array(benchmark_columns);
    std::cout << ",\"excludes\":[\"allocation\",\"Model\",\"H2D\",\"quantization\",\"DS4_pack\",\"CPU_reference\",\"output_reset\",\"D2H\"],\"validation\":\"all GPU outputs finite/redzones; all applicable CPU references and complete readonly payloads EVERY warmup/repeat\",\"element_counters\":\"sum over 20 validated intervals; excludes warmup\"}"
        ",\"ordering\":\"source; correctness0/1 after device cleanup BEFORE any timing; device,tensor,column,A1/B/A2 metrics after device cleanup; complete after Model/host/device cleanup\""
        ",\"abi\":{\"q8_bytes\":36,\"q8_qs_offset\":4,\"ds4_bytes\":144,\"ds4_alignment\":16,\"ds4_qs_offset\":16,\"layout\":\"[K128][column]\",\"ds_order\":\"d0,s0,d1,s1,d2,s2,d3,s3\",\"sum\":\"original raw-input half sum, not d8*sum(codes)\",\"padding\":\"missing halves/codes zero; unused K128 subblocks alone poisoned\"}"
        ",\"resources\":{\"stream\":\"borrowed explicit nonblocking\",\"exact_payloads\":true,\"weight_alignment\":2,\"weight_pointer_mod4\":2,\"q8_alignment\":4,\"output_alignment\":4,\"redzones_bytes\":{\"weights\":[2,2],\"raw\":[4,4],\"q8\":[4,4],\"ds4\":[16,16],\"flag\":[4,4],\"output\":[4,4]},\"redzone_bits\":\"half qNaN 0x7e00\",\"weight_readonly\":\"full original bytes in <=4MiB D2H chunks, no weight backup\",\"head_host_reads\":" << actual[2].payload_reads << '}'
        << ",\"coverage\":{\"types\":[\"Q4_0\",\"Q4_1\",\"Q5_0\",\"Q8_0\",\"Q6_K\"],\"new_format_cartesian\":true,\"q5_q8_widths\":";
    json_array(widths32); std::cout << ",\"q6_widths\":"; json_array(widths256);
    std::cout << ",\"rows\":"; json_array(row_counts); std::cout << ",\"columns\":"; json_array(column_counts);
    std::cout << ",\"q4_forward\":{\"widths\":[160,2560],\"rows\":[65,129],\"columns\":"; json_array(column_counts);
    std::cout << "},\"limits\":\"all five types M16384 N3; new formats K16384 grid; sole large exception Q6_K[2560,248320]\""
        ",\"manual\":{\"q5_q8\":[160,65,3],\"q6\":[768,65,3],\"checks\":[\"int8 -128/127\",\"rawsum != d8*sumcodes\",\"signed/subnormal/extreme halves\",\"16 distinct Q6 subscales\",\"all 8 distinct Q6 activation scales\",\"integer dot*subscale before FP32\"]}"
        ",\"quantizer_edges\":{\"shape\":[256,1],\"cases\":[\"zero\",\"negative_zero\",\"half_rounding\",\"minimum_half_scale\",\"signed_alternating\",\"small_sine\",\"unrounded_scale_codes\",\"large_cancellation\"]}"
        ",\"rejects\":[\"null\",\"shape\",\"type\",\"alignment\",\"range_wrap\",\"write_alias\",\"exact_head_exception\"],\"reuse\":\"all five formats; same producer/pack/linear allocations after rejection\"}"
        ",\"expected_per_device\":"; expected_counters().emit();
    std::cout << ",\"expected_protocol\":{\"source\":1,\"correctness\":2,\"metrics\":90,\"intervals\":1800,\"complete\":1,\"rows\":94},\"actual\":[";
    for (std::size_t i=0;i<actual.size();++i) {
        if (i) std::cout << ',';
        const auto& m=actual[i]; const auto& t=model.tensor(m.name);
        std::cout << "{\"tensor_index\":" << i << ",\"tensor\":"; json_string(m.name);
        std::cout << ",\"type\":"; json_string(qwen::type_name(m.type));
        std::cout << ",\"rank\":2,\"dimensions\":[" << m.width << ',' << m.rows << "],\"strides_bytes\":[" << t.strides[0] << ',' << t.strides[1]
            << "],\"elements\":" << t.elements << ",\"bytes\":" << m.bytes.size() << ",\"file_offset\":" << t.file_offset
            << ",\"read_api\":\"Model.read_tensor\",\"payload_reads\":" << m.payload_reads << ",\"unchanged\":true,\"reference_coverage\":";
        json_string(m.head()?"sampled_rows":"all_rows"); std::cout << ",\"reference_rows\":" << (m.head()?24:m.rows) << ",\"sample_rows\":";
        if (m.head()) json_array(head_rows); else std::cout << "[]";
        std::cout << ",\"finite_rows\":" << m.rows << ",\"finite_halfscales\":" << m.finite_halfscales << ",\"all_halfscales_finite\":true}";
    }
    std::cout << "],\"devices\":[";
    for (std::size_t i=0;i<devices.size();++i) { if (i) std::cout << ',';
        std::cout << "{\"device\":" << i << ",\"arch\":"; json_string(devices[i].gcnArchName); std::cout << ",\"wave\":" << devices[i].warpSize << '}';
    }
    // Exact required top-level field sets. Nested schema counts/error fields are
    // integers/finite numbers; IDs/paths text; proof flags booleans. No optional
    // fields or extra success records. source.schemas describes this protocol.
    std::cout << "],\"schemas\":{\"mmq_wide_source\":[\"kind\",\"protocol\",\"revision\",\"dirty\",\"model\",\"model_variant\",\"architecture\",\"donor_revision\",\"donor_license\",\"absolute_gate\",\"relative_gate\",\"scope\",\"oracle\",\"baseline\",\"timing\",\"ordering\",\"abi\",\"resources\",\"coverage\",\"expected_per_device\",\"expected_protocol\",\"actual\",\"devices\",\"schemas\"]"
        ",\"mmq_wide_correctness\":[\"kind\",\"protocol\",\"device\",\"counters\",\"error\",\"proofs\",\"cleanup\",\"passed\"]"
        ",\"mmq_wide_metric\":[\"kind\",\"protocol\",\"sequence\",\"device\",\"tensor_index\",\"column_index\",\"phase_index\",\"phase\",\"path\",\"tensor\",\"type\",\"width\",\"rows\",\"columns\",\"weight_bytes\",\"reference_coverage\",\"reference_rows\",\"sample_rows\",\"launches_per_repeat\",\"repeats\",\"validated_intervals\",\"finite_elements\",\"reference_elements\",\"completed_ms\",\"completed_ms_total\",\"resident_ms\",\"warmup_error\",\"error\",\"proofs\",\"cleanup\",\"passed\"]"
        ",\"mmq_wide_complete\":[\"kind\",\"protocol\",\"devices\",\"source_records\",\"correctness_records\",\"metric_records\",\"validated_intervals\",\"finite_elements\",\"reference_elements\",\"jsonl_records\",\"live_owners\",\"cleanup\",\"passed\"]"
        ",\"counters\":[";
    bool first=true;
#define NAME(name) if (!first) std::cout << ','; first=false; json_string(#name);
    WIDE_COUNTERS(NAME)
#undef NAME
    std::cout << "],\"error\":[\"max_abs\",\"max_bound_ratio\"],\"proofs\":[\"q8_bytes_exact\",\"ds4_bytes_exact\",\"finite\",\"redzones\",\"readonly\"]"
        ",\"actual\":[\"tensor_index\",\"tensor\",\"type\",\"rank\",\"dimensions\",\"strides_bytes\",\"elements\",\"bytes\",\"file_offset\",\"read_api\",\"payload_reads\",\"unchanged\",\"reference_coverage\",\"reference_rows\",\"sample_rows\",\"finite_rows\",\"finite_halfscales\",\"all_halfscales_finite\"]"
        ",\"devices\":[\"device\",\"arch\",\"wave\"],\"timing\":[\"clock\",\"repeats\",\"phases\",\"columns\",\"excludes\",\"validation\",\"element_counters\"]"
        ",\"abi\":[\"q8_bytes\",\"q8_qs_offset\",\"ds4_bytes\",\"ds4_alignment\",\"ds4_qs_offset\",\"layout\",\"ds_order\",\"sum\",\"padding\"]"
        ",\"resources\":[\"stream\",\"exact_payloads\",\"weight_alignment\",\"weight_pointer_mod4\",\"q8_alignment\",\"output_alignment\",\"redzones_bytes\",\"redzone_bits\",\"weight_readonly\",\"head_host_reads\"]"
        ",\"redzones_bytes\":[\"weights\",\"raw\",\"q8\",\"ds4\",\"flag\",\"output\"]"
        ",\"coverage\":[\"types\",\"new_format_cartesian\",\"q5_q8_widths\",\"q6_widths\",\"rows\",\"columns\",\"q4_forward\",\"limits\",\"manual\",\"quantizer_edges\",\"rejects\",\"reuse\"]"
        ",\"q4_forward\":[\"widths\",\"rows\",\"columns\"],\"manual\":[\"q5_q8\",\"q6\",\"checks\"],\"quantizer_edges\":[\"shape\",\"cases\"]"
        ",\"expected_protocol\":[\"source\",\"correctness\",\"metrics\",\"intervals\",\"complete\",\"rows\"],\"live_owners\":[\"buffers\",\"streams\",\"events\"]}}\n";
}
void proofs() { std::cout << "{\"q8_bytes_exact\":true,\"ds4_bytes_exact\":true,\"finite\":true,\"redzones\":true,\"readonly\":true}"; }
void correctness(int device,const Counters& c) {
    c.verify(expected_counters()); owners.empty(); record("mmq_wide_correctness");
    std::cout << ",\"device\":" << device << ",\"counters\":"; c.emit(); std::cout << ",\"error\":"; c.error.emit();
    std::cout << ",\"proofs\":"; proofs(); std::cout << ",\"cleanup\":true,\"passed\":true}\n";
}
void metric(const Metric& r,const Matrix& m,Totals& totals) {
    owners.empty(); const int n=benchmark_columns[r.column];
    require(r.sequence==totals.metrics && r.validated==repeats && r.finite_elements==std::size_t(m.rows)*n*repeats &&
        r.reference_elements==std::size_t(m.head()?24:m.rows)*n*repeats, "metric live coordinates/coverage");
    require(r.error.compared==r.reference_elements && r.error.finite==r.finite_elements && r.error.readonly==r.validated &&
        r.warmup.compared==std::size_t(m.head()?24:m.rows)*n && r.warmup.finite==std::size_t(m.rows)*n &&
        r.warmup.readonly==1, "observed metric validation counters");
    record("mmq_wide_metric"); std::cout << ",\"sequence\":" << r.sequence << ",\"device\":" << r.device << ",\"tensor_index\":" << r.tensor
        << ",\"column_index\":" << r.column << ",\"phase_index\":" << r.phase << ",\"phase\":"; json_string(phases[r.phase]);
    std::cout << ",\"path\":"; json_string(r.phase==1?"mmq_wide":"diagnostic_sliced_linear");
    std::cout << ",\"tensor\":"; json_string(m.name); std::cout << ",\"type\":"; json_string(qwen::type_name(m.type));
    std::cout << ",\"width\":" << m.width << ",\"rows\":" << m.rows << ",\"columns\":" << n << ",\"weight_bytes\":" << m.bytes.size()
        << ",\"reference_coverage\":"; json_string(m.head()?"sampled_rows":"all_rows");
    std::cout << ",\"reference_rows\":" << (m.head()?24:m.rows) << ",\"sample_rows\":";
    if (m.head()) json_array(head_rows); else std::cout << "[]";
    std::cout << ",\"launches_per_repeat\":" << (r.phase==1?1:(n+2)/3) << ",\"repeats\":20,\"validated_intervals\":" << r.validated
        << ",\"finite_elements\":" << r.finite_elements << ",\"reference_elements\":" << r.reference_elements << ",\"completed_ms\":"; json_array(r.ms);
    std::cout << ",\"completed_ms_total\":" << r.total << ",\"resident_ms\":" << r.total/repeats << ",\"warmup_error\":"; r.warmup.emit();
    std::cout << ",\"error\":"; r.error.emit(); std::cout << ",\"proofs\":"; proofs(); std::cout << ",\"cleanup\":true,\"passed\":true}\n";
    ++totals.metrics; totals.intervals+=r.validated; totals.finite_elements+=r.finite_elements; totals.reference_elements+=r.reference_elements;
}
void run(const char* path,Totals& totals) {
    const std::string_view revision=CORE_REVISION;
    require(revision.size()==40 && std::all_of(revision.begin(),revision.end(),[](char c) { return (c>='0'&&c<='9')||(c>='a'&&c<='f'); }), "CORE_REVISION must be 40 lowercase hex");
    require(CORE_DIRTY==0 || CORE_DIRTY==1, "CORE_DIRTY must be integer 0/1");
    require(std::fegetround()==FE_TONEAREST, "common-Q8 oracle requires round-to-nearest");
    int count=0; check(hipGetDeviceCount(&count)); require(count==2, "requires exactly GPU0/GPU1");
    std::array<hipDeviceProp_t,2> devices{};
    for (int d=0;d<2;++d) {
        check(hipGetDeviceProperties(&devices[d],d)); const std::string_view arch=devices[d].gcnArchName;
        require(devices[d].warpSize==64 && (arch=="gfx906" || arch.starts_with("gfx906:")), "requires exact gfx906 wave64");
    }
    qwen::Model model(path); require(model.metadata_value("general.architecture").get<std::string>()=="qwen4exp", "model architecture");
    const auto actual=actual_matrices(model); source(path,model,actual,devices); ++totals.source;
    for (int d=0;d<2;++d) { Counters c; qualification(d,actual,c); correctness(d,c); ++totals.correctness; }
    require(totals.correctness==2, "both devices must qualify before performance");
    for (int d=0;d<2;++d) {
        const auto results=performance(d,actual,totals.metrics); owners.empty();
        for (const auto& r:results) metric(r,actual[r.tensor],totals);
    }
    totals.verify();
}
} // namespace

int main(int argc,char** argv) {
    try {
        require(argc==2, "usage: core-mmq-wide MODEL.gguf"); std::cout << std::setprecision(17);
        Totals totals; run(argv[1],totals); // All Model/host/device owners destroyed BEFORE footer.
        totals.verify(); record("mmq_wide_complete");
        std::cout << ",\"devices\":2,\"source_records\":" << totals.source << ",\"correctness_records\":" << totals.correctness
            << ",\"metric_records\":" << totals.metrics << ",\"validated_intervals\":" << totals.intervals
            << ",\"finite_elements\":" << totals.finite_elements << ",\"reference_elements\":" << totals.reference_elements
            << ",\"jsonl_records\":" << totals.source+totals.correctness+totals.metrics+1
            << ",\"live_owners\":{\"buffers\":" << owners.buffers << ",\"streams\":" << owners.streams << ",\"events\":" << owners.events
            << "},\"cleanup\":true,\"passed\":true}\n";
        std::cout.flush(); require(bool(std::cout), "wide JSONL stdout failure"); return 0;
    } catch (const std::exception& e) { std::cerr << "core-mmq-wide: " << e.what() << '\n'; return 1; }
}
