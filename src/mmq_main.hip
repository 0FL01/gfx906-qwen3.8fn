#include "hip/mmq.cuh"
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
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// core-mmq MODEL.gguf; link qwen-mmq-gpu, qwen-linear-gpu, qwen-core.
// Component qualification ONLY: unchanged common-Q8 quantization/DS4 packing
// and canonical Q4 MMQ. The sliced N<=3 linear path is a diagnostic timing
// baseline, NOT a PP production implementation or an inference-speed claim.
// Build without fast-math, with -ffp-contract=off, for gfx906 wave64.
// JSONL v1 is deliberately fixed: source, correctness device0/device1, then
// device/tensor/columns/phase ordered measurements, then complete AFTER cleanup.
// Failure goes to stderr/exit1 and never produces a success footer.
namespace {
constexpr double absolute_gate = 2e-4, relative_gate = 2e-5;
constexpr int repeats = 20;
constexpr std::array widths{32, 160, 640, 2560, 10240, 16384};
constexpr std::array row_counts{1, 7, 63, 64, 65, 67};
constexpr std::array column_counts{1, 2, 3, 4, 8, 9, 16, 17, 32, 33, 64, 65, 128};
constexpr std::array benchmark_columns{1, 3, 8, 32, 128};
constexpr std::array types{qwen::TensorType::Q4_0, qwen::TensorType::Q4_1};
constexpr std::array<std::string_view, 3> phases{"A1", "B", "A2"};
constexpr std::size_t grid_cases = 936, actual_cases = 39, corner_cases = 3;
constexpr std::size_t matrix_cases = 979, quantization_cases = 984, packing_cases = 987;
constexpr std::size_t poison_cases = 316, poisoned_subblocks = 27540;
constexpr std::size_t compared_elements = 2568652, reference_slices = 9904;
constexpr std::size_t quantizer_rejects = 30, packer_rejects = 23, mmq_rejects = 36;
constexpr std::size_t host_rejects = quantizer_rejects + packer_rejects + mmq_rejects;
constexpr std::size_t measurements_per_device = 45, measurement_records = 90;
constexpr std::size_t jsonl_records = 94;
static_assert(sizeof(qwen::Q8_1) == 36 && offsetof(qwen::Q8_1, qs) == 4);
static_assert(sizeof(qwen::Q8MmqBlock) == 144 && alignof(qwen::Q8MmqBlock) == 16 &&
              offsetof(qwen::Q8MmqBlock, ds) == 0 && offsetof(qwen::Q8MmqBlock, qs) == 16);
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little);
static_assert(types.size() * widths.size() * row_counts.size() * column_counts.size() == grid_cases);

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) { std::cerr << "mmq cleanup: " << hipGetErrorString(status) << '\n'; std::abort(); }
}
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() {
        if (value) { cleanup(hipStreamSynchronize(value)); cleanup(hipStreamDestroy(value)); }
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Event {
    hipStream_t stream;
    hipEvent_t value = nullptr;
    explicit Event(hipStream_t s) : stream(s) { check(hipEventCreate(&value)); }
    ~Event() {
        if (value) { cleanup(hipStreamSynchronize(stream)); cleanup(hipEventDestroy(value)); }
    }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

// Exact payload capacities plus explicitly owned poisoned redzones, not padded
// rows/columns/K or a larger workspace hiding an invalid access. All async host
// copies complete inside the copy method. Every free drains its owning stream,
// including constructor failure and exception unwinding.
struct Buffer {
    hipStream_t stream;
    std::size_t prefix, bytes, suffix;
    std::vector<std::byte> expected;
    void* allocation = nullptr;
    Buffer(hipStream_t s, std::size_t n, std::size_t front, std::size_t back)
        : stream(s), prefix(front), bytes(n), suffix(back), expected(front + n + back, std::byte{0xa5}) {
        // Alternating half qNaN bits in the redzones (LE 0x7e00).
        for (std::size_t i = 0; i < prefix; ++i) expected[i] = i % 2 ? std::byte{0x7e} : std::byte{0};
        for (std::size_t i = 0; i < suffix; ++i) expected[prefix + bytes + i] = i % 2 ? std::byte{0x7e} : std::byte{0};
        check(hipMalloc(&allocation, expected.size()));
        try { upload_all(); }
        catch (...) { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation)); allocation = nullptr; throw; }
    }
    ~Buffer() {
        if (allocation) { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation)); }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as(std::size_t offset = 0) const {
        return reinterpret_cast<T*>(static_cast<std::byte*>(allocation) + prefix + offset);
    }
    void upload_all() {
        check(hipMemcpyAsync(allocation, expected.data(), expected.size(), hipMemcpyHostToDevice, stream));
        check(hipStreamSynchronize(stream));
    }
    void store(std::span<const std::byte> source, std::size_t offset = 0) {
        require(offset <= bytes && source.size() <= bytes - offset, "buffer store bounds");
        std::copy(source.begin(), source.end(), expected.begin() + prefix + offset);
        check(hipMemcpyAsync(as<std::byte>(offset), source.data(), source.size(), hipMemcpyHostToDevice, stream));
        check(hipStreamSynchronize(stream));
    }
    void expect(std::span<const std::byte> source) {
        require(source.size() == bytes, "buffer expectation bounds");
        std::copy(source.begin(), source.end(), expected.begin() + prefix);
    }
    std::vector<std::byte> read() const {
        std::vector<std::byte> result(expected.size());
        check(hipMemcpyAsync(result.data(), allocation, result.size(), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream)); return result;
    }
    void readonly() const { require(read() == expected, "readonly payload or poisoned redzone changed"); }
    void canaries(std::span<const std::byte> got) const {
        require(got.size() == expected.size() &&
            std::equal(got.begin(), got.begin() + prefix, expected.begin()) &&
            std::equal(got.begin() + prefix + bytes, got.end(), expected.begin() + prefix + bytes),
            "output redzone changed (padding store)");
    }
};

// Original fixed-string JSON helper; no std::quoted/unqualified ADL call.
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
template<class T, std::size_t N> void json_ints(const std::array<T, N>& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < N; ++i) { if (i) std::cout << ','; std::cout << values[i]; }
    std::cout << ']';
}
std::uint16_t half_bits(const std::byte* p) {
    return std::uint16_t(std::to_integer<unsigned>(p[0]) | (std::to_integer<unsigned>(p[1]) << 8));
}
void half_store(std::byte* p, std::uint16_t bits) {
    p[0] = std::byte(bits & 255); p[1] = std::byte(bits >> 8);
}
int tile_j(int rows, int n) { return qwen::mmq_q4_tile_j(rows, n); }
struct Error {
    double max_abs = 0, max_bound_ratio = 0;
    void compare(std::span<const float> got, std::span<const float> reference) {
        require(got.size() == reference.size(), "full common-Q8 comparison size");
        for (std::size_t i = 0; i < got.size(); ++i) {
            require(std::isfinite(got[i]) && std::isfinite(reference[i]), "nonfinite full-output comparison");
            const double e = std::abs(double(got[i]) - double(reference[i]));
            const double ratio = e / (absolute_gate + relative_gate * std::abs(double(reference[i])));
            max_abs = std::max(max_abs, e); max_bound_ratio = std::max(max_bound_ratio, ratio);
            if (ratio > 1) std::cerr << "mmq element=" << i << " got=" << got[i]
                << " reference=" << reference[i] << " error=" << e << " bound_ratio=" << ratio << '\n';
            require(ratio <= 1, "frozen common-Q8 linear gate failed");
        }
    }
    void merge(const Error& other) {
        max_abs = std::max(max_abs, other.max_abs);
        max_bound_ratio = std::max(max_bound_ratio, other.max_bound_ratio);
    }
};
struct Counters {
    std::size_t quantization = 0, packing = 0, matrices = 0, poison = 0, poison_blocks = 0;
    std::size_t elements = 0, slices = 0, rejects_quantizer = 0, rejects_packer = 0, rejects_mmq = 0;
    std::size_t manual = 0, int8_min = 0, raw_sum = 0, signed_scale = 0, half_pairs = 0;
    Error error;
    void verify() const {
        require(quantization == quantization_cases && packing == packing_cases && matrices == matrix_cases &&
            poison == poison_cases && poison_blocks == poisoned_subblocks && elements == compared_elements &&
            slices == reference_slices && rejects_quantizer == quantizer_rejects && rejects_packer == packer_rejects &&
            rejects_mmq == mmq_rejects && manual == 3 && int8_min == 3 && raw_sum == 3 && signed_scale == 3 &&
            half_pairs == 21, "frozen MMQ qualification coverage mismatch");
    }
};
struct Matrix {
    qwen::TensorType type = qwen::TensorType::Q4_0;
    int width = 0, rows = 0;
    std::string name;
    std::vector<std::byte> bytes;
    qwen::QMatrix view() const { return {type, width, rows, bytes}; }
};
void finite_weights(const Matrix& matrix) {
    const std::size_t block_bytes = matrix.type == qwen::TensorType::Q4_0 ? 18 : 20;
    require(matrix.bytes.size() == std::size_t(matrix.width / 32) * matrix.rows * block_bytes,
        "canonical weight exact byte capacity");
    for (std::size_t b = 0; b < matrix.bytes.size(); b += block_bytes) {
        require(std::isfinite(qwen::half_to_float(half_bits(matrix.bytes.data() + b))), "nonfinite static weight scale");
        if (matrix.type == qwen::TensorType::Q4_1)
            require(std::isfinite(qwen::half_to_float(half_bits(matrix.bytes.data() + b + 2))), "nonfinite static weight offset");
    }
}
Matrix synthetic(qwen::TensorType type, int width, int rows) {
    const std::size_t block_bytes = type == qwen::TensorType::Q4_0 ? 18 : 20;
    Matrix result{type, width, rows, "synthetic", std::vector<std::byte>(std::size_t(rows) * (width / 32) * block_bytes)};
    std::uint32_t random = 0x9e3779b9;
    for (auto& b : result.bytes) { random = random * 1664525 + 1013904223; b = std::byte(random >> 24); }
    for (std::size_t b = 0; b < result.bytes.size() / block_bytes; ++b) {
        auto* p = result.bytes.data() + b * block_bytes;
        const float scale = std::ldexp(float(b % 4 + 1), -12);
        half_store(p, qwen::float_to_half(b % 2 ? -scale : scale));
        if (type == qwen::TensorType::Q4_1) half_store(p + 2, qwen::float_to_half(float(int(b % 9) - 4) / 512));
    }
    finite_weights(result); return result;
}
std::vector<float> float_input(int width, int columns, int edge = -1) {
    std::vector<float> x(std::size_t(width) * columns);
    for (std::size_t j = 0; j < x.size(); ++j) {
        if (edge == 0) x[j] = 0;
        else if (edge == 1) x[j] = -0.0f;
        else if (edge == 2) x[j] = j % 32 == 0 ? 15.875f : float(int(j % 13) - 6) * .0625f;
        else if (edge == 3) x[j] = std::ldexp(j % 32 == 0 ? 127.0f : float(int(j % 63) - 31), -24);
        else if (edge == 4) x[j] = j % 2 ? -1.0003f : 1.0003f;
        else if (edge == 5) x[j] = std::sin(float(j)) * .013f;
        else if (edge == 6) x[j] = j % 32 == 0 ? 127.0381f : float(int(j % 11) - 5) * 1.0003f;
        else if (edge == 7) x[j] = j % 2 ? -500.0f : 500.0f;
        else x[j] = .7f * std::sin(float(j) * .137f) + .2f * std::cos(float(j) * .071f);
    }
    return x;
}

// Independent byte rearrangement expectation, not a copied GPU oracle kernel.
// Padding begins from value-initialized zeros; every original half/code is
// copied individually to [K128][column], without re-quantization/recomputed sum.
std::vector<qwen::Q8MmqBlock> expected_pack(std::span<const qwen::Q8_1> q, int width, int n) {
    const int blocks32 = width / 32, blocks128 = (blocks32 + 3) / 4;
    require(q.size() == std::size_t(blocks32) * n, "DS4 source exact capacity");
    std::vector<qwen::Q8MmqBlock> packed(std::size_t(blocks128) * n);
    for (int k = 0; k < blocks32; ++k) for (int c = 0; c < n; ++c) {
        const auto& source = q[std::size_t(c) * blocks32 + k];
        auto& target = packed[std::size_t(k / 4) * n + c];
        target.ds[2 * (k % 4)] = source.d; target.ds[2 * (k % 4) + 1] = source.s;
        std::copy_n(source.qs, 32, target.qs + 32 * (k % 4));
    }
    return packed;
}
std::vector<qwen::Q8_1> manual_input(int width, int n, bool half_corner = false) {
    std::vector<qwen::Q8_1> q(std::size_t(width / 32) * n);
    for (std::size_t b = 0; b < q.size(); ++b) {
        q[b].d = qwen::float_to_half((b % 2 ? -1.0f : 1.0f) * float(b % 7 + 1) / 512);
        q[b].s = qwen::float_to_half(float(int(b % 5) - 2) / 8);
        for (int j = 0; j < 32; ++j) q[b].qs[j] = std::int8_t(int((b * 37 + j * 17) % 256) - 128);
        q[b].qs[0] = -128;
    }
    if (half_corner) {
        require(width == 32 && n == 3, "half-RNE corner geometry");
        q[0].d = 0x3c01; q[0].s = 0x3555;
        q[1].d = 0x3800; q[1].s = 0x3800;
        q[2].d = 0xbc01; q[2].s = 0x8000; // preserve negative-zero raw-sum half
    } else {
        q.front().s = 0x8000; // packing preserves half bits, including signed zero
    }
    return q;
}
Matrix half_corner_matrix() {
    auto matrix = synthetic(qwen::TensorType::Q4_1, 32, 7); matrix.name = "half_RNE_products";
    // Normal rounding, signed scale, subnormal products, exact halfway-to-zero,
    // and halfway-to-even products. All operands and rounded products finite.
    constexpr std::array<std::uint16_t, 7> d{0x3c01, 0x3555, 0x0400, 0x0001, 0xbc01, 0x3bff, 0x4001};
    constexpr std::array<std::uint16_t, 7> m{0x3555, 0x3c01, 0x0001, 0x0400, 0xb555, 0x3bff, 0xc001};
    for (std::size_t r = 0; r < d.size(); ++r) {
        half_store(matrix.bytes.data() + r * 20, d[r]); half_store(matrix.bytes.data() + r * 20 + 2, m[r]);
    }
    finite_weights(matrix); return matrix;
}
void manual_proofs(std::span<const qwen::Q8_1> q, Counters& counters) {
    bool minimum = false, raw_differs = false, signed_scale = false;
    for (const auto& block : q) {
        require(std::isfinite(qwen::half_to_float(block.d)) && std::isfinite(qwen::half_to_float(block.s)), "manual Q8 numeric contract");
        int sum = 0;
        for (auto code : block.qs) { minimum |= code == -128; sum += int(code); }
        raw_differs |= qwen::half_to_float(block.s) != qwen::half_to_float(block.d) * float(sum);
        signed_scale |= qwen::half_to_float(block.d) < 0;
    }
    require(minimum && raw_differs && signed_scale, "manual signed-scale/raw-sum/int8 minimum proof");
    ++counters.manual; ++counters.int8_min; ++counters.raw_sum; ++counters.signed_scale;
}

// Only the three requested unchanged payloads are read. Validate ALL bindings
// before ANY payload read: rank, every dimension/stride, elements, byte_size and
// file range. Expert0 is one read_expert by name, HC is one read_slice by name.
std::array<Matrix, 3> actual_matrices(const qwen::Model& model) {
    constexpr std::array<std::string_view, 3> names{
        "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.hc_attn_down.weight"};
    constexpr std::array k{2560, 640, 10240}, rows{640, 2560, 320};
    constexpr std::array type{qwen::TensorType::Q4_0, qwen::TensorType::Q4_1, qwen::TensorType::Q4_0};
    std::array<Matrix, 3> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
        const auto& t = model.tensor(names[i]); const std::uint64_t bb = i == 1 ? 20 : 18;
        const std::uint64_t row_bytes = std::uint64_t(k[i] / 32) * bb;
        const std::uint64_t payload = row_bytes * rows[i], experts = i < 2 ? 512 : 1;
        require(t.type == type[i] && t.rank == (i < 2 ? 3u : 2u) && t.dimensions[0] == std::uint64_t(k[i]) &&
            t.dimensions[1] == std::uint64_t(rows[i]) && t.dimensions[2] == (i < 2 ? 512u : 0u) &&
            t.dimensions[3] == 0 && t.strides[0] == bb && t.strides[1] == row_bytes &&
            t.strides[2] == (i < 2 ? payload : 0) && t.strides[3] == 0 &&
            t.elements == std::uint64_t(k[i]) * rows[i] * experts && t.byte_size == payload * experts,
            "actual MMQ rank/dimensions/type/strides/elements/byte_size binding");
        require(t.relative_offset <= std::numeric_limits<std::uint64_t>::max() - model.data_offset() &&
            t.file_offset == model.data_offset() + t.relative_offset && t.file_offset <= model.file_size() &&
            t.byte_size <= model.file_size() - t.file_offset && t.file_offset % model.alignment() == 0,
            "actual MMQ file byte range/alignment");
        result[i] = {type[i], k[i], rows[i], std::string(names[i]), std::vector<std::byte>(std::size_t(payload))};
    }
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (i < 2) model.read_expert(result[i].name, 0, result[i].bytes);
        else model.read_slice(result[i].name, 0, result[i].bytes);
        finite_weights(result[i]);
    }
    return result;
}

struct Inputs {
    hipStream_t stream;
    int width, n;
    std::vector<float> x;
    std::vector<qwen::Q8_1> q;
    std::vector<qwen::Q8MmqBlock> packed;
    Buffer raw, canonical, ds4, flag;
    Inputs(hipStream_t s, int k, int columns, Counters* counters,
           std::span<const qwen::Q8_1> manual = {}, int edge = -1)
        : stream(s), width(k), n(columns), x(float_input(k, columns, edge)),
          q(std::size_t(k / 32) * columns),
          raw(s, x.size() * sizeof(float), 4, 4), canonical(s, q.size() * sizeof(qwen::Q8_1), 4, 4),
          ds4(s, std::size_t((k + 127) / 128) * columns * sizeof(qwen::Q8MmqBlock), 16, 16),
          flag(s, sizeof(int), 4, 4) {
        raw.store(std::as_bytes(std::span<const float>(x)));
        const int zero = 0; flag.store(std::as_bytes(std::span(&zero, 1)));
        if (manual.empty()) {
            // The existing CPU quantizer and its 36B ABI remain the producer
            // oracle. Check GPU bytes/flag BEFORE consuming them in the packer.
            qwen::quantize_q8(x, q);
            check(qwen::launch_quantize_q8_1(raw.as<float>(), canonical.as<qwen::Q8_1>(), width, n, flag.as<int>(), stream));
            canonical.expect(std::as_bytes(std::span<const qwen::Q8_1>(q)));
            canonical.readonly(); flag.readonly(); raw.readonly();
            if (counters) ++counters->quantization;
        } else {
            require(manual.size() == q.size(), "manual Q8 exact span");
            std::copy(manual.begin(), manual.end(), q.begin());
            canonical.store(std::as_bytes(std::span<const qwen::Q8_1>(q)));
            if (counters) manual_proofs(q, *counters);
        }
        packed = expected_pack(q, width, n);
        check(qwen::launch_q8_1_to_mmq(canonical.as<qwen::Q8_1>(), width, n, ds4.as<qwen::Q8MmqBlock>(), stream));
        ds4.expect(std::as_bytes(std::span<const qwen::Q8MmqBlock>(packed)));
        ds4.readonly(); readonly();
        if (counters) ++counters->packing;
    }
    void readonly() const { raw.readonly(); canonical.readonly(); ds4.readonly(); flag.readonly(); }
    void poison_tail(Counters& counters) {
        const int used = (width / 32) % 4;
        if (!used) return;
        const auto original = packed;
        for (int c = 0; c < n; ++c) for (int sub = used; sub < 4; ++sub) {
            const std::size_t b = std::size_t((width / 32) / 4) * n + c;
            auto& block = packed[b];
            block.ds[2 * sub] = 0x7e11; block.ds[2 * sub + 1] = 0xfe55;
            for (int j = 0; j < 32; ++j) block.qs[32 * sub + j] = j % 2 ? 127 : -128;
            // Upload ONLY the unused pair/code region, never active Q8 bytes.
            ds4.store(std::as_bytes(std::span<const std::uint16_t>(block.ds + 2 * sub, 2)),
                b * sizeof(block) + std::size_t(2 * sub) * sizeof(std::uint16_t));
            ds4.store(std::as_bytes(std::span<const std::int8_t>(block.qs + 32 * sub, 32)),
                b * sizeof(block) + offsetof(qwen::Q8MmqBlock, qs) + std::size_t(32 * sub));
            ++counters.poison_blocks;
        }
        // Independent active-subblock proof against the original common-Q8.
        for (int k = 0; k < width / 32; ++k) for (int c = 0; c < n; ++c) {
            const auto b = std::size_t(k / 4) * n + c; const int sub = k % 4;
            require(packed[b].ds[2 * sub] == original[b].ds[2 * sub] &&
                packed[b].ds[2 * sub + 1] == original[b].ds[2 * sub + 1] &&
                std::memcmp(packed[b].qs + 32 * sub, original[b].qs + 32 * sub, 32) == 0,
                "tail poison touched an active subblock");
        }
        readonly(); ++counters.poison;
    }
};

struct Case {
    const Matrix& matrix;
    hipStream_t stream;
    int n;
    Inputs input;
    Buffer weights, output;
    std::vector<float> reference;
    Case(const Matrix& m, hipStream_t s, int columns, Counters* counters,
         std::span<const qwen::Q8_1> manual = {})
        : matrix(m), stream(s), n(columns), input(s, m.width, columns, counters, manual),
          weights(s, m.bytes.size(), 2, 2), output(s, std::size_t(m.rows) * columns * sizeof(float), 4, 4),
          reference(std::size_t(m.rows) * columns) {
        weights.store(m.bytes);
        require(reinterpret_cast<std::uintptr_t>(weights.as<void>()) % 4 == 2, "weightpointeroffset2 not exercised");
        // Qualified reference permits ONLY N<=3. Full-N oracle is computed in
        // ascending disjoint 1..3-column slices with matching exact spans.
        for (int c = 0; c < n; c += 3) {
            const int count = std::min(3, n - c);
            qwen::matmul_q8_reference(m.view(), std::span<const qwen::Q8_1>(input.q).subspan(
                std::size_t(c) * (m.width / 32), std::size_t(count) * (m.width / 32)), count,
                std::span<float>(reference).subspan(std::size_t(c) * m.rows, std::size_t(count) * m.rows));
            if (counters) ++counters->slices;
        }
        for (float v : reference) require(std::isfinite(v), "CPU common-Q8 reference nonfinite");
    }
    qwen::QuantizedDeviceMatrix device_matrix() const {
        return {matrix.type, weights.as<void>(), matrix.width, matrix.rows};
    }
    void reset_output() {
        const auto bits = std::uint32_t{0x7fc12345};
        for (std::size_t i = 0; i < output.bytes; i += 4)
            std::memcpy(output.expected.data() + output.prefix + i, &bits, 4);
        output.upload_all();
    }
    void launch(bool mmq) const {
        if (mmq) check(qwen::launch_mmq_q4(device_matrix(), input.ds4.as<qwen::Q8MmqBlock>(), n, output.as<float>(), stream));
        else {
            // Diagnostic A only: current qualified launch, <=3 columns per
            // call, ordered slices covering the SAME Q8 inputs/output spans.
            for (int c = 0; c < n; c += 3)
                check(qwen::launch_quantized_linear(device_matrix(), input.canonical.as<qwen::Q8_1>() +
                    std::size_t(c) * (matrix.width / 32), std::min(3, n - c),
                    output.as<float>() + std::size_t(c) * matrix.rows, stream));
        }
    }
    Error validate() const {
        const auto bytes = output.read(); output.canaries(bytes);
        std::vector<float> got(reference.size());
        std::memcpy(got.data(), bytes.data() + output.prefix, output.bytes);
        Error error; error.compare(got, reference); input.readonly(); weights.readonly(); return error;
    }
    void qualify(Counters& counters, bool poison = true) {
        if (poison) input.poison_tail(counters);
        reset_output(); launch(true); const auto error = validate();
        counters.error.merge(error); ++counters.matrices; counters.elements += reference.size();
    }
};

// Deliberately invalid addresses appear ONLY in calls proven invalid by the
// frozen wrapper's pre-enqueue checks. Never send any such address to HIP copy,
// events, dereference or a valid launch. Check pending HIP error before/after,
// drain and compare ALL valid buffers, then re-pack and run the valid reuse.
void rejects_and_reuse(hipStream_t stream, Counters& counters) {
    const auto matrix = synthetic(qwen::TensorType::Q4_1, 160, 7);
    Case test(matrix, stream, 3, &counters);
    test.reset_output(); check(hipGetLastError());
    const auto good = test.device_matrix();
    auto* x = test.input.raw.as<float>(); auto* q = test.input.canonical.as<qwen::Q8_1>();
    auto* p = test.input.ds4.as<qwen::Q8MmqBlock>(); auto* y = test.output.as<float>();
    auto* flag = test.input.flag.as<int>();
    const auto top = std::numeric_limits<std::uintptr_t>::max();
    auto reject = [&](hipError_t status, std::size_t& counter, hipError_t expected = hipErrorInvalidValue) {
        require(status == expected, "host pre-enqueue rejection status"); check(hipGetLastError()); ++counter;
    };
    auto quant = [&](const float* a, qwen::Q8_1* b, int k, int n, int* f) {
        reject(qwen::launch_quantize_q8_1(a, b, k, n, f, stream), counters.rejects_quantizer);
    };
    for (int k : {0, -32, 31, 16416, std::numeric_limits<int>::max()}) quant(x, q, k, 3, flag);
    for (int n : {0, -1, 16385, std::numeric_limits<int>::max()}) quant(x, q, 160, n, flag);
    quant(nullptr, q, 160, 3, flag); quant(x, nullptr, 160, 3, flag); quant(x, q, 160, 3, nullptr);
    quant(test.input.raw.as<float>(2), q, 160, 3, flag);
    quant(x, test.input.canonical.as<qwen::Q8_1>(2), 160, 3, flag);
    quant(x, q, 160, 3, test.input.flag.as<int>(2));
    quant(reinterpret_cast<const float*>(top - 3), q, 160, 3, flag);
    quant(x, reinterpret_cast<qwen::Q8_1*>(top - 3), 160, 3, flag);
    quant(x, q, 160, 3, reinterpret_cast<int*>(top - 3));
    quant(x, reinterpret_cast<qwen::Q8_1*>(x), 160, 3, flag);
    quant(x, q, 160, 3, reinterpret_cast<int*>(x));
    quant(x, q, 160, 3, reinterpret_cast<int*>(q));
    // Overlap at a last aligned byte range, not merely equal pointers.
    quant(x, reinterpret_cast<qwen::Q8_1*>(x + 479), 160, 3, flag);
    quant(x, q, 160, 3, reinterpret_cast<int*>(x + 479));
    quant(x, q, 160, 3, reinterpret_cast<int*>(q + 14));
    quant(reinterpret_cast<float*>(top - 1), q, 160, 3, flag);
    quant(x, reinterpret_cast<qwen::Q8_1*>(top - 1), 160, 3, flag);
    quant(x, q, 160, 3, reinterpret_cast<int*>(top - 1));
    quant(x, q, -1, -1, flag); quant(nullptr, nullptr, 160, 3, nullptr);
    quant(x, q, 32, 0, flag);

    auto pack = [&](const qwen::Q8_1* a, int k, int n, qwen::Q8MmqBlock* b) {
        reject(qwen::launch_q8_1_to_mmq(a, k, n, b, stream), counters.rejects_packer);
    };
    for (int k : {0, -32, 31, 16416, std::numeric_limits<int>::max()}) pack(q, k, 3, p);
    for (int n : {0, -1, 129, std::numeric_limits<int>::max()}) pack(q, 160, n, p);
    pack(nullptr, 160, 3, p); pack(q, 160, 3, nullptr);
    pack(test.input.canonical.as<qwen::Q8_1>(2), 160, 3, p);
    pack(q, 160, 3, test.input.ds4.as<qwen::Q8MmqBlock>(8));
    pack(reinterpret_cast<qwen::Q8_1*>(top - 3), 160, 3, p);
    pack(q, 160, 3, reinterpret_cast<qwen::Q8MmqBlock*>(top - 15));
    pack(q, 160, 3, reinterpret_cast<qwen::Q8MmqBlock*>(q)); // rejected alignment/overlap
    pack(reinterpret_cast<qwen::Q8_1*>(p), 160, 3, p);
    pack(reinterpret_cast<qwen::Q8_1*>(p) + 1, 160, 3, p);
    pack(reinterpret_cast<qwen::Q8_1*>(top - 1), 160, 3, p);
    pack(q, 160, 3, reinterpret_cast<qwen::Q8MmqBlock*>(top - 7));
    pack(nullptr, 0, 0, nullptr); pack(q, 32, 129, p); pack(q, 33, 1, p);

    auto mmq = [&](qwen::QuantizedDeviceMatrix m, const qwen::Q8MmqBlock* a, int n, float* b,
                   hipError_t expected = hipErrorInvalidValue) {
        reject(qwen::launch_mmq_q4(m, a, n, b, stream), counters.rejects_mmq, expected);
    };
    for (int k : {0, -32, 31, 16416, std::numeric_limits<int>::max()}) {
        auto m = good; m.input = k; mmq(m, p, 3, y);
    }
    for (int r : {0, -1, 16385, std::numeric_limits<int>::max()}) {
        auto m = good; m.output = r; mmq(m, p, 3, y);
    }
    for (int n : {0, -1, 129, std::numeric_limits<int>::max()}) mmq(good, p, n, y);
    for (auto type : {qwen::TensorType::F32, qwen::TensorType::F16, qwen::TensorType::BF16,
                     qwen::TensorType::Q5_0, qwen::TensorType::Q8_0, qwen::TensorType::Q6_K,
                     static_cast<qwen::TensorType>(999)}) {
        auto m = good; m.type = type; mmq(m, p, 3, y, hipErrorNotSupported);
    }
    auto m = good; m.weights = nullptr; mmq(m, p, 3, y);
    m = good; m.weights = test.weights.as<std::byte>(1); mmq(m, p, 3, y);
    m = good; m.weights = reinterpret_cast<void*>(top - 1); mmq(m, p, 3, y);
    mmq(good, nullptr, 3, y); mmq(good, p, 3, nullptr);
    mmq(good, test.input.ds4.as<qwen::Q8MmqBlock>(8), 3, y);
    mmq(good, p, 3, test.output.as<float>(2));
    mmq(good, reinterpret_cast<qwen::Q8MmqBlock*>(top - 15), 3, y);
    mmq(good, p, 3, reinterpret_cast<float*>(top - 3));
    mmq(good, p, 3, reinterpret_cast<float*>(p));
    mmq(good, p, 3, reinterpret_cast<float*>(test.weights.as<std::byte>(2)));
    mmq(good, p, 3, reinterpret_cast<float*>(reinterpret_cast<std::byte*>(p) + test.input.ds4.bytes - 4));
    mmq(good, p, 3, reinterpret_cast<float*>(test.weights.as<std::byte>(test.weights.bytes - 6)));
    m = good; m.weights = reinterpret_cast<void*>(top); mmq(m, p, 3, y);
    mmq(good, reinterpret_cast<qwen::Q8MmqBlock*>(top - 7), 3, y);
    mmq(good, p, 3, reinterpret_cast<float*>(top - 1));
    check(hipStreamSynchronize(stream));
    test.input.readonly(); test.weights.readonly(); test.output.readonly();
    // Valid reuse of all three APIs on precisely the same allocations after
    // all rejects. These extra producer launches repeat an already counted
    // input/packing case; case counters are not kernel-launch counters.
    check(qwen::launch_quantize_q8_1(x, q, 160, 3, flag, stream)); test.input.readonly();
    check(qwen::launch_q8_1_to_mmq(q, 160, 3, p, stream)); test.input.readonly();
    test.qualify(counters);
}

void qualification(int device, const std::array<Matrix, 3>& actual, Counters& counters) {
    check(hipSetDevice(device)); Stream stream;
    rejects_and_reuse(stream.value, counters);
    for (int edge = 0; edge < 8; ++edge) { Inputs input(stream.value, 256, 1, &counters, {}, edge); }
    for (auto type : types) for (int width : widths) for (int rows : row_counts) {
        const auto matrix = synthetic(type, width, rows);
        for (int n : column_counts) { Case test(matrix, stream.value, n, &counters); test.qualify(counters); }
    }
    for (auto type : types) {
        const auto matrix = synthetic(type, 160, 7); const auto q = manual_input(160, 3);
        Case test(matrix, stream.value, 3, &counters, q); test.qualify(counters);
    }
    {
        const auto matrix = half_corner_matrix(); const auto q = manual_input(32, 3, true);
        bool rounding_differs = false, tie_to_zero = false;
        for (int r = 0; r < 7; ++r) for (const auto& block : q) {
            const float d = qwen::half_to_float(half_bits(matrix.bytes.data() + r * 20));
            const float m = qwen::half_to_float(half_bits(matrix.bytes.data() + r * 20 + 2));
            const float dd = d * qwen::half_to_float(block.d), ms = m * qwen::half_to_float(block.s);
            const float rounded_d = qwen::half_to_float(qwen::float_to_half(dd));
            const float rounded_m = qwen::half_to_float(qwen::float_to_half(ms));
            require(std::isfinite(rounded_d) && std::isfinite(rounded_m), "half-product numeric contract");
            rounding_differs |= rounded_d != dd || rounded_m != ms;
            tie_to_zero |= std::abs(dd) == std::ldexp(1.0f, -25) && rounded_d == 0;
            ++counters.half_pairs;
        }
        require(rounding_differs && tie_to_zero, "half-RNE product corner proof");
        Case test(matrix, stream.value, 3, &counters, q); test.qualify(counters);
    }
    for (const auto& matrix : actual) for (int n : column_counts) {
        Case test(matrix, stream.value, n, &counters); test.qualify(counters);
    }
    counters.verify(); check(hipStreamSynchronize(stream.value));
}

void measurement(Case& test, int device, std::size_t tensor_index, std::size_t column_index,
                 std::size_t phase_index, std::size_t sequence, Event& begin, Event& end) {
    const bool mmq = phase_index == 1;
    test.reset_output(); test.launch(mmq); const auto warmup = test.validate();
    Error error; std::array<double, repeats> intervals{}; double total = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        test.reset_output();
        check(hipEventRecord(begin.value, test.stream)); test.launch(mmq);
        check(hipEventRecord(end.value, test.stream)); check(hipEventSynchronize(end.value));
        float ms = 0; check(hipEventElapsedTime(&ms, begin.value, end.value));
        require(std::isfinite(ms) && ms > 0, "individually completed HIP event interval");
        intervals[rep] = ms; total += ms;
        // Full output, redzones and ALL readonly payloads validated EVERY time.
        // No reset/upload/quantization/pack/reference/readback in the interval.
        error.merge(test.validate());
    }
    std::cout << "{\"kind\":\"mmq_measurement\",\"protocol\":2,\"sequence\":" << sequence
        << ",\"device\":" << device << ",\"tensor_index\":" << tensor_index << ",\"column_index\":" << column_index
        << ",\"phase_index\":" << phase_index << ",\"phase\":"; json_string(phases[phase_index]);
    std::cout << ",\"path\":"; json_string(mmq ? "mmq" : "diagnostic_sliced_linear");
    std::cout << ",\"tensor\":"; json_string(test.matrix.name);
    std::cout << ",\"type\":"; json_string(qwen::type_name(test.matrix.type));
    std::cout << ",\"width\":" << test.matrix.width << ",\"rows\":" << test.matrix.rows << ",\"columns\":" << test.n
        << ",\"tile_j\":" << tile_j(test.matrix.rows, test.n) << ",\"column_tiles\":" << (test.n + tile_j(test.matrix.rows, test.n) - 1) / tile_j(test.matrix.rows, test.n)
        << ",\"launches_per_repeat\":" << (mmq ? 1 : (test.n + 2) / 3)
        << ",\"repeats\":20,\"validated_intervals\":20,\"elements_per_interval\":" << test.reference.size()
        << ",\"completed_ms\":"; json_ints(intervals);
    std::cout << ",\"completed_ms_total\":" << total << ",\"resident_ms\":" << total / repeats
        << ",\"max_abs\":" << error.max_abs << ",\"max_bound_ratio\":" << error.max_bound_ratio
        << ",\"warmup_max_abs\":" << warmup.max_abs << ",\"warmup_max_bound_ratio\":" << warmup.max_bound_ratio
        << ",\"finite\":true,\"canary\":true,\"readonly\":true,\"q8_bytes_exact\":true,\"packed_bytes_exact\":true,\"passed\":true}\n";
}
void performance(int device, const std::array<Matrix, 3>& actual, std::size_t& sequence) {
    check(hipSetDevice(device)); Stream stream; Event begin(stream.value), end(stream.value);
    for (std::size_t tensor = 0; tensor < actual.size(); ++tensor)
        for (std::size_t column = 0; column < benchmark_columns.size(); ++column) {
            Case test(actual[tensor], stream.value, benchmark_columns[column], nullptr);
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
                measurement(test, device, tensor, column, phase, sequence++, begin, end);
        }
    check(hipStreamSynchronize(stream.value));
}

void emit_expected() {
    std::cout << "{\"grid_cases\":936,\"actual_cases\":39,\"corner_cases\":3,\"reuse_cases\":1,\"quantizer_edges\":8"
        ",\"matrix_cases\":979,\"quantization_cases\":984,\"packing_cases\":987,\"poison_cases\":316,\"poisoned_subblocks\":27540"
        ",\"compared_elements\":2568652,\"reference_slices\":9904,\"quantizer_rejects\":30,\"packer_rejects\":23,\"mmq_rejects\":36"
        ",\"host_rejects\":89,\"manual_cases\":3,\"int8_min_cases\":3,\"raw_sum_difference_cases\":3,\"signed_q8_scale_cases\":3,\"half_product_pairs\":21}";
}
void source(const char* path, const qwen::Model& model, const std::array<Matrix, 3>& actual,
            const std::array<hipDeviceProp_t, 2>& devices) {
    std::cout << std::setprecision(17) << "{\"kind\":\"mmq_source\",\"protocol\":2,\"revision\":"; json_string(CORE_REVISION);
    std::cout << ",\"dirty\":" << CORE_DIRTY << ",\"model\":"; json_string(path);
    std::cout << ",\"model_variant\":\"qwen38-keep1-Q4_0\",\"architecture\":\"qwen4exp\",\"absolute_gate\":0.0002,\"relative_gate\":0.00002"
        ",\"scope\":\"component common-Q8 quantization/DS4 packing/canonical Q4 MMQ; no PP or inference speed claim\""
        ",\"numeric_contract\":\"finite static weight halves and valid finite Q8 halves/codes; numeric validity is caller-owned\""
        ",\"oracle\":\"unchanged matmul_q8_reference, ordered 1..3-column CPU slices, full-N finite comparison\""
        ",\"baseline\":\"diagnostic sliced current qualified launch_quantized_linear Columns<=3; benchmark only\""
        ",\"timing_scope\":\"20 individually completed resident HIP-event intervals per phase; all output/read-only/redzones validated each interval\""
        ",\"timing_excludes\":[\"allocation\",\"model_reads\",\"H2D\",\"quantization\",\"packing\",\"CPU_reference\",\"output_reset\",\"readback\"]"
        ",\"ordering\":\"source; correctness device0,device1 BEFORE performance; measurements device,tensor,column,phase A1/B/A2; complete after all cleanup\""
        ",\"stream\":\"explicit hipStreamNonBlocking\",\"q8_1\":{\"bytes\":36,\"qs_offset\":4,\"sum\":\"original half raw-input sum, not d*sum(codes)\"}"
        ",\"ds4\":{\"bytes\":144,\"alignment\":16,\"ds_offset\":0,\"qs_offset\":16,\"layout\":\"[K128][column]\",\"ds_order\":\"d0,s0,d1,s1,d2,s2,d3,s3\",\"missing_subblocks\":\"zero halves/codes; unused final subblocks alone poisoned before qualification MMQ\"}"
        ",\"tiles\":{\"rows\":64,\"k\":256,\"block\":[64,4],\"j\":[8,16,32,64],\"selection\":\"M<=640 J8 microtiles; otherwise smallest J>=N capped64\"}"
        ",\"buffers\":{\"exact_payloads\":true,\"weight_pointer_offset\":2,\"weight_alignment\":2,\"q8_alignment\":4,\"packed_alignment\":16,\"output_alignment\":4,\"redzones\":\"prefix/suffix bytes weights2/2, raw4/4, Q8_1 4/4, DS4 16/16, output4/4, flag4/4; half-qNaN poison\"}"
        ",\"coverage\":{\"types\":[\"Q4_0\",\"Q4_1\"],\"cartesian\":true,\"widths\":"; json_ints(widths);
    std::cout << ",\"rows\":"; json_ints(row_counts); std::cout << ",\"columns\":"; json_ints(column_counts);
    std::cout << ",\"quantizer_edges\":{\"width\":256,\"columns\":1,\"cases\":[\"zero\",\"negative_zero\",\"half_rounding\",\"minimum_half_scale\",\"signed_alternating\",\"small_sine\",\"unrounded_scale_codes\",\"large_cancellation\"]}"
        ",\"manual_corners\":[{\"type\":\"Q4_0\",\"width\":160,\"rows\":7,\"columns\":3},{\"type\":\"Q4_1\",\"width\":160,\"rows\":7,\"columns\":3},{\"type\":\"Q4_1\",\"width\":32,\"rows\":7,\"columns\":3}]"
        ",\"reuse\":{\"type\":\"Q4_1\",\"width\":160,\"rows\":7,\"columns\":3},\"reject_categories\":[\"null\",\"shape\",\"type\",\"alignment\",\"range_wrap\",\"write_alias\"]}"
        ",\"benchmark_columns\":"; json_ints(benchmark_columns);
    std::cout << ",\"benchmark_phases\":[\"A1\",\"B\",\"A2\"],\"repeats\":20,\"expected_per_device\":"; emit_expected();
    std::cout << ",\"expected_protocol\":{\"devices\":2,\"source_records\":1,\"correctness_records\":2,\"measurements_per_device\":45,\"measurement_records\":90,\"validated_intervals\":1800,\"complete_records\":1,\"jsonl_records\":94},\"actual\":[";
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (i) std::cout << ',';
        const auto& m = actual[i]; const auto& t = model.tensor(m.name);
        std::cout << "{\"tensor_index\":" << i << ",\"tensor\":"; json_string(m.name);
        std::cout << ",\"type\":"; json_string(qwen::type_name(m.type));
        std::cout << ",\"rank\":" << t.rank << ",\"dimensions\":[";
        for (std::size_t axis = 0; axis < t.rank; ++axis) { if (axis) std::cout << ','; std::cout << t.dimensions[axis]; }
        std::cout << "],\"strides_bytes\":[";
        for (std::size_t axis = 0; axis < t.rank; ++axis) { if (axis) std::cout << ','; std::cout << t.strides[axis]; }
        std::cout << "],\"elements\":" << t.elements << ",\"tensor_bytes\":" << t.byte_size << ",\"file_offset\":" << t.file_offset
            << ",\"read_api\":"; json_string(i < 2 ? "Model.read_expert" : "Model.read_slice");
        std::cout << ",\"expert\":" << (i < 2 ? "0" : "null") << ",\"slice_offset\":0,\"selected_bytes\":" << m.bytes.size()
            << ",\"width\":" << m.width << ",\"rows\":" << m.rows << ",\"unchanged\":true}";
    }
    std::cout << "],\"devices\":[";
    for (std::size_t d = 0; d < devices.size(); ++d) {
        if (d) std::cout << ',';
        std::cout << "{\"device\":" << d << ",\"arch\":"; json_string(devices[d].gcnArchName);
        std::cout << ",\"wave\":" << devices[d].warpSize << '}';
    }
    // Full fixed field schemas; no optional records/fields. Counts/coordinates
    // are integers, error/time fields finite numbers, proofs JSON booleans.
    std::cout << "],\"schemas\":{\"mmq_source\":[\"kind\",\"protocol\",\"revision\",\"dirty\",\"model\",\"model_variant\",\"architecture\",\"absolute_gate\",\"relative_gate\",\"scope\",\"numeric_contract\",\"oracle\",\"baseline\",\"timing_scope\",\"timing_excludes\",\"ordering\",\"stream\",\"q8_1\",\"ds4\",\"tiles\",\"buffers\",\"coverage\",\"benchmark_columns\",\"benchmark_phases\",\"repeats\",\"expected_per_device\",\"expected_protocol\",\"actual\",\"devices\",\"schemas\"]"
        ",\"mmq_correctness\":[\"kind\",\"protocol\",\"device\",\"counters\",\"max_abs\",\"max_bound_ratio\",\"q8_bytes_exact\",\"packed_bytes_exact\",\"missing_subblocks_zero\",\"poison_masked\",\"active_subblocks_unchanged\",\"half_RNE_corners\",\"rejects_unchanged\",\"valid_reuse\",\"finite\",\"canary\",\"readonly\",\"cleanup\",\"passed\"]"
        ",\"mmq_measurement\":[\"kind\",\"protocol\",\"sequence\",\"device\",\"tensor_index\",\"column_index\",\"phase_index\",\"phase\",\"path\",\"tensor\",\"type\",\"width\",\"rows\",\"columns\",\"tile_j\",\"column_tiles\",\"launches_per_repeat\",\"repeats\",\"validated_intervals\",\"elements_per_interval\",\"completed_ms\",\"completed_ms_total\",\"resident_ms\",\"max_abs\",\"max_bound_ratio\",\"warmup_max_abs\",\"warmup_max_bound_ratio\",\"finite\",\"canary\",\"readonly\",\"q8_bytes_exact\",\"packed_bytes_exact\",\"passed\"]"
        ",\"mmq_complete\":[\"kind\",\"protocol\",\"devices\",\"correctness_records\",\"measurement_records\",\"validated_intervals\",\"jsonl_records\",\"cleanup\",\"passed\"]}}\n";
}
void correctness(int device, const Counters& counters) {
    counters.verify();
    std::cout << "{\"kind\":\"mmq_correctness\",\"protocol\":2,\"device\":" << device << ",\"counters\":";
    emit_expected();
    std::cout << ",\"max_abs\":" << counters.error.max_abs << ",\"max_bound_ratio\":" << counters.error.max_bound_ratio
        << ",\"q8_bytes_exact\":true,\"packed_bytes_exact\":true,\"missing_subblocks_zero\":true,\"poison_masked\":true"
        ",\"active_subblocks_unchanged\":true,\"half_RNE_corners\":true,\"rejects_unchanged\":true,\"valid_reuse\":true"
        ",\"finite\":true,\"canary\":true,\"readonly\":true,\"cleanup\":true,\"passed\":true}\n";
}
void run(const char* path) {
    // Runtime provenance gate intentionally rejects CMake's unknown default.
    const std::string_view revision = CORE_REVISION;
    require(revision.size() == 40 && std::all_of(revision.begin(), revision.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }), "CORE_REVISION must be a quoted 40-lowercase-hex revision");
    require(CORE_DIRTY == 0 || CORE_DIRTY == 1, "CORE_DIRTY must be numeric 0/1");
    int count = 0; check(hipGetDeviceCount(&count)); require(count == 2, "core-mmq requires exactly two GPUs");
    std::array<hipDeviceProp_t, 2> devices{};
    for (int d = 0; d < 2; ++d) {
        check(hipGetDeviceProperties(&devices[d], d)); const std::string_view arch = devices[d].gcnArchName;
        require(devices[d].warpSize == 64 && (arch == "gfx906" || arch.starts_with("gfx906:")), "requires exact gfx906 ISA/wave64");
    }
    qwen::Model model(path);
    require(model.metadata_value("general.architecture").get<std::string>() == "qwen4exp", "actual model architecture");
    const auto actual = actual_matrices(model); source(path, model, actual, devices);
    for (int d = 0; d < 2; ++d) {
        Counters counters; qualification(d, actual, counters); // all device objects destroyed on return
        correctness(d, counters);
    }
    std::size_t sequence = 0;
    for (int d = 0; d < 2; ++d) performance(d, actual, sequence);
    require(sequence == measurement_records && measurements_per_device * 2 == sequence &&
        host_rejects == 89 && actual_cases + grid_cases + corner_cases + 1 == matrix_cases,
        "MMQ protocol totals");
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: core-mmq MODEL.gguf");
        run(argv[1]); // ALL streams/events/buffers/model/host payloads cleaned up
        std::cout << "{\"kind\":\"mmq_complete\",\"protocol\":2,\"devices\":2,\"correctness_records\":2,\"measurement_records\":90,\"validated_intervals\":1800,\"jsonl_records\":"
            << jsonl_records << ",\"cleanup\":true,\"passed\":true}\n";
        std::cout.flush(); require(bool(std::cout), "MMQ JSONL output failure"); return 0;
    } catch (const std::exception& e) { std::cerr << "core-mmq: " << e.what() << '\n'; return 1; }
}
