#include "dense.hpp"
#include "hc.hpp"
#include "ple.hpp"
#include "model.hpp"
#include "hip/blocks.cuh"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
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

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "blocks fixtures require finite checks and unfused FP32 CPU arithmetic"
#endif
#ifdef __clang__
#pragma clang fp contract(off)
#endif

// core-blocks target.gguf -- protocol 1, stdout is JSONL ONLY.
// Successful order: blocks_source; blocks_correctness(device=0,1); then
// blocks_measurement(device=0,1 outer, tokens=1,2,3,128 middle, HC,PLE inner);
// blocks_complete. Exactly 20 rows, 16 measurements, 320 timed repetitions.
// Missing complete, nonzero exit, false applicable parity, or a bound ratio>1
// is failure. Null parity fields mean not applicable (HC has no PLE history).
// Every error object maps named groups to {max_abs,max_bound_ratio,elements};
// each counter counts completed checks/cases, not launches or generated tokens.
// Timing: mean of 20 separately completed HIP event intervals, milliseconds.
// HC has 5 kernel launches; PLE has 7 (publication itself has two kernels).
// Scope is resident FP32 primitives with SAME immutable synthetic projections,
// actual stored gamma and decoded original F16 convolution. No projected matrix
// multiplication, tokenizer/hash/gather, quantized pipeline or request speed.
// CORE_REVISION/CORE_DIRTY are build provenance; absent definitions emit null,
// never a fabricated clean revision. Collector must retain those null values.

namespace {
constexpr int hidden = 2560, branches = 4, wide = 10240, rank = 320;
constexpr std::size_t history_size = 92160, weight_size = 40960;
constexpr int repeats = 20, protocol_rows = 20;
constexpr std::array<int, 4> token_counts{1, 2, 3, 128};
// Frozen BEFORE any GPU execution. CPU magnitude determines the bound.
constexpr double absolute_gate = 2e-4, relative_gate = 2e-4;
constexpr std::size_t guard_bytes = 64;
constexpr unsigned char guard_value = 0xa7, unused_value = 0xcd;
static_assert(sizeof(float) == 4 && sizeof(int) == 4);

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) {
        std::cerr << "blocks cleanup: " << hipGetErrorString(status) << '\n';
        std::abort();
    }
}
void json_string(std::string_view value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32) {
            constexpr char hex[] = "0123456789abcdef";
            std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else std::cout << char(c);
    }
    std::cout << '"';
}
template<class T> void exact(std::span<const T> got, std::span<const T> ref, const char* what) {
    require(got.size() == ref.size(), what);
    require(got.empty() || std::memcmp(got.data(), ref.data(), got.size_bytes()) == 0, what);
}
template<class T> void exact(const std::vector<T>& got, const std::vector<T>& ref, const char* what) {
    exact<T>(got, std::span<const T>(ref), what);
}
struct Gate {
    double max_abs = 0, max_bound_ratio = 0;
    std::size_t elements = 0;
    void compare(std::span<const float> got, std::span<const float> ref, const char* what) {
        require(got.size() == ref.size(), "blocks comparison shape");
        for (std::size_t i = 0; i < got.size(); ++i) {
            require(std::isfinite(got[i]) && std::isfinite(ref[i]), what);
            const double error = std::abs(double(got[i]) - double(ref[i]));
            const double bound = absolute_gate + relative_gate * std::abs(double(ref[i]));
            max_abs = std::max(max_abs, error);
            max_bound_ratio = std::max(max_bound_ratio, error / bound);
            ++elements;
            if (error > bound)
                std::cerr << what << ": element=" << i << " got=" << std::setprecision(17)
                          << got[i] << " reference=" << ref[i] << " bound=" << bound << '\n';
            require(error <= bound, what);
        }
    }
};
struct Errors { Gate rms, hc, ple, history; };
void gate_json(const Gate& g) {
    std::cout << "{\"max_abs\":" << g.max_abs << ",\"max_bound_ratio\":" << g.max_bound_ratio
              << ",\"elements\":" << g.elements << '}';
}
void errors_json(const Errors& e) {
    std::cout << "{\"rms\":"; gate_json(e.rms);
    std::cout << ",\"hc\":"; gate_json(e.hc);
    std::cout << ",\"ple\":"; gate_json(e.ple);
    std::cout << ",\"history\":"; gate_json(e.history); std::cout << '}';
}
struct Counters {
    std::size_t norm_cases = 0, activation_cases = 0, mix_cases = 0, injection_cases = 0;
    std::size_t combine_cases = 0, gate_cases = 0, conv_cases = 0, head_cases = 0;
    std::size_t broadcast_cases = 0, signed_zero_cases = 0, large_finite_cases = 0;
    std::size_t prefix_slots = 0, restore_cases = 0, repeated_accept0_cases = 0, mask_cases = 0;
    std::size_t readonly_active_checks = 0, publication_checks = 0, rejected_publications = 0;
    std::size_t numeric_rejects = 0, host_rejects = 0, readonly_alias_cases = 0;
    std::size_t numeric_invalid_cases = 0, numeric_arithmetic_cases = 0, numeric_combined_cases = 0;
    std::size_t immutable_checks = 0, canary_checks = 0, valid_reuses = 0;
};
void counters_json(const Counters& c) {
    std::cout << '{';
#define BLOCKS_COUNTER(name) std::cout << "\"" #name "\":" << c.name
    BLOCKS_COUNTER(norm_cases); std::cout << ','; BLOCKS_COUNTER(activation_cases);
    std::cout << ','; BLOCKS_COUNTER(mix_cases); std::cout << ','; BLOCKS_COUNTER(injection_cases);
    std::cout << ','; BLOCKS_COUNTER(combine_cases); std::cout << ','; BLOCKS_COUNTER(gate_cases);
    std::cout << ','; BLOCKS_COUNTER(conv_cases); std::cout << ','; BLOCKS_COUNTER(head_cases);
    std::cout << ','; BLOCKS_COUNTER(broadcast_cases); std::cout << ','; BLOCKS_COUNTER(signed_zero_cases);
    std::cout << ','; BLOCKS_COUNTER(large_finite_cases); std::cout << ','; BLOCKS_COUNTER(prefix_slots);
    std::cout << ','; BLOCKS_COUNTER(restore_cases); std::cout << ','; BLOCKS_COUNTER(repeated_accept0_cases);
    std::cout << ','; BLOCKS_COUNTER(mask_cases); std::cout << ','; BLOCKS_COUNTER(readonly_active_checks);
    std::cout << ','; BLOCKS_COUNTER(publication_checks); std::cout << ','; BLOCKS_COUNTER(rejected_publications);
    std::cout << ','; BLOCKS_COUNTER(numeric_rejects); std::cout << ','; BLOCKS_COUNTER(host_rejects);
    std::cout << ','; BLOCKS_COUNTER(numeric_invalid_cases); std::cout << ','; BLOCKS_COUNTER(numeric_arithmetic_cases);
    std::cout << ','; BLOCKS_COUNTER(numeric_combined_cases);
    std::cout << ','; BLOCKS_COUNTER(readonly_alias_cases); std::cout << ','; BLOCKS_COUNTER(immutable_checks);
    std::cout << ','; BLOCKS_COUNTER(canary_checks); std::cout << ','; BLOCKS_COUNTER(valid_reuses);
#undef BLOCKS_COUNTER
    std::cout << '}';
}
struct Summary { Errors errors; Counters counters; };

std::uint64_t unsigned_metadata(const qwen::Model& model, const char* key) {
    const auto& v = model.metadata_value(key);
    if (v.type == qwen::MetadataType::UINT32) return v.get<std::uint32_t>();
    if (v.type == qwen::MetadataType::UINT64) return v.get<std::uint64_t>();
    throw std::runtime_error(std::string("blocks unsigned metadata type: ") + key);
}
struct Data {
    qwen::HcConfig hc;
    qwen::PleLayerConfig ple;
    std::array<std::vector<float>, 3> hc_gamma, ple_gamma;
    std::array<std::vector<float>, 5> small_gamma;
    std::vector<float> conv;
    std::vector<qwen::TensorView> views;
    std::vector<float> load(const qwen::Model& model, const char* name,
                           std::initializer_list<std::uint64_t> dims, qwen::TensorType type) {
        const auto& t = model.tensor(name);
        require(t.type == type && t.rank == dims.size() &&
                std::equal(dims.begin(), dims.end(), t.dimensions.begin()), "blocks actual tensor type/shape");
        const std::uint64_t element_bytes = type == qwen::TensorType::F32 ? 4 : 2;
        std::uint64_t elements = 1, stride = element_bytes;
        std::size_t axis = 0;
        for (auto d : dims) {
            require(d > 0 && t.strides[axis++] == stride, "blocks tensor canonical strides");
            elements *= d; stride *= d; // fixed, checked geometry above
        }
        require(t.elements == elements && t.byte_size == elements * element_bytes &&
                t.relative_offset % model.alignment() == 0 &&
                t.file_offset >= model.data_offset() &&
                t.file_offset - model.data_offset() == t.relative_offset &&
                t.file_offset <= model.file_size() && t.byte_size <= model.file_size() - t.file_offset,
                "blocks tensor sizes/offsets");
        std::vector<std::byte> bytes(static_cast<std::size_t>(t.byte_size));
        model.read_tensor(name, bytes);
        std::vector<float> values(static_cast<std::size_t>(elements));
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = qwen::tensor_element(type, bytes, i);
            require(std::isfinite(values[i]), "blocks nonfinite actual parameter");
        }
        views.push_back(t);
        return values;
    }
    explicit Data(const qwen::Model& model) {
        const auto& arch = model.metadata_value("general.architecture");
        require(arch.type == qwen::MetadataType::STRING && arch.get<std::string>() == "qwen4exp",
                "blocks requires qwen4exp GGUF");
        hc.hidden_size = unsigned_metadata(model, "qwen4exp.embedding_length");
        hc.branches = unsigned_metadata(model, "qwen4exp.hyper_connection.count");
        hc.low_rank = unsigned_metadata(model, "qwen4exp.hyper_connection.low_rank");
        const auto& epsilon = model.metadata_value("qwen4exp.attention.layer_norm_rms_epsilon");
        require(epsilon.type == qwen::MetadataType::FLOAT32, "blocks RMS epsilon metadata type");
        hc.rms_epsilon = epsilon.get<float>();
        require(hc.hidden_size == hidden && hc.branches == branches && hc.low_rank == rank &&
                hc.widened_elements() == wide, "blocks H2560 C4 rank320 geometry");
        const auto hash = qwen::ple_hash_config(model, 1);
        ple = qwen::ple_layer_config(model, hash);
        require(ple.embedding_dim == hidden && ple.hidden_size == hidden && ple.hc_count == branches &&
                ple.conv_kernel == 4 && ple.ngram_size == 3 && ple.history_length() == 9 &&
                ple.history_elements() == history_size && ple.rms_epsilon == hc.rms_epsilon,
                "blocks actual layer1 PLE geometry");
        hc_gamma[0] = load(model, "blk.0.hc_attn_norm.weight", {wide}, qwen::TensorType::F32);
        hc_gamma[1] = load(model, "blk.0.hc_ffn_norm.weight", {wide}, qwen::TensorType::F32);
        hc_gamma[2] = load(model, "output_hc_norm.weight", {wide}, qwen::TensorType::F32);
        ple_gamma[0] = load(model, "blk.1.ple_norm_key.weight", {wide}, qwen::TensorType::F32);
        ple_gamma[1] = load(model, "blk.1.ple_norm_query.weight", {wide}, qwen::TensorType::F32);
        ple_gamma[2] = load(model, "blk.1.ple_norm_conv.weight", {wide}, qwen::TensorType::F32);
        conv = load(model, "blk.1.ple_conv1d.weight", {4, wide}, qwen::TensorType::F16);
        small_gamma[0] = load(model, "blk.0.ssm_norm.weight", {128}, qwen::TensorType::F32);
        small_gamma[1] = load(model, "blk.3.indexer.q_norm.weight", {128}, qwen::TensorType::F32);
        small_gamma[2] = load(model, "blk.3.indexer.k_norm.weight", {128}, qwen::TensorType::F32);
        small_gamma[3] = load(model, "blk.3.attn_q_norm.weight", {256}, qwen::TensorType::F32);
        small_gamma[4] = load(model, "blk.3.attn_k_norm.weight", {256}, qwen::TensorType::F32);
    }
};

// The following independent equations deliberately never call HcCpu/PleLayerCpu
// or matmul. All raw projections are synthetic SAME CPU/GPU inputs, not results
// of applying model matrices. Expected-math inputs and references remain const.
float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x); return e / (1.0f + e);
}
float silu(float x) { return x * sigmoid(x); }
std::vector<float> synthetic(std::size_t count, unsigned seed, float scale = 1.0f) {
    std::vector<float> v(count);
    for (std::size_t i = 0; i < count; ++i)
        v[i] = float(int((i * 37 + seed * 101 + (i / hidden) * 13) % 257) - 128) * (scale / 128.0f);
    return v;
}
struct Inputs {
    int n;
    std::vector<float> original, down, up, raw_inject, block, key, query, value;
    explicit Inputs(int tokens, unsigned seed = 1, bool large = false) : n(tokens),
        original(synthetic(std::size_t(n) * wide, seed, large ? 1e15f : 1.3f)),
        down(synthetic(std::size_t(n) * rank, seed + 1, large ? 1e30f : 9.0f)),
        up(synthetic(std::size_t(n) * wide, seed + 2, large ? 1e30f : 5.0f)),
        raw_inject(synthetic(std::size_t(n) * branches, seed + 3, large ? 1e30f : 15.0f)),
        block(synthetic(std::size_t(n) * hidden, seed + 4, large ? 1e15f : .7f)),
        key(synthetic(std::size_t(n) * wide, seed + 5, large ? 1e12f : 1.1f)),
        query(key), value(synthetic(std::size_t(n) * hidden, seed + 6, large ? 1e12f : .8f)) {
        for (std::size_t i = 0; i < query.size(); ++i) {
            const float sign = ((i / hidden) % branches) % 2 == 0 ? 1.0f : -1.0f;
            query[i] = sign * query[i] * .6f;
        }
    }
};
std::vector<float> rms(std::span<const float> x, std::span<const float> gamma,
                       int width, int group, float epsilon, bool shared = false) {
    require(x.size() % std::size_t(width) == 0 && width % group == 0 &&
            gamma.size() == std::size_t(shared ? group : width), "CPU RMS shape");
    std::vector<float> out(x.size());
    for (std::size_t base = 0; base < x.size(); base += group) {
        float sum = 0.0f;
        for (int d = 0; d < group; ++d) sum = sum + x[base + d] * x[base + d];
        const float inverse = 1.0f / std::sqrt(sum / float(group) + epsilon);
        for (int d = 0; d < group; ++d)
            out[base + d] = (x[base + d] * inverse) * gamma[(shared ? 0 : base % width) + d];
    }
    return out;
}
std::vector<float> lowrank(std::span<const float> raw) {
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) out[i] = silu(raw[i] / 4.0f);
    return out;
}
std::vector<float> mix(std::span<const float> norm, std::span<const float> up, int n) {
    std::vector<float> out(std::size_t(n) * hidden);
    for (int t = 0; t < n; ++t) for (int d = 0; d < hidden; ++d) {
        float sum = 0.0f;
        for (int b = 0; b < branches; ++b) {
            const int j = t * wide + b * hidden + d;
            sum = sum + sigmoid(up[j]) * norm[j];
        }
        out[t * hidden + d] = sum / 4.0f;
    }
    return out;
}
std::vector<float> inject(std::span<const float> raw) {
    std::vector<float> out(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) out[i] = 2.0f * sigmoid(raw[i] / 4.0f);
    return out;
}
std::vector<float> combine(const Inputs& x, std::span<const float> gates) {
    std::vector<float> out(x.original.size());
    for (int t = 0; t < x.n; ++t) for (int b = 0; b < branches; ++b) for (int d = 0; d < hidden; ++d) {
        const int j = t * wide + b * hidden + d;
        out[j] = x.original[j] + gates[t * branches + b] * x.block[t * hidden + d];
    }
    return out;
}
std::vector<float> gate_values(std::span<const float> key, std::span<const float> query,
                               std::span<const float> value, int n) {
    std::vector<float> out(std::size_t(n) * wide);
    for (int t = 0; t < n; ++t) for (int b = 0; b < branches; ++b) {
        const int base = t * wide + b * hidden;
        float dot = 0.0f;
        for (int d = 0; d < hidden; ++d) dot = dot + key[base + d] * query[base + d];
        const float score = dot / std::sqrt(float(hidden));
        const float sign = score > 0.0f ? 1.0f : score < 0.0f ? -1.0f : 0.0f;
        const float g = sigmoid(sign * std::sqrt(std::max(std::abs(score), 1e-6f)));
        for (int d = 0; d < hidden; ++d) out[base + d] = g * value[t * hidden + d];
    }
    return out;
}
struct ConvReference { std::vector<float> output, history, prefixes; };
ConvReference conv_reference(std::span<const float> gated, std::span<const float> norm,
                              std::span<const float> weights, std::span<const float> initial,
                              int n, std::span<const std::uint8_t> mask = {}, bool prefixes = false) {
    require(initial.size() == history_size && weights.size() == weight_size, "CPU conv shape");
    ConvReference r{std::vector<float>(std::size_t(n) * wide), std::vector<float>(initial.begin(), initial.end()), {}};
    if (prefixes) {
        r.prefixes.resize(std::size_t(n + 1) * history_size);
        std::copy(initial.begin(), initial.end(), r.prefixes.begin());
    }
    for (int t = 0; t < n; ++t) {
        const bool keep = mask.empty() || mask[t] == 1;
        for (int f = 0; f < wide; ++f) {
            const int j = t * wide + f, base = f * 9;
            const float current = keep ? norm[j] : 0.0f;
            float conv = 0.0f;
            for (int k = 0; k < 4; ++k)
                conv = conv + (k == 3 ? current : r.history[base + k * 3]) * weights[f * 4 + k];
            r.output[j] = (keep ? gated[j] : 0.0f) + silu(conv);
            for (int a = 0; a < 8; ++a) r.history[base + a] = r.history[base + a + 1];
            r.history[base + 8] = current;
        }
        if (prefixes) std::copy(r.history.begin(), r.history.end(), r.prefixes.begin() + std::size_t(t + 1) * history_size);
    }
    return r;
}
struct Reference {
    std::vector<float> normal, low, mixed, injection, combined, nk, nq, gated, ng;
    ConvReference conv;
    Reference(const Data& p, const Inputs& x, std::span<const float> initial) :
        normal(rms(x.original, p.hc_gamma[0], wide, hidden, p.hc.rms_epsilon)),
        low(lowrank(x.down)), mixed(mix(normal, x.up, x.n)), injection(inject(x.raw_inject)),
        combined(combine(x, injection)), nk(rms(x.key, p.ple_gamma[0], wide, hidden, p.ple.rms_epsilon)),
        nq(rms(x.query, p.ple_gamma[1], wide, hidden, p.ple.rms_epsilon)),
        gated(gate_values(nk, nq, x.value, x.n)), ng(rms(gated, p.ple_gamma[2], wide, hidden, p.ple.rms_epsilon)),
        conv(conv_reference(gated, ng, p.conv, initial, x.n)) {}
};

struct Buffer { void* base; void* data; std::size_t bytes; };
struct Fixture {
    int device;
    hipStream_t stream{};
    hipEvent_t begin{}, end{};
    std::vector<Buffer> owned;
    float *original{}, *down{}, *up{}, *raw_inject{}, *block{}, *key{}, *query{}, *value{};
    float *gamma{}, *gamma_key{}, *gamma_query{}, *gamma_conv{}, *weights{};
    float *normal{}, *low{}, *mixed{}, *injection{}, *combined{}, *nk{}, *nq{}, *gated{}, *ng{}, *output{};
    float *active{}, *staged{}, *prefixes{};
    std::uint8_t* mask{};
    int* error{};
    // Publication is a caller-owned fixture marker, never advanced on rejection.
    std::uint64_t published_tokens = 0;
    explicit Fixture(int d) : device(d) {}
    ~Fixture() {
        cleanup(hipSetDevice(device));
        if (stream) cleanup(hipStreamSynchronize(stream));
        for (const auto& b : owned) cleanup(hipFree(b.base));
        if (begin) cleanup(hipEventDestroy(begin));
        if (end) cleanup(hipEventDestroy(end));
        if (stream) cleanup(hipStreamDestroy(stream));
    }
    template<class T> T* allocate(std::size_t count) {
        const std::size_t bytes = count * sizeof(T);
        void* base = nullptr; check(hipMalloc(&base, bytes + 2 * guard_bytes));
        auto* data = static_cast<unsigned char*>(base) + guard_bytes;
        try { owned.push_back({base, data, bytes}); }
        catch (...) { cleanup(hipFree(base)); throw; }
        check(hipMemsetAsync(base, guard_value, bytes + 2 * guard_bytes, stream));
        check(hipMemsetAsync(data, unused_value, bytes, stream));
        return reinterpret_cast<T*>(data);
    }
    template<class T> void put(T* dest, std::span<const T> source) {
        check(hipMemcpyAsync(dest, source.data(), source.size_bytes(), hipMemcpyHostToDevice, stream));
        // Fixture uploads are outside timing; completion also protects temporary
        // host byte/vector lifetimes on HIP implementations without staging.
        check(hipStreamSynchronize(stream));
    }
    template<class T> void put(T* dest, const std::vector<T>& source) { put<T>(dest, std::span<const T>(source)); }
    template<class T> std::vector<T> get(const T* source, std::size_t n) {
        std::vector<T> result(n);
        if (n == 0) return result;
        check(hipMemcpyAsync(result.data(), source, n * sizeof(T), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream)); return result;
    }
    void initialize(const Data& p) {
        check(hipSetDevice(device));
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        check(hipEventCreate(&begin)); check(hipEventCreate(&end));
        constexpr std::size_t nw = 128 * wide, nh = 128 * hidden;
        original = allocate<float>(nw); down = allocate<float>(128 * rank); up = allocate<float>(nw);
        raw_inject = allocate<float>(128 * branches); block = allocate<float>(nh);
        key = allocate<float>(nw); query = allocate<float>(nw); value = allocate<float>(nh);
        gamma = allocate<float>(wide); gamma_key = allocate<float>(wide);
        gamma_query = allocate<float>(wide); gamma_conv = allocate<float>(wide); weights = allocate<float>(weight_size);
        normal = allocate<float>(nw); low = allocate<float>(128 * rank); mixed = allocate<float>(nh);
        injection = allocate<float>(128 * branches); combined = allocate<float>(nw);
        nk = allocate<float>(nw); nq = allocate<float>(nw); gated = allocate<float>(nw);
        ng = allocate<float>(nw); output = allocate<float>(nw);
        active = allocate<float>(history_size); staged = allocate<float>(history_size);
        prefixes = allocate<float>(4 * history_size); mask = allocate<std::uint8_t>(128); error = allocate<int>(1);
        parameters(p); clear_error(); check(hipStreamSynchronize(stream));
    }
    void parameters(const Data& p) {
        put(gamma, p.hc_gamma[0]); put(gamma_key, p.ple_gamma[0]);
        put(gamma_query, p.ple_gamma[1]); put(gamma_conv, p.ple_gamma[2]); put(weights, p.conv);
    }
    void upload(const Inputs& x) {
        put(original, x.original); put(down, x.down); put(up, x.up); put(raw_inject, x.raw_inject);
        put(block, x.block); put(key, x.key); put(query, x.query); put(value, x.value);
    }
    void clear_error() { check(hipMemsetAsync(error, 0, sizeof(int), stream)); }
    int status() { return get(error, 1)[0]; }
    void success() { require(status() == 0, "blocks unexpected numeric status"); }
    void reset_output(float* ptr) {
        const auto it = std::find_if(owned.begin(), owned.end(), [ptr](const Buffer& b) { return b.data == ptr; });
        require(it != owned.end(), "blocks output allocation");
        check(hipMemsetAsync(ptr, unused_value, it->bytes, stream));
    }
    void fill_outputs() {
        for (auto* ptr : {normal, low, mixed, injection, combined, nk, nq, gated, ng, output, staged, prefixes}) reset_output(ptr);
    }
    void canaries(Counters& c) {
        for (const auto& b : owned) {
            const auto left = get(static_cast<const unsigned char*>(b.base), guard_bytes);
            const auto right = get(static_cast<const unsigned char*>(b.data) + b.bytes, guard_bytes);
            require(std::all_of(left.begin(), left.end(), [](auto v) { return v == guard_value; }) &&
                    std::all_of(right.begin(), right.end(), [](auto v) { return v == guard_value; }), "blocks byte guard corrupted");
            c.canary_checks += 2;
        }
    }
    void unused(float* ptr, std::size_t used, Counters& c) {
        const auto it = std::find_if(owned.begin(), owned.end(), [ptr](const Buffer& b) { return b.data == ptr; });
        require(it != owned.end() && used * sizeof(float) <= it->bytes, "blocks unused range");
        const auto bytes = get(static_cast<const unsigned char*>(it->data) + used * sizeof(float), it->bytes - used * sizeof(float));
        require(std::all_of(bytes.begin(), bytes.end(), [](auto v) { return v == unused_value; }), "blocks logical output tail changed");
        ++c.canary_checks;
    }
    void immutable(const Inputs& x, const Data& p, Counters& c) {
        auto verify = [&](const float* ptr, const std::vector<float>& ref) {
            exact(get(ptr, ref.size()), ref, "blocks immutable input changed"); ++c.immutable_checks;
        };
        verify(original, x.original); verify(down, x.down); verify(up, x.up); verify(raw_inject, x.raw_inject);
        verify(block, x.block); verify(key, x.key); verify(query, x.query); verify(value, x.value);
        verify(gamma, p.hc_gamma[0]); verify(gamma_key, p.ple_gamma[0]);
        verify(gamma_query, p.ple_gamma[1]); verify(gamma_conv, p.ple_gamma[2]); verify(weights, p.conv);
    }
    qwen::PleConvDeviceWorkspace workspace(bool prefix = false) const { return {output, staged, prefix ? prefixes : nullptr}; }
    void rms_launch(const float* input, const float* g, int n, int width, int group,
                    float epsilon, float* out, bool shared = false) {
        check(qwen::launch_group_rms_norm(input, g, n, width, group, epsilon, shared, out, error, stream));
    }
    void conv_launch(int n, bool use_mask = false, bool prefix = false) {
        check(qwen::launch_ple_causal_conv(gated, ng, weights, active, n, use_mask ? mask : nullptr,
                                         workspace(prefix), error, stream));
    }
    void publish(const float* from) { check(qwen::launch_publish_ple_history(from, active, error, stream)); }
};

void compare(Fixture& f, float* ptr, const std::vector<float>& ref, Gate& g, Counters& c, const char* what) {
    f.success(); g.compare(f.get(ptr, ref.size()), ref, what); f.unused(ptr, ref.size(), c);
}
void check_norm(Fixture& f, const std::vector<float>& input, const std::vector<float>& gamma,
                int n, int width, int group, float epsilon, bool shared, Summary& s) {
    const auto expected = rms(input, gamma, width, group, epsilon, shared);
    f.put(f.original, input); f.put(f.gamma, gamma); f.reset_output(f.normal); f.clear_error();
    f.rms_launch(f.original, f.gamma, n, width, group, epsilon, f.normal, shared);
    compare(f, f.normal, expected, s.errors.rms, s.counters, "isolated RMS parity");
    exact(f.get(f.original, input.size()), input, "RMS input immutable");
    exact(f.get(f.gamma, gamma.size()), gamma, "RMS gamma immutable");
    s.counters.immutable_checks += 2; ++s.counters.norm_cases;
    if (shared) ++s.counters.broadcast_cases;
}

// Identical gated AND normalized-gated inputs permit exact history assertions.
// Output still uses the frozen bounded gate (host/device expf may differ).
ConvReference common_conv(Fixture& f, const Data& p, const std::vector<float>& gated,
                          const std::vector<float>& ng, const std::vector<float>& initial,
                          int n, std::span<const std::uint8_t> mask, bool prefixes,
                          bool publish, Summary& s, bool upload_active = true) {
    const auto ref = conv_reference(gated, ng, p.conv, initial, n, mask, prefixes);
    f.put(f.gated, gated); f.put(f.ng, ng);
    if (upload_active) f.put(f.active, initial);
    if (!mask.empty()) f.put<std::uint8_t>(f.mask, mask);
    f.reset_output(f.output); f.reset_output(f.staged);
    if (prefixes) f.reset_output(f.prefixes);
    f.clear_error(); f.conv_launch(n, !mask.empty(), prefixes); f.success();
    compare(f, f.output, ref.output, s.errors.ple, s.counters, "common-input conv output");
    const auto staged = f.get(f.staged, history_size);
    s.errors.history.compare(staged, ref.history, "common-input conv history");
    exact(staged, ref.history, "identical norm_gated requires exact staged history");
    exact(f.get(f.active, history_size), initial, "conv changed active before publication");
    ++s.counters.readonly_active_checks;
    if (prefixes) {
        const auto got = f.get(f.prefixes, ref.prefixes.size());
        s.errors.history.compare(got, ref.prefixes, "chronological conv prefixes");
        exact(got, ref.prefixes, "identical norm_gated requires exact prefix tails");
        f.unused(f.prefixes, ref.prefixes.size(), s.counters);
        s.counters.prefix_slots += std::size_t(n + 1);
    }
    exact(f.get(f.gated, gated.size()), gated, "conv gated immutable");
    exact(f.get(f.ng, ng.size()), ng, "conv norm_gated immutable");
    exact(f.get(f.weights, p.conv.size()), p.conv, "conv weights immutable");
    s.counters.immutable_checks += 3;
    if (!mask.empty()) {
        exact<std::uint8_t>(f.get(f.mask, mask.size()), mask, "conv mask immutable");
        ++s.counters.immutable_checks; ++s.counters.mask_cases;
    }
    ++s.counters.conv_cases;
    if (publish) {
        f.publish(f.staged); f.success();
        exact(f.get(f.active, history_size), ref.history, "common-input publication history");
        f.published_tokens += std::uint64_t(n); ++s.counters.publication_checks;
    }
    f.canaries(s.counters);
    return ref;
}

void isolated(Fixture& f, const Data& p, int n, Summary& s, bool large = false) {
    const Inputs x(n, 1, large);
    const auto initial = synthetic(history_size, 19, .25f);
    const Reference r(p, x, initial);
    for (const auto& gamma : p.hc_gamma)
        check_norm(f, x.original, gamma, n, wide, hidden, p.hc.rms_epsilon, false, s);
    check_norm(f, x.key, p.ple_gamma[0], n, wide, hidden, p.ple.rms_epsilon, false, s);
    check_norm(f, x.query, p.ple_gamma[1], n, wide, hidden, p.ple.rms_epsilon, false, s);
    check_norm(f, r.gated, p.ple_gamma[2], n, wide, hidden, p.ple.rms_epsilon, false, s);
    if (!large) {
        // Cover every supported broadcast shape using actual QSA/GDN gamma.
        for (std::size_t k = 0; k < p.small_gamma.size(); ++k) {
            const int group = k < 3 ? 128 : 256;
            for (int width : (group == 128 ? std::vector<int>{128, 512} : std::vector<int>{256, 512, 6144})) {
                const auto input = synthetic(std::size_t(n) * width, unsigned(k + 33));
                check_norm(f, input, p.small_gamma[k], n, width, group, p.hc.rms_epsilon, true, s);
            }
        }
        const std::vector<float> gamma(p.hc_gamma[0].begin(), p.hc_gamma[0].begin() + hidden);
        for (int width : {hidden, wide}) {
            const auto input = synthetic(std::size_t(n) * width, 39);
            check_norm(f, input, gamma, n, width, hidden, p.hc.rms_epsilon, true, s);
        }
    }
    f.upload(x); f.parameters(p); f.fill_outputs(); f.clear_error();
    check(qwen::launch_hc_lowrank_activation(f.down, n, f.low, f.error, f.stream));
    compare(f, f.low, r.low, s.errors.hc, s.counters, "isolated HC low SiLU(raw/4)");
    ++s.counters.activation_cases;
    f.put(f.normal, r.normal);
    check(qwen::launch_hc_mix(f.normal, f.up, n, f.mixed, f.error, f.stream));
    compare(f, f.mixed, r.mixed, s.errors.hc, s.counters, "isolated HC mean4 sigmoid mix");
    exact(f.get(f.normal, r.normal.size()), r.normal, "HC normalized input immutable");
    ++s.counters.immutable_checks; ++s.counters.mix_cases;
    check(qwen::launch_hc_injection(f.raw_inject, n, f.injection, f.error, f.stream));
    compare(f, f.injection, r.injection, s.errors.hc, s.counters, "isolated HC 2sigmoid(raw/4)");
    ++s.counters.injection_cases;
    // Isolate combine from preceding sigmoid rounding too.
    f.put(f.injection, r.injection);
    check(qwen::launch_hc_combine(f.original, f.block, f.injection, n, f.combined, f.error, f.stream));
    compare(f, f.combined, r.combined, s.errors.hc, s.counters, "original wide plus injection times block");
    exact(f.get(f.injection, r.injection.size()), r.injection, "combine injection immutable");
    ++s.counters.immutable_checks; ++s.counters.combine_cases;
    // Root head: normalization/mix only, no injection or combine. The original
    // full widened residual (MTP tap) is checked unchanged, not replaced by mix.
    const auto head_norm = rms(x.original, p.hc_gamma[2], wide, hidden, p.hc.rms_epsilon);
    const auto head = mix(head_norm, x.up, n);
    const auto old_injection = f.get(f.injection, r.injection.size());
    const auto old_combined = f.get(f.combined, r.combined.size());
    f.put(f.gamma, p.hc_gamma[2]); f.reset_output(f.normal); f.reset_output(f.mixed);
    f.rms_launch(f.original, f.gamma, n, wide, hidden, p.hc.rms_epsilon, f.normal);
    compare(f, f.normal, head_norm, s.errors.rms, s.counters, "root head RMS");
    check(qwen::launch_hc_mix(f.normal, f.up, n, f.mixed, f.error, f.stream));
    compare(f, f.mixed, head, s.errors.hc, s.counters, "root head mix without injection");
    exact(f.get(f.original, x.original.size()), x.original, "root full widened MTP tap unchanged");
    exact(f.get(f.injection, old_injection.size()), old_injection, "root head touched injection");
    exact(f.get(f.combined, old_combined.size()), old_combined, "root head touched combined residual");
    s.counters.immutable_checks += 3; ++s.counters.head_cases;
    f.put(f.gamma, p.hc_gamma[0]);
    f.put(f.nk, r.nk); f.put(f.nq, r.nq);
    check(qwen::launch_ple_gate(f.nk, f.nq, f.value, n, f.gated, f.error, f.stream));
    compare(f, f.gated, r.gated, s.errors.ple, s.counters, "isolated signed-root PLE gate");
    exact(f.get(f.nk, r.nk.size()), r.nk, "PLE normalized key immutable");
    exact(f.get(f.nq, r.nq.size()), r.nq, "PLE normalized query immutable");
    s.counters.immutable_checks += 2; ++s.counters.gate_cases;
    if (large) {
        // Isolated gate-interface stress, SAME finite synthetic inputs on both
        // paths: large positive/negative scores must use stable sigmoid. This
        // does not pretend these large values came from an actual projection.
        const auto large_gate = gate_values(x.key, x.query, x.value, n);
        f.put(f.nk, x.key); f.put(f.nq, x.query); f.reset_output(f.gated);
        check(qwen::launch_ple_gate(f.nk, f.nq, f.value, n, f.gated, f.error, f.stream));
        compare(f, f.gated, large_gate, s.errors.ple, s.counters, "large finite signed PLE scores");
        exact(f.get(f.nk, x.key.size()), x.key, "large gate key immutable");
        exact(f.get(f.nq, x.query.size()), x.query, "large gate query immutable");
        s.counters.immutable_checks += 2; ++s.counters.gate_cases;
    }
    static_cast<void>(common_conv(f, p, r.gated, r.ng, initial, n, {}, n <= 3, true, s));
    f.immutable(x, p, s.counters); f.canaries(s.counters);
    if (large) ++s.counters.large_finite_cases;
}

void zero_and_alias(Fixture& f, const Data& p, Summary& s) {
    const Inputs x(3, 47);
    const auto initial = synthetic(history_size, 49, .1f);
    const Reference r(p, x, initial);
    f.upload(x); f.parameters(p); f.fill_outputs(); f.clear_error();
    for (float zero : {0.0f, -0.0f}) {
        const std::vector<float> key(std::size_t(x.n) * wide, zero);
        const std::vector<float> query(key.size(), 1.0f);
        const auto expected = gate_values(key, query, x.value, x.n);
        // In addition to bounded parity, zero-score output is exactly value/2.
        for (int t = 0; t < x.n; ++t) for (int b = 0; b < branches; ++b) for (int d = 0; d < hidden; ++d)
            require(expected[t * wide + b * hidden + d] == .5f * x.value[t * hidden + d], "CPU signed-zero gate");
        f.put(f.nk, key); f.put(f.nq, query); f.reset_output(f.gated);
        check(qwen::launch_ple_gate(f.nk, f.nq, f.value, x.n, f.gated, f.error, f.stream));
        compare(f, f.gated, expected, s.errors.ple, s.counters, "signed-zero gate=0.5");
        exact(f.get(f.gated, expected.size()), expected, "signed-zero exact half-value");
        exact(f.get(f.nk, key.size()), key, "signed-zero input bits immutable");
        ++s.counters.signed_zero_cases; ++s.counters.gate_cases; ++s.counters.immutable_checks;
    }
    f.put(f.normal, r.normal); f.reset_output(f.mixed);
    const auto alias_mix = mix(r.normal, r.normal, x.n);
    check(qwen::launch_hc_mix(f.normal, f.normal, x.n, f.mixed, f.error, f.stream));
    compare(f, f.mixed, alias_mix, s.errors.hc, s.counters, "read-only HC alias accepted");
    exact(f.get(f.normal, r.normal.size()), r.normal, "read-only HC alias immutable");
    f.put(f.nk, r.nk); f.reset_output(f.gated);
    const auto alias_gate = gate_values(r.nk, r.nk, x.value, x.n);
    check(qwen::launch_ple_gate(f.nk, f.nk, f.value, x.n, f.gated, f.error, f.stream));
    compare(f, f.gated, alias_gate, s.errors.ple, s.counters, "read-only PLE alias accepted");
    exact(f.get(f.nk, r.nk.size()), r.nk, "read-only PLE alias immutable");
    s.counters.readonly_alias_cases += 2; s.counters.immutable_checks += 2;
    f.immutable(x, p, s.counters); f.canaries(s.counters);
}

void transactions(Fixture& f, const Data& p, Summary& s) {
    std::vector<float> warm(history_size, 0.0f);
    f.published_tokens = 0;
    // Two genuine warm windows populate all nine ages before verify3.
    for (unsigned window = 0; window < 2; ++window) {
        const Inputs x(6, 61 + window);
        const Reference r(p, x, warm);
        const auto ref = common_conv(f, p, r.gated, r.ng, warm, 6, {}, false, true, s);
        warm = ref.history;
    }
    require(f.published_tokens == 12, "warm publication marker");
    for (unsigned window = 0; window < 2; ++window) {
        const Inputs verify(3, 71 + window), replacement(1, 81 + window);
        const Reference v(p, verify, warm), replacement_ref(p, replacement, warm);
        const std::array<std::uint8_t, 3> mask = window == 0 ? std::array<std::uint8_t, 3>{1, 0, 1} :
                                                                    std::array<std::uint8_t, 3>{0, 1, 0};
        const std::uint64_t base_count = f.published_tokens;
        const auto chunk = common_conv(f, p, v.gated, v.ng, warm, 3, mask, true, false, s);
        require(f.published_tokens == base_count, "unpublished verify advanced marker");
        int attempt = 0;
        for (int accepted : {0, 1, 2, 0, 0}) {
            const int slot = 1 + accepted;
            const std::vector<float> tail(chunk.prefixes.begin() + std::size_t(slot) * history_size,
                                          chunk.prefixes.begin() + std::size_t(slot + 1) * history_size);
            f.publish(f.prefixes + std::size_t(slot) * history_size); f.success();
            exact(f.get(f.active, history_size), tail, "restore slot 1+accepted full tail");
            f.published_tokens = base_count + std::uint64_t(slot);
            ++s.counters.restore_cases; ++s.counters.publication_checks;
            const auto replaced = common_conv(f, p, replacement_ref.gated, replacement_ref.ng, tail,
                                             1, {}, false, true, s, false);
            require(f.published_tokens == base_count + std::uint64_t(slot + 1), "replacement publication marker");
            if (attempt >= 3) ++s.counters.repeated_accept0_cases;
            ++attempt;
            // Prefix storage must survive replacement; check all four full tails.
            exact(f.get(f.prefixes, chunk.prefixes.size()), chunk.prefixes, "replacement destroyed verify prefix storage");
            ++s.counters.immutable_checks;
            if (accepted == 0) warm = replaced.history;
        }
    }
    require(s.counters.restore_cases == 10 && s.counters.prefix_slots >= 8, "transaction coverage");
}

void hc_chain(Fixture& f, const Data& p, int n) {
    f.rms_launch(f.original, f.gamma, n, wide, hidden, p.hc.rms_epsilon, f.normal);
    check(qwen::launch_hc_lowrank_activation(f.down, n, f.low, f.error, f.stream));
    check(qwen::launch_hc_mix(f.normal, f.up, n, f.mixed, f.error, f.stream));
    check(qwen::launch_hc_injection(f.raw_inject, n, f.injection, f.error, f.stream));
    check(qwen::launch_hc_combine(f.original, f.block, f.injection, n, f.combined, f.error, f.stream));
}
void ple_chain(Fixture& f, const Data& p, int n) {
    f.rms_launch(f.key, f.gamma_key, n, wide, hidden, p.ple.rms_epsilon, f.nk);
    f.rms_launch(f.query, f.gamma_query, n, wide, hidden, p.ple.rms_epsilon, f.nq);
    check(qwen::launch_ple_gate(f.nk, f.nq, f.value, n, f.gated, f.error, f.stream));
    f.rms_launch(f.gated, f.gamma_conv, n, wide, hidden, p.ple.rms_epsilon, f.ng);
    f.conv_launch(n); f.publish(f.staged);
}
void validate_hc(Fixture& f, const Reference& r, Summary& s) {
    compare(f, f.normal, r.normal, s.errors.rms, s.counters, "HC chain parallel RMS");
    compare(f, f.low, r.low, s.errors.hc, s.counters, "HC chain lowrank");
    compare(f, f.mixed, r.mixed, s.errors.hc, s.counters, "HC chain mixed");
    compare(f, f.injection, r.injection, s.errors.hc, s.counters, "HC chain injection");
    compare(f, f.combined, r.combined, s.errors.hc, s.counters, "HC chain original-wide combine");
}
void validate_ple(Fixture& f, const Reference& r, Summary& s) {
    compare(f, f.nk, r.nk, s.errors.rms, s.counters, "PLE chain key RMS");
    compare(f, f.nq, r.nq, s.errors.rms, s.counters, "PLE chain query RMS");
    compare(f, f.gated, r.gated, s.errors.ple, s.counters, "PLE chain signed-root gate");
    compare(f, f.ng, r.ng, s.errors.rms, s.counters, "PLE chain conv RMS");
    compare(f, f.output, r.conv.output, s.errors.ple, s.counters, "PLE chain convolution");
    const auto staged = f.get(f.staged, history_size), active = f.get(f.active, history_size);
    // GPU's own parallel-RMS state versus ascending CPU state: BOUNDED only.
    s.errors.history.compare(staged, r.conv.history, "PLE own parallel RMS staged history");
    s.errors.history.compare(active, r.conv.history, "PLE own parallel RMS published history");
    // Publication is an exact COPY of that SAME GPU norm_gated history.
    exact(active, staged, "publication differs from its identical staged input");
    ++s.counters.publication_checks;
}

// Failure-only operands are separate mutable test vectors. Never corrupt or
// replace immutable expected-math inputs or the model parameters in Data.
void bad_float(Fixture& f, float* ptr, std::size_t count, std::size_t index, float bad) {
    auto failure_input = f.get(ptr, count); require(index < count, "numeric fixture index");
    failure_input[index] = bad; f.put(ptr, failure_input);
}
struct InputBytes { const unsigned char* pointer; std::vector<unsigned char> bytes; };
std::vector<InputBytes> snapshot_inputs(Fixture& f) {
    std::vector<InputBytes> result;
    for (void* p : std::array<void*, 14>{f.original, f.down, f.up, f.raw_inject, f.block, f.key, f.query,
                                       f.value, f.gamma, f.gamma_key, f.gamma_query, f.gamma_conv, f.weights, f.mask}) {
        const auto it = std::find_if(f.owned.begin(), f.owned.end(), [p](const Buffer& b) { return b.data == p; });
        require(it != f.owned.end(), "numeric input allocation");
        const auto* bytes = static_cast<const unsigned char*>(p);
        result.push_back({bytes, f.get(bytes, it->bytes)});
    }
    return result;
}
void check_snapshot(Fixture& f, const std::vector<InputBytes>& before, Counters& c) {
    for (const auto& b : before) {
        exact(f.get(b.pointer, b.bytes.size()), b.bytes, "failure mutated read-only input bytes");
        ++c.immutable_checks;
    }
}
void provisional_finite(Fixture& f) {
    // These are written by convolution, not intentionally bad borrowed inputs.
    for (float* ptr : {f.output, f.staged, f.prefixes}) {
        const auto it = std::find_if(f.owned.begin(), f.owned.end(), [ptr](const Buffer& b) { return b.data == ptr; });
        require(it != f.owned.end(), "provisional allocation");
        const auto values = f.get(ptr, it->bytes / sizeof(float));
        require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }), "nonfinite provisional helper output");
    }
}
void pointwise_finite(Fixture& f, float* output, std::size_t elements) {
    const auto values = f.get(output, elements);
    require(std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); }), "nonfinite provisional pointwise output");
}

void numeric_cases(Fixture& f, const Data& p, Summary& s) {
    const Inputs x(3, 91);
    const auto initial = synthetic(history_size, 92, .125f);
    const Reference r(p, x, initial);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const float max = std::numeric_limits<float>::max();
    auto initialize = [&] {
        f.upload(x); f.parameters(p); f.fill_outputs();
        f.put(f.normal, r.normal); f.put(f.injection, r.injection);
        f.put(f.nk, r.nk); f.put(f.nq, r.nq); f.put(f.gated, r.gated); f.put(f.ng, r.ng);
        f.put(f.active, initial); f.clear_error();
    };
    auto reuse = [&] {
        f.upload(x); f.parameters(p); f.fill_outputs(); f.clear_error();
        hc_chain(f, p, x.n); validate_hc(f, r, s);
        f.put(f.active, initial); ple_chain(f, p, x.n); validate_ple(f, r, s);
        f.published_tokens += std::uint64_t(x.n);
        f.immutable(x, p, s.counters); f.canaries(s.counters); ++s.counters.valid_reuses;
    };
    auto reject = [&](int bits, auto prepare, auto producer, bool masked = false, bool staged_only = false) {
        initialize(); prepare();
        const auto active_before = f.get(f.active, history_size);
        const auto marker_before = f.published_tokens;
        const auto input_before = snapshot_inputs(f);
        producer();
        if (!staged_only) f.conv_launch(x.n, masked, true);
        check(hipStreamSynchronize(f.stream));
        exact(f.get(f.active, history_size), active_before, "rejected conv changed active before publication");
        ++s.counters.readonly_active_checks;
        f.publish(f.staged);
        require(f.status() == bits, "numeric rejection sticky bits");
        if (bits == 1) ++s.counters.numeric_invalid_cases;
        else if (bits == 2) ++s.counters.numeric_arithmetic_cases;
        else if (bits == 3) ++s.counters.numeric_combined_cases;
        exact(f.get(f.active, history_size), active_before, "rejection changed ENTIRE active history");
        require(f.published_tokens == marker_before, "rejection changed publication marker");
        ++s.counters.numeric_rejects; ++s.counters.rejected_publications;
        check_snapshot(f, input_before, s.counters);
        if (!staged_only) provisional_finite(f);
        f.canaries(s.counters);
        reuse();
    };
    const auto nothing = [] {};
    auto norm = [&] {
        f.rms_launch(f.original, f.gamma, x.n, wide, hidden, p.hc.rms_epsilon, f.normal);
        pointwise_finite(f, f.normal, r.normal.size());
    };
    auto low = [&] {
        check(qwen::launch_hc_lowrank_activation(f.down, x.n, f.low, f.error, f.stream));
        pointwise_finite(f, f.low, r.low.size());
    };
    auto mixed = [&] {
        check(qwen::launch_hc_mix(f.normal, f.up, x.n, f.mixed, f.error, f.stream));
        pointwise_finite(f, f.mixed, r.mixed.size());
    };
    auto injection = [&] {
        check(qwen::launch_hc_injection(f.raw_inject, x.n, f.injection, f.error, f.stream));
        pointwise_finite(f, f.injection, r.injection.size());
    };
    auto combined = [&] {
        check(qwen::launch_hc_combine(f.original, f.block, f.injection, x.n, f.combined, f.error, f.stream));
        pointwise_finite(f, f.combined, r.combined.size());
    };
    auto gate = [&] {
        check(qwen::launch_ple_gate(f.nk, f.nq, f.value, x.n, f.gated, f.error, f.stream));
        pointwise_finite(f, f.gated, r.gated.size());
    };
    for (float bad : {nan, inf}) {
        reject(1, [&] { bad_float(f, f.original, x.original.size(), x.original.size() - 1, bad); }, norm);
        reject(1, [&] { bad_float(f, f.gamma, wide, wide - 1, bad); }, norm);
        reject(1, [&] { bad_float(f, f.down, x.down.size(), x.down.size() - 1, bad); }, low);
        reject(1, [&] { bad_float(f, f.normal, r.normal.size(), r.normal.size() - 1, bad); }, mixed);
        reject(1, [&] { bad_float(f, f.up, x.up.size(), x.up.size() - 1, bad); }, mixed);
        reject(1, [&] { bad_float(f, f.raw_inject, x.raw_inject.size(), x.raw_inject.size() - 1, bad); }, injection);
        reject(1, [&] { bad_float(f, f.original, x.original.size(), x.original.size() - 1, bad); }, combined);
        reject(1, [&] { bad_float(f, f.block, x.block.size(), x.block.size() - 1, bad); }, combined);
        reject(1, [&] { bad_float(f, f.injection, r.injection.size(), r.injection.size() - 1, bad); }, combined);
        reject(1, [&] { bad_float(f, f.nk, r.nk.size(), r.nk.size() - 1, bad); }, gate);
        reject(1, [&] { bad_float(f, f.nq, r.nq.size(), r.nq.size() - 1, bad); }, gate);
        reject(1, [&] { bad_float(f, f.value, x.value.size(), x.value.size() - 1, bad); }, gate);
        reject(1, [&] { bad_float(f, f.gated, r.gated.size(), r.gated.size() - 1, bad); }, nothing);
        reject(1, [&] { bad_float(f, f.ng, r.ng.size(), r.ng.size() - 1, bad); }, nothing);
        reject(1, [&] { bad_float(f, f.weights, weight_size, weight_size - 1, bad); }, nothing);
        reject(1, [&] { bad_float(f, f.active, history_size, history_size - 1, bad); }, nothing);
        // Standalone publish validates the LAST element before any active write.
        reject(1, [&] {
            f.put(f.staged, initial); bad_float(f, f.staged, history_size, history_size - 1, bad);
        }, nothing, false, true);
    }
    reject(2, [&] { bad_float(f, f.original, x.original.size(), x.original.size() - 1, max); }, norm);
    reject(2, [&] {
        f.put(f.normal, std::vector<float>(r.normal.size(), max));
        f.put(f.up, std::vector<float>(x.up.size(), 100.0f));
    }, mixed);
    for (float bad_gate : {-0.01f, 2.01f})
        reject(1, [&] { bad_float(f, f.injection, r.injection.size(), r.injection.size() - 1, bad_gate); }, combined);
    reject(2, [&] {
        bad_float(f, f.original, x.original.size(), x.original.size() - 1, max);
        bad_float(f, f.block, x.block.size(), x.block.size() - 1, max);
        bad_float(f, f.injection, r.injection.size(), r.injection.size() - 1, 2.0f);
    }, combined);
    reject(2, [&] {
        bad_float(f, f.nk, r.nk.size(), r.nk.size() - 1, max);
        bad_float(f, f.nq, r.nq.size(), r.nq.size() - 1, max);
    }, gate);
    // Late final-token/final-feature overflow must reject the WHOLE chunk.
    reject(2, [&] {
        bad_float(f, f.ng, r.ng.size(), r.ng.size() - 1, 4.0f);
        bad_float(f, f.weights, weight_size, weight_size - 1, max);
    }, nothing);
    reject(1, [&] {
        const std::vector<std::uint8_t> invalid_mask{1, 1, 2}; f.put(f.mask, invalid_mask);
    }, nothing, true);
    // keep=0 does not excuse a nonfinite input, and does not reset warm history.
    reject(1, [&] {
        const std::vector<std::uint8_t> zero_mask{1, 1, 0}; f.put(f.mask, zero_mask);
        bad_float(f, f.ng, r.ng.size(), r.ng.size() - 1, nan);
    }, nothing, true);
    for (int sticky : {1, 2, 3})
        reject(sticky, [&] { f.put<int>(f.error, std::span<const int>(&sticky, 1)); }, nothing);
    require(s.counters.numeric_rejects == 46 && s.counters.rejected_publications == 46 &&
            s.counters.valid_reuses == 46, "numeric rejection/reuse coverage");
}

struct HostArg { void* pointer; bool writable; bool optional = false; std::size_t alignment = 4; };
void host_reject(hipError_t status, Counters& c) {
    require(status == hipErrorInvalidValue, "host validation expected hipErrorInvalidValue");
    check(hipGetLastError()); ++c.host_rejects;
}
template<std::size_t N, class Call>
void pointer_rejections(const std::array<HostArg, N>& args, Call call, Counters& c) {
    std::array<void*, N> good{};
    for (std::size_t i = 0; i < N; ++i) good[i] = args[i].pointer;
    for (std::size_t i = 0; i < N; ++i) {
        if (!args[i].optional) {
            auto bad = good; bad[i] = nullptr; host_reject(call(bad), c);
        }
        if (args[i].alignment > 1) {
            auto bad = good;
            bad[i] = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(good[i]) + 1);
            host_reject(call(bad), c);
        }
        auto bad = good;
        const auto max = std::numeric_limits<std::uintptr_t>::max();
        bad[i] = reinterpret_cast<void*>(max - (args[i].alignment > 1 ? args[i].alignment - 1 : 1));
        host_reject(call(bad), c);
    }
    // Every writable argument must reject overlapping every other argument,
    // including error/scratch/prefix/active/mask, not merely input==output.
    for (std::size_t i = 0; i < N; ++i) if (args[i].writable)
        for (std::size_t j = 0; j < N; ++j) if (i != j) {
            auto bad = good; bad[i] = good[j]; host_reject(call(bad), c);
        }
}
void host_cases(Fixture& f, const Data& p, Summary& s) {
    auto& c = s.counters;
    check(hipGetLastError());
    const auto F = [](void* p) { return static_cast<float*>(p); };
    const auto I = [](void* p) { return static_cast<int*>(p); };
    pointer_rejections(std::array<HostArg, 4>{{{f.original, false}, {f.gamma, false}, {f.normal, true}, {f.error, true}}},
        [&](const auto& a) { return qwen::launch_group_rms_norm(F(a[0]), F(a[1]), 3, wide, hidden,
            p.hc.rms_epsilon, false, F(a[2]), I(a[3]), f.stream); }, c);
    pointer_rejections(std::array<HostArg, 3>{{{f.down, false}, {f.low, true}, {f.error, true}}},
        [&](const auto& a) { return qwen::launch_hc_lowrank_activation(F(a[0]), 3, F(a[1]), I(a[2]), f.stream); }, c);
    pointer_rejections(std::array<HostArg, 4>{{{f.normal, false}, {f.up, false}, {f.mixed, true}, {f.error, true}}},
        [&](const auto& a) { return qwen::launch_hc_mix(F(a[0]), F(a[1]), 3, F(a[2]), I(a[3]), f.stream); }, c);
    pointer_rejections(std::array<HostArg, 3>{{{f.raw_inject, false}, {f.injection, true}, {f.error, true}}},
        [&](const auto& a) { return qwen::launch_hc_injection(F(a[0]), 3, F(a[1]), I(a[2]), f.stream); }, c);
    pointer_rejections(std::array<HostArg, 5>{{{f.original, false}, {f.block, false}, {f.injection, false}, {f.combined, true}, {f.error, true}}},
        [&](const auto& a) { return qwen::launch_hc_combine(F(a[0]), F(a[1]), F(a[2]), 3, F(a[3]), I(a[4]), f.stream); }, c);
    pointer_rejections(std::array<HostArg, 5>{{{f.nk, false}, {f.nq, false}, {f.value, false}, {f.gated, true}, {f.error, true}}},
        [&](const auto& a) { return qwen::launch_ple_gate(F(a[0]), F(a[1]), F(a[2]), 3, F(a[3]), I(a[4]), f.stream); }, c);
    pointer_rejections(std::array<HostArg, 9>{{{f.gated, false}, {f.ng, false}, {f.weights, false}, {f.active, false},
        {f.mask, false, true, 1}, {f.output, true}, {f.staged, true}, {f.prefixes, true, true}, {f.error, true}}},
        [&](const auto& a) { return qwen::launch_ple_causal_conv(F(a[0]), F(a[1]), F(a[2]), F(a[3]), 3,
            static_cast<std::uint8_t*>(a[4]), {F(a[5]), F(a[6]), F(a[7])}, I(a[8]), f.stream); }, c);
    pointer_rejections(std::array<HostArg, 3>{{{f.staged, false}, {f.active, true}, {f.error, true}}},
        [&](const auto& a) { return qwen::launch_publish_ple_history(F(a[0]), F(a[1]), I(a[2]), f.stream); }, c);
    for (int n : {-1, 0, 129, std::numeric_limits<int>::max(), std::numeric_limits<int>::min()}) {
        host_reject(qwen::launch_group_rms_norm(f.original, f.gamma, n, wide, hidden, p.hc.rms_epsilon,
                                               false, f.normal, f.error, f.stream), c);
        host_reject(qwen::launch_hc_lowrank_activation(f.down, n, f.low, f.error, f.stream), c);
        host_reject(qwen::launch_hc_mix(f.normal, f.up, n, f.mixed, f.error, f.stream), c);
        host_reject(qwen::launch_hc_injection(f.raw_inject, n, f.injection, f.error, f.stream), c);
        host_reject(qwen::launch_hc_combine(f.original, f.block, f.injection, n, f.combined, f.error, f.stream), c);
        host_reject(qwen::launch_ple_gate(f.nk, f.nq, f.value, n, f.gated, f.error, f.stream), c);
        host_reject(qwen::launch_ple_causal_conv(f.gated, f.ng, f.weights, f.active, n, f.mask,
                                              f.workspace(true), f.error, f.stream), c);
    }
    for (float epsilon : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
        host_reject(qwen::launch_group_rms_norm(f.original, f.gamma, 3, wide, hidden, epsilon,
                                               false, f.normal, f.error, f.stream), c);
    for (auto shape : std::array<std::array<int, 2>, 8>{{{0, hidden}, {wide, 0}, {wide, -1}, {-1, hidden},
            {wide - 1, hidden}, {wide, 128}, {512, 64}, {std::numeric_limits<int>::max(), hidden}}})
        host_reject(qwen::launch_group_rms_norm(f.original, f.gamma, 3, shape[0], shape[1], p.hc.rms_epsilon,
                                               false, f.normal, f.error, f.stream), c);
    require(c.host_rejects == 224, "host rejection coverage");
    f.canaries(c);
}

void coverage(const Counters& c) {
    require(c.norm_cases == 86 && c.activation_cases == 5 && c.mix_cases == 5 &&
            c.injection_cases == 5 && c.combine_cases == 5 && c.gate_cases == 8 &&
            c.conv_cases == 19 && c.head_cases == 5 && c.broadcast_cases == 56 &&
            c.signed_zero_cases == 2 && c.large_finite_cases == 1 && c.prefix_slots == 21 &&
            c.restore_cases == 10 && c.repeated_accept0_cases == 4 && c.mask_cases == 2 &&
            c.readonly_active_checks == 65 && c.publication_checks == 73 && c.rejected_publications == 46 &&
            c.numeric_rejects == 46 && c.host_rejects == 224 && c.readonly_alias_cases == 2 &&
            c.numeric_invalid_cases == 39 && c.numeric_arithmetic_cases == 6 && c.numeric_combined_cases == 1 &&
            c.valid_reuses == 46 && c.immutable_checks > 0 && c.canary_checks > 0,
            "blocks incomplete correctness coverage");
}
Summary qualify(Fixture& f, const Data& p) {
    Summary result;
    host_cases(f, p, result);
    for (int n : token_counts) isolated(f, p, n, result);
    isolated(f, p, 3, result, true);
    zero_and_alias(f, p, result);
    transactions(f, p, result);
    numeric_cases(f, p, result);
    coverage(result.counters);
    return result;
}

void source_json(const qwen::Model& model, const Data& p, std::string_view path,
                 const std::array<hipDeviceProp_t, 2>& devices) {
    std::cout << "{\"kind\":\"blocks_source\",\"protocol\":1,\"scope\":\"isolated_resident_fp32_primitives_same_synthetic_projections_actual_gamma_conv\",\"revision\":";
#ifdef CORE_REVISION
    json_string(CORE_REVISION);
#else
    std::cout << "null";
#endif
    std::cout << ",\"dirty\":";
#ifdef CORE_DIRTY
    std::cout << (CORE_DIRTY ? "true" : "false");
#else
    std::cout << "null";
#endif
    std::cout << ",\"build_provenance_complete\":";
#if defined(CORE_REVISION) && defined(CORE_DIRTY)
    std::cout << "true";
#else
    std::cout << "false";
#endif
    std::cout << ",\"model_path\":"; json_string(path);
    std::cout << ",\"model_file_bytes\":" << model.file_size()
              << ",\"actual_metadata\":{\"architecture\":\"qwen4exp\",\"hidden\":" << p.hc.hidden_size
              << ",\"branches\":" << p.hc.branches << ",\"rank\":" << p.hc.low_rank
              << ",\"rms_epsilon\":" << p.hc.rms_epsilon << ",\"rms_epsilon_type\":\"FLOAT32\""
              << ",\"ple_layer\":1,\"ple_embedding_dim\":" << p.ple.embedding_dim
              << ",\"ple_conv_kernel\":" << p.ple.conv_kernel << ",\"ple_ngram_size\":" << p.ple.ngram_size
              << ",\"ple_history_length\":" << p.ple.history_length() << "},\"actual_tensors\":[";
    for (std::size_t i = 0; i < p.views.size(); ++i) {
        if (i) std::cout << ',';
        const auto& v = p.views[i];
        std::cout << "{\"name\":"; json_string(v.name);
        std::cout << ",\"type\":"; json_string(qwen::type_name(v.type));
        std::cout << ",\"rank\":" << v.rank << ",\"dimensions\":[";
        for (std::size_t j = 0; j < v.rank; ++j) { if (j) std::cout << ','; std::cout << v.dimensions[j]; }
        std::cout << "],\"strides_bytes\":[";
        for (std::size_t j = 0; j < v.rank; ++j) { if (j) std::cout << ','; std::cout << v.strides[j]; }
        std::cout << "],\"elements\":" << v.elements << ",\"bytes\":" << v.byte_size
                  << ",\"relative_offset\":" << v.relative_offset << ",\"file_offset\":" << v.file_offset << '}';
    }
    std::cout << "],\"devices\":[";
    for (int d = 0; d < 2; ++d) {
        if (d) std::cout << ',';
        std::cout << "{\"device\":" << d << ",\"arch\":"; json_string(devices[d].gcnArchName);
        std::cout << ",\"wave_size\":" << devices[d].warpSize << '}';
    }
    std::cout << "],\"gates\":{\"frozen_before_gpu\":true,\"absolute\":" << absolute_gate
              << ",\"relative_cpu_magnitude\":" << relative_gate
              << ",\"groups\":[\"rms\",\"hc\",\"ple\",\"history\"],\"max_bound_ratio_limit\":1"
              << ",\"finite_required\":true,\"exact_history\":\"identical_norm_gated_inputs_only\"}"
              << ",\"projection_inputs\":{\"source\":\"fixed_finite_synthetic_same_CPU_GPU_arrays\",\"immutable\":true"
              << ",\"names\":[\"raw_down\",\"raw_up\",\"raw_inject\",\"raw_key\",\"raw_query\",\"shared_value\"]"
              << ",\"actual_matrix_multiplication\":false,\"block_output\":\"synthetic\"}"
              << ",\"cpu_equations\":\"independent_ascending_FP32_direct_stored_gamma_unfused\""
              << ",\"history_modes\":{\"isolated_conv\":\"exact_common_norm_gated\",\"PLE_pipeline\":\"bounded_own_parallel_RMS\"}"
              << ",\"transaction_scope\":\"conv_only_owner_publication_marker_no_hash_or_EOS_reset\""
              << ",\"token_counts\":[1,2,3,128],\"repeats\":20,\"row_counts\":{\"source\":1,\"correctness\":2,\"measurement\":16,\"complete\":1,\"total\":20}"
              << ",\"order\":{\"correctness_devices\":[0,1],\"measurement_outer\":\"device\",\"measurement_middle\":\"tokens\",\"measurement_inner\":[\"HC\",\"PLE\"]}"
              << ",\"correctness_exact_counters\":{\"norm_cases\":86,\"activation_cases\":5,\"mix_cases\":5,\"injection_cases\":5,\"combine_cases\":5"
              << ",\"gate_cases\":8,\"conv_cases\":19,\"head_cases\":5,\"broadcast_cases\":56,\"signed_zero_cases\":2,\"large_finite_cases\":1"
              << ",\"prefix_slots\":21,\"restore_cases\":10,\"repeated_accept0_cases\":4,\"mask_cases\":2,\"readonly_active_checks\":65"
              << ",\"publication_checks\":73,\"rejected_publications\":46,\"numeric_rejects\":46,\"host_rejects\":224,\"readonly_alias_cases\":2,\"valid_reuses\":46"
              << ",\"numeric_invalid_cases\":39,\"numeric_arithmetic_cases\":6,\"numeric_combined_cases\":1}"
              << ",\"correctness_positive_counters\":[\"immutable_checks\",\"canary_checks\"]"
              << ",\"host_validation\":{\"expected_status\":\"hipErrorInvalidValue\",\"checks\":[\"geometry\",\"epsilon\",\"null_required\",\"all_writable_overlap\",\"misalignment\",\"wrapping\"]"
              << ",\"invalid_addresses_enqueued\":false}"
              << ",\"timing_scope\":\"individual_completed_HIP_event_intervals_resident_primitives_only\",\"launch_counts\":{\"HC\":5,\"PLE\":7}"
              << ",\"full_pipeline_qualification\":false,\"request_performance_claim\":false}\n";
    std::cout.flush();
}
void correctness_json(int device, const Summary& s) {
    std::cout << "{\"kind\":\"blocks_correctness\",\"protocol\":1,\"device\":" << device << ",\"counters\":";
    counters_json(s.counters); std::cout << ",\"error\":"; errors_json(s.errors);
    std::cout << ",\"parity\":{\"cpu_equations\":true,\"same_projections\":true,\"finite\":true,\"byte_canaries\":true,\"immutable_inputs\":true"
              << ",\"root_head_no_injection\":true,\"original_widened_tap_unchanged\":true,\"history_exact_common_inputs\":true"
              << ",\"history_bounded_own_parallel_rms\":true,\"all_four_prefix_tails\":true,\"restore_1_plus_accepted\":true"
              << ",\"repeated_accept0\":true,\"keep_mask_0_1_no_EOS_reset\":true,\"active_readonly_until_publish\":true"
              << ",\"rejected_entire_active_and_marker_unchanged\":true,\"valid_reuse\":true},\"passed\":true}\n";
    std::cout.flush();
}

void measurement(Fixture& f, const Data& p, int n, bool ple) {
    const Inputs x(n, 101);
    const auto initial = synthetic(history_size, 102, .125f);
    const Reference r(p, x, initial);
    f.upload(x); f.parameters(p);
    auto reset = [&] {
        f.fill_outputs(); f.clear_error(); f.put(f.active, initial);
        check(hipStreamSynchronize(f.stream));
    };
    auto chain = [&] { if (ple) ple_chain(f, p, n); else hc_chain(f, p, n); };
    auto validate = [&](Summary& s) {
        if (ple) validate_ple(f, r, s);
        else {
            validate_hc(f, r, s);
            exact(f.get(f.active, history_size), initial, "HC modified unrelated active history");
        }
        f.immutable(x, p, s.counters); f.canaries(s.counters);
    };
    // Complete and validate one untimed warmup; its errors/counters are separate.
    Summary warm;
    reset(); chain(); validate(warm);
    Summary measured;
    double total_ms = 0;
    for (int rep = 0; rep < repeats; ++rep) {
        reset();
        check(hipEventRecord(f.begin, f.stream)); chain(); check(hipEventRecord(f.end, f.stream));
        check(hipEventSynchronize(f.end));
        float elapsed_ms = 0; check(hipEventElapsedTime(&elapsed_ms, f.begin, f.end));
        require(std::isfinite(elapsed_ms) && elapsed_ms > 0.0f, "invalid completed event interval");
        total_ms += elapsed_ms;
        validate(measured); // EVERY repetition before reusing any state/output
    }
    std::cout << "{\"kind\":\"blocks_measurement\",\"protocol\":1,\"device\":" << f.device
              << ",\"tokens\":" << n << ",\"pipeline\":\"" << (ple ? "PLE" : "HC")
              << "\",\"repeats\":" << repeats << ",\"validated_repetitions\":" << repeats
              << ",\"resident_ms\":" << total_ms / repeats << ",\"launchcount\":" << (ple ? 7 : 5)
              << ",\"timing_excludes\":[\"model_load\",\"fixture_allocation\",\"CPU_reference\",\"H2D\",\"D2H\",\"history_restore\",\"output_reset\",\"error_clear\",\"validation\",\"warmup\"]"
              << ",\"error\":"; errors_json(measured.errors);
    std::cout << ",\"counters\":"; counters_json(measured.counters);
    std::cout << ",\"history_comparison\":\"" << (ple ? "bounded_own_parallel_RMS" : "not_applicable") << '"';
    std::cout << ",\"parity\":{\"cpu_equations\":true,\"same_projections\":true,\"finite\":true,\"byte_canaries\":true,\"immutable_inputs\":true"
              << ",\"history_bounded_own_parallel_rms\":" << (ple ? "true" : "null")
              << ",\"passed\":true}}\n";
    std::cout.flush();
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: core-blocks target.gguf");
        std::cout.exceptions(std::ios::badbit | std::ios::failbit);
        std::cout << std::setprecision(17);
        const qwen::Model model(argv[1]);
        const Data data(model);
        int count = 0; check(hipGetDeviceCount(&count));
        require(count == 2, "blocks requires exactly devices 0/1");
        std::array<hipDeviceProp_t, 2> devices{};
        for (int device = 0; device < 2; ++device) {
            check(hipGetDeviceProperties(&devices[device], device));
            const std::string_view arch(devices[device].gcnArchName);
            require((arch == "gfx906" || arch.starts_with("gfx906:")) && devices[device].warpSize == 64,
                    "blocks requires two gfx906 wave64 devices");
        }
        source_json(model, data, argv[1], devices);
        int correctness_rows = 0, measurement_rows = 0;
        for (int device = 0; device < 2; ++device) {
            Summary result;
            { Fixture fixture(device); fixture.initialize(data); result = qualify(fixture, data); }
            correctness_json(device, result); ++correctness_rows;
        }
        for (int device = 0; device < 2; ++device) {
            Fixture fixture(device); fixture.initialize(data);
            for (int n : token_counts) {
                measurement(fixture, data, n, false); ++measurement_rows;
                measurement(fixture, data, n, true); ++measurement_rows;
            }
        }
        require(correctness_rows == 2 && measurement_rows == 16, "blocks missing protocol rows");
        // All fixture destructors, including checked cleanup, have completed.
        std::cout << "{\"kind\":\"blocks_complete\",\"protocol\":1,\"correctness_rows\":" << correctness_rows
                  << ",\"measurement_rows\":" << measurement_rows << ",\"measurement_repetitions\":" << measurement_rows * repeats
                  << ",\"jsonl_rows\":" << protocol_rows << ",\"passed\":true}\n";
        std::cout.flush();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "core-blocks: " << e.what() << '\n'; return 1;
    }
}
