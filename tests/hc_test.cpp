#include "hc.hpp"
#include "dense.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef CORE_REVISION
#define CORE_REVISION "unknown"
#define CORE_DIRTY 1
#endif

// Count C++ allocations, including aligned allocations, across successful calls.
// malloc-backed new remains visible to ASan/LSan. Out-of-line free avoids GCC's
// mismatched-new-delete false positives when replacing global new/delete.
namespace {
std::size_t allocation_count = 0;
[[gnu::noinline]] void* allocate(std::size_t size) {
    ++allocation_count;
    if (void* pointer = std::malloc(size == 0 ? 1 : size)) return pointer;
    throw std::bad_alloc();
}
[[gnu::noinline]] void* allocate_aligned(std::size_t size, std::size_t alignment) {
    ++allocation_count;
    void* pointer = nullptr;
    if (posix_memalign(&pointer, alignment, size == 0 ? 1 : size) == 0) return pointer;
    throw std::bad_alloc();
}
[[gnu::noinline]] void release(void* pointer) noexcept { std::free(pointer); }
}
void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void operator delete(void* pointer) noexcept { release(pointer); }
void operator delete[](void* pointer) noexcept { release(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { release(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { release(pointer); }
void* operator new(std::size_t size, std::align_val_t alignment) {
    return allocate_aligned(size, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return allocate_aligned(size, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer, std::align_val_t) noexcept { release(pointer); }
void operator delete[](void* pointer, std::align_val_t) noexcept { release(pointer); }
void operator delete(void* pointer, std::size_t, std::align_val_t) noexcept { release(pointer); }
void operator delete[](void* pointer, std::size_t, std::align_val_t) noexcept { release(pointer); }

namespace {

using qwen::HcConfig;
using qwen::TensorType;
std::size_t checks = 0, rejections = 0, arithmetic_rejections = 0, fixtures = 0;
double max_abs_error = 0.0;

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void near(float actual, double expected) {
    const double error = std::abs(static_cast<double>(actual) - expected);
    max_abs_error = std::max(max_abs_error, error);
    // Fixed beforehand: scalar FP32 reductions vs independent FP64 HF algebra.
    check(std::isfinite(actual) && error <= 4.0e-6 * (1.0 + std::abs(expected)),
          "FP32 HC differs from independent double HF equations");
}

void near(std::span<const float> actual, std::span<const double> expected) {
    check(actual.size() == expected.size(), "reference shape");
    for (std::size_t i = 0; i < actual.size(); ++i) near(actual[i], expected[i]);
}

void same(std::span<const float> actual, std::span<const float> expected) {
    check(actual.size() == expected.size(), "exact shape");
    for (std::size_t i = 0; i < actual.size(); ++i)
        check(std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected[i]),
              "caller buffer changed or token/chunk mismatch");
}

template<class Function> void rejected(Function function) {
    try { function(); }
    catch (const std::invalid_argument&) { ++rejections; return; }
    throw std::runtime_error("bad HC argument accepted or wrong exception type");
}

template<class Function> void arithmetic_rejected(Function function) {
    try { function(); }
    catch (const std::runtime_error&) { ++arithmetic_rejections; return; }
    throw std::runtime_error("HC arithmetic overflow accepted or wrong exception type");
}

void put16(std::span<std::byte> data, std::size_t at, std::uint16_t value) {
    data[at] = std::byte(value & 255);
    data[at + 1] = std::byte(value >> 8);
}

void put32(std::span<std::byte> data, std::size_t at, std::uint32_t value) {
    for (std::size_t i = 0; i < 4; ++i) data[at + i] = std::byte((value >> (8 * i)) & 255);
}

struct Entry { std::size_t row, column; double value; };

// Logical sparse entries are the independent reference, not decoded production
// bytes. This also exercises full real geometry without a 50+ MB FP64 matrix.
struct Matrix {
    TensorType type;
    int input, output;
    std::vector<std::byte> bytes;
    std::vector<Entry> entries;

    Matrix(TensorType dtype, std::size_t width, std::size_t rows)
        : type(dtype), input(static_cast<int>(width)), output(static_cast<int>(rows)) {
        if (type == TensorType::Q4_0) {
            check(width % 32 == 0, "fixture Q4 row alignment");
            bytes.resize(width * rows / 32 * 18);
            for (std::size_t block = 0; block < width * rows / 32; ++block) {
                put16(bytes, block * 18, 0x2800); // binary16 exactly +1/32
                std::fill_n(bytes.begin() + block * 18 + 2, 16, std::byte{0x88});
            }
        } else {
            check(type == TensorType::F32 || type == TensorType::BF16, "fixture type");
            bytes.resize(width * rows * (type == TensorType::F32 ? 4 : 2));
        }
    }

    qwen::QMatrix view() const { return {type, input, output, bytes}; }

    void set(std::size_t row, std::size_t column, float value) {
        const auto index = row * static_cast<std::size_t>(input) + column;
        check(row < static_cast<std::size_t>(output) && column < static_cast<std::size_t>(input),
              "fixture matrix index");
        if (type == TensorType::Q4_0) {
            const int code = static_cast<int>(std::lround(value * 32.0f));
            check(code >= -8 && code <= 7 && value == static_cast<float>(code) / 32.0f,
                  "fixture exact Q4 value");
            const auto at = (index / 32) * 18 + 2 + index % 16;
            const auto old = std::to_integer<unsigned>(bytes[at]);
            const auto shift = (index % 32 < 16 ? 0 : 4);
            bytes[at] = std::byte((old & ~(15U << shift)) | (static_cast<unsigned>(code + 8) << shift));
        } else {
            const auto bits = std::bit_cast<std::uint32_t>(value);
            if (type == TensorType::F32) put32(bytes, index * 4, bits);
            else {
                check((bits & 65535U) == 0, "fixture exact BF16 value");
                put16(bytes, index * 2, static_cast<std::uint16_t>(bits >> 16));
            }
        }
        const auto it = std::find_if(entries.begin(), entries.end(), [=](const Entry& e) {
            return e.row == row && e.column == column;
        });
        if (it != entries.end()) it->value = value;
        else if (value != 0.0f) entries.push_back({row, column, value});
    }
};

struct Weights {
    HcConfig config;
    Matrix down, up, inject;
    std::vector<float> norm;

    explicit Weights(HcConfig cfg, TensorType dtype = TensorType::F32,
                     TensorType inject_dtype = TensorType::F32)
        : config(cfg), down(dtype, cfg.hidden_size * cfg.branches, cfg.low_rank),
          up(dtype, cfg.low_rank, cfg.hidden_size * cfg.branches),
          inject(inject_dtype, cfg.hidden_size * cfg.branches, cfg.branches),
          norm(cfg.hidden_size * cfg.branches, 1.0f) {}

    qwen::HcParameters parameters(bool with_inject = true) const {
        return {down.view(), up.view(), norm,
                with_inject ? std::optional(inject.view()) : std::nullopt};
    }
};

std::vector<double> linear(const Matrix& matrix, std::span<const double> input) {
    std::vector<double> output(static_cast<std::size_t>(matrix.output), 0.0);
    for (const auto& e : matrix.entries) output[e.row] += e.value * input[e.column];
    return output;
}

struct Reference {
    std::vector<double> mixed, injection;
};

// Independent HF algebra, directly from pinned Qwen4ExpTextGatedResidual
// lines 551-563 and Qwen4ExpTextRMSNorm lines 305-310. No production norm,
// sigmoid, decoder, matmul or HC helper is called. Logical weights are supplied
// above; gamma is already 1+HF weight as stored by the GGUF converter/graph.
Reference reference(const Weights& w, std::span<const float> x, std::size_t tokens) {
    const auto h = w.config.hidden_size, c = w.config.branches, wide = h * c;
    Reference result{std::vector<double>(tokens * h), std::vector<double>(tokens * c)};
    for (std::size_t t = 0; t < tokens; ++t) {
        std::vector<double> normalized(wide);
        for (std::size_t branch = 0; branch < c; ++branch) {
            double mean_square = 0.0;
            for (std::size_t j = 0; j < h; ++j) {
                const double value = x[t * wide + branch * h + j];
                mean_square += value * value;
            }
            mean_square /= static_cast<double>(h);
            const double denominator = std::sqrt(mean_square + w.config.rms_epsilon);
            for (std::size_t j = 0; j < h; ++j)
                normalized[branch * h + j] = x[t * wide + branch * h + j] /
                    denominator * w.norm[branch * h + j];
        }
        auto low = linear(w.down, normalized);
        for (double& value : low) {
            const double scaled = value / static_cast<double>(c);
            value = scaled / (1.0 + std::exp(-scaled));
        }
        const auto gates = linear(w.up, low);
        for (std::size_t j = 0; j < h; ++j) {
            double mixed = 0.0;
            for (std::size_t branch = 0; branch < c; ++branch)
                mixed += normalized[branch * h + j] / (1.0 + std::exp(-gates[branch * h + j]));
            result.mixed[t * h + j] = mixed / static_cast<double>(c);
        }
        const auto inject = linear(w.inject, normalized);
        for (std::size_t branch = 0; branch < c; ++branch)
            result.injection[t * c + branch] = 2.0 /
                (1.0 + std::exp(-inject[branch] / static_cast<double>(c)));
    }
    return result;
}

std::vector<double> post_reference(HcConfig cfg, std::span<const float> residual,
                                  std::span<const float> block, std::span<const double> gates,
                                  std::size_t tokens) {
    const auto h = cfg.hidden_size, c = cfg.branches, wide = h * c;
    std::vector<double> result(tokens * wide);
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t branch = 0; branch < c; ++branch)
            for (std::size_t j = 0; j < h; ++j)
                result[t * wide + branch * h + j] = residual[t * wide + branch * h + j] +
                    gates[t * c + branch] * static_cast<double>(block[t * h + j]);
    return result;
}

void exercise(Weights& weights, std::span<const float> input, std::size_t tokens) {
    ++fixtures;
    const auto cfg = weights.config;
    const auto expected = reference(weights, input, tokens);
    qwen::HcCpu block(cfg, weights.parameters(), tokens);
    qwen::HcCpu head(cfg, weights.parameters(false), tokens);
    std::vector<float> mixed(tokens * cfg.hidden_size, 99), gates(tokens * cfg.branches, 99);
    std::vector<float> collapsed(mixed.size(), 99), combined(input.size(), 99);
    std::vector<float> block_out(mixed.size());
    for (std::size_t i = 0; i < block_out.size(); ++i)
        block_out[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 8.0f;
    const std::vector<float> original(input.begin(), input.end());
    block.mix(input, tokens, mixed, gates);
    head.mix(input, tokens, collapsed);
    near(mixed, expected.mixed);
    near(gates, expected.injection);
    same(collapsed, mixed);
    same(input, original);
    block.combine(input, block_out, gates, tokens, combined);
    near(combined, post_reference(cfg, input, block_out, expected.injection, tokens));
    const auto tap = combined;
    head.mix(combined, tokens, collapsed);
    near(collapsed, reference(weights, combined, tokens).mixed);
    same(combined, tap); // the full-width pre-head Tap remains intact
    check(block.has_injection() && !head.has_injection(), "head has no inject tensor");
    check(block.capacity() == tokens && block.config().widened_elements() == cfg.hidden_size * cfg.branches,
          "config/capacity accessors");

    // Reuse after combine overwrote disposable normalization scratch, varying
    // token count, not merely repeating one cached result.
    qwen::HcCpu one(cfg, weights.parameters());
    std::vector<float> single(cfg.hidden_size), single_gates(cfg.branches);
    const auto wide = cfg.widened_elements();
    for (std::size_t t = 0; t < tokens; ++t) {
        one.mix(input.subspan(t * wide, wide), 1, single, single_gates);
        same(single, std::span(mixed).subspan(t * cfg.hidden_size, cfg.hidden_size));
        same(single_gates, std::span(gates).subspan(t * cfg.branches, cfg.branches));
    }
    std::vector<float> expanded(input.size());
    const auto before = allocation_count;
    for (int repeat = 0; repeat < 3; ++repeat) {
        block.mix(input, tokens, mixed, gates);
        block.combine(input, block_out, gates, tokens, combined);
        head.mix(combined, tokens, collapsed);
        block.expand(block_out, tokens, expanded);
        block.mix(input.first(wide), 1, single, single_gates);
    }
    check(allocation_count == before, "HC successful calls allocate hidden memory");
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t branch = 0; branch < cfg.branches; ++branch)
            same(std::span(expanded).subspan(t * wide + branch * cfg.hidden_size, cfg.hidden_size),
                 std::span(block_out).subspan(t * cfg.hidden_size, cfg.hidden_size));
    near(mixed, expected.mixed);
    near(gates, expected.injection);
}

void rectangular() {
    Weights w({5, 3, 7, 0.03125f});
    for (std::size_t row = 0; row < 7; ++row)
        for (std::size_t col = 0; col < 15; ++col)
            w.down.set(row, col, static_cast<float>(static_cast<int>((row * 13 + col * 3) % 11) - 5) / 32);
    for (std::size_t row = 0; row < 15; ++row)
        for (std::size_t col = 0; col < 7; ++col)
            w.up.set(row, col, static_cast<float>(static_cast<int>((row * 5 + col * 7) % 13) - 6) / 16);
    for (std::size_t row = 0; row < 3; ++row)
        for (std::size_t col = 0; col < 15; ++col)
            w.inject.set(row, col, static_cast<float>(static_cast<int>((row * 3 + col * 7) % 9) - 4) / 8);
    for (std::size_t i = 0; i < w.norm.size(); ++i)
        w.norm[i] = static_cast<float>(static_cast<int>(i % 7) - 2) / 4;
    std::vector<float> x(3 * 15);
    for (std::size_t t = 0; t < 3; ++t)
        for (std::size_t branch = 0; branch < 3; ++branch)
            for (std::size_t j = 0; j < 5; ++j)
                x[t * 15 + branch * 5 + j] = std::ldexp(
                    static_cast<float>(static_cast<int>((t * 3 + j * 5 + branch) % 17) - 8) / 8,
                    static_cast<int>(branch * 3));
    exercise(w, x, 3);
}

void constants_and_epsilon() {
    for (const float eps : {1.0e-6f, 0.25f, 1.0e-20f}) {
        Weights w({2, 4, 3, eps}); // zero low-rank/inject => G=.5, I=1
        w.norm = {0.25f, -1.5f, 2.0f, 0.0f, 0.5f, 3.0f, -0.25f, 1.25f};
        std::array<float, 24> x{};
        for (std::size_t branch = 0; branch < 4; ++branch)
            for (std::size_t j = 0; j < 2; ++j) {
                x[8 + branch * 2 + j] = branch % 2 == 0 ? 2.0f : -3.0f;
                x[16 + branch * 2 + j] = (branch % 2 == 0 ? 1.0f : -1.0f) * 1.0e-10f;
            }
        exercise(w, x, 3);
    }
    // Single feature per branch cannot hide a wrong grouping axis or omitted
    // gamma fold. With zero mix weights the result is .5*mean(normalized*gamma),
    // not a softmax average (unit gates), and GGUF gamma is not incremented.
    Weights w({1, 2, 3, 0.25f});
    w.norm = {0.25f, 2.5f};
    const std::array<float, 4> x{3, -2, 0, 0};
    exercise(w, x, 2);
}

void gate_extremes() {
    Weights w({2, 3, 2, 1.0e-6f}, TensorType::BF16, TensorType::BF16);
    for (std::size_t branch = 0; branch < 3; ++branch) {
        w.down.set(0, branch * 2, 1024);
        w.down.set(1, branch * 2, -1024);
        w.up.set(branch * 2, 0, 1024);
        w.up.set(branch * 2 + 1, 0, -1024);
        w.inject.set(0, branch * 2, 1024);
        w.inject.set(1, branch * 2, -1024);
    }
    const std::array<float, 6> x{1, 1, 2, 2, 4, 4};
    exercise(w, x, 1);
    qwen::HcCpu cpu(w.config, w.parameters());
    std::array<float, 2> mixed{};
    std::array<float, 3> gates{};
    cpu.mix(x, 1, mixed, gates);
    check(gates == std::array<float, 3>{2, 0, 1}, "saturated injection gates [2,0,1]");
    check(mixed[0] > 0.99f && mixed[1] == 0, "sigmoid feature gates saturate independently");
    // Explicit endpoint gates are valid; zero preserves exactly one branch,
    // and 2 amplifies only the matching branch, rather than broadcasting I[0].
    const std::array<float, 2> out{3, -2};
    std::array<float, 6> result{};
    cpu.combine(x, out, gates, 1, result);
    const std::array<float, 6> expected{7, -3, 2, 2, 7, 2};
    same(result, expected);
}

void real_geometry() {
    const HcConfig cfg{};
    check(cfg.widened_elements() == 10240 && cfg.low_rank == 320, "actual HC geometry");
    for (const auto type : {TensorType::Q4_0, TensorType::BF16}) {
        // Layer: Q4_0 down/up + BF16 inject. Head: BF16 down/up without inject
        // is exercised inside exercise(); BF16 block also checks the shared ABI.
        Weights w(cfg, type, TensorType::BF16);
        for (std::size_t r = 0; r < cfg.low_rank; ++r) {
            w.down.set(r, (37 * r + 7) % 10240, 3.0f / 32);
            w.down.set(r, (53 * r + 8191) % 10240, -2.0f / 32);
        }
        for (std::size_t row = 0; row < 10240; ++row)
            w.up.set(row, (row * 7 + row / 2560) % 320,
                     static_cast<float>(static_cast<int>(row % 7) - 3) / 32);
        for (std::size_t branch = 0; branch < 4; ++branch)
            for (std::size_t other = 0; other < 4; ++other)
                w.inject.set(branch, other * 2560 + (branch * 17 + other * 31) % 2560,
                             static_cast<float>(static_cast<int>(branch) - static_cast<int>(other)) / 32);
        for (std::size_t i = 0; i < w.norm.size(); ++i)
            w.norm[i] = static_cast<float>(8 + i % 23) / 16;
        std::vector<float> x(2 * 10240);
        for (std::size_t t = 0; t < 2; ++t)
            for (std::size_t branch = 0; branch < 4; ++branch)
                for (std::size_t j = 0; j < 2560; ++j)
                    x[t * 10240 + branch * 2560 + j] = std::ldexp(
                        static_cast<float>(static_cast<int>((j * 5 + t * 11 + branch * 7) % 37) - 18) / 16,
                        static_cast<int>(branch));
        exercise(w, x, 2);
    }
}

// Actual-weight oracle: the equations are the same independent HF algebra as
// reference() above, with each elementary product/sum rounded explicitly to
// FP32. A 10240-term dense reduction is not an exact FP64 reduction; this keeps
// the ORIGINAL 4e-6*(1+abs(reference)) acceptance rule without disguising that
// distinction as a larger tolerance. All arithmetic here is evaluated in double
// before explicit rounding. Only canonical weight decoding uses dense.hpp; no
// production HC, normalization, activation or matrix multiply is reused.
double round_f32(double value) { return static_cast<double>(static_cast<float>(value)); }

double reference_sigmoid_f32(double value) {
    const double exponent = round_f32(std::exp(-std::abs(value)));
    const double denominator = round_f32(1.0 + exponent);
    return round_f32((value >= 0.0 ? 1.0 : exponent) / denominator);
}

std::vector<double> reference_rows_f32(const qwen::QMatrix& matrix,
                                     std::span<const double> input, std::size_t tokens,
                                     std::span<const std::size_t> rows) {
    const auto width = static_cast<std::size_t>(matrix.input);
    check(input.size() == tokens * width, "loaded reference projection input shape");
    std::vector<double> output(tokens * rows.size(), 0.0);
    // Decode each selected weight once for all tokens, keeping ascending-k
    // reduction order independently for every [token][row] accumulator.
    for (std::size_t r = 0; r < rows.size(); ++r) {
        check(rows[r] < static_cast<std::size_t>(matrix.output), "loaded reference row range");
        for (std::size_t k = 0; k < width; ++k) {
            const double weight = qwen::tensor_element(matrix.type, matrix.weights, rows[r] * width + k);
            for (std::size_t t = 0; t < tokens; ++t) {
                auto& sum = output[t * rows.size() + r];
                sum = round_f32(sum + round_f32(weight * input[t * width + k]));
            }
        }
    }
    return output;
}

Reference sampled_reference_f32(HcConfig cfg, qwen::HcParameters p,
                               std::span<const float> x, std::size_t tokens,
                               std::span<const std::size_t> features) {
    const auto h = cfg.hidden_size, c = cfg.branches, wide = cfg.widened_elements();
    check(x.size() == tokens * wide && p.norm_weight.size() == wide, "loaded reference norm shape");
    std::vector<double> normalized(x.size());
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t branch = 0; branch < c; ++branch) {
            const auto start = t * wide + branch * h;
            double squares = 0.0;
            for (std::size_t j = 0; j < h; ++j) {
                const double value = x[start + j];
                squares = round_f32(squares + round_f32(value * value));
            }
            const double variance = round_f32(round_f32(squares / static_cast<double>(h)) + cfg.rms_epsilon);
            const double inverse = round_f32(1.0 / round_f32(std::sqrt(variance)));
            for (std::size_t j = 0; j < h; ++j)
                normalized[start + j] = round_f32(round_f32(x[start + j] * inverse) *
                                                 p.norm_weight[branch * h + j]);
        }
    std::vector<std::size_t> down_rows(cfg.low_rank);
    std::iota(down_rows.begin(), down_rows.end(), std::size_t{0});
    auto low = reference_rows_f32(p.down, normalized, tokens, down_rows);
    for (double& value : low) {
        const double scaled = round_f32(value / static_cast<double>(c));
        value = round_f32(scaled * reference_sigmoid_f32(scaled));
    }
    std::vector<std::size_t> up_rows;
    up_rows.reserve(c * features.size());
    for (std::size_t branch = 0; branch < c; ++branch)
        for (const auto feature : features) {
            check(feature < h, "loaded reference feature range");
            up_rows.push_back(branch * h + feature);
        }
    const auto logits = reference_rows_f32(p.up, low, tokens, up_rows);
    Reference result{std::vector<double>(tokens * features.size()), {}};
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t j = 0; j < features.size(); ++j) {
            double mixed = 0.0;
            for (std::size_t branch = 0; branch < c; ++branch) {
                const double gate = reference_sigmoid_f32(logits[t * up_rows.size() + branch * features.size() + j]);
                mixed = round_f32(mixed + round_f32(gate * normalized[t * wide + branch * h + features[j]]));
            }
            result.mixed[t * features.size() + j] = round_f32(mixed / static_cast<double>(c));
        }
    if (p.inject) {
        std::vector<std::size_t> inject_rows(c);
        std::iota(inject_rows.begin(), inject_rows.end(), std::size_t{0});
        result.injection = reference_rows_f32(*p.inject, normalized, tokens, inject_rows);
        for (double& value : result.injection)
            value = round_f32(2.0 * reference_sigmoid_f32(round_f32(value / static_cast<double>(c))));
    }
    return result;
}

struct ActualCase {
    std::string_view mixer;
    bool injection;
    std::size_t mix_samples = 0, injection_samples = 0, combine_samples = 0;
    double mix_error = 0.0, injection_error = 0.0, combine_error = 0.0;
};

void compare_sample(float actual, double expected, double& maximum) {
    near(actual, expected); // The original tolerance is shared, never widened.
    maximum = std::max(maximum, std::abs(static_cast<double>(actual) - expected));
}

std::vector<float> actual_input(HcConfig cfg) {
    constexpr std::size_t tokens = 3;
    const auto h = cfg.hidden_size, c = cfg.branches, wide = cfg.widened_elements();
    std::vector<float> input(tokens * wide);
    // Immutable synthetic residual, intentionally distinct by token, branch
    // and feature. Binary fractions avoid platform-dependent random generators.
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t branch = 0; branch < c; ++branch)
            for (std::size_t j = 0; j < h; ++j)
                input[t * wide + branch * h + j] = std::ldexp(
                    static_cast<float>(static_cast<int>((j * 5 + t * 11 + branch * 7) % 37) - 18) / 16.0f,
                    static_cast<int>(branch)) + static_cast<float>(branch + t) / 32.0f;
    return input;
}

ActualCase actual_case(HcConfig cfg, qwen::HcParameters parameters, std::string_view mixer,
                      std::span<const float> input, std::span<const std::size_t> features) {
    constexpr std::size_t tokens = 3;
    const auto h = cfg.hidden_size, c = cfg.branches, wide = cfg.widened_elements();
    qwen::HcCpu cpu(cfg, parameters, tokens);
    const bool inject = parameters.inject.has_value();
    check(cpu.has_injection() == inject, "actual head/block injection convention");
    const auto expected = sampled_reference_f32(cfg, parameters, input, tokens, features);
    const std::vector<float> original(input.begin(), input.end());
    std::vector<float> mixed(tokens * h), gates(inject ? tokens * c : 0), combined(tokens * wide, 53);
    std::vector<float> block(tokens * h);
    for (std::size_t i = 0; i < block.size(); ++i)
        block[i] = static_cast<float>(static_cast<int>((i * 7 + i / h * 11) % 29) - 14) / 32.0f;
    auto allocations = allocation_count;
    cpu.mix(input, tokens, mixed, gates);
    check(allocation_count == allocations, "loaded HC mix allocates");
    const auto batch_mixed = mixed, batch_gates = gates;
    if (inject) {
        allocations = allocation_count;
        cpu.combine(input, block, gates, tokens, combined);
        check(allocation_count == allocations, "loaded HC combine allocates");
    } else {
        // The actual root has neither an injection projection nor post-combine.
        // Supplying gates must fail atomically, including the collapsed output.
        std::vector<float> unexpected_gates(tokens * c, 53);
        const auto old_gates = unexpected_gates;
        const auto old_combined = combined;
        rejected([&] { cpu.mix(input, tokens, mixed, unexpected_gates); });
        same(mixed, batch_mixed);
        same(unexpected_gates, old_gates);
        rejected([&] { cpu.combine(input, block, unexpected_gates, tokens, combined); });
        same(combined, old_combined);
        check(expected.injection.empty(), "head reference must not inject");
    }
    const auto batch_combined = combined;
    ActualCase report{mixer, inject};
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t j = 0; j < features.size(); ++j) {
            compare_sample(batch_mixed[t * h + features[j]], expected.mixed[t * features.size() + j], report.mix_error);
            ++report.mix_samples;
        }
        if (inject)
            for (std::size_t branch = 0; branch < c; ++branch) {
                compare_sample(batch_gates[t * c + branch], expected.injection[t * c + branch], report.injection_error);
                ++report.injection_samples;
                for (std::size_t j = 0; j < h; ++j) {
                    const auto at = t * wide + branch * h + j;
                    const double next = round_f32(input[at] + round_f32(expected.injection[t * c + branch] * block[t * h + j]));
                    compare_sample(batch_combined[at], next, report.combine_error);
                    ++report.combine_samples;
                }
            }
    }
    // Only three projected calls per mixer: N3 baseline followed by N1/N2
    // prefixes of EXACTLY the same input. The independent oracle above runs
    // once for N3, projecting all rank rows but only sampled output features.
    for (const std::size_t n : {std::size_t{1}, std::size_t{2}}) {
        auto out = std::span(mixed).first(n * h);
        auto injection = std::span(gates).first(inject ? n * c : 0);
        allocations = allocation_count;
        cpu.mix(input.first(n * wide), n, out, injection);
        if (inject)
            cpu.combine(input.first(n * wide), std::span(block).first(n * h), injection,
                        n, std::span(combined).first(n * wide));
        check(allocation_count == allocations, "loaded HC prefix reuse allocates");
        same(out, std::span(batch_mixed).first(n * h));
        same(injection, std::span(batch_gates).first(inject ? n * c : 0));
        if (inject) same(std::span(combined).first(n * wide), std::span(batch_combined).first(n * wide));
    }
    for (const auto value : batch_mixed) check(std::isfinite(value), "actual HC output nonfinite");
    for (const auto value : batch_gates) check(std::isfinite(value) && value >= 0 && value <= 2, "actual HC gate range");
    if (inject)
        for (const auto value : batch_combined) check(std::isfinite(value), "actual HC widened output nonfinite");
    same(input, original); // Full-width residual/MTP Tap survives every call.
    return report;
}

std::size_t actual_dimension(const qwen::MetadataValue& value) {
    std::uint64_t dimension = 0;
    if (value.type == qwen::MetadataType::UINT32) dimension = value.get<std::uint32_t>();
    else if (value.type == qwen::MetadataType::UINT64) dimension = value.get<std::uint64_t>();
    else throw std::invalid_argument("HC fixture: geometry metadata must be UINT32/UINT64");
    if (dimension == 0 || dimension > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("HC fixture: geometry metadata outside positive QMatrix ABI");
    return static_cast<std::size_t>(dimension);
}

void validate_actual_tensor(const qwen::TensorView& tensor, TensorType type,
                            std::span<const std::uint64_t> dimensions) {
    if (dimensions.empty() || dimensions.size() > tensor.dimensions.size() ||
        tensor.type != type || tensor.rank != dimensions.size() ||
        !std::equal(dimensions.begin(), dimensions.end(), tensor.dimensions.begin()))
        throw std::invalid_argument("HC fixture: wrong dtype/rank/dimensions for " + tensor.name);
    const auto layout = qwen::type_layout(type);
    if (dimensions[0] % layout.block_elements != 0)
        throw std::invalid_argument("HC fixture: matrix row block alignment");
    std::uint64_t elements = 1;
    for (const auto dimension : dimensions) {
        if (dimension == 0 || dimension > std::numeric_limits<std::uint64_t>::max() / elements)
            throw std::invalid_argument("HC fixture: tensor element overflow");
        elements *= dimension;
    }
    const auto blocks = elements / layout.block_elements;
    if (blocks > std::numeric_limits<std::uint64_t>::max() / layout.block_bytes ||
        tensor.elements != elements || tensor.byte_size != blocks * layout.block_bytes ||
        tensor.byte_size > std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("HC fixture: inconsistent tensor byte/element count");
}

struct ActualViews {
    const qwen::TensorView *down, *up, *norm, *inject;
};

ActualViews actual_views(const qwen::Model& model, HcConfig cfg, std::string_view prefix, bool inject) {
    auto tensor = [&](const char* suffix) -> const qwen::TensorView* {
        return &model.tensor(std::string(prefix) + suffix + ".weight");
    };
    ActualViews views{tensor("down"), tensor("up"), tensor("norm"), inject ? tensor("inject") : nullptr};
    const std::uint64_t wide = cfg.widened_elements(), rank = cfg.low_rank, branches = cfg.branches;
    const std::array down{wide, rank}, up{rank, wide}, injection{wide, branches};
    const std::array norm{wide};
    const auto type = inject ? TensorType::Q4_0 : TensorType::BF16;
    validate_actual_tensor(*views.down, type, down);
    validate_actual_tensor(*views.up, type, up);
    validate_actual_tensor(*views.norm, TensorType::F32, norm);
    if (inject) validate_actual_tensor(*views.inject, TensorType::BF16, injection);
    else if (model.find_tensor(std::string(prefix) + "inject.weight"))
        throw std::invalid_argument("HC fixture: root head must not have inject weights");
    return views;
}

struct ActualWeights {
    std::vector<std::byte> down, up, inject;
    std::vector<float> norm;

    ActualWeights(const qwen::Model& model, const ActualViews& views) {
        auto read = [&](const qwen::TensorView& tensor, std::vector<std::byte>& data) {
            data.resize(static_cast<std::size_t>(tensor.byte_size));
            model.read_tensor(tensor.name, data);
        };
        read(*views.down, down); read(*views.up, up);
        if (views.inject) read(*views.inject, inject);
        std::vector<std::byte> bytes;
        read(*views.norm, bytes);
        norm.resize(static_cast<std::size_t>(views.norm->elements));
        for (std::size_t i = 0; i < norm.size(); ++i)
            norm[i] = qwen::tensor_element(TensorType::F32, bytes, i); // Direct GGUF gamma, NOT 1+gamma.
    }

    qwen::HcParameters parameters(const ActualViews& views) const {
        auto matrix = [](const qwen::TensorView& tensor, std::span<const std::byte> data) {
            return qwen::QMatrix{tensor.type, static_cast<int>(tensor.dimensions[0]),
                                 static_cast<int>(tensor.dimensions[1]), data};
        };
        return {matrix(*views.down, down), matrix(*views.up, up), norm,
                views.inject ? std::optional(matrix(*views.inject, inject)) : std::nullopt};
    }
};

// Exercise the new loaded-byte reference and prefix runner locally, even when
// the real GGUF exists only on the target machine. The pre-existing sparse
// FP64 oracle supplies an additional, independently encoded cross-check.
void loaded_reference_tests() {
    for (const auto type : {TensorType::F32, TensorType::Q4_0, TensorType::BF16}) {
        const HcConfig cfg = type == TensorType::Q4_0 ? HcConfig{8, 4, 32, 0.03125f} : HcConfig{5, 3, 7, 0.03125f};
        Weights w(cfg, type, TensorType::BF16);
        for (Matrix* matrix : {&w.down, &w.up, &w.inject})
            for (std::size_t row = 0; row < static_cast<std::size_t>(matrix->output); ++row)
                for (std::size_t col = 0; col < static_cast<std::size_t>(matrix->input); ++col)
                    matrix->set(row, col, static_cast<float>(static_cast<int>((row * 13 + col * 7) % 9) - 4) / 32.0f);
        for (std::size_t i = 0; i < w.norm.size(); ++i) w.norm[i] = static_cast<float>(3 + i % 11) / 8.0f;
        const auto input = actual_input(cfg);
        std::vector<std::size_t> features(cfg.hidden_size);
        std::iota(features.begin(), features.end(), std::size_t{0});
        const auto double_expected = reference(w, input, 3);
        const auto rounded = sampled_reference_f32(cfg, w.parameters(), input, 3, features);
        for (std::size_t i = 0; i < rounded.mixed.size(); ++i) near(static_cast<float>(rounded.mixed[i]), double_expected.mixed[i]);
        for (std::size_t i = 0; i < rounded.injection.size(); ++i) near(static_cast<float>(rounded.injection[i]), double_expected.injection[i]);
        const auto report = actual_case(cfg, w.parameters(type != TensorType::BF16), "synthetic_loaded_bytes", input, features);
        check(report.mix_samples == 3 * cfg.hidden_size && report.injection == (type != TensorType::BF16), "loaded oracle coverage");
        ++fixtures;
    }
    check(actual_dimension({qwen::MetadataType::UINT32, std::uint32_t{2560}}) == 2560, "HC UINT32 geometry metadata");
    check(actual_dimension({qwen::MetadataType::UINT64, std::uint64_t{320}}) == 320, "HC UINT64 geometry metadata");
    rejected([&] { (void)actual_dimension({qwen::MetadataType::UINT32, std::uint32_t{0}}); });
    rejected([&] { (void)actual_dimension({qwen::MetadataType::UINT64, std::uint64_t{1} << 32}); });
    rejected([&] { (void)actual_dimension({qwen::MetadataType::FLOAT32, 320.0f}); });
    qwen::TensorView tensor{};
    tensor.name = "preflight"; tensor.type = TensorType::Q4_0; tensor.rank = 2;
    tensor.dimensions = {32, 3, 0, 0}; tensor.elements = 96; tensor.byte_size = 54;
    const std::array<std::uint64_t, 2> dimensions{32, 3};
    validate_actual_tensor(tensor, TensorType::Q4_0, dimensions);
    auto bad = tensor; bad.type = TensorType::BF16;
    rejected([&] { validate_actual_tensor(bad, TensorType::Q4_0, dimensions); });
    bad = tensor; bad.rank = 3;
    rejected([&] { validate_actual_tensor(bad, TensorType::Q4_0, dimensions); });
    bad = tensor; bad.dimensions[0] = 64;
    rejected([&] { validate_actual_tensor(bad, TensorType::Q4_0, dimensions); });
    bad = tensor; --bad.byte_size;
    rejected([&] { validate_actual_tensor(bad, TensorType::Q4_0, dimensions); });
    bad = tensor; --bad.elements;
    rejected([&] { validate_actual_tensor(bad, TensorType::Q4_0, dimensions); });
}

void json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::cout << '"';
    for (const unsigned char byte : value) {
        if (byte == '"' || byte == '\\') std::cout << '\\' << static_cast<char>(byte);
        else if (byte < 32) std::cout << "\\u00" << hex[byte >> 4] << hex[byte & 15];
        else std::cout << static_cast<char>(byte);
    }
    std::cout << '"';
}

void actual_model(const char* path) {
    const qwen::Model model(path);
    const auto& architecture = model.metadata_value("general.architecture");
    if (architecture.type != qwen::MetadataType::STRING || architecture.get<std::string>() != "qwen4exp")
        throw std::invalid_argument("HC fixture: expected qwen4exp architecture");
    HcConfig cfg;
    cfg.hidden_size = actual_dimension(model.metadata_value("qwen4exp.embedding_length"));
    cfg.branches = actual_dimension(model.metadata_value("qwen4exp.hyper_connection.count"));
    cfg.low_rank = actual_dimension(model.metadata_value("qwen4exp.hyper_connection.low_rank"));
    const auto& epsilon = model.metadata_value("qwen4exp.attention.layer_norm_rms_epsilon");
    if (epsilon.type != qwen::MetadataType::FLOAT32)
        throw std::invalid_argument("HC fixture: RMS epsilon metadata must be FLOAT32");
    cfg.rms_epsilon = epsilon.get<float>();
    const auto wide = cfg.widened_elements();
    check(cfg.hidden_size == 2560 && cfg.branches == 4 && cfg.low_rank == 320,
          "HC actual fixture requires validated target H2560 C4 R320 geometry");
    // Preflight ALL eleven selected tensors before reading ANY payload.
    const std::array views{actual_views(model, cfg, "blk.0.hc_attn_", true),
                           actual_views(model, cfg, "blk.0.hc_ffn_", true),
                           actual_views(model, cfg, "output_hc_", false)};
    const std::array<std::string_view, 3> names{"blk.0.hc_attn", "blk.0.hc_ffn", "output_hc"};
    const std::array<std::size_t, 10> features{0, 1, 15, 16, 31, 32, 63, cfg.hidden_size / 3, cfg.hidden_size / 2, cfg.hidden_size - 1};
    const auto input = actual_input(cfg);
    std::array<ActualCase, 3> reports{};
    for (std::size_t i = 0; i < views.size(); ++i) {
        // Borrowed spans never escape this iteration; only one mixer's payload
        // and scratch are resident at a time, no full-model dequantization.
        const ActualWeights weights(model, views[i]);
        reports[i] = actual_case(cfg, weights.parameters(views[i]), names[i], input, features);
    }
    std::cout << "{\"kind\":\"hc_actual_model\",\"passed\":true,\"actual_model\":true,\"model_path\":";
    json_string(path);
    std::cout << ",\"model_file_bytes\":" << model.file_size()
              << ",\"scope\":\"actual blk0 attention/FFN and root HC weights; synthetic widened residual; sampled HF oracle, not full inference\""
              << ",\"reference\":\"Transformers a005fc82babfe8871d87746decad2dbee100a125 Qwen4ExpTextGatedResidual:551-563\""
              << ",\"expected_arithmetic\":\"double HF equations with explicit per-operation FP32 rounding, ascending-k reductions\""
              << ",\"tolerance\":{\"absolute\":4e-6,\"relative\":4e-6,\"rule\":\"error <= absolute + relative * abs(expected)\"}"
              << ",\"norm_convention\":\"stored GGUF gamma, already 1+HF weight; RMS per H-sized branch\""
              << ",\"geometry\":{\"hidden_size\":" << cfg.hidden_size << ",\"branches\":" << cfg.branches
              << ",\"widened_elements\":" << wide << ",\"low_rank\":" << cfg.low_rank << ",\"rms_epsilon\":" << cfg.rms_epsilon << '}'
              << ",\"sampled_features\":[";
    for (std::size_t i = 0; i < features.size(); ++i) std::cout << (i == 0 ? "" : ",") << features[i];
    std::cout << "],\"cases\":[";
    for (std::size_t i = 0; i < reports.size(); ++i) {
        const auto& r = reports[i];
        std::cout << (i == 0 ? "" : ",") << "{\"mixer\":";
        json_string(r.mixer);
        std::cout << ",\"tokens\":[1,2,3],\"injection\":" << (r.injection ? "true" : "false")
                  << ",\"down_up_dtype\":\"" << (r.injection ? "Q4_0" : "BF16") << "\",\"norm_dtype\":\"F32\",\"inject_dtype\":"
                  << (r.injection ? "\"BF16\"" : "null")
                  << ",\"reference_tokens\":3,\"down_rows_per_token\":" << cfg.low_rank
                  << ",\"up_rows_per_token\":" << features.size() * cfg.branches
                  << ",\"mix_samples\":" << r.mix_samples << ",\"injection_samples\":" << r.injection_samples
                  << ",\"combine_samples\":" << r.combine_samples
                  << ",\"mix_max_abs_error\":" << r.mix_error << ",\"injection_max_abs_error\":" << r.injection_error
                  << ",\"combine_max_abs_error\":" << r.combine_error
                  << ",\"prefix_exact\":true,\"residual_preserved\":true,\"finite_outputs\":true,\"success_call_allocations\":0}";
    }
    std::cout << "],\"token_cases\":9,\"root_injection_rejected\":true,\"mtp_tap_preserved\":true}\n";
}

void argument_failures() {
    Weights w({3, 2, 4, 1.0e-6f});
    for (const HcConfig cfg : {HcConfig{0, 2, 4, 1.0e-6f}, HcConfig{3, 0, 4, 1.0e-6f},
                              HcConfig{3, 1, 4, 1.0e-6f}, HcConfig{3, 2, 0, 1.0e-6f},
                              HcConfig{3, 2, 4, 0}, HcConfig{3, 2, 4, -1},
                              HcConfig{3, 2, 4, std::numeric_limits<float>::infinity()},
                              HcConfig{3, 2, 4, std::numeric_limits<float>::quiet_NaN()},
                              HcConfig{std::numeric_limits<std::size_t>::max(), 2, 4, 1.0e-6f},
                              HcConfig{3, 2, std::numeric_limits<std::size_t>::max(), 1.0e-6f}})
        rejected([&] { qwen::HcCpu cpu(cfg, w.parameters()); });
    rejected([&] { qwen::HcCpu cpu(w.config, w.parameters(), 0); });
    rejected([&] { qwen::HcCpu cpu(w.config, w.parameters(), std::numeric_limits<std::size_t>::max()); });
    auto params = w.parameters();
    params.down.input = 5;
    rejected([&] { qwen::HcCpu cpu(w.config, params); });
    params = w.parameters(); params.up.output = 5;
    rejected([&] { qwen::HcCpu cpu(w.config, params); });
    params = w.parameters(); params.inject->output = 3;
    rejected([&] { qwen::HcCpu cpu(w.config, params); });
    params = w.parameters(); params.norm_weight = std::span(w.norm).first(5);
    rejected([&] { qwen::HcCpu cpu(w.config, params); });
    params = w.parameters(); params.down.weights = params.down.weights.first(params.down.weights.size() - 1);
    rejected([&] { qwen::HcCpu cpu(w.config, params); });
    params = w.parameters(); params.up.type = static_cast<TensorType>(99);
    rejected([&] { qwen::HcCpu cpu(w.config, params); });
    params = w.parameters(); params.down.type = TensorType::Q4_0;
    rejected([&] { qwen::HcCpu cpu(w.config, params); });
    params = w.parameters(); params.norm_weight = {static_cast<const float*>(nullptr), 6};
    rejected([&] { qwen::HcCpu cpu(w.config, params); });
    for (const float nonfinite : {std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN()}) {
        w.norm.back() = nonfinite;
        rejected([&] { qwen::HcCpu cpu(w.config, w.parameters()); });
        w.norm.back() = 1;
        for (Matrix* matrix : {&w.down, &w.up, &w.inject}) {
            matrix->set(static_cast<std::size_t>(matrix->output - 1),
                        static_cast<std::size_t>(matrix->input - 1), nonfinite);
            rejected([&] { qwen::HcCpu cpu(w.config, w.parameters()); });
            matrix->set(static_cast<std::size_t>(matrix->output - 1),
                        static_cast<std::size_t>(matrix->input - 1), 0);
        }
    }
    qwen::HcCpu cpu(w.config, w.parameters(), 2), head(w.config, w.parameters(false), 2);
    std::vector<float> x(12, 1), mixed(6, 53), gates(4, 53), out(12, 53), block(6, 2);
    const auto old_mixed = mixed, old_gates = gates, old_out = out, old_x = x;
    auto unchanged = [&] { same(mixed, old_mixed); same(gates, old_gates); same(out, old_out); };
    auto reject_call = [&](auto function) { rejected(function); unchanged(); };
    reject_call([&] { cpu.mix(x, 0, mixed, gates); });
    reject_call([&] { cpu.mix(x, 3, mixed, gates); });
    reject_call([&] { cpu.mix(std::span(x).first(11), 2, mixed, gates); });
    reject_call([&] { cpu.mix(x, 2, std::span(mixed).first(5), gates); });
    reject_call([&] { cpu.mix(x, 2, mixed, std::span(gates).first(3)); });
    reject_call([&] { cpu.mix(x, 2, mixed); });
    reject_call([&] { head.mix(x, 2, mixed, gates); });
    reject_call([&] { head.combine(x, block, gates, 2, out); });
    reject_call([&] { cpu.mix(x, 2, std::span(x).first(6), gates); });
    same(x, old_x);
    reject_call([&] { cpu.mix(x, 2, mixed, std::span(mixed).first(4)); });
    reject_call([&] { cpu.mix(std::span(x).first(6), 1,
                             std::span(w.norm).first(3), std::span(gates).first(2)); });
    reject_call([&] { cpu.mix(std::span(x).first(6), 1,
                             {reinterpret_cast<float*>(w.down.bytes.data()), 3}, std::span(gates).first(2)); });
    for (Matrix* matrix : {&w.up, &w.inject})
        reject_call([&] { cpu.mix(x, 2, mixed,
                                 {reinterpret_cast<float*>(matrix->bytes.data()), 4}); });
    // partial overlap, not just equal starting pointers
    reject_call([&] { cpu.mix(x, 2, std::span(x).subspan(3, 6), gates); });
    reject_call([&] { cpu.combine(x, block, std::span(gates).first(3), 2, out); });
    reject_call([&] { cpu.combine(x, std::span(block).first(5), gates, 2, out); });
    reject_call([&] { cpu.combine(x, block, gates, 2, std::span(out).first(11)); });
    reject_call([&] { cpu.combine(x, block, gates, 2, x); });
    reject_call([&] { cpu.combine(x, std::span(out).first(6), gates, 2, out); });
    reject_call([&] { cpu.combine(x, block, std::span(out).first(4), 2, out); });
    reject_call([&] { cpu.expand(std::span(block).first(5), 2, out); });
    reject_call([&] { cpu.expand(block, 2, std::span(out).first(11)); });
    reject_call([&] { cpu.expand(std::span(x).first(6), 2, x); });
    for (const float nonfinite : {std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN()}) {
        x.back() = nonfinite;
        reject_call([&] { cpu.mix(x, 2, mixed, gates); });
        reject_call([&] { cpu.combine(x, block, gates, 2, out); });
        x.back() = 1;
        block.back() = nonfinite;
        reject_call([&] { cpu.expand(block, 2, out); });
        reject_call([&] { cpu.combine(x, block, gates, 2, out); });
        block.back() = 2;
    }
    for (const float gate : {-0.00001f, 2.00001f, std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::quiet_NaN()}) {
        auto bad_gates = std::vector<float>(4, 1);
        bad_gates.back() = gate;
        reject_call([&] { cpu.combine(x, block, bad_gates, 2, out); });
    }
    // All failing calls left the instance usable; output's prior NaN contents
    // are permitted, because every successful result is fully overwritten.
    std::fill(mixed.begin(), mixed.end(), std::numeric_limits<float>::quiet_NaN());
    cpu.mix(x, 2, mixed, gates);
    near(mixed, reference(w, x, 2).mixed);
    check(std::all_of(gates.begin(), gates.end(), [](float v) { return v == 1; }), "reuse after bad calls");
    // Read-only overlap is valid. The block and activated gates here are views
    // into the original residual, while only the separate output is written.
    const auto block_view = std::span<const float>(x).first(6);
    const auto gate_view = std::span<const float>(x).first(4);
    cpu.combine(x, block_view, gate_view, 2, out);
    check(std::all_of(out.begin(), out.end(), [](float v) { return v == 2; }), "read-only aliases allowed");
    same(x, old_x);
}

void arithmetic_failures() {
    const float huge = std::numeric_limits<float>::max();
    for (int stage = 0; stage < 5; ++stage) {
        Weights w({1, 3, 2, 1.0e-6f});
        if (stage == 1) { w.norm[0] = 2; w.down.set(0, 0, huge); }
        if (stage == 2) { w.down.set(0, 0, 32); w.up.set(2, 0, huge); }
        if (stage == 3) { w.norm[0] = 2; w.inject.set(2, 0, huge); }
        if (stage == 4) std::fill(w.norm.begin(), w.norm.end(), huge);
        qwen::HcCpu cpu(w.config, w.parameters(), 2);
        // First token is zero and safe. The later token fails in norm/down/up/
        // inject/branch reduction; even that first token must not be published.
        std::array<float, 6> x{0, 0, 0, 1, 1, 1};
        if (stage == 0) x.back() = huge;
        std::array<float, 2> mixed{53, 53};
        std::array<float, 6> gates{53, 53, 53, 53, 53, 53};
        const auto old_mixed = mixed;
        const auto old_gates = gates;
        arithmetic_rejected([&] { cpu.mix(x, 2, mixed, gates); });
        same(mixed, old_mixed); same(gates, old_gates);
        x.fill(0);
        cpu.mix(x, 2, mixed, gates);
        check(mixed == std::array<float, 2>{0, 0}, "reuse after arithmetic failure");
        for (const float gate : gates) check(gate == 1, "zero injection after arithmetic failure");
    }
    Weights w({2, 3, 2, 1.0e-6f});
    qwen::HcCpu cpu(w.config, w.parameters(), 2);
    std::array<float, 12> x{}, out{};
    std::array<float, 4> block{0, 0, huge, huge};
    std::array<float, 6> gates{1, 1, 1, 1, 1, 2};
    out.fill(53);
    const auto old = out;
    arithmetic_rejected([&] { cpu.combine(x, block, gates, 2, out); });
    same(out, old);
    gates.fill(1);
    x.back() = huge;
    arithmetic_rejected([&] { cpu.combine(x, block, gates, 2, out); });
    same(out, old);
    x.fill(0); block.fill(1);
    cpu.combine(x, block, gates, 2, out);
    for (const float value : out) check(value == 1, "combine reuse after failure");
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool has_model = argc == 3 && std::string_view(argv[1]) == "--model";
        if (argc != 1 && !has_model) throw std::invalid_argument("usage: hc-test [--model GGUF]");
        std::cout << std::setprecision(9);
        rectangular();
        constants_and_epsilon();
        gate_extremes();
        real_geometry();
        argument_failures();
        arithmetic_failures();
        loaded_reference_tests();
        if (has_model) actual_model(argv[2]);
        std::cout << "{\"kind\":\"hc_cpu\",\"passed\":true,\"checks\":" << checks
                  << ",\"rejections\":" << rejections
                  << ",\"arithmetic_rejections\":" << arithmetic_rejections
                  << ",\"fixtures\":" << fixtures << ",\"max_abs_error\":" << max_abs_error
                  << ",\"success_call_allocations\":0,\"real_geometry\":\"4x2560/rank320\",\"actual_model\":"
                  << (has_model ? "true" : "false") << ",\"actual_cases\":" << (has_model ? 9 : 0)
                  << ",\"revision\":\"" << CORE_REVISION << "\",\"dirty\":" << CORE_DIRTY << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "hc-test: " << error.what() << '\n';
        return 1;
    }
}
