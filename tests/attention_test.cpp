#include "attention.hpp"

#include <algorithm>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
std::size_t allocations = 0;
[[gnu::noinline]] void* allocate(std::size_t n) {
    ++allocations;
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void* allocate_aligned(std::size_t n, std::size_t alignment) {
    ++allocations;
    void* p = nullptr;
    if (posix_memalign(&p, alignment, n == 0 ? 1 : n) == 0) return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void release(void* p) noexcept { std::free(p); }
}
void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void operator delete(void* p) noexcept { release(p); }
void operator delete[](void* p) noexcept { release(p); }
void operator delete(void* p, std::size_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t) noexcept { release(p); }
void* operator new(std::size_t n, std::align_val_t a) { return allocate_aligned(n, static_cast<std::size_t>(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocate_aligned(n, static_cast<std::size_t>(a)); }
void operator delete(void* p, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::align_val_t) noexcept { release(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { release(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { release(p); }

namespace {
using qwen::AttentionQ4Mode;
std::size_t checks = 0, rejections = 0, arithmetic_rejections = 0, fixtures = 0;
double maximum_error = 0.0;
double maximum_gathered_error = 0.0, maximum_direct_error = 0.0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejected(F f) {
    try { f(); }
    catch (const std::invalid_argument&) { ++rejections; return; }
    throw std::runtime_error("invalid attention argument accepted");
}
template<class F> void arithmetic_rejected(F f) {
    try { f(); }
    catch (const std::runtime_error&) { ++arithmetic_rejections; return; }
    throw std::runtime_error("nonfinite attention arithmetic accepted");
}

// Independent binary16 decoder and RNE rounding, using exact double powers of
// two. Neither reference calls quant/kv/attention helpers nor copies their
// bit-conversion algorithm. Q4 scale*integer fits FP32 exactly before rounding.
double half(std::uint16_t bits) {
    const int exponent = (bits >> 10) & 31, fraction = bits & 1023;
    require(exponent != 31, "reference finite half required");
    const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(fraction), -24) :
        std::ldexp(static_cast<double>(1024 + fraction), exponent - 25);
    return bits & 0x8000 ? -magnitude : magnitude;
}
double round_even(double x) {
    const double lower = std::floor(x), fraction = x - lower;
    return lower + (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0) ? 1.0 : 0.0);
}
double half_roundtrip(double x) {
    int exponent = 0;
    const double magnitude = std::abs(x);
    if (magnitude == 0) return x;
    (void)std::frexp(magnitude, &exponent);
    const double step = std::ldexp(1.0, std::max(-24, exponent - 11));
    const double rounded = round_even(magnitude / step) * step;
    return std::copysign(rounded >= 65536.0 ? std::numeric_limits<double>::infinity() : rounded, x);
}
double decode(const qwen::Q4_0* row, int d, AttentionQ4Mode mode) {
    const auto& b = row[d / 32];
    const unsigned shift = d % 32 < 16 ? 0 : 4;
    const int code = (b.qs[d % 16] >> shift) & 15;
    const double value = half(b.d) * static_cast<double>(code - 8);
    return mode == AttentionQ4Mode::gathered_fp16 ? half_roundtrip(value) : value;
}

struct Fixture {
    std::size_t capacity, visible;
    std::vector<qwen::Q4_0> keys, values;
    std::vector<float> query, scores, staged, output;
    std::vector<std::int32_t> ids;
    Fixture(std::size_t cap, std::size_t count) : capacity(cap), visible(cap),
        keys(cap * 16), values(cap * 16), query(qwen::attention_query_elements),
        scores(count), staged(qwen::attention_query_elements), output(qwen::attention_query_elements), ids(count) {
        for (std::size_t t = 0; t < cap; ++t) for (std::size_t b = 0; b < 16; ++b) {
            auto& k = keys[t * 16 + b]; auto& v = values[t * 16 + b];
            // Signed, deliberately non-power-of-two scales; both nibble halves
            // vary independently and the two KV heads have distinct content.
            k.d = static_cast<std::uint16_t>(0x2803 + ((t * 19 + b * 31) % 511) + ((t + b) % 2 ? 0x8000 : 0));
            v.d = static_cast<std::uint16_t>(0x2c01 + ((t * 7 + b * 43) % 701) + ((t + 2 * b) % 3 ? 0x8000 : 0));
            for (std::size_t j = 0; j < 16; ++j) {
                k.qs[j] = static_cast<std::uint8_t>(((t * 5 + b + j * 3) % 16) | (((t + b * 7 + j * 11) % 16) << 4));
                v.qs[j] = static_cast<std::uint8_t>(((t + b * 13 + j * 7) % 16) | (((t * 3 + b * 5 + j) % 16) << 4));
            }
        }
        for (std::size_t i = 0; i < query.size(); ++i)
            query[i] = static_cast<float>(0.7 * std::sin(static_cast<double>(i) * 0.173) + 0.1 * std::cos(static_cast<double>(i) * 0.037));
        for (std::size_t j = 0; j < count; ++j) ids[j] = static_cast<std::int32_t>((j * 17 + 3) % cap);
    }
    qwen::AttentionQ4Cache cache() const { return {keys, values, capacity}; }
    qwen::AttentionQ4Scratch scratch() { return {scores, staged}; }
    void call(AttentionQ4Mode mode = AttentionQ4Mode::gathered_fp16) {
        const auto before = allocations;
        qwen::attention_q4(cache(), query, ids, visible, mode, scratch(), output);
        require(allocations == before, "successful hot attention allocated");
    }
    void sentinel() {
        std::fill(scores.begin(), scores.end(), -17.0f);
        std::fill(staged.begin(), staged.end(), -19.0f);
        std::fill(output.begin(), output.end(), -23.0f);
    }
    void unchanged(bool include_scratch = true) const {
        for (float v : output) require(v == -23.0f, "failed attention changed output");
        if (include_scratch) {
            for (float v : staged) require(v == -19.0f, "invalid attention changed staged output");
            for (float v : scores) require(v == -17.0f, "invalid attention changed score scratch");
        }
    }
    template<class F> void invalid(F f) { sentinel(); rejected(f); unchanged(); }
};

// Independent double attention equations. Only selected rows are materialized,
// in test-owned memory, to make the long-count oracle reasonably inexpensive.
std::vector<double> reference(const Fixture& f, AttentionQ4Mode mode) {
    const auto n = f.ids.size();
    std::vector<double> k(n * 512), v(n * 512), probabilities(n), out(6144);
    for (std::size_t j = 0; j < n; ++j) for (int d = 0; d < 512; ++d) {
        k[j * 512 + d] = decode(f.keys.data() + static_cast<std::size_t>(f.ids[j]) * 16, d, mode);
        v[j * 512 + d] = decode(f.values.data() + static_cast<std::size_t>(f.ids[j]) * 16, d, mode);
    }
    for (int h = 0; h < 24; ++h) {
        const int kv = h < 12 ? 0 : 1; // Independent explicit two-group mapping.
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < n; ++j) {
            double score = 0;
            for (int d = 0; d < 256; ++d) score += static_cast<double>(f.query[h * 256 + d]) * k[j * 512 + kv * 256 + d];
            probabilities[j] = score / std::sqrt(256.0);
            maximum = std::max(maximum, probabilities[j]);
        }
        double denominator = 0;
        for (double& p : probabilities) { p = std::exp(p - maximum); denominator += p; }
        for (int d = 0; d < 256; ++d) {
            double weighted = 0;
            for (std::size_t j = 0; j < n; ++j) weighted += probabilities[j] * v[j * 512 + kv * 256 + d];
            out[h * 256 + d] = weighted / denominator;
        }
    }
    return out;
}

void parity(Fixture& f, AttentionQ4Mode mode) {
    const auto original_keys = f.keys, original_values = f.values;
    const auto original_query = f.query; const auto original_ids = f.ids;
    const auto expected = reference(f, mode);
    f.call(mode);
    ++fixtures;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double error = std::abs(static_cast<double>(f.output[i]) - expected[i]);
        maximum_error = std::max(maximum_error, error);
        auto& mode_error = mode == AttentionQ4Mode::gathered_fp16 ? maximum_gathered_error : maximum_direct_error;
        mode_error = std::max(mode_error, error);
        // Frozen before execution: stricter CPU FP32-vs-double gate, NOT the
        // parent's GPU common-gather 2e-4 absolute + 2e-4 relative gate.
        require(std::isfinite(f.output[i]) && error <= 2e-5 + 2e-5 * std::abs(expected[i]), "attention differs from independent double oracle");
    }
    require(std::memcmp(original_keys.data(), f.keys.data(), f.keys.size() * sizeof(qwen::Q4_0)) == 0, "K cache mutated");
    require(std::memcmp(original_values.data(), f.values.data(), f.values.size() * sizeof(qwen::Q4_0)) == 0, "V cache mutated");
    require(original_query == f.query && original_ids == f.ids, "query or selected IDs mutated");
}

void numeric_fixtures() {
    for (std::size_t n : {1U, 2U, 3U, 5U, 7U, 31U, 63U, 64U, 65U, 127U, 128U, 129U, 2048U, 2049U, 2050U, 2051U}) {
        Fixture f(n + 7, n);
        for (auto mode : {AttentionQ4Mode::gathered_fp16, AttentionQ4Mode::direct_fp32}) parity(f, mode);
    }
    Fixture f(8, 5);
    f.ids = {6, 0, 3, 6, 2}; // Sparse, unsorted, repeated, no inferred mask.
    parity(f, AttentionQ4Mode::gathered_fp16);
    const auto gathered = f.output;
    parity(f, AttentionQ4Mode::direct_fp32);
    require(f.output != gathered, "fixture must distinguish direct FP32 from gathered FP16");
    std::fill(f.query.begin(), f.query.end(), 0.0f);
    parity(f, AttentionQ4Mode::gathered_fp16); // Uniform softmax, ascending mean.
    std::fill(f.query.begin(), f.query.end(), 1000.0f);
    parity(f, AttentionQ4Mode::gathered_fp16); // Stable softmax at large scores.
    Fixture subnormal(3, 3);
    for (auto& b : subnormal.keys) b.d = 0x0001;
    for (auto& b : subnormal.values) b.d = 0x8001;
    parity(subnormal, AttentionQ4Mode::gathered_fp16);
    for (auto& b : subnormal.keys) b.d = 0x8000;
    for (auto& b : subnormal.values) b.d = 0x0000;
    parity(subnormal, AttentionQ4Mode::gathered_fp16);
}

void grouped_heads() {
    Fixture f(3, 1); f.ids[0] = 1;
    for (std::size_t b = 0; b < 16; ++b) {
        auto& k = f.keys[16 + b]; auto& v = f.values[16 + b];
        k.d = 0; v.d = 0x3400; // 1/4
        std::fill(std::begin(k.qs), std::end(k.qs), 0x88);
        std::fill(std::begin(v.qs), std::end(v.qs), b < 8 ? 0x66 : 0xbb);
    }
    f.call();
    for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d)
        require(f.output[h * 256 + d] == (h < 12 ? -0.5f : 0.75f), "GQA must map h/12, not h%2");
}

void invalid_inputs() {
    Fixture f(7, 3); f.visible = 5; f.ids = {0, 3, 4};
    auto run = [&] { qwen::attention_q4(f.cache(), f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); };
    for (std::int32_t id : {-1, 5, 7, std::numeric_limits<std::int32_t>::max(), std::numeric_limits<std::int32_t>::min()}) {
        f.ids.back() = id; f.invalid(run);
    }
    f.ids.back() = 4;
    for (float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
        f.query.back() = bad; f.invalid(run);
    }
    f.query.back() = 0.0f;
    for (bool key : {false, true}) for (std::uint16_t bits : {0x7c00, 0xfc00, 0x7e00, 0x7c01}) {
        auto& b = (key ? f.keys : f.values)[4 * 16 + 15];
        const auto old = b.d; b.d = bits;
        std::feclearexcept(FE_ALL_EXCEPT);
        f.invalid(run);
        require((std::fetestexcept(FE_INVALID) & FE_INVALID) == 0, "invalid half scale decoded/arithmetic before rejection");
        b.d = old;
    }
    for (std::size_t cap : {0U, 131073U}) {
        auto cache = f.cache(); cache.capacity = cap;
        f.invalid([&] { qwen::attention_q4(cache, f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    }
    for (std::size_t visible : {0U, 8U})
        f.invalid([&] { qwen::attention_q4(f.cache(), f.query, f.ids, visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    auto cache = f.cache(); cache.keys = cache.keys.first(cache.keys.size() - 1);
    f.invalid([&] { qwen::attention_q4(cache, f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    cache = f.cache(); cache.values = cache.values.first(cache.values.size() - 1);
    f.invalid([&] { qwen::attention_q4(cache, f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    f.invalid([&] { qwen::attention_q4(f.cache(), std::span(f.query).first(6143), f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    f.invalid([&] { qwen::attention_q4(f.cache(), f.query, {}, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    f.invalid([&] { qwen::attention_q4(f.cache(), f.query, f.ids, f.visible, static_cast<AttentionQ4Mode>(99), f.scratch(), f.output); });
    f.invalid([&] { qwen::attention_q4(f.cache(), f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, {std::span(f.scores).first(2), f.staged}, f.output); });
    f.invalid([&] { qwen::attention_q4(f.cache(), f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, {f.scores, std::span(f.staged).first(6143)}, f.output); });
    f.invalid([&] { qwen::attention_q4(f.cache(), f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), std::span(f.output).first(6143)); });
    f.ids.resize(2052, 0); f.scores.resize(2052);
    f.invalid(run);
    f.ids.resize(3); f.scores.resize(3);

    // Every positive count is in the public shape domain, even non-QSA-budget
    // counts. A late invalid query must be the rejecting operand, not the count.
    std::vector<std::int32_t> ids(2051, 0); std::vector<float> scores(2051, -17);
    f.query.back() = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t n = 1; n <= 2051; ++n) {
        try {
            qwen::attention_q4(f.cache(), f.query, std::span(ids).first(n), f.visible,
                AttentionQ4Mode::gathered_fp16, {std::span(scores).first(n), f.staged}, f.output);
            throw std::runtime_error("late invalid query accepted");
        } catch (const std::invalid_argument& e) {
            ++rejections;
            require(std::string_view(e.what()) == "attention: nonfinite query", "legal selected count rejected as a shape");
        }
    }
    f.unchanged();
    for (float v : scores) require(v == -17, "late query validation changed score scratch");
    f.query.back() = 0;

    // Invalid *unselected* and future rows are deliberately not decoded/read.
    for (int t : {1, 2, 5, 6}) for (int b = 0; b < 16; ++b) {
        f.keys[t * 16 + b].d = 0x7c01; f.values[t * 16 + b].d = 0x7e00;
    }
    parity(f, AttentionQ4Mode::gathered_fp16);
}

void aliases_and_ranges() {
    Fixture f(7, 3);
    auto invoke = [&](std::span<const float> q, qwen::AttentionQ4Scratch scratch, std::span<float> output) {
        qwen::attention_q4(f.cache(), q, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, scratch, output);
    };
    f.invalid([&] { invoke(f.query, f.scratch(), f.query); });
    f.invalid([&] { invoke(f.query, f.scratch(), f.staged); });
    f.invalid([&] { invoke(f.query, {std::span(f.query).first(3), f.staged}, f.output); });
    f.invalid([&] { invoke(f.query, {std::span(f.output).first(3), f.staged}, f.output); });
    f.invalid([&] { invoke(f.query, {std::span(f.staged).first(3), f.staged}, f.output); });
    f.invalid([&] { invoke(f.query, {f.scores, f.query}, f.output); });
    f.invalid([&] { invoke(f.query, {f.scores, f.output}, f.output); });
    f.invalid([&] { invoke(f.query, {std::span(reinterpret_cast<float*>(f.keys.data()), 3), f.staged}, f.output); });
    f.invalid([&] { invoke(f.query, f.scratch(), {reinterpret_cast<float*>(f.values.data()), 6144}); });
    f.invalid([&] { invoke(f.query, {std::span(reinterpret_cast<float*>(f.ids.data()), 3), f.staged}, f.output); });
    const auto* misaligned = reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(f.query.data()) + 1);
    f.invalid([&] { invoke({misaligned, 6144}, f.scratch(), f.output); });
    const auto* wrapping = reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max() & ~std::uintptr_t{3});
    f.invalid([&] { invoke({wrapping, 6144}, f.scratch(), f.output); });
    // Read-only overlaps are legal: the very same canonical rows as K and V.
    f.values = f.keys;
    const auto expected = reference(f, AttentionQ4Mode::gathered_fp16);
    auto cache = f.cache(); cache.values = cache.keys;
    qwen::attention_q4(cache, f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output);
    for (std::size_t i = 0; i < expected.size(); ++i)
        require(std::abs(static_cast<double>(f.output[i]) - expected[i]) < 2e-5, "legal read-only overlap failed");
}

void overflow_and_reuse() {
    Fixture f(5, 3); f.ids = {0, 2, 4};
    std::fill(f.query.begin(), f.query.end(), 0.0f);
    auto& b = f.values[4 * 16 + 15];
    b.d = 0x7bff; std::fill(std::begin(b.qs), std::end(b.qs), 0x00);
    f.invalid([&] { qwen::attention_q4(f.cache(), f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    parity(f, AttentionQ4Mode::direct_fp32); // Large finite Q4 value, half-only overflow.
    b.d = 0;
    auto& k = f.keys[4 * 16 + 15];
    k.d = 0x7bff; std::fill(std::begin(k.qs), std::end(k.qs), 0x00);
    f.invalid([&] { qwen::attention_q4(f.cache(), f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    k.d = 0x3c00; k.qs[15] = 0xf8;
    f.query.back() = std::numeric_limits<float>::max(); // Last Q head, last dim.
    f.sentinel();
    arithmetic_rejected([&] { qwen::attention_q4(f.cache(), f.query, f.ids, f.visible, AttentionQ4Mode::gathered_fp16, f.scratch(), f.output); });
    f.unchanged(false);
    require(std::any_of(f.staged.begin(), f.staged.end(), [](float v) { return v != -19; }), "fixture did not reach late arithmetic failure");
    f.query.back() = 0;
    parity(f, AttentionQ4Mode::gathered_fp16); // Immediate reuse after rejection.
    Fixture maximum_capacity(131072, 1);
    maximum_capacity.ids[0] = 131071;
    parity(maximum_capacity, AttentionQ4Mode::gathered_fp16);
}
} // namespace

int main() {
    try {
        numeric_fixtures(); grouped_heads(); invalid_inputs(); aliases_and_ranges(); overflow_and_reuse();
        std::cout << "{\"test\":\"attention\",\"checks\":" << checks << ",\"rejections\":" << rejections
                  << ",\"arithmetic_rejections\":" << arithmetic_rejections << ",\"fixtures\":" << fixtures
                  << ",\"max_abs_error\":" << maximum_error << ",\"gathered_max_abs_error\":" << maximum_gathered_error
                  << ",\"direct_max_abs_error\":" << maximum_direct_error << ",\"passed\":true}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
