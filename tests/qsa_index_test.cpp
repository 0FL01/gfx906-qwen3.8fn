#include "qsa_index.hpp"
#include "qsa.hpp"
#include "kv.hpp"
#include "dense.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
bool count_allocations = false;
std::size_t allocations = 0;
}

// The allocation assertion covers successful numeric hot calls, not exceptions
// or the existing diagnostic qsa_select (which explicitly allocates).
[[gnu::noinline]] void* operator new(std::size_t bytes) {
    if (count_allocations) ++allocations;
    if (void* p = std::malloc(bytes == 0 ? 1 : bytes)) return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
std::size_t checks = 0, rejects = 0, numeric_rejects = 0, selections = 0, restores = 0;
double max_key_error = 0.0, max_score_error = 0.0, max_roundtrip_error = 0.0;

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejected(F fn) {
    try { fn(); } catch (const std::invalid_argument&) { ++rejects; return; }
    throw std::runtime_error("invalid input unexpectedly accepted");
}
template<class F> void numerical_rejected(F fn) {
    try { fn(); } catch (const std::runtime_error&) { ++numeric_rejects; return; }
    throw std::runtime_error("nonfinite arithmetic unexpectedly accepted");
}
void exact(std::span<const float> a, std::span<const float> b, const char* message) {
    check(a.size() == b.size(), message);
    for (std::size_t i = 0; i < a.size(); ++i)
        check(std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i]), message);
}
void near(std::span<const float> a, std::span<const float> b, double& maximum, const char* message) {
    check(a.size() == b.size(), message);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double error = std::fabs(static_cast<double>(a[i]) - b[i]);
        maximum = std::max(maximum, error);
        // Same scalar FP32 contract. No tolerance introduced to hide a source or
        // arithmetic disagreement; independent dense oracle must be bit-exact.
        check(error == 0.0, message);
    }
}

struct Fixture {
    qwen::QsaIndexConfig config;
    std::array<float, 128> qweight{}, kweight{};
    std::vector<float> frequencies;
    explicit Fixture(std::size_t capacity = 131072, std::size_t rotary = 64) {
        config.capacity = capacity;
        config.rotary_dim = rotary;
        config.rms_epsilon = 1.0e-6f;
        for (std::size_t i = 0; i < 128; ++i) {
            qweight[i] = 0.5f + static_cast<float>(i % 17) / 16.0f;
            kweight[i] = 0.75f + static_cast<float>((i * 7) % 19) / 32.0f;
        }
        // Explicit SYNTHETIC theta, not a claimed model metadata constant.
        // Formula is inherited HF default RoPE, using rotary_dim, not index D.
        frequencies.resize(rotary / 2);
        for (std::size_t i = 0; i < frequencies.size(); ++i)
            frequencies[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(rotary));
    }
    qwen::QsaIndexParameters parameters() const { return {qweight, kweight, frequencies}; }
};

std::vector<float> keys(std::size_t tokens, std::size_t salt = 0) {
    std::vector<float> result(tokens * 128);
    for (std::size_t t = 0; t < tokens; ++t) for (std::size_t d = 0; d < 128; ++d) {
        const int value = static_cast<int>((t * 71 + d * 29 + d * t * 3 + salt * 43) % 257) - 128;
        result[t * 128 + d] = static_cast<float>(value) / 64.0f;
    }
    return result;
}
std::array<float, 512> query(std::size_t salt = 0) {
    std::array<float, 512> result{};
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = static_cast<float>(static_cast<int>((i * 47 + salt * 17) % 193) - 96) / 32.0f;
    return result;
}

// Independent pack formula + dense canonical element reader, rather than the
// implementation's kv.hpp roundtrip. Read-only fixtures never get overwritten.
std::vector<float> roundtrip(std::span<const float> raw) {
    std::vector<qwen::Q4_0> packed(raw.size() / 32);
    for (std::size_t b = 0; b < packed.size(); ++b) {
        const auto row = raw.subspan(b * 32, 32);
        std::size_t winner = 0;
        for (std::size_t j = 1; j < 32; ++j)
            if (std::fabs(row[j]) > std::fabs(row[winner])) winner = j;
        const float scale = row[winner] / -8.0f;
        const float reciprocal = scale == 0.0f ? 0.0f : 1.0f / scale;
        packed[b].d = qwen::float_to_half(scale);
        for (std::size_t j = 0; j < 16; ++j) {
            const int lo = std::min(15, static_cast<int>(row[j] * reciprocal + 8.5f));
            const int hi = std::min(15, static_cast<int>(row[j + 16] * reciprocal + 8.5f));
            packed[b].qs[j] = static_cast<std::uint8_t>(lo + 16 * hi);
        }
    }
    const auto bytes = std::as_bytes(std::span(packed));
    std::vector<float> result(raw.size());
    for (std::size_t i = 0; i < result.size(); ++i)
        result[i] = qwen::tensor_element(qwen::TensorType::Q4_0, bytes, i);
    return result;
}

std::vector<float> transformed(std::span<const float> raw, std::span<const float> gamma,
                               const Fixture& fixture, std::size_t position) {
    float sum = 0.0f;
    for (float x : raw) { const float square = x * x; sum += square; }
    const float inverse_rms = 1.0f / std::sqrt(sum / static_cast<float>(raw.size()) + fixture.config.rms_epsilon);
    std::vector<float> normalized(raw.size()), result(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) normalized[i] = (raw[i] * inverse_rms) * gamma[i];
    const auto half = fixture.config.rotary_dim / 2;
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (i >= fixture.config.rotary_dim) { result[i] = normalized[i]; continue; }
        const auto f = i % half;
        const float angle = static_cast<float>(position) * fixture.frequencies[f];
        const float cosine = std::cos(angle) * fixture.config.rope_attention_scale;
        const float sine = std::sin(angle) * fixture.config.rope_attention_scale;
        const float rotated = i < half ? -normalized[i + half] : normalized[i - half];
        result[i] = normalized[i] * cosine + rotated * sine;
    }
    return result;
}

struct Oracle {
    std::vector<float> pooled, tail, scores;
};
Oracle oracle(std::span<const float> raw, std::span<const float> q, const Fixture& fixture) {
    const auto visible = raw.size() / 128, blocks = visible / 4;
    const auto rounded = roundtrip(raw);
    Oracle out;
    out.pooled.resize(blocks * 128);
    out.tail.assign(rounded.begin() + blocks * 4 * 128, rounded.end());
    for (std::size_t b = 0; b < blocks; ++b) {
        std::array<float, 128> mean{};
        for (std::size_t d = 0; d < 128; ++d) {
            float accumulator = rounded[(4 * b) * 128 + d];
            for (std::size_t member = 1; member < 4; ++member)
                accumulator += rounded[(4 * b + member) * 128 + d];
            mean[d] = accumulator / 4.0f;
        }
        const auto result = transformed(mean, fixture.kweight, fixture, 4 * b);
        std::copy(result.begin(), result.end(), out.pooled.begin() + b * 128);
    }
    if (visible == 0) return out;
    std::vector<float> normalized_query;
    for (std::size_t h = 0; h < 4; ++h) {
        const auto row = transformed(q.subspan(h * 128, 128), fixture.qweight, fixture, visible - 1);
        normalized_query.insert(normalized_query.end(), row.begin(), row.end());
    }
    out.scores.resize(blocks);
    if (blocks != 0) {
        // Independent DENSE FP32 scorer: weight row b is the completed key;
        // each query head is a separate input column. ReLU is AFTER each dot.
        std::vector<float> dots(4 * blocks);
        const qwen::QMatrix matrix{qwen::TensorType::F32, 128, static_cast<int>(blocks),
                                   std::as_bytes(std::span(out.pooled))};
        qwen::matmul_f32(matrix, normalized_query, 4, dots);
        for (std::size_t b = 0; b < blocks; ++b) {
            float sum = 0.0f;
            for (std::size_t h = 0; h < 4; ++h) sum += std::max(dots[h * blocks + b], 0.0f);
            out.scores[b] = sum / std::sqrt(128.0f);
        }
    }
    return out;
}

void check_selection(std::span<const float> scores, std::size_t visible) {
    constexpr std::int32_t canary = -31337;
    std::array<std::int32_t, 2053> tokens;
    std::array<std::int32_t, 514> blocks;
    tokens.fill(canary); blocks.fill(canary);
    const auto count = qwen::qsa_select(scores, visible, std::span(tokens).subspan(1, 2051),
                                      std::span(blocks).subspan(1, 512));
    ++selections;
    check(count.block_count == std::min<std::size_t>(512, visible / 4), "selection block count");
    check(count.token_count == count.block_count * 4 + visible % 4, "selection actual tail count");
    check(tokens[0] == canary && blocks[0] == canary, "selection leading canaries");
    for (std::size_t i = count.token_count + 1; i < tokens.size(); ++i)
        check(tokens[i] == canary, "selection padding was written");
    for (std::size_t i = count.block_count + 1; i < blocks.size(); ++i)
        check(blocks[i] == canary, "selection block padding was written");
    // Independent rank count: every block that beats b defines one rank.
    for (std::size_t b = 0; b < scores.size(); ++b) {
        std::size_t rank = 0;
        for (std::size_t other = 0; other < scores.size(); ++other)
            if (scores[other] > scores[b] || (scores[other] == scores[b] && other < b)) ++rank;
        if (rank >= count.block_count) continue;
        check(blocks[rank + 1] == static_cast<std::int32_t>(b), "selection rank/tie mismatch");
        for (std::size_t d = 0; d < 4; ++d)
            check(tokens[rank * 4 + d + 1] == static_cast<std::int32_t>(4 * b + d), "selection split block");
    }
    for (std::size_t i = 0; i < visible % 4; ++i)
        check(tokens[count.block_count * 4 + i + 1] == static_cast<std::int32_t>(visible / 4 * 4 + i), "selection tail ID");
    if (visible == 2052) check(count.token_count == 2048, "2052 HF count (mx expanded count is 2051)");
}

void assert_oracle(qwen::QsaIndexCpu& cpu, std::span<const float> raw,
                   std::span<const float> q, const Fixture& fixture) {
    const auto ref = oracle(raw, q, fixture);
    check(cpu.consumed_tokens() == raw.size() / 128, "oracle consumed count");
    near(cpu.completed_keys(), ref.pooled, max_key_error, "pooled norm/RoPE oracle mismatch");
    exact(cpu.raw_tail(), ref.tail, "Q4 raw tail oracle mismatch");
    if (cpu.consumed_tokens() != 0) {
        std::vector<float> scores(cpu.completed_blocks(), -991.0f);
        cpu.score(q, scores);
        near(scores, ref.scores, max_score_error, "independent dense score oracle mismatch");
        check_selection(scores, cpu.consumed_tokens());
    }
}

void q4_fixtures() {
    std::array<float, 128> raw{};
    // Signed first-max ties, clamping, zero blocks, a representable subnormal
    // half scale, and values whose code would differ with a rounded reciprocal.
    for (std::size_t d = 0; d < 32; ++d) raw[d] = static_cast<float>(static_cast<int>(d) - 16) / 8.0f;
    raw[0] = 8.003f; raw[1] = -8.003f; raw[2] = 0.5002f;
    for (std::size_t d = 64; d < 96; ++d) raw[d] = std::ldexp(static_cast<float>(static_cast<int>(d) - 80), -20);
    raw[96] = -17.3f; raw[97] = 17.3f; raw[98] = -0.0f;
    const auto readonly = raw;
    std::array<qwen::Q4_0, 4> packed{};
    std::array<float, 128> actual{};
    qwen::quantize_q4(raw, packed); qwen::dequantize_q4(packed, actual);
    const auto expected = roundtrip(raw);
    near(actual, expected, max_roundtrip_error, "independent Q4 roundtrip mismatch");
    exact(raw, readonly, "read-only Q4 fixture changed");
    check((packed[0].d & 0x8000U) != 0, "first signed tie scale");
    check((packed[3].d & 0x8000U) == 0, "negative first tie scale");
    // Frozen known GGML bytes: d=-1, nibble j and 15-j. Dense reader checks
    // canonical low-first16/high-last16 layout independently from kv dequant.
    const qwen::Q4_0 known{0xbc00U, {0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,
                                    0x78,0x69,0x5a,0x4b,0x3c,0x2d,0x1e,0x0f}};
    std::array<float, 32> decoded{};
    qwen::dequantize_q4(std::span(&known, 1), decoded);
    for (std::size_t i = 0; i < 16; ++i) {
        check(decoded[i] == static_cast<float>(8 - static_cast<int>(i)), "known low nibble");
        check(decoded[i + 16] == static_cast<float>(static_cast<int>(i) - 7), "known high nibble");
    }
    Fixture f(32);
    qwen::QsaIndexCpu cpu(f.config, f.parameters());
    cpu.append(raw, 1);
    exact(cpu.raw_tail(), expected, "raw index K did not use operational Q4");
}

void numeric_and_chunks() {
    for (const auto rotary : {std::size_t(2), std::size_t(64), std::size_t(128)}) {
        Fixture fixture(64, rotary);
        fixture.config.rope_attention_scale = 1.125f;
        const auto raw = keys(24), readonly = raw;
        const auto q = query(9);
        qwen::QsaIndexCpu cpu(fixture.config, fixture.parameters());
        for (std::size_t n = 1; n <= 24; ++n) {
            cpu.append(std::span(raw).subspan((n - 1) * 128, 128), 1);
            assert_oracle(cpu, std::span(raw).first(n * 128), q, fixture);
        }
        exact(raw, readonly, "read-only projected keys changed");
        // Constructor owns params: later caller changes cannot alter the cache.
        const Fixture original_fixture = fixture;
        fixture.qweight.fill(0.0f); fixture.kweight.fill(0.0f);
        fixture.frequencies.assign(fixture.frequencies.size(), 7.0f);
        const auto next = keys(4, 83);
        cpu.append(next, 4);
        auto continued = raw;
        continued.insert(continued.end(), next.begin(), next.end());
        assert_oracle(cpu, continued, q, original_fixture);
        qwen::QsaIndexCheckpoint cp;
        cpu.save(cp);
        check(cp.raw_tail().empty(), "aligned snapshot tail");
    }
    Fixture fixture;
    const auto raw = keys(2056);
    const auto qraw = query(3);
    qwen::QsaIndexCpu sequential(fixture.config, fixture.parameters());
    for (std::size_t n = 1; n <= 2056; ++n) {
        sequential.append(std::span(raw).subspan((n - 1) * 128, 128), 1);
        if (n < 9 || (n >= 2047 && n <= 2056))
            assert_oracle(sequential, std::span(raw).first(n * 128), qraw, fixture);
    }
    for (const auto chunk : {std::size_t(1), std::size_t(2), std::size_t(3), std::size_t(7),
                             std::size_t(128), std::size_t(1024), std::size_t(2056)}) {
        qwen::QsaIndexCpu cpu(fixture.config, fixture.parameters());
        for (std::size_t n = 0; n < 2056;) {
            const auto count = std::min(chunk, 2056 - n);
            cpu.append(std::span(raw).subspan(n * 128, count * 128), count);
            n += count;
        }
        exact(cpu.completed_keys(), sequential.completed_keys(), "chunk pooled keys not bit-exact");
        exact(cpu.raw_tail(), sequential.raw_tail(), "chunk raw tail not bit-exact");
        std::vector<float> a(cpu.completed_blocks()), b(sequential.completed_blocks());
        cpu.score(qraw, a); sequential.score(qraw, b);
        exact(a, b, "chunk scores not bit-exact");
        // Explicit sequential/chunk parity for each ending remainder across the
        // budget transition, rather than only the aligned 2056-token final state.
        for (std::size_t visible = 2047; visible < 2056; ++visible) {
            cpu.reset();
            for (std::size_t n = 0; n < visible;) {
                const auto count = std::min(chunk, visible - n);
                cpu.append(std::span(raw).subspan(n * 128, count * 128), count);
                n += count;
            }
            // Every completed block is immutable under appending the suffix.
            exact(cpu.completed_keys(), sequential.completed_keys().first(cpu.completed_keys().size()),
                  "boundary chunk keys not bit-exact");
            const auto tail = roundtrip(std::span(raw).subspan((visible / 4) * 4 * 128, (visible % 4) * 128));
            exact(cpu.raw_tail(), tail, "boundary chunk tail not bit-exact");
            assert_oracle(cpu, std::span(raw).first(visible * 128), qraw, fixture);
        }
    }
    // Zero Q produces exact tied zero scores regardless of all K values.
    for (std::size_t visible = 2047; visible <= 2056; ++visible) {
        qwen::QsaIndexCpu cpu(fixture.config, fixture.parameters());
        const std::vector<float> zeros(visible * 128, 0.0f);
        const std::array<float, 512> qzero{};
        cpu.append(zeros, visible);
        std::vector<float> scores(cpu.completed_blocks()); cpu.score(qzero, scores);
        check(std::all_of(scores.begin(), scores.end(), [](float s) { return s == 0.0f; }), "zero scores");
        check_selection(scores, visible);
    }
    // Equal NONZERO scores with the signal solely outside the rotary prefix.
    // Opposite query heads make ReLU-before-head-sum observably different from
    // ReLU-after-sum. No special frequency/score constant is used to force ties.
    Fixture tied(2056, 2);
    tied.qweight.fill(1.0f); tied.kweight.fill(1.0f);
    std::vector<float> identical(2056 * 128);
    for (std::size_t t = 0; t < 2056; ++t) identical[t * 128 + 127] = 1.0f;
    std::array<float, 512> opposing{};
    for (std::size_t h = 0; h < 4; ++h) opposing[h * 128 + 127] = h % 2 == 0 ? 1.0f : -1.0f;
    qwen::QsaIndexCpu equal(tied.config, tied.parameters());
    equal.append(std::span(identical).first(2046 * 128), 2046);
    for (std::size_t visible = 2047; visible <= 2056; ++visible) {
        equal.append(std::span(identical).subspan((visible - 1) * 128, 128), 1);
        std::vector<float> scores(equal.completed_blocks()); equal.score(opposing, scores);
        check(scores[0] > 0.0f, "headwise ReLU must not cancel opposite heads");
        check(std::all_of(scores.begin(), scores.end(), [&](float s) { return s == scores[0]; }), "nonzero exact ties");
        assert_oracle(equal, std::span(identical).first(visible * 128), opposing, tied);
    }
}

void rollback() {
    Fixture fixture(128);
    const auto qraw = query(5);
    for (std::size_t remainder = 0; remainder < 4; ++remainder) for (std::size_t accept = 0; accept <= 2; ++accept) {
        qwen::QsaIndexCpu cpu(fixture.config, fixture.parameters());
        std::vector<float> committed = keys(8 + remainder);
        cpu.append(committed, committed.size() / 128);
        std::array<qwen::QsaIndexCheckpoint, 4> slots;
        for (std::size_t window = 0; window < 7; ++window) {
            const auto verify = keys(3, window + 17);
            cpu.append(verify, 3, slots);
            const auto retained = 1 + accept;
            cpu.restore(slots[retained]); ++restores;
            committed.insert(committed.end(), verify.begin(), verify.begin() + retained * 128);
            assert_oracle(cpu, committed, qraw, fixture);
            const auto replacement = keys(1, window + 79);
            cpu.append(replacement, 1);
            committed.insert(committed.end(), replacement.begin(), replacement.end());
            assert_oracle(cpu, committed, qraw, fixture);
        }
    }
    qwen::QsaIndexCpu cpu(fixture.config, fixture.parameters()), other(fixture.config, fixture.parameters());
    const auto raw = keys(16);
    cpu.append(std::span(raw).first(3 * 128), 3);
    qwen::QsaIndexCheckpoint early, later;
    cpu.save(early);
    cpu.append(std::span(raw).subspan(3 * 128, 9 * 128), 9); cpu.save(later);
    cpu.restore(early); ++restores;
    cpu.restore(later); ++restores; // retained completed prefix is still intact
    cpu.restore(early); ++restores;
    const auto branch = keys(1, 234);
    cpu.append(branch, 1); // overwrites completed block 0, invalidating old suffix
    rejected([&] { cpu.restore(later); });
    check(cpu.consumed_tokens() == 4, "stale restore changed state");
    rejected([&] { other.restore(early); });
    rejected([&] { other.restore(qwen::QsaIndexCheckpoint{}); });
    cpu.reset(); rejected([&] { cpu.restore(early); });
    // Empty-prefix and partial-tail checkpoints remain sufficient after a
    // branch; they own their small raw tail and require no completed history.
    cpu.save(early); cpu.append(std::span(raw).first(128), 1); cpu.restore(early);
    check(cpu.consumed_tokens() == 0 && cpu.raw_tail().empty(), "empty-prefix restore");

    // An address is not owner identity: destruction/reconstruction at the same
    // address must reject a snapshot, even with the same geometry and prefix.
    alignas(qwen::QsaIndexCpu) std::array<std::byte, sizeof(qwen::QsaIndexCpu)> storage{};
    auto* first = new (storage.data()) qwen::QsaIndexCpu(fixture.config, fixture.parameters());
    qwen::QsaIndexCheckpoint previous_lifetime;
    first->save(previous_lifetime);
    first->~QsaIndexCpu();
    auto* second = new (storage.data()) qwen::QsaIndexCpu(fixture.config, fixture.parameters());
    rejected([&] { second->restore(previous_lifetime); });
    second->~QsaIndexCpu();
}

struct StateCopy {
    std::size_t consumed;
    std::vector<float> pooled, tail;
    explicit StateCopy(const qwen::QsaIndexCpu& cpu) : consumed(cpu.consumed_tokens()),
        pooled(cpu.completed_keys().begin(), cpu.completed_keys().end()), tail(cpu.raw_tail().begin(), cpu.raw_tail().end()) {}
    void verify(const qwen::QsaIndexCpu& cpu) const {
        check(consumed == cpu.consumed_tokens(), "rejection changed consumed count");
        exact(pooled, cpu.completed_keys(), "rejection changed pooled cache");
        exact(tail, cpu.raw_tail(), "rejection changed raw tail");
    }
};

void invalid_and_atomic() {
    Fixture fixture(16);
    for (int field = 0; field < 10; ++field) {
        auto c = fixture.config;
        switch (field) {
            case 0: c.head_dim = 64; break; case 1: c.query_heads = 3; break;
            case 2: c.key_heads = 2; break; case 3: c.compress_ratio = 8; break;
            case 4: c.capacity = 0; break; case 5: c.capacity = std::numeric_limits<std::size_t>::max(); break;
            case 6: c.rotary_dim = 0; break; case 7: c.rotary_dim = 129; break;
            case 8: c.rms_epsilon = 0.0f; break; case 9: c.rope_attention_scale = -1.0f; break;
        }
        rejected([&] { qwen::QsaIndexCpu bad(c, fixture.parameters()); });
    }
    auto p = fixture.parameters(); p.key_norm_weight = p.key_norm_weight.first(127);
    rejected([&] { qwen::QsaIndexCpu bad(fixture.config, p); });
    p = fixture.parameters(); p.inverse_frequencies = p.inverse_frequencies.first(1);
    rejected([&] { qwen::QsaIndexCpu bad(fixture.config, p); });
    fixture.frequencies[0] = 0.0f;
    rejected([&] { qwen::QsaIndexCpu bad(fixture.config, fixture.parameters()); });
    fixture.frequencies[0] = std::numeric_limits<float>::infinity();
    rejected([&] { qwen::QsaIndexCpu bad(fixture.config, fixture.parameters()); });
    fixture.frequencies[0] = 1.0f;
    fixture.kweight[0] = std::numeric_limits<float>::quiet_NaN();
    rejected([&] { qwen::QsaIndexCpu bad(fixture.config, fixture.parameters()); });
    fixture.kweight[0] = 1.0f;
    qwen::QsaIndexCpu cpu(fixture.config, fixture.parameters());
    const auto raw = keys(8);
    const auto qraw = query();
    std::array<float, 2> output{-77.0f, -77.0f};
    rejected([&] { cpu.score(qraw, {}); });
    cpu.append(std::span(raw).first(5 * 128), 5);
    const StateCopy original(cpu);
    std::array<qwen::QsaIndexCheckpoint, 4> slots;
    auto bad = keys(3);
    for (const auto value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::max(), std::numeric_limits<float>::denorm_min()}) {
        bad = keys(3);
        if (value == std::numeric_limits<float>::denorm_min())
            std::fill(bad.end() - 32, bad.end(), value);
        else bad.back() = value;
        rejected([&] { cpu.append(bad, 3, slots); });
        original.verify(cpu);
        check(slots[0].consumed_tokens() == 0, "invalid chunk wrote prefixes");
    }
    rejected([&] { cpu.append(raw, std::numeric_limits<std::size_t>::max(), slots); });
    rejected([&] { cpu.append(raw, 7); });
    rejected([&] { cpu.append(raw, 8, slots); });
    rejected([&] { cpu.append(cpu.raw_tail(), 1); });
    rejected([&] { cpu.append(cpu.completed_keys(), 1); });
    qwen::QsaIndexCheckpoint cp; cpu.save(cp);
    rejected([&] { cpu.append(cp.raw_tail(), 1, std::span(&cp, 1)); });
    // Correct prefix count, so rejection must inspect the checkpoint-tail alias.
    std::array<qwen::QsaIndexCheckpoint, 2> pair; cpu.save(pair[1]);
    rejected([&] { cpu.append(pair[1].raw_tail(), 1, pair); });
    auto moved = std::move(pair[1]);
    rejected([&] { cpu.save(pair[1]); });
    rejected([&] { cpu.restore(pair[1]); });
    cpu.restore(moved);
    auto badq = qraw; badq.back() = std::numeric_limits<float>::quiet_NaN();
    rejected([&] { cpu.score(badq, std::span(output).first(1)); });
    rejected([&] { cpu.score(qraw, output); });
    rejected([&] { cpu.score(std::span(qraw).first(511), std::span(output).first(1)); });
    auto overlapq = qraw;
    rejected([&] { cpu.score(overlapq, std::span(overlapq).first(1)); });
    auto* owned = const_cast<float*>(cpu.completed_keys().data());
    rejected([&] { cpu.score(qraw, std::span(owned, 1)); });
    original.verify(cpu);
    cpu.append(std::span(raw).first(3 * 128), 3);
    // Two blocks = 256 values, too short for Q. A four-block cache allows a
    // correctly sized owned input alias and tests it before any normalization.
    cpu.append(raw, 8);
    std::array<float, 4> owned_alias_output{};
    rejected([&] { cpu.score(cpu.completed_keys(), owned_alias_output); });
    check(output[0] == -77.0f && output[1] == -77.0f, "invalid score changed output");
    rejected([&] { cpu.append(std::span(raw).first(128), 1); });
    const StateCopy full(cpu);
    full.verify(cpu);

    // Nonfinite FP32 RMS reduction: finite query, unchanged output/state.
    std::vector<float> scores(cpu.completed_blocks(), -123.0f);
    badq.fill(std::numeric_limits<float>::max());
    numerical_rejected([&] { cpu.score(badq, scores); });
    check(std::all_of(scores.begin(), scores.end(), [](float x) { return x == -123.0f; }), "failed query wrote output");
    full.verify(cpu); cpu.score(qraw, scores);

    // Finite key gamma overflows only on the fourth token. Whole-chunk invalid
    // checks pass; prefixes 0..3 commit; token4 and its prefix remain untouched.
    Fixture explosive(16);
    explosive.kweight.fill(std::numeric_limits<float>::max());
    qwen::QsaIndexCpu failure(explosive.config, explosive.parameters());
    const auto repeated = keys(4);
    std::array<qwen::QsaIndexCheckpoint, 5> five;
    numerical_rejected([&] { failure.append(repeated, 4, five); });
    check(failure.consumed_tokens() == 3 && failure.completed_blocks() == 0, "failed completed key consumed input");
    check(five[3].consumed_tokens() == 3 && five[4].consumed_tokens() == 0, "numerical failure wrote failed prefix");
    exact(failure.raw_tail(), roundtrip(std::span(repeated).first(384)), "failed token corrupted raw tail");
    const StateCopy before_failure(failure);
    numerical_rejected([&] { failure.append(std::span(repeated).last(128), 1); });
    before_failure.verify(failure);
    failure.restore(five[0]);
    const std::vector<float> zeros(4 * 128);
    failure.append(zeros, 4); // valid reuse even with very large finite gamma
    check(failure.consumed_tokens() == 4, "reuse after numerical failure");

    // Score overflow AFTER valid query/key normalization, not merely input RMS.
    Fixture large(8); large.qweight.fill(1.0e20f); large.kweight.fill(1.0e20f);
    qwen::QsaIndexCpu dot_failure(large.config, large.parameters());
    const std::vector<float> ones(4 * 128, 1.0f);
    const std::array<float, 512> qones = [] { std::array<float, 512> a{}; a.fill(1.0f); return a; }();
    dot_failure.append(ones, 4);
    std::array<float, 1> guard{-123.0f};
    numerical_rejected([&] { dot_failure.score(qones, guard); });
    check(guard[0] == -123.0f, "dot overflow wrote output");
    const std::array<float, 512> qzeros{};
    dot_failure.score(qzeros, guard); check(guard[0] == 0.0f, "reuse after score overflow");
}

void capacity_and_allocations() {
    Fixture fixture;
    qwen::QsaIndexCpu cpu(fixture.config, fixture.parameters());
    std::array<qwen::QsaIndexCheckpoint, 4> slots;
    std::array<float, 512> qzero{};
    const std::vector<float> zero(131072 * 128);
    std::vector<float> scores(32768);
    count_allocations = true;
    cpu.append(std::span(zero).first(3 * 128), 3, slots);
    cpu.restore(slots[1]);
    cpu.append(std::span(zero).subspan(128, (131072 - 1) * 128), 131072 - 1);
    cpu.score(qzero, scores);
    cpu.save(slots[0]);
    cpu.restore(slots[0]);
    cpu.append({}, 0);
    count_allocations = false;
    check(allocations == 0, "allocation in successful numeric hot call");
    check(cpu.consumed_tokens() == 131072 && cpu.completed_keys().size() == 32768 * 128 &&
          cpu.raw_tail().empty(), "actual capacity131072 append");
    check(std::all_of(scores.begin(), scores.end(), [](float s) { return s == 0.0f; }), "capacity scores");
    rejected([&] { cpu.append(std::span(zero).first(128), 1); });
    count_allocations = true; cpu.reset(); count_allocations = false;
    check(allocations == 0 && cpu.consumed_tokens() == 0, "reset allocation/length");
    // Non-multiple-of-four capacity and capacities smaller than one block.
    for (std::size_t capacity = 1; capacity <= 7; ++capacity) {
        Fixture small(capacity);
        qwen::QsaIndexCpu tail(small.config, small.parameters());
        const auto raw = keys(capacity); const auto qraw = query(31);
        tail.append(raw, capacity); assert_oracle(tail, raw, qraw, small);
        rejected([&] { tail.append(std::span(raw).first(128), 1); });
    }
}

} // namespace

int main() {
    try {
        q4_fixtures();
        numeric_and_chunks();
        rollback();
        invalid_and_atomic();
        capacity_and_allocations();
        std::cout << "{\"kind\":\"qsa_index_cpu\",\"checks\":" << checks
                  << ",\"rejects\":" << rejects << ",\"numeric_rejects\":" << numeric_rejects
                  << ",\"selections\":" << selections << ",\"restores\":" << restores
                  << ",\"hot_allocations\":" << allocations << ",\"max_key_error\":" << max_key_error
                  << ",\"max_score_error\":" << max_score_error << ",\"max_roundtrip_error\":" << max_roundtrip_error
                  << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        count_allocations = false;
        std::cerr << "qsa index test: " << error.what() << '\n';
        std::cout << "{\"kind\":\"qsa_index_cpu\",\"checks\":" << checks << ",\"rejects\":" << rejects
                  << ",\"numeric_rejects\":" << numeric_rejects << ",\"selections\":" << selections
                  << ",\"restores\":" << restores << ",\"hot_allocations\":" << allocations
                  << ",\"max_key_error\":" << max_key_error << ",\"max_score_error\":" << max_score_error
                  << ",\"max_roundtrip_error\":" << max_roundtrip_error << ",\"passed\":false}\n";
        return 1;
    }
}
