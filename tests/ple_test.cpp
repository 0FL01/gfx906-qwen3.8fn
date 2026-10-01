#include "ple.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>

#ifndef CORE_REVISION
#define CORE_REVISION "unknown"
#define CORE_DIRTY 1
#endif

namespace allocation_probe {
bool enabled = false;
std::size_t count = 0;
}
// Count successful hot calls, including file-backed read_slice, save/restore.
// Keep both sides of the malloc/free-backed replacement out of allocator
// inlining: GCC 13 otherwise mistakes this legal paired override for new/free.
[[gnu::noinline]] void* operator new(std::size_t n) {
    if (allocation_probe::enabled) ++allocation_probe::count;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
[[gnu::noinline]] void* operator new[](std::size_t n) { return ::operator new(n); }
[[gnu::noinline]] void operator delete(void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete(void* p, std::size_t) noexcept { std::free(p); }
[[gnu::noinline]] void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace qwen;
using Bytes = std::vector<std::byte>;
std::size_t checks = 0, rejected = 0;
double max_error = 0;
void check(bool ok, const char* what) { ++checks; if (!ok) throw std::runtime_error(what); }
template<class F> void reject(F f) {
    try { f(); } catch (const std::exception&) { ++rejected; return; }
    throw std::runtime_error("invalid PLE call accepted after " + std::to_string(rejected) + " rejections");
}
template<class T> bool same(std::span<const T> a, std::span<const T> b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0);
}
bool same_hash(const PleHashCheckpoint& a, const PleHashCheckpoint& b) {
    return a.config() == b.config() && same(a.history(), b.history()) &&
        a.history_count() == b.history_count() && a.segment_tokens() == b.segment_tokens() &&
        a.consumed_tokens() == b.consumed_tokens();
}
bool same_layer(const PleLayerCheckpoint& a, const PleLayerCheckpoint& b) {
    return a.config() == b.config() && same(a.conv_history(), b.conv_history()) &&
        a.consumed_tokens() == b.consumed_tokens();
}
void near(float actual, double reference, double tolerance = 3e-6) {
    const double error = std::abs(double(actual) - reference);
    max_error = std::max(max_error, error);
    check(std::isfinite(actual) && error <= tolerance * (1 + std::abs(reference)), "double oracle mismatch");
}

PleHashConfig tiny_hash() {
    PleHashConfig c;
    c.ngram_size = 3; c.heads_per_ngram = 2; c.head_dim = 2;
    c.table_rows = 11; c.token_vocab_size = 300000;
    // Deliberately exercise 64-bit multiplication overflow and signed mixed IDs.
    c.multipliers = {0xf123456789abcdefULL, 0x743219876543210fULL, 0xa5a5f0f012345679ULL};
    c.vocab_sizes = {3, 4, 3, 1}; c.offsets = {0, 3, 7, 10};
    c.validate(); return c;
}
PleHashConfig keep1_hash(bool physical) {
    PleHashConfig c;
    c.token_vocab_size = 300000;
    c.multipliers = {23456789012345ULL, 3456789012345ULL, 9876543211235ULL};
    c.table_rows = physical ? 40000085 : 11;
    c.vocab_sizes.assign(16, 1); c.offsets.assign(16, physical ? 40000084 : 10);
    c.vocab_sizes[0] = physical ? 20000003 : 3;
    c.vocab_sizes[8] = physical ? 20000081 : 7;
    c.offsets[0] = 0; c.offsets[8] = c.vocab_sizes[0];
    c.validate(); return c;
}
// Independent low-64 multiplication via 32-bit limbs, rather than using the
// production uint64 product. Signed bit_cast + signed % gives torch remainder.
std::uint64_t product(std::uint64_t a, std::uint64_t b) {
    constexpr std::uint64_t mask = 0xffffffff;
    const std::uint64_t low = (a & mask) * (b & mask);
    const std::uint64_t high = ((a >> 32) * (b & mask) + (b >> 32) * (a & mask)) & mask;
    return low + (high << 32);
}
std::vector<std::uint64_t> formula(const PleHashConfig& c, const std::vector<std::int64_t>& sequence) {
    std::vector<std::uint64_t> result(sequence.size() * c.heads());
    std::size_t start = 0;
    for (std::size_t t = 0; t < sequence.size(); ++t) {
        for (std::size_t order = 2; order <= c.ngram_size; ++order) {
            std::uint64_t mixed = 0;
            for (std::size_t p = 0; p < order; ++p) {
                const auto id = p <= t - start ? sequence[t - p] : c.eos_token_id;
                mixed ^= product(static_cast<std::uint64_t>(id), c.multipliers[p]);
            }
            const auto signed_id = std::bit_cast<std::int64_t>(mixed);
            for (std::size_t g = 0; g < c.heads_per_ngram; ++g) {
                const auto h = (order - 2) * c.heads_per_ngram + g;
                const auto divisor = static_cast<std::int64_t>(c.vocab_sizes[h]);
                auto r = signed_id % divisor;
                if (r < 0) r += divisor;
                result[t * c.heads() + h] = std::uint64_t(r) + c.offsets[h];
            }
        }
        if (sequence[t] == c.eos_token_id) start = t + 1;
    }
    return result;
}

struct HashProof {
    std::uint64_t maximum_valid_token_id_including_eos;
    std::uint64_t hf_generator_multiplier_max, valid_id_multiplier_bound;
    bool nonnegative_hash_proven_for_all_valid_ids = true;
    bool generator_bound_satisfied = true;
    std::vector<bool> products_fit_int64;
};
HashProof hash_proof(const PleHashConfig& c) {
    c.validate();
    // HF a005fc82 modular_qwen4_exp.py:576-588 bounds each positive multiplier
    // by INT64_MAX / unigram_vocab. The actual checkpoint's constants, NOT an
    // assumed generator/config, determine this sufficient proof. Every valid
    // ID (including the EOS padding ID) is nonnegative. If every product fits
    // INT64_MAX its high bit is zero; XOR of ANY such products also has high
    // bit zero, for every context/order. Signed and unsigned remainders then
    // coincide throughout the vocabulary, not merely on sampled sequences.
    const auto limit = std::uint64_t(INT64_MAX);
    const auto maximum = std::max(c.token_vocab_size - 1, std::uint64_t(c.eos_token_id));
    HashProof p{maximum, limit / c.token_vocab_size, maximum ? limit / maximum : UINT64_MAX,
                true, true, {}};
    for (auto multiplier : c.multipliers) {
        const bool fits = !maximum || multiplier <= limit / maximum; // no overflowing product in proof
        p.products_fit_int64.push_back(fits);
        p.nonnegative_hash_proven_for_all_valid_ids &= fits;
        p.generator_bound_satisfied &= multiplier <= p.hf_generator_multiplier_max;
    }
    return p;
}
struct HashDiagnostic {
    std::size_t negative_hash_count = 0, mismatched_row_count = 0;
    std::vector<std::uint64_t> signed_rows, unsigned_rows;
};
HashDiagnostic hash_diagnostic(const PleHashConfig& c, const std::vector<std::int64_t>& sequence) {
    c.validate();
    HashDiagnostic d;
    d.signed_rows = formula(c, sequence); d.unsigned_rows.resize(d.signed_rows.size());
    std::size_t segment_start = 0;
    for (std::size_t t = 0; t < sequence.size(); ++t) {
        check(sequence[t] >= 0 && std::uint64_t(sequence[t]) < c.token_vocab_size, "diagnostic valid token");
        std::uint64_t mixed = product(std::uint64_t(sequence[t]), c.multipliers[0]);
        for (std::size_t lag = 1; lag < c.ngram_size; ++lag) {
            const auto previous = lag <= t - segment_start ? sequence[t - lag] : c.eos_token_id;
            mixed ^= product(std::uint64_t(previous), c.multipliers[lag]);
            d.negative_hash_count += std::size_t(mixed >> 63); // one count per token/order, not per head
            for (std::size_t g = 0; g < c.heads_per_ngram; ++g) {
                const auto h = (lag - 1) * c.heads_per_ngram + g;
                const auto i = t * c.heads() + h;
                d.unsigned_rows[i] = mixed % c.vocab_sizes[h] + c.offsets[h];
                d.mismatched_row_count += d.unsigned_rows[i] != d.signed_rows[i];
            }
        }
        if (sequence[t] == c.eos_token_id) segment_start = t + 1;
    }
    if (hash_proof(c).nonnegative_hash_proven_for_all_valid_ids)
        check(d.negative_hash_count == 0 && d.mismatched_row_count == 0, "vocabulary-wide proof implies sampled equivalence");
    return d;
}
void hash_diagnostic_tests() {
    auto c = keep1_hash(true);
    const auto proof = hash_proof(c);
    check(proof.nonnegative_hash_proven_for_all_valid_ids && proof.generator_bound_satisfied,
          "positive bounded products prove every XOR has high bit zero");
    check(proof.maximum_valid_token_id_including_eos == c.token_vocab_size - 1, "proof includes valid EOS");
    auto d = hash_diagnostic(c, {299999, c.eos_token_id, 248046, 0, 1, 299999});
    check(d.negative_hash_count == 0 && d.mismatched_row_count == 0, "bounded actual-like constants diagnostics");
    c = tiny_hash(); c.eos_token_id = 0; c.multipliers = {0x8000000000000000ULL, 0, 0};
    check(!hash_proof(c).nonnegative_hash_proven_for_all_valid_ids, "synthetic negative products are not proven nonnegative");
    d = hash_diagnostic(c, {1});
    check(d.negative_hash_count == 2 && d.mismatched_row_count > 0, "synthetic signed/unsigned diagnostic remains active");
    // Inclusive maximum-valid-ID bound is slightly less restrictive than HF's
    // exclusive vocabulary-size bound. It is sufficient on its own.
    c.token_vocab_size = 2; c.multipliers = {std::uint64_t(INT64_MAX), 0, 0};
    const auto inclusive = hash_proof(c);
    check(inclusive.nonnegative_hash_proven_for_all_valid_ids && !inclusive.generator_bound_satisfied,
          "proof uses loaded constants and inclusive maximum valid ID");
    c.token_vocab_size = 1; c.multipliers = {UINT64_MAX, UINT64_MAX, UINT64_MAX};
    check(hash_proof(c).nonnegative_hash_proven_for_all_valid_ids, "zero-only vocabulary has zero products even for large multiplier bits");
    d = hash_diagnostic(c, {0, 0}); check(d.negative_hash_count == 0, "zero-only vocabulary diagnostic");
}

// Small independent GGUF writer, literal format IDs/byte geometry. No model
// copies, sparse multi-GB files or loader sizing helpers in these fixtures.
struct Writer {
    Bytes bytes;
    void number(std::uint64_t v, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) bytes.push_back(std::byte((v >> (8 * i)) & 255));
    }
    void string(std::string_view s) {
        number(s.size(), 8);
        for (auto c : s) bytes.push_back(std::byte(static_cast<unsigned char>(c)));
    }
};
Bytes f32bytes(std::span<const float> data) {
    Writer w; for (float f : data) w.number(std::bit_cast<std::uint32_t>(f), 4); return w.bytes;
}
struct Tensor { std::string name; std::vector<std::uint64_t> dims; std::uint32_t type; Bytes bytes; };
struct Fixture {
    std::string path;
    int fd = -1;
    Fixture(const PleHashConfig& c, std::vector<Tensor> tensors, bool float_hash_metadata = false) {
        Writer w, md;
        std::uint64_t count = 0;
        auto key = [&](std::string_view name, unsigned type) { md.string(name); md.number(type, 4); ++count; };
        auto scalar = [&](std::string_view name, std::uint64_t v) { key(name, 4); md.number(v, 4); };
        auto array = [&](std::string_view name, std::span<const std::uint64_t> data, unsigned type) {
            key(name, 9); md.number(type, 4); md.number(data.size(), 8);
            for (auto v : data) md.number(v, type == 10 ? 8 : 4);
        };
        key("general.architecture", 8); md.string("qwen4exp");
        scalar("qwen4exp.block_count", 48); scalar("qwen4exp.vocab_size", c.token_vocab_size);
        scalar("qwen4exp.ple.ngram_size", c.ngram_size);
        scalar("qwen4exp.ple.heads_per_ngram", c.heads_per_ngram);
        scalar("qwen4exp.embedding_length_per_layer_input", c.head_dim);
        scalar("qwen4exp.ple.eos_token_id", c.eos_token_id);
        scalar("tokenizer.ggml.eos_token_id", 248046);
        const std::array<std::uint64_t, 1> layers{c.layer_index};
        array("qwen4exp.ple.layers", layers, 4);
        array("qwen4exp.ple.layer_multipliers", c.multipliers, float_hash_metadata ? 6 : 10);
        array("qwen4exp.ple.head_vocab_sizes", c.vocab_sizes, 10);
        array("qwen4exp.ple.head_offsets", c.offsets, 10);
        scalar("qwen4exp.embedding_length", 3); scalar("qwen4exp.hyper_connection.count", 2);
        scalar("qwen4exp.ple.conv_kernel", 4);
        key("qwen4exp.attention.layer_norm_rms_epsilon", 6);
        md.number(std::bit_cast<std::uint32_t>(1e-4f), 4);
        w.number(0x46554747, 4); w.number(3, 4); w.number(tensors.size(), 8); w.number(count, 8);
        w.bytes.insert(w.bytes.end(), md.bytes.begin(), md.bytes.end());
        std::uint64_t offset = 0;
        for (const auto& t : tensors) {
            w.string(t.name); w.number(t.dims.size(), 4);
            for (auto dim : t.dims) w.number(dim, 8);
            w.number(t.type, 4); w.number(offset, 8);
            offset += (t.bytes.size() + 31) / 32 * 32;
        }
        while (w.bytes.size() % 32) w.bytes.push_back(std::byte{0});
        for (const auto& t : tensors) {
            w.bytes.insert(w.bytes.end(), t.bytes.begin(), t.bytes.end());
            while (w.bytes.size() % 32) w.bytes.push_back(std::byte{0});
        }
        const auto pattern = (std::filesystem::temp_directory_path() / "ple-fixture-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end()); name.push_back('\0');
        fd = mkstemp(name.data());
        if (fd < 0) throw std::runtime_error("mkstemp");
        path = name.data();
        std::size_t done = 0;
        while (done < w.bytes.size()) {
            const auto n = write(fd, w.bytes.data() + done, w.bytes.size() - done);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) throw std::runtime_error("fixture write");
            done += static_cast<std::size_t>(n);
        }
    }
    ~Fixture() { if (fd >= 0) close(fd); if (!path.empty()) unlink(path.c_str()); }
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
};

void hash_tests() {
    for (auto c : {tiny_hash(), keep1_hash(true)}) {
        std::vector<std::int64_t> tokens{0, 1, 17, c.eos_token_id, 123, 248046, 299999, c.eos_token_id,
                                        c.eos_token_id, 0, 5, 8, 19, 7, 2};
        const auto expected = formula(c, tokens);
        PleHashCpu cpu(c), split(c);
        std::vector<std::uint64_t> rows(expected.size()), divided(expected.size());
        std::vector<PleHashCheckpoint> prefix;
        for (std::size_t i = 0; i <= tokens.size(); ++i) prefix.emplace_back(c);
        cpu.run(tokens, rows, prefix);
        check(rows == expected, "hash independent formula");
        for (std::size_t t = 0; t <= tokens.size(); ++t) {
            check(prefix[t].consumed_tokens() == t, "chronological hash prefix");
            check(prefix[t].history_count() == std::min(t, c.ngram_size - 1), "raw history count");
        }
        check(prefix[4].segment_tokens() == 0 && prefix[3].segment_tokens() == 2,
              "EOS boundary saved after, not before current hash");
        std::vector<std::uint64_t> ids(c.heads());
        ple_head_ids(c, c.eos_token_id, std::span<const std::int64_t>(tokens).subspan(1, 2), ids);
        check(std::equal(ids.begin(), ids.end(), expected.begin() + 3 * c.heads()), "current EOS retains context");
        for (std::size_t cut = 0; cut <= tokens.size(); ++cut) {
            split.reset();
            split.run(std::span(tokens).first(cut), std::span(divided).first(cut * c.heads()));
            split.run(std::span(tokens).subspan(cut), std::span(divided).subspan(cut * c.heads()));
            check(divided == rows && same_hash(cpu.state(), split.state()), "every hash chunk boundary exact");
        }
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const auto h = i % c.heads();
            check(rows[i] >= c.offsets[h] && rows[i] - c.offsets[h] < c.vocab_sizes[h], "row limit");
            if (c.heads() == 16 && h != 0 && h != 8) check(rows[i] == 40000084, "14 logical zero rows");
        }
        PleHashCheckpoint saved(c); cpu.save(saved);
        const auto unchanged = rows;
        reject([&] { cpu.run(std::array<std::int64_t, 2>{1, -1}, std::span(rows).first(2 * c.heads())); });
        check(rows == unchanged && same_hash(saved, cpu.state()), "late invalid ID atomic");
        reject([&] { cpu.step(300000, ids); });
        reject([&] { cpu.run(tokens, rows, std::span(prefix).first(2)); });
        reject([&] { ple_head_ids(cpu.state().config(), 1, cpu.state().history(),
                                std::span<std::uint64_t>(const_cast<std::uint64_t*>(cpu.state().config().offsets.data()), c.heads())); });
        reject([&] { cpu.run(cpu.state().history(), std::span(rows).first((c.ngram_size - 1) * c.heads())); });
        check(same_hash(saved, cpu.state()), "hash error reuse state");
        cpu.reset(); check(cpu.state().consumed_tokens() == 0 && cpu.state().segment_tokens() == 0, "hash reset");
        cpu.restore(saved); check(same_hash(saved, cpu.state()), "hash restore");
    }
    // General order/count, not a hardcoded two-order implementation.
    auto c = tiny_hash(); c.ngram_size = 5; c.heads_per_ngram = 1;
    c.multipliers = {UINT64_MAX, 17, 39, 0x8000000000000000ULL, 27};
    std::vector<std::int64_t> seq{1, 2, 9, 12, c.eos_token_id, 4, 17};
    PleHashCpu cpu(c); std::vector<std::uint64_t> rows(seq.size() * c.heads()); cpu.run(seq, rows);
    check(rows == formula(c, seq), "metadata-driven order 5");
    auto bad = c; bad.offsets.back() = UINT64_MAX;
    reject([&] { PleHashCpu b(bad); });
    bad = c; bad.vocab_sizes[0] = 0; reject([&] { bad.validate(); });
    bad = c; bad.multipliers.pop_back(); reject([&] { bad.validate(); });
    bad = c; bad.ngram_size = SIZE_MAX; reject([&] { bad.validate(); });
    check(ple_layer_from_hf(2, 48) == 1, "one-based HF layer conversion");
    reject([] { ple_layer_from_hf(0, 48); }); reject([] { ple_layer_from_hf(49, 48); });
    // Explicit INT64_MIN mixed and unsigned/signed mismatch fixture.
    c = tiny_hash(); c.eos_token_id = 0; c.multipliers = {0x8000000000000000ULL, 0, 0};
    std::vector<std::uint64_t> ids(c.heads()); ple_head_ids(c, 1, {}, ids);
    check(ids == formula(c, {1}), "INT64_MIN torch remainder");
    check(ids[0] != (0x8000000000000000ULL % c.vocab_sizes[0]), "mx unsigned differs on signed hash");
}

struct LayerFixture {
    PleLayerConfig c{8, 3, 2, 4, 3, 1e-4f};
    std::vector<float> key, value, conv, nk, nq, nc;
    Bytes kb, vb, cb;
    LayerFixture() : key(48), value(24), conv(24), nk(6), nq(6), nc(6) {
        for (std::size_t i = 0; i < key.size(); ++i) key[i] = float(int(i * 7 % 19) - 9) / 23;
        for (std::size_t i = 0; i < value.size(); ++i) value[i] = float(int(i * 11 % 17) - 8) / 13;
        for (std::size_t i = 0; i < conv.size(); ++i) conv[i] = float(int(i * 13 % 23) - 11) / 31;
        for (std::size_t i = 0; i < 6; ++i) { nk[i] = float(i + 1) / 5; nq[i] = float(7 - i) / 6; nc[i] = float(i + 3) / 7; }
        kb = f32bytes(key); vb = f32bytes(value); cb = f32bytes(conv);
    }
    PleParameters parameters() const {
        return {{TensorType::F32, 8, 6, kb}, {TensorType::F32, 8, 3, vb},
                {TensorType::F32, 4, 6, cb}, nk, nq, nc};
    }
    std::vector<Tensor> tensors(Bytes table, std::uint32_t type = 0,
                                std::vector<std::uint64_t> dims = {2, 11}) const {
        return {{"per_layer_token_embd.weight", dims, type, table},
            {"blk.1.ple_key.weight", {8, 6}, 0, kb}, {"blk.1.ple_value.weight", {8, 3}, 0, vb},
            {"blk.1.ple_conv1d.weight", {4, 6}, 0, cb},
            {"blk.1.ple_norm_key.weight", {6}, 0, f32bytes(nk)},
            {"blk.1.ple_norm_query.weight", {6}, 0, f32bytes(nq)},
            {"blk.1.ple_norm_conv.weight", {6}, 0, f32bytes(nc)}};
    }
};
std::vector<double> double_norm(std::span<const double> x, std::span<const float> gamma,
                                std::size_t group, double epsilon) {
    std::vector<double> out(x.size());
    for (std::size_t start = 0; start < x.size(); start += group) {
        double ss = 0; for (std::size_t d = 0; d < group; ++d) ss += x[start + d] * x[start + d];
        const double denominator = std::sqrt(ss / double(group) + epsilon);
        for (std::size_t d = 0; d < group; ++d) out[start + d] = x[start + d] / denominator * gamma[start + d];
    }
    return out;
}
double dsigmoid(double x) { return 1 / (1 + std::exp(-x)); }
struct DoubleOracle {
    const LayerFixture& f;
    std::vector<std::vector<double>> past;
    explicit DoubleOracle(const LayerFixture& fixture) : f(fixture) {}
    std::vector<double> step(std::span<const float> e, std::span<const float> h, bool keep) {
        std::vector<double> key(6, 0), value(3, 0), hidden(h.begin(), h.end()), gated(6);
        for (std::size_t r = 0; r < 6; ++r) for (std::size_t d = 0; d < 8; ++d) key[r] += double(f.key[r * 8 + d]) * e[d];
        for (std::size_t r = 0; r < 3; ++r) for (std::size_t d = 0; d < 8; ++d) value[r] += double(f.value[r * 8 + d]) * e[d];
        key = double_norm(key, f.nk, 3, f.c.rms_epsilon);
        hidden = double_norm(hidden, f.nq, 3, f.c.rms_epsilon);
        for (std::size_t stream = 0; stream < 2; ++stream) {
            double dot = 0; for (std::size_t d = 0; d < 3; ++d) dot += key[stream * 3 + d] * hidden[stream * 3 + d];
            dot /= std::sqrt(3.0);
            const double signed_root = dot == 0 ? 0 : std::copysign(std::sqrt(std::max(std::abs(dot), 1e-6)), dot);
            for (std::size_t d = 0; d < 3; ++d) gated[stream * 3 + d] = dsigmoid(signed_root) * value[d];
        }
        auto normed = double_norm(gated, f.nc, 3, f.c.rms_epsilon);
        if (!keep) { std::fill(gated.begin(), gated.end(), 0); std::fill(normed.begin(), normed.end(), 0); }
        past.push_back(normed);
        const auto t = past.size() - 1;
        std::vector<double> out(6);
        for (std::size_t channel = 0; channel < 6; ++channel) {
            double convolution = 0;
            for (std::size_t k = 0; k < 4; ++k) {
                const std::size_t lag = (3 - k) * 3;
                if (lag <= t) convolution += double(f.conv[channel * 4 + k]) * past[t - lag][channel];
            }
            out[channel] = gated[channel] + convolution * dsigmoid(convolution);
        }
        return out;
    }
};

void lookup_tests() {
    auto c = tiny_hash(); LayerFixture f;
    std::vector<float> table(22);
    for (std::size_t i = 0; i < table.size(); ++i) table[i] = float(i) / 9;
    Fixture file(c, f.tensors(f32bytes(table))); Model model(file.path);
    const auto loaded = ple_hash_config(model, 1);
    check(loaded == c && loaded.eos_token_id == 248044, "exact metadata/EOS independent of tokenizer");
    check(ple_layer_config(model, c) == f.c, "actual tensor geometry matches metadata");
    reject([&] { ple_hash_config(model, 2); });
    auto one_based_metadata = c; one_based_metadata.layer_index = 2;
    Fixture one_based_file(one_based_metadata, f.tensors(f32bytes(table))); Model one_based_model(one_based_file.path);
    reject([&] { ple_hash_config(one_based_model, 1); });
    PleHashCpu hash(c); PleLookup lookup(model, c);
    std::vector<std::uint64_t> rows(c.heads()); std::vector<float> out(8), expected(8);
    for (std::int64_t t : {1, 17, 248044, 248046}) {
        hash.step(t, rows); lookup.lookup(rows, out); ple_lookup_f32(c, table, rows, expected);
        check(out == expected, "file FP32 exact read_slice lookup");
        for (std::size_t h = 0; h < c.heads(); ++h)
            for (std::size_t d = 0; d < c.head_dim; ++d) check(out[h * c.head_dim + d] == table[rows[h] * c.head_dim + d], "head flatten order");
    }
    const auto before = out;
    auto invalid_rows = rows; invalid_rows.back() = c.table_rows;
    reject([&] { lookup.lookup(invalid_rows, out); }); check(out == before, "row bound atomic");
    invalid_rows = rows; invalid_rows[0] = 3;
    reject([&] { lookup.lookup(invalid_rows, out); }); check(out == before, "wrong head range rejected");
    auto badtable = table; badtable[rows.back() * 2] = std::numeric_limits<float>::infinity();
    reject([&] { ple_lookup_f32(c, badtable, rows, out); }); check(out == before, "late NaN gather atomic");
    reject([&] { ple_lookup_f32(c, table, rows, std::span(table).first(8)); });
    Fixture nonfinite(c, f.tensors(f32bytes(badtable))); Model nm(nonfinite.path); PleLookup nl(nm, c);
    reject([&] { nl.lookup(rows, out); }); check(out == before, "file late nonfinite atomic");
    Fixture rounded(c, f.tensors(f32bytes(table)), true); Model rm(rounded.path);
    reject([&] { ple_hash_config(rm, 1); });
    Fixture rank(c, {{"per_layer_token_embd.weight", {22}, 0, f32bytes(table)}}); Model rankm(rank.path);
    reject([&] { PleLookup l(rankm, c); });
    auto wrong = c; wrong.table_rows = 12; reject([&] { PleLookup l(model, wrong); });
    // Read errors must not expose partially assembled embeddings.
    check(ftruncate(file.fd, static_cast<off_t>(model.data_offset())) == 0, "truncate fixture");
    reject([&] { lookup.lookup(rows, out); }); check(out == before, "short read lookup atomic");

    // Full 16 logical heads and target Q4_0 width; only row counts scaled down.
    c = keep1_hash(false); Writer q4;
    for (std::size_t r = 0; r < 11; ++r) for (std::size_t block = 0; block < 5; ++block) {
        q4.number(r == 10 ? 0 : 0x3c00, 2); // half scale 0 or 1, including common zero
        for (std::size_t j = 0; j < 16; ++j) q4.number(0xa9, 1); // low=+1, high=+2
    }
    Fixture qfile(c, {{"per_layer_token_embd.weight", {160, 11}, 2, q4.bytes}});
    Model qm(qfile.path); const auto qc = ple_hash_config(qm, 1); PleLookup ql(qm, qc); PleHashCpu qh(qc);
    rows.resize(16); out.resize(2560); qh.step(123, rows); ql.lookup(rows, out);
    for (std::size_t h = 0; h < 16; ++h) for (std::size_t d = 0; d < 160; ++d)
        check(out[h * 160 + d] == (h == 0 || h == 8 ? (d % 32 < 16 ? 1.0f : 2.0f) : 0.0f), "keep1 logical 16x160 Q4 zero row");
    check(qm.file_size() < 4096, "fixture does not duplicate 3.6 GB table");
    allocation_probe::count = 0; allocation_probe::enabled = true;
    for (int i = 0; i < 5; ++i) { qh.step(123 + i, rows); ql.lookup(rows, out); }
    allocation_probe::enabled = false;
    check(allocation_probe::count == 0, "hash/file lookup per-step allocation");
}

void layer_tests() {
    LayerFixture f; const auto& c = f.c;
    const std::size_t n = 25;
    std::vector<float> e(n * 8), h(n * 6), out(n * 6), split_out(n * 6);
    std::vector<std::uint8_t> mask(n, 1); mask[4] = mask[12] = 0;
    for (std::size_t i = 0; i < e.size(); ++i) e[i] = float(int(i * 17 % 53) - 26) / 29;
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = float(int(i * 19 % 59) - 29) / 31;
    std::fill(h.begin(), h.begin() + 6, 0.0f); // exact zero dot -> gate 1/2
    PleLayerCpu cpu(c, f.parameters()), split(c, f.parameters());
    std::vector<PleLayerCheckpoint> prefixes;
    for (std::size_t i = 0; i <= n; ++i) prefixes.emplace_back(c);
    cpu.run(e, h, n, out, prefixes, mask);
    DoubleOracle reference(f);
    for (std::size_t t = 0; t < n; ++t) {
        const auto expected = reference.step(std::span(e).subspan(t * 8, 8), std::span(h).subspan(t * 6, 6), mask[t]);
        for (std::size_t d = 0; d < 6; ++d) near(out[t * 6 + d], expected[d]);
        check(prefixes[t + 1].consumed_tokens() == t + 1, "layer chronological prefix");
        for (std::size_t d = 0; d < 6; ++d) for (std::size_t age = 0; age < 9; ++age) {
            const double x = t + 1 + age < 9 ? 0 : reference.past[t + 1 + age - 9][d];
            near(prefixes[t + 1].conv_history()[d * 9 + age], x);
        }
    }
    check(std::any_of(out.begin() + 24, out.begin() + 30, [](float v) { return v != 0; }),
          "masked current token retains past conv contribution");
    for (std::size_t cut = 0; cut <= n; ++cut) {
        split.reset();
        split.run(std::span(e).first(cut * 8), std::span(h).first(cut * 6), cut,
                  std::span(split_out).first(cut * 6), {}, std::span(mask).first(cut));
        split.run(std::span(e).subspan(cut * 8), std::span(h).subspan(cut * 6), n - cut,
                  std::span(split_out).subspan(cut * 6), {}, std::span(mask).subspan(cut));
        check(same<float>(out, split_out) && same_layer(cpu.state(), split.state()), "every conv chunk boundary exact");
    }
    PleLayerCheckpoint saved(c); cpu.save(saved); const auto unchanged = out;
    auto bige = e; bige.back() = std::numeric_limits<float>::quiet_NaN();
    reject([&] { cpu.run(bige, h, n, out); });
    check(out == unchanged && same_layer(saved, cpu.state()), "late nonfinite input chunk atomic");
    reject([&] { cpu.run(e, h, n, out, std::span(prefixes).first(3)); });
    reject([&] { cpu.run(e, h, n, h); });
    reject([&] { cpu.step(std::span(e).first(8), cpu.state().conv_history().first(6), std::span(out).first(6)); });
    reject([&] { cpu.step(std::span(e).first(8), std::span(h).first(6), f.nk); });
    mask.back() = 2; reject([&] { cpu.run(e, h, n, out, {}, mask); }); mask.back() = 1;
    check(out == unchanged && same_layer(saved, cpu.state()), "aliases/geometry/mask atomic");
    auto bad = c; bad.hidden_size = 0; reject([&] { bad.validate(); });
    bad = c; bad.ngram_size = SIZE_MAX; reject([&] { bad.validate(); });
    bad = c; bad.rms_epsilon = std::numeric_limits<float>::infinity(); reject([&] { bad.validate(); });
    auto params = f.parameters(); params.key.output = 5; reject([&] { PleLayerCpu p(c, params); });
    auto bad_conv = f.cb;
    const auto inf = f32bytes(std::array<float, 1>{std::numeric_limits<float>::infinity()});
    std::copy(inf.begin(), inf.end(), bad_conv.end() - 4);
    params = f.parameters(); params.conv.weights = bad_conv;
    reject([&] { PleLayerCpu p(c, params); });
    PleLayerCpu clean(c, f.parameters()); clean.restore(saved);
    std::vector<float> huge(8, std::numeric_limits<float>::max()), one(6, -999);
    const auto sentinel = one;
    reject([&] { clean.step(huge, std::span(h).first(6), one); });
    check(one == sentinel && same_layer(saved, clean.state()), "arithmetic overflow failing token atomic");
    auto corrupt = saved;
    const_cast<float*>(corrupt.conv_history().data())[53] = std::numeric_limits<float>::quiet_NaN();
    reject([&] { clean.restore(corrupt); }); check(same_layer(saved, clean.state()), "nonfinite restore atomic");
    PleLayerCheckpoint bad_padding(c); const_cast<float*>(bad_padding.conv_history().data())[0] = 1;
    reject([&] { clean.restore(bad_padding); }); check(same_layer(saved, clean.state()), "history padding restore atomic");
    clean.reset(); check(clean.state().consumed_tokens() == 0, "layer reset count");
    for (float x : clean.state().conv_history()) check(std::bit_cast<std::uint32_t>(x) == 0, "reset positive zeros");
    allocation_probe::count = 0; allocation_probe::enabled = true;
    clean.run(e, h, n, out, prefixes, mask); clean.save(saved); clean.restore(prefixes[2]); clean.reset();
    allocation_probe::enabled = false;
    check(allocation_probe::count == 0, "layer run/save/restore/reset per-step allocation");

    // F16 depthwise weights with an independent exact dyadic encoder/oracle.
    LayerFixture half;
    Writer hw;
    for (std::size_t i = 0; i < half.conv.size(); ++i) {
        const int integer = int(i * 13 % 23) - 11;
        half.conv[i] = float(integer) / 32;
        const double magnitude = std::abs(double(half.conv[i]));
        std::uint16_t bits = 0;
        if (magnitude) {
            const int power = std::ilogb(magnitude);
            bits = std::uint16_t((power + 15) * 1024 + std::lround((std::ldexp(magnitude, -power) - 1) * 1024));
            if (integer < 0) bits |= 0x8000;
        }
        hw.number(bits, 2);
    }
    auto hp = half.parameters(); hp.conv = {TensorType::F16, 4, 6, hw.bytes};
    PleLayerCpu half_cpu(c, hp); DoubleOracle half_reference(half);
    half_cpu.run(e, h, n, out, {}, mask);
    for (std::size_t t = 0; t < n; ++t) {
        const auto expected = half_reference.step(std::span(e).subspan(t * 8, 8), std::span(h).subspan(t * 6, 6), mask[t]);
        for (std::size_t d = 0; d < 6; ++d) near(out[t * 6 + d], expected[d]);
    }
    // The HF layer returns an injection, never an internally merged residual.
    const Bytes zero_value(24 * 4, std::byte{0});
    auto zp = f.parameters(); zp.value.weights = zero_value;
    PleLayerCpu zero_layer(c, zp); std::array<float, 6> zero_injection{};
    zero_layer.step(std::span(e).first(8), std::span(h).subspan(6, 6), zero_injection);
    check(std::all_of(zero_injection.begin(), zero_injection.end(), [](float v) { return v == 0; }),
          "injection does not add residual or collapse HC");
    // A caller can borrow a checkpoint as immutable parameter storage; saving
    // into that same storage must be rejected before changing the parameters.
    PleLayerCheckpoint borrowed(c); auto bp = f.parameters(); bp.norm_key = borrowed.conv_history().first(6);
    PleLayerCpu borrowed_layer(c, bp);
    reject([&] { borrowed_layer.save(borrowed); });
    reject([&] { borrowed_layer.run({}, {}, 0, {}, std::span<PleLayerCheckpoint>(&borrowed, 1)); });
    // Empty ranges never alias, even when their pointers fall inside state.
    const auto empty_h = clean.state().conv_history().first(0);
    clean.run(empty_h, empty_h, 0, {});

    std::array<float, 6> x{1, -2, 3, 4, 0, -1}, gamma{1, 2, 3, 4, 5, 6}, result{};
    ple_group_rms_norm(x, gamma, 3, 1e-4f, result);
    const std::vector<double> xd(x.begin(), x.end()); const auto rn = double_norm(xd, gamma, 3, 1e-4f);
    for (std::size_t i = 0; i < 6; ++i) near(result[i], rn[i]);
    const auto old = result;
    gamma.back() = std::numeric_limits<float>::quiet_NaN();
    reject([&] { ple_group_rms_norm(x, gamma, 3, 1e-4f, result); }); check(result == old, "norm rejection atomic");
    gamma.back() = 6;
    reject([&] { ple_group_rms_norm(x, gamma, 0, 1e-4f, result); });
    reject([&] { ple_group_rms_norm(x, gamma, 3, 1e-4f, x); });
    std::array<float, 6> zero{}, key{1, 0, 0, -1e-8f, 0, 0}, query{1, 0, 0, 1, 0, 0};
    std::array<float, 3> v{2, -4, 0.5};
    ple_gate_values(zero, query, v, result);
    for (std::size_t i = 0; i < 6; ++i) check(result[i] == v[i % 3] * 0.5f, "gate exact zero");
    ple_gate_values(key, query, v, result);
    for (std::size_t i = 0; i < 6; ++i) near(result[i], v[i % 3] * dsigmoid(i < 3 ? std::sqrt(1 / std::sqrt(3.0)) : -0.001));
    const auto gated_before = result; query.back() = std::numeric_limits<float>::infinity();
    reject([&] { ple_gate_values(key, query, v, result); }); check(result == gated_before, "gate nonfinite atomic");
    query.back() = 0; reject([&] { ple_gate_values(key, query, v, query); });
}

// Couple hash+conv checkpoints exactly as verify3 integration will: token at slot
// 1 is already consumed target input, slots 2/3 are drafts. Rejects can cross EOS;
// accepted drafts are consumed inputs, replacement/bonus remains pending.
void rollback_tests() {
    auto c = tiny_hash(); LayerFixture f;
    PleHashCpu hash(c), sequential_hash(c); PleLayerCpu layer(f.c, f.parameters()), sequential_layer(f.c, f.parameters());
    std::vector<float> table(22); for (std::size_t i = 0; i < 22; ++i) table[i] = float(int(i) - 11) / 13;
    std::vector<PleHashCheckpoint> hp; std::vector<PleLayerCheckpoint> lp;
    for (int i = 0; i < 4; ++i) { hp.emplace_back(c); lp.emplace_back(f.c); }
    std::array<std::uint64_t, 12> rows{}; std::array<float, 24> embedding{};
    std::array<float, 18> hidden{}, out{}; std::array<float, 8> se{}; std::array<float, 6> so{};
    std::array<std::uint64_t, 4> sr{};
    for (std::size_t i = 0; i < hidden.size(); ++i) hidden[i] = float(int(i % 7) - 3) / 5;
    // Multiple repeated rejection windows, 0/1/2 accepts, EOS before/inside/after.
    for (std::size_t window = 0; window < 24; ++window) {
        const std::array<std::int64_t, 3> tokens{window % 4 == 0 ? c.eos_token_id : std::int64_t(window + 1),
                                               window % 4 == 1 ? c.eos_token_id : 123,
                                               window % 4 == 2 ? c.eos_token_id : 248046};
        const std::size_t accepts = window < 6 ? 0 : (window - 6) % 3;
        hash.run(tokens, rows, hp);
        for (std::size_t t = 0; t < 3; ++t)
            ple_lookup_f32(c, table, std::span(rows).subspan(t * 4, 4), std::span(embedding).subspan(t * 8, 8));
        layer.run(embedding, hidden, 3, out, lp);
        hash.restore(hp[1 + accepts]); layer.restore(lp[1 + accepts]);
        for (std::size_t t = 0; t < 1 + accepts; ++t) {
            sequential_hash.step(tokens[t], sr); ple_lookup_f32(c, table, sr, se);
            sequential_layer.step(se, std::span(hidden).subspan(t * 6, 6), so);
            check(std::equal(so.begin(), so.end(), out.begin() + t * 6), "verify accepted prefix output exact");
        }
        check(same_hash(hash.state(), sequential_hash.state()) && same_layer(layer.state(), sequential_layer.state()),
              "accept 0/1/2 hash/conv restore exact");
        // Actually consume the replacement next; outputs and states must match.
        const std::int64_t continuation = window % 5 == 0 ? c.eos_token_id : 299999;
        hash.step(continuation, sr); ple_lookup_f32(c, table, sr, se); layer.step(se, std::span(hidden).first(6), so);
        const auto previous = so;
        sequential_hash.step(continuation, sr); ple_lookup_f32(c, table, sr, se);
        sequential_layer.step(se, std::span(hidden).first(6), so);
        check(previous == so && same_hash(hash.state(), sequential_hash.state()) && same_layer(layer.state(), sequential_layer.state()),
              "continuation after repeated rollback/EOS exact");
    }
}

// Independent literal GGML Q4_0 / IEEE half readers for the actual raw-FP32
// reference. No matmul_f32, tensor_element, or production PLE math in this
// oracle. Norm gammas below are the already-folded GGUF values.
std::uint16_t literal_u16(std::span<const std::byte> data, std::size_t pos) {
    return std::uint16_t(std::to_integer<unsigned>(data[pos]) |
                         (std::to_integer<unsigned>(data[pos + 1]) << 8));
}
float literal_half(std::uint16_t bits) {
    const int exponent = (bits / 1024) % 32, fraction = bits % 1024;
    check(exponent != 31, "actual half parameter is finite");
    const double magnitude = exponent ? std::ldexp(double(1024 + fraction), exponent - 25) :
                                        std::ldexp(double(fraction), -24);
    return float(bits & 0x8000 ? -magnitude : magnitude);
}
float literal_weight(const QMatrix& matrix, std::size_t element) {
    if (matrix.type == TensorType::Q4_0) {
        const auto block = (element / 32) * 18;
        const auto lane = element % 32;
        const int code = int((std::to_integer<unsigned>(matrix.weights[block + 2 + lane % 16]) >>
                              (lane < 16 ? 0 : 4)) & 15);
        return float(code - 8) * literal_half(literal_u16(matrix.weights, block));
    }
    if (matrix.type == TensorType::F16) return literal_half(literal_u16(matrix.weights, element * 2));
    if (matrix.type == TensorType::F32) {
        std::uint32_t bits = 0;
        for (std::size_t b = 0; b < 4; ++b) bits |= std::uint32_t(std::to_integer<unsigned>(matrix.weights[element * 4 + b])) << (8 * b);
        return std::bit_cast<float>(bits);
    }
    throw std::runtime_error("raw reference supports only Q4_0, F16 and F32");
}
std::vector<float> raw_projection(const QMatrix& weights, std::span<const float> input) {
    check(input.size() == std::size_t(weights.input), "raw projection input geometry");
    std::vector<float> out(std::size_t(weights.output), 0.0f);
    // All weights are prevalidated finite by PleLayerCpu. Zero products do not
    // change an ascending FP32 sum (starting at +0); keep1's fourteen zero heads
    // can therefore be skipped in the REFERENCE without changing a weight or
    // rounding a nonzero product. Production still receives the original Q4.
    for (std::size_t row = 0; row < out.size(); ++row) {
        float sum = 0;
        for (std::size_t k = 0; k < input.size(); ++k) if (input[k] != 0) {
            const float term = literal_weight(weights, row * input.size() + k) * input[k];
            sum += term;
        }
        check(std::isfinite(sum), "raw projection finite"); out[row] = sum;
    }
    return out;
}
void raw_norm(std::span<const float> input, std::span<const float> gamma,
              const PleLayerConfig& c, std::span<float> output) {
    for (std::size_t stream = 0; stream < c.hc_count; ++stream) {
        const auto start = stream * c.hidden_size;
        float square_sum = 0;
        for (std::size_t d = 0; d < c.hidden_size; ++d) square_sum += input[start + d] * input[start + d];
        const float inverse = 1.0f / std::sqrt(square_sum / float(c.hidden_size) + c.rms_epsilon);
        for (std::size_t d = 0; d < c.hidden_size; ++d) output[start + d] = (input[start + d] * inverse) * gamma[start + d];
    }
}
struct RawReference {
    std::vector<float> normalized, injection;
    std::size_t cached_projection_cases = 0, oldest_tap_read_count = 0, oldest_nonzero_tap_product_count = 0;
};
RawReference raw_reference(const PleLayerConfig& c, const PleParameters& p,
                           std::span<const float> embeddings, std::span<const float> hidden,
                           std::size_t tokens) {
    const auto wide = c.wide_elements();
    check(embeddings.size() == tokens * c.embedding_dim && hidden.size() == tokens * wide, "raw reference token geometry");
    RawReference ref{std::vector<float>(tokens * wide), std::vector<float>(tokens * wide), 0, 0, 0};
    struct Projection { std::size_t token; std::vector<float> key, value; };
    std::vector<Projection> cache;
    std::vector<float> key(wide), query(wide), gated(wide), kernel(wide * c.conv_kernel);
    for (std::size_t i = 0; i < kernel.size(); ++i) kernel[i] = literal_weight(p.conv, i);
    // HF a005fc82 modular_qwen4_exp.py:719-783: full chronological timeline,
    // group RMS, shared value/signed-root gate, dilated taps, SiLU, injection.
    // Using the full timeline gives a reference independent of the incremental
    // history layout, chunk boundaries, save/restore, or EOS cache management.
    for (std::size_t t = 0; t < tokens; ++t) {
        const auto e = embeddings.subspan(t * c.embedding_dim, c.embedding_dim);
        std::size_t selected = 0;
        for (; selected < cache.size(); ++selected)
            if (same(e, embeddings.subspan(cache[selected].token * c.embedding_dim, c.embedding_dim))) break;
        if (selected == cache.size()) cache.push_back({t, raw_projection(p.key, e), raw_projection(p.value, e)});
        raw_norm(cache[selected].key, p.norm_key, c, key);
        raw_norm(hidden.subspan(t * wide, wide), p.norm_query, c, query);
        for (std::size_t stream = 0; stream < c.hc_count; ++stream) {
            const auto start = stream * c.hidden_size;
            float dot = 0;
            for (std::size_t d = 0; d < c.hidden_size; ++d) dot += key[start + d] * query[start + d];
            const float score = dot / std::sqrt(float(c.hidden_size));
            const float sign = score == 0 ? 0.0f : score > 0 ? 1.0f : -1.0f;
            const float root = sign * std::sqrt(std::max(std::abs(score), 1e-6f));
            const float gate = 1.0f / (1.0f + std::exp(-root));
            for (std::size_t d = 0; d < c.hidden_size; ++d) gated[start + d] = gate * cache[selected].value[d];
        }
        raw_norm(gated, p.norm_conv, c, std::span(ref.normalized).subspan(t * wide, wide));
        for (std::size_t f = 0; f < wide; ++f) {
            float convolution = 0;
            for (std::size_t k = 0; k < c.conv_kernel; ++k) {
                const auto lag = (c.conv_kernel - 1 - k) * c.ngram_size;
                const float x = lag <= t ? ref.normalized[(t - lag) * wide + f] : 0.0f;
                const float term = x * kernel[f * c.conv_kernel + k];
                convolution += term;
                if (!k && lag <= t) {
                    ++ref.oldest_tap_read_count;
                    ref.oldest_nonzero_tap_product_count += term != 0;
                }
            }
            ref.injection[t * wide + f] = gated[f] + convolution / (1.0f + std::exp(-convolution));
        }
    }
    ref.cached_projection_cases = cache.size();
    return ref;
}
struct ActualError {
    std::size_t elements = 0, bit_mismatches = 0;
    double max_abs = 0, max_scaled = 0;
};
void actual_near(float actual, float reference, ActualError& error) {
    const double absolute = std::abs(double(actual) - double(reference));
    const double scaled = absolute / (1 + std::abs(double(reference)));
    ++error.elements;
    error.bit_mismatches += std::bit_cast<std::uint32_t>(actual) != std::bit_cast<std::uint32_t>(reference);
    error.max_abs = std::max(error.max_abs, absolute); error.max_scaled = std::max(error.max_scaled, scaled);
    // Same threshold as the original unit oracle; no actual-only relaxation.
    check(std::isfinite(actual) && std::isfinite(reference) && scaled <= 3e-6, "actual raw FP32 reference mismatch");
}
void actual_exact(std::span<const float> actual, std::span<const float> reference, ActualError& error) {
    check(same(actual, reference), "actual chunk/prefix/continuation bitwise mismatch");
    for (std::size_t i = 0; i < actual.size(); ++i) actual_near(actual[i], reference[i], error);
}
void hash_history_proof(const PleHashCheckpoint& s, std::span<const std::int64_t> tokens) {
    const auto history = s.history();
    check(s.consumed_tokens() == tokens.size() && s.history_count() == std::min(history.size(), tokens.size()),
          "actual independent consumed/history count");
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto lag = history.size() - i;
        check(history[i] == (lag <= tokens.size() ? tokens[tokens.size() - lag] : s.config().eos_token_id),
              "actual independent raw history including EOS");
    }
    std::size_t segment = 0;
    while (segment < history.size() && segment < tokens.size() && tokens[tokens.size() - 1 - segment] != s.config().eos_token_id) ++segment;
    check(s.segment_tokens() == segment, "actual independent EOS segment boundary");
}
void conv_history_proof(const PleLayerCheckpoint& s, const RawReference& ref, std::size_t consumed, ActualError& error) {
    const auto& c = s.config(); const auto hist = c.history_length(), wide = c.wide_elements();
    check(s.consumed_tokens() == consumed && s.conv_history().size() == wide * hist, "actual convolution history count/geometry");
    for (std::size_t f = 0; f < wide; ++f) for (std::size_t age = 0; age < hist; ++age) {
        const auto lag = hist - age;
        const float expected = lag <= consumed ? ref.normalized[(consumed - lag) * wide + f] : 0.0f;
        actual_near(s.conv_history()[f * hist + age], expected, error);
    }
}
void actual_reference_tests() {
    // Exercise the actual-only readers/reference/history proofs locally on small
    // Q4/F16 parameters, without requiring the multi-GB target model.
    const PleLayerConfig c{32, 2, 2, 4, 3, 1e-4f};
    Writer key, value, conv;
    for (std::size_t r = 0; r < 4; ++r) {
        key.number(0x3800, 2);
        for (std::size_t b = 0; b < 16; ++b) key.number(((r + b) % 16) | (((r * 3 + b + 5) % 16) << 4), 1);
    }
    for (std::size_t r = 0; r < 2; ++r) {
        value.number(0x3400, 2);
        for (std::size_t b = 0; b < 16; ++b) value.number(((r * 5 + b) % 16) | (((r + b + 3) % 16) << 4), 1);
    }
    for (std::size_t i = 0; i < 16; ++i) conv.number(i % 2 ? 0xb000 : 0x3400, 2);
    const std::array<float, 4> gamma{1, 1.5, 0.5, 2};
    const PleParameters p{{TensorType::Q4_0, 32, 4, key.bytes}, {TensorType::Q4_0, 32, 2, value.bytes},
                           {TensorType::F16, 4, 4, conv.bytes}, gamma, gamma, gamma};
    PleLayerCpu cpu(c, p);
    std::vector<float> e(12 * 32), h(12 * 4), out(12 * 4);
    for (std::size_t t = 0; t < 12; ++t) {
        for (std::size_t d = 0; d < 32; ++d) e[t * 32 + d] = d % 3 ? 0 : float(int((d + (t % 4) * 7) % 17) - 8) / 19;
        for (std::size_t d = 0; d < 4; ++d) h[t * 4 + d] = float(int((d + t % 4) % 7) - 3) / 5;
    }
    std::vector<PleLayerCheckpoint> prefixes; for (int i = 0; i < 13; ++i) prefixes.emplace_back(c);
    cpu.run(e, h, 12, out, prefixes); const auto ref = raw_reference(c, p, e, h, 12);
    ActualError error;
    for (std::size_t i = 0; i < out.size(); ++i) actual_near(out[i], ref.injection[i], error);
    for (std::size_t t = 0; t <= 12; ++t) conv_history_proof(prefixes[t], ref, t, error);
    check(ref.cached_projection_cases == 4 && ref.oldest_tap_read_count == 12 && ref.oldest_nonzero_tap_product_count > 0,
          "actual reference cache and oldest dilated tap exercised locally");
}

void json_string(std::string_view text) {
    std::cout << '"';
    for (const unsigned char c : text) {
        switch (c) {
        case '"': std::cout << "\\\""; break;
        case '\\': std::cout << "\\\\"; break;
        case '\n': std::cout << "\\n"; break;
        case '\r': std::cout << "\\r"; break;
        case '\t': std::cout << "\\t"; break;
        default:
            if (c < 32) std::cout << "\\u00" << "0123456789abcdef"[c / 16] << "0123456789abcdef"[c % 16];
            else std::cout << char(c);
        }
    }
    std::cout << '"';
}
template<class T> void json_array(const T& values) {
    std::cout << '['; bool comma = false;
    for (auto value : values) {
        if (comma) std::cout << ',';
        comma = true;
        if constexpr (std::is_same_v<typename T::value_type, bool>) std::cout << (value ? "true" : "false");
        else std::cout << value;
    }
    std::cout << ']';
}
void json_error(const ActualError& e) {
    std::cout << "{\"elements\":" << e.elements << ",\"bit_mismatches\":" << e.bit_mismatches
              << ",\"max_abs\":" << e.max_abs << ",\"max_scaled\":" << e.max_scaled << '}';
}
struct ActualRollback {
    std::size_t accepted = 0, restored_count = 0, continued_count = 0, restored_segment = 0, continued_segment = 0;
    std::vector<std::int64_t> restored_history, continued_history;
    ActualError output, history;
    explicit ActualRollback(std::size_t history_size) : restored_history(history_size), continued_history(history_size) {}
};

// Optional target-machine actual-weight fixture. Reads only selected table rows
// and the six layer tensors; production uses original Q4/F16 weights and FP32
// math throughout. No GPU/full inference or duplicate table payload.
std::size_t actual_model(const char* path) {
    const auto checks_before = checks;
    Model model(path); const auto hc = ple_hash_config(model, 1); const auto lc = ple_layer_config(model, hc);
    check(hc.heads() == 16 && hc.head_dim == 160 && hc.table_rows == 40000085 && hc.eos_token_id == 248044,
          "target keep1 geometry");
    check(hc.vocab_sizes[0] == 20000003 && hc.offsets[0] == 0 &&
          hc.vocab_sizes[8] == 20000081 && hc.offsets[8] == 20000003,
          "target retained heads 0/8 ranges");
    for (std::size_t head = 0; head < 16; ++head) if (head != 0 && head != 8)
        check(hc.vocab_sizes[head] == 1 && hc.offsets[head] == 40000084, "target shared zero row ranges");
    check(lc.embedding_dim == 2560 && lc.hidden_size == 2560 && lc.hc_count == 4 &&
          lc.conv_kernel == 4 && lc.ngram_size == 3 && lc.history_length() == 9,
          "target widened 4x2560 PLE residual");
    const auto& multiplier_metadata = model.metadata_value("qwen4exp.ple.layer_multipliers");
    check(multiplier_metadata.type == MetadataType::ARRAY, "actual multipliers array metadata");
    const auto& multiplier_array = multiplier_metadata.get<MetadataArray>();
    check(multiplier_array.element_type == MetadataType::UINT64 &&
          std::get<std::vector<std::uint64_t>>(multiplier_array.values) == hc.multipliers,
          "actual multipliers preserved as exact uint64 metadata bits");
    const auto& tokenizer_eos_metadata = model.metadata_value("tokenizer.ggml.eos_token_id");
    check(tokenizer_eos_metadata.type == MetadataType::UINT32, "actual tokenizer EOS metadata type");
    const auto tokenizer_eos = tokenizer_eos_metadata.get<std::uint32_t>();
    check(tokenizer_eos == 248046 && tokenizer_eos != hc.eos_token_id && tokenizer_eos < hc.token_vocab_size,
          "actual PLE EOS is distinct from tokenizer EOS");
    const auto proof = hash_proof(hc);
    const std::size_t n = 12, wide = lc.wide_elements(), heads = hc.heads();
    std::vector<std::int64_t> tokens;
    for (std::size_t t = 0; t < n; ++t)
        tokens.push_back(std::array<std::int64_t, 4>{1, hc.eos_token_id, std::int64_t(tokenizer_eos),
                                                    std::int64_t(hc.token_vocab_size - 1)}[t % 4]);
    const auto diagnostic = hash_diagnostic(hc, tokens);
    auto matrix = [&](const char* suffix, std::size_t input, std::size_t output, Bytes& data) {
        const auto& t = model.tensor(std::string("blk.1.ple_") + suffix + ".weight");
        data.resize(static_cast<std::size_t>(t.byte_size)); model.read_tensor(t.name, data);
        return QMatrix{t.type, static_cast<int>(input), static_cast<int>(output), data};
    };
    Bytes kb, vb, cb; std::array<std::vector<float>, 3> norms;
    PleParameters p{matrix("key", lc.embedding_dim, lc.wide_elements(), kb),
                    matrix("value", lc.embedding_dim, lc.hidden_size, vb),
                    matrix("conv1d", lc.conv_kernel, lc.wide_elements(), cb), {}, {}, {}};
    check(p.key.type == TensorType::Q4_0 && p.value.type == TensorType::Q4_0 && p.conv.type == TensorType::F16,
          "target Q4 projections/F16 convolution");
    std::size_t i = 0;
    for (const auto* name : {"norm_key", "norm_query", "norm_conv"}) {
        const auto& t = model.tensor(std::string("blk.1.ple_") + name + ".weight");
        Bytes b(static_cast<std::size_t>(t.byte_size)); model.read_tensor(t.name, b);
        norms[i].resize(lc.wide_elements());
        for (std::size_t d = 0; d < lc.wide_elements(); ++d) norms[i][d] = tensor_element(t.type, b, d);
        ++i;
    }
    p.norm_key = norms[0]; p.norm_query = norms[1]; p.norm_conv = norms[2];
    PleHashCpu hash(hc); PleLookup lookup(model, hc); PleLayerCpu layer(lc, p);
    std::vector<std::uint64_t> rows(n * heads), divided_rows(rows.size()), single_rows(heads);
    std::vector<float> e(n * lc.embedding_dim), h(n * wide), out(n * wide), divided(out.size());
    std::vector<PleHashCheckpoint> hp;
    std::vector<PleLayerCheckpoint> lp;
    for (std::size_t t = 0; t <= n; ++t) { hp.emplace_back(hc); lp.emplace_back(lc); }
    for (std::size_t t = 0; t < n; ++t) for (std::size_t d = 0; d < wide; ++d)
        h[t * wide + d] = float(int((d * 17 + (t % 4) * 11) % 43) - 21) / 31;

    allocation_probe::count = 0; allocation_probe::enabled = true;
    hash.run(tokens, rows, hp);
    for (std::size_t t = 0; t < n; ++t)
        lookup.lookup(std::span(rows).subspan(t * heads, heads), std::span(e).subspan(t * lc.embedding_dim, lc.embedding_dim));
    layer.run(e, h, n, out, lp);
    allocation_probe::enabled = false;
    std::size_t runtime_formula_mismatches = 0;
    for (std::size_t i = 0; i < rows.size(); ++i) runtime_formula_mismatches += rows[i] != diagnostic.signed_rows[i];
    check(runtime_formula_mismatches == 0, "actual exact hash vs independent limb/signed formula");

    // Independently inspect exactly the selected Q4 rows, including the common
    // zero row's half scales. This remains a row-sized buffer, not a table copy.
    const auto& table = model.tensor("per_layer_token_embd.weight");
    check(table.type == TensorType::Q4_0, "actual table Q4_0");
    Bytes row_bytes(std::size_t(table.strides[1]));
    const QMatrix row_matrix{TensorType::Q4_0, int(hc.head_dim), 1, row_bytes};
    std::size_t logical_zero_elements = 0;
    for (std::size_t t = 0; t < n; ++t) for (std::size_t head = 0; head < heads; ++head) {
        model.read_slice(table.name, rows[t * heads + head] * table.strides[1], row_bytes);
        for (std::size_t d = 0; d < hc.head_dim; ++d) {
            const float value = e[t * lc.embedding_dim + head * hc.head_dim + d];
            check(std::bit_cast<std::uint32_t>(value) == std::bit_cast<std::uint32_t>(literal_weight(row_matrix, d)),
                  "actual selected table row independent Q4 decode exact");
            if (head != 0 && head != 8) { check(value == 0, "actual fourteen logical zero heads"); ++logical_zero_elements; }
        }
    }
    model.read_slice(table.name, 40000084ULL * table.strides[1], row_bytes);
    for (std::size_t block = 0; block < hc.head_dim / 32; ++block)
        check((literal_u16(row_bytes, block * 18) & 0x7fff) == 0, "actual common zero Q4 row has zero scales");

    const auto ref = raw_reference(lc, p, e, h, n);
    check(ref.cached_projection_cases <= 6 && ref.oldest_tap_read_count == (n - lc.history_length()) * wide,
          "actual bounded reference projections and nine-token-old taps exercised");
    ActualError raw_output_error, raw_history_error, split_output_error, split_history_error;
    for (std::size_t i = 0; i < out.size(); ++i) actual_near(out[i], ref.injection[i], raw_output_error);
    std::size_t eos_prefix_checks = 0, eos_prior_nonzero_history_elements = 0;
    for (std::size_t t = 0; t <= n; ++t) {
        hash_history_proof(hp[t], std::span(tokens).first(t));
        conv_history_proof(lp[t], ref, t, raw_history_error);
        if (t && tokens[t - 1] == hc.eos_token_id) {
            ++eos_prefix_checks;
            const auto predecessor_count = std::min(t - 1, lc.history_length() - 1);
            for (std::size_t f = 0; f < wide; ++f)
                for (std::size_t lag = 1; lag <= predecessor_count; ++lag)
                    eos_prior_nonzero_history_elements += lp[t].conv_history()[f * lc.history_length() + lc.history_length() - 1 - lag] != 0;
            const auto begin = t - 1 > hc.ngram_size - 1 ? t - hc.ngram_size : 0;
            ple_head_ids(hc, hc.eos_token_id, std::span(tokens).subspan(begin, t - 1 - begin), single_rows);
            check(std::equal(single_rows.begin(), single_rows.end(), rows.begin() + (t - 1) * heads),
                  "actual current EOS retains predecessor context");
            check(hp[t].segment_tokens() == 0, "actual EOS boundary established after consumption");
            if (t < n) {
                ple_head_ids(hc, tokens[t], {}, single_rows);
                check(std::equal(single_rows.begin(), single_rows.end(), rows.begin() + t * heads),
                      "actual token after EOS matches a fresh EOS-padded hash");
            }
        }
        if (t && tokens[t - 1] == tokenizer_eos) check(hp[t].segment_tokens() > 0, "tokenizer EOS does not reset PLE hash");
    }

    // One late split (10+2) crosses an already-filled nine-token conv history.
    const std::size_t cut = 10;
    allocation_probe::enabled = true;
    hash.reset(); layer.reset();
    hash.run(std::span(tokens).first(cut), std::span(divided_rows).first(cut * heads));
    layer.run(std::span(e).first(cut * lc.embedding_dim), std::span(h).first(cut * wide), cut, std::span(divided).first(cut * wide));
    check(same_hash(hash.state(), hp[cut]), "actual split boundary hash state exact");
    actual_exact(layer.state().conv_history(), lp[cut].conv_history(), split_history_error);
    hash.run(std::span(tokens).subspan(cut), std::span(divided_rows).subspan(cut * heads));
    layer.run(std::span(e).subspan(cut * lc.embedding_dim), std::span(h).subspan(cut * wide), n - cut, std::span(divided).subspan(cut * wide));
    allocation_probe::enabled = false;
    check(divided_rows == rows && same_hash(hash.state(), hp[n]) && same_layer(layer.state(), lp[n]), "actual split final states exact");
    actual_exact(divided, out, split_output_error);
    actual_exact(layer.state().conv_history(), lp[n].conv_history(), split_history_error);

    // Use a filled base history. Each 3-input window spans EOS in draft slot 1;
    // accept 0 rejects that EOS, accept 1/2 retains it. Prefix 1+a is restored,
    // then a genuinely consumed continuation is hashed/looked up/executed.
    const std::size_t base = 8;
    std::vector<PleHashCheckpoint> verify_hp;
    std::vector<PleLayerCheckpoint> verify_lp;
    for (int t = 0; t < 4; ++t) { verify_hp.emplace_back(hc); verify_lp.emplace_back(lc); }
    std::vector<std::uint64_t> verify_rows(3 * heads);
    std::vector<float> verify_e(3 * lc.embedding_dim), verify_out(3 * wide), next_e(lc.embedding_dim), next_out(wide);
    std::vector<ActualRollback> rollback;
    for (int a = 0; a < 3; ++a) rollback.emplace_back(hc.ngram_size - 1);
    // Evaluate the same verify window once. Its immutable chronological snapshots
    // can exercise all three restore decisions without decoding/projecting the
    // same actual Q4 weights three times for identical inputs.
    allocation_probe::enabled = true;
    hash.restore(hp[base]); layer.restore(lp[base]);
    hash.run(std::span(tokens).subspan(base, 3), verify_rows, verify_hp);
    for (std::size_t t = 0; t < 3; ++t)
        lookup.lookup(std::span(verify_rows).subspan(t * heads, heads), std::span(verify_e).subspan(t * lc.embedding_dim, lc.embedding_dim));
    layer.run(verify_e, std::span(h).subspan(base * wide, 3 * wide), 3, verify_out, verify_lp);
    allocation_probe::enabled = false;
    for (std::size_t accepted = 0; accepted < 3; ++accepted) {
        auto& r = rollback[accepted]; r.accepted = accepted;
        allocation_probe::enabled = true;
        hash.restore(verify_hp[1 + accepted]); layer.restore(verify_lp[1 + accepted]);
        allocation_probe::enabled = false;
        check(same<std::uint64_t>(verify_rows, std::span<const std::uint64_t>(rows).subspan(base * heads, 3 * heads)) &&
              same<float>(verify_e, std::span<const float>(e).subspan(base * lc.embedding_dim, 3 * lc.embedding_dim)),
              "actual verify3 rows and loaded embeddings exact");
        actual_exact(verify_out, std::span<const float>(out).subspan(base * wide, 3 * wide), r.output);
        for (std::size_t slot = 0; slot < 4; ++slot) {
            check(same_hash(verify_hp[slot], hp[base + slot]), "actual verify3 chronological hash prefix exact");
            actual_exact(verify_lp[slot].conv_history(), lp[base + slot].conv_history(), r.history);
            check(verify_lp[slot].consumed_tokens() == base + slot, "actual verify3 chronological conv prefix count");
        }
        const auto consumed = base + 1 + accepted;
        check(same_hash(hash.state(), hp[consumed]) && same_layer(layer.state(), lp[consumed]), "actual accepted prefix restores both states exactly");
        hash_history_proof(hash.state(), std::span(tokens).first(consumed));
        conv_history_proof(layer.state(), ref, consumed, raw_history_error);
        r.restored_count = std::size_t(hash.state().consumed_tokens()); r.restored_segment = hash.state().segment_tokens();
        std::copy(hash.state().history().begin(), hash.state().history().end(), r.restored_history.begin());
        allocation_probe::enabled = true;
        hash.step(tokens[consumed], single_rows); lookup.lookup(single_rows, next_e);
        layer.step(next_e, std::span(h).subspan(consumed * wide, wide), next_out);
        allocation_probe::enabled = false;
        check(same<std::uint64_t>(single_rows, std::span<const std::uint64_t>(rows).subspan(consumed * heads, heads)) &&
              same<float>(next_e, std::span<const float>(e).subspan(consumed * lc.embedding_dim, lc.embedding_dim)),
              "actual continuation uses correct rolled-back hash/lookup");
        actual_exact(next_out, std::span<const float>(out).subspan(consumed * wide, wide), r.output);
        check(same_hash(hash.state(), hp[consumed + 1]) && same_layer(layer.state(), lp[consumed + 1]), "actual continuation states exact");
        actual_exact(layer.state().conv_history(), lp[consumed + 1].conv_history(), r.history);
        hash_history_proof(hash.state(), std::span(tokens).first(consumed + 1));
        conv_history_proof(layer.state(), ref, consumed + 1, raw_history_error);
        r.continued_count = std::size_t(hash.state().consumed_tokens()); r.continued_segment = hash.state().segment_tokens();
        std::copy(hash.state().history().begin(), hash.state().history().end(), r.continued_history.begin());
    }
    const auto hot_allocations = allocation_probe::count;
    check(hot_allocations == 0, "actual preallocated hash/lookup/layer/chunk/restore zero allocations");

    // JSON integer multipliers remain exact uint64 values (never float). Sampled
    // signed/unsigned counts are diagnostics, not claims of an actual mx bug.
    std::cout << std::setprecision(17) << "{\"kind\":\"ple_actual_model\",\"passed\":true,\"model\":";
    json_string(path);
    std::cout << ",\"mode\":\"cpu_raw_fp32\",\"actual_model_cases\":5,\"layer_index_zero_based\":" << hc.layer_index
              << ",\"hf_layer_one_based\":" << hc.layer_index + 1 << ",\"token_vocab_size\":" << hc.token_vocab_size
              << ",\"table_rows\":" << hc.table_rows << ",\"head_dim\":" << hc.head_dim << ",\"heads\":" << heads
              << ",\"table_name\":\"per_layer_token_embd.weight\",\"table_type\":\"Q4_0\",\"table_dimensions\":[" << hc.head_dim << ',' << hc.table_rows
              << "],\"table_row_bytes\":" << table.strides[1]
              << ",\"ngram_size\":" << hc.ngram_size << ",\"heads_per_ngram\":" << hc.heads_per_ngram
              << ",\"ple_eos_token_id\":" << hc.eos_token_id << ",\"tokenizer_eos_token_id\":" << tokenizer_eos
              << ",\"multipliers_encoding\":\"uint64_bits\",\"multipliers\":";
    json_array(hc.multipliers); std::cout << ",\"head_vocab_sizes\":"; json_array(hc.vocab_sizes);
    std::cout << ",\"head_offsets\":"; json_array(hc.offsets);
    std::cout << ",\"head_ranges\":[";
    for (std::size_t head = 0; head < heads; ++head) {
        if (head) std::cout << ',';
        std::cout << "{\"head\":" << head << ",\"order\":" << 2 + head / hc.heads_per_ngram
                  << ",\"vocab_size\":" << hc.vocab_sizes[head] << ",\"offset\":" << hc.offsets[head]
                  << ",\"last_row_inclusive\":" << hc.offsets[head] + hc.vocab_sizes[head] - 1 << '}';
    }
    std::cout << "],\"maximum_valid_token_id_including_eos\":" << proof.maximum_valid_token_id_including_eos
              << ",\"hf_generator_multiplier_max\":" << proof.hf_generator_multiplier_max
              << ",\"valid_id_multiplier_bound\":" << proof.valid_id_multiplier_bound
              << ",\"generator_bound_satisfied\":" << (proof.generator_bound_satisfied ? "true" : "false")
              << ",\"nonnegative_hash_proven_for_all_valid_ids\":" << (proof.nonnegative_hash_proven_for_all_valid_ids ? "true" : "false")
              << ",\"product_fits_int64_per_multiplier\":";
    json_array(proof.products_fit_int64);
    std::cout << ",\"maximum_valid_token_products\":[";
    for (std::size_t m = 0; m < hc.multipliers.size(); ++m) {
        if (m) std::cout << ',';
        if (!proof.maximum_valid_token_id_including_eos || hc.multipliers[m] <= UINT64_MAX / proof.maximum_valid_token_id_including_eos)
            std::cout << hc.multipliers[m] * proof.maximum_valid_token_id_including_eos;
        else std::cout << "null"; // actual product exceeds UINT64, never report a wrapped bound
    }
    std::cout << "],\"proof_method\":\"max_valid_token_product_le_int64_max_implies_xor_high_bit_zero\",\"diagnostic_tokens\":";
    json_array(tokens);
    std::cout << ",\"negative_hash_count\":" << diagnostic.negative_hash_count
              << ",\"negative_hash_count_unit\":\"token_order\",\"mismatched_row_count\":" << diagnostic.mismatched_row_count
              << ",\"runtime_vs_formula_mismatched_row_count\":" << runtime_formula_mismatches
              << ",\"hash_semantics\":\"signed_int64_torch_remainder\",\"first_rows\":[";
    for (std::size_t t = 0; t < 4; ++t) {
        if (t) std::cout << ',';
        std::cout << "{\"token_index\":" << t << ",\"token_id\":" << tokens[t] << ",\"rows\":";
        json_array(std::span<const std::uint64_t>(rows).subspan(t * heads, heads));
        std::cout << ",\"unsigned_rows\":"; json_array(std::span<const std::uint64_t>(diagnostic.unsigned_rows).subspan(t * heads, heads)); std::cout << '}';
    }
    std::cout << "],\"hidden_size\":" << lc.hidden_size << ",\"hc_count\":" << lc.hc_count
              << ",\"conv_kernel\":" << lc.conv_kernel << ",\"dilation\":" << lc.ngram_size
              << ",\"history_length\":" << lc.history_length() << ",\"rms_epsilon\":" << lc.rms_epsilon
              << ",\"canonical_tokens\":" << n << ",\"canonical_prefixes\":" << hp.size()
              << ",\"production_forward_steps\":" << n * 2 + 3 + rollback.size()
              << ",\"verify3_window_evaluations\":1,\"verify3_acceptance_cases\":" << rollback.size()
              << ",\"reference_projection_cases\":" << ref.cached_projection_cases
              << ",\"oldest_tap_read_count\":" << ref.oldest_tap_read_count
              << ",\"oldest_nonzero_tap_product_count\":" << ref.oldest_nonzero_tap_product_count
              << ",\"logical_zero_elements_checked\":" << logical_zero_elements << ",\"eos_prefix_checks\":" << eos_prefix_checks
              << ",\"eos_prior_nonzero_history_elements\":" << eos_prior_nonzero_history_elements
              << ",\"eos_hash_semantics_proven_for_selected_sequence\":true,\"tokenizer_eos_is_not_ple_boundary\":true"
              << ",\"hash_history_proven_exact\":true,\"conv_history_proven_against_raw_fp32\":true"
              << ",\"eos_does_not_reset_conv_history\":true,\"raw_fp32_output_error\":";
    json_error(raw_output_error); std::cout << ",\"raw_fp32_history_error\":"; json_error(raw_history_error);
    std::cout << ",\"chunk_split_after_consumed\":" << cut << ",\"chunk_bitwise_exact\":true,\"chunk_output_error\":";
    json_error(split_output_error); std::cout << ",\"chunk_history_error\":"; json_error(split_history_error);
    std::cout << ",\"verify3\":[";
    for (const auto& r : rollback) {
        if (r.accepted) std::cout << ',';
        std::cout << "{\"accepted_drafts\":" << r.accepted << ",\"prefix_slot\":" << 1 + r.accepted
                  << ",\"base_consumed_tokens\":" << base << ",\"restored_consumed_tokens\":" << r.restored_count
                  << ",\"continued_consumed_tokens\":" << r.continued_count << ",\"restored_segment_tokens\":" << r.restored_segment
                  << ",\"continued_segment_tokens\":" << r.continued_segment << ",\"restored_hash_history\":";
        json_array(r.restored_history); std::cout << ",\"continued_hash_history\":"; json_array(r.continued_history);
        std::cout << ",\"chronological_prefix_count\":4,\"prefix_hash_and_conv_bitwise_exact\":true"
                  << ",\"continuation_hash_and_conv_bitwise_exact\":true,\"output_error\":";
        json_error(r.output); std::cout << ",\"history_error\":"; json_error(r.history); std::cout << '}';
    }
    std::cout << "],\"verify3_prefixes_bitwise_exact\":true,\"continuations_bitwise_exact\":true,\"hot_allocations\":" << hot_allocations
              << ",\"checks\":" << checks - checks_before << "}\n";
    return 5; // full raw oracle + late chunk split + accept-0/1/2 verify windows
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "usage: ple-test [--model GGUF]\n"; return 0;
        }
        const bool run_actual = argc != 1;
        if (run_actual && (argc != 3 || std::string_view(argv[1]) != "--model" ||
                           std::string_view(argv[2]).empty() || std::string_view(argv[2]).starts_with("--")))
            throw std::runtime_error("usage: ple-test [--model GGUF]");
        hash_tests(); lookup_tests(); layer_tests(); rollback_tests();
        hash_diagnostic_tests(); actual_reference_tests();
        const auto actual_cases = run_actual ? actual_model(argv[2]) : 0;
        std::cout << "{\"test\":\"ple\",\"passed\":true,\"checks\":" << checks << ",\"rejected\":" << rejected
                  << ",\"max_abs_double_error\":" << max_error << ",\"actual_model\":" << (run_actual ? "true" : "false")
                  << ",\"actual_model_cases\":" << actual_cases
                  << ",\"revision\":\"" << CORE_REVISION << "\",\"dirty\":" << CORE_DIRTY << "}\n";
        return 0;
    } catch (const std::exception& e) {
        allocation_probe::enabled = false;
        std::cerr << "ple-test: " << e.what() << '\n'; return 1;
    }
}
