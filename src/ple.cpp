#include "ple.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

// WRITE OURS: no donor code copied. Equation/ABI sources:
// Transformers (Apache-2.0), a005fc82babfe8871d87746decad2dbee100a125,
// modular_qwen4_exp.py:608-708 (hash/EOS/head order), 719-783 (PLE), 796/811-814
// (layer numbering/injection). Its inherited Qwen3_5 RMSNorm is defined at
// modeling_qwen3_5.py:840-854 (FP32 norm and 1+w).
// mx-llama.cpp (MIT), dcd685463d597d31f5ca759d32c94592a2740fa4,
// conversion/qwen4exp.py:120-163/182-188 (metadata and folded gamma),
// src/models/qwen4exp.cpp:1228-1252/1328-1418 (hash and GGUF PLE graph).
// HF uses signed remainder; mx uses unsigned mixed%v. These are equivalent when
// loaded multipliers keep every valid token product <= INT64_MAX (XOR then also
// has its high bit clear). Actual-model fixtures check that bound; synthetic
// negative/overflow cases alone are NOT evidence of a selected-model fork bug.

namespace qwen {
namespace {

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }

std::size_t mul(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) invalid("PLE: size overflow");
    return a * b;
}
std::size_t narrow(std::uint64_t n) {
    if (n > std::numeric_limits<std::size_t>::max()) invalid("PLE: size overflow");
    return static_cast<std::size_t>(n);
}
struct Range { std::uintptr_t first, last; };
Range range(const void* p, std::size_t bytes) {
    const auto first = reinterpret_cast<std::uintptr_t>(p);
    if (bytes && (!p || bytes > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) ||
                  bytes > std::numeric_limits<std::uintptr_t>::max() - first))
        invalid("PLE: invalid address range");
    return {first, first + bytes};
}
template<class T> Range range(std::span<T> s) { return range(s.data(), mul(s.size(), sizeof(T))); }
template<class T> Range range(const std::vector<T>& v) { return range(std::span<const T>(v)); }
bool overlap(Range a, Range b) {
    return a.first != a.last && b.first != b.last && a.first < b.last && b.first < a.last;
}
void disjoint(Range a, Range b) { if (overlap(a, b)) invalid("PLE: alias"); }
void finite(std::span<const float> values) {
    for (float x : values) if (!std::isfinite(x)) invalid("PLE: nonfinite input/parameter");
}
float checked(float x) {
    if (!std::isfinite(x)) invalid("PLE: nonfinite arithmetic");
    return x;
}
void token_valid(const PleHashConfig& c, std::int64_t token) {
    if (token < 0 || std::uint64_t(token) >= c.token_vocab_size) invalid("PLE: invalid token ID");
}
void config_disjoint(Range a, const PleHashConfig& c) {
    disjoint(a, range(&c, sizeof(c)));
    disjoint(a, range(c.multipliers)); disjoint(a, range(c.vocab_sizes)); disjoint(a, range(c.offsets));
}
void rows_valid(const PleHashConfig& c, std::span<const std::uint64_t> rows) {
    if (rows.size() != c.heads()) invalid("PLE: head ID count");
    range(rows);
    for (std::size_t h = 0; h < rows.size(); ++h)
        if (rows[h] < c.offsets[h] || rows[h] - c.offsets[h] >= c.vocab_sizes[h])
            invalid("PLE: head row out of range");
}

// Avoid signed overflow, implementation-defined casts and negating INT64_MIN.
std::uint64_t torch_remainder(std::uint64_t bits, std::uint64_t divisor) {
    if (!(bits >> 63)) return bits % divisor;
    const std::uint64_t magnitude = (~bits) + 1;
    const auto r = magnitude % divisor;
    return r ? divisor - r : 0;
}
void hash_unchecked(const PleHashConfig& c, std::int64_t token,
                    std::span<const std::int64_t> history, std::span<std::uint64_t> output) {
    std::uint64_t mixed = std::uint64_t(token) * c.multipliers[0];
    bool cut = false;
    for (std::size_t p = 1; p < c.ngram_size; ++p) {
        const auto t = p > history.size() ? c.eos_token_id : history[history.size() - p];
        cut = cut || p > history.size() || t == c.eos_token_id;
        mixed ^= std::uint64_t(cut ? c.eos_token_id : t) * c.multipliers[p];
        for (std::size_t g = 0; g < c.heads_per_ngram; ++g) {
            const auto h = (p - 1) * c.heads_per_ngram + g;
            output[h] = torch_remainder(mixed, c.vocab_sizes[h]) + c.offsets[h];
        }
    }
}

std::uint64_t integer(const MetadataValue& v) {
    return std::visit([](const auto& x) -> std::uint64_t {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            if constexpr (std::is_signed_v<T>) if (x < 0) invalid("PLE: negative metadata");
            return static_cast<std::uint64_t>(x);
        } else invalid("PLE: integer metadata required");
    }, v.value);
}
std::uint64_t meta(const Model& m, std::string_view key) {
    const auto* v = m.find_metadata(key);
    if (!v) invalid("PLE: missing metadata");
    return integer(*v);
}
std::vector<std::uint64_t> array(const Model& m, std::string_view key, bool u64) {
    const auto* v = m.find_metadata(key);
    if (!v || v->type != MetadataType::ARRAY) invalid("PLE: missing array metadata");
    const auto& a = v->get<MetadataArray>();
    if (u64 && a.element_type != MetadataType::UINT64) invalid("PLE: exact UINT64 array required");
    return std::visit([](const auto& xs) -> std::vector<std::uint64_t> {
        using T = typename std::decay_t<decltype(xs)>::value_type;
        if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
            std::vector<std::uint64_t> result;
            result.reserve(xs.size());
            for (T x : xs) {
                if constexpr (std::is_signed_v<T>) if (x < 0) invalid("PLE: negative metadata");
                result.push_back(static_cast<std::uint64_t>(x));
            }
            return result;
        } else invalid("PLE: integer array required");
    }, a.values);
}
std::size_t table_geometry(const TensorView& t, const PleHashConfig& c) {
    const auto l = type_layout(t.type);
    if (t.rank != 2 || t.dimensions[0] != c.head_dim || t.dimensions[1] != c.table_rows ||
        c.head_dim % l.block_elements != 0) invalid("PLE: table geometry");
    const auto row_bytes = mul(c.head_dim / l.block_elements, l.block_bytes);
    if (t.strides[0] != l.block_bytes || t.strides[1] != row_bytes ||
        t.elements != mul(c.head_dim, narrow(c.table_rows)) ||
        t.byte_size != mul(row_bytes, narrow(c.table_rows))) invalid("PLE: table bytes/strides");
    return row_bytes;
}
void tensor_shape(const Model& m, const std::string& name, std::size_t a, std::size_t b = 0) {
    const auto& t = m.tensor(name);
    if (t.rank != (b ? 2U : 1U) || t.dimensions[0] != a || (b && t.dimensions[1] != b))
        invalid("PLE: parameter tensor geometry");
    if (!b && t.type != TensorType::F32) invalid("PLE: norm tensor must be F32");
}
void matrix_valid(const QMatrix& m, std::size_t input, std::size_t output) {
    const auto l = type_layout(m.type);
    if (input > std::size_t(std::numeric_limits<int>::max()) ||
        output > std::size_t(std::numeric_limits<int>::max()) ||
        m.input != static_cast<int>(input) || m.output != static_cast<int>(output) ||
        input % l.block_elements != 0 ||
        m.weights.size() != mul(mul(input / l.block_elements, l.block_bytes), output))
        invalid("PLE: matrix geometry");
    range(m.weights);
    for (std::size_t i = 0; i < mul(input, output); ++i)
        checked(tensor_element(m.type, m.weights, i));
}
float sigmoid(float x) {
    if (x >= 0) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
}
float norm_scale(std::span<const float> x, float epsilon) {
    float sum = 0.0f;
    for (float v : x) sum += v * v;
    checked(sum);
    return checked(1.0f / std::sqrt(checked(sum / float(x.size()) + epsilon)));
}
float gate_scale(std::span<const float> k, std::span<const float> q) {
    float sum = 0.0f;
    for (std::size_t d = 0; d < k.size(); ++d) sum += k[d] * q[d];
    const float s = checked(sum / std::sqrt(float(k.size())));
    const float sign = s > 0 ? 1.0f : s < 0 ? -1.0f : 0.0f;
    return sigmoid(sign * std::sqrt(std::max(std::abs(s), 1.0e-6f)));
}

} // namespace

std::size_t PleHashConfig::heads() const {
    if (ngram_size < 2 || !heads_per_ngram) invalid("PLE: ngram/head geometry");
    return mul(ngram_size - 1, heads_per_ngram);
}
std::size_t PleHashConfig::embedding_elements() const { return mul(heads(), head_dim); }
void PleHashConfig::validate() const {
    const auto h = heads();
    if (!head_dim || !table_rows || table_rows > std::uint64_t(INT64_MAX) ||
        !token_vocab_size || token_vocab_size > std::uint64_t(INT64_MAX) ||
        multipliers.size() != ngram_size || vocab_sizes.size() != h || offsets.size() != h)
        invalid("PLE: hash metadata geometry");
    token_valid(*this, eos_token_id);
    mul(embedding_elements(), sizeof(float));
    mul(ngram_size - 1, sizeof(std::int64_t));
    for (std::size_t i = 0; i < h; ++i)
        if (!vocab_sizes[i] || offsets[i] >= table_rows || vocab_sizes[i] > table_rows - offsets[i])
            invalid("PLE: head range exceeds table");
}
std::uint32_t ple_layer_from_hf(std::uint32_t one_based, std::uint32_t blocks) {
    if (!one_based || one_based > blocks) invalid("PLE: invalid one-based layer");
    return one_based - 1;
}
PleHashConfig ple_hash_config(const Model& m, std::uint32_t layer) {
    const auto& arch = m.metadata_value("general.architecture");
    if (arch.type != MetadataType::STRING || arch.get<std::string>() != "qwen4exp")
        invalid("PLE: architecture must be qwen4exp");
    const auto layers = array(m, "qwen4exp.ple.layers", false);
    const auto blocks = meta(m, "qwen4exp.block_count");
    if (layers.size() != 1 || layers[0] >= blocks || layers[0] != layer)
        invalid("PLE: zero-based layer metadata mismatch");
    PleHashConfig c;
    c.layer_index = layer;
    c.ngram_size = narrow(meta(m, "qwen4exp.ple.ngram_size"));
    c.heads_per_ngram = narrow(meta(m, "qwen4exp.ple.heads_per_ngram"));
    c.head_dim = narrow(meta(m, "qwen4exp.embedding_length_per_layer_input"));
    const auto eos = meta(m, "qwen4exp.ple.eos_token_id");
    if (eos > std::uint64_t(INT64_MAX)) invalid("PLE: EOS overflow");
    c.eos_token_id = static_cast<std::int64_t>(eos);
    if (m.find_metadata("qwen4exp.vocab_size")) c.token_vocab_size = meta(m, "qwen4exp.vocab_size");
    else {
        const auto& t = m.tensor("token_embd.weight");
        if (t.rank != 2) invalid("PLE: token vocabulary geometry");
        c.token_vocab_size = t.dimensions[1];
    }
    c.multipliers = array(m, "qwen4exp.ple.layer_multipliers", true);
    c.vocab_sizes = array(m, "qwen4exp.ple.head_vocab_sizes", true);
    c.offsets = array(m, "qwen4exp.ple.head_offsets", true);
    const auto& table = m.tensor("per_layer_token_embd.weight");
    c.table_rows = table.dimensions[1];
    c.validate();
    table_geometry(table, c);
    return c;
}
void ple_head_ids(const PleHashConfig& c, std::int64_t token,
                  std::span<const std::int64_t> history, std::span<std::uint64_t> output) {
    c.validate(); token_valid(c, token);
    if (history.size() > c.ngram_size - 1 || output.size() != c.heads()) invalid("PLE: hash span size");
    for (auto t : history) token_valid(c, t);
    disjoint(range(output), range(history)); config_disjoint(range(output), c);
    hash_unchecked(c, token, history, output);
}
PleHashCheckpoint::PleHashCheckpoint(PleHashConfig c) : config_(std::move(c)) {
    config_.validate(); history_.assign(config_.ngram_size - 1, config_.eos_token_id);
}
PleHashCpu::PleHashCpu(PleHashConfig c) : state_(std::move(c)) {}
void PleHashCpu::reset() noexcept {
    std::fill(state_.history_.begin(), state_.history_.end(), state_.config_.eos_token_id);
    state_.history_count_ = state_.segment_tokens_ = 0; state_.consumed_tokens_ = 0;
}
void PleHashCpu::validate_checkpoint(const PleHashCheckpoint& s) const {
    if (!(s.config_ == state_.config_) || s.history_.size() != state_.history_.size() ||
        s.history_count_ != std::min<std::uint64_t>(s.consumed_tokens_, s.history_.size()) ||
        s.segment_tokens_ > s.history_count_) invalid("PLE: hash checkpoint geometry/count");
    for (auto t : s.history_) token_valid(s.config_, t);
    std::size_t segment = 0;
    for (auto it = s.history_.rbegin(); it != s.history_.rend(); ++it) {
        if (*it == s.config_.eos_token_id) break;
        ++segment;
    }
    if (segment != s.segment_tokens_) invalid("PLE: hash checkpoint EOS boundary");
    for (std::size_t i = 0; i < s.history_.size() - s.history_count_; ++i)
        if (s.history_[i] != s.config_.eos_token_id) invalid("PLE: hash checkpoint padding");
}
void PleHashCpu::save(PleHashCheckpoint& s) const {
    validate_checkpoint(s);
    if (&s == &state_) return;
    std::copy(state_.history_.begin(), state_.history_.end(), s.history_.begin());
    s.history_count_ = state_.history_count_; s.segment_tokens_ = state_.segment_tokens_;
    s.consumed_tokens_ = state_.consumed_tokens_;
}
void PleHashCpu::restore(const PleHashCheckpoint& s) {
    validate_checkpoint(s);
    if (&s == &state_) return;
    std::copy(s.history_.begin(), s.history_.end(), state_.history_.begin());
    state_.history_count_ = s.history_count_; state_.segment_tokens_ = s.segment_tokens_;
    state_.consumed_tokens_ = s.consumed_tokens_;
}
void PleHashCpu::step(std::int64_t token, std::span<std::uint64_t> rows) {
    run(std::span<const std::int64_t>(&token, 1), rows);
}
void PleHashCpu::run(std::span<const std::int64_t> tokens, std::span<std::uint64_t> rows,
                     std::span<PleHashCheckpoint> prefixes) {
    const auto& c = state_.config_;
    if (rows.size() != mul(tokens.size(), c.heads()) ||
        (!prefixes.empty() && (tokens.size() == SIZE_MAX || prefixes.size() != tokens.size() + 1)) ||
        tokens.size() > UINT64_MAX - state_.consumed_tokens_) invalid("PLE: hash chunk geometry/count");
    const auto r = range(rows), t = range(tokens);
    disjoint(r, t); config_disjoint(r, c);
    disjoint(r, range(&state_, sizeof(state_))); disjoint(t, range(&state_, sizeof(state_)));
    disjoint(r, range(state_.history_)); disjoint(t, range(state_.history_));
    disjoint(r, range(prefixes)); disjoint(t, range(prefixes));
    for (auto token : tokens) token_valid(c, token);
    for (auto& s : prefixes) {
        validate_checkpoint(s);
        if (&s == &state_) invalid("PLE: prefix aliases state");
        disjoint(r, range(s.history_)); disjoint(t, range(s.history_)); config_disjoint(r, s.config_);
    }
    if (!prefixes.empty()) save(prefixes[0]);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        hash_unchecked(c, tokens[i], state_.history_, rows.subspan(i * c.heads(), c.heads()));
        std::move(state_.history_.begin() + 1, state_.history_.end(), state_.history_.begin());
        state_.history_.back() = tokens[i];
        state_.history_count_ = std::min(state_.history_count_ + 1, state_.history_.size());
        state_.segment_tokens_ = tokens[i] == c.eos_token_id ? 0 :
            std::min(state_.segment_tokens_ + 1, state_.history_.size());
        ++state_.consumed_tokens_;
        if (!prefixes.empty()) save(prefixes[i + 1]);
    }
}

void ple_lookup_f32(const PleHashConfig& c, std::span<const float> table,
                    std::span<const std::uint64_t> rows, std::span<float> output) {
    c.validate(); rows_valid(c, rows);
    if (table.size() != mul(narrow(c.table_rows), c.head_dim) || output.size() != c.embedding_elements())
        invalid("PLE: FP32 table span size");
    disjoint(range(output), range(table)); disjoint(range(output), range(rows)); config_disjoint(range(output), c);
    for (auto row : rows) finite(table.subspan(mul(narrow(row), c.head_dim), c.head_dim));
    for (std::size_t h = 0; h < rows.size(); ++h)
        std::copy_n(table.begin() + mul(narrow(rows[h]), c.head_dim), c.head_dim, output.begin() + h * c.head_dim);
}
PleLookup::PleLookup(const Model& m, PleHashConfig c)
    : model_(m), config_(std::move(c)), table_(m.tensor("per_layer_token_embd.weight")) {
    config_.validate(); row_.resize(table_geometry(table_, config_));
    scratch_.resize(config_.embedding_elements());
}
void PleLookup::lookup(std::span<const std::uint64_t> rows, std::span<float> output) {
    rows_valid(config_, rows);
    if (output.size() != scratch_.size()) invalid("PLE: lookup output size");
    disjoint(range(output), range(rows)); config_disjoint(range(output), config_);
    disjoint(range(output), range(row_)); disjoint(range(output), range(scratch_));
    for (std::size_t h = 0; h < rows.size(); ++h) {
        // Geometry checked at construction and row IDs checked above. Exact one
        // canonical row read (target 5 Q4_0 blocks = 90 bytes), never full table.
        model_.read_slice(table_.name, mul(narrow(rows[h]), row_.size()), row_);
        for (std::size_t d = 0; d < config_.head_dim; ++d)
            scratch_[h * config_.head_dim + d] = checked(tensor_element(table_.type, row_, d));
    }
    std::copy(scratch_.begin(), scratch_.end(), output.begin());
}

std::size_t PleLayerConfig::wide_elements() const { return mul(hidden_size, hc_count); }
std::size_t PleLayerConfig::history_length() const {
    if (!conv_kernel) invalid("PLE: zero convolution kernel");
    return mul(conv_kernel - 1, ngram_size);
}
std::size_t PleLayerConfig::history_elements() const { return mul(wide_elements(), history_length()); }
void PleLayerConfig::validate() const {
    if (!embedding_dim || !hidden_size || !hc_count || !conv_kernel || ngram_size < 2 ||
        !std::isfinite(rms_epsilon) || rms_epsilon <= 0 ||
        embedding_dim > std::size_t(INT_MAX) || wide_elements() > std::size_t(INT_MAX) ||
        conv_kernel > std::size_t(INT_MAX)) invalid("PLE: layer geometry/epsilon");
    mul(history_elements(), sizeof(float)); mul(wide_elements(), sizeof(float));
}
PleLayerConfig ple_layer_config(const Model& m, const PleHashConfig& hash) {
    hash.validate();
    if (!(hash == ple_hash_config(m, hash.layer_index))) invalid("PLE: hash/model metadata mismatch");
    PleLayerConfig c;
    c.embedding_dim = hash.embedding_elements(); c.ngram_size = hash.ngram_size;
    c.hidden_size = narrow(meta(m, "qwen4exp.embedding_length"));
    c.hc_count = narrow(meta(m, "qwen4exp.hyper_connection.count"));
    c.conv_kernel = narrow(meta(m, "qwen4exp.ple.conv_kernel"));
    const auto* eps = m.find_metadata("qwen4exp.attention.layer_norm_rms_epsilon");
    if (!eps || eps->type != MetadataType::FLOAT32) invalid("PLE: FP32 RMS epsilon required");
    c.rms_epsilon = eps->get<float>(); c.validate();
    const auto base = "blk." + std::to_string(hash.layer_index) + ".ple_";
    tensor_shape(m, base + "key.weight", c.embedding_dim, c.wide_elements());
    tensor_shape(m, base + "value.weight", c.embedding_dim, c.hidden_size);
    tensor_shape(m, base + "conv1d.weight", c.conv_kernel, c.wide_elements());
    for (const auto* name : {"norm_key.weight", "norm_query.weight", "norm_conv.weight"})
        tensor_shape(m, base + name, c.wide_elements());
    return c;
}
void ple_group_rms_norm(std::span<const float> x, std::span<const float> gamma,
                        std::size_t group, float eps, std::span<float> out) {
    if (!group || x.empty() || x.size() % group || gamma.size() != x.size() || out.size() != x.size() ||
        !std::isfinite(eps) || eps <= 0) invalid("PLE: RMS geometry/epsilon");
    disjoint(range(out), range(x)); disjoint(range(out), range(gamma)); finite(x); finite(gamma);
    for (int pass = 0; pass < 2; ++pass)
        for (std::size_t g = 0; g < x.size(); g += group) {
            const float scale = norm_scale(x.subspan(g, group), eps);
            for (std::size_t d = g; d < g + group; ++d) {
                const float v = checked((x[d] * scale) * gamma[d]);
                if (pass) out[d] = v;
            }
        }
}
void ple_gate_values(std::span<const float> k, std::span<const float> q,
                     std::span<const float> v, std::span<float> out) {
    if (v.empty() || k.empty() || k.size() % v.size() || q.size() != k.size() || out.size() != k.size())
        invalid("PLE: gate geometry");
    disjoint(range(out), range(k)); disjoint(range(out), range(q)); disjoint(range(out), range(v));
    finite(k); finite(q); finite(v);
    for (int pass = 0; pass < 2; ++pass)
        for (std::size_t g = 0; g < k.size(); g += v.size()) {
            const float scale = gate_scale(k.subspan(g, v.size()), q.subspan(g, v.size()));
            for (std::size_t d = 0; d < v.size(); ++d) {
                const float result = checked(scale * v[d]);
                if (pass) out[g + d] = result;
            }
        }
}
PleLayerCheckpoint::PleLayerCheckpoint(PleLayerConfig c) : config_(c) {
    c.validate(); history_.resize(c.history_elements(), 0.0f);
}
PleLayerCpu::PleLayerCpu(PleLayerConfig c, PleParameters p) : state_(c), parameters_(p) {
    matrix_valid(p.key, c.embedding_dim, c.wide_elements());
    matrix_valid(p.value, c.embedding_dim, c.hidden_size);
    matrix_valid(p.conv, c.conv_kernel, c.wide_elements());
    for (auto n : {p.norm_key, p.norm_query, p.norm_conv}) {
        if (n.size() != c.wide_elements()) invalid("PLE: norm parameter size");
        range(n); finite(n);
    }
    key_.resize(c.wide_elements()); query_.resize(c.wide_elements()); value_.resize(c.hidden_size);
    gated_.resize(c.wide_elements()); normalized_.resize(c.wide_elements()); output_.resize(c.wide_elements());
}
void PleLayerCpu::reset() noexcept {
    std::fill(state_.history_.begin(), state_.history_.end(), 0.0f); state_.consumed_tokens_ = 0;
}
void PleLayerCpu::validate_checkpoint(const PleLayerCheckpoint& s) const {
    if (!(s.config_ == state_.config_) || s.history_.size() != state_.history_.size())
        invalid("PLE: convolution checkpoint geometry");
    finite(s.history_);
    const auto hist = s.config_.history_length();
    const auto padding = hist - std::min<std::uint64_t>(hist, s.consumed_tokens_);
    for (std::size_t f = 0; f < s.config_.wide_elements(); ++f)
        for (std::size_t p = 0; p < padding; ++p)
            if (s.history_[f * hist + p] != 0.0f) invalid("PLE: convolution checkpoint padding");
}
void PleLayerCpu::save(PleLayerCheckpoint& s) const {
    validate_checkpoint(s);
    if (&s == &state_) return;
    const auto dest = range(s.history_);
    for (auto n : {parameters_.norm_key, parameters_.norm_query, parameters_.norm_conv}) disjoint(dest, range(n));
    for (const auto& m : {parameters_.key, parameters_.value, parameters_.conv}) disjoint(dest, range(m.weights));
    std::copy(state_.history_.begin(), state_.history_.end(), s.history_.begin());
    s.consumed_tokens_ = state_.consumed_tokens_;
}
void PleLayerCpu::restore(const PleLayerCheckpoint& s) {
    validate_checkpoint(s);
    if (&s == &state_) return;
    std::copy(s.history_.begin(), s.history_.end(), state_.history_.begin());
    state_.consumed_tokens_ = s.consumed_tokens_;
}
void PleLayerCpu::step(std::span<const float> e, std::span<const float> h, std::span<float> out, bool keep) {
    const std::uint8_t mask = keep ? 1 : 0;
    run(e, h, 1, out, {}, std::span<const std::uint8_t>(&mask, 1));
}
void PleLayerCpu::run(std::span<const float> e, std::span<const float> h, std::size_t tokens,
                      std::span<float> out, std::span<PleLayerCheckpoint> prefixes,
                      std::span<const std::uint8_t> mask) {
    const auto& c = state_.config_;
    if (e.size() != mul(tokens, c.embedding_dim) || h.size() != mul(tokens, c.wide_elements()) ||
        out.size() != h.size() || (!mask.empty() && mask.size() != tokens) ||
        (!prefixes.empty() && (tokens == SIZE_MAX || prefixes.size() != tokens + 1)) ||
        tokens > UINT64_MAX - state_.consumed_tokens_) invalid("PLE: layer chunk geometry/count");
    const auto er = range(e), hr = range(h), wr = range(out), mr = range(mask);
    for (auto r : {er, hr, wr, mr}) disjoint(r, range(&state_, sizeof(state_)));
    disjoint(wr, er); disjoint(wr, hr); disjoint(wr, mr); disjoint(wr, range(prefixes));
    disjoint(er, range(prefixes)); disjoint(hr, range(prefixes)); disjoint(mr, range(prefixes));
    for (const auto& b : {std::span<const float>(state_.history_), std::span<const float>(key_),
                        std::span<const float>(query_), std::span<const float>(value_),
                        std::span<const float>(gated_), std::span<const float>(normalized_),
                        std::span<const float>(output_)}) {
        for (auto r : {er, hr, wr, mr}) disjoint(r, range(b));
    }
    for (auto b : {parameters_.norm_key, parameters_.norm_query, parameters_.norm_conv}) {
        disjoint(wr, range(b));
        for (const auto& s : prefixes) disjoint(range(s.history_), range(b));
    }
    for (const auto& m : {parameters_.key, parameters_.value, parameters_.conv}) {
        disjoint(wr, range(m.weights));
        for (const auto& s : prefixes) disjoint(range(s.history_), range(m.weights));
    }
    finite(e); finite(h);
    for (auto keep : mask) if (keep > 1) invalid("PLE: mask must be 0/1");
    for (auto& s : prefixes) {
        validate_checkpoint(s);
        if (&s == &state_) invalid("PLE: prefix aliases state");
        for (auto r : {er, hr, wr, mr}) disjoint(r, range(s.history_));
    }
    if (!prefixes.empty()) save(prefixes[0]);
    for (std::size_t t = 0; t < tokens; ++t) {
        step_unchecked(e.subspan(t * c.embedding_dim, c.embedding_dim),
                       h.subspan(t * c.wide_elements(), c.wide_elements()),
                       out.subspan(t * c.wide_elements(), c.wide_elements()), mask.empty() || mask[t]);
        if (!prefixes.empty()) save(prefixes[t + 1]);
    }
}
void PleLayerCpu::step_unchecked(std::span<const float> e, std::span<const float> h,
                                 std::span<float> out, bool keep) {
    const auto& c = state_.config_;
    matmul_f32(parameters_.key, e, 1, key_); matmul_f32(parameters_.value, e, 1, value_);
    ple_group_rms_norm(key_, parameters_.norm_key, c.hidden_size, c.rms_epsilon, query_);
    std::copy(query_.begin(), query_.end(), key_.begin());
    ple_group_rms_norm(h, parameters_.norm_query, c.hidden_size, c.rms_epsilon, query_);
    ple_gate_values(key_, query_, value_, gated_);
    ple_group_rms_norm(gated_, parameters_.norm_conv, c.hidden_size, c.rms_epsilon, normalized_);
    if (!keep) { std::fill(gated_.begin(), gated_.end(), 0.0f); std::fill(normalized_.begin(), normalized_.end(), 0.0f); }
    const auto hist = c.history_length();
    for (std::size_t f = 0; f < c.wide_elements(); ++f) {
        float conv = 0.0f;
        for (std::size_t k = 0; k < c.conv_kernel; ++k) {
            const auto lag = (c.conv_kernel - 1 - k) * c.ngram_size;
            const float x = lag ? state_.history_[f * hist + hist - lag] : normalized_[f];
            conv += x * tensor_element(parameters_.conv.type, parameters_.conv.weights, f * c.conv_kernel + k);
        }
        checked(conv);
        output_[f] = checked(gated_[f] + conv * sigmoid(conv));
    }
    // Commit only after every feature has passed. EOS is a hash-only boundary.
    if (hist) for (std::size_t f = 0; f < c.wide_elements(); ++f) {
        auto first = state_.history_.begin() + f * hist;
        std::move(first + 1, first + hist, first); *(first + hist - 1) = normalized_[f];
    }
    ++state_.consumed_tokens_;
    std::copy(output_.begin(), output_.end(), out.begin());
}

} // namespace qwen
