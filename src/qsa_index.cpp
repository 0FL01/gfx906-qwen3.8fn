#include "qsa_index.hpp"
#include "kv.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace qwen {
namespace {

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }
[[noreturn]] void numerical() { throw std::runtime_error("qsa index: nonfinite FP32 arithmetic"); }

bool overlap(const void* a, std::size_t as, const void* b, std::size_t bs) {
    const auto* x = static_cast<const std::byte*>(a);
    const auto* y = static_cast<const std::byte*>(b);
    const std::less<const std::byte*> before;
    return as != 0 && bs != 0 && before(x, y + bs) && before(y, x + as);
}

void finite(std::span<const float> values) {
    for (const auto value : values)
        if (!std::isfinite(value)) invalid("qsa index: nonfinite input/parameter");
}

std::uint64_t new_owner() {
    // Construction only. Do not recycle addresses as identity after destruction.
    static std::atomic<std::uint64_t> owners{0};
    auto previous = owners.load(std::memory_order_relaxed);
    do {
        if (previous == std::numeric_limits<std::uint64_t>::max())
            invalid("qsa index: owner counter exhausted");
    } while (!owners.compare_exchange_weak(previous, previous + 1, std::memory_order_relaxed));
    return previous + 1;
}

void validate_config(QsaIndexConfig c, QsaIndexParameters p) {
    if (c.head_dim != 128 || c.query_heads != 4 || c.key_heads != 1 || c.compress_ratio != 4)
        invalid("qsa index: expected target D128/Q4/K1/compress4");
    if (c.capacity == 0 || c.capacity > 131072 || c.rotary_dim == 0 ||
        c.rotary_dim > 128 || c.rotary_dim % 2 != 0)
        invalid("qsa index: invalid capacity/rotary dimension");
    if (!std::isfinite(c.rms_epsilon) || c.rms_epsilon <= 0.0f ||
        !std::isfinite(c.rope_attention_scale) || c.rope_attention_scale <= 0.0f)
        invalid("qsa index: invalid epsilon/RoPE attention scale");
    if (p.query_norm_weight.size() != 128 || p.key_norm_weight.size() != 128 ||
        p.inverse_frequencies.size() != c.rotary_dim / 2)
        invalid("qsa index: wrong norm/frequency dimensions");
    finite(p.query_norm_weight);
    finite(p.key_norm_weight);
    finite(p.inverse_frequencies);
    for (float f : p.inverse_frequencies)
        if (f <= 0.0f) invalid("qsa index: inverse frequencies must be positive");
}

} // namespace

// Source semantics (no upstream code copied): HF Qwen4ExpTextQSAIndexer.forward,
// apply_rotary_pos_emb, inherited Qwen3_5RMSNorm/TextRotaryEmbedding/rotate_half,
// transformers a005fc82babfe8871d87746decad2dbee100a125 (Qwen/HF, Apache-2.0).
// mx dcd685463d597d31f5ca759d32c94592a2740fa4:
// src/models/qwen4exp.cpp::build_qsa_top_k: raw K cache -> mean -> norm -> RoPE;
// src/llama-memory-hybrid-idx.cpp::set_input_qsa: blk_pos = b*ratio;
// conversion/qwen4exp.py::modify_tensors: indexer norm gamma += 1 at conversion.
// Difference: HF raw K is not Q4; this oracle intentionally includes the GGUF
// operational Q4 roundtrip. FP32 stays FP32 (no HF activation-dtype recast).
// mx's score omits 1/sqrt(D), a positive ranking-invariant factor; we retain the
// reference numeric score. mx expanded-position selection is NOT our semantics:
// qsa_select implements HF whole-block/actual-tail with lower-ID tie policy.

QsaIndexCheckpoint::QsaIndexCheckpoint() : tail_(3 * 128) {}

std::span<const float> QsaIndexCheckpoint::raw_tail() const noexcept {
    // A moved-from checkpoint is inspectable but not save/restore compatible.
    return std::span<const float>(tail_).first(std::min(tail_.size(), (consumed_ % 4) * 128));
}

QsaIndexCpu::QsaIndexCpu(QsaIndexConfig config, QsaIndexParameters parameters) : config_(config) {
    validate_config(config, parameters);
    query_weight_.assign(parameters.query_norm_weight.begin(), parameters.query_norm_weight.end());
    key_weight_.assign(parameters.key_norm_weight.begin(), parameters.key_norm_weight.end());
    inverse_frequencies_.assign(parameters.inverse_frequencies.begin(), parameters.inverse_frequencies.end());
    keys_.resize((config.capacity / 4) * 128);
    score_scratch_.resize(config.capacity / 4);
    block_tags_.resize(config.capacity / 4);
    owner_ = new_owner();
}

std::span<const float> QsaIndexCpu::completed_keys() const noexcept {
    return std::span<const float>(keys_).first(completed_blocks() * 128);
}
std::span<const float> QsaIndexCpu::raw_tail() const noexcept {
    return std::span<const float>(tail_).first((consumed_ % 4) * 128);
}

bool QsaIndexCpu::aliases_owned(const void* data, std::size_t bytes) const {
    const auto aliases = [&](const auto& values) {
        return overlap(data, bytes, values.data(), values.size() * sizeof(values[0]));
    };
    return overlap(data, bytes, this, sizeof(*this)) || aliases(query_weight_) || aliases(key_weight_) ||
           aliases(inverse_frequencies_) || aliases(keys_) || aliases(score_scratch_) || aliases(block_tags_);
}

void QsaIndexCpu::validate_destination(const QsaIndexCheckpoint& checkpoint) const {
    if (checkpoint.tail_.size() != 3 * 128 ||
        aliases_owned(&checkpoint, sizeof(checkpoint)) ||
        aliases_owned(checkpoint.tail_.data(), checkpoint.tail_.size() * sizeof(float)))
        invalid("qsa index: invalid/aliased checkpoint storage");
}

void QsaIndexCpu::reset() {
    if (epoch_ == std::numeric_limits<std::uint64_t>::max()) invalid("qsa index: epoch overflow");
    ++epoch_;
    consumed_ = 0;
    retained_blocks_ = 0;
    tail_.fill(0.0f);
}

void QsaIndexCpu::save(QsaIndexCheckpoint& checkpoint) const {
    validate_destination(checkpoint);
    checkpoint.consumed_ = consumed_;
    checkpoint.owner_ = owner_;
    checkpoint.epoch_ = epoch_;
    checkpoint.prefix_tag_ = completed_blocks() == 0 ? 0 : block_tags_[completed_blocks() - 1];
    std::copy(tail_.begin(), tail_.end(), checkpoint.tail_.begin());
}

void QsaIndexCpu::restore(const QsaIndexCheckpoint& checkpoint) {
    validate_destination(checkpoint);
    const auto blocks = checkpoint.consumed_ / 4;
    if (checkpoint.owner_ != owner_ || checkpoint.epoch_ != epoch_ ||
        checkpoint.consumed_ > config_.capacity || blocks > retained_blocks_ ||
        checkpoint.prefix_tag_ != (blocks == 0 ? 0 : block_tags_[blocks - 1]))
        invalid("qsa index: cross-owner/reset/overwritten checkpoint prefix");
    finite(checkpoint.raw_tail());
    consumed_ = checkpoint.consumed_;
    std::copy(checkpoint.tail_.begin(), checkpoint.tail_.end(), tail_.begin());
}

void QsaIndexCpu::norm_rope(std::span<float> values, std::span<const float> weight,
                            std::size_t position) const {
    float squares = 0.0f;
    for (float x : values) squares = squares + x * x;
    const float variance = squares / 128.0f + config_.rms_epsilon;
    if (!std::isfinite(variance)) numerical();
    const float scale = 1.0f / std::sqrt(variance);
    for (std::size_t i = 0; i < 128; ++i) {
        values[i] = (values[i] * scale) * weight[i];
        if (!std::isfinite(values[i])) numerical();
    }
    const auto half = config_.rotary_dim / 2;
    for (std::size_t i = 0; i < half; ++i) {
        const float angle = inverse_frequencies_[i] * static_cast<float>(position);
        if (!std::isfinite(angle)) numerical();
        const float cosine = std::cos(angle) * config_.rope_attention_scale;
        const float sine = std::sin(angle) * config_.rope_attention_scale;
        const float a = values[i], b = values[i + half];
        values[i] = a * cosine + (-b) * sine;
        values[i + half] = b * cosine + a * sine;
        if (!std::isfinite(values[i]) || !std::isfinite(values[i + half])) numerical();
    }
}

void QsaIndexCpu::append_one(std::span<const float> raw_key) {
    std::array<Q4_0, 4> packed{};
    quantize_q4(raw_key, packed);
    dequantize_q4(packed, raw_scratch_);
    const auto remainder = consumed_ % 4;
    if (remainder == 3) {
        for (std::size_t d = 0; d < 128; ++d) {
            float sum = tail_[d];
            sum = sum + tail_[128 + d];
            sum = sum + tail_[256 + d];
            sum = sum + raw_scratch_[d];
            pooled_scratch_[d] = sum * 0.25f;
        }
        norm_rope(pooled_scratch_, key_weight_, consumed_ - 3);
        const auto block = completed_blocks();
        // No semantic mutation until the candidate completed key is finite.
        std::copy(pooled_scratch_.begin(), pooled_scratch_.end(), keys_.begin() + block * 128);
        block_tags_[block] = ++next_tag_;
        retained_blocks_ = block + 1; // writing a branch invalidates its old suffix
        tail_.fill(0.0f);
    } else {
        std::copy(raw_scratch_.begin(), raw_scratch_.end(), tail_.begin() + remainder * 128);
    }
    ++consumed_;
}

void QsaIndexCpu::append(std::span<const float> raw_keys, std::size_t tokens,
                         std::span<QsaIndexCheckpoint> prefixes) {
    // Bound tokens BEFORE any products, tokens+1 or borrowed-span dereference.
    if (tokens > config_.capacity - consumed_ || raw_keys.size() != tokens * 128 ||
        (!prefixes.empty() && prefixes.size() != tokens + 1))
        invalid("qsa index: chunk size/capacity/prefix count");
    const auto completions = (consumed_ + tokens) / 4 - completed_blocks();
    if (completions > std::numeric_limits<std::uint64_t>::max() - next_tag_)
        invalid("qsa index: prefix tag overflow");
    if (aliases_owned(raw_keys.data(), raw_keys.size_bytes()) ||
        overlap(raw_keys.data(), raw_keys.size_bytes(), prefixes.data(), prefixes.size_bytes()))
        invalid("qsa index: input aliases state/prefixes");
    for (const auto& prefix : prefixes) {
        validate_destination(prefix);
        if (overlap(raw_keys.data(), raw_keys.size_bytes(), prefix.tail_.data(), prefix.tail_.size() * sizeof(float)))
            invalid("qsa index: input aliases checkpoint tail");
    }
    // kv quantize checks all four blocks before writes, including finite/scales.
    std::array<Q4_0, 4> packed{};
    for (std::size_t t = 0; t < tokens; ++t) quantize_q4(raw_keys.subspan(t * 128, 128), packed);
    if (!prefixes.empty()) save(prefixes[0]);
    for (std::size_t t = 0; t < tokens; ++t) {
        append_one(raw_keys.subspan(t * 128, 128));
        if (!prefixes.empty()) save(prefixes[t + 1]);
    }
}

void QsaIndexCpu::score(std::span<const float> raw_query, std::span<float> scores) {
    if (consumed_ == 0 || raw_query.size() != 4 * 128 || scores.size() != completed_blocks())
        invalid("qsa index: query/score shape or empty visibility");
    if (aliases_owned(raw_query.data(), raw_query.size_bytes()) ||
        aliases_owned(scores.data(), scores.size_bytes()) ||
        overlap(raw_query.data(), raw_query.size_bytes(), scores.data(), scores.size_bytes()))
        invalid("qsa index: query/score aliases state or each other");
    finite(raw_query);
    std::copy(raw_query.begin(), raw_query.end(), query_scratch_.begin());
    for (std::size_t h = 0; h < 4; ++h)
        norm_rope(std::span<float>(query_scratch_).subspan(h * 128, 128), query_weight_, consumed_ - 1);
    const float divisor = std::sqrt(128.0f);
    for (std::size_t block = 0; block < completed_blocks(); ++block) {
        float sum = 0.0f;
        for (std::size_t h = 0; h < 4; ++h) {
            float dot = 0.0f;
            for (std::size_t d = 0; d < 128; ++d)
                dot = dot + query_scratch_[h * 128 + d] * keys_[block * 128 + d];
            if (!std::isfinite(dot)) numerical();
            sum = sum + std::max(dot, 0.0f);
        }
        score_scratch_[block] = sum / divisor;
        if (!std::isfinite(score_scratch_[block])) numerical();
    }
    std::copy_n(score_scratch_.begin(), completed_blocks(), scores.begin());
}

} // namespace qwen
