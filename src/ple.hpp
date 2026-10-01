#pragma once

#include "dense.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace qwen {

// Metadata, not a special two-head keep1 architecture. Heads are ordered by
// n-gram order 2..ngram_size, then head within that order. Multipliers are exact
// int64 bit patterns packed as UINT64 in GGUF, never converted through float.
struct PleHashConfig {
    std::size_t ngram_size = 3;
    std::size_t heads_per_ngram = 8;
    std::size_t head_dim = 160;
    std::uint64_t table_rows = 0;
    std::uint64_t token_vocab_size = 0;
    std::int64_t eos_token_id = 248044;
    std::uint32_t layer_index = 1; // zero-based decoder block, NOT PLE ordinal
    std::vector<std::uint64_t> multipliers;
    std::vector<std::uint64_t> vocab_sizes;
    std::vector<std::uint64_t> offsets;

    void validate() const;
    std::size_t heads() const;
    std::size_t embedding_elements() const;
    bool operator==(const PleHashConfig&) const = default;
};

// HF ple_layer_ids=[2] -> GGUF ple.layers=[1]. Reject zero/out-of-range HF ids.
std::uint32_t ple_layer_from_hf(std::uint32_t one_based, std::uint32_t blocks);
// Selected GGUF ABI carries one PLE layer / one multiplier vector. Reject
// multiple layers rather than silently applying the wrong PLE ordinal's hash.
PleHashConfig ple_hash_config(const Model& model, std::uint32_t zero_based_layer);

// Pure single-token hash. History is chronological, at most ngram_size-1 RAW
// consumed token IDs. Missing predecessors and everything at/before a previous
// EOS hash as EOS. A current EOS still uses its PRE-EOS predecessors.
// Products wrap modulo 2^64, XOR keeps those bits; interpret the result as signed
// int64 and use nonnegative torch.remainder, then add the head's offset.
// Negative/out-of-vocabulary token IDs are invalid. Output is exactly heads().
// Invalid geometry/IDs/alias throw invalid_argument before any output writes.
void ple_head_ids(const PleHashConfig& config, std::int64_t token,
                  std::span<const std::int64_t> history,
                  std::span<std::uint64_t> output);

class PleHashCheckpoint {
public:
    explicit PleHashCheckpoint(PleHashConfig config);
    const PleHashConfig& config() const noexcept { return config_; }
    // Fixed-size, EOS-padded oldest-first raw history, including consumed EOS.
    std::span<const std::int64_t> history() const noexcept { return history_; }
    std::size_t history_count() const noexcept { return history_count_; }
    // Capped at ngram_size-1; zero immediately after consuming EOS.
    std::size_t segment_tokens() const noexcept { return segment_tokens_; }
    std::uint64_t consumed_tokens() const noexcept { return consumed_tokens_; }
private:
    friend class PleHashCpu;
    PleHashConfig config_;
    std::vector<std::int64_t> history_;
    std::size_t history_count_ = 0, segment_tokens_ = 0;
    std::uint64_t consumed_tokens_ = 0;
};

class PleHashCpu {
public:
    explicit PleHashCpu(PleHashConfig config);
    PleHashCpu(const PleHashCpu&) = delete;
    PleHashCpu& operator=(const PleHashCpu&) = delete;
    const PleHashCheckpoint& state() const noexcept { return state_; }
    void reset() noexcept;
    void save(PleHashCheckpoint& checkpoint) const;
    void restore(const PleHashCheckpoint& checkpoint);
    void step(std::int64_t consumed_token, std::span<std::uint64_t> rows);
    // Token-major [tokens][heads]. Optional preallocated prefixes: exactly N+1,
    // slot 0 is pre-run; slot n is after n CONSUMED inputs (verify3 uses 1+a).
    // Empty chunks allowed. Entire argument validation is atomic, no allocation.
    void run(std::span<const std::int64_t> consumed_tokens,
             std::span<std::uint64_t> rows,
             std::span<PleHashCheckpoint> prefixes = {});
private:
    void validate_checkpoint(const PleHashCheckpoint& checkpoint) const;
    PleHashCheckpoint state_;
};

// Gather one token's logical heads. Caller-owned small FP32 tables are useful
// fixtures; the file-backed path below never reads/duplicates the whole table.
// Rows must belong to the respective head's range, not merely to the table.
void ple_lookup_f32(const PleHashConfig& config, std::span<const float> table,
                    std::span<const std::uint64_t> rows, std::span<float> output);

class PleLookup {
public:
    // Model must outlive this lookup. Validates rank 2, canonical byte strides,
    // dimensions and exact size; allocates one row and one embedding scratch.
    PleLookup(const Model& model, PleHashConfig config);
    void lookup(std::span<const std::uint64_t> rows, std::span<float> output);
private:
    const Model& model_;
    PleHashConfig config_;
    const TensorView& table_;
    std::vector<std::byte> row_;
    std::vector<float> scratch_;
};

struct PleLayerConfig {
    std::size_t embedding_dim = 2560, hidden_size = 2560, hc_count = 4;
    std::size_t conv_kernel = 4, ngram_size = 3;
    float rms_epsilon = 1.0e-6f;
    void validate() const;
    std::size_t wide_elements() const;
    std::size_t history_length() const; // (conv_kernel-1)*ngram_size, target 9
    std::size_t history_elements() const;
    bool operator==(const PleLayerConfig&) const = default;
};
PleLayerConfig ple_layer_config(const Model& model, const PleHashConfig& hash);

struct PleParameters {
    QMatrix key;   // [embedding_dim, hc_count*hidden_size], canonical GGML rows
    QMatrix value; // [embedding_dim, hidden_size], one shared value per token
    QMatrix conv;  // [conv_kernel, wide_elements], oldest tap first (target F16)
    // GGUF converter has ALREADY folded HF zero-centred weights to (1+weight).
    std::span<const float> norm_key, norm_query, norm_conv; // each [wide]
};

// Original scalar FP32 helpers, ascending reductions, -ffp-contract=off.
// Direct GGUF gamma. Output must be disjoint; failures leave it unchanged.
void ple_group_rms_norm(std::span<const float> input, std::span<const float> gamma,
                        std::size_t group_size, float epsilon, std::span<float> output);
// s_c = dot(key_c,query_c)/sqrt(hidden_size), g=sign(s)*sqrt(max(abs(s),1e-6)),
// result[c,d]=sigmoid(g_c)*shared_value[d]. Zero s -> sigmoid(0), not sqrt(eps).
void ple_gate_values(std::span<const float> key, std::span<const float> query,
                     std::span<const float> shared_value, std::span<float> output);

class PleLayerCheckpoint {
public:
    explicit PleLayerCheckpoint(PleLayerConfig config = {});
    const PleLayerConfig& config() const noexcept { return config_; }
    // [wide_feature][history_length], oldest -> newest NORMALIZED gated values.
    // EOS does NOT clear this convolution history. reset() clears it to +0.
    std::span<const float> conv_history() const noexcept { return history_; }
    std::uint64_t consumed_tokens() const noexcept { return consumed_tokens_; }
private:
    friend class PleLayerCpu;
    PleLayerConfig config_;
    std::vector<float> history_;
    std::uint64_t consumed_tokens_ = 0;
};

class PleLayerCpu {
public:
    // Borrow immutable weights/norms (must outlive the instance); own state and
    // scratch. Allocates only at construction. No threads/allocations on success.
    PleLayerCpu(PleLayerConfig config, PleParameters parameters);
    PleLayerCpu(const PleLayerCpu&) = delete;
    PleLayerCpu& operator=(const PleLayerCpu&) = delete;
    const PleLayerCheckpoint& state() const noexcept { return state_; }
    void reset() noexcept;
    void save(PleLayerCheckpoint& checkpoint) const;
    void restore(const PleLayerCheckpoint& checkpoint);
    // Returns PLE INJECTION [hc_count*hidden_size], not the merged residual.
    // Caller adds it once to the widened residual BEFORE the attention HC mix.
    // keep=false zeros current gated+normed values, but prior conv taps survive
    // (HF conv_mask); caller supplies EOS to the separate hash for masked tokens.
    void step(std::span<const float> embedding, std::span<const float> hidden,
              std::span<float> injection, bool keep = true);
    // Token-major inputs/output. Optional keep mask has exactly tokens entries,
    // each 0/1. Prefix convention matches PleHashCpu. Preflight invalid sizes,
    // aliases/nonfinites/count overflow before mutation. Arithmetic overflow in
    // a token throws: failing token's state/output unchanged, earlier tokens in
    // the chunk remain consumed. All hash/lookup/layer states are separate;
    // integration saves/restores matching prefixes together, including EOS state.
    void run(std::span<const float> embeddings, std::span<const float> hidden,
             std::size_t tokens, std::span<float> injection,
             std::span<PleLayerCheckpoint> prefixes = {},
             std::span<const std::uint8_t> keep_mask = {});
private:
    void validate_checkpoint(const PleLayerCheckpoint& checkpoint) const;
    void step_unchecked(std::span<const float> embedding, std::span<const float> hidden,
                        std::span<float> injection, bool keep);
    PleLayerCheckpoint state_;
    PleParameters parameters_;
    std::vector<float> key_, query_, value_, gated_, normalized_, output_;
};

} // namespace qwen
