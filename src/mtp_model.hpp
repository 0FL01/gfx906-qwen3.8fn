#pragma once

#include "model.hpp"

namespace qwen {

// Fixed trained blk.48 sidecar geometry. GGUF axes remain fastest-first; no
// payload is copied, dequantized, or repacked by this descriptor.
struct MtpGeometry {
    std::uint32_t layer;
    std::uint32_t block_count;
    std::uint32_t predict_layers;
    std::uint32_t hidden_size;
    std::uint32_t hc_count;
    std::uint32_t hc_rank;
    std::uint32_t widened_size;
    std::uint32_t query_heads;
    std::uint32_t kv_heads;
    std::uint32_t head_dim;
    std::uint32_t experts;
    std::uint32_t intermediate_size;
    std::uint32_t vocabulary_size;
    std::uint32_t rotary_dim;
    float rope_base;
    float rms_epsilon;
    std::array<std::int32_t, 4> rope_sections;

    // Checked against typed metadata as well as tensor geometry. blk.48 is
    // dense-only despite its indexer tensors. Native context metadata is 262144;
    // it does not raise the Session runtime capacity limit of 131072.
    static constexpr std::uint32_t required_experts_used = 10;
    static constexpr std::uint32_t required_compression_ratio = 0;
    static constexpr TensorType kv_type = TensorType::Q4_0;
};

struct MtpTypeCounts {
    std::size_t q8_0 = 0;
    std::size_t f32 = 0;
    std::size_t bf16 = 0;
};

struct MtpHeadWeights {
    const TensorView& norm;
    const TensorView& down;
    const TensorView& up;
};

struct MtpResidualWeights {
    MtpHeadWeights mixer;
    const TensorView& inject;
};

struct MtpAttentionWeights {
    const TensorView& query;
    const TensorView& key;
    const TensorView& value;
    const TensorView& output;
    const TensorView& query_norm;
    const TensorView& key_norm;
};

struct MtpExpertWeights {
    const TensorView& router;
    const TensorView& shared_router;
    const TensorView& gate;
    const TensorView& up;
    const TensorView& down;
    const TensorView& shared_gate;
    const TensorView& shared_up;
    const TensorView& shared_down;
};

struct MtpIndexerWeights {
    const TensorView& query;
    const TensorView& key;
    const TensorView& query_norm;
    const TensorView& key_norm;
};

struct MtpWeights {
    const TensorView& embedding_norm;
    const TensorView& hidden_norm;
    const TensorView& embedding_hidden_projection;
    MtpHeadWeights head; // nextn.hc_head_*, NEVER the target root HC mixer
    MtpResidualWeights attention_hc;
    MtpResidualWeights ffn_hc;
    MtpAttentionWeights attention;
    MtpExpertWeights experts;
    MtpIndexerWeights indexer;
};

struct MtpSharedTargetWeights {
    const TensorView& embedding; // target token_embd.weight, Q4_0
    const TensorView& output;    // DISTINCT target output.weight, Q6_K
};

// Mandatory construction conditions, all enforced before views are exposed.
// These flags describe loader validation, not outstanding parent-side checks or
// evidence of trained forward/teacher-history/rollback correctness.
struct MtpValidationRequirements {
    bool per_role_type_inventory = true;
    bool dense_compression_metadata = true;
    bool attention_expert_metadata = true;
    bool target_tokenizer_equivalence = true;
};

// Owns the sidecar's Model parser/file descriptor and parsed inventory.
// The target and its TWO borrowed views must outlive this object and must not
// be moved from or move-assigned while borrowed. All public views are const.
// Construction validates the complete 32-tensor inventory before exposing it.
// Each role has an exact dtype: BF16 only for indexer Q/K projections, F32 for
// norms/router weights, and Q8_0 for all other projections and expert weights.
// Aggregate counts are 11/2/19 F32/BF16/Q8_0. Common target geometry and every
// typed tokenizer.* key/value (including optional-key presence) must agree.
// This is a loader prerequisite, not trained MTP execution/restore proof.
//
// Production convention (not HF training proof): fresh history row 0 uses zero
// hidden; teacher row p (p > 0) consumes widened target tap T[p-1] and token x[p]
// at position p.
// Recursive proposal 2 consumes widened D[N], not normalized head input.
// Recursive KV must be discarded and teacher-rebuilt before prefix truncation.
// No history/KV, forward, sampling, or speculative state is implemented here.
class MtpModel {
public:
    MtpModel(const std::filesystem::path& sidecar_path, const Model& target);
    MtpModel(const std::filesystem::path&, Model&&) = delete;
    MtpModel(const std::filesystem::path&, const Model&&) = delete;
    MtpModel(const MtpModel&) = delete;
    MtpModel& operator=(const MtpModel&) = delete;
    MtpModel(MtpModel&&) = delete;
    MtpModel& operator=(MtpModel&&) = delete;

    [[nodiscard]] const Model& sidecar() const noexcept { return sidecar_; }
    [[nodiscard]] const Model& target() const noexcept { return target_; }
    [[nodiscard]] const MtpGeometry& geometry() const noexcept { return geometry_; }
    [[nodiscard]] const MtpWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] const MtpSharedTargetWeights& shared_target() const noexcept { return shared_; }
    [[nodiscard]] const MtpTypeCounts& type_counts() const noexcept { return type_counts_; }
    [[nodiscard]] MtpValidationRequirements validation_requirements() const noexcept { return {}; }

private:
    Model sidecar_;
    const Model& target_;
    const MtpGeometry geometry_;
    const MtpWeights weights_;
    const MtpSharedTargetWeights shared_;
    const MtpTypeCounts type_counts_;
};

} // namespace qwen
