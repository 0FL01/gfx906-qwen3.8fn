#include "mtp_model.hpp"

#include <algorithm>
#include <bit>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

namespace qwen {
namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("MTP sidecar: " + message);
}

template<class T> const T& metadata(const Model& model, std::string_view key, MetadataType type) {
    const auto* value = model.find_metadata(key);
    if (!value) invalid("missing metadata key: " + std::string(key));
    if (value->type != type) invalid("wrong metadata type: " + std::string(key));
    return value->get<T>();
}

std::uint32_t number(const Model& model, std::string_view key, std::uint32_t expected) {
    const auto value = metadata<std::uint32_t>(model, key, MetadataType::UINT32);
    if (value != expected) invalid("unexpected metadata value: " + std::string(key));
    return value;
}

float floating(const Model& model, std::string_view key, float expected) {
    const auto value = metadata<float>(model, key, MetadataType::FLOAT32);
    if (std::bit_cast<std::uint32_t>(value) != std::bit_cast<std::uint32_t>(expected))
        invalid("unexpected metadata value: " + std::string(key));
    return value;
}

template<class T> const std::vector<T>& array(const Model& model, std::string_view key,
                                             MetadataType element_type) {
    const auto& value = metadata<MetadataArray>(model, key, MetadataType::ARRAY);
    if (value.element_type != element_type)
        invalid("wrong array element type: " + std::string(key));
    return value.get<T>();
}

template<class T> bool same_float_bits(T a, T b) {
    using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
    return std::bit_cast<Bits>(a) == std::bit_cast<Bits>(b);
}

// Model owns the parsed assets. Compare them directly, including floating-point
// bit patterns (+/-0 and NaN payloads), with no hashes, samples or asset copies.
bool same_array(const MetadataArray& a, const MetadataArray& b) {
    if (a.element_type != b.element_type) return false;
    return std::visit([&](const auto& values) {
        using Vector = std::decay_t<decltype(values)>;
        const auto* other = std::get_if<Vector>(&b.values);
        if (!other || values.size() != other->size()) return false;
        if constexpr (std::is_floating_point_v<typename Vector::value_type>) {
            for (std::size_t i = 0; i < values.size(); ++i)
                if (!same_float_bits(values[i], (*other)[i])) return false;
            return true;
        } else return values == *other;
    }, a.values);
}

bool same_metadata(const MetadataValue& a, const MetadataValue& b) {
    if (a.type != b.type) return false;
    return std::visit([&](const auto& value) {
        using T = std::decay_t<decltype(value)>;
        const auto* other = std::get_if<T>(&b.value);
        if (!other) return false;
        if constexpr (std::is_same_v<T, MetadataArray>) return same_array(value, *other);
        else if constexpr (std::is_floating_point_v<T>) return same_float_bits(value, *other);
        else return value == *other;
    }, a.value);
}

void tokenizer_assets(const Model& model, std::uint32_t vocabulary) {
    const auto& tokens = array<std::string>(model, "tokenizer.ggml.tokens", MetadataType::STRING);
    const auto& types = array<std::int32_t>(model, "tokenizer.ggml.token_type", MetadataType::INT32);
    if (tokens.size() != vocabulary || types.size() != vocabulary)
        invalid("tokenizer vocabulary cardinality disagrees with borrowed matrices");
    (void)array<std::string>(model, "tokenizer.ggml.merges", MetadataType::STRING);
    (void)metadata<std::string>(model, "tokenizer.ggml.model", MetadataType::STRING);
    (void)metadata<std::string>(model, "tokenizer.ggml.pre", MetadataType::STRING);
    (void)metadata<std::string>(model, "tokenizer.chat_template", MetadataType::STRING);
    (void)metadata<bool>(model, "tokenizer.ggml.add_bos_token", MetadataType::BOOL);
    for (const auto key : {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                           "tokenizer.ggml.padding_token_id"})
        if (metadata<std::uint32_t>(model, key, MetadataType::UINT32) >= vocabulary)
            invalid("tokenizer special token outside borrowed vocabulary: " + std::string(key));
    // scores/add_eos and future tokenizer assets are not invented or defaulted.
    // Their presence, types and complete values are checked by the map comparison.
}

void tokenizer_equivalence(const Model& sidecar, const Model& target, std::uint32_t vocabulary) {
    tokenizer_assets(sidecar, vocabulary);
    tokenizer_assets(target, vocabulary);
    constexpr std::string_view prefix = "tokenizer.";
    for (const auto* model : {&sidecar, &target}) {
        const auto& other = model == &sidecar ? target : sidecar;
        for (auto it = model->metadata().lower_bound(prefix);
             it != model->metadata().end() && it->first.starts_with(prefix); ++it) {
            const auto* value = other.find_metadata(it->first);
            if (!value) invalid("tokenizer key-set mismatch: " + it->first);
            if (!same_metadata(it->second, *value))
                invalid("tokenizer metadata mismatch: " + it->first);
        }
    }
}

void compression(const Model& model, std::uint32_t blocks, bool draft) {
    constexpr std::string_view key = "qwen4exp.attention.compress_ratios";
    const auto& ratios = array<std::int32_t>(model, key, MetadataType::INT32);
    if (ratios.size() != blocks) invalid("unexpected metadata length: " + std::string(key));
    // Check the complete array's supported values, but not an irrelevant trunk
    // ordering. Only the sidecar's final layer establishes dense MTP attention.
    for (const auto ratio : ratios)
        if (ratio != 0 && ratio != 4)
            invalid("unexpected metadata value: " + std::string(key));
    if (draft && ratios.back() != static_cast<std::int32_t>(MtpGeometry::required_compression_ratio))
        invalid("blk.48 compression ratio must be zero");
}

MtpGeometry common_geometry(const Model& model) {
    MtpGeometry geometry{};
    geometry.hidden_size = number(model, "qwen4exp.embedding_length", 2560);
    geometry.hc_count = number(model, "qwen4exp.hyper_connection.count", 4);
    geometry.hc_rank = number(model, "qwen4exp.hyper_connection.low_rank", 320);
    geometry.widened_size = geometry.hidden_size * geometry.hc_count;
    geometry.query_heads = number(model, "qwen4exp.attention.head_count", 24);
    geometry.kv_heads = number(model, "qwen4exp.attention.head_count_kv", 2);
    geometry.head_dim = number(model, "qwen4exp.attention.key_length", 256);
    (void)number(model, "qwen4exp.attention.value_length", 256);
    geometry.experts = number(model, "qwen4exp.expert_count", 512);
    (void)number(model, "qwen4exp.expert_used_count", MtpGeometry::required_experts_used);
    geometry.intermediate_size = number(model, "qwen4exp.expert_feed_forward_length", 640);
    (void)number(model, "qwen4exp.expert_shared_feed_forward_length", 640);
    (void)number(model, "qwen4exp.context_length", 262144);
    geometry.rotary_dim = number(model, "qwen4exp.rope.dimension_count", 64);
    geometry.rope_base = floating(model, "qwen4exp.rope.freq_base", 1.0e7F);
    geometry.rms_epsilon = floating(model, "qwen4exp.attention.layer_norm_rms_epsilon", 1.0e-6F);
    constexpr std::string_view key = "qwen4exp.rope.dimension_sections";
    const auto& values = array<std::int32_t>(model, key, MetadataType::INT32);
    constexpr std::array<std::int32_t, 4> expected_sections{11, 11, 10, 0};
    if (values.size() != expected_sections.size() ||
        !std::equal(values.begin(), values.end(), expected_sections.begin()))
        invalid("unexpected metadata value: " + std::string(key));
    geometry.rope_sections = expected_sections;
    return geometry;
}

void architecture(const Model& model) {
    if (metadata<std::string>(model, "general.architecture", MetadataType::STRING) != "qwen4exp")
        invalid("expected qwen4exp architecture");
}

const TensorView& tensor(const Model& model, std::string_view name) {
    const auto* view = model.find_tensor(name);
    if (!view) invalid("missing tensor: " + std::string(name));
    return *view;
}

void shape(const TensorView& view, std::initializer_list<std::uint64_t> dimensions) {
    if (view.rank != dimensions.size() ||
        !std::equal(dimensions.begin(), dimensions.end(), view.dimensions.begin()))
        invalid("unexpected tensor shape/rank: " + view.name);
    // Bytes, strides, axis-0 divisibility, checked products and file ranges have
    // already been validated by Model. Accept only its unmodified const views.
}

const TensorView& block(const Model& model, std::string_view suffix) {
    return tensor(model, "blk.48." + std::string(suffix) + ".weight");
}

MtpTypeCounts count_types(const Model& sidecar) {
    MtpTypeCounts counts;
    for (const auto& view : sidecar.tensors()) {
        switch (view.type) {
        case TensorType::Q8_0: ++counts.q8_0; break;
        case TensorType::F32: ++counts.f32; break;
        case TensorType::BF16: ++counts.bf16; break;
        default: invalid("unsupported sidecar tensor type: " + view.name);
        }
    }
    return counts;
}

MtpGeometry validate(const Model& sidecar, const Model& target) {
    architecture(sidecar);
    architecture(target);
    (void)number(target, "qwen4exp.block_count", 48);
    (void)common_geometry(target); // Validate compatibility before borrowing widened taps.
    compression(target, 48, false);
    if (!metadata<bool>(sidecar, "qwen4exp.nextn_shared_target_tensors", MetadataType::BOOL))
        invalid("nextn_shared_target_tensors must be true");

    auto geometry = common_geometry(sidecar);
    geometry.block_count = number(sidecar, "qwen4exp.block_count", 49);
    geometry.predict_layers = number(sidecar, "qwen4exp.nextn_predict_layers", 1);
    geometry.layer = geometry.block_count - geometry.predict_layers;
    compression(sidecar, geometry.block_count, true);

    // Required names plus an exact count reject foreign layers, duplicate roles,
    // embedded copies of target weights, and even otherwise valid extra tensors.
    if (sidecar.tensors().size() != 32) invalid("expected exactly 32 blk.48 tensors");
    const auto counts = count_types(sidecar);
    const auto check = [&](std::string_view suffix, TensorType type,
                           std::initializer_list<std::uint64_t> dims) {
        const auto& view = block(sidecar, suffix);
        shape(view, dims);
        if (view.type != type) invalid("unexpected tensor type for role: " + view.name);
    };
    check("nextn.enorm", TensorType::F32, {2560});
    check("nextn.hnorm", TensorType::F32, {10240});
    check("nextn.eh_proj", TensorType::Q8_0, {5120, 2560});
    for (const auto prefix : {"nextn.hc_head", "hc_attn", "hc_ffn"}) {
        check(std::string(prefix) + "_norm", TensorType::F32, {10240});
        check(std::string(prefix) + "_down", TensorType::Q8_0, {10240, 320});
        check(std::string(prefix) + "_up", TensorType::Q8_0, {320, 10240});
    }
    check("hc_attn_inject", TensorType::Q8_0, {10240, 4});
    check("hc_ffn_inject", TensorType::Q8_0, {10240, 4});
    check("attn_q", TensorType::Q8_0, {2560, 12288}); // interleaved Q + gate, 2*24*256
    check("attn_k", TensorType::Q8_0, {2560, 512});
    check("attn_v", TensorType::Q8_0, {2560, 512});
    check("attn_output", TensorType::Q8_0, {6144, 2560});
    check("attn_q_norm", TensorType::F32, {256});
    check("attn_k_norm", TensorType::F32, {256});
    check("ffn_gate_inp", TensorType::F32, {2560, 512});
    check("ffn_gate_inp_shexp", TensorType::F32, {2560});
    check("ffn_gate_exps", TensorType::Q8_0, {2560, 640, 512});
    check("ffn_up_exps", TensorType::Q8_0, {2560, 640, 512});
    check("ffn_down_exps", TensorType::Q8_0, {640, 2560, 512});
    check("ffn_gate_shexp", TensorType::Q8_0, {2560, 640});
    check("ffn_up_shexp", TensorType::Q8_0, {2560, 640});
    check("ffn_down_shexp", TensorType::Q8_0, {640, 2560});
    check("indexer.q_proj", TensorType::BF16, {2560, 512});
    check("indexer.k_proj", TensorType::BF16, {2560, 128});
    check("indexer.q_norm", TensorType::F32, {128});
    check("indexer.k_norm", TensorType::F32, {128});
    if (counts.q8_0 != 19 || counts.f32 != 11 || counts.bf16 != 2)
        invalid("expected type counts Q8_0=19 F32=11 BF16=2");

    const auto& embedding = tensor(target, "token_embd.weight");
    const auto& output = tensor(target, "output.weight");
    shape(embedding, {2560, 248320});
    shape(output, {2560, 248320});
    if (embedding.type != TensorType::Q4_0) invalid("target embedding must be Q4_0");
    if (output.type != TensorType::Q6_K) invalid("target output must be Q6_K");
    if (&embedding == &output) invalid("target embedding/output must be distinct views");
    geometry.vocabulary_size = static_cast<std::uint32_t>(embedding.dimensions[1]);
    tokenizer_equivalence(sidecar, target, geometry.vocabulary_size);
    return geometry;
}

MtpWeights bind_weights(const Model& sidecar) {
    const auto get = [&](std::string_view suffix) -> const TensorView& { return block(sidecar, suffix); };
    return {
        get("nextn.enorm"), get("nextn.hnorm"), get("nextn.eh_proj"),
        {get("nextn.hc_head_norm"), get("nextn.hc_head_down"), get("nextn.hc_head_up")},
        {{get("hc_attn_norm"), get("hc_attn_down"), get("hc_attn_up")}, get("hc_attn_inject")},
        {{get("hc_ffn_norm"), get("hc_ffn_down"), get("hc_ffn_up")}, get("hc_ffn_inject")},
        {get("attn_q"), get("attn_k"), get("attn_v"), get("attn_output"),
         get("attn_q_norm"), get("attn_k_norm")},
        {get("ffn_gate_inp"), get("ffn_gate_inp_shexp"), get("ffn_gate_exps"),
         get("ffn_up_exps"), get("ffn_down_exps"), get("ffn_gate_shexp"),
         get("ffn_up_shexp"), get("ffn_down_shexp")},
        {get("indexer.q_proj"), get("indexer.k_proj"), get("indexer.q_norm"), get("indexer.k_norm")}
    };
}

} // namespace

MtpModel::MtpModel(const std::filesystem::path& sidecar_path, const Model& target)
    : sidecar_(sidecar_path), target_(target), geometry_(validate(sidecar_, target_)),
      weights_(bind_weights(sidecar_)),
      shared_{target_.tensor("token_embd.weight"), target_.tensor("output.weight")},
      type_counts_(count_types(sidecar_)) {}

} // namespace qwen
