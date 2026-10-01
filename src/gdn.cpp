#include "gdn.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace qwen {
namespace {

// WRITE OURS: scalar equations, no donor kernel code imported.
// Architecture: Transformers a005fc82babfe8871d87746decad2dbee100a125,
// qwen4_exp/modular_qwen4_exp.py Qwen4ExpTextGatedDeltaNet ->
// qwen3_5/modeling_qwen3_5.py Qwen3_5GatedDeltaNet.forward,
// causal_conv1d_{fn,update}, l2norm, torch_recurrent_gated_delta_rule;
// qwen3_next/modeling_qwen3_next.py Qwen3NextRMSNormGated.forward.
// All under src/transformers/models/. Q/K epsilon is the recurrence's fixed
// 1e-6, independently of the output RMS epsilon. Output gate is sigmoid.
// KEEP external recurrent layout: mx dcd685463d597d31f5ca759d32c94592a2740fa4
// ggml/src/ggml-cuda/gated_delta_net{,_chunk}.cu (col * D + k), also furnace
// 905021dbad71c5056ef51f9fd45d545403fc989c gated_delta_net.cu resident Wave64.
// Reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8 kernels/
// gdn_recurrent_step_fused.cpp / gdn_recurrent_batched_v2.cpp use the opposite
// HBM orientation [k][v]; their LDS swizzle is private, NOT this snapshot ABI.
// GGUF tiled heads: mx conversion/qwen.py _LinearAttentionVReorderBase and
// conversion/qwen4exp.py Qwen4ExpTextModel. HF's repeat_interleave becomes h%K
// after this value-preserving reorder; this is not a changed head association.

std::size_t multiply(std::size_t a, std::size_t b) {
    constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (a > maximum || (b != 0 && a > maximum / b))
        throw std::invalid_argument("GDN size overflow");
    return a * b;
}

std::size_t add(std::size_t a, std::size_t b) {
    constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    if (a > maximum || b > maximum - a)
        throw std::invalid_argument("GDN size overflow");
    return a + b;
}

void validate_config(const GdnConfig& c) {
    if (c.key_heads == 0 || c.value_heads == 0 || c.key_head_dim == 0 ||
        c.value_head_dim == 0 || c.conv_width == 0 || c.value_heads % c.key_heads != 0 ||
        !std::isfinite(c.rms_epsilon) || c.rms_epsilon <= 0)
        throw std::invalid_argument("invalid GDN geometry/epsilon");
    const auto key = multiply(c.key_heads, c.key_head_dim);
    const auto value = multiply(c.value_heads, c.value_head_dim);
    const auto features = add(multiply(2, key), value);
    (void)multiply(features, c.conv_width);
    (void)multiply(value, c.key_head_dim);
}

void finite(std::span<const float> values, const char* name) {
    for (float value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument(std::string("nonfinite GDN ") + name);
}

void exact_size(std::span<const float> values, std::size_t size, const char* name) {
    if (values.size() != size)
        throw std::invalid_argument(std::string("wrong GDN ") + name + " size");
}

float checked(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("GDN FP32 arithmetic overflow");
    return value;
}

float sigmoid(float x) {
    const float e = std::exp(-std::abs(x));
    return x >= 0 ? 1.0f / (1.0f + e) : e / (1.0f + e);
}

float softplus(float x) {
    // PyTorch F.softplus default beta=1, threshold=20, without exp overflow.
    return x > 20.0f ? x : std::log1p(std::exp(x));
}

bool overlaps(std::span<const float> a, std::span<const float> b) {
    if (a.empty() || b.empty()) return false;
    const auto aa = reinterpret_cast<std::uintptr_t>(a.data());
    const auto bb = reinterpret_cast<std::uintptr_t>(b.data());
    // Difference comparisons avoid end-address overflow and unrelated-pointer UB.
    return aa <= bb ? bb - aa < a.size_bytes() : aa - bb < b.size_bytes();
}

bool same_config(const GdnConfig& a, const GdnConfig& b) {
    return a.key_heads == b.key_heads && a.value_heads == b.value_heads &&
           a.key_head_dim == b.key_head_dim && a.value_head_dim == b.value_head_dim &&
           a.conv_width == b.conv_width && a.rms_epsilon == b.rms_epsilon;
}

}

std::size_t GdnConfig::key_elements() const {
    validate_config(*this);
    return multiply(key_heads, key_head_dim);
}
std::size_t GdnConfig::value_elements() const {
    validate_config(*this);
    return multiply(value_heads, value_head_dim);
}
std::size_t GdnConfig::qkv_elements() const {
    validate_config(*this);
    return add(multiply(2, key_elements()), value_elements());
}
std::size_t GdnConfig::recurrent_elements() const {
    validate_config(*this);
    return multiply(value_elements(), key_head_dim);
}
std::size_t GdnConfig::conv_elements() const {
    validate_config(*this);
    return multiply(qkv_elements(), conv_width - 1);
}

GdnCheckpoint::GdnCheckpoint(GdnConfig config)
    : config_(config), recurrent_(config.recurrent_elements(), 0.0f),
      conv_history_(config.conv_elements(), 0.0f) {}

GdnCpu::GdnCpu(GdnConfig config, GdnParameters p)
    : state_(config) {
    exact_size(p.conv_weights, multiply(config.qkv_elements(), config.conv_width), "conv weights");
    exact_size(p.dt_bias, config.value_heads, "dt bias");
    exact_size(p.ssm_a, config.value_heads, "ssm A");
    exact_size(p.norm_weight, config.value_head_dim, "norm weight");
    finite(p.conv_weights, "conv weights");
    finite(p.dt_bias, "dt bias");
    finite(p.ssm_a, "ssm A");
    finite(p.norm_weight, "norm weight");
    if (std::any_of(p.ssm_a.begin(), p.ssm_a.end(), [](float a) { return a > 0; }))
        throw std::invalid_argument("GDN ssm A must be -exp(A_log), not A_log/positive A");
    conv_weights_.assign(p.conv_weights.begin(), p.conv_weights.end());
    dt_bias_.assign(p.dt_bias.begin(), p.dt_bias.end());
    ssm_a_.assign(p.ssm_a.begin(), p.ssm_a.end());
    norm_weight_.assign(p.norm_weight.begin(), p.norm_weight.end());
    qkv_scratch_.resize(config.qkv_elements());
    output_scratch_.resize(config.value_elements());
    next_recurrent_.resize(config.recurrent_elements());
}

void GdnCpu::reset() noexcept {
    std::fill(state_.recurrent_.begin(), state_.recurrent_.end(), 0.0f);
    std::fill(state_.conv_history_.begin(), state_.conv_history_.end(), 0.0f);
    state_.consumed_tokens_ = 0;
}

void GdnCpu::validate_checkpoint(const GdnCheckpoint& checkpoint) const {
    if (!same_config(state_.config_, checkpoint.config_) ||
        checkpoint.recurrent_.size() != state_.recurrent_.size() ||
        checkpoint.conv_history_.size() != state_.conv_history_.size())
        throw std::invalid_argument("GDN checkpoint geometry mismatch");
    finite(checkpoint.recurrent_, "checkpoint recurrent state");
    finite(checkpoint.conv_history_, "checkpoint conv state");
}

void GdnCpu::save(GdnCheckpoint& checkpoint) const {
    validate_checkpoint(checkpoint);
    if (&checkpoint == &state_) return;
    std::copy(state_.recurrent_.begin(), state_.recurrent_.end(), checkpoint.recurrent_.begin());
    std::copy(state_.conv_history_.begin(), state_.conv_history_.end(), checkpoint.conv_history_.begin());
    checkpoint.consumed_tokens_ = state_.consumed_tokens_;
}

void GdnCpu::restore(const GdnCheckpoint& checkpoint) {
    validate_checkpoint(checkpoint);
    if (&checkpoint == &state_) return;
    std::copy(checkpoint.recurrent_.begin(), checkpoint.recurrent_.end(), state_.recurrent_.begin());
    std::copy(checkpoint.conv_history_.begin(), checkpoint.conv_history_.end(), state_.conv_history_.begin());
    state_.consumed_tokens_ = checkpoint.consumed_tokens_;
}

void GdnCpu::step(GdnInput input, std::span<float> output) {
    run(input, 1, output);
}

void GdnCpu::run(GdnInput input, std::size_t tokens, std::span<float> output,
                 std::span<GdnCheckpoint> prefixes) {
    const auto& c = state_.config_;
    const auto features = c.qkv_elements();
    const auto values = c.value_elements();
    exact_size(input.qkv, multiply(tokens, features), "qkv input");
    exact_size(input.z, multiply(tokens, values), "z input");
    exact_size(input.alpha, multiply(tokens, c.value_heads), "alpha input");
    exact_size(input.beta, multiply(tokens, c.value_heads), "beta input");
    exact_size(output, multiply(tokens, values), "output");
    if (tokens > std::numeric_limits<std::uint64_t>::max() - state_.consumed_tokens_)
        throw std::invalid_argument("GDN consumed token count overflow");
    if (!prefixes.empty() && (tokens == std::numeric_limits<std::size_t>::max() || prefixes.size() != tokens + 1))
        throw std::invalid_argument("GDN needs tokens+1 prefix checkpoints");
    const std::array<std::span<const float>, 4> inputs{input.qkv, input.z, input.alpha, input.beta};
    const std::array<std::span<const float>, 9> owned{
        state_.recurrent_, state_.conv_history_, conv_weights_, dt_bias_, ssm_a_, norm_weight_,
        qkv_scratch_, output_scratch_, next_recurrent_};
    for (auto in : inputs) {
        finite(in, "input");
        if (overlaps(in, output)) throw std::invalid_argument("GDN input/output overlap");
        for (auto buffer : owned)
            if (overlaps(in, buffer)) throw std::invalid_argument("GDN input aliases owned buffer");
    }
    for (auto buffer : owned)
        if (overlaps(output, buffer)) throw std::invalid_argument("GDN output aliases owned buffer");
    for (const auto& prefix : prefixes) {
        validate_checkpoint(prefix);
        if (&prefix == &state_) throw std::invalid_argument("GDN active state is not a prefix destination");
        for (auto buffer : {prefix.recurrent(), prefix.conv_history()}) {
            if (overlaps(output, buffer)) throw std::invalid_argument("GDN output/prefix overlap");
            for (auto in : inputs)
                if (overlaps(in, buffer)) throw std::invalid_argument("GDN input/prefix overlap");
        }
    }
    if (!prefixes.empty()) save(prefixes[0]);
    for (std::size_t t = 0; t < tokens; ++t) {
        step_unchecked({input.qkv.subspan(t * features, features), input.z.subspan(t * values, values),
                        input.alpha.subspan(t * c.value_heads, c.value_heads),
                        input.beta.subspan(t * c.value_heads, c.value_heads)},
                       output.subspan(t * values, values));
        if (!prefixes.empty()) save(prefixes[t + 1]);
    }
}

void GdnCpu::step_unchecked(GdnInput input, std::span<float> output) {
    const auto& c = state_.config_;
    const auto keys = c.key_elements();
    const auto features = c.qkv_elements();
    const auto history_width = c.conv_width - 1;
    for (std::size_t f = 0; f < features; ++f) {
        float x = 0;
        for (std::size_t age = 0; age < history_width; ++age)
            x += state_.conv_history_[f * history_width + age] * conv_weights_[f * c.conv_width + age];
        x = checked(x + input.qkv[f] * conv_weights_[f * c.conv_width + history_width]);
        qkv_scratch_[f] = checked(x * sigmoid(x)); // convolution uses SiLU, output uses sigmoid
    }
    const float query_scale = 1.0f / std::sqrt(float(c.key_head_dim));
    for (std::size_t h = 0; h < 2 * c.key_heads; ++h) {
        const auto offset = h * c.key_head_dim;
        float sum = 0;
        for (std::size_t k = 0; k < c.key_head_dim; ++k)
            sum += qkv_scratch_[offset + k] * qkv_scratch_[offset + k];
        const float inv_norm = 1.0f / std::sqrt(checked(sum + 1.0e-6f));
        for (std::size_t k = 0; k < c.key_head_dim; ++k) {
            float normalized = qkv_scratch_[offset + k] * inv_norm;
            if (h < c.key_heads) normalized *= query_scale;
            qkv_scratch_[offset + k] = checked(normalized);
        }
    }
    for (std::size_t h = 0; h < c.value_heads; ++h) {
        const auto kh = h % c.key_heads;
        const float beta = sigmoid(input.beta[h]);
        const float log_decay = checked(ssm_a_[h] * softplus(checked(input.alpha[h] + dt_bias_[h])));
        const float decay = std::exp(log_decay);
        for (std::size_t v = 0; v < c.value_head_dim; ++v) {
            const auto base = (h * c.value_head_dim + v) * c.key_head_dim;
            float predicted = 0;
            for (std::size_t k = 0; k < c.key_head_dim; ++k) {
                next_recurrent_[base + k] = checked(decay * state_.recurrent_[base + k]);
                predicted += next_recurrent_[base + k] * qkv_scratch_[keys + kh * c.key_head_dim + k];
            }
            const float delta = checked((qkv_scratch_[2 * keys + h * c.value_head_dim + v] - checked(predicted)) * beta);
            float read = 0;
            for (std::size_t k = 0; k < c.key_head_dim; ++k) {
                next_recurrent_[base + k] = checked(next_recurrent_[base + k] +
                    qkv_scratch_[keys + kh * c.key_head_dim + k] * delta);
                read += next_recurrent_[base + k] * qkv_scratch_[kh * c.key_head_dim + k];
            }
            output_scratch_[h * c.value_head_dim + v] = checked(read);
        }
        float sum = 0;
        for (std::size_t v = 0; v < c.value_head_dim; ++v) {
            const float x = output_scratch_[h * c.value_head_dim + v];
            sum += x * x;
        }
        const float inv_rms = 1.0f / std::sqrt(checked(sum / float(c.value_head_dim) + c.rms_epsilon));
        for (std::size_t v = 0; v < c.value_head_dim; ++v) {
            const auto i = h * c.value_head_dim + v;
            output_scratch_[i] = checked(checked(output_scratch_[i] * inv_rms) * norm_weight_[v]);
            output_scratch_[i] = checked(output_scratch_[i] * sigmoid(input.z[i]));
        }
    }
    // Publish only after all FP32 intermediates/outputs are finite. Scratch may
    // be dirty after an exception, but the active state and history never are.
    std::copy(next_recurrent_.begin(), next_recurrent_.end(), state_.recurrent_.begin());
    for (std::size_t f = 0; f < features; ++f) {
        for (std::size_t age = 0; age + 1 < history_width; ++age)
            state_.conv_history_[f * history_width + age] = state_.conv_history_[f * history_width + age + 1];
        if (history_width != 0)
            state_.conv_history_[f * history_width + history_width - 1] = input.qkv[f];
    }
    ++state_.consumed_tokens_;
    std::copy(output_scratch_.begin(), output_scratch_.end(), output.begin());
}

}
