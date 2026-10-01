#include "hc.hpp"

#include "dense.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

// WRITE OURS: equations only, no donor implementation imported.
// Mathematical reference: Transformers a005fc82babfe8871d87746decad2dbee100a125,
// src/transformers/models/qwen4_exp/modular_qwen4_exp.py:
// Qwen4ExpTextRMSNorm::_norm, lines 299-310 (group = hidden_size);
// Qwen4ExpTextGatedResidual, lines 531-563 (mix and injection);
// Qwen4ExpTextDecoderLayer::forward, lines 815-838 (both residual combines);
// Qwen4ExpTextModel, lines 906, 1014-1027 (repeat and head-only mixer).
// Stored-weight / GGUF layout reference: mx-llama.cpp
// dcd685463d597d31f5ca759d32c94592a2740fa4, src/models/qwen4exp.cpp:
// load_arch_tensors, lines 145-149, 216-225;
// graph::build_hc_mix/build_hc_combine, lines 255-325;
// graph constructor, lines 536-570 (full widened pre-head t_h_nextn Tap).
// That graph explicitly confirms the converter folded gamma = 1 + HF weight.
// We consume the actual GGUF gamma directly, NEVER add one again.
//
// For C branches, H features, R rank, X[c,j] and GGUF gamma[c,j]:
// N[c,j] = (X[c,j] / sqrt(mean_j(X[c,j]^2) + eps)) * gamma[c,j]
// L[r]   = SiLU((sum_cj down[r,cj] * N[c,j]) / C)
// G[c,j] = sigmoid(sum_r up[cj,r] * L[r])
// M[j]   = mean_c(G[c,j] * N[c,j])
// I[c]   = 2 * sigmoid((sum_cj inject[c,cj] * N[c,j]) / C)
// Y[c,j] = X[c,j] + I[c] * block_output[j]
// The head returns M alone, using its own norm/down/up and NO injection tensor.
// There is no tanh, softmax, cross-branch normalization, residual remix or extra
// norm in this HC. All math here is FP32 (not HF BF16 autocast); canonical stored
// weights are decoded with dense.hpp, without Q8 activation quantization.

namespace qwen {
namespace {

[[noreturn]] void invalid(const char* message) { throw std::invalid_argument(message); }

std::size_t multiply(std::size_t a, std::size_t b) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        invalid("HC: size overflow");
    return a * b;
}

struct Range { std::uintptr_t begin, end; };

Range range(const void* pointer, std::size_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if ((bytes != 0 && pointer == nullptr) ||
        bytes > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        invalid("HC: invalid address range");
    return {begin, begin + bytes};
}

template<class T> Range range(std::span<T> values) {
    return range(values.data(), multiply(values.size(), sizeof(T)));
}

bool overlaps(Range a, Range b) {
    return a.begin < a.end && b.begin < b.end && a.begin < b.end && b.begin < a.end;
}

template<class A, class B> void disjoint(std::span<A> a, std::span<B> b) {
    if (overlaps(range(a), range(b))) invalid("HC: writable span aliases another buffer");
}

void finite(std::span<const float> values) {
    range(values);
    for (const float value : values)
        if (!std::isfinite(value)) invalid("HC: nonfinite argument");
}

float arithmetic(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("HC: nonfinite FP32 arithmetic");
    return value;
}

float sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    const float e = std::exp(x);
    return e / (1.0f + e);
}

void validate_matrix(const QMatrix& matrix, std::size_t input, std::size_t output) {
    if (matrix.input <= 0 || matrix.output <= 0 ||
        static_cast<std::size_t>(matrix.input) != input ||
        static_cast<std::size_t>(matrix.output) != output)
        invalid("HC: matrix geometry mismatch");
    TypeLayout layout{};
    switch (matrix.type) {
    case TensorType::F32: case TensorType::F16: case TensorType::BF16:
    case TensorType::Q4_0: case TensorType::Q4_1: case TensorType::Q5_0:
    case TensorType::Q8_0: case TensorType::Q6_K:
        layout = type_layout(matrix.type);
        break;
    default: invalid("HC: unsupported tensor type");
    }
    if (input % layout.block_elements != 0 || matrix.weights.size() !=
        multiply(multiply(input / layout.block_elements, layout.block_bytes), output))
        invalid("HC: matrix row alignment or weight size mismatch");
    range(matrix.weights);
    const auto count = multiply(input, output);
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(tensor_element(matrix.type, matrix.weights, i)))
            invalid("HC: nonfinite matrix weight");
}

void project(const QMatrix& matrix, std::span<const float> input,
             std::size_t tokens, std::span<float> output) {
    // Constructor checked every weight and dimension; borrowed weights remain
    // immutable and intermediate activations were checked before this call.
    // dense reports projection overflow as invalid_argument; at this level it
    // is an arithmetic failure, so preserve HC's runtime_error contract.
    try {
        matmul_f32(matrix, input, static_cast<int>(tokens), output);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("HC: nonfinite FP32 projection");
    }
}

std::size_t scratch_count(std::size_t width, std::size_t capacity) {
    const auto count = multiply(width, capacity);
    if (multiply(count, sizeof(float)) >
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()))
        invalid("HC: scratch size overflow");
    return count;
}

} // namespace

std::size_t HcConfig::widened_elements() const {
    if (hidden_size == 0 || branches <= 1 || low_rank == 0 ||
        !std::isfinite(rms_epsilon) || rms_epsilon <= 0.0f)
        invalid("HC: invalid configuration");
    const auto wide = multiply(hidden_size, branches);
    const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (wide > limit || low_rank > limit) invalid("HC: dimension exceeds QMatrix ABI");
    return wide;
}

HcCpu::HcCpu(HcConfig config, HcParameters parameters, std::size_t capacity)
    : config_(config), parameters_(parameters), capacity_(capacity) {
    const auto wide = config_.widened_elements();
    if (capacity == 0 || capacity > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        invalid("HC: invalid token capacity");
    const auto wide_count = scratch_count(wide, capacity);
    const auto low_count = scratch_count(config.low_rank, capacity);
    const auto hidden_count = scratch_count(config.hidden_size, capacity);
    const auto branch_count = scratch_count(config.branches, capacity);
    if (parameters.norm_weight.size() != wide) invalid("HC: norm size mismatch");
    finite(parameters.norm_weight);
    validate_matrix(parameters.down, wide, config.low_rank);
    validate_matrix(parameters.up, config.low_rank, wide);
    if (parameters.inject) validate_matrix(*parameters.inject, wide, config.branches);
    normalized_.resize(wide_count);
    low_.resize(low_count);
    gates_.resize(wide_count);
    mixed_.resize(hidden_count);
    if (parameters.inject) injection_.resize(branch_count);
}

void HcCpu::validate_tokens(std::size_t tokens) const {
    if (tokens == 0 || tokens > capacity_) invalid("HC: token count exceeds capacity");
}

void HcCpu::validate_output(std::span<float> output) const {
    range(output);
    disjoint(output, parameters_.norm_weight);
    disjoint(output, parameters_.down.weights);
    disjoint(output, parameters_.up.weights);
    if (parameters_.inject) disjoint(output, parameters_.inject->weights);
    disjoint(output, std::span<const float>(normalized_));
    disjoint(output, std::span<const float>(low_));
    disjoint(output, std::span<const float>(gates_));
    disjoint(output, std::span<const float>(mixed_));
    disjoint(output, std::span<const float>(injection_));
}

void HcCpu::expand(std::span<const float> embedding, std::size_t tokens,
                   std::span<float> widened) {
    validate_tokens(tokens);
    const auto h = config_.hidden_size, c = config_.branches, wide = h * c;
    if (embedding.size() != tokens * h || widened.size() != tokens * wide)
        invalid("HC: expand span size mismatch");
    validate_output(widened);
    disjoint(embedding, widened);
    finite(embedding);
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t branch = 0; branch < c; ++branch)
            std::copy_n(embedding.data() + t * h, h, widened.data() + t * wide + branch * h);
}

void HcCpu::mix(std::span<const float> residual, std::size_t tokens,
                std::span<float> block_input, std::span<float> injection_weights) {
    validate_tokens(tokens);
    const auto h = config_.hidden_size, c = config_.branches, wide = h * c;
    const bool inject = parameters_.inject.has_value();
    if (residual.size() != tokens * wide || block_input.size() != tokens * h ||
        injection_weights.size() != (inject ? tokens * c : 0))
        invalid("HC: mix span size mismatch");
    validate_output(block_input);
    validate_output(injection_weights);
    disjoint(residual, block_input);
    disjoint(residual, injection_weights);
    disjoint(block_input, injection_weights);
    finite(residual);

    auto normalized = std::span(normalized_).first(tokens * wide);
    auto low = std::span(low_).first(tokens * config_.low_rank);
    auto gates = std::span(gates_).first(tokens * wide);
    auto mixed = std::span(mixed_).first(tokens * h);
    auto injection = std::span(injection_).first(inject ? tokens * c : 0);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t branch = 0; branch < c; ++branch) {
            const auto start = t * wide + branch * h;
            float squares = 0.0f;
            for (std::size_t j = 0; j < h; ++j) {
                const float x = residual[start + j];
                squares += x * x;
            }
            const float inverse = arithmetic(1.0f / std::sqrt(
                arithmetic(squares / static_cast<float>(h) + config_.rms_epsilon)));
            for (std::size_t j = 0; j < h; ++j)
                normalized[start + j] = arithmetic((residual[start + j] * inverse) *
                    parameters_.norm_weight[branch * h + j]);
        }
    }
    project(parameters_.down, normalized, tokens, low);
    for (float& value : low) {
        const float scaled = value / static_cast<float>(c);
        value = arithmetic(scaled * sigmoid(scaled));
    }
    project(parameters_.up, low, tokens, gates);
    for (float& value : gates) value = sigmoid(value);
    for (std::size_t t = 0; t < tokens; ++t) {
        for (std::size_t j = 0; j < h; ++j) {
            float sum = 0.0f;
            for (std::size_t branch = 0; branch < c; ++branch) {
                const auto index = t * wide + branch * h + j;
                sum += gates[index] * normalized[index];
            }
            mixed[t * h + j] = arithmetic(sum / static_cast<float>(c));
        }
    }
    if (inject) {
        project(*parameters_.inject, normalized, tokens, injection);
        for (float& value : injection)
            value = 2.0f * sigmoid(value / static_cast<float>(c));
    }
    // Commit only after every token, projection and activated output succeeded.
    std::copy(mixed.begin(), mixed.end(), block_input.begin());
    std::copy(injection.begin(), injection.end(), injection_weights.begin());
}

void HcCpu::combine(std::span<const float> residual, std::span<const float> block_output,
                    std::span<const float> injection_weights, std::size_t tokens,
                    std::span<float> widened) {
    validate_tokens(tokens);
    const auto h = config_.hidden_size, c = config_.branches, wide = h * c;
    if (!parameters_.inject) invalid("HC: head mixer has no injection");
    if (residual.size() != tokens * wide || block_output.size() != tokens * h ||
        injection_weights.size() != tokens * c || widened.size() != tokens * wide)
        invalid("HC: combine span size mismatch");
    validate_output(widened);
    disjoint(residual, widened);
    disjoint(block_output, widened);
    disjoint(injection_weights, widened);
    finite(residual);
    finite(block_output);
    finite(injection_weights);
    for (const float gate : injection_weights)
        if (gate < 0.0f || gate > 2.0f) invalid("HC: injection gate outside [0,2]");
    auto next = std::span(normalized_).first(tokens * wide);
    for (std::size_t t = 0; t < tokens; ++t)
        for (std::size_t branch = 0; branch < c; ++branch)
            for (std::size_t j = 0; j < h; ++j) {
                const auto index = t * wide + branch * h + j;
                const float contribution = arithmetic(injection_weights[t * c + branch] *
                    block_output[t * h + j]);
                next[index] = arithmetic(residual[index] + contribution);
            }
    std::copy(next.begin(), next.end(), widened.begin());
}

} // namespace qwen
