#include "attention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace qwen {
namespace {

struct Range { std::uintptr_t begin, end; bool writable; };

template<class T>
Range range(std::span<T> values, bool writable = false) {
    const auto begin = reinterpret_cast<std::uintptr_t>(values.data());
    if (values.empty() || values.data() == nullptr || begin % alignof(T) != 0 ||
        values.size() > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T) ||
        values.size() > (std::numeric_limits<std::uintptr_t>::max() - begin) / sizeof(T))
        throw std::invalid_argument("attention: invalid address range");
    return {begin, begin + values.size_bytes(), writable};
}

template<std::size_t N>
void disjoint(const std::array<Range, N>& buffers) {
    for (std::size_t i = 0; i < N; ++i) for (std::size_t j = i + 1; j < N; ++j) {
        const auto& a = buffers[i]; const auto& b = buffers[j];
        if ((a.writable || b.writable) && a.begin < b.end && b.begin < a.end)
            throw std::invalid_argument("attention: writable overlap");
    }
}

float decode(const Q4_0* row, int d, AttentionQ4Mode mode) {
    const auto& block = row[d / 32];
    const int i = d % 32;
    const int code = i < 16 ? block.qs[i] & 15 : block.qs[i - 16] >> 4;
    const float value = half_to_float(block.d) * static_cast<float>(code - 8);
    return mode == AttentionQ4Mode::gathered_fp16 ? half_to_float(float_to_half(value)) : value;
}

float finite(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("attention: nonfinite FP32 arithmetic");
    return value;
}

} // namespace

void attention_q4(AttentionQ4Cache cache, std::span<const float> query,
                  std::span<const std::int32_t> ids, std::size_t visible,
                  AttentionQ4Mode mode, AttentionQ4Scratch scratch,
                  std::span<float> output) {
    if (cache.capacity == 0 || cache.capacity > attention_max_capacity || visible == 0 ||
        visible > cache.capacity || ids.empty() || ids.size() > attention_max_selected ||
        query.size() != attention_query_elements || output.size() != attention_query_elements ||
        cache.keys.size() != cache.capacity * 16 || cache.values.size() != cache.capacity * 16 ||
        scratch.scores.size() != ids.size() || scratch.output.size() != attention_query_elements ||
        (mode != AttentionQ4Mode::gathered_fp16 && mode != AttentionQ4Mode::direct_fp32))
        throw std::invalid_argument("attention: shape/count/mode");
    disjoint(std::array{range(cache.keys), range(cache.values), range(query), range(ids),
                        range(scratch.scores, true), range(scratch.output, true), range(output, true)});
    // Whole-call input preflight: no scratch/output mutation, and no speculative
    // decode of even a finite scale until every selected scale bit pattern passes.
    for (auto id : ids) if (id < 0 || static_cast<std::size_t>(id) >= visible)
        throw std::invalid_argument("attention: selected ID outside causal visibility");
    for (float q : query) if (!std::isfinite(q))
        throw std::invalid_argument("attention: nonfinite query");
    for (auto id : ids) {
        const auto base = static_cast<std::size_t>(id) * 16;
        for (std::size_t b = 0; b < 16; ++b)
            if ((cache.keys[base + b].d & 0x7c00) == 0x7c00 ||
                (cache.values[base + b].d & 0x7c00) == 0x7c00)
                throw std::invalid_argument("attention: nonfinite Q4 scale");
    }
    if (mode == AttentionQ4Mode::gathered_fp16) for (auto id : ids) {
        const auto base = static_cast<std::size_t>(id) * 16;
        for (const auto* row : {cache.keys.data() + base, cache.values.data() + base})
            for (int d = 0; d < attention_kv_heads * attention_head_dim; ++d) {
                const float value = decode(row, d, AttentionQ4Mode::direct_fp32);
                if ((float_to_half(value) & 0x7c00) == 0x7c00)
                    throw std::invalid_argument("attention: selected FP16 gather overflow");
            }
    }
    for (int h = 0; h < attention_query_heads; ++h) {
        const int kv = h / attention_gqa;
        const float* q = query.data() + h * attention_head_dim;
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t j = 0; j < ids.size(); ++j) {
            const auto* k = cache.keys.data() + static_cast<std::size_t>(ids[j]) * 16 + kv * 8;
            float score = 0.0f;
            for (int d = 0; d < attention_head_dim; ++d)
                score = finite(score + finite(q[d] * decode(k, d, mode)));
            score = finite(score * attention_query_coefficient);
            scratch.scores[j] = score;
            maximum = std::max(maximum, score);
        }
        float sum = 0.0f;
        for (float& score : scratch.scores) {
            score = finite(std::exp(finite(score - maximum)));
            sum = finite(sum + score);
        }
        if (!(sum > 0.0f)) throw std::runtime_error("attention: empty softmax denominator");
        const float inverse = finite(1.0f / sum);
        for (int d = 0; d < attention_head_dim; ++d) {
            float numerator = 0.0f;
            for (std::size_t j = 0; j < ids.size(); ++j) {
                const auto* v = cache.values.data() + static_cast<std::size_t>(ids[j]) * 16 + kv * 8;
                numerator = finite(numerator + finite(scratch.scores[j] * decode(v, d, mode)));
            }
            scratch.output[h * attention_head_dim + d] = finite(numerator * inverse);
        }
    }
    std::copy(scratch.output.begin(), scratch.output.end(), output.begin());
}

} // namespace qwen
