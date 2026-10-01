#include "qsa.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

std::size_t checks = 0;
std::size_t rejections = 0;
std::size_t prefix_queries = 0;
constexpr std::int32_t canary = -1234567;

void check(bool condition, const char* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

template<class Function> void rejected(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        ++rejections;
        return;
    }
    throw std::runtime_error("invalid QSA arguments unexpectedly accepted");
}

struct Selection {
    std::vector<std::int32_t> tokens;
    std::vector<std::int32_t> blocks;
};

Selection select(std::span<const float> scores, std::size_t visible, qwen::QsaConfig config = {}) {
    // Guard both ends and the unused capacity, including the three unused
    // entries at 2052. Caller supplies preallocated output, not padding IDs.
    const auto capacity = config.token_budget + config.compress_ratio - 1;
    std::vector<std::int32_t> tokens(capacity + 2, canary);
    std::vector<std::int32_t> blocks(config.token_budget / config.compress_ratio + 2, canary);
    const auto counts = qwen::qsa_select(scores, visible,
        std::span(tokens).subspan(1, capacity), std::span(blocks).subspan(1, blocks.size() - 2), config);
    check(counts.block_count == std::min(visible / config.compress_ratio,
                                      config.token_budget / config.compress_ratio), "actual block count");
    check(counts.token_count == counts.block_count * config.compress_ratio +
                               visible % config.compress_ratio, "actual token count");
    check(tokens.front() == canary && blocks.front() == canary, "leading output guards");
    check(std::all_of(tokens.begin() + 1 + counts.token_count, tokens.end(),
                      [](auto id) { return id == canary; }), "unused token capacity changed");
    check(std::all_of(blocks.begin() + 1 + counts.block_count, blocks.end(),
                      [](auto id) { return id == canary; }), "unused block capacity changed");
    Selection result{{tokens.begin() + 1, tokens.begin() + 1 + counts.token_count},
                     {blocks.begin() + 1, blocks.begin() + 1 + counts.block_count}};
    std::vector<bool> seen(visible, false);
    for (const auto id : result.tokens) {
        check(id >= 0 && static_cast<std::size_t>(id) < visible, "noncausal or padding token ID");
        check(!seen[static_cast<std::size_t>(id)], "duplicate selected token ID");
        seen[static_cast<std::size_t>(id)] = true;
    }
    for (std::size_t rank = 0; rank < result.blocks.size(); ++rank) {
        const auto block = result.blocks[rank];
        check(block >= 0 && static_cast<std::size_t>(block) < scores.size(), "future block ID");
        for (std::size_t member = 0; member < config.compress_ratio; ++member)
            check(result.tokens[rank * config.compress_ratio + member] ==
                  static_cast<std::int32_t>(static_cast<std::size_t>(block) * config.compress_ratio + member),
                  "selected block not expanded intact/in order");
    }
    for (std::size_t id = visible / config.compress_ratio * config.compress_ratio; id < visible; ++id)
        check(result.tokens[result.blocks.size() * config.compress_ratio +
                            id % config.compress_ratio] == static_cast<std::int32_t>(id),
              "actual tail not appended intact/in order");
    return result;
}

// Independent rank-count oracle, deliberately not another sort implementation.
// Every candidate's rank is the number of blocks that beat it. This checks
// sorted output and the cutoff's tied IDs rather than relying on plausible text.
Selection reference(std::span<const float> scores, std::size_t visible, qwen::QsaConfig config = {}) {
    const auto ratio = config.compress_ratio;
    const auto count = std::min(scores.size(), config.token_budget / ratio);
    Selection result{std::vector<std::int32_t>(count * ratio + visible % ratio),
                     std::vector<std::int32_t>(count)};
    for (std::size_t block = 0; block < scores.size(); ++block) {
        std::size_t rank = 0;
        for (std::size_t other = 0; other < scores.size(); ++other)
            if (scores[other] > scores[block] || (scores[other] == scores[block] && other < block)) ++rank;
        if (rank >= count) continue;
        result.blocks[rank] = static_cast<std::int32_t>(block);
        for (std::size_t member = 0; member < ratio; ++member)
            result.tokens[rank * ratio + member] = static_cast<std::int32_t>(block * ratio + member);
    }
    const auto tail_start = scores.size() * ratio;
    for (std::size_t id = tail_start; id < visible; ++id)
        result.tokens[count * ratio + id - tail_start] = static_cast<std::int32_t>(id);
    return result;
}

void same(const Selection& actual, const Selection& expected) {
    check(actual.blocks == expected.blocks, "selected block IDs/order differ");
    check(actual.tokens == expected.tokens, "selected token IDs/order differ");
}

std::vector<float> increasing_scores(std::size_t blocks) {
    std::vector<float> scores(blocks);
    for (std::size_t block = 0; block < blocks; ++block) scores[block] = static_cast<float>(block + 1);
    return scores;
}

void selection_tests() {
    for (std::size_t visible = 0; visible <= 20; ++visible) {
        const auto scores = increasing_scores(visible / 4);
        same(select(scores, visible), reference(scores, visible));
    }
    for (std::size_t visible = 2047; visible <= 2056; ++visible) {
        auto scores = increasing_scores(visible / 4);
        same(select(scores, visible), reference(scores, visible));
        // Exact ties at the cutoff and at zero are common after per-head ReLU.
        for (const float value : {0.0f, 7.0f}) {
            std::fill(scores.begin(), scores.end(), value);
            const auto result = select(scores, visible);
            same(result, reference(scores, visible));
            for (std::size_t rank = 0; rank < result.blocks.size(); ++rank)
                check(result.blocks[rank] == static_cast<std::int32_t>(rank), "tied block ID convention");
        }
        // Shuffled unique scores and quantized/tied scores, including a mixed
        // cutoff; catches accidentally selecting chronological/newest blocks.
        for (const bool ties : {false, true}) {
            for (std::size_t block = 0; block < scores.size(); ++block) {
                const auto permuted = (block * 137 + 71) % 521;
                scores[block] = static_cast<float>(ties ? permuted % 11 : permuted);
            }
            same(select(scores, visible), reference(scores, visible));
        }
    }
    const std::array<float, 5> extremes{std::numeric_limits<float>::lowest(), -0.0f,
        0.0f, std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::max()};
    same(select(extremes, 23), reference(extremes, 23));
    check(select(extremes, 23).blocks == std::vector<std::int32_t>({4, 3, 1, 2, 0}),
          "finite signed/zero/extreme score ordering");
    // Geometry is passed from metadata; defaults do not bypass config checks.
    for (const qwen::QsaConfig config : {qwen::QsaConfig{8, 4}, qwen::QsaConfig{3, 1}, qwen::QsaConfig{9, 3}}) {
        for (std::size_t visible = 0; visible <= 33; ++visible) {
            const auto scores = increasing_scores(visible / config.compress_ratio);
            same(select(scores, visible, config), reference(scores, visible, config));
        }
    }
}

void causal_chunk_tests() {
    constexpr std::size_t length = 2060;
    const auto scores = increasing_scores(length / 4);
    std::vector<Selection> sequential;
    sequential.reserve(length + 1);
    for (std::size_t visible = 0; visible <= length; ++visible)
        sequential.push_back(select(std::span(scores).first(visible / 4), visible));
    for (const std::size_t chunk_size : {1, 3, 4, 5, 127, 256, 1024, 2052}) {
        for (std::size_t start = 0; start < length; start += chunk_size) {
            const auto end = std::min(length, start + chunk_size);
            // All chunk keys may already exist, including a partial block that
            // only finishes in a later chunk. Only the per-query completed
            // prefix of scores is supplied; block boundaries are global.
            auto prepared = scores;
            for (std::size_t visible = start + 1; visible <= end; ++visible) {
                std::fill(prepared.begin() + visible / 4, prepared.end(),
                          std::numeric_limits<float>::quiet_NaN());
                same(select(std::span(prepared).first(visible / 4), visible), sequential[visible]);
                ++prefix_queries;
                prepared = scores;
            }
        }
    }
}

void invalid_tests() {
    std::array<std::int32_t, 2051> tokens;
    std::array<std::int32_t, 512> blocks;
    tokens.fill(canary);
    blocks.fill(canary);
    auto scores = increasing_scores(513);
    const auto fail = [&](std::span<const float> input, std::size_t visible,
                          std::span<std::int32_t> out_tokens, std::span<std::int32_t> out_blocks,
                          qwen::QsaConfig config = {}) {
        rejected([&] { (void)qwen::qsa_select(input, visible, out_tokens, out_blocks, config); });
        check(std::all_of(tokens.begin(), tokens.end(), [](auto id) { return id == canary; }) &&
              std::all_of(blocks.begin(), blocks.end(), [](auto id) { return id == canary; }),
              "rejected input changed outputs");
    };
    fail(std::span(scores).first(512), 2052, tokens, blocks); // missing visible block
    scores.push_back(1.0f);
    fail(scores, 2052, tokens, blocks); // extra future block
    scores.pop_back();
    fail(scores, 2052, std::span(tokens).first(2047), blocks);
    fail(scores, 2052, tokens, std::span(blocks).first(511));
    fail({}, 3, std::span(tokens).first(2), {}); // tail capacity, no complete blocks
    fail(scores, 2052, {}, blocks);
    fail(scores, 2052, tokens, {});
    fail(scores, 2052, tokens, std::span(tokens).first(512));
    // Check partial overlap in both directions, not just identical pointers.
    fail(scores, 2052, tokens, std::span(tokens).subspan(1536, 512));
    fail(std::span(scores).first(2), 8, std::span(tokens).subspan(1, 8), std::span(tokens).first(2));
    for (const float value : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity(),
                              -std::numeric_limits<float>::infinity()}) {
        scores.front() = value; // check even the otherwise discarded lowest block
        fail(scores, 2052, tokens, blocks);
    }
    scores.front() = 1.0f;
    for (const qwen::QsaConfig config : {qwen::QsaConfig{0, 4}, qwen::QsaConfig{2048, 0},
                                        qwen::QsaConfig{2049, 4}, qwen::QsaConfig{2, 4},
                                        qwen::QsaConfig{std::numeric_limits<std::size_t>::max(),
                                                        std::numeric_limits<std::size_t>::max()}})
        fail(scores, 2052, tokens, blocks, config);
    fail({}, std::size_t(std::numeric_limits<std::int32_t>::max()) + 2, tokens, blocks);
    fail({}, std::numeric_limits<std::size_t>::max(), tokens, blocks);
    fail(std::span(scores).first(1), 3, tokens, blocks); // incomplete block has no score
    fail(std::span(scores).first(1), 0, tokens, blocks);
    // Exactly the actual count fits, although the maximum capacity is 2051.
    const auto counts = qwen::qsa_select(scores, 2052, std::span(tokens).first(2048), blocks);
    check(counts.token_count == 2048 && counts.block_count == 512, "2052 exact-size outputs");
    check(tokens[2048] == canary && tokens[2050] == canary, "capacity is not actual count");
    const auto empty = qwen::qsa_select({}, 0, {}, {});
    check(empty.token_count == 0 && empty.block_count == 0, "empty output spans");
    std::array<std::int32_t, 15> adjacent;
    adjacent.fill(canary);
    const auto exact = qwen::qsa_select(std::span(scores).first(3), 12,
        std::span(adjacent).first(12), std::span(adjacent).subspan(12));
    check(exact.token_count == 12 && exact.block_count == 3, "adjacent nonoverlapping output regions");
    check(adjacent[0] == 8 && adjacent[12] == 2 && adjacent[14] == 0, "adjacent output IDs");
}

// TEST-ONLY diagnostic, not a runtime compatibility backend. Source pin:
// mx dcd685463d597d31f5ca759d32c94592a2740fa4 (verified remote HEAD),
// /home/radneon/src/worktrees/qwen38-pp-trace-75/:
// src/models/qwen4exp.cpp build_qsa_top_k (683-822), build_attn_qsa (826-904);
// src/llama-memory-hybrid-idx.cpp set_input_qsa (353-476), both bias modes;
// src/llama-kv-cache.cpp get_n_kv (1259-1273), at least 256-cell padding;
// ggml/src/ggml.c ggml_top_k (5463-5475);
// ggml/src/ggml-cuda/top-k.cu ggml_cuda_op_top_k (325-402), non-CUB HIP;
// ggml/src/ggml-cuda/argsort.cu k_argsort_f32_i32_large (223-269).
// ADAPT sorting-network comparisons for exact strict-comparison/padding/tie
// behavior, executing on CPU; GPU dispatch/timing is not tested here.
// Donor license for this diagnostic primitive:
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
std::vector<std::int32_t> fork_bitonic_topk(std::span<const float> scores, std::size_t width) {
    std::size_t padded = 1;
    while (padded < scores.size()) padded *= 2;
    std::vector<std::int32_t> order(padded);
    std::iota(order.begin(), order.end(), 0);
    for (std::size_t group = 2; group <= padded; group *= 2) {
        for (std::size_t stride = group / 2; stride != 0; stride /= 2) {
            for (std::size_t col = 0; col < padded; ++col) {
                const auto other = col ^ stride;
                if (other <= col) continue;
                const auto left = static_cast<std::size_t>(order[col]);
                const auto right = static_cast<std::size_t>(order[other]);
                const bool left_valid = left < scores.size();
                const bool right_valid = right < scores.size();
                bool swap;
                if ((col & group) == 0)
                    swap = !left_valid || (right_valid && scores[left] < scores[right]);
                else
                    swap = !right_valid || (left_valid && scores[left] > scores[right]);
                if (swap) std::swap(order[col], order[other]);
            }
        }
    }
    order.resize(width);
    return order;
}

struct ForkSelection {
    std::vector<std::int32_t> topk_cells; // includes any selected masked slots
    std::vector<std::int32_t> valid_tokens; // after build_attn_qsa causal mask
};

// Contiguous one-stream text cache, cell ID == token position, all current
// chunk cells loaded. Pooled block scores are supplied; pooling/norm/RoPE are
// outside this fixture. The partial block's gathered score is finite zero.
// Both bias modes force actual visible tails with +1e9, not +infinity.
ForkSelection fork_select(std::span<const float> pooled_scores, std::size_t loaded,
                          std::size_t visible, std::size_t n_kv, bool block_bias) {
    check(visible <= loaded && loaded <= n_kv && pooled_scores.size() == loaded / 4,
          "invalid diagnostic cache layout");
    check(n_kv <= 16384 && std::all_of(pooled_scores.begin(), pooled_scores.end(),
          [](float value) { return std::isfinite(value); }), "diagnostic outside inspected bitonic path");
    const float masked = -std::numeric_limits<float>::infinity();
    const auto tail_start = visible / 4 * 4;
    std::vector<float> expanded(n_kv, masked);
    for (std::size_t cell = 0; cell < loaded; ++cell) {
        const auto block = cell / 4;
        const bool complete = block < pooled_scores.size();
        float score = complete ? pooled_scores[block] : 0.0f;
        if (block_bias) {
            const float bias = block * 4 >= tail_start ? 1.0e9f : (complete ? 0.0f : masked);
            score = score + bias;
            expanded[cell] = score + (cell < visible ? 0.0f : masked);
        } else {
            // The per-cell mode maps an incomplete block's cell_blk to block 0.
            if (!complete) score = pooled_scores.empty() ? 0.0f : pooled_scores[0];
            const float bias = cell >= visible ? masked :
                               (cell >= tail_start ? 1.0e9f : (complete ? 0.0f : masked));
            expanded[cell] = score + bias;
        }
    }
    ForkSelection result{fork_bitonic_topk(expanded, std::min(n_kv, std::size_t{2051})), {}};
    for (const auto cell : result.topk_cells) {
        check(cell >= 0 && static_cast<std::size_t>(cell) < n_kv, "fork sort returned network padding");
        if (static_cast<std::size_t>(cell) < visible) result.valid_tokens.push_back(cell);
    }
    return result;
}

std::vector<std::int32_t> sorted_ids(std::vector<std::int32_t> ids) {
    std::sort(ids.begin(), ids.end());
    check(std::adjacent_find(ids.begin(), ids.end()) == ids.end(), "diagnostic selected duplicate IDs");
    return ids;
}

void fork_diagnostic_tests() {
    // Exact source-derived bitonic ties are intentionally not replaced by
    // stable_sort. A strict comparison network is not globally stable.
    const std::array<float, 4> tied{2.0f, 1.0f, 1.0f, 2.0f};
    check(fork_bitonic_topk(tied, 4) == std::vector<std::int32_t>({0, 3, 2, 1}), "fork strict bitonic tie fixture");
    constexpr std::array<std::size_t, 10> official_counts{2047, 2048, 2049, 2050, 2051,
                                                         2048, 2049, 2050, 2051, 2048};
    for (std::size_t visible = 2047; visible <= 2056; ++visible) {
        const auto n_kv = (visible + 255) / 256 * 256;
        const auto scores = increasing_scores(visible / 4);
        const auto official = select(scores, visible);
        const auto fork = fork_select(scores, visible, visible, n_kv, true);
        const auto per_cell = fork_select(scores, visible, visible, n_kv, false);
        check(sorted_ids(fork.valid_tokens) == sorted_ids(per_cell.valid_tokens), "fork bias mode valid-ID parity");
        check(official.tokens.size() == official_counts[visible - 2047], "official 2047-2056 count table");
        check(fork.valid_tokens.size() == std::min(visible, std::size_t{2051}), "fork actual valid count");
        const bool diverges = official.tokens.size() != fork.valid_tokens.size();
        check(diverges == (visible >= 2052 && visible % 4 != 3), "source-derived divergence boundary");
        if (!diverges)
            check(sorted_ids(official.tokens) == sorted_ids(fork.valid_tokens), "fork ID sets below cutoff");
        std::vector<std::int32_t> extra_ids;
        const auto official_set = sorted_ids(official.tokens);
        for (const auto id : fork.valid_tokens)
            if (!std::binary_search(official_set.begin(), official_set.end(), id)) extra_ids.push_back(id);
        extra_ids = sorted_ids(std::move(extra_ids));
        const auto extras = extra_ids.size();
        check(extras == fork.valid_tokens.size() - official.tokens.size(), "fork valid extra IDs");
        if (visible == 2052) {
            check(extras == 3, "2052 valid-ID divergence must be concrete, not only capacity");
            std::size_t from_lowest_block = 0;
            for (const auto id : fork.valid_tokens) if (id < 4) ++from_lowest_block;
            check(from_lowest_block == 3, "2052 fork includes three tokens from rejected block 0");
        }
        std::cout << "{\"diagnostic\":\"mx-expanded-position\",\"visible\":" << visible
                  << ",\"n_kv\":" << n_kv << ",\"official_count\":" << official.tokens.size()
                  << ",\"fork_width\":" << fork.topk_cells.size()
                  << ",\"fork_valid_count\":" << fork.valid_tokens.size()
                  << ",\"fork_extra_valid_ids\":" << extras << ",\"extra_token_ids\":[";
        for (std::size_t i = 0; i < extra_ids.size(); ++i) {
            if (i != 0) std::cout << ',';
            std::cout << extra_ids[i];
        }
        std::cout << "]}\n";
        for (const float value : {0.0f, 7.0f}) {
            const std::vector<float> equal(scores.size(), value);
            const auto legacy = fork_select(equal, visible, visible, n_kv, true);
            check(legacy.valid_tokens.size() == std::min(visible, std::size_t{2051}), "fork tied/zero valid count");
            for (std::size_t id = visible / 4 * 4; id < visible; ++id)
                check(std::find(legacy.valid_tokens.begin(), legacy.valid_tokens.end(),
                                static_cast<std::int32_t>(id)) != legacy.valid_tokens.end(), "fork forced tail");
        }
    }
    // A whole prefill chunk is loaded but each query has its own causal mask.
    const auto scores = increasing_scores(2056 / 4);
    for (std::size_t visible = 2047; visible <= 2056; ++visible) {
        const auto fork = fork_select(scores, 2056, visible, 2304, true);
        check(fork.valid_tokens.size() == std::min(visible, std::size_t{2051}), "fork chunk-prefix valid count");
        for (const auto id : fork.valid_tokens)
            check(id >= 0 && static_cast<std::size_t>(id) < visible, "fork final mask leaked future token");
    }
    // A tiny prefix can have a large top-k width with many masked slots: final
    // valid count is essential when diagnosing, even in the old expanded path.
    for (std::size_t visible = 1; visible <= 12; ++visible) {
        const auto fork = fork_select(increasing_scores(3), 12, visible, 256, true);
        check(fork.topk_cells.size() == 256 && fork.valid_tokens.size() == visible, "masked fork width vs actual count");
    }
}

}

int main() {
    try {
        selection_tests();
        causal_chunk_tests();
        invalid_tests();
        fork_diagnostic_tests();
        std::cout << "{\"test\":\"qsa\",\"checks\":" << checks << ",\"rejections\":" << rejections
                  << ",\"chunk_prefix_queries\":" << prefix_queries << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "qsa test: " << error.what() << '\n';
        return 1;
    }
}
