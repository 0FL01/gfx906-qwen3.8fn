// Separate diagnostic executable. Never add this translation unit to core targets.
// API/layout: mx dcd685463d597d31f5ca759d32c94592a2740fa4, include/llama.h.
#include "llama.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#if !defined(CORE_ORACLE_LIBRARY_REVISION) || !defined(CORE_ORACLE_LIBRARY_DIR)
#error "Build with tools/build-oracle.sh and revision-attested production libraries"
#endif

namespace {
namespace fs = std::filesystem;
constexpr std::string_view revision = "dcd685463d597d31f5ca759d32c94592a2740fa4";
constexpr std::string_view hf_revision = "a005fc82babfe8871d87746decad2dbee100a125";
constexpr float hf_gdn_l2_epsilon = 1.0e-6f;
constexpr size_t gdn_l2_nodes_per_token = 2 * 36;
constexpr size_t qsa_nodes_per_token = 12;
constexpr size_t qsa_max_visible = 2048;
constexpr size_t qsa_head_dim = 256;
constexpr size_t qsa_query_heads = 24;
constexpr size_t qsa_kv_heads = 2;
constexpr size_t qsa_output_elements = qsa_head_dim * qsa_query_heads;
constexpr size_t qsa_probe_row_limit = 8192;
constexpr size_t qsa_probe_manifest_limit = qsa_max_visible * qsa_nodes_per_token * qsa_probe_row_limit;
constexpr float qsa_coefficient = 0.0625f;
constexpr int32_t vocab_size = 248320;
constexpr uint32_t capacity = 4096;
constexpr size_t gdn_layer_count = 36;
constexpr size_t gdn_state_elements = 128 * 128 * 48;
constexpr size_t gdn_output_elements = 128 * 48;
constexpr size_t gdn_state_probe_row_limit = 8192;
constexpr size_t gdn_state_probe_manifest_limit = capacity * gdn_layer_count * gdn_state_probe_row_limit;
constexpr int32_t threads = 16;
constexpr size_t tensor_limit = 16 * 1024 * 1024;
constexpr size_t token_capture_limit = 64 * 1024 * 1024;
constexpr std::array<int, 4> layers{0, 1, 3, 47};
constexpr std::array<const char *, 6> required_names{
    "hc_init", "l_last-0", "l_last-1", "l_last-3", "l_last-47", "result_norm"};
// Exactly common/common.h::LLM_FFN_EXPS_REGEX / llm_ffn_exps_cpu_override().
constexpr const char * cpu_expert_pattern = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little);
static_assert(GGML_MAX_DIMS == 4);

std::string json_string(std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    result += '"';
    return result;
}

std::ofstream open_output(const fs::path & path, bool binary = false) {
    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(path, std::ios::out | std::ios::trunc | (binary ? std::ios::binary : std::ios::openmode{}));
    return stream;
}

std::set<std::string, std::less<>> allowlist(bool intermediates) {
    std::set<std::string, std::less<>> names(required_names.begin(), required_names.end());
    if (intermediates) {
        // qwen4exp.cpp::build_hc_mix/build_hc_combine and
        // llama-graph.cpp::build_moe_ffn. Repeated HC names are intentional:
        // attention/FFN use the same name; occurrences distinguish their nodes.
        constexpr const char * extra[] = {
            "hc_norm", "hc_gate", "hc_mixed", "hc_inject", "hc_combine",
            "ffn_moe_logits", "ffn_moe_probs", "ffn_moe_argsort", "ffn_moe_topk",
            "ffn_moe_weights", "ffn_moe_weights_softmax", "ffn_moe_weights_norm",
            "ffn_moe_weights_scaled", "ffn_moe_cache_slots", "ffn_moe_up",
            "ffn_moe_gate", "ffn_moe_swiglu", "ffn_moe_down", "ffn_moe_cache_up",
            "ffn_moe_cache_gate", "ffn_moe_cache_down", "ffn_moe_cache_merged",
            "ffn_moe_weighted", "ffn_moe_out", "ffn_shexp_gated", "ffn_out"};
        for (const int layer : {0, 1, 2, 3, 47}) {
            for (const auto * name : extra) {
                names.insert(std::string(name) + "-" + std::to_string(layer));
            }
        }
        names.insert("hc_norm"); // head mixer, il=-1: suffix is omitted
        names.insert("hc_gate"); // hc_mixed here is renamed to result_norm
        for (int layer : {0, 1, 2}) {
            for (const char* name : {"linear_attn_qkv_mixed", "z", "alpha", "beta", "conv_output_raw", "conv_output_silu",
                                     "q_conv_predelta", "k_conv_predelta", "v_conv_predelta", "final_output", "linear_attn_out"})
                names.insert(std::string(name) + "-" + std::to_string(layer));
        }
        for (const char* name : {"ple_embd", "ple_key", "ple_value", "ple_gated_value", "ple_conv_out"})
            names.insert(std::string(name) + "-1");
        names.insert("kqv_out-3");
        names.insert("kqv_out-47");
    }
    return names;
}

struct Options {
    fs::path model;
    fs::path output;
    bool intermediates = false;
    bool all_layers = false;
    std::set<int> gdn_probe_layers;
    std::set<int> qsa_probe_layers;
    bool hf_gdn_l2_control = false;
    bool hf_qsa_f32_control = false;
    int cache_warmup_passes = 0;
    int cache_inserts = 2;
    std::vector<llama_token> tokens;
};

void usage() {
    std::fprintf(stderr,
        "usage: core-oracle [--intermediates] [--all-layers] [--warm-cache PASSES] [--cache-inserts INT]\n"
        "                   [--gdn-probe-layer INT] [--qsa-probe-layer INT]\n"
        "                   [--hf-gdn-l2-control] [--hf-qsa-f32-control]\n"
        "                   model.gguf NEW-output-dir token-id...\n"
        "Feeds 1..4096 fixed IDs at consecutive positions 0..N-1, sequence 0.\n"
        "Writes all 248320 FP32 logits after every consumed token, without sampling.\n"
        "--intermediates adds exact HC/MoE names at layers 0,1,3,47 and checks cache activation.\n"
        "--all-layers instead records only hc_init, result_norm, all 48 layer outputs and cache slots.\n"
        "--gdn-probe-layer adds exact GDN/HC/final-FFN sites for one layer; repeatable, requires --all-layers.\n"
        "It also records original log-decay, sigmoid beta, PRE/POST state and raw recurrence output\n"
        "at attn_output, reading POST from the persistent cache destination; gdn-state-probes.jsonl.\n"
        "--qsa-probe-layer records original FLASH query/selected Q4 K/V and corrected FP32 output,\n"
        "before inverse H64, only for recorded tokens; repeatable unique layers 3,7,...,47,\n"
        "requires --hf-qsa-f32-control; writes exclusive artifacts and qsa-probes.jsonl.\n"
        "--cache-inserts accepts 1..112 (default 2); --warm-cache accepts 1..64 passes.\n"
        "--hf-gdn-l2-control explicitly corrects all 36 GDN Q/K L2 norms to HF additive epsilon,\n"
        "including warmup. This experimental semantic diagnostic is not a baseline-performance reference.\n"
        "--hf-qsa-f32-control independently recomputes all 12 masked QSA outputs on CPU before\n"
        "inverse Hadamard, including warmup, with Q4->FP16 RNE gather and FP32 normalized attention.\n"
        "It supports only positions 0..2047 and is not a baseline-performance reference.\n");
}

Options parse_options(int argc, char ** argv) {
    Options opts;
    int first = 1;
    while (first < argc) {
        const std::string_view arg(argv[first]);
        if (arg == "--intermediates") { opts.intermediates = true; ++first; }
        else if (arg == "--all-layers") { opts.all_layers = true; ++first; }
        else if (arg == "--hf-gdn-l2-control") { opts.hf_gdn_l2_control = true; ++first; }
        else if (arg == "--hf-qsa-f32-control") { opts.hf_qsa_f32_control = true; ++first; }
        else if (arg == "--warm-cache") {
            if (++first == argc) throw std::runtime_error("missing warm-cache passes");
            const std::string_view value(argv[first++]);
            const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), opts.cache_warmup_passes);
            if (ec != std::errc{} || end != value.data() + value.size() || opts.cache_warmup_passes < 1 || opts.cache_warmup_passes > 64)
                throw std::runtime_error("warm-cache passes must be 1..64");
        } else if (arg == "--gdn-probe-layer") {
            if (++first == argc) throw std::runtime_error("missing GDN probe layer");
            const std::string_view value(argv[first++]);
            int layer = -1;
            const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), layer);
            if (ec != std::errc{} || end != value.data() + value.size() || layer < 0 || layer >= 48 || layer % 4 == 3 ||
                    !opts.gdn_probe_layers.insert(layer).second)
                throw std::runtime_error("GDN probe requires a unique GDN layer in 0..47");
        } else if (arg == "--qsa-probe-layer") {
            if (++first == argc) throw std::runtime_error("missing QSA probe layer");
            const std::string_view value(argv[first++]);
            int layer = -1;
            const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), layer);
            if (ec != std::errc{} || end != value.data() + value.size() || layer < 3 || layer >= 48 || layer % 4 != 3 ||
                    !opts.qsa_probe_layers.insert(layer).second)
                throw std::runtime_error("QSA probe requires a unique QSA layer in 3,7,...,47");
        } else if (arg == "--cache-inserts") {
            if (++first == argc) throw std::runtime_error("missing cache-inserts value");
            const std::string_view value(argv[first++]);
            const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), opts.cache_inserts);
            if (ec != std::errc{} || end != value.data() + value.size() || opts.cache_inserts < 1 || opts.cache_inserts > 112)
                throw std::runtime_error("cache-inserts must be 1..112");
        } else break;
    }
    if (!opts.gdn_probe_layers.empty() && !opts.all_layers)
        throw std::runtime_error("GDN probes require --all-layers to avoid observing router normalization sites");
    if (!opts.qsa_probe_layers.empty() && !opts.hf_qsa_f32_control)
        throw std::runtime_error("QSA probes require --hf-qsa-f32-control");
    if (argc - first < 3) {
        throw std::runtime_error("expected a model, a new output directory and token IDs");
    }
    opts.model = argv[first++];
    opts.output = argv[first++];
    if (static_cast<uint32_t>(argc - first) > capacity) {
        throw std::runtime_error("token count exceeds fixed capacity 4096");
    }
    for (; first < argc; ++first) {
        const std::string_view arg(argv[first]);
        llama_token token = -1;
        const auto [end, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), token);
        if (ec != std::errc{} || end != arg.data() + arg.size() || token < 0 || token >= vocab_size) {
            throw std::runtime_error("each token ID must be an integer in [0,248320)");
        }
        opts.tokens.push_back(token);
    }
    if (!fs::is_regular_file(opts.model)) {
        throw std::runtime_error("model is not a regular file");
    }
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
    if (opts.hf_gdn_l2_control || opts.hf_qsa_f32_control) {
        throw std::runtime_error("HF diagnostic controls require compilation without fast-math");
    }
#endif
    return opts;
}

size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) {
        throw std::runtime_error("diagnostic storage span addition overflow");
    }
    return a + b;
}

size_t checked_multiply(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error("diagnostic storage span multiplication overflow");
    }
    return a * b;
}

struct L2Storage {
    size_t bytes;
    uintptr_t address;
};

L2Storage check_storage_allocation(const ggml_tensor * tensor, size_t span, std::string_view purpose) {
    // Same buffer-or-view-source contract as the public synchronous get/set API.
    auto * buffer = tensor->buffer ? tensor->buffer : (tensor->view_src ? tensor->view_src->buffer : nullptr);
    if (!buffer) {
        throw std::runtime_error(std::string(purpose) + " has no backend buffer");
    }
    const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const auto address = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t allocation = ggml_backend_buffer_get_size(buffer);
    if (base == 0 || address < base || address - base > allocation ||
        span > allocation - (address - base) || span > std::numeric_limits<uintptr_t>::max() - address) {
        throw std::runtime_error(std::string(purpose) + " byte range exceeds its backend allocation");
    }
    return {span, address};
}

L2Storage check_l2_storage(const ggml_tensor * tensor) {
    constexpr std::array<int64_t, 4> shape{128, 16, 1, 1};
    if (!tensor || tensor->type != GGML_TYPE_F32 || !tensor->data) {
        throw std::runtime_error("GDN L2 tensor/source must have available F32 data");
    }
    size_t span = sizeof(float);
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (tensor->ne[dim] != shape[dim] || tensor->nb[dim] < sizeof(float) ||
            tensor->nb[dim] % sizeof(float) != 0) {
            throw std::runtime_error("GDN L2 tensor/source must have ne=[128,16,1,1] and valid F32 strides");
        }
        span = checked_add(span, checked_multiply(static_cast<size_t>(shape[dim] - 1), tensor->nb[dim]));
    }
    // Positive, F32-aligned strides can be padded or permuted. Their smallest
    // collision has coordinate differences nb[1]/gcd and nb[0]/gcd; reject it
    // only if both differences fit inside this fixed 128x16 grid.
    const size_t stride_gcd = std::gcd(tensor->nb[0], tensor->nb[1]);
    if ((tensor->nb[1] / stride_gcd < 128 && tensor->nb[0] / stride_gcd < 16) ||
        span > tensor_limit || ggml_nbytes(tensor) != span) {
        throw std::runtime_error("GDN L2 tensor/source has overlapping elements or an invalid/bounded storage span");
    }
    return check_storage_allocation(tensor, span, "GDN L2 tensor/source");
}

L2Storage check_qsa_storage(const ggml_tensor * tensor, ggml_type type,
                          const std::array<int64_t, 4> & shape) {
    if (!tensor || tensor->type != type || !tensor->data ||
        (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_Q4_0)) {
        throw std::runtime_error("QSA tensor/source type or available data mismatch");
    }
    const size_t unit = type == GGML_TYPE_F32 ? sizeof(float) :
                        type == GGML_TYPE_F16 ? sizeof(ggml_fp16_t) : 18;
    const size_t alignment = type == GGML_TYPE_F32 ? alignof(float) : alignof(ggml_fp16_t);
    if (reinterpret_cast<uintptr_t>(tensor->data) % alignment != 0 ||
        (type == GGML_TYPE_Q4_0 && (ggml_blck_size(type) != 32 || ggml_type_size(type) != 18 ||
                                  tensor->nb[0] != 18 || shape[0] % 32 != 0))) {
        throw std::runtime_error("QSA tensor/source alignment or original Q4_0 block layout mismatch");
    }
    std::array<std::pair<size_t, size_t>, 4> dimensions{};
    size_t span = unit;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (shape[dim] <= 0 || shape[dim] > static_cast<int64_t>(qsa_output_elements) ||
            tensor->ne[dim] != shape[dim] || tensor->nb[dim] < unit || tensor->nb[dim] % alignment != 0) {
            throw std::runtime_error("QSA tensor/source dimensions or strides mismatch");
        }
        // Quantized nb[0] advances a BLOCK, not one of its 32 nibble values.
        // In particular a 256-wide row spans 8*18=144 bytes, not 256*18.
        const size_t count = static_cast<size_t>(shape[dim]) / (type == GGML_TYPE_Q4_0 && dim == 0 ? 32 : 1);
        dimensions[dim] = {tensor->nb[dim], count};
        span = checked_add(span, checked_multiply(count - 1, tensor->nb[dim]));
    }
    // Sorting active axes accepts both interleaved heads (nb1=288, nb2=144)
    // and capacity-strided head planes, while rejecting overlapping elements.
    std::sort(dimensions.begin(), dimensions.end());
    size_t occupied = unit;
    for (const auto & [stride, count] : dimensions) {
        if (count > 1) {
            if (stride < occupied) {
                throw std::runtime_error("QSA tensor/source has overlapping strided elements/blocks");
            }
            occupied = checked_add(occupied, checked_multiply(count - 1, stride));
        }
    }
    if (span > tensor_limit || ggml_nbytes(tensor) != span) {
        throw std::runtime_error("QSA tensor/source exceeds bounded storage or ggml_nbytes differs from checked span");
    }
    return check_storage_allocation(tensor, span, "QSA tensor/source");
}

bool overlaps(const L2Storage & a, const L2Storage & b) {
    // Both ranges have already passed the pointer-end overflow check.
    return a.address < b.address + b.bytes && b.address < a.address + a.bytes;
}

size_t f32_flat_offset(const ggml_tensor * tensor, size_t index) {
    size_t offset = 0;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        const size_t count = static_cast<size_t>(tensor->ne[dim]);
        offset = checked_add(offset, checked_multiply(index % count, tensor->nb[dim]));
        index /= count;
    }
    if (index != 0) throw std::runtime_error("QSA flattened output index exceeds tensor geometry");
    return offset;
}

struct QsaSite {
    const ggml_tensor * flash;
    std::vector<const ggml_tensor *> wrappers; // target toward FLASH, excluding FLASH
    std::vector<L2Storage> wrapper_storage;
    L2Storage destination;
    L2Storage flash_storage;
};

QsaSite check_qsa_site(const ggml_tensor * target) {
    QsaSite site{};
    site.destination = check_qsa_storage(target, GGML_TYPE_F32, {6144, 1, 1, 1});
    const ggml_tensor * node = target;
    std::set<const ggml_tensor *> visited;
    while (node && node->op != GGML_OP_FLASH_ATTN_EXT) {
        if (site.wrappers.size() == 8 || !visited.insert(node).second || !node->src[0] ||
            (node->op != GGML_OP_VIEW && node->op != GGML_OP_RESHAPE &&
             node->op != GGML_OP_PERMUTE && node->op != GGML_OP_CONT)) {
            throw std::runtime_error("QSA kqv_out must reach one FLASH_ATTN_EXT through at most 8 layout-only nodes");
        }
        for (size_t i = 1; i < GGML_MAX_SRC; ++i) {
            if (node->src[i]) throw std::runtime_error("QSA layout wrapper has an unexpected second source");
        }
        std::array<int64_t, 4> shape{};
        size_t elements = 1;
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            shape[dim] = node->ne[dim];
            if (shape[dim] <= 0) throw std::runtime_error("QSA wrapper has a nonpositive dimension");
            elements = checked_multiply(elements, static_cast<size_t>(shape[dim]));
        }
        if (elements != qsa_output_elements) throw std::runtime_error("QSA layout wrapper changes element count");
        site.wrapper_storage.push_back(check_qsa_storage(node, GGML_TYPE_F32, shape));
        site.wrappers.push_back(node);
        node = node->src[0];
    }
    if (!node || !visited.insert(node).second) throw std::runtime_error("QSA FLASH ancestor is missing/cyclic");
    site.flash = node;
    site.flash_storage = check_qsa_storage(node, GGML_TYPE_F32, {256, 24, 1, 1});
    if (!ggml_is_contiguous(node)) throw std::runtime_error("QSA FLASH output must be contiguous head-major F32");

    // Prove the final flattened element j still denotes FLASH head j/256,
    // channel j%256. Layout names alone do not prove this (a permutation may
    // reorder real axes). Carry the exact logical-to-FLASH map through the chain.
    std::vector<size_t> mapping(qsa_output_elements);
    std::iota(mapping.begin(), mapping.end(), 0);
    const ggml_tensor * source = node;
    for (auto it = site.wrappers.rbegin(); it != site.wrappers.rend(); ++it) {
        const auto * wrapper = *it;
        std::vector<size_t> next(qsa_output_elements);
        if (wrapper->op == GGML_OP_CONT || wrapper->op == GGML_OP_RESHAPE) {
            if (wrapper->op == GGML_OP_RESHAPE &&
                (!ggml_is_contiguous(source) || !ggml_is_contiguous(wrapper))) {
                throw std::runtime_error("QSA reshape is not a contiguous, order-preserving reinterpretation");
            }
            next = mapping; // CONT copies the source's logical flattened order.
        } else {
            size_t view_offset = 0;
            if (wrapper->op == GGML_OP_VIEW) {
                static_assert(sizeof(view_offset) <= sizeof(wrapper->op_params));
                std::memcpy(&view_offset, wrapper->op_params, sizeof(view_offset));
            }
            std::map<size_t, size_t> source_indices;
            for (size_t j = 0; j < qsa_output_elements; ++j) {
                if (!source_indices.emplace(f32_flat_offset(source, j), j).second) {
                    throw std::runtime_error("QSA layout source has duplicate physical F32 elements");
                }
            }
            for (size_t j = 0; j < qsa_output_elements; ++j) {
                const auto found = source_indices.find(checked_add(view_offset, f32_flat_offset(wrapper, j)));
                if (found == source_indices.end()) throw std::runtime_error("QSA view/permutation escapes source elements");
                next[j] = mapping[found->second];
            }
        }
        mapping.swap(next);
        source = wrapper;
    }
    for (size_t j = 0; j < mapping.size(); ++j) {
        if (mapping[j] != j) throw std::runtime_error("QSA kqv_out flattening does not preserve FLASH head/channel order");
    }
    return site;
}

float qsa_finite(float value) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite FP32 QSA control arithmetic");
    return value;
}

float qsa_multiply(float a, float b) {
    const volatile float value = a * b;
    return qsa_finite(value);
}

float qsa_add(float a, float b) {
    const volatile float value = a + b;
    return qsa_finite(value);
}

float qsa_decode_q4(const std::vector<char> & raw, const ggml_tensor * tensor,
                    size_t position, size_t head, size_t channel) {
    const size_t offset = position * tensor->nb[1] + head * tensor->nb[2] + (channel / 32) * tensor->nb[0];
    ggml_fp16_t scale_bits{};
    std::memcpy(&scale_bits, raw.data() + offset, sizeof(scale_bits));
    const float scale = ggml_fp16_to_fp32(scale_bits);
    if (!std::isfinite(scale)) throw std::runtime_error("non-finite selected original QSA Q4_0 scale");
    const size_t lane = channel % 32;
    const auto packed = static_cast<unsigned char>(raw[offset + sizeof(scale_bits) + lane % 16]);
    const int code = lane < 16 ? packed & 15 : packed >> 4;
    // Decode the original nibble representation independently. Half gather is
    // applied to BOTH K and V, not to probabilities or the V accumulation.
    const float value = qsa_multiply(scale, static_cast<float>(code - 8));
    const auto half = ggml_fp32_to_fp16(value);
    if ((half & 0x7c00) == 0x7c00) throw std::runtime_error("selected QSA Q4->FP16 RNE gather overflow");
    return qsa_finite(ggml_fp16_to_fp32(half)); // finite zero/negative Q4 scales are valid
}

void write_tensor_description(std::ostream & out, const ggml_tensor * tensor, const L2Storage & storage) {
    out << "{\"name\":" << json_string(tensor->name)
        << ",\"type\":" << json_string(ggml_type_name(tensor->type))
        << ",\"type_id\":" << static_cast<int>(tensor->type)
        << ",\"op\":" << json_string(ggml_op_name(tensor->op))
        << ",\"op_id\":" << static_cast<int>(tensor->op) << ",\"ne\":[";
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) out << (dim ? "," : "") << tensor->ne[dim];
    out << "],\"nb\":[";
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) out << (dim ? "," : "") << tensor->nb[dim];
    out << "],\"bytes\":" << storage.bytes << ",\"view_offset\":" << tensor->view_offs << '}';
}

// open_output() intentionally retains the existing oracle behavior. Probes use
// the exclusive POSIX writer pattern from attention_oracle.cpp::write_output:
// no truncation, symlink following or linking; EINTR/short writes are checked.
struct ProbeFile {
    int fd = -1;
    size_t bytes = 0;

    explicit ProbeFile(const fs::path & path)
        : fd(::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0644)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "create exclusive QSA probe " + path.string());
    }
    ProbeFile(const ProbeFile &) = delete;
    ProbeFile & operator=(const ProbeFile &) = delete;
    ~ProbeFile() {
        if (fd >= 0 && ::close(fd) != 0) std::fprintf(stderr, "core-oracle: QSA probe cleanup close failed\n");
    }

    void check_size() const {
        struct stat status{};
        if (::fstat(fd, &status) != 0) throw std::system_error(errno, std::generic_category(), "fstat QSA probe");
        if (!S_ISREG(status.st_mode) || status.st_nlink != 1 || status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) != bytes)
            throw std::runtime_error("QSA probe must be a single-link regular file with the exact written byte count");
    }

    void append(const void * data, size_t count, size_t limit) {
        const size_t expected = checked_add(bytes, count);
        if (!data || count == 0 || expected > limit) throw std::runtime_error("QSA probe write exceeds its byte bound");
        check_size();
        const auto * input = static_cast<const char *>(data);
        size_t offset = 0;
        while (offset < count) {
            const auto written = ::write(fd, input + offset, count - offset);
            if (written < 0) {
                if (errno == EINTR) continue;
                throw std::system_error(errno, std::generic_category(), "write QSA probe");
            }
            if (written == 0) throw std::runtime_error("zero-byte write of QSA probe");
            offset += static_cast<size_t>(written);
        }
        bytes = expected;
        check_size();
    }

    void close_checked() {
        check_size();
        const int descriptor = std::exchange(fd, -1);
        if (::close(descriptor) != 0) throw std::system_error(errno, std::generic_category(), "close QSA probe");
    }
};

void write_probe_blob(const fs::path & path, const void * data, size_t bytes) {
    ProbeFile file(path);
    file.append(data, bytes, tensor_limit);
    file.close_checked();
}

L2Storage check_gdn_state_storage(const ggml_tensor * tensor, const std::array<int64_t, 4> & shape,
                                const std::array<size_t, 4> & strides, std::string_view purpose,
                                size_t byte_limit = tensor_limit) {
    if (!tensor || tensor->type != GGML_TYPE_F32 || !tensor->data ||
        reinterpret_cast<uintptr_t>(tensor->data) % alignof(float) != 0)
        throw std::runtime_error(std::string(purpose) + " must have available, aligned F32 storage");
    size_t span = sizeof(float);
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
        if (shape[dim] <= 0 || tensor->ne[dim] != shape[dim] || tensor->nb[dim] != strides[dim] ||
            strides[dim] < sizeof(float) || strides[dim] % sizeof(float) != 0)
            throw std::runtime_error(std::string(purpose) + " geometry/strides differ from the pinned one-token primitive");
        span = checked_add(span, checked_multiply(static_cast<size_t>(shape[dim] - 1), strides[dim]));
    }
    if (span > byte_limit || ggml_nbytes(tensor) != span)
        throw std::runtime_error(std::string(purpose) + " storage span is invalid or exceeds its bound");
    return check_storage_allocation(tensor, span, purpose);
}

void check_gdn_state_view(const ggml_tensor * view, const ggml_tensor * primitive, size_t offset) {
    if (!view || view->op != GGML_OP_VIEW || view->src[0] != primitive || view->view_src != primitive ||
        view->view_offs != offset)
        throw std::runtime_error("GDN state probe requires a direct, correctly offset VIEW of the same primitive");
    for (size_t i = 1; i < GGML_MAX_SRC; ++i) {
        if (view->src[i]) throw std::runtime_error("GDN state/output VIEW has an unexpected extra source");
    }
    size_t parameter_offset = 0;
    static_assert(sizeof(parameter_offset) <= sizeof(view->op_params));
    std::memcpy(&parameter_offset, view->op_params, sizeof(parameter_offset));
    if (parameter_offset != offset)
        throw std::runtime_error("GDN state/output VIEW parameter offset disagrees with its storage offset");
}

std::vector<char> copy_gdn_state_f32(const ggml_tensor * tensor, const L2Storage & storage, size_t elements) {
    const size_t bytes = checked_multiply(elements, sizeof(float));
    if (bytes == 0 || bytes > tensor_limit || storage.bytes < sizeof(float))
        throw std::runtime_error("GDN state probe canonical copy exceeds its byte bound");
    size_t actual_elements = 1;
    for (int dim = 0; dim < GGML_MAX_DIMS; ++dim)
        actual_elements = checked_multiply(actual_elements, static_cast<size_t>(tensor->ne[dim]));
    if (actual_elements != elements) throw std::runtime_error("GDN state probe canonical element count mismatch");
    std::vector<char> raw(storage.bytes), packed(bytes);
    ggml_backend_tensor_get(tensor, raw.data(), 0, raw.size());
    for (size_t index = 0; index < elements; ++index) {
        const size_t offset = f32_flat_offset(tensor, index);
        if (offset > storage.bytes - sizeof(float)) throw std::runtime_error("GDN state probe copy escapes its live storage");
        float value{};
        std::memcpy(&value, raw.data() + offset, sizeof(value));
        if (!std::isfinite(value)) throw std::runtime_error("non-finite original GDN control/state/output probe value");
        // Copy original bits, including signed zero. No arithmetic or normalization.
        std::memcpy(packed.data() + index * sizeof(float), raw.data() + offset, sizeof(float));
    }
    return packed;
}

struct GdnStateSite {
    const ggml_tensor * primitive = nullptr;
    const ggml_tensor * new_state = nullptr;
    const ggml_tensor * copy = nullptr;
    const ggml_tensor * destination = nullptr;
    size_t primitive_node = 0;
    size_t new_state_node = 0;
    size_t copy_node = 0;
};

struct Capture {
    fs::path output;
    std::set<std::string, std::less<>> names;
    std::ofstream records;
    std::ofstream control_records;
    std::ofstream qsa_control_records;
    std::unique_ptr<ProbeFile> qsa_probe_records;
    std::set<int> qsa_probe_layers;
    std::array<bool, qsa_nodes_per_token> qsa_probe_seen{};
    std::array<size_t, qsa_nodes_per_token> qsa_probe_totals{};
    size_t qsa_probe_count = 0;
    size_t qsa_probe_rows = 0;
    size_t qsa_probe_tokens_checked = 0;
    std::unique_ptr<ProbeFile> gdn_state_probe_records;
    std::set<int> gdn_state_probe_layers;
    std::map<std::string, size_t, std::less<>> gdn_state_output_names;
    std::map<std::string, size_t, std::less<>> gdn_state_view_names;
    std::map<const ggml_tensor *, size_t> gdn_primitive_nodes;
    std::map<const ggml_tensor *, size_t> gdn_state_tensor_ids;
    std::array<GdnStateSite, 48> gdn_state_sites{};
    std::array<bool, 48> gdn_state_probe_seen{};
    std::array<size_t, 48> gdn_state_probe_totals{};
    size_t gdn_state_probe_count = 0;
    size_t gdn_state_probe_rows = 0;
    size_t gdn_state_probe_tokens_checked = 0;
    std::vector<char> buffer;
    std::vector<char> control_source;
    std::vector<char> control_destination;
    std::map<std::string, size_t, std::less<>> occurrences;
    std::map<std::string, size_t, std::less<>> totals;
    int32_t token_index = -1;
    llama_token token_id = -1;
    // Warmup has token_index=-1 (no tensor artifacts), but it still has a real
    // evaluation position, input ID and monotonically increasing epoch.
    int32_t evaluation_position = -1;
    bool evaluation_active = false;
    int warm_pass = -1;
    size_t evaluation_epoch = 0;
    bool hf_gdn_l2_control = false;
    std::map<std::string, size_t, std::less<>> control_names;
    std::array<bool, 48 * 2> control_seen{};
    size_t control_count = 0;
    size_t recorded_control_count = 0;
    size_t warm_control_count = 0;
    size_t recorded_control_tokens_checked = 0;
    size_t warm_control_tokens_checked = 0;
    std::vector<size_t> warm_pass_control_counts;
    bool hf_qsa_f32_control = false;
    bool all_layers = false;
    std::map<std::string, size_t, std::less<>> probe_expected;
    std::map<std::string, size_t, std::less<>> qsa_control_names;
    std::array<bool, qsa_nodes_per_token> qsa_control_seen{};
    size_t qsa_control_count = 0;
    size_t recorded_qsa_control_count = 0;
    size_t warm_qsa_control_count = 0;
    size_t recorded_qsa_control_tokens_checked = 0;
    size_t warm_qsa_control_tokens_checked = 0;
    std::vector<size_t> warm_pass_qsa_control_counts;
    size_t ask_index = 0;
    size_t pending_node = 0;
    size_t token_bytes = 0;
    size_t captures = 0;
    std::array<char, 512> error{};

    Capture(const fs::path & dir, bool intermediates, bool control, bool qsa_control, bool all,
            const std::set<int>& probes, const std::set<int>& qsa_probes)
        : output(dir), names(allowlist(intermediates)), records(open_output(dir / "tensors.jsonl")),
          qsa_probe_layers(qsa_probes), gdn_state_probe_layers(probes),
          hf_gdn_l2_control(control), hf_qsa_f32_control(qsa_control), all_layers(all) {
        if (all) {
            // Keep the diagnostic manifest bounded: 98 observations/token, not
            // every intermediate of every layer. Controls remain independent.
            names = {"hc_init", "result_norm"};
            for (int layer = 0; layer < 48; ++layer) {
                names.emplace("l_last-" + std::to_string(layer));
                names.emplace("ffn_moe_cache_slots-" + std::to_string(layer));
            }
        }
        for (int layer : probes) {
            const auto suffix = "-" + std::to_string(layer);
            gdn_state_output_names.emplace("attn_output" + suffix, static_cast<size_t>(layer));
            gdn_state_view_names.emplace("new_state" + suffix, static_cast<size_t>(layer));
            // This is the only new callback stop: controls/PRE/POST are read
            // through ancestry after the unobserved cache CPY has completed.
            probe_expected.emplace("attn_output" + suffix, 1);
            for (const char* name : {"linear_attn_qkv_mixed", "z", "alpha", "beta", "conv_output_raw", "conv_output_silu",
                                     "q_conv_predelta", "k_conv_predelta", "v_conv_predelta", "final_output", "linear_attn_out", "ffn_out"})
                probe_expected.emplace(std::string(name) + suffix, 1);
            for (const char* name : {"hc_norm", "hc_gate", "hc_mixed", "hc_inject"})
                probe_expected.emplace(std::string(name) + suffix, 2);
        }
        if (all) {
            // Bounded post-attention observations on explicitly requested QSA
            // probe layers only. No router probability/normalization captures:
            // keep the routing-fusion observation contract unchanged.
            for (int layer : qsa_probes) {
                const auto suffix = "-" + std::to_string(layer);
                for (const char* name : {"Qcur_full", "gate_reshaped", "attn_pregate",
                                        "gate_sigmoid", "attn_gated", "attn_output", "ffn_out"})
                    probe_expected.emplace(std::string(name) + suffix, 1);
                for (const char* name : {"hc_norm", "hc_gate", "hc_mixed", "hc_inject"})
                    probe_expected.emplace(std::string(name) + suffix, 2);
            }
        }
        for (const auto& [name, count] : probe_expected) {
            (void) count;
            names.emplace(name);
        }
        if (control) {
            control_records = open_output(dir / "gdn-l2-control.jsonl");
            for (int layer = 0; layer < 48; ++layer) {
                if (layer % 4 != 3) {
                    control_names.emplace("q_conv_predelta-" + std::to_string(layer), 2 * layer);
                    control_names.emplace("k_conv_predelta-" + std::to_string(layer), 2 * layer + 1);
                }
            }
        }
        if (qsa_control) {
            qsa_control_records = open_output(dir / "qsa-f32-control.jsonl");
            qsa_control_records << std::setprecision(std::numeric_limits<double>::max_digits10);
            for (int layer = 3; layer < 48; layer += 4) {
                qsa_control_names.emplace("kqv_out-" + std::to_string(layer), static_cast<size_t>((layer - 3) / 4));
            }
        }
        if (!qsa_probe_layers.empty()) {
            if (!qsa_control) throw std::runtime_error("QSA probes require the existing independent FP32 control");
            qsa_probe_records = std::make_unique<ProbeFile>(dir / "qsa-probes.jsonl");
        }
    }

    void begin(int32_t index, llama_token id, int pass = -1) {
        if (evaluation_active) {
            throw std::runtime_error("previous oracle evaluation was not validated and finished");
        }
        if (hf_qsa_f32_control && (index < 0 || static_cast<size_t>(index) >= qsa_max_visible)) {
            throw std::runtime_error("HF QSA F32 control supports only positions 0..2047 (max_control_visible=2048)");
        }
        token_index = pass < 0 ? index : -1;
        token_id = id;
        evaluation_position = index;
        evaluation_active = true;
        warm_pass = pass;
        ++evaluation_epoch;
        control_seen.fill(false);
        control_count = 0;
        qsa_control_seen.fill(false);
        qsa_control_count = 0;
        qsa_probe_seen.fill(false);
        qsa_probe_count = 0;
        gdn_state_probe_seen.fill(false);
        gdn_state_probe_count = 0;
        gdn_state_sites.fill(GdnStateSite{});
        gdn_primitive_nodes.clear();
        gdn_state_tensor_ids.clear();
        ask_index = 0;
        token_bytes = 0;
        occurrences.clear();
    }

    void open_gdn_state_probes() {
        if (gdn_state_probe_layers.empty()) return;
        if (gdn_state_probe_records) throw std::runtime_error("GDN state probe manifest was already opened");
        // Called after Capture exists and initializing metadata was written, so
        // even an exclusive-create failure is reported as failed metadata.
        gdn_state_probe_records = std::make_unique<ProbeFile>(output / "gdn-state-probes.jsonl");
    }

    void scan_gdn_state_node(const ggml_tensor * tensor) {
        if (token_index < 0 || warm_pass >= 0 || gdn_state_probe_layers.empty()) return;
        if (tensor->op == GGML_OP_GATED_DELTA_NET) {
            if (gdn_primitive_nodes.size() >= gdn_layer_count ||
                !gdn_primitive_nodes.emplace(tensor, pending_node).second)
                throw std::runtime_error("duplicate or excessive one-token GDN primitive in probe graph scan");
        }
        const auto view = gdn_state_view_names.find(tensor->name);
        if (view != gdn_state_view_names.end()) {
            auto & site = gdn_state_sites[view->second];
            if (site.new_state || tensor->op != GGML_OP_VIEW || !tensor->src[0] ||
                tensor->src[0]->op != GGML_OP_GATED_DELTA_NET)
                throw std::runtime_error("duplicate or unexpected requested new_state GDN VIEW");
            check_gdn_state_view(tensor, tensor->src[0], gdn_output_elements * sizeof(float));
            const auto primitive = gdn_primitive_nodes.find(tensor->src[0]);
            if (primitive == gdn_primitive_nodes.end() || primitive->second >= pending_node)
                throw std::runtime_error("requested new_state VIEW lacks a preceding scanned GDN primitive");
            site.primitive = tensor->src[0];
            site.primitive_node = primitive->second;
            site.new_state = tensor;
            site.new_state_node = pending_node;
        }
        if (tensor->op == GGML_OP_CPY && tensor->src[0]) {
            const auto source = gdn_state_view_names.find(tensor->src[0]->name);
            if (source != gdn_state_view_names.end()) {
                auto & site = gdn_state_sites[source->second];
                if (site.copy || !site.new_state || tensor->src[0] != site.new_state ||
                    site.new_state->src[0] != site.primitive || !tensor->src[1] ||
                    site.new_state_node >= pending_node)
                    throw std::runtime_error("duplicate or unexpected requested GDN state cache CPY");
                for (size_t i = 2; i < GGML_MAX_SRC; ++i) {
                    if (tensor->src[i]) throw std::runtime_error("GDN cache CPY has an unexpected extra source");
                }
                site.copy = tensor;
                site.destination = tensor->src[1];
                site.copy_node = pending_node;
            }
        }
        // This scan NEVER selects new_state or CPY. Inserting a stop there can
        // disable GDN->cache-CPY fusion or expose an unwritten primitive tail.
    }

    size_t gdn_state_tensor_identity(const ggml_tensor * tensor) {
        if (!tensor) throw std::runtime_error("missing tensor identity in GDN state probe");
        const auto found = gdn_state_tensor_ids.find(tensor);
        if (found != gdn_state_tensor_ids.end()) return found->second;
        if (gdn_state_tensor_ids.size() >= gdn_layer_count * 16)
            throw std::runtime_error("GDN state probe tensor identity table exceeds its bound");
        const size_t identity = gdn_state_tensor_ids.size();
        gdn_state_tensor_ids.emplace(tensor, identity);
        return identity;
    }

    void describe_gdn_state_tensor(std::ostream & out, const ggml_tensor * tensor, const L2Storage & storage) {
        out << "{\"tensor_identity\":" << gdn_state_tensor_identity(tensor) << ",\"live_tensor\":";
        write_tensor_description(out, tensor, storage);
        out << ",\"view_source_identity\":";
        if (tensor->view_src) out << gdn_state_tensor_identity(tensor->view_src);
        else out << "null";
        out << '}';
    }

    void save_gdn_state_probe(const ggml_tensor * tensor) {
        const auto found = gdn_state_output_names.find(tensor->name);
        if (!evaluation_active || warm_pass >= 0 || token_index < 0 || evaluation_position != token_index ||
            found == gdn_state_output_names.end() || !gdn_state_probe_records)
            throw std::runtime_error("invalid or warm GDN state probe observation");
        const size_t layer = found->second;
        const auto & site = gdn_state_sites[layer];
        if (gdn_state_probe_seen[layer] || !site.primitive || !site.new_state || !site.copy || !site.destination ||
            !(site.primitive_node < site.new_state_node && site.new_state_node < site.copy_node && site.copy_node < pending_node))
            throw std::runtime_error("duplicate GDN state probe or missing/incorrectly ordered primitive/new_state/cache CPY");
        check_gdn_state_view(tensor, site.primitive, 0);
        check_gdn_state_view(site.new_state, site.primitive, gdn_output_elements * sizeof(float));
        if (site.primitive->op != GGML_OP_GATED_DELTA_NET || site.copy->op != GGML_OP_CPY ||
            site.copy->src[0] != site.new_state || site.copy->src[1] != site.destination)
            throw std::runtime_error("live GDN primitive/cache-copy ancestry changed after graph scan");
        for (size_t i = 2; i < GGML_MAX_SRC; ++i) {
            if (site.copy->src[i]) throw std::runtime_error("live GDN cache CPY has an unexpected extra source");
        }
        int32_t snapshots = 0;
        std::memcpy(&snapshots, site.primitive->op_params, sizeof(snapshots));
        if (snapshots != 1) throw std::runtime_error("GDN state probe requires one primitive snapshot slot");
        for (size_t i = 6; i < GGML_MAX_SRC; ++i) {
            if (site.primitive->src[i]) throw std::runtime_error("GDN primitive has an unexpected extra source");
        }
        const std::array<const ggml_tensor *, 6> sources{
            site.primitive->src[0], site.primitive->src[1], site.primitive->src[2],
            site.primitive->src[3], site.primitive->src[4], site.primitive->src[5]};
        constexpr size_t state_bytes = gdn_state_elements * sizeof(float);
        constexpr size_t output_bytes = gdn_output_elements * sizeof(float);
        constexpr size_t primitive_bytes = state_bytes + output_bytes;
        const std::array<L2Storage, 6> storage{
            check_gdn_state_storage(sources[0], {128,16,1,1}, {4,512,8192,8192}, "GDN primitive Q"),
            check_gdn_state_storage(sources[1], {128,16,1,1}, {4,512,8192,8192}, "GDN primitive K"),
            check_gdn_state_storage(sources[2], {128,48,1,1}, {4,512,40960,40960}, "GDN primitive V"),
            check_gdn_state_storage(sources[3], {1,48,1,1}, {4,4,192,192}, "GDN primitive log-decay"),
            check_gdn_state_storage(sources[4], {1,48,1,1}, {4,4,192,192}, "GDN primitive sigmoid beta"),
            check_gdn_state_storage(sources[5], {128,128,48,1}, {4,512,65536,state_bytes}, "GDN primitive PRE-state")};
        const auto primitive = check_gdn_state_storage(site.primitive, {6144,129,1,1},
            {4,output_bytes,primitive_bytes,primitive_bytes}, "GDN packed primitive");
        const auto output_storage = check_gdn_state_storage(tensor, {128,48,1,1},
            {4,512,output_bytes,output_bytes}, "GDN raw recurrence output");
        const auto new_state = check_gdn_state_storage(site.new_state, {128,128,48,1},
            {4,512,65536,state_bytes}, "GDN new_state VIEW (not read)");
        const auto destination = check_gdn_state_storage(site.destination, {786432,1,1,1},
            {4,state_bytes,state_bytes,state_bytes}, "GDN persistent POST-state cache destination");
        const auto copy = check_gdn_state_storage(site.copy, {786432,1,1,1},
            {4,state_bytes,state_bytes,state_bytes}, "GDN cache CPY destination alias");
        const auto * cache = site.destination->src[0];
        if (!cache || cache->op != GGML_OP_NONE || cache->type != GGML_TYPE_F32 || !cache->data ||
            reinterpret_cast<uintptr_t>(cache->data) % alignof(float) != 0 ||
            cache->ne[0] != static_cast<int64_t>(gdn_state_elements) || cache->ne[1] < 1 || cache->ne[1] > capacity ||
            cache->ne[2] != 1 || cache->ne[3] != 1 || site.destination->view_offs % state_bytes != 0)
            throw std::runtime_error("GDN POST-state CPY destination must be a view of the persistent F32 cache leaf");
        const size_t cache_bytes = checked_multiply(state_bytes, static_cast<size_t>(cache->ne[1]));
        // Validate the cache owner's complete allocation without copying it.
        // Only its one-state destination is read under the normal tensor bound.
        const auto cache_storage = check_gdn_state_storage(cache, {786432,cache->ne[1],1,1},
            {4,state_bytes,cache_bytes,cache_bytes}, "GDN persistent state cache leaf", checked_multiply(state_bytes, capacity));
        if (site.destination->view_offs > cache_storage.bytes ||
            destination.bytes > cache_storage.bytes - site.destination->view_offs)
            throw std::runtime_error("GDN POST-state VIEW escapes the persistent cache leaf's logical storage");
        check_gdn_state_view(site.destination, cache, site.destination->view_offs);
        const auto cache_address = reinterpret_cast<uintptr_t>(cache->data);
        if (site.destination->view_offs > std::numeric_limits<uintptr_t>::max() - cache_address ||
            destination.address != cache_address + site.destination->view_offs || copy.address != destination.address ||
            copy.bytes != destination.bytes || output_storage.address != primitive.address ||
            new_state.address != primitive.address + output_bytes)
            throw std::runtime_error("GDN live output/state/cache VIEW or CPY byte identity mismatch");
        if (overlaps(primitive, destination)) throw std::runtime_error("GDN persistent POST cache aliases the primitive output/tail");
        for (const auto & source : storage) {
            if (overlaps(source, primitive) || overlaps(source, destination))
                throw std::runtime_error("GDN raw primitive input/PRE-state aliases produced output or POST-state cache");
        }
        const std::array<L2Storage, 5> captured_storage{storage[3], storage[4], storage[5], destination, output_storage};
        for (size_t i = 0; i < captured_storage.size(); ++i) {
            for (size_t j = 0; j < i; ++j) {
                if (overlaps(captured_storage[i], captured_storage[j]))
                    throw std::runtime_error("GDN state probe controls/PRE/POST/raw-output storage overlaps");
            }
        }
        // The scheduler has synchronized the producing backend. CPY preceded
        // this selected raw VIEW, so read its persistent destination, NEVER
        // the primitive's possibly unwritten state tail/new_state VIEW.
        const std::array<std::vector<char>, 5> blobs{
            copy_gdn_state_f32(sources[3], storage[3], 48),
            copy_gdn_state_f32(sources[4], storage[4], 48),
            copy_gdn_state_f32(sources[5], storage[5], gdn_state_elements),
            copy_gdn_state_f32(site.destination, destination, gdn_state_elements),
            copy_gdn_state_f32(tensor, output_storage, gdn_output_elements)};
        std::array<char, 96> stem{};
        const int length = std::snprintf(stem.data(), stem.size(), "gdn-state-t%04d-e%06zu-l%02zu",
            token_index, evaluation_epoch, layer);
        if (length < 0 || static_cast<size_t>(length) >= stem.size()) throw std::runtime_error("GDN state probe filename overflow");
        const std::array<const char *, 5> roles{"log_decay", "beta", "pre_state", "post_state", "raw_output"};
        const std::array<const char *, 5> origins{
            "GATED_DELTA_NET.src[3]", "GATED_DELTA_NET.src[4]", "GATED_DELTA_NET.src[5]",
            "retained cache CPY.src[1]", "attn_output VIEW of GATED_DELTA_NET"};
        std::array<std::string, 5> paths{};
        for (size_t i = 0; i < paths.size(); ++i) paths[i] = std::string(stem.data()) + "." + roles[i] + ".f32.bin";
        std::ostringstream row;
        row.exceptions(std::ios::badbit | std::ios::failbit);
        row << "{\"format\":\"gfx906-mx-gdn-state-probe-v1\",\"layer\":" << layer
            << ",\"evaluation_epoch\":" << evaluation_epoch << ",\"phase\":\"recorded\",\"warm_pass\":-1"
            << ",\"token_index\":" << token_index << ",\"position\":" << evaluation_position
            << ",\"seq_id\":0,\"input_token_id\":" << token_id << ",\"node_index\":" << pending_node << ",\"occurrence\":0"
            << ",\"primitive_node_index\":" << site.primitive_node << ",\"new_state_node_index\":" << site.new_state_node
            << ",\"cache_copy_node_index\":" << site.copy_node << ",\"tensor_identity_scope\":\"evaluation_epoch\""
            << ",\"read_only_observation\":true,\"sources_unchanged\":true,\"controls_unchanged\":true"
            << ",\"post_from_persistent_cache_destination\":true,\"primitive_state_tail_read\":false"
            << ",\"new_state_callback_selected\":false,\"cache_copy_callback_selected\":false"
            << ",\"decay_materialized\":false,\"log_decay_semantics\":\"original g; exp(g) is internal to the primitive\""
            << ",\"production_source_revision\":" << json_string(revision)
            << ",\"library_revision_attested\":" << json_string(CORE_ORACLE_LIBRARY_REVISION) << ",\"primitive\":";
        describe_gdn_state_tensor(row, site.primitive, primitive);
        row << ",\"raw_output\":";
        describe_gdn_state_tensor(row, tensor, output_storage);
        row << ",\"new_state_view_not_read\":";
        describe_gdn_state_tensor(row, site.new_state, new_state);
        row << ",\"cache_copy\":";
        describe_gdn_state_tensor(row, site.copy, copy);
        row << ",\"cache_copy_source0_identity\":" << gdn_state_tensor_identity(site.new_state)
            << ",\"cache_copy_source1_identity\":" << gdn_state_tensor_identity(site.destination) << ",\"post_destination\":";
        describe_gdn_state_tensor(row, site.destination, destination);
        row << ",\"persistent_cache_leaf_not_read\":";
        describe_gdn_state_tensor(row, cache, cache_storage);
        row << ",\"sources\":[";
        for (size_t i = 0; i < sources.size(); ++i) {
            row << (i ? "," : "") << "{\"source_index\":" << i << ",\"source\":";
            describe_gdn_state_tensor(row, sources[i], storage[i]);
            row << '}';
        }
        row << "],\"artifacts\":{";
        for (size_t i = 0; i < paths.size(); ++i) {
            const bool state = i == 2 || i == 3;
            const size_t elements = i < 2 ? 48 : state ? gdn_state_elements : gdn_output_elements;
            const size_t bytes = checked_multiply(elements, sizeof(float));
            if (blobs[i].size() != bytes) throw std::runtime_error("GDN state probe artifact byte count mismatch");
            row << (i ? "," : "") << json_string(roles[i]) << ":{\"path\":" << json_string(paths[i])
                << ",\"from\":" << json_string(origins[i]) << ",\"type\":\"f32\",\"elements\":" << elements
                << ",\"bytes\":" << bytes << ",\"endianness\":\"little\",\"header_bytes\":0,\"original_bits_copy\":true"
                << ",\"canonical_layout\":" << json_string(i < 2 ? "[head]" : state ? "[head][value][key]" : "[head][value]")
                << ",\"shape\":" << (i < 2 ? "[48]" : state ? "[48,128,128]" : "[48,128]")
                << ",\"strides_bytes\":" << (i < 2 ? "[4]" : state ? "[65536,512,4]" : "[512,4]") << '}';
        }
        row << "},\"all_finite\":true}\n";
        const auto text = row.str();
        if (text.size() > gdn_state_probe_row_limit ||
            gdn_state_probe_rows >= checked_multiply(gdn_state_probe_layers.size(), capacity))
            throw std::runtime_error("GDN state probe manifest exceeds its row/coverage bound");
        // Publish coverage only after ALL five exclusive files have exact sizes
        // and successful checked closes. Partial artifacts never publish a row.
        for (size_t i = 0; i < paths.size(); ++i) write_probe_blob(output / paths[i], blobs[i].data(), blobs[i].size());
        gdn_state_probe_records->append(text.data(), text.size(), gdn_state_probe_manifest_limit);
        gdn_state_probe_seen[layer] = true;
        ++gdn_state_probe_count;
        ++gdn_state_probe_totals[layer];
        ++gdn_state_probe_rows;
    }

    void correct_l2(ggml_tensor * tensor) {
        const auto found = control_names.find(tensor->name);
        if (found == control_names.end() || tensor->op != GGML_OP_L2_NORM || !tensor->src[0]) {
            throw std::runtime_error("unexpected GDN L2 control name/op/source-0 handshake");
        }
        const size_t slot = found->second;
        if (control_seen[slot]) {
            throw std::runtime_error("duplicate GDN L2 correction for this token/layer/direction");
        }
        const auto * source = tensor->src[0];
        const auto src = check_l2_storage(source);
        const auto dst = check_l2_storage(tensor);
        if (src.address < dst.address + dst.bytes && dst.address < src.address + src.bytes) {
            throw std::runtime_error("GDN L2 source-0 aliases the already-normalized destination; raw input unavailable");
        }
        control_source.resize(src.bytes);
        control_destination.resize(dst.bytes);
        ggml_backend_tensor_get(source, control_source.data(), 0, src.bytes);
        // Preserve every existing destination gap/padding byte, replacing only
        // its actual F32 elements. Source and destination strides may differ.
        ggml_backend_tensor_get(tensor, control_destination.data(), 0, dst.bytes);
        for (size_t head = 0; head < 16; ++head) {
            std::array<float, 128> raw{};
            float sum = 0.0f;
            for (size_t i = 0; i < raw.size(); ++i) {
                const size_t offset = head * source->nb[1] + i * source->nb[0];
                std::memcpy(&raw[i], control_source.data() + offset, sizeof(float));
                if (!std::isfinite(raw[i])) {
                    throw std::runtime_error("non-finite raw GDN L2 source-0 element");
                }
                // Volatile FP32 intermediates enforce separately rounded mul
                // and ascending add. The compiler cannot contract these to FMA.
                const volatile float square = raw[i] * raw[i];
                const volatile float next_sum = sum + square;
                sum = next_sum;
            }
            const volatile float shifted_sum = sum + hf_gdn_l2_epsilon;
            const float denominator = std::sqrt(static_cast<float>(shifted_sum));
            if (!std::isfinite(sum) || !std::isfinite(denominator) || denominator <= 0.0f) {
                throw std::runtime_error("non-finite/invalid FP32 GDN L2 sum or denominator");
            }
            for (size_t i = 0; i < raw.size(); ++i) {
                const size_t offset = head * tensor->nb[1] + i * tensor->nb[0];
                float previous_value = 0.0f;
                std::memcpy(&previous_value, control_destination.data() + offset, sizeof(float));
                if (!std::isfinite(previous_value)) {
                    throw std::runtime_error("non-finite production GDN L2 output before control");
                }
                // Validate the old output, but never use it to derive the
                // correction: only the independently copied raw source-0.
                const float value = raw[i] / denominator;
                if (!std::isfinite(value)) {
                    throw std::runtime_error("non-finite corrected GDN L2 output");
                }
                std::memcpy(control_destination.data() + offset, &value, sizeof(float));
            }
        }
        // ask=false runs after the pinned scheduler's backend synchronization,
        // before it enqueues any subsequent graph range/consumer. This is the
        // synchronous get/set API, not an asynchronous upload of borrowed data.
        ggml_backend_tensor_set(tensor, control_destination.data(), 0, dst.bytes);
        control_seen[slot] = true; // count writes, never scheduler ask requests
        ++control_count;
        ++(warm_pass < 0 ? recorded_control_count : warm_control_count);
        control_records << "{\"evaluation_epoch\":" << evaluation_epoch
                        << ",\"phase\":" << json_string(warm_pass < 0 ? "recorded" : "warm")
                        << ",\"warm_pass\":" << warm_pass << ",\"token_index\":" << token_index
                        << ",\"position\":" << evaluation_position << ",\"seq_id\":0,\"input_token_id\":" << token_id
                        << ",\"node_index\":" << pending_node << ",\"layer\":" << slot / 2
                        << ",\"direction\":" << json_string(slot % 2 == 0 ? "q" : "k")
                        << ",\"name\":" << json_string(tensor->name)
                        << ",\"source_name\":" << json_string(source->name)
                        << ",\"op\":\"L2_NORM\",\"type\":\"f32\",\"ne\":[128,16,1,1],\"source_nb\":[";
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            control_records << (dim ? "," : "") << source->nb[dim];
        }
        control_records << "],\"destination_nb\":[";
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            control_records << (dim ? "," : "") << tensor->nb[dim];
        }
        control_records << "],\"source_bytes\":" << src.bytes << ",\"destination_bytes\":" << dst.bytes
                        << ",\"write_index_in_token\":" << control_count - 1 << "}\n";
        control_records.flush();
    }

    void correct_qsa(ggml_tensor * tensor) {
        const auto found = qsa_control_names.find(tensor->name);
        if (found == qsa_control_names.end() || evaluation_position < 0 ||
            static_cast<size_t>(evaluation_position) >= qsa_max_visible) {
            throw std::runtime_error("unexpected QSA correction name/layer/position handshake");
        }
        const size_t slot = found->second;
        if (qsa_control_seen[slot]) throw std::runtime_error("duplicate QSA correction for this token/layer");
        const auto site = check_qsa_site(tensor);
        const auto * flash = site.flash;
        std::array<float, 3> parameters{};
        std::memcpy(parameters.data(), flash->op_params, sizeof(parameters));
        if (parameters[0] != qsa_coefficient || parameters[1] != 0.0f || parameters[2] != 0.0f ||
            ggml_flash_attn_ext_get_prec(flash) != GGML_PREC_F32) {
            throw std::runtime_error("QSA FLASH requires scale=0.0625, max_bias=0, softcap=0, precision=F32");
        }
        for (size_t i = 4; i < GGML_MAX_SRC; ++i) {
            if (flash->src[i]) throw std::runtime_error("QSA FLASH has an unsupported sink/extra source");
        }
        const std::array<const ggml_tensor *, 4> sources{
            flash->src[0], flash->src[1], flash->src[2], flash->src[3]};
        if (!sources[0] || !sources[1] || !sources[2] || !sources[3]) {
            throw std::runtime_error("QSA FLASH requires original Q, K, V and mask sources");
        }
        const int64_t physical = sources[1]->ne[1];
        const size_t visible = static_cast<size_t>(evaluation_position) + 1;
        if (physical < static_cast<int64_t>(visible) || physical > capacity ||
            sources[3]->ne[1] < 1 || sources[3]->ne[1] > capacity) {
            throw std::runtime_error("QSA physical KV/mask capacity cannot represent the supported causal prefix");
        }
        const std::array<L2Storage, 4> storage{
            check_qsa_storage(sources[0], GGML_TYPE_F32, {256, 1, 24, 1}),
            check_qsa_storage(sources[1], GGML_TYPE_Q4_0, {256, physical, 2, 1}),
            check_qsa_storage(sources[2], GGML_TYPE_Q4_0, {256, physical, 2, 1}),
            check_qsa_storage(sources[3], GGML_TYPE_F16, {physical, sources[3]->ne[1], 1, 1})};
        for (const auto & input : storage) {
            if (overlaps(input, site.destination) || overlaps(input, site.flash_storage)) {
                throw std::runtime_error("QSA FLASH/target aliases raw Q/K/V/mask; independent input unavailable");
            }
            for (const auto & wrapper : site.wrapper_storage) {
                if (overlaps(input, wrapper)) {
                    throw std::runtime_error("QSA intermediate layout storage aliases raw Q/K/V/mask; independent input unavailable");
                }
            }
        }
        // Verify the installed public half helper's round-to-nearest-even ties.
        // No private ggml structs, GPU gather implementation or core linkage.
        if (ggml_fp32_to_fp16(1.00048828125f) != 0x3c00 ||
            ggml_fp32_to_fp16(1.00146484375f) != 0x3c02 ||
            ggml_fp32_to_fp16(-1.00048828125f) != 0xbc00) {
            throw std::runtime_error("installed ggml FP32->FP16 helper does not satisfy the required RNE gather");
        }
        std::array<std::vector<char>, 4> raw;
        for (size_t i = 0; i < raw.size(); ++i) {
            raw[i].resize(storage[i].bytes);
            ggml_backend_tensor_get(sources[i], raw[i].data(), 0, storage[i].bytes);
        }
        std::vector<char> destination(site.destination.bytes);
        ggml_backend_tensor_get(tensor, destination.data(), 0, site.destination.bytes);
        size_t selected = 0;
        // The initial fixture is deliberately restricted to the dense causal
        // prefix, before the fork's long-context selection differs from HF.
        // Do not decode scales from unselected/uninitialized future KV rows.
        for (size_t key = 0; key < static_cast<size_t>(physical); ++key) {
            ggml_fp16_t bits{};
            std::memcpy(&bits, raw[3].data() + key * sources[3]->nb[0], sizeof(bits));
            const float value = ggml_fp16_to_fp32(bits);
            if (std::isfinite(value) && value == 0.0f) {
                if (key >= visible) throw std::runtime_error("QSA mask selects a future physical KV position");
                ++selected;
            } else if (std::isinf(value) && value < 0.0f) {
                if (key < visible) throw std::runtime_error("QSA mask excludes a required causal prefix position");
            } else {
                throw std::runtime_error("QSA mask must contain only finite zero selections or negative infinity exclusions");
            }
        }
        if (selected != visible) throw std::runtime_error("QSA selected KV IDs are not exactly 0..currentPosition");

        std::array<float, qsa_output_elements> query{}, previous{}, corrected{};
        for (size_t head = 0; head < qsa_query_heads; ++head) {
            for (size_t d = 0; d < qsa_head_dim; ++d) {
                const size_t index = head * qsa_head_dim + d;
                const size_t offset = head * sources[0]->nb[2] + d * sources[0]->nb[0];
                std::memcpy(&query[index], raw[0].data() + offset, sizeof(float));
                qsa_finite(query[index]);
                std::memcpy(&previous[index], destination.data() + f32_flat_offset(tensor, index), sizeof(float));
                if (!std::isfinite(previous[index])) throw std::runtime_error("non-finite production QSA target before control");
            }
        }
        // Preflight every selected scale before decoding any selected value.
        for (size_t input : {size_t{1}, size_t{2}}) {
            for (size_t key = 0; key < visible; ++key) {
                for (size_t head = 0; head < qsa_kv_heads; ++head) {
                    for (size_t block = 0; block < qsa_head_dim / 32; ++block) {
                        const size_t offset = key * sources[input]->nb[1] + head * sources[input]->nb[2] +
                                              block * sources[input]->nb[0];
                        ggml_fp16_t scale{};
                        std::memcpy(&scale, raw[input].data() + offset, sizeof(scale));
                        if ((scale & 0x7c00) == 0x7c00) throw std::runtime_error("non-finite selected original QSA Q4_0 scale");
                    }
                }
            }
        }
        const size_t gathered_elements = checked_multiply(visible, qsa_head_dim * qsa_kv_heads);
        std::vector<float> keys(gathered_elements), values(gathered_elements), scores(visible);
        for (size_t key = 0; key < visible; ++key) {
            for (size_t head = 0; head < qsa_kv_heads; ++head) {
                for (size_t d = 0; d < qsa_head_dim; ++d) {
                    const size_t index = (key * qsa_kv_heads + head) * qsa_head_dim + d;
                    keys[index] = qsa_decode_q4(raw[1], sources[1], key, head, d);
                    values[index] = qsa_decode_q4(raw[2], sources[2], key, head, d);
                }
            }
        }
        for (size_t head = 0; head < qsa_query_heads; ++head) {
            const size_t kv_head = head / (qsa_query_heads / qsa_kv_heads);
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t key = 0; key < visible; ++key) {
                const size_t base = (key * qsa_kv_heads + kv_head) * qsa_head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < qsa_head_dim; ++d) {
                    dot = qsa_add(dot, qsa_multiply(query[head * qsa_head_dim + d], keys[base + d]));
                }
                scores[key] = qsa_multiply(dot, qsa_coefficient);
                maximum = std::max(maximum, scores[key]);
            }
            float denominator = 0.0f;
            for (float & score : scores) {
                const volatile float shifted = score - maximum;
                const volatile float weight = std::exp(qsa_finite(shifted));
                score = qsa_finite(weight);
                denominator = qsa_add(denominator, score);
            }
            if (!(denominator > 0.0f)) throw std::runtime_error("QSA FP32 softmax has an empty denominator");
            const volatile float inverse = 1.0f / denominator;
            qsa_finite(inverse);
            for (size_t d = 0; d < qsa_head_dim; ++d) {
                float numerator = 0.0f;
                for (size_t key = 0; key < visible; ++key) {
                    const size_t index = (key * qsa_kv_heads + kv_head) * qsa_head_dim + d;
                    numerator = qsa_add(numerator, qsa_multiply(scores[key], values[index]));
                }
                // Probability one returns the stored Q4->half->float V itself,
                // including signed zero; no GGML max-offset or biased rescale.
                corrected[head * qsa_head_dim + d] = visible == 1 ? values[kv_head * qsa_head_dim + d] :
                                                    qsa_multiply(numerator, inverse);
            }
        }
        double squared_error = 0.0, max_abs_error = 0.0;
        for (size_t index = 0; index < corrected.size(); ++index) {
            qsa_finite(corrected[index]);
            const double error_value = static_cast<double>(corrected[index]) - previous[index];
            max_abs_error = std::max(max_abs_error, std::abs(error_value));
            squared_error += error_value * error_value;
            const size_t offset = f32_flat_offset(tensor, index);
            std::memcpy(destination.data() + offset, &corrected[index], sizeof(float));
        }
        const double rms_error = std::sqrt(squared_error / corrected.size());
        if (!std::isfinite(max_abs_error) || !std::isfinite(rms_error)) {
            throw std::runtime_error("non-finite informational QSA old/new numeric metrics");
        }
        // All inputs, output elements and padding were checked/copied before
        // this single synchronous write. Only kqv_out changes; caches and Q/mask
        // remain intact. The next graph range performs inverse H64/gate/proj.
        std::vector<char> readback(site.destination.bytes);
        ggml_backend_tensor_set(tensor, destination.data(), 0, site.destination.bytes);
        qsa_control_seen[slot] = true; // count actual writes, never scheduler ask requests
        ++qsa_control_count;
        ++(warm_pass < 0 ? recorded_qsa_control_count : warm_qsa_control_count);
        ggml_backend_tensor_get(tensor, readback.data(), 0, readback.size());
        if (readback != destination) throw std::runtime_error("QSA controlled target/padding readback differs from committed bytes");
        for (size_t index = 0; index < corrected.size(); ++index) {
            float value{};
            std::memcpy(&value, readback.data() + f32_flat_offset(tensor, index), sizeof(value));
            if (!std::isfinite(value)) throw std::runtime_error("non-finite controlled QSA target readback");
        }
        qsa_control_records << "{\"evaluation_epoch\":" << evaluation_epoch
            << ",\"phase\":" << json_string(warm_pass < 0 ? "recorded" : "warm")
            << ",\"warm_pass\":" << warm_pass << ",\"token_index\":" << token_index
            << ",\"position\":" << evaluation_position << ",\"seq_id\":0,\"input_token_id\":" << token_id
            << ",\"node_index\":" << pending_node << ",\"occurrence\":0,\"layer\":" << 3 + 4 * slot
            << ",\"name\":" << json_string(tensor->name) << ",\"target\":";
        write_tensor_description(qsa_control_records, tensor, site.destination);
        qsa_control_records << ",\"flash_ancestor\":";
        write_tensor_description(qsa_control_records, flash, site.flash_storage);
        qsa_control_records << ",\"layout_wrappers\":[";
        for (size_t i = 0; i < site.wrappers.size(); ++i) {
            qsa_control_records << (i ? "," : "");
            write_tensor_description(qsa_control_records, site.wrappers[i], site.wrapper_storage[i]);
        }
        qsa_control_records << "],\"sources\":[";
        for (size_t i = 0; i < sources.size(); ++i) {
            qsa_control_records << (i ? "," : "");
            write_tensor_description(qsa_control_records, sources[i], storage[i]);
        }
        qsa_control_records << "],\"query_count\":1,\"selected_count\":" << selected
            << ",\"physical_kv_count\":" << physical << ",\"selected_ids\":\"0..currentPosition\""
            << ",\"raw_quant_type\":\"q4_0\",\"gather\":\"Q4_0->FP16 RNE->FP32, K and V\""
            << ",\"scale\":0.0625,\"max_bias\":0,\"softcap\":0,\"source_precision\":\"F32\""
            << ",\"softmax_accumulation\":\"FP32 ascending separately rounded dot/exp/sum/V-sum; numerator*(1/denominator)\""
            << ",\"stage_change_before_inverse_hadamard\":true,\"cache_unchanged\":true"
            << ",\"weights_unchanged\":true,\"diagnostic_only\":true,\"baseline_performance_reference\":false"
            << ",\"production_source_revision\":" << json_string(revision)
            << ",\"library_revision_attested\":" << json_string(CORE_ORACLE_LIBRARY_REVISION)
            << ",\"hf_source_revision\":" << json_string(hf_revision)
            << ",\"metrics_informational_only\":true,\"old_new_max_abs_error\":" << max_abs_error
            << ",\"old_new_rms_error\":" << rms_error
            << ",\"old_first_scalar\":" << previous[0] << ",\"new_first_scalar\":" << corrected[0]
            << ",\"all_finite\":true,\"readback_exact\":true,\"padding_preserved\":true"
            << ",\"write_index_in_token\":" << qsa_control_count - 1 << "}\n";
        qsa_control_records.flush();
        const int layer = static_cast<int>(3 + 4 * slot);
        if (warm_pass < 0 && token_index >= 0 && qsa_probe_layers.contains(layer)) {
            save_qsa_probe(slot, sources, storage, raw, query, corrected, selected);
        }
    }

    void save_qsa_probe(size_t slot, const std::array<const ggml_tensor *, 4> & sources,
                        const std::array<L2Storage, 4> & storage,
                        const std::array<std::vector<char>, 4> & raw,
                        const std::array<float, qsa_output_elements> & query,
                        const std::array<float, qsa_output_elements> & corrected, size_t selected) {
        // All original inputs and the correction are already independently
        // validated by correct_qsa. No new backend get/set, decode or arithmetic.
        const int layer = static_cast<int>(3 + 4 * slot);
        if (!evaluation_active || warm_pass >= 0 || token_index < 0 || slot >= qsa_probe_seen.size() ||
            !qsa_probe_records || !qsa_probe_layers.contains(layer) || qsa_probe_seen[slot] ||
            selected == 0 || selected > qsa_max_visible || selected != static_cast<size_t>(evaluation_position) + 1)
            throw std::runtime_error("invalid, duplicate or warm QSA probe observation");

        constexpr size_t block_bytes = 18;
        constexpr size_t blocks_per_head = qsa_head_dim / 32;
        constexpr size_t row_bytes = qsa_kv_heads * blocks_per_head * block_bytes;
        constexpr size_t f32_bytes = qsa_output_elements * sizeof(float);
        const size_t kv_bytes = checked_multiply(selected, row_bytes);
        std::array<std::vector<char>, 2> packed{std::vector<char>(kv_bytes), std::vector<char>(kv_bytes)};
        for (size_t input = 1; input <= 2; ++input) {
            if (raw[input].size() != storage[input].bytes || storage[input].bytes < block_bytes)
                throw std::runtime_error("QSA probe original KV storage span mismatch");
            for (size_t key = 0; key < selected; ++key) {
                for (size_t head = 0; head < qsa_kv_heads; ++head) {
                    for (size_t block = 0; block < blocks_per_head; ++block) {
                        const size_t offset = checked_add(checked_add(checked_multiply(key, sources[input]->nb[1]),
                            checked_multiply(head, sources[input]->nb[2])), checked_multiply(block, sources[input]->nb[0]));
                        if (offset > storage[input].bytes - block_bytes)
                            throw std::runtime_error("QSA probe selected original Q4 block escapes validated storage");
                        const size_t target = ((key * qsa_kv_heads + head) * blocks_per_head + block) * block_bytes;
                        std::memcpy(packed[input - 1].data() + target, raw[input].data() + offset, block_bytes);
                    }
                }
            }
        }
        std::array<char, 96> stem{};
        const int length = std::snprintf(stem.data(), stem.size(), "qsa-t%04d-e%06zu-l%02d", token_index, evaluation_epoch, layer);
        if (length < 0 || static_cast<size_t>(length) >= stem.size()) throw std::runtime_error("QSA probe filename overflow");
        const std::array<std::string, 4> paths{
            std::string(stem.data()) + ".query.f32.bin", std::string(stem.data()) + ".k.q4_0.bin",
            std::string(stem.data()) + ".v.q4_0.bin", std::string(stem.data()) + ".corrected.f32.bin"};
        std::ostringstream row;
        row.exceptions(std::ios::badbit | std::ios::failbit);
        row << "{\"format\":\"gfx906-mx-qsa-probe-v1\",\"layer\":" << layer
            << ",\"evaluation_epoch\":" << evaluation_epoch << ",\"phase\":\"recorded\",\"warm_pass\":-1"
            << ",\"token_index\":" << token_index << ",\"position\":" << evaluation_position
            << ",\"seq_id\":0,\"input_token_id\":" << token_id << ",\"node_index\":" << pending_node
            << ",\"selected_count\":" << selected << ",\"selected_ids\":{\"first\":0,\"last\":" << evaluation_position
            << ",\"step\":1},\"physical_kv_count\":" << sources[1]->ne[1]
            << ",\"source_inputs\":\"original FLASH_ATTN_EXT src[0..3]: query,K,V,mask\",\"sources\":[";
        for (size_t i = 0; i < sources.size(); ++i) {
            row << (i ? "," : "");
            write_tensor_description(row, sources[i], storage[i]);
        }
        row << "],\"stage\":\"before_inverse_h64\",\"read_only_observation\":true,\"sources_unchanged\":true"
            << ",\"production_source_revision\":" << json_string(revision)
            << ",\"library_revision_attested\":" << json_string(CORE_ORACLE_LIBRARY_REVISION)
            << ",\"artifacts\":{\"query\":{\"path\":" << json_string(paths[0])
            << ",\"source_index\":0,\"type\":\"f32\",\"elements\":" << query.size() << ",\"bytes\":" << f32_bytes
            << ",\"endianness\":\"little\",\"header_bytes\":0,\"canonical_layout\":\"[query_head][channel]\""
            << ",\"shape\":[24,256],\"strides_bytes\":[1024,4],\"original_strided_f32_copy\":true}";
        for (size_t i = 1; i <= 2; ++i) {
            row << "," << json_string(i == 1 ? "key" : "value") << ":{\"path\":" << json_string(paths[i])
                << ",\"source_index\":" << i << ",\"type\":\"q4_0\",\"elements\":" << selected * qsa_kv_heads * qsa_head_dim
                << ",\"bytes\":" << kv_bytes << ",\"endianness\":\"little\",\"header_bytes\":0"
                << ",\"canonical_layout\":\"[visible_position][kv_head][block][18 original bytes]\""
                << ",\"shape\":[" << selected << ",2,8],\"strides_bytes\":[288,144,18]"
                << ",\"blocks\":" << selected * qsa_kv_heads * blocks_per_head
                << ",\"block_elements\":32,\"block_bytes\":18,\"original_block_copy\":true,\"requantized\":false}";
        }
        row << ",\"corrected\":{\"path\":" << json_string(paths[3])
            << ",\"type\":\"f32\",\"elements\":" << corrected.size() << ",\"bytes\":" << f32_bytes
            << ",\"endianness\":\"little\",\"header_bytes\":0,\"canonical_layout\":\"[query_head][channel]\""
            << ",\"shape\":[24,256],\"strides_bytes\":[1024,4],\"from\":\"independent hf_qsa_f32_control corrected array\""
            << ",\"controlled_target_readback_exact\":true}}}\n";
        const auto text = row.str();
        if (text.size() > qsa_probe_row_limit || qsa_probe_rows >= checked_multiply(qsa_probe_layers.size(), qsa_max_visible))
            throw std::runtime_error("QSA probe manifest exceeds its row/coverage bound");
        // Publish the row only after all four exclusive artifacts have exact
        // lengths and checked closes. A partial observation cannot claim coverage.
        write_probe_blob(output / paths[0], query.data(), f32_bytes);
        write_probe_blob(output / paths[1], packed[0].data(), kv_bytes);
        write_probe_blob(output / paths[2], packed[1].data(), kv_bytes);
        write_probe_blob(output / paths[3], corrected.data(), f32_bytes);
        qsa_probe_records->append(text.data(), text.size(), qsa_probe_manifest_limit);
        qsa_probe_seen[slot] = true;
        ++qsa_probe_count;
        ++qsa_probe_totals[slot];
        ++qsa_probe_rows;
    }

    void save(const ggml_tensor * tensor) {
        // ggml_nbytes is the storage span, including gaps for strided views.
        // Do not reinterpret every intermediate as a contiguous FP32 tensor.
        const size_t bytes = ggml_nbytes(tensor);
        if (!tensor->data || bytes == 0 || bytes > tensor_limit || bytes > token_capture_limit - token_bytes) {
            throw std::runtime_error("selected tensor has unavailable data or exceeds capture byte limits");
        }
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            if (tensor->ne[dim] <= 0) {
                throw std::runtime_error("selected tensor has an invalid dimension");
            }
        }
        const size_t occurrence = occurrences[tensor->name]++;
        std::array<char, 192> filename{};
        const int length = std::snprintf(filename.data(), filename.size(), "t%04d-n%06zu-o%03zu-%s.bin",
            token_index, pending_node, occurrence, tensor->name);
        if (length < 0 || static_cast<size_t>(length) >= filename.size()) {
            throw std::runtime_error("callback artifact filename overflow");
        }
        buffer.resize(bytes);
        // The pinned scheduler synchronizes the producing backend before ask=false.
        // Copy into owned storage now, before scheduler reuse of tensor memory.
        ggml_backend_tensor_get(tensor, buffer.data(), 0, bytes);
        auto blob = open_output(output / filename.data(), true);
        blob.write(buffer.data(), static_cast<std::streamsize>(bytes));
        blob.close();
        records << "{\"token_index\":" << token_index << ",\"position\":" << token_index
                << ",\"seq_id\":0,\"input_token_id\":" << token_id
                << ",\"node_index\":" << pending_node << ",\"occurrence\":" << occurrence
                << ",\"name\":" << json_string(tensor->name)
                << ",\"type\":" << json_string(ggml_type_name(tensor->type))
                << ",\"type_id\":" << static_cast<int>(tensor->type)
                << ",\"op\":" << json_string(ggml_op_name(tensor->op))
                << ",\"op_id\":" << static_cast<int>(tensor->op) << ",\"ne\":[";
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            records << (dim ? "," : "") << tensor->ne[dim];
        }
        records << "],\"nb\":[";
        for (int dim = 0; dim < GGML_MAX_DIMS; ++dim) {
            records << (dim ? "," : "") << tensor->nb[dim];
        }
        records << "],\"bytes\":" << bytes
                << ",\"contiguous\":" << (ggml_is_contiguous(tensor) ? "true" : "false")
                << ",\"view_offset\":" << tensor->view_offs;
        if (hf_gdn_l2_control && control_names.contains(tensor->name)) {
            records << ",\"hf_gdn_l2_controlled\":true,\"evaluation_epoch\":" << evaluation_epoch;
        }
        if (hf_qsa_f32_control && qsa_control_names.contains(tensor->name)) {
            records << ",\"hf_qsa_f32_controlled\":true,\"evaluation_epoch\":" << evaluation_epoch;
        }
        records << ",\"file\":" << json_string(filename.data()) << "}\n";
        records.flush();
        token_bytes += bytes;
        ++captures;
        ++totals[tensor->name];
    }

    static bool callback(ggml_tensor * tensor, bool ask, void * opaque) noexcept {
        auto & self = *static_cast<Capture *>(opaque);
        if (!self.evaluation_active) {
            return !ask;
        }
        if (self.error[0] != '\0') {
            return false;
        }
        try {
            if (!tensor) {
                throw std::runtime_error("scheduler returned a null tensor during an active evaluation");
            }
            const bool control = self.hf_gdn_l2_control && self.control_names.contains(tensor->name);
            const bool qsa_control = self.hf_qsa_f32_control && self.qsa_control_names.contains(tensor->name);
            const bool observe = self.token_index >= 0 && self.names.contains(tensor->name);
            const bool gdn_state_probe = self.token_index >= 0 && self.gdn_state_output_names.contains(tensor->name);
            if (ask) {
                self.pending_node = self.ask_index++;
                self.scan_gdn_state_node(tensor); // retain ancestry only; never select state VIEW/CPY
                if (self.hf_gdn_l2_control && tensor->op == GGML_OP_L2_NORM &&
                    tensor->ne[0] == 128 && tensor->ne[1] == 16 && !control) {
                    throw std::runtime_error("unnamed/unexpected 128x16 GDN L2 control node");
                }
                if (self.hf_qsa_f32_control && std::string_view(tensor->name).starts_with("kqv_out-") && !qsa_control) {
                    // GGML derives layout-only names such as "kqv_out-3 (reshaped)".
                    // Only an exact integer-suffixed canonical site is a handshake;
                    // correcting its aliases again would duplicate the same output.
                    const auto suffix = std::string_view(tensor->name).substr(8);
                    int layer = -1;
                    const auto [end, error] = std::from_chars(suffix.data(), suffix.data() + suffix.size(), layer);
                    if (error == std::errc{} && end == suffix.data() + suffix.size()) {
                        throw std::runtime_error("unexpected canonical kqv_out layer/site for QSA control");
                    }
                }
                // Exact artifact and independent diagnostic control allowlists.
                // Warmup suppresses only observations, never either correction.
                return observe || control || qsa_control;
            }
            if (!control && !qsa_control && !observe) {
                throw std::runtime_error("scheduler returned an unrequested tensor");
            }
            if (control) {
                self.correct_l2(tensor);
            }
            if (qsa_control) {
                self.correct_qsa(tensor);
            }
            if (gdn_state_probe) {
                self.save_gdn_state_probe(tensor);
            }
            if (observe) {
                self.save(tensor); // requested Q/K and kqv_out captures see corrected outputs
            }
            return true;
        } catch (const std::exception & ex) {
            std::snprintf(self.error.data(), self.error.size(),
                "epoch=%zu pass=%d position=%d input_token_id=%d node=%s: %s",
                self.evaluation_epoch, self.warm_pass, self.evaluation_position, self.token_id,
                tensor ? tensor->name : "<null>", ex.what());
        } catch (...) {
            std::snprintf(self.error.data(), self.error.size(),
                "epoch=%zu pass=%d position=%d input_token_id=%d: unknown callback error",
                self.evaluation_epoch, self.warm_pass, self.evaluation_position, self.token_id);
        }
        // Never propagate a C++ exception through the C callback boundary.
        return false;
    }

    void check(bool intermediates) const {
        // The pinned scheduler's cancellation can break just a backend split;
        // check our own error flag even when llama_decode reports success.
        if (error[0] != '\0') {
            throw std::runtime_error(std::string("callback failed: ") + error.data());
        }
        if (!evaluation_active) {
            throw std::runtime_error("no active oracle evaluation to validate");
        }
        if (hf_gdn_l2_control) {
            for (const auto & [name, slot] : control_names) {
                if (!control_seen[slot]) {
                    throw std::runtime_error("missing HF GDN L2 correction: " + name + " at epoch " +
                        std::to_string(evaluation_epoch) + ", pass " + std::to_string(warm_pass) +
                        ", position " + std::to_string(evaluation_position) + ", input token " + std::to_string(token_id));
                }
            }
            if (control_count != gdn_l2_nodes_per_token) {
                throw std::runtime_error("HF GDN L2 control requires exactly 72 unique Q/K writes per token");
            }
        }
        if (hf_qsa_f32_control) {
            for (const auto & [name, slot] : qsa_control_names) {
                if (!qsa_control_seen[slot]) {
                    throw std::runtime_error("missing HF QSA F32 correction: " + name + " at epoch " +
                        std::to_string(evaluation_epoch) + ", pass " + std::to_string(warm_pass) +
                        ", position " + std::to_string(evaluation_position) + ", input token " + std::to_string(token_id));
                }
            }
            if (qsa_control_count != qsa_nodes_per_token) {
                throw std::runtime_error("HF QSA F32 control requires exactly 12 unique output writes per token");
            }
        }
        if (token_index < 0) {
            if (qsa_probe_count != 0 || std::any_of(qsa_probe_seen.begin(), qsa_probe_seen.end(), [](bool seen) { return seen; }))
                throw std::runtime_error("QSA probes must never record warmup observations");
            if (gdn_state_probe_count != 0 || !gdn_primitive_nodes.empty() ||
                std::any_of(gdn_state_probe_seen.begin(), gdn_state_probe_seen.end(), [](bool seen) { return seen; }))
                throw std::runtime_error("GDN state probes must never scan or record warmup observations");
            return; // Warm control validation does not demand artifact captures.
        }
        if (qsa_probe_count != qsa_probe_layers.size()) throw std::runtime_error("incomplete per-token QSA probe coverage");
        for (int layer : qsa_probe_layers) {
            if (!qsa_probe_seen[static_cast<size_t>((layer - 3) / 4)])
                throw std::runtime_error("missing recorded QSA probe at layer " + std::to_string(layer));
        }
        if (gdn_state_probe_count != gdn_state_probe_layers.size())
            throw std::runtime_error("incomplete per-token GDN state probe coverage");
        for (int layer : gdn_state_probe_layers) {
            if (!gdn_state_probe_seen[static_cast<size_t>(layer)])
                throw std::runtime_error("missing recorded GDN state probe at layer " + std::to_string(layer));
        }
        for (const auto * name : required_names) {
            if (!occurrences.contains(name)) {
                throw std::runtime_error(std::string("required callback tensor missing: ") + name);
            }
        }
        if (intermediates) {
            for (const int layer : layers) {
                const auto name = "ffn_moe_cache_slots-" + std::to_string(layer);
                if (!occurrences.contains(name)) {
                    throw std::runtime_error("requested GPU expert cache is not active at layer " + std::to_string(layer));
                }
            }
        }
        if (all_layers) {
            for (const auto & name : names) {
                const auto found = occurrences.find(name);
                const auto probe = probe_expected.find(name);
                const size_t expected = probe == probe_expected.end() ? 1 : probe->second;
                if (found == occurrences.end() || found->second != expected) {
                    throw std::runtime_error("all-layer diagnostic has an incorrect exact capture count: " + name);
                }
            }
        }
    }

    void finish() {
        if (warm_pass < 0 && !qsa_probe_layers.empty()) ++qsa_probe_tokens_checked;
        if (warm_pass < 0 && !gdn_state_probe_layers.empty()) ++gdn_state_probe_tokens_checked;
        if (hf_gdn_l2_control) {
            ++(warm_pass < 0 ? recorded_control_tokens_checked : warm_control_tokens_checked);
        }
        if (hf_qsa_f32_control) {
            ++(warm_pass < 0 ? recorded_qsa_control_tokens_checked : warm_qsa_control_tokens_checked);
        }
        token_index = -1;
        evaluation_active = false;
    }

    void check_warm_pass(size_t previous_count, size_t previous_tokens,
                         size_t previous_qsa_count, size_t previous_qsa_tokens, size_t tokens) {
        if (hf_gdn_l2_control) {
            const size_t count = warm_control_count - previous_count;
            if (count != checked_multiply(gdn_l2_nodes_per_token, tokens) ||
                warm_control_tokens_checked - previous_tokens != tokens) {
                throw std::runtime_error("HF GDN L2 warm pass did not complete 72 corrections for every fixed token");
            }
            warm_pass_control_counts.push_back(count);
        }
        if (hf_qsa_f32_control) {
            const size_t count = warm_qsa_control_count - previous_qsa_count;
            if (count != checked_multiply(qsa_nodes_per_token, tokens) ||
                warm_qsa_control_tokens_checked - previous_qsa_tokens != tokens) {
                throw std::runtime_error("HF QSA F32 warm pass did not complete 12 corrections for every fixed token");
            }
            warm_pass_qsa_control_counts.push_back(count);
        }
    }

    void check_control_totals(size_t tokens, int passes) const {
        if (hf_gdn_l2_control) {
            const size_t recorded_expected = checked_multiply(gdn_l2_nodes_per_token, tokens);
            const size_t warm_tokens_expected = checked_multiply(tokens, static_cast<size_t>(passes));
            if (recorded_control_count != recorded_expected || recorded_control_tokens_checked != tokens ||
                warm_control_count != checked_multiply(gdn_l2_nodes_per_token, warm_tokens_expected) ||
                warm_control_tokens_checked != warm_tokens_expected ||
                warm_pass_control_counts.size() != static_cast<size_t>(passes)) {
                throw std::runtime_error("HF GDN L2 run has incomplete recorded/warm correction counts");
            }
        }
        if (hf_qsa_f32_control) {
            const size_t recorded_expected = checked_multiply(qsa_nodes_per_token, tokens);
            const size_t warm_tokens_expected = checked_multiply(tokens, static_cast<size_t>(passes));
            if (recorded_qsa_control_count != recorded_expected || recorded_qsa_control_tokens_checked != tokens ||
                warm_qsa_control_count != checked_multiply(qsa_nodes_per_token, warm_tokens_expected) ||
                warm_qsa_control_tokens_checked != warm_tokens_expected ||
                warm_pass_qsa_control_counts.size() != static_cast<size_t>(passes)) {
                throw std::runtime_error("HF QSA F32 run has incomplete recorded/warm correction counts");
            }
        }
    }

    void check_qsa_probe_totals(size_t tokens) const {
        if (qsa_probe_layers.empty()) return;
        if (!qsa_probe_records || qsa_probe_rows != checked_multiply(tokens, qsa_probe_layers.size()) ||
            qsa_probe_tokens_checked != tokens)
            throw std::runtime_error("QSA probe run has incomplete recorded-token/requested-layer coverage");
        for (size_t slot = 0; slot < qsa_probe_totals.size(); ++slot) {
            const size_t expected = qsa_probe_layers.contains(static_cast<int>(3 + 4 * slot)) ? tokens : 0;
            if (qsa_probe_totals[slot] != expected) throw std::runtime_error("QSA probe run has an incorrect exact layer count");
        }
        qsa_probe_records->check_size();
    }

    void check_gdn_state_probe_totals(size_t tokens) const {
        if (gdn_state_probe_layers.empty()) return;
        if (!gdn_state_probe_records || gdn_state_probe_rows != checked_multiply(tokens, gdn_state_probe_layers.size()) ||
            gdn_state_probe_tokens_checked != tokens)
            throw std::runtime_error("GDN state probe run has incomplete recorded-token/requested-layer coverage");
        for (size_t layer = 0; layer < gdn_state_probe_totals.size(); ++layer) {
            const size_t expected = gdn_state_probe_layers.contains(static_cast<int>(layer)) ? tokens : 0;
            if (gdn_state_probe_totals[layer] != expected)
                throw std::runtime_error("GDN state probe run has an incorrect exact layer count");
        }
        gdn_state_probe_records->check_size();
    }
};

struct DeviceInfo {
    std::string name;
    std::string description;
    std::string backend;
    std::string id;
    size_t total;
    size_t free_before_load;
};

struct Metadata {
    std::string library_version;
    std::vector<DeviceInfo> devices;
    int32_t actual_vocab = 0;
    int32_t actual_layers = 0;
    uint32_t actual_ctx = 0;
    uint32_t actual_batch = 0;
    uint32_t actual_ubatch = 0;
    size_t completed_tokens = 0;
};

void write_metadata(const Options & opts, const Metadata & meta, const Capture & capture,
                    std::string_view status, std::string_view error = {}) {
    auto out = open_output(opts.output / "metadata.json");
    out << "{\n\"format\":\"gfx906-mx-oracle-v1\",\"status\":" << json_string(status)
        << ",\"error\":" << json_string(error)
        << ",\"source_revision\":" << json_string(revision)
        << ",\"library_revision_attested\":" << json_string(CORE_ORACLE_LIBRARY_REVISION)
        << ",\"library_revision_runtime_verified\":false,\"library_version\":" << json_string(meta.library_version)
        << ",\"model\":" << json_string(opts.model.filename().string())
        << ",\"context\":{\"capacity_requested\":4096,\"capacity_actual\":" << meta.actual_ctx
        << ",\"batch_requested\":1,\"ubatch_requested\":1,\"batch_actual\":" << meta.actual_batch
        << ",\"ubatch_actual\":" << meta.actual_ubatch
        << ",\"seq_id\":0,\"seq_max\":1,\"threads\":16,\"threads_batch\":16"
        << ",\"type_k\":\"q4_0\",\"type_v\":\"q4_0\",\"flash_attention_requested\":\"enabled\""
        << ",\"offload_kqv\":true,\"op_offload\":true,\"recurrent_snapshots\":0}"
        << ",\"placement\":{\"split_mode\":\"layer\",\"tensor_split\":[1,1],\"n_gpu_layers\":-1"
        << ",\"cpu_expert_pattern\":" << json_string(cpu_expert_pattern)
        << ",\"use_extra_bufts\":false,\"load_mode\":\"direct_io\",\"lazy_mode\":\"off\""
        << ",\"moe_cache_slots_requested\":112,\"moe_cache_inserts_requested\":" << opts.cache_inserts
        << ",\"cache_activation_checked_layers\":" << (opts.intermediates ? "[0,1,3,47]" : "null") << "}"
        << ",\"expert_cache_warmup\":{\"passes\":" << opts.cache_warmup_passes
        << ",\"tokens_per_pass\":" << opts.tokens.size()
        << ",\"state_clear_after_each_pass\":true,\"warmup_logits_discarded\":true}"
        << ",\"hf_gdn_l2_control\":{\"enabled\":" << (opts.hf_gdn_l2_control ? "true" : "false")
        << ",\"mode\":" << json_string(opts.hf_gdn_l2_control ? "experimental_hf_semantic_correction" : "disabled")
        << ",\"diagnostic_only\":true,\"baseline_performance_reference\":false,\"weights_unchanged\":true"
        << ",\"hf_source_revision\":" << json_string(hf_revision)
        << ",\"production_source_revision\":" << json_string(revision)
        << ",\"production_formula\":\"x/sqrt(max(sum(x*x),eps*eps))\""
        << ",\"control_formula\":\"x/sqrt(sum(x*x)+1e-6f)\",\"epsilon_type\":\"f32\""
        << ",\"epsilon_f32\":" << std::setprecision(std::numeric_limits<float>::max_digits10) << hf_gdn_l2_epsilon
        << ",\"epsilon_f32_bits\":" << std::bit_cast<uint32_t>(hf_gdn_l2_epsilon)
        << ",\"arithmetic\":\"ascending 128-element FP32 sum; separately rounded multiply/add, no FMA; raw source-0 recomputation\""
        << ",\"handshake\":{\"name_patterns\":[\"q_conv_predelta-L\",\"k_conv_predelta-L\"]"
        << ",\"layers\":\"0..47 except L%4==3\",\"op\":\"L2_NORM\",\"type\":\"f32\""
        << ",\"ne\":[128,16,1,1],\"source_index\":0,\"source_ne\":[128,16,1,1],\"source_type\":\"f32\""
        << ",\"source_destination_disjoint_required\":true,\"strided_storage_checked\":true"
        << ",\"preserve_destination_padding\":true,\"layer_count\":36,\"corrections_per_token\":72}"
        << ",\"records\":" << (opts.hf_gdn_l2_control ? "\"gdn-l2-control.jsonl\"" : "null")
        << ",\"recorded_control_count\":" << capture.recorded_control_count
        << ",\"warm_control_count\":" << capture.warm_control_count
        << ",\"recorded_tokens_checked\":" << capture.recorded_control_tokens_checked
        << ",\"warm_tokens_checked\":" << capture.warm_control_tokens_checked
        << ",\"warm_pass_expected_count\":" << (opts.hf_gdn_l2_control ? gdn_l2_nodes_per_token * opts.tokens.size() : 0)
        << ",\"warm_pass_control_counts\":[";
    for (size_t i = 0; i < capture.warm_pass_control_counts.size(); ++i) {
        out << (i ? "," : "") << capture.warm_pass_control_counts[i];
    }
    out << "],\"last_evaluation\":{\"epoch\":" << capture.evaluation_epoch
        << ",\"warm_pass\":" << capture.warm_pass << ",\"position\":" << capture.evaluation_position
        << ",\"active\":" << (capture.evaluation_active ? "true" : "false")
        << ",\"input_token_id\":" << capture.token_id << ",\"control_count\":" << capture.control_count << "}}"
        << ",\"hf_qsa_f32_control\":{\"enabled\":" << (opts.hf_qsa_f32_control ? "true" : "false")
        << ",\"mode\":" << json_string(opts.hf_qsa_f32_control ? "experimental_hf_normalized_attention_precision" : "disabled")
        << ",\"diagnostic_only\":true,\"baseline_performance_reference\":false,\"weights_unchanged\":true"
        << ",\"independent_of_hf_gdn_l2_control\":true,\"hf_source_revision\":" << json_string(hf_revision)
        << ",\"production_source_revision\":" << json_string(revision)
        << ",\"library_revision_attested\":" << json_string(CORE_ORACLE_LIBRARY_REVISION)
        << ",\"max_control_visible\":2048,\"supported_positions\":[0,2047]"
        << ",\"q4_cache_unchanged\":true,\"query_and_mask_unchanged\":true"
        << ",\"raw_quant_type\":\"q4_0\",\"gather\":\"Q4_0->FP16 RNE->FP32 for K and V\""
        << ",\"gather_helpers\":[\"ggml_fp16_to_fp32\",\"ggml_fp32_to_fp16\"]"
        << ",\"fp32_rounding\":\"round-to-nearest-even required; environment is not modified\""
        << ",\"softmax_accumulation\":\"FP32\",\"arithmetic\":\"ascending separate FP32 mul/add; scores=dot*0.0625; exp(score-max); denominator=sum(weights); output=sum(weights*V)*(1/denominator), probability-one copies gathered V; no FMA or GGML max-offset\""
        << ",\"original_output_used_only_for_finite_guards_and_informational_metrics\":true"
        << ",\"stage_change_before_inverse_hadamard\":true,\"no_math_ancestor_wrappers\":true"
        << ",\"handshake\":{\"name_pattern\":\"kqv_out-L\",\"layers\":[3,7,11,15,19,23,27,31,35,39,43,47]"
        << ",\"target_type\":\"f32\",\"target_ne\":[6144,1,1,1]"
        << ",\"ancestor_op\":\"FLASH_ATTN_EXT\",\"ancestor_count\":1,\"ancestor_ne\":[256,24,1,1]"
        << ",\"wrapper_ops\":[\"VIEW\",\"RESHAPE\",\"PERMUTE\",\"CONT\"],\"max_wrappers\":8"
        << ",\"head_major_flattening_proved\":true,\"query_type\":\"f32\",\"query_ne\":[256,1,24,1]"
        << ",\"kv_type\":\"q4_0\",\"kv_ne\":[256,\"physicalKV\",2,1],\"q4_block_bytes\":18,\"q4_row_bytes\":144"
        << ",\"mask_type\":\"f16\",\"mask_ne\":[\"physicalKV\",\">=1\",1,1]"
        << ",\"mask_row\":0,\"mask_selected_value\":0,\"mask_excluded_value\":\"-inf\""
        << ",\"selected_ids_exact\":\"0..currentPosition\",\"query_count\":1,\"seq_id\":0,\"gqa_group\":12"
        << ",\"scale\":0.0625,\"max_bias\":0,\"softcap\":0,\"precision\":\"F32\",\"sinks\":false"
        << ",\"source_destination_disjoint_required\":true,\"strided_storage_checked\":true"
        << ",\"preserve_destination_padding\":true,\"exact_readback_required\":true,\"corrections_per_token\":12}"
        << ",\"records\":" << (opts.hf_qsa_f32_control ? "\"qsa-f32-control.jsonl\"" : "null")
        << ",\"recorded_control_count\":" << capture.recorded_qsa_control_count
        << ",\"warm_control_count\":" << capture.warm_qsa_control_count
        << ",\"recorded_tokens_checked\":" << capture.recorded_qsa_control_tokens_checked
        << ",\"warm_tokens_checked\":" << capture.warm_qsa_control_tokens_checked
        << ",\"recorded_expected_count\":" << (opts.hf_qsa_f32_control ? qsa_nodes_per_token * opts.tokens.size() : 0)
        << ",\"warm_expected_count\":" << (opts.hf_qsa_f32_control ? qsa_nodes_per_token * opts.tokens.size() * opts.cache_warmup_passes : 0)
        << ",\"warm_pass_expected_count\":" << (opts.hf_qsa_f32_control ? qsa_nodes_per_token * opts.tokens.size() : 0)
        << ",\"warm_pass_control_counts\":[";
    for (size_t i = 0; i < capture.warm_pass_qsa_control_counts.size(); ++i) {
        out << (i ? "," : "") << capture.warm_pass_qsa_control_counts[i];
    }
    out << "],\"last_evaluation\":{\"epoch\":" << capture.evaluation_epoch
        << ",\"warm_pass\":" << capture.warm_pass << ",\"position\":" << capture.evaluation_position
        << ",\"active\":" << (capture.evaluation_active ? "true" : "false")
        << ",\"input_token_id\":" << capture.token_id << ",\"control_count\":" << capture.qsa_control_count << "}}";
    if (!opts.gdn_probe_layers.empty()) {
        out << ",\"gdn_state_probes\":{\"format\":\"gfx906-mx-gdn-state-probe-v1\",\"records\":\"gdn-state-probes.jsonl\""
            << ",\"read_only_observation\":true,\"warmup_suppressed\":true,\"rows\":" << capture.gdn_state_probe_rows
            << ",\"expected_rows\":" << checked_multiply(opts.tokens.size(), opts.gdn_probe_layers.size())
            << ",\"recorded_tokens_checked\":" << capture.gdn_state_probe_tokens_checked
            << ",\"artifacts_per_row\":5,\"row_byte_limit\":" << gdn_state_probe_row_limit
            << ",\"manifest_byte_limit\":" << gdn_state_probe_manifest_limit
            << ",\"tensor_identity_scope\":\"evaluation_epoch\",\"callback_site\":\"attn_output-L\""
            << ",\"new_state_and_cache_copy_selected\":false,\"post_from_persistent_cache_destination\":true"
            << ",\"primitive_state_tail_read\":false,\"decay_materialized\":false"
            << ",\"control_elements\":48,\"state_elements\":" << gdn_state_elements
            << ",\"raw_output_elements\":" << gdn_output_elements
            << ",\"state_layout\":\"[head][value][key], key contiguous\",\"requested_layers\":[";
        bool comma = false;
        for (int layer : opts.gdn_probe_layers) {
            out << (comma ? "," : "") << layer;
            comma = true;
        }
        out << "],\"layer_counts\":[";
        comma = false;
        for (int layer : opts.gdn_probe_layers) {
            out << (comma ? "," : "") << "{\"layer\":" << layer << ",\"rows\":"
                << capture.gdn_state_probe_totals[static_cast<size_t>(layer)] << '}';
            comma = true;
        }
        out << "],\"last_evaluation_probe_count\":" << capture.gdn_state_probe_count << '}';
    }
    if (!opts.qsa_probe_layers.empty()) {
        out << ",\"qsa_probes\":{\"format\":\"gfx906-mx-qsa-probe-v1\",\"records\":\"qsa-probes.jsonl\""
            << ",\"read_only_observation\":true,\"warmup_suppressed\":true,\"rows\":" << capture.qsa_probe_rows
            << ",\"expected_rows\":" << checked_multiply(opts.tokens.size(), opts.qsa_probe_layers.size())
            << ",\"recorded_tokens_checked\":" << capture.qsa_probe_tokens_checked
            << ",\"row_byte_limit\":" << qsa_probe_row_limit << ",\"manifest_byte_limit\":" << qsa_probe_manifest_limit
            << ",\"requested_layers\":[";
        bool comma = false;
        for (int layer : opts.qsa_probe_layers) {
            out << (comma ? "," : "") << layer;
            comma = true;
        }
        out << "],\"layer_counts\":[";
        comma = false;
        for (int layer : opts.qsa_probe_layers) {
            out << (comma ? "," : "") << "{\"layer\":" << layer << ",\"rows\":"
                << capture.qsa_probe_totals[static_cast<size_t>((layer - 3) / 4)] << '}';
            comma = true;
        }
        out << "]}";
    }
    out << ",\"model_layers_actual\":" << meta.actual_layers << ",\"vocab_actual\":" << meta.actual_vocab
        << ",\"sampler\":null,\"mtp\":false,\"token_ids\":[";
    for (size_t i = 0; i < opts.tokens.size(); ++i) {
        out << (i ? "," : "") << opts.tokens[i];
    }
    out << "],\"completed_tokens\":" << meta.completed_tokens
        << ",\"logits\":{\"file\":\"logits.f32.bin\",\"dtype\":\"float32\",\"endianness\":\"little\""
        << ",\"layout\":\"token-major,vocabulary-minor\",\"columns\":248320,\"header_bytes\":0"
        << ",\"row_bytes\":" << vocab_size * sizeof(float)
        << ",\"bytes_written\":" << meta.completed_tokens * vocab_size * sizeof(float)
        << ",\"semantics\":\"row i is the next-token distribution after consuming token_ids[i] at position i\"}"
        << ",\"callbacks\":{\"records\":\"tensors.jsonl\",\"captures\":" << capture.captures
        << ",\"all_layer_outputs_and_slots\":" << (opts.all_layers ? "true" : "false")
        << ",\"tensor_byte_limit\":" << tensor_limit << ",\"token_byte_limit\":" << token_capture_limit
        << ",\"binary_layout\":\"native little-endian storage span with original ne/nb; includes stride gaps\""
        << ",\"indices\":\"token_index,node_index,occurrence are zero-based; node_index counts scheduler ask calls\""
        << ",\"allowlist\":[";
    bool comma = false;
    for (const auto & name : capture.names) {
        out << (comma ? "," : "") << json_string(name);
        comma = true;
    }
    out << "],\"gdn_probe_layers\":[";
    comma = false;
    for (int layer : opts.gdn_probe_layers) {
        out << (comma ? "," : "") << layer;
        comma = true;
    }
    out << "],\"observed_counts\":{";
    comma = false;
    for (const auto & [name, count] : capture.totals) {
        out << (comma ? "," : "") << json_string(name) << ':' << count;
        comma = true;
    }
    out << "}},\"devices\":[";
    for (size_t i = 0; i < meta.devices.size(); ++i) {
        const auto & dev = meta.devices[i];
        out << (i ? "," : "") << "{\"name\":" << json_string(dev.name)
            << ",\"description\":" << json_string(dev.description) << ",\"backend\":" << json_string(dev.backend)
            << ",\"device_id\":" << json_string(dev.id) << ",\"memory_total\":" << dev.total
            << ",\"memory_free_before_load\":" << dev.free_before_load << '}';
    }
    out << "]\n}\n";
    out.close();
}

struct BackendLifetime {
    BackendLifetime() {
        ggml_backend_load_all_from_path(CORE_ORACLE_LIBRARY_DIR);
        llama_backend_init();
    }
    ~BackendLifetime() { llama_backend_free(); }
};

void run(const Options & opts, Metadata & meta, Capture & capture) {
    if (opts.hf_qsa_f32_control && opts.tokens.size() > qsa_max_visible) {
        throw std::runtime_error("HF QSA F32 control rejects more than 2048 fixed tokens: supported positions 0..2047, max_control_visible=2048");
    }
    if (opts.hf_qsa_f32_control && std::fegetround() != FE_TONEAREST) {
        throw std::runtime_error("HF QSA F32 control requires round-to-nearest-even FP32 arithmetic");
    }
    capture.open_gdn_state_probes();
    BackendLifetime backend;
    meta.library_version = llama_version(); // version string does NOT expose a commit
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        const std::string name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (name != "ROCm" && name != "HIP") {
            continue;
        }
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(dev, &props);
        devices.push_back(dev);
        meta.devices.push_back({props.name, props.description, name, props.device_id ? props.device_id : "",
            props.memory_total, props.memory_free});
    }
    if (devices.size() != 2 || !ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        throw std::runtime_error("oracle requires exactly two HIP GPUs and an installed CPU backend");
    }
    devices.push_back(nullptr);
    std::vector<float> split(llama_max_devices(), 0.0f);
    split.at(0) = split.at(1) = 1.0f;
    const llama_model_tensor_buft_override overrides[] = {
        {cpu_expert_pattern, ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    auto mp = llama_model_default_params();
    mp.devices = devices.data();
    mp.tensor_buft_overrides = overrides;
    mp.n_gpu_layers = -1; // llama_model::n_gpu_layers(): negative means n_layer_all+1
    mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    mp.tensor_split = split.data();
    mp.main_gpu = 0;
    mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    // CPU_REPACK is excluded by llama_moe_cache::create's canonical_host check.
    // Canonical CPU/device-pinned host storage qualifies; no common init needed.
    mp.use_extra_bufts = false;
    mp.no_host = false;
    mp.load_mtp = false;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(opts.model.c_str(), mp), llama_model_free);
    if (!model) {
        throw std::runtime_error("production model load failed; inspect stderr");
    }
    meta.actual_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    meta.actual_layers = llama_model_n_layer(model.get());
    if (meta.actual_vocab != vocab_size || meta.actual_layers != 48) {
        throw std::runtime_error("oracle fixture requires the 48-layer, 248320-token target model");
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = capacity;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_seq_max = 1;
    cp.n_rs_seq = 0;
    cp.n_outputs_max = 1;
    cp.n_outputs_max_per_seq = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    cp.type_k = GGML_TYPE_Q4_0;
    cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    cp.n_moe_cache_slots = 112;
    cp.n_moe_cache_inserts = opts.cache_inserts;
    cp.offload_kqv = true;
    cp.op_offload = true;
    cp.embeddings = false;
    cp.no_perf = true;
    cp.cb_eval = Capture::callback;
    cp.cb_eval_user_data = &capture;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        throw std::runtime_error("Q4 K/V, flash-attention or expert-cache context initialization failed; inspect stderr");
    }
    meta.actual_ctx = llama_n_ctx(ctx.get());
    meta.actual_batch = llama_n_batch(ctx.get());
    meta.actual_ubatch = llama_n_ubatch(ctx.get());
    if (meta.actual_ctx != capacity || llama_n_seq_max(ctx.get()) != 1 || meta.actual_ubatch != 1) {
        throw std::runtime_error("context does not match the requested fixture capacity/sequence/ubatch");
    }
    // Matched GPU arithmetic diagnostic, explicitly separate from the default
    // cold baseline. CPU Q8_1 stores d*sum(codes); HIP stores the raw input sum.
    // Warm immutable expert slots, then clear ALL recurrent/KV/PLE input state.
    // llama_memory_clear only touches the memory module, not context.moe_cache.
    // Each pass uses the complete identical teacher-forced timeline; no callback
    // artifacts, performance samples or output tokens are counted for warmup.
    // HF control, when requested, must run on the warm timeline too: otherwise
    // it would admit experts selected by a different mathematical forward.
    for (int pass = 0; pass < opts.cache_warmup_passes; ++pass) {
        const size_t previous_controls = capture.warm_control_count;
        const size_t previous_tokens = capture.warm_control_tokens_checked;
        const size_t previous_qsa_controls = capture.warm_qsa_control_count;
        const size_t previous_qsa_tokens = capture.warm_qsa_control_tokens_checked;
        for (size_t i = 0; i < opts.tokens.size(); ++i) {
            llama_token token = opts.tokens[i]; llama_pos position = static_cast<llama_pos>(i);
            int32_t count = 1; llama_seq_id sequence = 0; llama_seq_id* sequences = &sequence; int8_t wanted = 1;
            llama_batch batch{1, &token, nullptr, &position, &count, &sequences, &wanted};
            if (opts.hf_gdn_l2_control || opts.hf_qsa_f32_control) {
                capture.begin(position, token, pass);
            }
            const int32_t result = llama_decode(ctx.get(), batch);
            if (opts.hf_gdn_l2_control || opts.hf_qsa_f32_control) {
                llama_synchronize(ctx.get());
                if (capture.error[0] != '\0') {
                    throw std::runtime_error(std::string("warm callback failed: ") + capture.error.data());
                }
            }
            if (result != 0) {
                throw std::runtime_error("cache warmup decode failed: pass " + std::to_string(pass) +
                    ", position " + std::to_string(i) + ", input token " + std::to_string(token) +
                    ", result " + std::to_string(result));
            }
            if (opts.hf_gdn_l2_control || opts.hf_qsa_f32_control) {
                capture.check(opts.intermediates);
                capture.finish();
            }
        }
        llama_synchronize(ctx.get());
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        capture.check_warm_pass(previous_controls, previous_tokens, previous_qsa_controls, previous_qsa_tokens, opts.tokens.size());
    }
    auto logits_file = open_output(opts.output / "logits.f32.bin", true);
    auto token_records = open_output(opts.output / "tokens.jsonl");
    token_records << std::setprecision(std::numeric_limits<double>::max_digits10);
    std::vector<float> logits(vocab_size);
    write_metadata(opts, meta, capture, "running");
    for (size_t i = 0; i < opts.tokens.size(); ++i) {
        llama_token token = opts.tokens[i];
        llama_pos position = static_cast<llama_pos>(i);
        int32_t n_seq_id = 1;
        llama_seq_id seq_id = 0;
        llama_seq_id * seq_ids = &seq_id;
        int8_t output_logits = 1;
        // All batch fields are explicit. No implicit position tracking or BOS/EOS edits.
        llama_batch batch{1, &token, nullptr, &position, &n_seq_id, &seq_ids, &output_logits};
        capture.begin(position, token);
        const auto start = std::chrono::steady_clock::now();
        const int32_t result = llama_decode(ctx.get(), batch); // ONLY this separate oracle
        llama_synchronize(ctx.get());
        if (capture.error[0] != '\0') {
            throw std::runtime_error(std::string("callback failed: ") + capture.error.data());
        }
        if (result != 0) {
            throw std::runtime_error("llama_decode failed at position " + std::to_string(i) + ": " + std::to_string(result));
        }
        capture.check(opts.intermediates);
        capture.finish();
        const auto * source = llama_get_logits_ith(ctx.get(), 0);
        if (!source) {
            throw std::runtime_error("logits row is unavailable at position " + std::to_string(i));
        }
        std::copy_n(source, logits.size(), logits.data());
        for (const float value : logits) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("non-finite logit at position " + std::to_string(i));
            }
        }
        const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        const size_t bytes = logits.size() * sizeof(float);
        logits_file.write(reinterpret_cast<const char *>(logits.data()), static_cast<std::streamsize>(bytes));
        logits_file.flush();
        const auto best = std::max_element(logits.begin(), logits.end());
        token_records << "{\"token_index\":" << i << ",\"position\":" << position
                      << ",\"seq_id\":0,\"input_token_id\":" << token
                      << ",\"logits_byte_offset\":" << i * bytes << ",\"logits_bytes\":" << bytes
                      << ",\"all_finite\":true,\"argmax_token_id\":" << best - logits.begin()
                      << ",\"max_logit\":" << *best << ",\"callback_captures\":";
        size_t count = 0;
        for (const auto & entry : capture.occurrences) {
            count += entry.second;
        }
        token_records << count;
        if (opts.hf_gdn_l2_control) {
            token_records << ",\"hf_gdn_l2_control_count\":" << capture.control_count
                          << ",\"evaluation_epoch\":" << capture.evaluation_epoch;
        }
        if (opts.hf_qsa_f32_control) {
            token_records << ",\"hf_qsa_f32_control_count\":" << capture.qsa_control_count;
            if (!opts.hf_gdn_l2_control) token_records << ",\"evaluation_epoch\":" << capture.evaluation_epoch;
        }
        token_records << ",\"diagnostic_decode_with_callbacks_ms\":" << milliseconds << "}\n";
        token_records.flush();
        ++meta.completed_tokens;
    }
    capture.check_control_totals(opts.tokens.size(), opts.cache_warmup_passes);
    capture.check_qsa_probe_totals(opts.tokens.size());
    capture.check_gdn_state_probe_totals(opts.tokens.size());
    logits_file.close();
    token_records.close();
    capture.records.close();
    if (capture.control_records.is_open()) {
        capture.control_records.close();
    }
    if (capture.qsa_control_records.is_open()) {
        capture.qsa_control_records.close();
    }
    if (capture.qsa_probe_records) {
        capture.qsa_probe_records->close_checked();
    }
    if (capture.gdn_state_probe_records) {
        capture.gdn_state_probe_records->close_checked();
    }
}
} // namespace

int main(int argc, char ** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        usage();
        return 0;
    }
    Options opts;
    Metadata meta;
    std::unique_ptr<Capture> capture;
    try {
        if (std::string_view(CORE_ORACLE_LIBRARY_REVISION) != revision) {
            throw std::runtime_error("library provenance revision does not match the pinned production source");
        }
        opts = parse_options(argc, argv);
        const auto parent = opts.output.has_parent_path() ? opts.output.parent_path() : fs::path(".");
        if (!fs::is_directory(parent) || !fs::create_directory(opts.output)) {
            throw std::runtime_error("output directory must be new and its parent must already exist");
        }
        capture = std::make_unique<Capture>(opts.output, opts.intermediates, opts.hf_gdn_l2_control, opts.hf_qsa_f32_control,
            opts.all_layers, opts.gdn_probe_layers, opts.qsa_probe_layers);
        write_metadata(opts, meta, *capture, "initializing");
        run(opts, meta, *capture);
        write_metadata(opts, meta, *capture, "complete");
        std::fprintf(stderr, "core-oracle: saved %zu rows of %d logits and %zu callback tensors\n",
            meta.completed_tokens, vocab_size, capture->captures);
        return 0;
    } catch (const std::exception & ex) {
        if (capture) {
            try {
                write_metadata(opts, meta, *capture, "failed", ex.what());
            } catch (...) {
                std::fprintf(stderr, "core-oracle: could not write failure metadata\n");
            }
        }
        std::fprintf(stderr, "core-oracle: %s\n", ex.what());
        if (opts.tokens.empty()) {
            usage();
        }
        return 1;
    }
}
