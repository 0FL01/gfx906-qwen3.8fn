#pragma once

#include "sampling.hpp"
#include "session.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qwen::session_cli {

inline constexpr int vocabulary = 248320;
inline constexpr int eos_token = 248046;
inline constexpr int maximum_capacity = 131072;

struct Options {
    SessionConfig config;
    std::string model, logits_path;
    std::vector<std::int32_t> tokens;
    int generate = 0;
    bool ignore_eos = false, help = false, sample = false;
    bool explicit_prefill = false;
    // Parsing stores configuration only; it never constructs or advances an RNG.
    SamplingConfig sampling{0};
    bool request_protocol() const noexcept { return explicit_prefill || sample; }
};

template<class Number> inline Number number(std::string_view text, std::string_view label) {
    if (text.empty()) throw std::invalid_argument("empty " + std::string(label));
    Number result{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), result);
    if (error != std::errc{} || end != text.data() + text.size())
        throw std::invalid_argument("invalid " + std::string(label) + ": " + std::string(text));
    return result;
}

// Validate the complete request before artifact creation, model construction or
// sampler construction. Options may appear before/after the model and token IDs.
inline Options parse(int argc, const char* const* argv) {
    if (argc < 1 || argv == nullptr) throw std::invalid_argument("invalid argument array");
    for (int i = 0; i < argc; ++i)
        if (argv[i] == nullptr) throw std::invalid_argument("null argument");
    Options options;
    std::optional<std::uint64_t> seed;
    bool filters = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--trace" || arg == "--logits" || arg == "--capacity" || arg == "--slots" ||
            arg == "--generate" || arg == "--prefill-chunk" || arg == "--seed" ||
            arg == "--temperature" || arg == "--top-p" || arg == "--top-k") {
            if (++i == argc) throw std::invalid_argument("missing option value: " + std::string(arg));
            const std::string_view value(argv[i]);
            if (value.empty()) throw std::invalid_argument("empty option value: " + std::string(arg));
            if (arg == "--trace") options.config.trace_directory = value;
            else if (arg == "--logits") options.logits_path = value;
            else if (arg == "--capacity") options.config.capacity = number<int>(value, arg);
            else if (arg == "--slots") options.config.expert_slots = number<int>(value, arg);
            else if (arg == "--generate") options.generate = number<int>(value, arg);
            else if (arg == "--prefill-chunk") {
                options.config.max_batch_tokens = number<int>(value, arg);
                options.explicit_prefill = true;
            } else if (arg == "--seed") {
                if (seed) throw std::invalid_argument("duplicate --seed");
                seed = number<std::uint64_t>(value, arg);
            } else {
                filters = true;
                if (arg == "--temperature") options.sampling.temperature = number<double>(value, arg);
                else if (arg == "--top-p") options.sampling.top_p = number<double>(value, arg);
                else options.sampling.top_k = number<int>(value, arg);
            }
        } else if (arg == "--ignore-eos") options.ignore_eos = true;
        else if (arg == "--sample") options.sample = true;
        else if (arg == "--help") options.help = true;
        else if (arg.starts_with("--")) throw std::invalid_argument("unknown option: " + std::string(arg));
        else if (options.model.empty()) {
            if (arg.empty()) throw std::invalid_argument("empty model path");
            options.model = arg;
        } else {
            if (options.tokens.size() == static_cast<std::size_t>(maximum_capacity))
                throw std::invalid_argument("too many prompt tokens");
            const auto token = number<std::int32_t>(arg, "token ID");
            if (token < 0 || token >= vocabulary) throw std::invalid_argument("token outside vocabulary");
            options.tokens.push_back(token);
        }
    }
    const auto& config = options.config;
    if (config.capacity < 4 || config.capacity > maximum_capacity || config.capacity % 4 != 0)
        throw std::invalid_argument("capacity must be a multiple of four in 4..131072");
    if (config.expert_slots < 1 || config.expert_slots > 512)
        throw std::invalid_argument("slots must be in 1..512");
    if (config.max_batch_tokens < 1 || config.max_batch_tokens > 1024)
        throw std::invalid_argument("prefill chunk must be in 1..1024");
    if (options.generate < 0 || options.generate > maximum_capacity)
        throw std::invalid_argument("generate must be in 0..131072");
    if (!std::isfinite(options.sampling.temperature) || options.sampling.temperature <= 0.0 ||
        !std::isfinite(options.sampling.top_p) || options.sampling.top_p <= 0.0 || options.sampling.top_p > 1.0 ||
        options.sampling.top_k < 0 || options.sampling.top_k > vocabulary)
        throw std::invalid_argument("invalid stochastic filter configuration");
    if (!options.sample && (seed || filters))
        throw std::invalid_argument("seed and sampling filters require --sample");
    if (options.sample && !seed) throw std::invalid_argument("--sample requires --seed UINT64");
    if (options.sample && options.generate == 0)
        throw std::invalid_argument("--sample requires positive --generate");
    if (!config.trace_directory.empty() && config.max_batch_tokens > 1)
        throw std::invalid_argument("--trace requires prefill chunk 1 to preserve trace order");
    if (seed) options.sampling.seed = *seed;
    if (options.help) return options;
    if (options.model.empty() || options.tokens.empty()) throw std::invalid_argument("model/token IDs required");
    // The first output comes from PP and the last remains pending, including EOS.
    // Use subtraction so neither a large prompt nor a large output count can wrap.
    const auto remaining_forwards = static_cast<std::size_t>(options.generate > 0 ? options.generate - 1 : 0);
    const auto capacity = static_cast<std::size_t>(config.capacity);
    if (options.tokens.size() > capacity || remaining_forwards > capacity - options.tokens.size())
        throw std::invalid_argument("request exceeds capacity");
    return options;
}

inline constexpr std::string_view usage =
    "core-session [--capacity N] [--slots N] [--trace DIR] [--logits FILE] [--generate N] [--ignore-eos] "
    "[--prefill-chunk N] [--sample --seed UINT64 [--temperature T] [--top-p P] [--top-k K]] model.gguf token-id...\n"
    "capacity: 4..131072, multiple of 4; slots: 1..512; prefill chunk: 1..1024 (default 1).\n"
    "Sampling requires --generate >0 and an explicit seed; defaults: temperature 1, top-p .95, top-k 20.\n"
    "Temperature must be finite >0, top-p finite in (0,1], top-k 0..248320 (0 disables top-k).\n"
    "Grouped logits are written token-major; --trace requires chunk 1. No sampling flags means greedy diagnostics.\n";

} // namespace qwen::session_cli
