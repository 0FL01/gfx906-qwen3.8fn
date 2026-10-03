#include "session_cli.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point begin) {
    return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
}

int legacy_request(const qwen::session_cli::Options& options) {
    const auto& config = options.config;
    const auto& model = options.model;
    const auto& logits_path = options.logits_path;
    const auto& tokens = options.tokens;
    const int generate = options.generate;
    const bool ignore_eos = options.ignore_eos;
    std::ofstream logits_file;
    if (!logits_path.empty()) {
        if (std::filesystem::exists(logits_path)) throw std::runtime_error("refusing existing logits artifact");
        logits_file.open(logits_path, std::ios::binary);
        if (!logits_file) throw std::runtime_error("cannot open logits artifact");
    }
    std::cout << std::setprecision(12);
    std::cout << "{\"kind\":\"session_source\",\"revision\":\"" << CORE_REVISION
              << "\",\"dirty\":" << CORE_DIRTY << ",\"model\":" << std::quoted(model)
              << ",\"capacity\":" << config.capacity << ",\"expert_slots\":" << config.expert_slots
              << ",\"trace\":" << (!config.trace_directory.empty() ? "true" : "false")
              << ",\"sampling\":\"greedy_diagnostic\",\"runtime\":\"own_48_layer_HIP\"}\n";
    std::cout.flush();
    const auto load_begin = std::chrono::steady_clock::now();
    auto session = std::make_unique<qwen::Session>(model, config);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_begin).count();
    std::span<const float> logits;
    auto consume = [&](int token) {
        logits = session->step(token);
        if (logits_file.is_open()) {
            logits_file.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size_bytes()));
            if (!logits_file) throw std::runtime_error("logits write failed");
        }
        const auto stats = session->stats();
        const auto best = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::cout << "{\"kind\":\"session_token\",\"position\":" << stats.consumed_tokens - 1
                  << ",\"token\":" << token << ",\"argmax\":" << best
                  << ",\"completed_ms\":" << stats.last_completed_ms
                  << ",\"expert_hits\":" << stats.expert_hits << ",\"expert_misses\":" << stats.expert_misses
                  << ",\"expert_upload_bytes\":" << stats.expert_upload_bytes << ",\"finite\":true}\n";
        std::cout.flush();
    };
    const auto request_begin = std::chrono::steady_clock::now();
    for (int token : tokens) consume(token);
    int outputs = 0;
    for (; outputs < generate;) {
        const int next = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        std::cout << "{\"kind\":\"session_output\",\"index\":" << outputs << ",\"token\":" << next << "}\n";
        ++outputs;
        if (next == 248046 && !ignore_eos) break;
        // Last emitted token remains pending; there is no unnecessary decode.
        if (outputs < generate) consume(next);
    }
    const double request_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - request_begin).count();
    const auto stats = session->stats();
    session.reset();
    if (logits_file.is_open()) { logits_file.close(); if (!logits_file) throw std::runtime_error("logits close failed"); }
    std::cout << "{\"kind\":\"session_complete\",\"input_tokens\":" << tokens.size()
              << ",\"output_tokens\":" << outputs << ",\"consumed_tokens\":" << stats.consumed_tokens
              << ",\"load_ms\":" << load_ms << ",\"request_ms\":" << request_ms
              << ",\"scope\":\"ordered_decode_no_prefill_no_sampling_speed_claim\",\"passed\":true}\n";
    std::cout.flush(); if (!std::cout) throw std::runtime_error("stdout failed");
    return 0;
}

void json_string(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::cout << '"';
    for (const unsigned char c : text) {
        if (c == '"' || c == '\\') std::cout << '\\' << static_cast<char>(c);
        else if (c < 0x20) std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
        else std::cout << static_cast<char>(c);
    }
    std::cout << '"';
}

void token_ids(std::span<const std::int32_t> ids) {
    std::cout << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << ids[i];
    }
    std::cout << ']';
}

int full_request(const qwen::session_cli::Options& options) {
    constexpr auto vocabulary = static_cast<std::size_t>(qwen::session_cli::vocabulary);
    const bool diagnostic_rows = !options.sample || !options.config.trace_directory.empty() || !options.logits_path.empty();
    // All CLI/sampler storage is bounded and prepared before the model is loaded.
    std::vector<std::int32_t> generated(static_cast<std::size_t>(options.generate));
    std::vector<float> probabilities(options.sample ? vocabulary : 0);
    std::unique_ptr<qwen::Sampler> sampler;
    if (options.sample) sampler = std::make_unique<qwen::Sampler>(vocabulary, options.sampling);
    std::ofstream logits_file;
    if (!options.logits_path.empty()) {
        if (std::filesystem::exists(options.logits_path)) throw std::runtime_error("refusing existing logits artifact");
        logits_file.open(options.logits_path, std::ios::binary);
        if (!logits_file) throw std::runtime_error("cannot open logits artifact");
    }
    const bool primary_sampling = options.sample && options.sampling.temperature == 1.0 &&
        options.sampling.top_p == 0.95 && options.sampling.top_k == 20;
    std::cout << std::setprecision(17);
    std::cout << "{\"kind\":\"session_request_source\",\"protocol\":1,\"revision\":\"" << CORE_REVISION
              << "\",\"dirty\":" << (CORE_DIRTY != 0 ? "true" : "false") << ",\"model\":";
    json_string(options.model);
    std::cout << ",\"runtime\":\"own_48_layer_HIP\",\"kv\":\"Q4_0\",\"candidate\":true,\"mtp\":false,\"mtp_acceptance\":null"
                 ",\"series\":\"" << (primary_sampling ? "primary_sampling" : options.sample ? "custom_sampling" : "greedy_diagnostic")
              << "\",\"session_start\":\"fresh\",\"prefix_reuse\":false,\"expert_cache_warmness\":\"unknown\""
                 ",\"config\":{\"capacity\":" << options.config.capacity << ",\"expert_slots\":" << options.config.expert_slots
              << ",\"max_batch_tokens\":" << options.config.max_batch_tokens
              << ",\"requested_output_tokens\":" << options.generate << ",\"ignore_eos\":" << (options.ignore_eos ? "true" : "false")
              << ",\"trace\":" << (!options.config.trace_directory.empty() ? "true" : "false")
              << ",\"trace_directory\":";
    json_string(options.config.trace_directory);
    std::cout << ",\"logits_path\":"; json_string(options.logits_path);
    std::cout << ",\"diagnostic_rows\":" << (diagnostic_rows ? "true" : "false") << "},\"sampling\":{\"mode\":\""
              << (options.sample ? "stochastic" : "greedy_diagnostic") << '"';
    if (options.sample)
        std::cout << ",\"seed\":" << options.sampling.seed << ",\"temperature\":" << options.sampling.temperature
                  << ",\"top_p\":" << options.sampling.top_p << ",\"top_k\":" << options.sampling.top_k
                  << ",\"rng\":\"mt19937_64_high53_ascending_id_cdf\",\"baseline_rng_equivalent\":false";
    std::cout << "},\"timing_scope\":\"completed_wall_sampling_and_cli_io_excludes_load_cleanup\""
                 ",\"pp_scope\":\"completed_prompt_calls_and_row_io\""
                 ",\"tg_scope\":\"remaining_output_forwards_sampling_and_io\",\"prompt_ids\":";
    token_ids(options.tokens);
    std::cout << "}\n";
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("stdout failed");
    const auto load_begin = Clock::now();
    auto session = std::make_unique<qwen::Session>(options.model, options.config);
    const double load_ms = elapsed_ms(load_begin);
    std::span<const float> last_logits;
    std::size_t pp_calls = 0, tg_forwards = 0;
    auto consume_window = [&](std::span<const std::int32_t> ids, bool prompt) {
        const auto position = session->stats().consumed_tokens;
        const auto logits = prompt ? session->step_batch(ids) : session->step(ids[0]);
        if (logits.size() != ids.size() * vocabulary) throw std::runtime_error("unexpected logit extent");
        const auto stats = session->stats();
        if (stats.consumed_tokens != position + ids.size()) throw std::runtime_error("unexpected consumed count");
        // Consume artifact/diagnostic rows now, never after the next accepted call.
        // Completion/cache counters belong to the entire call, not individual rows.
        for (std::size_t row = 0; row < ids.size(); ++row) {
            const auto values = logits.subspan(row * vocabulary, vocabulary);
            if (logits_file.is_open()) {
                logits_file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
                if (!logits_file) throw std::runtime_error("logits write failed");
            }
            if (diagnostic_rows) {
                const auto best = std::max_element(values.begin(), values.end()) - values.begin();
                std::cout << "{\"kind\":\"session_request_row\",\"protocol\":1,\"phase\":\"" << (prompt ? "pp" : "tg")
                          << "\",\"position\":" << position + row << ",\"token\":" << ids[row]
                          << ",\"argmax\":" << best << ",\"finite\":true}\n";
            }
            last_logits = values;
        }
        if (diagnostic_rows)
            std::cout << "{\"kind\":\"session_request_window\",\"protocol\":1,\"phase\":\"" << (prompt ? "pp" : "tg")
                      << "\",\"first_position\":" << position << ",\"rows\":" << ids.size()
                      << ",\"completed_ms\":" << stats.last_completed_ms
                      << ",\"expert_hits\":" << stats.expert_hits << ",\"expert_misses\":" << stats.expert_misses
                      << ",\"expert_upload_bytes\":" << stats.expert_upload_bytes << "}\n";
        if (!std::cout) throw std::runtime_error("stdout failed");
    };
    auto emit = [&](std::size_t index) {
        std::int32_t next;
        if (sampler) {
            sampler->distribution(last_logits, probabilities);
            next = static_cast<std::int32_t>(sampler->draw(probabilities));
        } else next = static_cast<std::int32_t>(std::max_element(last_logits.begin(), last_logits.end()) - last_logits.begin());
        generated[index] = next;
        std::cout << "{\"kind\":\"session_request_output\",\"protocol\":1,\"index\":" << index
                  << ",\"token\":" << next << "}\n";
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("stdout failed");
        return next == qwen::session_cli::eos_token && !options.ignore_eos;
    };
    const auto request_begin = Clock::now();
    const std::span<const std::int32_t> prompt(options.tokens);
    for (std::size_t first = 0; first < prompt.size();) {
        const auto count = std::min(static_cast<std::size_t>(options.config.max_batch_tokens), prompt.size() - first);
        consume_window(prompt.subspan(first, count), true);
        first += count; ++pp_calls;
    }
    const double pp_ms = elapsed_ms(request_begin);
    std::size_t outputs = 0;
    bool stopped_eos = false;
    double first_output_ms = 0, first_output_after_pp_ms = 0, tg_ms = 0;
    if (!generated.empty()) {
        const auto first_begin = Clock::now();
        stopped_eos = emit(outputs++); // Draw from the final prompt row, before it expires.
        first_output_after_pp_ms = elapsed_ms(first_begin);
        first_output_ms = elapsed_ms(request_begin);
        if (!stopped_eos && outputs < generated.size()) {
            const auto tg_begin = Clock::now();
            while (!stopped_eos && outputs < generated.size()) {
                consume_window(std::span<const std::int32_t>(&generated[outputs - 1], 1), false);
                ++tg_forwards;
                stopped_eos = emit(outputs++);
            }
            tg_ms = elapsed_ms(tg_begin);
        }
    }
    // The final emitted token (also an early EOS) is deliberately not forwarded.
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("stdout failed");
    if (logits_file.is_open()) { logits_file.flush(); if (!logits_file) throw std::runtime_error("logits flush failed"); }
    const double total_ms = elapsed_ms(request_begin);
    const auto stats = session->stats();
    if (stats.consumed_tokens != prompt.size() + (outputs > 0 ? outputs - 1 : 0) ||
        tg_forwards != (outputs > 0 ? outputs - 1 : 0) || (sampler && sampler->random_draws() != outputs))
        throw std::runtime_error("request accounting mismatch");
    session.reset();
    if (logits_file.is_open()) { logits_file.close(); if (!logits_file) throw std::runtime_error("logits close failed"); }
    std::cout << "{\"kind\":\"session_request_complete\",\"protocol\":1,\"candidate\":true,\"mtp\":false,\"mtp_acceptance\":null"
              << ",\"input_tokens\":" << prompt.size() << ",\"output_tokens\":" << outputs
              << ",\"consumed_tokens\":" << stats.consumed_tokens << ",\"pp_calls\":" << pp_calls
              << ",\"tg_forwards\":" << tg_forwards << ",\"random_draws\":" << (sampler ? sampler->random_draws() : 0)
              << ",\"load_ms\":" << load_ms << ",\"pp_ms\":" << pp_ms << ",\"tg_ms\":" << tg_ms
              << ",\"total_ms\":" << total_ms << ",\"first_output_ms\":";
    if (outputs) std::cout << first_output_ms; else std::cout << "null";
    std::cout << ",\"first_output_after_pp_ms\":" << first_output_after_pp_ms
              << ",\"stop_reason\":\"" << (stopped_eos ? "eos" : outputs ? "output_limit" : "prompt_only")
              << "\",\"pending_token\":";
    if (outputs) std::cout << generated[outputs - 1]; else std::cout << "null";
    std::cout << ",\"expert_hits\":" << stats.expert_hits << ",\"expert_misses\":" << stats.expert_misses
              << ",\"expert_upload_bytes\":" << stats.expert_upload_bytes << ",\"generated_ids\":";
    token_ids(std::span<const std::int32_t>(generated).first(outputs));
    std::cout << ",\"scope\":\"candidate_non_mtp_full_request_no_speed_claim\",\"passed\":true}\n";
    std::cout.flush();
    if (!std::cout) throw std::runtime_error("stdout failed");
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = qwen::session_cli::parse(argc, argv);
        if (options.help) { std::cout << qwen::session_cli::usage; return 0; }
        return options.request_protocol() ? full_request(options) : legacy_request(options);
    } catch (const std::exception& e) { std::cerr << "core-session: " << e.what() << '\n'; return 1; }
}
