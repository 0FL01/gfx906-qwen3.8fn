#include "session.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
int integer(const std::string& text) {
    int n = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), n);
    if (error != std::errc{} || end != text.data() + text.size()) throw std::invalid_argument("invalid integer: " + text);
    return n;
}
}

int main(int argc, char** argv) {
    try {
        qwen::SessionConfig config;
        std::string model, logits_path;
        std::vector<std::int32_t> tokens;
        int generate = 0; bool ignore_eos = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--trace" || arg == "--logits" || arg == "--capacity" || arg == "--slots" || arg == "--generate") {
                if (++i == argc) throw std::invalid_argument("missing option value");
                if (arg == "--trace") config.trace_directory = argv[i];
                else if (arg == "--logits") logits_path = argv[i];
                else if (arg == "--capacity") config.capacity = integer(argv[i]);
                else if (arg == "--slots") config.expert_slots = integer(argv[i]);
                else generate = integer(argv[i]);
            } else if (arg == "--ignore-eos") ignore_eos = true;
            else if (arg == "--help") {
                std::cout << "core-session [--capacity N] [--slots N] [--trace DIR] [--logits FILE] [--generate N] [--ignore-eos] model.gguf token-id...\n";
                return 0;
            } else if (model.empty()) model = arg;
            else tokens.push_back(integer(arg));
        }
        if (model.empty() || tokens.empty() || generate < 0) throw std::invalid_argument("model/token IDs required");
        for (int token : tokens) if (token < 0 || token >= 248320) throw std::invalid_argument("token outside vocabulary");
        if (config.capacity < 4 || tokens.size() + static_cast<std::size_t>(std::max(0, generate - 1)) > static_cast<std::size_t>(config.capacity))
            throw std::invalid_argument("request exceeds capacity");
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
    } catch (const std::exception& e) { std::cerr << "core-session: " << e.what() << '\n'; return 1; }
}
