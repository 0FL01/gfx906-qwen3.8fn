#include "session.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifndef CORE_REVISION
#error "CORE_REVISION required"
#endif
#ifndef CORE_DIRTY
#error "CORE_DIRTY required"
#endif

int main(int argc, char** argv) {
    const bool tokens32 = argc == 4 && std::string_view(argv[1]) == "--tokens32";
    if (argc != 3 && !tokens32) {
        std::cerr << "usage: core-prefill-trace [--tokens32] MODEL.gguf NEW_TRACE_DIRECTORY\n";
        return 2;
    }
    try {
        constexpr std::size_t vocabulary = 248320;
        std::vector<std::int32_t> ids(tokens32 ? 32 : 4);
        ids[0] = 248044;
        for (std::size_t t = 1; t < ids.size(); ++t) ids[t] = 99 + static_cast<std::int32_t>(t);
        const auto trace = std::filesystem::absolute(argv[tokens32 ? 3 : 2]);
        std::uintmax_t n1_end = 0, batch_end = 0;
        bool numeric_gate_passed = true;
        std::cout << std::setprecision(17);
        std::cout << "{\"kind\":\"prefill_trace_source\",\"revision\":"
                  << std::quoted(CORE_REVISION) << ",\"dirty\":" << CORE_DIRTY
                  << ",\"trace\":" << std::quoted(trace.string())
                  << ",\"diagnostic_only\":true,\"absolute_gate\":0.02,\"relative_gate\":0.002";
        if (tokens32) std::cout << ",\"protocol\":2,\"token_count\":32";
        std::cout << "}\n";
        {
            qwen::SessionConfig config;
            config.capacity = 40;
            config.expert_slots = 1;
            config.max_batch_tokens = 32;
            config.trace_directory = trace.string();
            qwen::Session session(argv[tokens32 ? 2 : 1], config);
            std::vector<float> reference(ids.size() * vocabulary);
            for (std::size_t t = 0; t < ids.size(); ++t) {
                const auto output = session.step(ids[t]);
                if (output.size() != vocabulary) throw std::runtime_error("N1 logit extent");
                std::copy(output.begin(), output.end(), reference.begin() + t * vocabulary);
            }
            n1_end = std::filesystem::file_size(trace / "tensors.jsonl");
            session.reset();
            const auto output = session.step_batch(ids);
            if (output.size() != reference.size()) throw std::runtime_error(tokens32 ? "N32 logit extent" : "N4 logit extent");
            batch_end = std::filesystem::file_size(trace / "tensors.jsonl");
            for (std::size_t t = 0; t < ids.size(); ++t) {
                double maximum = 0, sum_squared = 0, maximum_ratio = 0;
                std::uint64_t violations = 0, bit_mismatches = 0;
                for (std::size_t j = 0; j < vocabulary; ++j) {
                    const float actual = output[t * vocabulary + j];
                    const float expected = reference[t * vocabulary + j];
                    if (!std::isfinite(actual) || !std::isfinite(expected))
                        throw std::runtime_error("nonfinite diagnostic logits");
                    const double error = std::abs(double(actual) - double(expected));
                    const double bound = 0.02 + 0.002 * std::abs(double(expected));
                    maximum = std::max(maximum, error);
                    maximum_ratio = std::max(maximum_ratio, error / bound);
                    sum_squared += error * error;
                    violations += error > bound;
                    bit_mismatches += std::bit_cast<std::uint32_t>(actual) !=
                                      std::bit_cast<std::uint32_t>(expected);
                }
                numeric_gate_passed = numeric_gate_passed && violations == 0;
                std::cout << "{\"kind\":\"prefill_trace_row\",\"position\":" << t
                          << ",\"token\":" << ids[t] << ",\"elements\":" << vocabulary
                          << ",\"max_abs\":" << maximum << ",\"rms\":"
                          << std::sqrt(sum_squared / vocabulary)
                          << ",\"max_bound_ratio\":" << maximum_ratio
                          << ",\"violations\":" << violations
                          << ",\"bit_mismatches\":" << bit_mismatches << "}\n";
            }
        }
        std::cout << "{\"kind\":\"prefill_trace_complete\",\"diagnostic_only\":true,"
                     "\"capture_completed\":true,\"raii_cleanup_completed\":true,"
                     "\"numeric_gate_passed\":" << (numeric_gate_passed ? "true" : "false")
                  << ",\"n1_manifest_end_bytes\":" << n1_end
                  << ",\"batch_manifest_end_bytes\":" << batch_end << "}\n";
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("diagnostic stdout write");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core-prefill-trace: " << error.what() << '\n';
        return 1;
    }
}
