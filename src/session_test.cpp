#include "session.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr std::size_t vocabulary = 248320;
constexpr std::array<std::int32_t, 4> teacher_ids{248044, 100, 101, 102};
// Session::step executes 48 layers; Layer::ffn visits ten distinct expert IDs.
// With one retained slot only the first route per layer can possibly hit.
constexpr std::uint64_t routes_per_step = 48 * 10;
constexpr std::uint64_t minimum_misses_per_step = 48 * 9;
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_zero_stats(const qwen::SessionStats& stats, const std::string& where) {
    require(stats.consumed_tokens == 0 && stats.expert_hits == 0 &&
            stats.expert_misses == 0 && stats.expert_upload_bytes == 0 &&
            stats.last_completed_ms == 0, where + ": reset statistics are not zero");
}

void check_completed_stats(const qwen::SessionStats& stats, const qwen::SessionStats& previous,
                           std::size_t position, const std::string& where) {
    require(stats.consumed_tokens == position + 1, where + ": consumed count");
    require(std::isfinite(stats.last_completed_ms) && stats.last_completed_ms >= 0,
            where + ": invalid completed time");
    require(stats.expert_hits >= previous.expert_hits && stats.expert_misses >= previous.expert_misses,
            where + ": expert counters decreased");
    const auto hits = stats.expert_hits - previous.expert_hits;
    const auto misses = stats.expert_misses - previous.expert_misses;
    require(hits <= routes_per_step && misses <= routes_per_step && hits + misses == routes_per_step,
            where + ": incomplete expert route accounting");
    require(misses >= minimum_misses_per_step, where + ": single-slot reuse was not exercised");
    require(stats.expert_upload_bytes > previous.expert_upload_bytes,
            where + ": missing expert upload accounting");
}

void check_finite_logits(std::span<const float> logits, const std::string& where) {
    require(logits.size() == vocabulary, where + ": logits size is not 248320");
    for (std::size_t i = 0; i < logits.size(); ++i)
        if (!std::isfinite(logits[i]))
            throw std::runtime_error(where + ": nonfinite logit at " + std::to_string(i));
}

void check_exact_logits(std::span<const float> actual, std::span<const float> expected,
                        const std::string& where) {
    require(actual.size() == vocabulary && expected.size() == vocabulary, where + ": logits size");
    for (std::size_t i = 0; i < vocabulary; ++i) {
        const auto actual_bits = std::bit_cast<std::uint32_t>(actual[i]);
        const auto expected_bits = std::bit_cast<std::uint32_t>(expected[i]);
        if (actual_bits != expected_bits)
            throw std::runtime_error(where + ": logit bits differ at " + std::to_string(i) +
                                     " (actual=" + std::to_string(actual_bits) +
                                     ", expected=" + std::to_string(expected_bits) + ")");
    }
}

void expect_rejection(qwen::Session& session, std::int32_t token,
                      std::span<const float> prior_logits, std::span<const float> saved_logits,
                      const std::string& where) {
    const auto before = session.stats();
    bool rejected = false;
    try {
        static_cast<void>(session.step(token));
    } catch (const std::invalid_argument&) {
        rejected = true;
    } catch (const std::exception& error) {
        throw std::runtime_error(where + ": expected std::invalid_argument, got " + error.what());
    }
    require(rejected, where + ": step accepted a rejected input");
    const auto after = session.stats();
    require(after.consumed_tokens == before.consumed_tokens && after.expert_hits == before.expert_hits &&
            after.expert_misses == before.expert_misses && after.expert_upload_bytes == before.expert_upload_bytes &&
            std::bit_cast<std::uint64_t>(after.last_completed_ms) ==
                std::bit_cast<std::uint64_t>(before.last_completed_ms),
            where + ": rejection mutated statistics");
    // These argument checks precede the mutation/invalidating try block in step().
    // Its existing host_logits storage must therefore still contain the prior result.
    check_exact_logits(prior_logits, saved_logits, where + ": rejection mutated logits");
}

void json_string(std::ostream& out, std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    out.put('"');
    for (char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') { out.put('\\'); out.put(c); }
        else if (byte < 0x20) { out << "\\u00"; out.put(hex[byte >> 4]); out.put(hex[byte & 15]); }
        else out.put(c);
    }
    out.put('"');
}

void json_stats(std::ostream& out, const qwen::SessionStats& stats) {
    out << "{\"consumed_tokens\":" << stats.consumed_tokens
        << ",\"expert_hits\":" << stats.expert_hits
        << ",\"expert_misses\":" << stats.expert_misses
        << ",\"expert_upload_bytes\":" << stats.expert_upload_bytes
        << ",\"last_completed_ms\":" << stats.last_completed_ms << '}';
}

void json_steps(std::ostream& out, const std::array<qwen::SessionStats, teacher_ids.size()>& steps) {
    out << '[';
    for (std::size_t i = 0; i < steps.size(); ++i) {
        if (i) out << ',';
        out << "{\"token\":" << teacher_ids[i] << ",\"stats\":";
        json_stats(out, steps[i]);
        out << '}';
    }
    out << ']';
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: core-session-test MODEL.gguf\n";
        return 2;
    }
    try {
        require(argv[1][0] != '\0', "model path is empty");
        qwen::SessionConfig config;
        config.capacity = 4;
        config.expert_slots = 1;
        // Leave trace_directory empty: no synchronous diagnostic captures.
        std::vector<float> reference(teacher_ids.size() * vocabulary);
        const auto saved_logits = [&](std::size_t position) {
            return std::span<const float>(reference).subspan(position * vocabulary, vocabulary);
        };
        std::array<qwen::SessionStats, teacher_ids.size()> baseline{}, replay{};
        qwen::SessionStats initial{}, after_reset{}, final_reset{};
        std::size_t finite_values = 0, compared_values = 0;
        unsigned invalid_token_rejections = 0, capacity_rejections = 0;
        {
            // Impl initializes Model/PLE before validating capacity/slots. Do not
            // construct additional Sessions for capacity 0/3 or slots 0 here.
            qwen::Session session(argv[1], config);
            initial = session.stats();
            check_zero_stats(initial, "construction");
            auto previous = initial;
            std::span<const float> logits;
            for (std::size_t i = 0; i < teacher_ids.size(); ++i) {
                const auto where = "baseline step " + std::to_string(i);
                logits = session.step(teacher_ids[i]);
                check_finite_logits(logits, where);
                finite_values += logits.size();
                baseline[i] = session.stats();
                check_completed_stats(baseline[i], previous, i, where);
                previous = baseline[i];
                std::copy(logits.begin(), logits.end(),
                          reference.begin() + static_cast<std::ptrdiff_t>(i * vocabulary));
            }
            // A valid ID isolates the capacity guard from the token-ID guard.
            expect_rejection(session, 100, logits, saved_logits(3), "baseline full capacity");
            ++capacity_rejections;
            logits = {};
            session.reset();
            after_reset = session.stats();
            check_zero_stats(after_reset, "reset after capacity rejection");
            previous = after_reset;
            for (std::size_t i = 0; i < teacher_ids.size(); ++i) {
                const auto where = "reset replay step " + std::to_string(i);
                logits = session.step(teacher_ids[i]);
                check_finite_logits(logits, where);
                finite_values += logits.size();
                check_exact_logits(logits, saved_logits(i), where);
                compared_values += logits.size();
                replay[i] = session.stats();
                check_completed_stats(replay[i], previous, i, where);
                previous = replay[i];
                if (i == 0) {
                    // Capacity remains available. Continue WITHOUT reset after
                    // both errors; subsequent logits must match the clean run.
                    expect_rejection(session, -1, logits, saved_logits(i), "negative token");
                    ++invalid_token_rejections;
                    expect_rejection(session, static_cast<std::int32_t>(vocabulary), logits,
                                     saved_logits(i), "token at vocabulary limit");
                    ++invalid_token_rejections;
                }
            }
            expect_rejection(session, 100, logits, saved_logits(3), "replay full capacity");
            ++capacity_rejections;
            logits = {};
            session.reset();
            final_reset = session.stats();
            check_zero_stats(final_reset, "final reset");
        } // Complete RAII Session cleanup before emitting any success record.

        std::ostringstream record;
        record.imbue(std::locale::classic());
        record << std::setprecision(std::numeric_limits<double>::max_digits10);
        record << "{\"kind\":\"session_test_complete\",\"revision\":";
        json_string(record, CORE_REVISION);
        record << ",\"dirty\":" << (CORE_DIRTY ? "true" : "false") << ",\"model\":";
        json_string(record, argv[1]);
        record << ",\"runtime\":\"own_48_layer_HIP\",\"capacity\":" << config.capacity
               << ",\"expert_slots\":" << config.expert_slots << ",\"trace\":false"
               << ",\"sampling\":\"teacher_forced\",\"scope\":\"reset_replay_and_rejection_self_parity\""
               << ",\"performance_claim\":false,\"initial_stats\":";
        json_stats(record, initial);
        record << ",\"baseline_steps\":";
        json_steps(record, baseline);
        record << ",\"after_reset_stats\":";
        json_stats(record, after_reset);
        record << ",\"replay_steps\":";
        json_steps(record, replay);
        record << ",\"final_reset_stats\":";
        json_stats(record, final_reset);
        record << ",\"checks\":{\"finite_logit_values\":" << finite_values
               << ",\"bitwise_compared_logit_values\":" << compared_values
               << ",\"invalid_token_rejections\":" << invalid_token_rejections
               << ",\"capacity_rejections\":" << capacity_rejections
               << ",\"rejection_stats_preserved\":true,\"rejection_logits_preserved\":true"
               << ",\"invalid_token_continuation\":true,\"reset_reused\":true"
               << ",\"minimum_slot_reuse_misses_per_step\":" << minimum_misses_per_step << '}'
               << ",\"invalid_config_checks\":{\"tested\":false,\"capacity_cases\":[0,3],\"expert_slots_cases\":[0],"
               << "\"reason\":\"constructor validation follows Model and PLE initialization\"}"
               << ",\"passed\":true}\n";
        require(static_cast<bool>(record), "JSON record formatting failed");
        std::cout << record.str();
        std::cout.flush();
        require(static_cast<bool>(std::cout), "stdout failed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core-session-test: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "core-session-test: non-standard exception\n";
        return 1;
    }
}
