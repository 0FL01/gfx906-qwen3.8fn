#include "session.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#ifndef CORE_REVISION
#error "core-prefill-failure-trace requires compiled 40-hex CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-prefill-failure-trace requires compiled CORE_DIRTY (0 or 1)"
#endif

namespace {
constexpr std::size_t vocabulary = 248320, prefix = 13200, captured_tokens = 8, batch = 1024;
constexpr int capacity = 16416, slots = 112;
// Upper bound for this driver's positive trace_first_token=13200 diagnostics,
// including the five layer-8 witnesses: up to 241 F32 files / 4,513,085 elements
// per token. This is not a default-tracing capture count. Metadata is separate.
constexpr std::uint64_t capture_files_bound = captured_tokens * 241;
constexpr std::uint64_t capture_payload_bound = captured_tokens * 4513085ULL * sizeof(float);
constexpr std::int32_t token_at(std::size_t position) {
    return position == 0 ? 248044 : 99 + static_cast<std::int32_t>(position);
}
constexpr bool revision_valid(std::string_view revision) {
    if (revision.size() != 40) return false;
    for (const char c : revision)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) return false;
    return true;
}
static_assert(revision_valid(CORE_REVISION));
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(prefix == 12 * batch + 912 && prefix + captured_tokens <= capacity);
static_assert(token_at(0) == 248044 && token_at(1) == 100 && token_at(13207) == 13306);
static_assert(capture_payload_bound == 144418720 && capture_files_bound == 1928);

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
void quote(std::ostream& out, std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    out << '"';
    for (const char c : text) {
        const auto byte = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (byte < 32) out << "\\u00" << hex[byte >> 4] << hex[byte & 15];
        else out << c;
    }
    out << '"';
}
void number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
}
struct Writer {
    std::uint64_t records = 0;
    template<class F> void emit(std::string_view kind, F fields) {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "{\"kind\":\"prefill_failure_trace_" << kind << "\",\"protocol\":1";
        fields(out);
        out << "}\n";
        require(static_cast<bool>(out), "JSONL formatting failed");
        std::cout << out.str();
        std::cout.flush();
        require(static_cast<bool>(std::cout), "JSONL output failed");
        ++records;
    }
};
struct Run {
    std::string trace;
    std::string_view mode, where = "trace_path";
    std::size_t offset = 0, rows = 0, prefix_calls = 0, windows = 0;
    std::uint64_t prefix_logit_values = 0, manifest_bytes = 0;
    qwen::SessionStats stats{};
    qwen::SessionRouteStats routes{};
    bool stats_available = false, session_constructed = false, cleanup_completed = false;
    void snapshot(const qwen::Session& session) {
        stats = session.stats();
        routes = session.route_stats();
        stats_available = true;
    }
};
struct CleanupScope {
    bool& completed;
    ~CleanupScope() { completed = true; }
};
void json_stats(std::ostream& out, const Run& run) {
    if (!run.stats_available) { out << "null"; return; }
    const auto& s = run.stats;
    out << "{\"consumed_tokens\":" << s.consumed_tokens << ",\"expert_hits\":" << s.expert_hits
        << ",\"expert_misses\":" << s.expert_misses << ",\"expert_upload_bytes\":" << s.expert_upload_bytes
        << ",\"last_completed_ms\":";
    number(out, s.last_completed_ms);
    out << '}';
}
void json_routes(std::ostream& out, const Run& run) {
    if (!run.stats_available) { out << "null"; return; }
    out << "{\"last_max_expert_group_assignments\":" << run.routes.last_max_expert_group_assignments
        << ",\"expert_groups_gt128\":" << run.routes.expert_groups_gt128 << '}';
}
void context(std::ostream& out, const Run& run) {
    out << ",\"mode\":"; quote(out, run.mode);
    out << ",\"trace_directory\":"; quote(out, run.trace);
    out << ",\"where\":"; quote(out, run.where);
    out << ",\"context\":{\"offset\":" << run.offset << ",\"rows\":" << run.rows;
    if (run.rows) out << ",\"first_token\":" << token_at(run.offset)
                      << ",\"last_token\":" << token_at(run.offset + run.rows - 1);
    out << "},\"actual_consumed\":";
    if (run.stats_available) out << run.stats.consumed_tokens;
    else out << "null";
    out << ",\"stats\":"; json_stats(out, run);
    out << ",\"route_stats\":"; json_routes(out, run);
}
struct Logits {
    std::size_t finite = 0, argmax = 0;
    float minimum = 0, maximum = 0;
};
Logits inspect(std::span<const float> values) {
    require(values.size() == vocabulary, "trace N1 logit extent");
    Logits result;
    result.minimum = result.maximum = values.front();
    for (std::size_t i = 0; i < values.size(); ++i) {
        require(std::isfinite(values[i]), "trace N1 nonfinite logits");
        ++result.finite;
        result.minimum = std::min(result.minimum, values[i]);
        if (values[i] > result.maximum) { result.maximum = values[i]; result.argmax = i; }
    }
    return result;
}
void source(Writer& writer, const Run& run, std::string_view model) {
    writer.emit("source", [&](std::ostream& out) {
        out << ",\"revision\":"; quote(out, CORE_REVISION);
        out << ",\"dirty\":" << bool(CORE_DIRTY) << ",\"model\":"; quote(out, model);
        out << ",\"mode\":"; quote(out, run.mode);
        out << ",\"trace_directory\":"; quote(out, run.trace);
        out << ",\"diagnostic_only\":true,\"parity_test\":false,\"speed_test\":false"
               ",\"model_variant_assumed\":\"qwen38-keep1-Q4_0\",\"target_kv\":\"Q4_0\",\"mtp\":false"
               ",\"weights_changed\":false,\"arithmetic_changed\":false,\"epsilon_changed\":false"
               ",\"quantization_changed\":false,\"tolerances_changed\":false,\"sampling\":\"none; teacher IDs\""
               ",\"source_formula\":\"position 0 = 248044; position p > 0 = 99 + p; through 13207 inclusive\""
            << ",\"capacity\":" << capacity << ",\"expert_slots\":" << slots << ",\"max_batch_tokens\":" << batch
            << ",\"trace_first_token\":" << prefix << ",\"trace_end_exclusive\":" << prefix + captured_tokens
            << ",\"prefix_tokens\":" << prefix << ",\"prefix_calls\":" << (run.mode == "n1" ? prefix : 13)
            << ",\"prefix_schedule\":";
        quote(out, run.mode == "n1" ? "13200 N1" : "12 N1024 + N912");
        out << ",\"trace_schedule\":\"eight N1 calls at 13200..13207\""
            << ",\"capture_files_upper_bound\":" << capture_files_bound
            << ",\"capture_f32_payload_upper_bound_bytes\":" << capture_payload_bound
            << ",\"capture_metadata_bytes_separate\":true,\"retained_logit_history_bytes\":0"
               ",\"reference_assumption\":\"Reported real4K N1/chunk self-parity passed; N1 correctness beyond 4K is not established. This run makes no parity claim.\""
               ",\"reported_original_failure\":{\"source_job\":\"1791018955150-729\",\"revision\":\"2e9848d43cf9f908dc8280f81e111c0cde86f01c\",\"dirty\":true"
               ",\"phase\":\"16K sequential N1\",\"offset\":13206,\"rows\":1,\"consumed_tokens\":13206"
               ",\"error\":\"router layer 8: numeric flag 2\",\"cause\":\"unknown; router check includes preceding sticky flags\"}"
               ",\"expected_records_no_error\":{\"source\":1,\"prefix_summary\":1,\"window\":8,\"complete\":1,\"total\":11}"
               ",\"failure_contract\":\"stop at first exception; snapshot stats before Session destruction; failure record after RAII; no complete record\""
               ",\"stats_semantics\":\"consumed_tokens counts successful windows; expert counters may include work in a failed window; route_stats retain the last successful call\"";
    });
}
void execute(Writer& writer, Run& run, const char* model) {
    run.trace = std::filesystem::absolute(run.trace).string();
    source(writer, run, model);
    {
        CleanupScope cleanup{run.cleanup_completed};
        qwen::SessionConfig config;
        config.capacity = capacity;
        config.expert_slots = slots;
        config.max_batch_tokens = static_cast<int>(batch);
        config.trace_directory = run.trace;
        config.trace_first_token = static_cast<int>(prefix);
        run.where = "construct_session";
        qwen::Session session(model, config);
        run.session_constructed = true;
        try {
            run.snapshot(session);
            // Reuse one small ID buffer. Session owns the sole current-logit span;
            // neither prefix nor diagnostic logits are copied or retained here.
            std::array<std::int32_t, batch> ids{};
            std::size_t offset = 0;
            while (offset < prefix) {
                run.where = "prefix_forward";
                run.offset = offset;
                run.rows = run.mode == "n1" ? 1 : std::min(batch, prefix - offset);
                for (std::size_t row = 0; row < run.rows; ++row) ids[row] = token_at(offset + row);
                const auto values = run.mode == "n1" ? session.step(ids[0]) :
                    session.step_batch(std::span<const std::int32_t>(ids).first(run.rows));
                run.snapshot(session);
                require(values.size() == run.rows * vocabulary, "prefix logit extent");
                offset += run.rows;
                ++run.prefix_calls;
                run.prefix_logit_values += values.size();
                require(run.stats.consumed_tokens == offset, "prefix consumed accounting");
            }
            run.where = "prefix_summary";
            run.offset = 0;
            run.rows = prefix;
            require(run.prefix_calls == (run.mode == "n1" ? prefix : 13), "prefix call accounting");
            run.manifest_bytes = std::filesystem::file_size(std::filesystem::path(run.trace) / "tensors.jsonl");
            require(run.manifest_bytes == 0, "capture range wrote prefix metadata");
            writer.emit("prefix_summary", [&](std::ostream& out) {
                context(out, run);
                out << ",\"completed_calls\":" << run.prefix_calls << ",\"returned_logit_values\":" << run.prefix_logit_values
                    << ",\"retained_logit_history_bytes\":0,\"trace_manifest_bytes\":" << run.manifest_bytes;
            });
            for (std::size_t position = prefix; position < prefix + captured_tokens; ++position) {
                run.where = "trace_forward";
                run.offset = position;
                run.rows = 1;
                const auto values = session.step(token_at(position));
                run.snapshot(session);
                ++run.windows;
                require(run.stats.consumed_tokens == position + 1, "trace consumed accounting");
                run.where = "trace_logit_summary";
                const auto logits = inspect(values);
                run.manifest_bytes = std::filesystem::file_size(std::filesystem::path(run.trace) / "tensors.jsonl");
                writer.emit("window", [&](std::ostream& out) {
                    context(out, run);
                    out << ",\"completed_trace_calls\":" << run.windows << ",\"logits\":{\"elements\":" << values.size()
                        << ",\"finite\":" << logits.finite << ",\"minimum\":" << logits.minimum
                        << ",\"maximum\":" << logits.maximum << ",\"argmax\":" << logits.argmax
                        << "},\"trace_manifest_bytes\":" << run.manifest_bytes;
                });
            }
            require(run.stats.consumed_tokens == prefix + captured_tokens && run.windows == captured_tokens,
                    "diagnostic completion accounting");
        } catch (...) {
            // Accessors do not synchronize/allocate. Preserve actual failed-call
            // counters while Session is alive, then unwind before reporting.
            run.snapshot(session);
            throw;
        }
    }
    run.where = "complete";
    run.offset = prefix + captured_tokens;
    run.rows = 0;
    require(run.cleanup_completed && writer.records == 10, "diagnostic record/cleanup accounting");
    writer.emit("complete", [&](std::ostream& out) {
        context(out, run);
        out << ",\"diagnostic_only\":true,\"parity_test\":false,\"speed_test\":false,\"capture_completed\":true"
            << ",\"raii_cleanup_completed\":" << run.cleanup_completed
            << ",\"completed_prefix_calls\":" << run.prefix_calls << ",\"completed_trace_calls\":" << run.windows
            << ",\"records_including_this\":" << writer.records + 1 << ",\"trace_manifest_bytes\":" << run.manifest_bytes;
    });
}
}

int main(int argc, char** argv) {
    if (argc != 4 || (std::string_view(argv[3]) != "n1" && std::string_view(argv[3]) != "chunk1024")) {
        std::cerr << "usage: core-prefill-failure-trace MODEL.gguf NEW_TRACE_DIR MODE\n"
                     "MODE must be exactly n1 or chunk1024\n";
        return 2;
    }
    Writer writer;
    Run run;
    run.trace = argv[2];
    run.mode = argv[3];
    try {
        execute(writer, run, argv[1]);
        return 0;
    } catch (const std::exception& error) {
        // execute's Session scope (including partial construction) has unwound.
        // The first exception is the diagnostic result; never retry/reset here.
        try {
            writer.emit("failure", [&](std::ostream& out) {
                context(out, run);
                out << ",\"error\":"; quote(out, error.what());
                out << ",\"diagnostic_only\":true,\"parity_test\":false,\"speed_test\":false"
                    << ",\"session_constructed\":" << run.session_constructed
                    << ",\"raii_cleanup_completed\":" << run.cleanup_completed
                    << ",\"completed_prefix_calls\":" << run.prefix_calls << ",\"completed_trace_calls\":" << run.windows
                    << ",\"last_completed_trace_manifest_bytes\":" << run.manifest_bytes
                    << ",\"trace_files_removed\":false,\"records_including_this\":" << writer.records + 1;
            });
        } catch (const std::exception& output_error) {
            std::cerr << "failure record: " << output_error.what() << '\n';
        }
        std::cerr << "core-prefill-failure-trace: " << error.what() << '\n';
        return 1;
    }
}
