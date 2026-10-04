#include "session_cli.hpp"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::size_t checks = 0, rejections = 0;

void check(bool condition, const char* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

qwen::session_cli::Options parse(std::initializer_list<const char*> arguments) {
    return qwen::session_cli::parse(static_cast<int>(arguments.size()), arguments.begin());
}

template<class Function> void rejected(Function function, std::string_view guard = {}) {
    try {
        function();
    } catch (const std::invalid_argument& error) {
        ++rejections;
        if (!guard.empty())
            check(std::string_view(error.what()).find(guard) != std::string_view::npos, "wrong rejection guard");
        return;
    }
    throw std::runtime_error("invalid CLI arguments accepted");
}

void defaults_and_compatibility() {
    const auto original = parse({"core-session", "model.gguf", "248044", "100", "248319"});
    check(original.model == "model.gguf" && original.tokens == std::vector<std::int32_t>{248044, 100, 248319}, "legacy prompt IDs");
    check(original.config.capacity == 4096 && original.config.expert_slots == 112 && original.config.max_batch_tokens == 1,
          "legacy allocation defaults");
    check(original.generate == 0 && !original.sample && !original.ignore_eos && !original.help && !original.request_protocol(),
          "legacy greedy protocol default");
    check(original.config.trace_directory.empty() && original.logits_path.empty(), "legacy artifacts off");
    check(original.protocol_version() == 0 && !original.explicit_execution && !original.attention_enabled &&
          original.config.cpu_workers == 0 && original.config.attention_query_tile == 1 &&
          original.hybrid_policy.mode == qwen::SessionHybridMode::disabled && original.hybrid_policy.gpu_miss_groups == 2 &&
          !original.config.hybrid_probe && !original.hybrid_policy.diagnostic_fail_after_admission,
          "legacy execution defaults and source selection");
    const auto teacher = parse({"core-session", "--trace", "trace-dir", "model.gguf", "248044", "--capacity", "40",
                                "100", "--logits", "logits.bin", "--slots", "1", "--generate", "32", "--ignore-eos"});
    check(!teacher.request_protocol() && teacher.config.trace_directory == "trace-dir" && teacher.logits_path == "logits.bin",
          "legacy diagnostic protocol with interspersed options");
    check(teacher.tokens == std::vector<std::int32_t>{248044, 100} && teacher.generate == 32 && teacher.ignore_eos &&
          teacher.config.capacity == 40 && teacher.config.expert_slots == 1, "legacy diagnostic config");
    check(parse({"core-session", "--help"}).help, "help without a model");
    check(parse({"core-session", "model.gguf", "0", "--help"}).help, "late help");
    // The existing contract permits last-value-wins except for --seed. Keep it
    // even when the earlier numeric value would fail final range validation.
    const auto duplicates = parse({"core-session", "--capacity", "3", "--capacity", "8", "--slots", "513", "--slots", "2",
                                   "--generate", "-1", "--generate", "1", "--trace", "first", "--trace", "last",
                                   "--logits", "first.bin", "--logits", "last.bin", "--ignore-eos", "--ignore-eos", "m", "0"});
    check(duplicates.protocol_version() == 0 && duplicates.config.capacity == 8 && duplicates.config.expert_slots == 2 &&
          duplicates.generate == 1 && duplicates.config.trace_directory == "last" && duplicates.logits_path == "last.bin" &&
          duplicates.ignore_eos, "legacy accepted duplicate behavior");
    const auto sampled_duplicates = parse({"core-session", "--sample", "--sample", "--seed", "7", "--generate", "1",
                                           "--temperature", "0", "--temperature", "1", "--top-p", "0", "--top-p", ".95",
                                           "--top-k", "-1", "--top-k", "20", "--prefill-chunk", "1025", "--prefill-chunk", "1", "m", "0"});
    check(sampled_duplicates.protocol_version() == 1 && !sampled_duplicates.explicit_execution &&
          sampled_duplicates.sampling.temperature == 1 && sampled_duplicates.sampling.top_p == .95 &&
          sampled_duplicates.sampling.top_k == 20 && sampled_duplicates.config.max_batch_tokens == 1,
          "protocol 1 accepted duplicate behavior and source selection");
}

void opt_in_and_bounds() {
    const auto sampled = parse({"core-session", "--sample", "--seed", "0", "--generate", "512", "--ignore-eos",
                                "--capacity", "4608", "--prefill-chunk", "1024", "model.gguf", "248044"});
    check(sampled.sample && sampled.request_protocol() && sampled.explicit_prefill && sampled.ignore_eos &&
          sampled.config.max_batch_tokens == 1024 && sampled.generate == 512, "stochastic grouped opt-in");
    check(sampled.sampling.seed == 0 && sampled.sampling.temperature == 1.0 && sampled.sampling.top_p == 0.95 &&
           sampled.sampling.top_k == 20, "original primary sampling defaults and explicit zero seed");
    check(sampled.protocol_version() == 1 && !sampled.explicit_execution, "old stochastic source remains protocol 1");
    const auto maximum_seed = parse({"core-session", "model.gguf", "0", "--temperature", "2.5", "--top-p", "1",
                                     "--top-k", "248320", "--seed", "18446744073709551615", "--sample", "--generate", "1"});
    check(maximum_seed.sampling.seed == std::numeric_limits<std::uint64_t>::max() && maximum_seed.sampling.top_k == 248320 &&
          maximum_seed.sampling.temperature == 2.5 && maximum_seed.sampling.top_p == 1.0, "late options and UINT64_MAX");
    const auto no_top_k = parse({"core-session", "--sample", "--seed", "7", "--top-k", "0", "--temperature", "1e-300",
                                "--top-p", ".01", "--generate", "1", "--prefill-chunk", "1", "--trace", "trace", "m", "0"});
    check(no_top_k.sampling.top_k == 0 && no_top_k.sampling.temperature > 0 && no_top_k.config.trace_directory == "trace",
          "sampler-supported disabled top-k, positive small temperature and N1 trace");
    const auto teacher_batch = parse({"core-session", "--prefill-chunk", "32", "--logits", "rows.bin", "m", "0", "1"});
    check(teacher_batch.request_protocol() && !teacher_batch.sample && teacher_batch.generate == 0 &&
           teacher_batch.logits_path == "rows.bin", "grouped teacher/logits opt-in without a seed");
    check(teacher_batch.protocol_version() == 1 && !teacher_batch.explicit_execution, "old grouped source remains protocol 1");
    const auto capacity_edge = parse({"core-session", "--capacity", "4", "--slots", "512", "--generate", "4", "m", "0"});
    check(capacity_edge.generate == 4 && capacity_edge.config.capacity == 4, "one prompt plus three forwards fits capacity four");
    check(parse({"core-session", "--capacity", "4", "--generate", "1", "m", "0", "1", "2", "3"}).tokens.size() == 4,
          "last prompt plus pending first output fits full capacity");
    const auto max_outputs = parse({"core-session", "--capacity", "131072", "--generate", "131072", "m", "0"});
    check(max_outputs.generate == 131072, "maximum generated vector bounded with pending final token");
    rejected([] { parse({"core-session", "--capacity", "4", "--generate", "5", "m", "0"}); }, "exceeds capacity");
    rejected([] { parse({"core-session", "--capacity", "4", "--generate", "2", "m", "0", "1", "2", "3"}); }, "exceeds capacity");
    std::vector<const char*> long_prompt{"core-session", "--capacity", "131072", "m"};
    long_prompt.resize(4 + 131072, "0");
    check(qwen::session_cli::parse(static_cast<int>(long_prompt.size()), long_prompt.data()).tokens.size() == 131072,
          "maximum prompt bound accepted");
    long_prompt.push_back("0");
    rejected([&] { qwen::session_cli::parse(static_cast<int>(long_prompt.size()), long_prompt.data()); }, "too many prompt");
}

void execution_options() {
    using Mode = qwen::SessionHybridMode;
    for (const auto option : {"--cpu-workers", "--hybrid-mode", "--gpu-miss-groups", "--attention-query-tile"}) {
        const auto value = std::string_view(option) == "--cpu-workers" ? "0" :
            std::string_view(option) == "--hybrid-mode" ? "disabled" :
            std::string_view(option) == "--gpu-miss-groups" ? "2" : "1";
        const auto baseline = parse({"core-session", "m", "0", option, value});
        check(baseline.protocol_version() == 2 && baseline.request_protocol() && baseline.explicit_execution &&
              !baseline.sample && baseline.config.cpu_workers == 0 && baseline.hybrid_policy.mode == Mode::disabled &&
              baseline.hybrid_policy.gpu_miss_groups == 2 && baseline.config.attention_query_tile == 1 &&
              !baseline.attention_enabled, "explicit baseline execution value selects distinct protocol 2");
        rejected([&] { parse({"core-session", option, value, "m", "0", option, value}); }, "duplicate");
    }
    for (int workers = 1; workers <= 15; ++workers) {
        const auto text = std::to_string(workers);
        const auto automatic = parse({"core-session", "m", "0", "--cpu-workers", text.c_str()});
        check(automatic.config.cpu_workers == workers && automatic.hybrid_policy.mode == Mode::mixed &&
              automatic.hybrid_policy.gpu_miss_groups == 2 && automatic.protocol_version() == 2 &&
              !automatic.config.hybrid_probe && !automatic.hybrid_policy.diagnostic_fail_after_admission,
              "nonzero workers default to bounded mixed without probes");
    }
    const auto prepared_disabled = parse({"core-session", "--hybrid-mode", "disabled", "--cpu-workers", "15",
                                         "--gpu-miss-groups", "0", "--generate", "4", "--capacity", "4", "m", "0"});
    check(prepared_disabled.hybrid_policy.mode == Mode::disabled && prepared_disabled.config.cpu_workers == 15 &&
          prepared_disabled.hybrid_policy.gpu_miss_groups == 0 && prepared_disabled.generate == 4,
          "explicit disabled overrides implicit mixed and retains pending-token capacity");
    const auto full_capacity = parse({"core-session", "--attention-query-tile", "128", "--capacity", "4", "--generate", "1",
                                     "m", "0", "1", "2", "3"});
    check(full_capacity.attention_enabled && full_capacity.tokens.size() == 4, "protocol 2 pending first output fits full capacity");
    for (const auto name : {"disabled", "mixed", "force-cpu", "force-gpu-misses"}) {
        const auto mode = qwen::session_cli::hybrid_mode(name);
        check(qwen::session_cli::hybrid_mode_name(mode) == name, "canonical hybrid mode round trip");
        for (const auto quota : {"0", "1", "2"}) {
            if (mode == Mode::mixed && std::string_view(quota) == "0") continue;
            const auto configured = parse({"core-session", "--hybrid-mode", name, "m", "0", "--gpu-miss-groups", quota,
                                           "--cpu-workers", "1"});
            check(configured.hybrid_policy.mode == mode &&
                  configured.hybrid_policy.gpu_miss_groups == qwen::session_cli::number<int>(quota, "quota") &&
                  configured.config.cpu_workers == 1, "explicit modes and permitted quotas, late workers");
        }
    }
    for (int tile = 1; tile <= 128; ++tile) {
        const auto text = std::to_string(tile);
        const auto attention = parse({"core-session", "--attention-query-tile", text.c_str(), "m", "0"});
        check(attention.config.attention_query_tile == tile && attention.attention_enabled == (tile > 1) &&
              attention.config.max_batch_tokens == 1 && attention.config.cpu_workers == 0 &&
              attention.hybrid_policy.mode == Mode::disabled && attention.protocol_version() == 2,
              "attention tile is constructor capacity plus explicit enable, independent of chunk size");
    }
    const auto trace = parse({"core-session", "--attention-query-tile", "1", "--prefill-chunk", "1", "--trace", "trace", "m", "0"});
    check(!trace.attention_enabled && trace.protocol_version() == 2 && trace.config.trace_directory == "trace",
          "tile 1 disables attention batching and permits historical trace ordering");
    const auto wide = parse({"core-session", "--prefill-chunk", "1024", "--attention-query-tile", "128", "--cpu-workers", "15",
                             "--hybrid-mode", "mixed", "--gpu-miss-groups", "1", "m", "0"});
    check(wide.config.max_batch_tokens == 1024 && wide.config.attention_query_tile == 128 && wide.attention_enabled &&
          wide.config.cpu_workers == 15 && wide.hybrid_policy.mode == Mode::mixed && wide.hybrid_policy.gpu_miss_groups == 1,
          "all execution knobs together with wide prefill");
    check(parse({"core-session", "--cpu-workers", "0", "--capacity", "3", "--capacity", "4", "m", "0"}).config.capacity == 4,
          "new protocol preserves accepted old-option duplicate semantics");
}

void execution_rejections() {
    for (const auto option : {"--cpu-workers", "--hybrid-mode", "--gpu-miss-groups", "--attention-query-tile"}) {
        rejected([&] { parse({"core-session", "m", "0", option}); }, "missing option value");
        rejected([&] { parse({"core-session", option, "", "m", "0"}); }, "empty option value");
    }
    for (const auto workers : {"-1", "16", "2147483648", "1.0", "+1", " 1", "1x"})
        rejected([&] { parse({"core-session", "--cpu-workers", workers, "m", "0"}); });
    for (const auto quota : {"-1", "3", "2147483648", "1.0", "+1", " 1", "1x"})
        rejected([&] { parse({"core-session", "--gpu-miss-groups", quota, "m", "0"}); });
    for (const auto tile : {"0", "-1", "129", "2147483648", "1.0", "+1", " 1", "1x"})
        rejected([&] { parse({"core-session", "--attention-query-tile", tile, "m", "0"}); });
    // No undocumented enum spellings, flag aliases, probe or fault controls.
    for (const auto mode : {"off", "cpu", "gpu", "force_cpu", "force_gpu_misses", "MIXED", "production", "mixedx"})
        rejected([&] { parse({"core-session", "--cpu-workers", "1", "--hybrid-mode", mode, "m", "0"}); }, "hybrid mode must");
    for (const auto alias : {"--cpu_workers", "--hybrid_mode", "--gpu_miss_groups", "--attention_query_tile",
                             "--hybrid-probe", "--hybrid_probe", "--fault", "--diagnostic-fail-after-admission",
                             "--diagnostic_fail_after_admission", "--attention-batch"})
        rejected([&] { parse({"core-session", alias, "m", "0"}); }, "unknown option");
    for (const auto mode : {"mixed", "force-cpu", "force-gpu-misses"}) {
        rejected([&] { parse({"core-session", "--hybrid-mode", mode, "m", "0"}); }, "requires --cpu-workers");
        rejected([&] { parse({"core-session", "--cpu-workers", "0", "--hybrid-mode", mode, "m", "0"}); }, "requires --cpu-workers");
    }
    rejected([] { parse({"core-session", "--cpu-workers", "1", "--gpu-miss-groups", "0", "m", "0"}); }, "mixed mode requires");
    rejected([] { parse({"core-session", "--gpu-miss-groups", "0", "--hybrid-mode", "mixed", "m", "0", "--cpu-workers", "1"}); },
             "mixed mode requires");
    rejected([] { parse({"core-session", "--trace", "trace", "m", "0", "--attention-query-tile", "2"}); }, "incompatible");
    rejected([] { parse({"core-session", "--attention-query-tile", "128", "--trace", "trace", "m", "0"}); }, "incompatible");
    rejected([] { parse({"core-session", "--attention-query-tile", "1", "--prefill-chunk", "2", "--trace", "trace", "m", "0"}); },
             "preserve trace order");
    rejected([] { parse({"core-session", "--help", "--attention-query-tile", "2", "--trace", "trace"}); }, "incompatible");
    rejected([] { parse({"core-session", "--cpu-workers", "0", "m", "0", "--cpu-workers", "1"}); }, "duplicate --cpu-workers");
    rejected([] { parse({"core-session", "--hybrid-mode", "disabled", "m", "0", "--hybrid-mode", "mixed", "--cpu-workers", "1"}); },
             "duplicate --hybrid-mode");
    rejected([] { parse({"core-session", "--gpu-miss-groups", "2", "m", "0", "--gpu-miss-groups", "1"}); }, "duplicate --gpu-miss-groups");
    rejected([] { parse({"core-session", "--attention-query-tile", "1", "m", "0", "--attention-query-tile", "2"}); },
             "duplicate --attention-query-tile");
    rejected([] { parse({"core-session", "--cpu-workers", "1", "--capacity", "4", "--generate", "5", "m", "0"}); }, "exceeds capacity");
}

void execution_sampling() {
    // Protocol 2 keeps exact integer seeds across signed-INT64 boundaries and
    // the entire UINT64 source range; no conversion through floating point.
    for (const auto text : {"0", "9223372036854775807", "9223372036854775808", "18446744073709551615"}) {
        const auto sampled = parse({"core-session", "--cpu-workers", "1", "--attention-query-tile", "128", "--sample",
                                    "--seed", text, "--generate", "512", "--ignore-eos", "m", "0"});
        check(sampled.sample && sampled.protocol_version() == 2 &&
              sampled.sampling.seed == qwen::session_cli::number<std::uint64_t>(text, "seed") &&
              sampled.sampling.temperature == 1.0 && sampled.sampling.top_p == .95 && sampled.sampling.top_k == 20,
              "new execution modes preserve explicit seed bounds and RNG/filter defaults");
    }
    rejected([] { parse({"core-session", "--cpu-workers", "1", "--sample", "--generate", "1", "m", "0"}); }, "requires --seed");
    rejected([] { parse({"core-session", "--attention-query-tile", "2", "--seed", "1", "m", "0"}); }, "require --sample");
    rejected([] { parse({"core-session", "--gpu-miss-groups", "1", "--sample", "--seed", "1", "m", "0"}); }, "positive --generate");
    rejected([] { parse({"core-session", "--hybrid-mode", "disabled", "--sample", "--seed", "0", "--seed", "0", "--generate", "1", "m", "0"}); },
             "duplicate --seed");
    for (const auto seed : {"-1", "-9223372036854775808", "18446744073709551616", "+1", "1x", "1.0", " 1"})
        rejected([&] { parse({"core-session", "--attention-query-tile", "2", "--sample", "--seed", seed, "--generate", "1", "m", "0"}); });
}

void invalid_arguments() {
    rejected([] { qwen::session_cli::parse(0, nullptr); }, "argument array");
    rejected([] { qwen::session_cli::parse(-1, nullptr); }, "argument array");
    rejected([] { qwen::session_cli::parse(1, nullptr); }, "argument array");
    rejected([] { parse({nullptr}); }, "null argument");
    rejected([] { parse({"core-session", "m", nullptr}); }, "null argument");
    rejected([] { parse({"core-session", "--seed", nullptr}); }, "null argument");
    rejected([] { parse({"core-session", "--help", nullptr}); }, "null argument");
    rejected([] { qwen::session_cli::number<int>({}, "integer"); }, "empty integer");
    rejected([] { parse({"core-session"}); }, "model/token");
    rejected([] { parse({"core-session", "m"}); }, "model/token");
    rejected([] { parse({"core-session", "", "0"}); }, "empty model");
    rejected([] { parse({"core-session", "m", ""}); }, "empty token");
    rejected([] { parse({"core-session", "--unknown", "m", "0"}); }, "unknown option");
    rejected([] { parse({"core-session", "m", "0", "--unknown"}); }, "unknown option");
    for (const auto option : {"--trace", "--logits", "--capacity", "--slots", "--generate", "--prefill-chunk",
                              "--seed", "--temperature", "--top-p", "--top-k"}) {
        rejected([&] { parse({"core-session", "m", "0", option}); }, "missing option value");
        rejected([&] { parse({"core-session", option, "", "m", "0"}); }, "empty option value");
    }
    for (const auto token : {"-1", "248320", "2147483648", "-2147483649", "18446744073709551616", "100x", " 0", "+0"})
        rejected([&] { parse({"core-session", "m", "248044", "100", token}); });
    for (const auto capacity : {"-4", "0", "3", "5", "131076", "2147483648", "4096x"})
        rejected([&] { parse({"core-session", "--capacity", capacity, "m", "0"}); });
    for (const auto slots : {"0", "-1", "513", "2147483648"})
        rejected([&] { parse({"core-session", "--slots", slots, "m", "0"}); });
    for (const auto chunk : {"0", "-1", "1025", "2147483648", "1.5"})
        rejected([&] { parse({"core-session", "--prefill-chunk", chunk, "m", "0"}); });
    for (const auto generate : {"-1", "131073", "2147483648", "4294967296", "1x"})
        rejected([&] { parse({"core-session", "--generate", generate, "m", "0"}); });
    for (const auto seed : {"-1", "+1", "18446744073709551616", "1x", "1.0", " 1"})
        rejected([&] { parse({"core-session", "--sample", "--seed", seed, "--generate", "1", "m", "0"}); });
    for (const auto temperature : {"0", "-0", "-1", "nan", "inf", "-inf", "1e999", "1e-999", "1x"})
        rejected([&] { parse({"core-session", "--sample", "--seed", "1", "--temperature", temperature, "--generate", "1", "m", "0"}); });
    for (const auto top_p : {"0", "-0", "-.1", "1.0001", "nan", "inf", "1e999", "1x"})
        rejected([&] { parse({"core-session", "--sample", "--seed", "1", "--top-p", top_p, "--generate", "1", "m", "0"}); });
    for (const auto top_k : {"-1", "248321", "2147483648", "1.0"})
        rejected([&] { parse({"core-session", "--sample", "--seed", "1", "--top-k", top_k, "--generate", "1", "m", "0"}); });
    rejected([] { parse({"core-session", "--sample", "--generate", "1", "m", "0"}); }, "requires --seed");
    rejected([] { parse({"core-session", "--sample", "--seed", "1", "m", "0"}); }, "positive --generate");
    rejected([] { parse({"core-session", "--seed", "1", "m", "0"}); }, "require --sample");
    for (const auto filter : {"--temperature", "--top-p", "--top-k"})
        rejected([&] { parse({"core-session", filter, "1", "m", "0"}); }, "require --sample");
    rejected([] { parse({"core-session", "--sample", "--seed", "1", "--seed", "2", "--generate", "1", "m", "0"}); }, "duplicate --seed");
    rejected([] { parse({"core-session", "--trace", "trace", "--prefill-chunk", "2", "m", "0"}); }, "preserve trace order");
    rejected([] { parse({"core-session", "--sample", "--seed", "1", "--generate", "1", "--prefill-chunk", "1024",
                         "--trace", "trace", "m", "0"}); }, "preserve trace order");
}

void preflight_before_execution() {
    // A parser-only executable has no Session/Sampler linkage. Exercise the
    // caller boundary with stand-ins to check that late invalid arguments cannot
    // reach construction or randomness; this is not a GPU runtime qualification.
    std::uint64_t constructions = 0;
    std::mt19937_64 rng(42);
    const auto original_rng = rng;
    auto execute = [&](std::initializer_list<const char*> args) {
        const auto options = parse(args);
        (void)options;
        ++constructions;
        (void)rng();
    };
    rejected([&] { execute({"core-session", "--sample", "--seed", "42", "--generate", "1", "m", "0", "248320"}); });
    rejected([&] { execute({"core-session", "--sample", "--seed", "42", "--generate", "1", "m", "0", "--top-p", "nan"}); });
    rejected([&] { execute({"core-session", "--sample", "--seed", "42", "--generate", "1", "m", "0", "--capacity", "5"}); });
    rejected([&] { execute({"core-session", "--logits", "must-not-create.bin", "m", "0", "--attention-query-tile", "2", "--trace", "must-not-create"}); },
             "incompatible");
    rejected([&] { execute({"core-session", "--logits", "must-not-create.bin", "m", "0", "--hybrid-mode", "mixed"}); }, "requires --cpu-workers");
    rejected([&] { execute({"core-session", "--cpu-workers", "1", "--sample", "--seed", "42", "--generate", "1", "m", "0", "--gpu-miss-groups", "0"}); },
             "mixed mode requires");
    rejected([&] { execute({"core-session", "--attention-query-tile", "2", "m", "0", "--attention-query-tile", "2"}); }, "duplicate");
    rejected([&] { execute({"core-session", "--attention-query-tile", "2", "--sample", "--generate", "1", "m", "0", "--seed", "18446744073709551616"}); });
    check(constructions == 0 && rng == original_rng, "invalid request reached execution or RNG");
}
} // namespace

int main() {
    try {
        defaults_and_compatibility();
        opt_in_and_bounds();
        invalid_arguments();
        execution_options();
        execution_rejections();
        execution_sampling();
        preflight_before_execution();
        std::cout << "session-cli: " << checks << " checks, " << rejections << " rejections passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "session-cli-test: " << error.what() << '\n';
        return 1;
    }
}
