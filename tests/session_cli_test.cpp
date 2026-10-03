#include "session_cli.hpp"

#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
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
    const auto teacher = parse({"core-session", "--trace", "trace-dir", "model.gguf", "248044", "--capacity", "40",
                                "100", "--logits", "logits.bin", "--slots", "1", "--generate", "32", "--ignore-eos"});
    check(!teacher.request_protocol() && teacher.config.trace_directory == "trace-dir" && teacher.logits_path == "logits.bin",
          "legacy diagnostic protocol with interspersed options");
    check(teacher.tokens == std::vector<std::int32_t>{248044, 100} && teacher.generate == 32 && teacher.ignore_eos &&
          teacher.config.capacity == 40 && teacher.config.expert_slots == 1, "legacy diagnostic config");
    check(parse({"core-session", "--help"}).help, "help without a model");
    check(parse({"core-session", "model.gguf", "0", "--help"}).help, "late help");
}

void opt_in_and_bounds() {
    const auto sampled = parse({"core-session", "--sample", "--seed", "0", "--generate", "512", "--ignore-eos",
                                "--capacity", "4608", "--prefill-chunk", "1024", "model.gguf", "248044"});
    check(sampled.sample && sampled.request_protocol() && sampled.explicit_prefill && sampled.ignore_eos &&
          sampled.config.max_batch_tokens == 1024 && sampled.generate == 512, "stochastic grouped opt-in");
    check(sampled.sampling.seed == 0 && sampled.sampling.temperature == 1.0 && sampled.sampling.top_p == 0.95 &&
          sampled.sampling.top_k == 20, "original primary sampling defaults and explicit zero seed");
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
    check(constructions == 0 && rng == original_rng, "invalid request reached execution or RNG");
}
} // namespace

int main() {
    try {
        defaults_and_compatibility();
        opt_in_and_bounds();
        invalid_arguments();
        preflight_before_execution();
        std::cout << "session-cli: " << checks << " checks, " << rejections << " rejections passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "session-cli-test: " << error.what() << '\n';
        return 1;
    }
}
