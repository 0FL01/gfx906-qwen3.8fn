// Diagnostic only: explicit completed PP ranges separate loading and cleanup.
#include "session.hpp"
#include <rocprofiler-sdk-roctx/roctx.h>
#include <algorithm>
#include <chrono>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>
#ifndef CORE_REVISION
#error "compiled source revision required"
#endif
#ifndef CORE_DIRTY
#error "compiled dirty flag required"
#endif
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
struct Range {
    explicit Range(const char* name) { (void)roctxRangePushA(name); }
    ~Range() { (void)roctxRangePop(); }
};
}
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("core-profile-prefill MODEL whitespace-token-ids.txt");
        std::ifstream input(argv[2]);
        if (!input) throw std::runtime_error("cannot open token fixture");
        std::vector<std::int32_t> ids;
        std::string token;
        while (input >> token) {
            std::int32_t id = 0;
            const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
            if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() ||
                id < 0 || id >= 248320 || ids.size() >= 16384)
                throw std::invalid_argument("fixture token/count bounds");
            ids.push_back(static_cast<std::int32_t>(id));
        }
        if (!input.eof() || (ids.size() != 4096 && ids.size() != 16384))
            throw std::invalid_argument("expected exactly 4096 or 16384 integer IDs");
        qwen::SessionConfig cfg;
        cfg.capacity = static_cast<int>(ids.size()) + 512;
        cfg.max_batch_tokens = 1024;
        cfg.expert_slots = 112;
        cfg.attention_query_tile = 8;
        std::cout << std::setprecision(17)
            << "{\"kind\":\"profile_prefill_source\",\"revision\":\"" << CORE_REVISION
            << "\",\"dirty\":" << (CORE_DIRTY ? "true" : "false")
            << ",\"prompt_tokens\":" << ids.size()
            << ",\"capacity\":" << cfg.capacity
            << ",\"chunk\":1024,\"slots\":112,\"attention_tile\":8,\"cpu_workers\":0,"
               "\"mtp\":false,\"prefix_reuse\":false,\"performance_qualified\":false,\"prompt_ids\":[";
        for (std::size_t i = 0; i < ids.size(); ++i) std::cout << (i ? "," : "") << ids[i];
        std::cout << "]}\n";
        auto start = Clock::now();
        qwen::Session session(argv[1], cfg);
        session.set_attention_batch(true);
        const double load_ms = elapsed(start);
        start = Clock::now();
        std::size_t calls = 0;
        {
            Range pp("CORE_PP_COMPLETED");
            for (std::size_t first = 0; first < ids.size(); first += 1024) {
                Range chunk("CORE_PP_CHUNK_COMPLETED");
                const auto logits = session.step_batch(std::span(ids).subspan(first, 1024));
                if (logits.size() != 1024ULL * 248320)
                    throw std::runtime_error("full-logit extent");
                ++calls;
            }
        }
        const double pp_ms = elapsed(start);
        const auto stats = session.stats();
        if (stats.consumed_tokens != ids.size() || !std::isfinite(pp_ms) || pp_ms <= 0)
            throw std::runtime_error("completed prefill contract");
        std::cout << "{\"kind\":\"profile_prefill_complete\",\"prompt_tokens\":" << ids.size()
            << ",\"calls\":" << calls << ",\"load_ms\":" << load_ms << ",\"pp_ms\":" << pp_ms
            << ",\"expert_upload_bytes\":" << stats.expert_upload_bytes
            << ",\"scope\":\"ROCTX completed PP only; load and cleanup outside; trace overhead; not speed qualification\","
               "\"passed\":true}\n";
        std::cout.flush();
        if (!std::cout) throw std::runtime_error("stdout failed");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "profile_prefill: " << e.what() << '\n';
        return 1;
    }
}
