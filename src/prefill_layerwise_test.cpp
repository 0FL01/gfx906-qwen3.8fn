#include "session.hpp"
#include "routes.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef CORE_REVISION
#error "core-prefill-layerwise-test requires compiled CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-prefill-layerwise-test requires compiled CORE_DIRTY"
#endif

namespace {
constexpr std::size_t V = 248320, W = 10240, guards = 16;
constexpr std::uint32_t sentinel = 0x4b71abcd;
constexpr std::uint64_t inventory = 68262297600ULL;
static_assert(std::string_view(CORE_REVISION).size() == 40 && (CORE_DIRTY == 0 || CORE_DIRTY == 1));
void require(bool ok, std::string_view message) {
    if (!ok) throw std::runtime_error(std::string(message));
}
void hip(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
template<class F> void invalid(F&& call) {
    bool rejected = false;
    try { call(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "expected atomic invalid_argument");
}
struct Guarded {
    std::size_t extent;
    std::vector<float> storage;
    explicit Guarded(std::size_t count) : extent(count), storage(count + 2 * guards, std::bit_cast<float>(sentinel)) {}
    float* data() { return storage.data() + guards; }
    std::span<const float> values() const { return std::span(storage).subspan(guards, extent); }
    void check() const {
        for (std::size_t i = 0; i < guards; ++i)
            require(std::bit_cast<std::uint32_t>(storage[i]) == sentinel &&
                std::bit_cast<std::uint32_t>(storage[guards + extent + i]) == sentinel, "fixture guard overwritten");
    }
};
void compare(std::span<const float> actual, std::span<const float> reference) {
    require(actual.size() == reference.size(), "honest full-vocabulary shape");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], r = reference[i];
        if (!std::isfinite(a) || !std::isfinite(r) || std::abs(a - r) > .02 + .002 * std::abs(r))
            throw std::runtime_error("frozen .02+.002absref full-V gate index=" + std::to_string(i));
    }
}
std::vector<float> tap_copy(qwen::SessionTargetTap tap) {
    if (!tap.pointer) { require(tap.rows == 0, "empty tap extent"); return {}; }
    require(tap.device == 1 && tap.width == static_cast<int>(W) && tap.rows > 0, "typed tap geometry");
    Guarded result(static_cast<std::size_t>(tap.rows) * W);
    hip(hipSetDevice(1)); hip(hipMemcpy(result.data(), tap.pointer, result.extent * sizeof(float), hipMemcpyDeviceToHost));
    result.check();
    return {result.values().begin(), result.values().end()};
}
template<class T, std::size_t N> auto words(const T& x) { return std::bit_cast<std::array<std::uint64_t, N>>(x); }
struct Snapshot {
    qwen::SessionStats stats;
    qwen::SessionRouteStats routes;
    qwen::SessionAttentionStats attention;
    qwen::SessionHybridStats hybrid;
    qwen::SessionSpeculativeStats speculative;
    qwen::SessionCheckpointState checkpoint;
    qwen::SessionTargetTap tap;
    qwen::SessionMemory memory;
    const float* borrowed;
    std::vector<float> logits, tap_bytes;
    Snapshot(qwen::Session& s, std::span<const float> output)
        : stats(s.stats()), routes(s.route_stats()), attention(s.attention_stats()), hybrid(s.hybrid_stats()), speculative(s.speculative_stats()),
          checkpoint(s.checkpoint_state()), tap(s.target_tap()), memory(s.memory()), borrowed(output.data()),
          logits(output.begin(), output.end()), tap_bytes(tap_copy(tap)) {}
    void check(qwen::Session& s, std::span<const float> output) const {
        require((words<qwen::SessionStats, 5>(s.stats()) == words<qwen::SessionStats, 5>(stats)), "rejection stats");
        require((words<qwen::SessionRouteStats, 2>(s.route_stats()) == words<qwen::SessionRouteStats, 2>(routes)), "rejection routes");
        require((words<qwen::SessionAttentionStats, 6>(s.attention_stats()) == words<qwen::SessionAttentionStats, 6>(attention)), "rejection attention");
        require((words<qwen::SessionHybridStats, 26>(s.hybrid_stats()) == words<qwen::SessionHybridStats, 26>(hybrid)), "rejection hybrid physical counters");
        require(s.speculative_stats() == speculative && s.checkpoint_state() == checkpoint && s.target_tap() == tap, "rejection owner/tap");
        require(output.data() == borrowed && output.size() == logits.size() &&
            std::memcmp(output.data(), logits.data(), logits.size() * sizeof(float)) == 0, "borrowed logits changed");
        const auto bytes = tap_copy(tap);
        require(bytes.size() == tap_bytes.size() && (bytes.empty() ||
            std::memcmp(bytes.data(), tap_bytes.data(), bytes.size() * sizeof(float)) == 0), "borrowed tap bytes changed");
        const auto now = s.memory();
        require(now.ownership_verified && now.host_logit_capacity == memory.host_logit_capacity &&
            now.host_layerwise_activations == memory.host_layerwise_activations, "rejection host allocations");
        for (int id = 0; id < 2; ++id)
            require(now.devices[id].owned_bytes == memory.devices[id].owned_bytes &&
                now.devices[id].owned_buffers == memory.devices[id].owned_buffers, "rejection GPU allocations");
    }
};
void routes_gate() {
    invalid([] { qwen::RouteGroups old(1025); });
    invalid([] { qwen::RouteGroups large(16385, qwen::RouteGroups::Bound::layerwise16384); });
    qwen::RouteGroups groups(16384, qwen::RouteGroups::Bound::layerwise16384);
    std::vector<std::int32_t> ids(16384 * 10); std::vector<float> weights(ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<std::int32_t>(i % 10);
        weights[i] = std::bit_cast<float>(0x3dcccccdU + static_cast<std::uint32_t>(i % 7));
    }
    groups.prepare(ids, weights, 16384);
    require(groups.assignments().size() == ids.size() && groups.group(9).size() == 16384, "global >128 owner");
    const std::vector<qwen::RouteAssignment> previous(groups.assignments().begin(), groups.assignments().end());
    for (int e = 0; e < 10; ++e) for (int t = 0; t < 16384; ++t) {
        const auto& a = groups.group(e)[t];
        require(a.token == t && a.rank == e && a.expert == e &&
            std::bit_cast<std::uint32_t>(a.weight) == std::bit_cast<std::uint32_t>(weights[t * 10 + e]), "stable token/rank/raw bits");
    }
    ids.back() = 512; invalid([&] { groups.prepare(ids, weights, 16384); });
    require(std::memcmp(previous.data(), groups.assignments().data(), previous.size() * sizeof(previous[0])) == 0, "late group rejection mutated owner");
    std::cout << "{\"kind\":\"layerwise_routes_host_gate\",\"rows\":16384,\"group_assignments\":16384,\"passed\":true}\n";
}
void resources(qwen::Session& s, int n) {
    const auto m = s.memory(); const auto N = static_cast<std::uint64_t>(n);
    require(m.ownership_verified && m.layerwise_prefill_capacity == n, "independent buffer ownership ledger");
    require(m.host_layerwise_activations == 2 * N * W * 4 && m.host_layerwise_ffn_input == N * 2560 * 4 &&
        m.host_layerwise_injection == N * 4 * 4 && m.host_layerwise_probabilities == N * 512 * 4 &&
        m.host_layerwise_routes == N * 10 * 8 && m.host_layerwise_groups == N * 10 * 16 &&
        m.pinned_layerwise_metadata == N * 10 * 8 && m.pinned_handoff == 1024ULL * W * 4, "full-N/bounded-frame capacities");
    for (int id = 0; id < 2; ++id) {
        require(m.layerwise_original_q8_bytes[id] == N * 80 * 36 &&
            m.layerwise_contribution_bytes[id] == N * 10 * 2560 * 4 &&
            m.layerwise_metadata_bytes[id] == N * 10 * 8, "global GPU extents");
        require(m.target_tap_bytes[id] == (m.speculative_checkpoints && id == 1 ? std::max<std::uint64_t>(N, 1024) * W * 4 : 0) &&
            m.target_tap_staging_bytes[id] == m.target_tap_bytes[id], "separate full teacher-history tap capacity");
    }
    std::cout << "{\"kind\":\"layerwise_resources\",\"rows\":" << n << ",\"activations_each_bytes\":" << N * W * 4
        << ",\"original_q8_each_bytes\":" << N * 80 * 36 << ",\"contrib_each_bytes\":" << N * 10 * 2560 * 4
        << ",\"dto_each_bytes\":" << N * 10 * 8 << ",\"router_host_bytes\":" << m.host_layerwise_probabilities
        << ",\"host_logits_capacity_bytes\":" << m.host_logit_capacity << ",\"root_tap_each_bytes\":" << m.target_tap_bytes[1]
        << ",\"free_vram0\":" << m.devices[0].free_vram << ",\"free_vram1\":" << m.devices[1].free_vram
        << ",\"ownership_verified\":true}\n";
}
void record(qwen::Session& s, std::string_view phase, int rows, std::size_t compared, double wall_ms,
            qwen::SessionStats before = {}) {
    const auto a = s.stats();
    require(a.expert_hits - before.expert_hits + a.expert_misses - before.expert_misses == static_cast<std::uint64_t>(rows) * 480,
        "logical reuse + first physical acquisitions do not cover actual assignments");
    std::cout << "{\"kind\":\"layerwise_gate\",\"phase\":" << std::quoted(std::string(phase)) << ",\"rows\":" << rows
        << ",\"compared_full_vocab_values\":" << compared << ",\"expert_upload_bytes\":" << a.expert_upload_bytes - before.expert_upload_bytes
        << ",\"hits\":" << a.expert_hits - before.expert_hits << ",\"misses\":" << a.expert_misses - before.expert_misses
        << ",\"max_actual_group\":" << s.route_stats().last_max_expert_group_assignments
        << ",\"groups_gt128\":" << s.route_stats().expert_groups_gt128 << ",\"wall_ms\":" << wall_ms
        << ",\"last_call_completed_ms\":" << a.last_completed_ms << ",\"wall_includes_fixture_comparison\":true"
        << ",\"warm_cache_retained_by_reset\":true,\"passed\":true}\n";
}
void run(const std::string& model, int n) {
    qwen::SessionConfig config; config.capacity = n + 8; config.max_batch_tokens = 1024;
    config.attention_query_tile = 8; config.layerwise_prefill_capacity = n;
    config.speculative_checkpoints = n <= 4096; // 16k all-rows avoids two 16.27GB publication owners.
    qwen::Session s(model, config); s.set_attention_batch(true); resources(s, n);
    const auto gpu_loaded = s.memory();
    std::vector<std::int32_t> ids(n + 8 + 2, -123456789);
    auto tokens = std::span(ids).subspan(1, n + 8); tokens[0] = 248044;
    for (int t = 1; t < n + 8; ++t) tokens[t] = 99 + t;
    Guarded reference(static_cast<std::size_t>(n + 8) * V);
    Guarded old_reference(static_cast<std::size_t>(n + 8) * V);
    // Fixture-only independent retained oracles, NOT runtime allocations. At
    // optional N16384 these two guarded refs add ~32.56GB beyond Session RAM.
    Guarded teacher(config.speculative_checkpoints ? static_cast<std::size_t>(n) * W : 0);
    // Independent N1 hot path, FULL V for EVERY row, including continuation.
    for (int t = 0; t < n + 8; ++t) {
        const auto out = s.step(tokens[t]); require(out.size() == V, "N1 vocabulary");
        for (float x : out) require(std::isfinite(x), "N1 reference nonfinite");
        std::copy(out.begin(), out.end(), reference.data() + static_cast<std::size_t>(t) * V);
        if (config.speculative_checkpoints && t < n) {
            const auto tap = tap_copy(s.target_tap()); require(tap.size() == W, "N1 tap shape");
            std::copy(tap.begin(), tap.end(), teacher.data() + static_cast<std::size_t>(t) * W);
        }
    }
    auto expected = [&](int first, int count) { return reference.values().subspan(static_cast<std::size_t>(first) * V, static_cast<std::size_t>(count) * V); };
    auto compare_all = [&](std::span<const float> out, int first, int count) {
        compare(out, expected(first, count));
        compare(out, old_reference.values().subspan(static_cast<std::size_t>(first) * V, static_cast<std::size_t>(count) * V));
    };
    auto continuation = [&] {
        for (int t = n; t < n + 8; ++t) compare_all(s.step(tokens[t]), t, 1);
    };
    auto check_tap = [&](int first, int count) {
        if (!config.speculative_checkpoints) { require(s.target_tap().rows == 0, "disabled tap"); return; }
        const auto tap = s.target_tap(); require(tap.rows == count && tap.first_position == static_cast<std::uint64_t>(first), "ALL actual ordinary tap rows");
        compare(tap_copy(tap), teacher.values().subspan(static_cast<std::size_t>(first) * W, static_cast<std::size_t>(count) * W));
    };
    s.reset();
    const auto old_begin = std::chrono::steady_clock::now();
    for (int first = 0; first < n; first += 1024) {
        const int count = std::min(1024, n - first);
        const auto out = s.step_batch(tokens.subspan(first, count)); compare(out, expected(first, count));
        std::copy(out.begin(), out.end(), old_reference.data() + static_cast<std::size_t>(first) * V);
    }
    record(s, "old_C1024_all_rows", n, static_cast<std::size_t>(n) * V,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - old_begin).count());
    for (int t = n; t < n + 8; ++t) {
        const auto out = s.step(tokens[t]); compare(out, expected(t, 1));
        std::copy(out.begin(), out.end(), old_reference.data() + static_cast<std::size_t>(t) * V);
    }
    for (int repeat = 0; repeat < 2; ++repeat) {
        s.reset(); const auto begin = std::chrono::steady_clock::now();
        const auto out = s.prefill_layerwise(tokens.first(n), qwen::Session::OutputMode::all_rows);
        compare_all(out, 0, n); check_tap(0, n);
        require(s.stats().expert_upload_bytes <= inventory, "more than one canonical inventory uploaded in one full window");
        if (n >= 4096) require(s.route_stats().last_max_expert_group_assignments > 128 && s.route_stats().expert_groups_gt128 > 0,
            "optional real model window must exercise actual global group >128");
        const auto sp = s.speculative_stats();
        require(!s.checkpoint_state().pending && sp.verify_windows == 0 && sp.gdn_prefix_calls == 0 &&
            sp.ple_prefix_calls == 0 && sp.qsa_tail_prefix_calls == 0 &&
            sp.target_forward_rows == (config.speculative_checkpoints ? static_cast<std::uint64_t>(n) : 0), "ordinary PP is not verifyN3");
        record(s, repeat ? "layerwise_all_rows_repeat" : "layerwise_all_rows", n, out.size(),
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count());
        continuation();
    }
    resources(s, n); const auto stable = s.memory();
    s.reset(); const auto last = s.prefill_layerwise(tokens.first(n), qwen::Session::OutputMode::last_row);
    compare_all(last, n - 1, 1); check_tap(0, n); require(last.size() == V, "last_row is NOT N-row logit evidence");
    record(s, "explicit_last_row_only", n, V, s.stats().last_completed_ms);
    { Snapshot full_capacity(s, last); invalid([&] { (void)s.prefill_layerwise(tokens.first(9)); }); full_capacity.check(s, last); }
    continuation();
    // Call-boundary invariance with occupied chronology and global-group owners
    // refilled across two windows, then real continuation state.
    s.reset(); compare_all(s.prefill_layerwise(tokens.first(17)), 0, 17);
    compare_all(s.prefill_layerwise(tokens.subspan(17, n - 17)), 17, n - 17);
    check_tap(17, n - 17); continuation();
    s.reset(); compare(s.step_batch(tokens.first(5)), expected(0, 5));
    const auto occupied_before = s.stats(); const auto occupied_begin = std::chrono::steady_clock::now();
    compare_all(s.prefill_layerwise(tokens.subspan(5, n - 5)), 5, n - 5); check_tap(5, n - 5);
    record(s, "occupied5_layerwise", n - 5, static_cast<std::size_t>(n - 5) * V,
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - occupied_begin).count(), occupied_before);
    continuation();
    // A previous actual successful view, with enough room to scan the ENTIRE
    // rejected N-row suffix (including its deliberately invalid final ID).
    s.reset(); const auto prior = s.step(tokens[0]); Snapshot snapshot(s, prior);
    auto reject = [&](auto&& call) { invalid(call); snapshot.check(s, prior); };
    std::vector<std::int32_t> bad(tokens.begin(), tokens.begin() + n); bad.back() = 248320;
    reject([&] { (void)s.prefill_layerwise(bad); });
    reject([&] { (void)s.prefill_layerwise({}); });
    reject([&] { (void)s.prefill_layerwise(tokens.first(n + 1)); });
    reject([&] { (void)s.prefill_layerwise(tokens.first(1), static_cast<qwen::Session::OutputMode>(77)); });
    std::vector<std::int32_t> old_too_large(1025, 100);
    reject([&] { (void)s.step_batch(old_too_large); }); // Legacy max remains EXACTLY 1024.
    if (config.speculative_checkpoints) {
        const auto verified = s.verify_window(tokens.subspan(1, 3)); Snapshot pending(s, verified);
        invalid([&] { (void)s.prefill_layerwise(tokens.first(n)); }); pending.check(s, verified);
        invalid([&] { (void)s.prefill_layerwise(bad); }); pending.check(s, verified);
        s.restore_prefix(0); require(s.stats().consumed_tokens == 1, "pending restore retained pre-window cursor");
        require(s.target_tap() == pending.tap && std::memcmp(verified.data(), pending.logits.data(), verified.size_bytes()) == 0, "restore borrowed lifetime");
        const auto physical = s.speculative_stats();
        compare_all(s.prefill_layerwise(tokens.subspan(1, n - 1)), 1, n - 1); check_tap(1, n - 1);
        const auto after = s.speculative_stats();
        require(after.verify_windows == physical.verify_windows && after.verify_rows == physical.verify_rows &&
            after.gdn_prefix_calls == physical.gdn_prefix_calls && after.ple_prefix_calls == physical.ple_prefix_calls &&
            after.qsa_tail_prefix_calls == physical.qsa_tail_prefix_calls &&
            after.target_forward_rows == physical.target_forward_rows + static_cast<std::uint64_t>(n - 1), "ordinary teacher history after rejected verify owner");
    }
    s.reset(); require(s.stats().consumed_tokens == 0 && s.route_stats().expert_groups_gt128 == 0 &&
        s.target_tap().rows == 0 && !s.checkpoint_state().pending && s.speculative_stats() == qwen::SessionSpeculativeStats{}, "complete reset");
    compare_all(s.prefill_layerwise(tokens.first(n)), 0, n);
    const auto final = s.memory(); require(final.host_logit_capacity == stable.host_logit_capacity, "steady logit owners");
    for (int id = 0; id < 2; ++id) require(final.devices[id].owned_bytes == gpu_loaded.devices[id].owned_bytes &&
        final.devices[id].owned_buffers == gpu_loaded.devices[id].owned_buffers, "no per-layer/matmul GPU allocation");
    reference.check(); old_reference.check(); teacher.check(); require(ids.front() == -123456789 && ids.back() == -123456789, "input guards");
    std::cout << "{\"kind\":\"layerwise_fixture_complete\",\"rows\":" << n << ",\"model\":" << std::quoted(model)
        << ",\"revision\":\"" << CORE_REVISION << "\",\"dirty\":" << CORE_DIRTY
        << ",\"frozen_abs_gate\":0.02,\"frozen_rel_gate\":0.002,\"checkpoint_tap_enabled\":" << std::boolalpha << config.speculative_checkpoints
        << ",\"short_tg_speed_qualified\":false,\"r4_target_speed_qualified\":false,\"passed\":true}\n";
}
void disabled_gate(const std::string& model) {
    for (int c : {-1, 16385}) {
        qwen::SessionConfig bad; bad.layerwise_prefill_capacity = c;
        invalid([&] { qwen::Session rejected(model, bad); });
    }
    { qwen::SessionConfig bad; bad.max_batch_tokens = 3; bad.layerwise_prefill_capacity = 32;
      invalid([&] { qwen::Session rejected(model, bad); }); }
    { qwen::SessionConfig bad; bad.max_batch_tokens = 32; bad.layerwise_prefill_capacity = 32; bad.trace_directory = "reject-before-trace-create";
      invalid([&] { qwen::Session rejected(model, bad); }); }
    qwen::SessionConfig config; config.capacity = 4;
    qwen::Session s(model, config); const auto m = s.memory();
    require(m.layerwise_prefill_capacity == 0 && !m.host_layerwise_owner && !m.host_layerwise_activations &&
        !m.host_layerwise_ffn_input && !m.host_layerwise_injection && !m.host_layerwise_probabilities && !m.host_layerwise_routes &&
        !m.host_layerwise_groups && !m.pinned_layerwise_metadata && m.layerwise_original_q8_bytes == std::array<std::uint64_t, 2>{} &&
        m.layerwise_contribution_bytes == std::array<std::uint64_t, 2>{} && m.layerwise_metadata_bytes == std::array<std::uint64_t, 2>{}, "default no R4 payload allocation");
    const std::int32_t token = 100; invalid([&] { (void)s.prefill_layerwise(std::span(&token, 1)); });
    require(s.stats().consumed_tokens == 0 && s.memory().devices[0].owned_bytes == m.devices[0].owned_bytes, "default resource rejection");
}
}
int main(int argc, char** argv) {
    try {
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        routes_gate();
        if (argc == 2 && std::string_view(argv[1]) == "--routes-only") return 0;
        require(argc == 2 || argc == 3, "usage: core-prefill-layerwise-test MODEL [32|128|4096|16384]");
        disabled_gate(argv[1]);
        if (argc == 2) { run(argv[1], 32); run(argv[1], 128); }
        else {
            int n = 0; const std::string_view text(argv[2]); const auto parsed = std::from_chars(text.data(), text.data() + text.size(), n);
            require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() &&
                (n == 32 || n == 128 || n == 4096 || n == 16384), "explicit fixture row bound");
            run(argv[1], n);
        }
        return 0;
    } catch (const std::exception& e) { std::cerr << "layerwise fixture: " << e.what() << '\n'; return 1; }
}
