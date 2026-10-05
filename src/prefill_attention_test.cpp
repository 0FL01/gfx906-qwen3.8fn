#include "session.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
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
#include <system_error>
#include <vector>

#ifndef CORE_REVISION
#error "core-prefill-attention-test requires the compiled 40-hex CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-prefill-attention-test requires compiled CORE_DIRTY (0 or 1)"
#endif

namespace {
constexpr std::size_t vocabulary = 248320, teacher_rows = 2056, continuation_rows = 32;
constexpr int capacity = 2088, max_tokens = 1024, query_tile = 8;
constexpr double absolute_gate = .02, relative_gate = .002;
constexpr std::uint64_t routes_per_row = 480, q40_bytes = 2764800, q41_bytes = 2867200;
constexpr std::uint64_t payload_bytes = (6 * q41_bytes + 42 * q40_bytes) * 512;
constexpr std::uint64_t reference_bytes = capacity * vocabulary * sizeof(float);
constexpr std::uint64_t preservation_bytes = max_tokens * vocabulary * sizeof(float);
constexpr std::size_t success_records = 31, guards = 16;
constexpr std::uint32_t guard_bits = 0x4b71abcd;
constexpr std::int32_t input_guard = -123456789;
constexpr std::array<std::string_view, 3> phases{"reference_old_n1", "enabled1024", "occupied5_enabled997"};
constexpr std::array<std::int32_t, capacity> source_ids = [] {
    std::array<std::int32_t, capacity> ids{};
    ids[0] = 248044;
    for (std::size_t p = 1; p < ids.size(); ++p) ids[p] = 99 + static_cast<std::int32_t>(p);
    return ids;
}();
static_assert(sizeof(float) == 4 && sizeof(double) == 8 && std::numeric_limits<float>::is_iec559);
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
static_assert(std::string_view(CORE_REVISION).size() == 40);
static_assert(teacher_rows + continuation_rows == capacity && 5 + 997 + 997 + 57 == teacher_rows);
static_assert(source_ids[0] == 248044 && source_ids[2055] == 2154 && source_ids[2056] == 2155 && source_ids.back() == 2186);
static_assert(payload_bytes == 68262297600ULL && reference_bytes == 2073968640ULL && preservation_bytes == 1017118720ULL);
// source + loaded + three(reset, windows, phase) + 8 rejections + final_reset + footer.
static_assert(success_records == 2 + 3 * 2 + (2 + 5 + 6) + 8 + 2);

void require(bool ok, std::string_view message) {
    if (!ok) throw std::runtime_error(std::string(message));
}
void format(std::ostringstream& out) {
    out.imbue(std::locale::classic());
    out << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10);
}
void text(std::ostream& out, std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    out.put('"');
    for (char c : value) {
        const auto b = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') { out.put('\\'); out.put(c); }
        else if (b < 32) { out << "\\u00"; out.put(hex[b >> 4]); out.put(hex[b & 15]); }
        else out.put(c);
    }
    out.put('"');
}
void number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value; else out << "null";
}
template<class T, std::size_t N> void array(std::ostream& out, const std::array<T, N>& values) {
    out << '[';
    for (std::size_t i = 0; i < N; ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}

// Separate JSONL protocol 2 (bounded head logits): source; memory(loaded/reset/final_reset); window;
// rejection; phase; complete/failure. Exactly 31 successful, bounded raw records.
// Window rows are chronological *logical accepted rows*, NOT GPU selected IDs or
// measured per-query visibility. N1 groups summarize immediately inspected calls.
// Batch counters count successfully completed API invocations, NOT GPU kernels.
// All lines are buffered until Session AND the two large float owners unwind.
// Failure stops execution without a complete footer; bit/argmax are diagnostics.
struct Records {
    std::array<std::string, success_records + 1> lines{};
    std::size_t count = 0;
    void add(std::ostringstream& out) {
        require(count < lines.size() && static_cast<bool>(out), "record bound/format");
        lines[count++] = out.str() + '\n';
    }
    void begin(std::ostringstream& out, std::string_view kind, int phase = -1) const {
        format(out);
        out << "{\"protocol\":2,\"kind\":"; text(out, kind);
        out << ",\"record_index\":" << count << ",\"phase_index\":";
        if (phase < 0) out << "null,\"phase\":null";
        else { out << phase << ",\"phase\":"; text(out, phases.at(static_cast<std::size_t>(phase))); }
    }
    void flush() const {
        for (std::size_t i = 0; i < count; ++i) {
            std::cout << lines[i]; std::cout.flush();
            require(static_cast<bool>(std::cout), "JSONL output failure");
        }
    }
};

void stats_json(std::ostream& out, const qwen::SessionStats& s) {
    out << "{\"consumed_tokens\":" << s.consumed_tokens << ",\"expert_hits\":" << s.expert_hits
        << ",\"expert_misses\":" << s.expert_misses << ",\"expert_upload_bytes\":" << s.expert_upload_bytes
        << ",\"last_completed_ms\":"; number(out, s.last_completed_ms);
    out << ",\"last_completed_ms_bits\":" << std::bit_cast<std::uint64_t>(s.last_completed_ms) << '}';
}
void routes_json(std::ostream& out, const qwen::SessionRouteStats& s) {
    out << "{\"last_max_expert_group_assignments\":" << s.last_max_expert_group_assignments
        << ",\"expert_groups_gt128\":" << s.expert_groups_gt128 << '}';
}
void attention_json(std::ostream& out, const qwen::SessionAttentionStats& s) {
    out << "{\"batch_calls\":" << s.batch_calls << ",\"query_rows\":" << s.query_rows
        << ",\"multiquery_calls\":" << s.multiquery_calls << ",\"multiquery_rows\":" << s.multiquery_rows
        << ",\"singleton_tail_calls\":" << s.singleton_tail_calls << ",\"max_query_rows\":" << s.max_query_rows << '}';
}
struct State {
    qwen::SessionStats stats;
    qwen::SessionRouteStats routes;
    qwen::SessionAttentionStats attention;
    bool enabled;
};
State state(qwen::Session& session) {
    const auto policy = session.hybrid_policy();
    require(policy.mode == qwen::SessionHybridMode::disabled && policy.gpu_miss_groups == 2 && !policy.diagnostic_fail_after_admission,
        "fixture dispatched hybrid policy");
    const auto h = session.hybrid_stats();
    for (const auto n : {h.short_layers, h.gpu_only_wide_layers, h.ready_hit_assignments, h.physical_miss_assignments,
        h.group_reuse_assignments, h.cpu_groups, h.cpu_assignments, h.gpu_hit_groups, h.gpu_hit_assignments,
        h.gpu_miss_groups, h.gpu_miss_assignments, h.admitted_groups, h.evicted_ready_slots, h.input_extractions,
        h.input_bytes, h.cpu_return_bytes, h.cpu_input_bytes_checked, h.all_hit_layers, h.forced_cpu_layers,
        h.forced_gpu_layers, h.cpu_gate_up_jobs, h.cpu_down_jobs, h.gpu_middle_columns, h.gpu_middle_batches,
        h.paired_gate_up_bytes, h.middle_q8_bytes}) require(n == 0, "disabled hybrid statistics changed");
    const auto probe = session.hybrid_probe_diagnostics();
    require(probe.accepted_cpu_jobs == 0 && probe.queued_admission_copies == 0 && probe.pending_slots_at_failure == 0 &&
        probe.cpu_return_bytes_before_failure == 0 && probe.ready_cache_ids == 0 && probe.pending_cache_ids == 0 &&
        probe.failure_layer == -1 && !probe.synthetic_failure && !probe.cpu_pool_drained && !probe.gpu_streams_drained,
        "probe-off diagnostics changed");
    const auto intermediates = session.hybrid_intermediates();
    const auto inputs = session.hybrid_inputs();
    require(intermediates.routed_down.empty() && intermediates.ffn_output.empty() && inputs.original_q8.empty() &&
        inputs.expert_ids.empty() && inputs.route_weights.empty() && inputs.input_available.empty() && inputs.cpu_assignment.empty(),
        "probe-off published frames");
    return {session.stats(), session.route_stats(), session.attention_stats(), session.attention_batch()};
}
void state_json(std::ostream& out, const State& s) {
    out << "{\"stats\":"; stats_json(out, s.stats);
    out << ",\"route_stats\":"; routes_json(out, s.routes);
    out << ",\"attention_stats\":"; attention_json(out, s.attention);
    out << ",\"attention_batch\":" << s.enabled << ",\"hybrid_disabled_stats_all_zero\":true,\"probe_off_publication_empty\":true}";
}
std::string state_key(State s, bool ignore_mode = false) {
    if (ignore_mode) s.enabled = false;
    std::ostringstream out; format(out); state_json(out, s); return out.str();
}
void attention_equal(const qwen::SessionAttentionStats& a, const qwen::SessionAttentionStats& b) {
    std::ostringstream x, y; attention_json(x, a); attention_json(y, b);
    require(x.str() == y.str(), "completed attention counters mismatch");
}

void memory_json(std::ostream& out, const qwen::SessionMemory& m, bool stable_only = false) {
    out << "{\"capacity\":" << m.capacity << ",\"expert_slots\":" << m.expert_slots
        << ",\"attention_query_tile\":" << m.attention_query_tile << ",\"ownership_verified\":" << m.ownership_verified;
#define HOST_FIELD(name) out << ",\"" #name "\":" << m.name
    HOST_FIELD(ram_expert_capacity); HOST_FIELD(ram_expert_payload); HOST_FIELD(host_embedding_capacity);
    HOST_FIELD(host_logit_capacity); HOST_FIELD(pinned_handoff); HOST_FIELD(expert_payload_reads);
    HOST_FIELD(expert_payload_bytes_read); HOST_FIELD(pinned_expert_staging); HOST_FIELD(pinned_hybrid_input);
    HOST_FIELD(pinned_hybrid_output); HOST_FIELD(pinned_hybrid_error); HOST_FIELD(host_hybrid_plans);
    HOST_FIELD(host_cpu_expert_views); HOST_FIELD(cpu_pool_metadata); HOST_FIELD(cpu_pool_scratch);
    HOST_FIELD(cpu_workers); HOST_FIELD(host_hybrid_probe); HOST_FIELD(host_routing_capacity);
    HOST_FIELD(host_route_group_payload); HOST_FIELD(pinned_route_metadata); HOST_FIELD(host_hybrid_input_probe);
    HOST_FIELD(pinned_hybrid_middle);
#undef HOST_FIELD
#define HOST_ARRAY(name) out << ",\"" #name "\":"; array(out, m.name)
    HOST_ARRAY(hybrid_contribution_bytes); HOST_ARRAY(expert_stage_capacity_bytes); HOST_ARRAY(route_metadata_bytes);
    HOST_ARRAY(hybrid_gate_up_bytes); HOST_ARRAY(hybrid_middle_float_bytes); HOST_ARRAY(hybrid_middle_q8_bytes);
    HOST_ARRAY(hybrid_middle_error_bytes);
#undef HOST_ARRAY
    out << ",\"devices\":[";
    for (std::size_t i = 0; i < m.devices.size(); ++i) {
        if (i) out << ',';
        const auto& d = m.devices[i];
        out << "{\"device\":" << d.device << ",\"first_layer\":" << d.first_layer << ",\"last_layer\":" << d.last_layer
            << ",\"gdn_layers\":" << d.gdn_layers << ",\"qsa_layers\":" << d.qsa_layers;
#define DEVICE_FIELD(name) out << ",\"" #name "\":" << d.name
        DEVICE_FIELD(weights); DEVICE_FIELD(expert_slots); DEVICE_FIELD(qsa_kv); DEVICE_FIELD(qsa_index);
        DEVICE_FIELD(gdn_state); DEVICE_FIELD(ple_state); DEVICE_FIELD(workspace); DEVICE_FIELD(head_logits_bytes); DEVICE_FIELD(owned_bytes);
        DEVICE_FIELD(owned_peak_bytes); DEVICE_FIELD(owned_buffers); DEVICE_FIELD(total_vram);
        if (!stable_only) { DEVICE_FIELD(free_vram); }
#undef DEVICE_FIELD
        const auto& a = m.attention[i];
        out << ",\"attention\":{\"gathered_key_bytes\":" << a.gathered_key_bytes;
#define ATTENTION_FIELD(name) out << ",\"" #name "\":" << a.name
        ATTENTION_FIELD(gathered_value_bytes); ATTENTION_FIELD(partial_output_bytes); ATTENTION_FIELD(partial_max_sum_bytes);
        ATTENTION_FIELD(staged_buffer_bytes); ATTENTION_FIELD(output_buffer_bytes); ATTENTION_FIELD(selected_id_bytes);
        ATTENTION_FIELD(selected_block_bytes); ATTENTION_FIELD(selected_count_bytes); ATTENTION_FIELD(private_workspace_bytes);
#undef ATTENTION_FIELD
        out << "}}";
    }
    out << "]}";
}
std::string memory_key(const qwen::SessionMemory& m) {
    std::ostringstream out; format(out); memory_json(out, m, true); return out.str();
}
void check_memory(const qwen::SessionMemory& m) {
    require(m.capacity == capacity && m.expert_slots == 1 && m.attention_query_tile == query_tile && m.ownership_verified,
        "memory config/independent ownership ledger");
    require(m.ram_expert_payload == payload_bytes && m.ram_expert_capacity >= payload_bytes &&
        m.expert_payload_reads == 144 && m.expert_payload_bytes_read == payload_bytes, "RAM payload reread/capacity");
    require(m.host_embedding_capacity > 0 && m.host_logit_capacity >= preservation_bytes && m.pinned_handoff == 41943040 &&
        m.pinned_expert_staging == 183500800, "host capacity/handoff/staging");
    require(m.cpu_workers == 0 && m.pinned_hybrid_input == 0 && m.pinned_hybrid_output == 0 && m.pinned_hybrid_error == 0 &&
        m.host_hybrid_plans == 0 && m.host_cpu_expert_views == 0 && m.cpu_pool_metadata == 0 && m.cpu_pool_scratch == 0 &&
        m.host_hybrid_probe == 0 && m.host_hybrid_input_probe == 0 && m.pinned_hybrid_middle == 0, "CPU/probe-off resources");
    constexpr std::array<std::uint64_t, 2> slots{6 * q41_bytes + 18 * q40_bytes, 24 * q40_bytes};
    for (std::size_t i = 0; i < m.devices.size(); ++i) {
        const auto& d = m.devices[i]; const auto& a = m.attention[i];
        require(d.device == static_cast<int>(i) && d.first_layer == static_cast<int>(i) * 24 &&
            d.last_layer == static_cast<int>(i) * 24 + 23 && d.gdn_layers == 18 && d.qsa_layers == 6, "24/24 owner geometry");
        require(d.expert_slots == slots[i] && d.qsa_kv == 6ULL * capacity * 32 * 18 &&
            d.qsa_index == 6ULL * (capacity / 4 * 128 + 384) * sizeof(float) &&
            d.gdn_state == 18ULL * (786432 + 30720) * sizeof(float) &&
            d.ple_state == (i == 0 ? 92160ULL * sizeof(float) : 0), "unchanged cache/index/GDN/PLE geometry");
        std::uint64_t sum = 0;
        require(d.head_logits_bytes == 128ULL * 248320 * sizeof(float), "actual bounded head logits");
        require(m.selection_score_buffer_bytes[i] == max_tokens * 12288ULL * sizeof(float) &&
            m.selection_histogram_bytes[i] == 8ULL * 8192 * sizeof(int) &&
            m.selection_state_bytes[i] == 8ULL * 16 * sizeof(std::uint64_t) &&
            m.selection_candidate_bytes[i] == 8ULL * 512 * sizeof(std::uint64_t),
            "actual B8 selector backing capacities");

        for (auto n : {d.weights, d.expert_slots, d.qsa_kv, d.qsa_index, d.gdn_state, d.ple_state, d.workspace}) {
            require(n <= std::numeric_limits<std::uint64_t>::max() - sum, "owned byte sum overflow"); sum += n;
        }
        require(d.weights > 0 && d.owned_bytes == sum && d.owned_buffers > 0 && d.owned_peak_bytes >= sum &&
            d.free_vram > 0 && d.free_vram <= d.total_vram && sum <= d.total_vram - d.free_vram, "GPU ownership/VRAM");
        require(a.gathered_key_bytes == 8ULL * 1050112 * sizeof(std::uint16_t) && a.gathered_value_bytes == a.gathered_key_bytes &&
            a.partial_output_bytes == 8ULL * 202752 * sizeof(float) && a.partial_max_sum_bytes == 8ULL * 792 * 8 &&
            a.staged_buffer_bytes == max_tokens * 12288ULL * sizeof(float) && a.output_buffer_bytes == a.staged_buffer_bytes &&
            a.selected_id_bytes == 8ULL * 2051 * sizeof(std::int32_t) && a.selected_block_bytes == 8ULL * 512 * sizeof(std::int32_t) &&
            a.selected_count_bytes == 8ULL * 2 * sizeof(int) && a.private_workspace_bytes == 40338944,
            "actual B8 per-buffer capacities/private workspace");
        require(a.private_workspace_bytes == a.gathered_key_bytes + a.gathered_value_bytes + a.partial_output_bytes +
            a.partial_max_sum_bytes + 8 * 6144ULL * sizeof(float) &&
            a.selected_id_bytes + a.selected_block_bytes + a.selected_count_bytes == 8 * 10260ULL, "private workspace/selection sums");
        require(m.route_metadata_bytes[i] == max_tokens * 10ULL * 8 && m.expert_stage_capacity_bytes[i] == 16 * q41_bytes &&
            m.hybrid_contribution_bytes[i] == 0 && m.hybrid_gate_up_bytes[i] == 0 && m.hybrid_middle_float_bytes[i] == 0 &&
            m.hybrid_middle_q8_bytes[i] == 0 && m.hybrid_middle_error_bytes[i] == 0, "fixed route/stage/CPU-off GPU capacities");
    }
    require(m.pinned_route_metadata == 2 * max_tokens * 10ULL * 8 && m.host_routing_capacity > 0 &&
        m.host_route_group_payload > 0, "indexed route metadata owner");
}

struct Guarded {
    std::size_t extent;
    std::vector<float> storage;
    std::size_t allocation;
    explicit Guarded(std::size_t n) : extent(n), storage(n + 2 * guards), allocation(storage.capacity()) {
        std::fill_n(storage.begin(), guards, std::bit_cast<float>(guard_bits));
        std::fill_n(storage.end() - guards, guards, std::bit_cast<float>(guard_bits));
    }
    std::span<float> values() { return std::span<float>(storage).subspan(guards, extent); }
    void check() const {
        require(storage.size() == extent + 2 * guards && storage.capacity() == allocation, "fixture float capacity changed");
        for (std::size_t i = 0; i < guards; ++i)
            require(std::bit_cast<std::uint32_t>(storage[i]) == guard_bits &&
                std::bit_cast<std::uint32_t>(storage[extent + guards + i]) == guard_bits, "fixture float guard");
    }
};
struct Reference {
    // Two upfront large CPU allocations, no second model or logits reallocation.
    Guarded logits{capacity * vocabulary}, saved{max_tokens * vocabulary};
    std::array<std::int32_t, max_tokens + 1 + 2 * guards> invalid{};
    Reference() {
        invalid.fill(100);
        std::fill_n(invalid.begin(), guards, input_guard); std::fill_n(invalid.end() - guards, guards, input_guard);
        for (std::size_t p = 0; p < source_ids.size(); ++p)
            require(source_ids[p] == (p ? 99 + static_cast<std::int32_t>(p) : 248044) && source_ids[p] >= 0 &&
                source_ids[p] < static_cast<std::int32_t>(vocabulary), "source token IDs");
        check();
    }
    void check() const {
        logits.check(); saved.check();
        for (std::size_t i = 0; i < guards; ++i)
            require(invalid[i] == input_guard && invalid[invalid.size() - guards + i] == input_guard, "fixture input guard");
    }
    std::span<float> row_window(std::size_t p, std::size_t n) {
        require(n && p < capacity && n <= capacity - p, "reference window range");
        return logits.values().subspan(p * vocabulary, n * vocabulary);
    }
    std::span<const float> snapshot(std::span<const float> active) {
        require(active.size() <= saved.extent, "active span exceeds preservation allocation");
        std::copy(active.begin(), active.end(), saved.values().begin()); return saved.values().first(active.size());
    }
    std::span<const std::int32_t> invalid_window(std::size_t n, std::int32_t last) {
        require(n > 0 && n <= max_tokens + 1, "invalid window size");
        auto ids = std::span<std::int32_t>(invalid).subspan(guards, n);
        std::fill(ids.begin(), ids.end(), 100); ids.back() = last; return ids;
    }
};
void exact(std::span<const float> active, std::span<const float> saved) {
    require(active.size() == saved.size(), "preservation extent");
    for (std::size_t i = 0; i < active.size(); ++i)
        if (std::bit_cast<std::uint32_t>(active[i]) != std::bit_cast<std::uint32_t>(saved[i]))
            throw std::runtime_error("active full-span bits changed at " + std::to_string(i));
}
struct Audit {
    qwen::SessionMemory loaded{}, last{};
    std::string stable;
    std::array<std::uint64_t, 2> minimum_free{};
    std::uint64_t snapshots = 0, preserved_values = 0;
    void observe(qwen::Session& session, Reference& ref, std::span<const float> active) {
        ref.check(); const auto saved = ref.snapshot(active); const auto before = state(session);
        last = session.memory(); check_memory(last);
        require(state_key(state(session)) == state_key(before), "memory mutated published state");
        exact(active, saved); ref.check();
        const auto key = memory_key(last);
        if (!snapshots) { loaded = last; stable = key; minimum_free = {last.devices[0].free_vram, last.devices[1].free_vram}; }
        else require(key == stable, "owned GPU/known host capacity or payload reads changed");
        for (std::size_t i = 0; i < 2; ++i) minimum_free[i] = std::min(minimum_free[i], last.devices[i].free_vram);
        ++snapshots; preserved_values += active.size();
    }
};

struct Errors {
    std::uint64_t values = 0, finite = 0, compared = 0, finite_pairs = 0, nonfinite_actual = 0, nonfinite_reference = 0;
    std::uint64_t violations = 0, bit_mismatches = 0, argmax_rows = 0, argmax_agree = 0;
    std::size_t maxabs_row = 0, maxabs_column = 0, maxratio_row = 0, maxratio_column = 0;
    std::size_t first_bad_row = 0, first_bad_column = 0;
    double maxabs = 0, maxratio = 0, squares = 0;
    void inspect(std::span<const float> actual, std::span<const float> expected, std::size_t offset) {
        require(!actual.empty() && actual.size() % vocabulary == 0 && (expected.empty() || expected.size() == actual.size()),
            "full-vocabulary inspection extent");
        values += actual.size(); compared += expected.size();
        for (std::size_t row = 0; row < actual.size() / vocabulary; ++row) {
            std::size_t ai = 0, bi = 0; bool finite_row = true;
            for (std::size_t col = 0; col < vocabulary; ++col) {
                const auto i = row * vocabulary + col;
                const double a = actual[i];
                if (std::isfinite(a)) ++finite; else { ++nonfinite_actual; finite_row = false; }
                if (expected.empty()) continue;
                const double b = expected[i];
                if (!std::isfinite(b)) { ++nonfinite_reference; finite_row = false; }
                bit_mismatches += std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]);
                bool bad = !std::isfinite(a) || !std::isfinite(b);
                if (!bad) {
                    ++finite_pairs; const double error = std::abs(a - b), bound = absolute_gate + relative_gate * std::abs(b);
                    squares += error * error;
                    if (finite_pairs == 1 || error > maxabs) { maxabs = error; maxabs_row = offset + row; maxabs_column = col; }
                    if (finite_pairs == 1 || error / bound > maxratio) {
                        maxratio = error / bound; maxratio_row = offset + row; maxratio_column = col;
                    }
                    bad = error > bound;
                }
                if (bad && violations++ == 0) { first_bad_row = offset + row; first_bad_column = col; }
                if (actual[i] > actual[row * vocabulary + ai]) ai = col;
                if (expected[i] > expected[row * vocabulary + bi]) bi = col;
            }
            if (!expected.empty() && finite_row) { ++argmax_rows; argmax_agree += ai == bi; }
        }
    }
    bool passed() const { return nonfinite_actual == 0 && nonfinite_reference == 0 && violations == 0; }
};
void errors_json(std::ostream& out, const Errors& e) {
    out << "{\"values\":" << e.values << ",\"finite\":" << e.finite << ",\"compared\":" << e.compared
        << ",\"finite_pairs\":" << e.finite_pairs << ",\"nonfinite_actual\":" << e.nonfinite_actual
        << ",\"nonfinite_reference\":" << e.nonfinite_reference << ",\"violations\":" << e.violations
        << ",\"bit_mismatches\":" << e.bit_mismatches << ",\"argmax_rows\":" << e.argmax_rows
        << ",\"argmax_agree\":" << e.argmax_agree << ",\"maxabs\":"; number(out, e.maxabs);
    out << ",\"rms\":"; number(out, e.finite_pairs ? std::sqrt(e.squares / static_cast<double>(e.finite_pairs)) : 0);
    out << ",\"maxboundratio\":"; number(out, e.maxratio);
    out << ",\"maxabs_row\":" << e.maxabs_row << ",\"maxabs_column\":" << e.maxabs_column
        << ",\"maxratio_row\":" << e.maxratio_row << ",\"maxratio_column\":" << e.maxratio_column
        << ",\"first_violation_row\":";
    if (e.violations) out << e.first_bad_row; else out << "null";
    out << ",\"first_violation_column\":";
    if (e.violations) out << e.first_bad_column; else out << "null";
    out << '}';
}

qwen::SessionAttentionStats expected_attention(qwen::SessionAttentionStats s, bool enabled, std::size_t n) {
    // Expected contract used ONLY to check independently reported completed-call
    // counters. No logical-N -> physical-kernel-count conversion.
    if (!enabled || n == 1) return s;
    const auto tiles = (n + query_tile - 1) / query_tile;
    const auto singles = n % query_tile == 1 ? 1ULL : 0ULL;
    s.batch_calls += 12 * tiles; s.query_rows += 12 * n;
    s.singleton_tail_calls += 12 * singles; s.multiquery_calls += 12 * (tiles - singles);
    s.multiquery_rows += 12 * (n - singles); s.max_query_rows = std::max(s.max_query_rows, std::min<std::uint64_t>(n, query_tile));
    return s;
}
void check_completed(const State& a, const State& b, std::size_t offset, std::size_t n) {
    require(b.stats.consumed_tokens == offset && a.stats.consumed_tokens == offset + n && a.enabled == b.enabled,
        "completed position/mode");
    require(std::isfinite(a.stats.last_completed_ms) && a.stats.last_completed_ms > 0, "completed timing");
    require(a.stats.expert_hits >= b.stats.expert_hits && a.stats.expert_misses >= b.stats.expert_misses &&
        a.stats.expert_upload_bytes >= b.stats.expert_upload_bytes, "nonmonotone routes");
    const auto hits = a.stats.expert_hits - b.stats.expert_hits, misses = a.stats.expert_misses - b.stats.expert_misses;
    const auto bytes = a.stats.expert_upload_bytes - b.stats.expert_upload_bytes;
    require(hits <= routes_per_row * n && misses <= routes_per_row * n && hits + misses == routes_per_row * n &&
        bytes >= misses * q40_bytes && bytes <= misses * q41_bytes, "unchanged 480*N route/payload accounting");
    require(a.routes.last_max_expert_group_assignments >= 1 && a.routes.last_max_expert_group_assignments <= n &&
        a.routes.expert_groups_gt128 >= b.routes.expert_groups_gt128, "route group bounds");
    const auto groups = a.routes.expert_groups_gt128 - b.routes.expert_groups_gt128;
    require(groups <= routes_per_row * n / 129 && (groups > 0) == (a.routes.last_max_expert_group_assignments > 128),
        "route threshold diagnostic");
    attention_equal(a.attention, expected_attention(b.attention, b.enabled, n));
}
void memory_record(Records& records, std::string_view event, int phase, qwen::Session& s, const Audit& audit) {
    std::ostringstream out; records.begin(out, "prefill_attention_memory", phase);
    out << ",\"event\":"; text(out, event);
    out << ",\"state\":"; state_json(out, state(s));
    out << ",\"memory_snapshot_index\":" << audit.snapshots - 1 << ",\"memory\":"; memory_json(out, audit.last);
    out << ",\"passed\":true}"; records.add(out);
}

struct Phase {
    int index;
    std::size_t calls = 0, windows = 0, accepted = 0, boundary_rows = 0, rejections = 0, toggle_checks = 0;
    std::array<std::uint64_t, 4> visible_mod4{};
    std::uint64_t finite_values = 0, compared_values = 0, bit_mismatches = 0, argmax_agree = 0;
    State teacher{}, final{};
};
struct Totals {
    std::uint64_t calls = 0, windows = 0, accepted = 0, finite_values = 0, compared_values = 0;
    std::uint64_t rejections = 0, toggle_checks = 0, boundary_rows = 0;
    qwen::SessionAttentionStats attention{};
};

// Return ONLY the latest accepted active span; never inspect an earlier span
// after a subsequent accepted call (or reset). The retained reference is owned.
std::span<const float> window(Records& records, qwen::Session& session, Reference& ref, Audit& audit, Phase& phase,
    std::size_t offset, std::size_t rows, bool singles, std::string_view segment) {
    require(offset == phase.accepted && rows && rows <= capacity - offset, "window chronology");
    const auto before = state(session); Errors errors; std::span<const float> active;
    const std::size_t calls_before = phase.calls;
    std::size_t completed_rows = 0;
    try {
        for (std::size_t t = 0; t < rows;) {
            const auto n = singles ? 1 : rows;
            const auto p = offset + t; const auto call_before = state(session);
            active = session.step_batch(std::span<const std::int32_t>(source_ids).subspan(p, n));
            ++phase.calls;
            require(active.size() == n * vocabulary, "completed full-vocabulary extent");
            completed_rows += n;
            // Freeze/apply gate BEFORE another Session call or preservation read.
            errors.inspect(active, phase.index ? ref.row_window(p, n) : std::span<const float>{}, p);
            if (!errors.passed()) throw std::runtime_error("frozen full-vocabulary gate failed");
            if (!phase.index) std::copy(active.begin(), active.end(), ref.row_window(p, n).begin());
            check_completed(state(session), call_before, p, n); audit.observe(session, ref, active);
            for (std::size_t r = p; r < p + n; ++r) {
                ++phase.visible_mod4[(r + 1) % 4]; phase.boundary_rows += r >= 2047 && r <= 2056;
            }
            t += n; phase.accepted += n;
        }
    } catch (const std::exception& e) {
        std::ostringstream out; records.begin(out, "prefill_attention_window", phase.index);
        out << ",\"window_index\":" << phase.windows << ",\"offset\":" << offset << ",\"rows\":" << rows
            << ",\"completed_rows\":" << completed_rows << ",\"completed_calls\":" << phase.calls - calls_before
            << ",\"segment\":"; text(out, segment);
        out << ",\"errors\":"; errors_json(out, errors);
        out << ",\"state_before\":"; state_json(out, before);
        out << ",\"state_after\":"; state_json(out, state(session));
        out << ",\"passed\":false,\"failure\":"; text(out, std::string_view(e.what()).substr(0, 1024)); out << '}';
        records.add(out); throw;
    }
    const auto after = state(session);
    phase.finite_values += errors.finite; phase.compared_values += errors.compared;
    phase.bit_mismatches += errors.bit_mismatches; phase.argmax_agree += errors.argmax_agree;
    std::ostringstream out; records.begin(out, "prefill_attention_window", phase.index);
    out << ",\"window_index\":" << phase.windows++ << ",\"offset\":" << offset << ",\"rows\":" << rows
        << ",\"completed_rows\":" << completed_rows << ",\"completed_calls\":" << phase.calls - calls_before
        << ",\"segment\":"; text(out, segment);
    out << ",\"expected_routes\":" << routes_per_row * rows << ",\"route_hits_delta\":" << after.stats.expert_hits - before.stats.expert_hits
        << ",\"route_misses_delta\":" << after.stats.expert_misses - before.stats.expert_misses
        << ",\"route_upload_bytes_delta\":" << after.stats.expert_upload_bytes - before.stats.expert_upload_bytes;
    out << ",\"state_before\":"; state_json(out, before); out << ",\"state_after\":"; state_json(out, after);
    out << ",\"errors\":"; errors_json(out, errors);
    out << ",\"accepted_rows_expected_visible_mod4\":"; array(out, phase.visible_mod4);
    out << ",\"accepted_rows_logical2047_through2056\":" << phase.boundary_rows
        << ",\"memory_snapshot_index\":" << audit.snapshots - 1 << ",\"memory\":"; memory_json(out, audit.last);
    out << ",\"passed\":true}"; records.add(out); return active;
}

void toggles(qwen::Session& s, Reference& ref, Audit& audit, Phase& phase, std::span<const float> active) {
    require(active.size() >= 997 * vocabulary && s.attention_batch(), "full old-span toggle witness");
    const auto before = state(s);
    for (bool enabled : {false, true, true}) {
        const auto saved = ref.snapshot(active); s.set_attention_batch(enabled);
        require(s.attention_batch() == enabled && state_key(state(s), true) == state_key(before, true), "toggle changed publication");
        exact(active, saved); audit.observe(s, ref, active); ++phase.toggle_checks;
    }
}
void reject(Records& records, qwen::Session& s, Reference& ref, Audit& audit, Phase& phase,
    std::span<const float> active, std::string_view reason, std::size_t rows, std::int32_t last) {
    const auto before = state(s); audit.observe(s, ref, active); const auto saved = ref.snapshot(active);
    bool rejected = false;
    try { (void)s.step_batch(ref.invalid_window(rows, last)); }
    catch (const std::invalid_argument&) { rejected = true; }
    const auto after = state(s);
    require(rejected && state_key(before) == state_key(after), "preflight rejection changed state/counters/mode");
    exact(active, saved); ref.check(); audit.observe(s, ref, active); ++phase.rejections;
    std::ostringstream out; records.begin(out, "prefill_attention_rejection", phase.index);
    out << ",\"reason\":"; text(out, reason);
    out << ",\"attempted_rows\":" << rows << ",\"last_token_id\":" << last << ",\"remaining_capacity\":"
        << capacity - before.stats.consumed_tokens << ",\"preserved_full_span_values\":" << active.size()
        << ",\"state_before\":"; state_json(out, before); out << ",\"state_after\":"; state_json(out, after);
    out << ",\"memory_snapshot_index\":" << audit.snapshots - 1 << ",\"memory\":"; memory_json(out, audit.last);
    out << ",\"rejected\":true,\"full_span_bit_preserved\":true,\"passed\":true}"; records.add(out);
}
void late_rejections(Records& records, qwen::Session& s, Reference& ref, Audit& audit, Phase& p, std::span<const float> active) {
    // Both bad IDs occur after a VALID 1023-token prefix, still within remaining
    // capacity. Length=1025 probe is in-range for remaining Session capacity too.
    reject(records, s, ref, audit, p, active, "late_negative_id", 1024, -1);
    reject(records, s, ref, audit, p, active, "late_vocab_id", 1024, static_cast<std::int32_t>(vocabulary));
    reject(records, s, ref, audit, p, active, "oversized_1025", 1025, 100);
}

void source_record(Records& records, const std::string& model, std::uint64_t model_bytes) {
    std::ostringstream out; records.begin(out, "prefill_attention_source");
    out << ",\"revision\":"; text(out, CORE_REVISION); out << ",\"dirty\":" << CORE_DIRTY << ",\"model\":"; text(out, model);
    out << ",\"model_bytes\":" << model_bytes << ",\"config\":{\"capacity\":2088,\"expert_slots\":1,\"max_batch_tokens\":1024,"
        "\"cpu_workers\":0,\"hybrid_probe\":false,\"trace_directory\":\"\",\"attention_query_tile\":8,\"initial_attention_batch\":false}"
        ",\"source_ids\":{\"bos\":248044,\"formula_after_bos\":\"99+absolute_position\",\"teacher_rows\":2056,"
        "\"teacher_last\":2154,\"continuation_rows\":32,\"continuation_first\":2155,\"continuation_last\":2186}"
        ",\"schedules\":[{\"phase\":\"reference_old_n1\",\"teacher_calls\":2056,\"teacher_call_rows\":1,\"continuation_calls\":32},"
        "{\"phase\":\"enabled1024\",\"teacher_chunks\":[1024,1024,8],\"continuation_calls\":32},"
        "{\"phase\":\"occupied5_enabled997\",\"prefix_n1_calls\":5,\"teacher_chunks\":[997,997,57],\"continuation_calls\":32}]"
        ",\"gate\":{\"absolute\":" << absolute_gate << ",\"relative\":" << relative_gate
        << ",\"formula\":\"abs(actual-reference)<=.02+.002*abs(reference)\",\"all_vocabulary\":true,\"immediate\":true,"
        "\"bit_identity_required\":false,\"argmax_identity_required\":false}"
        ",\"reference_bytes\":" << reference_bytes << ",\"preservation_bytes\":" << preservation_bytes
        << ",\"guard_bytes\":" << 4 * guards * sizeof(float) << ",\"private_workspace_bytes_per_device\":40338944,"
        "\"selection_bytes_per_device\":82080,\"expected_success_records\":31,\"scope\":{\"reference\":\"same-session old N1\","
        "\"component_gate_required_separately\":true,\"actual_selected_id_trace\":false,\"actual_visibility_trace\":false,"
        "\"hf_oracle\":false,\"performance_qualification\":false,\"long4k16k\":false,"
        "\"logical_row_coordinate\":\"zero-based absolute source position\","
        "\"expected_visibility\":\"position+1; computed from accepted rows, not GPU-observed\"},\"passed\":true}";
    records.add(out);
}
Totals run(Records& records, const std::string& model, bool& constructed, bool& cleaned,
    std::uint64_t& memory_snapshots, std::uint64_t& preserved_values, std::array<std::uint64_t, 2>& minimum_free) {
    Reference ref; qwen::SessionConfig config;
    config.capacity = capacity; config.expert_slots = 1; config.max_batch_tokens = max_tokens;
    config.attention_query_tile = query_tile;
    Totals totals;
    {
        qwen::Session session(model, config); constructed = true; Audit audit;
        require(!session.attention_batch(), "constructor did not default OFF");
        audit.observe(session, ref, {}); memory_record(records, "loaded", -1, session, audit);
        for (int index = 0; index < 3; ++index) {
            session.reset(); // No read of a span from the previous phase afterward.
            const auto reset_state = state(session);
            require(state_key(reset_state, true) == state_key({{}, {}, {}, false}), "reset publication not empty");
            require(reset_state.enabled == (index == 2), "reset did not retain mode");
            if (index == 1) {
                session.set_attention_batch(true);
                require(session.attention_batch() && state_key(state(session), true) == state_key(reset_state, true),
                    "initial enable changed publication");
            }
            audit.observe(session, ref, {}); memory_record(records, "reset", index, session, audit);
            Phase phase{index}; std::span<const float> active;
            if (!index) {
                active = window(records, session, ref, audit, phase, 0, teacher_rows, true, "teacher");
                phase.teacher = state(session);
                active = window(records, session, ref, audit, phase, teacher_rows, continuation_rows, true, "continuation");
            } else {
                if (index == 2) active = window(records, session, ref, audit, phase, 0, 5, true, "occupied_prefix");
                const std::size_t chunk = index == 1 ? 1024 : 997;
                bool first = true;
                while (phase.accepted < teacher_rows) {
                    const auto offset = phase.accepted, n = std::min(chunk, teacher_rows - offset);
                    active = window(records, session, ref, audit, phase, offset, n, false, "teacher");
                    if (first) {
                        toggles(session, ref, audit, phase, active); late_rejections(records, session, ref, audit, phase, active); first = false;
                    }
                }
                phase.teacher = state(session);
                active = window(records, session, ref, audit, phase, teacher_rows, 31, true, "continuation");
                reject(records, session, ref, audit, phase, active, "near_capacity_2_with_1_remaining", 2, 100);
                active = window(records, session, ref, audit, phase, capacity - 1, 1, true, "continuation_final");
            }
            phase.final = state(session);
            require(phase.accepted == capacity && phase.boundary_rows == 10 && phase.visible_mod4 == std::array<std::uint64_t, 4>{522,522,522,522},
                "accepted logical boundary/mod4 coverage");
            require(phase.calls == (index == 0 ? 2088U : index == 1 ? 35U : 40U) && phase.windows == (index == 0 ? 2U : index == 1 ? 5U : 6U),
                "phase calls/windows");
            require(phase.rejections == (index ? 4U : 0U) && phase.toggle_checks == (index ? 3U : 0U) &&
                phase.finite_values == capacity * vocabulary && phase.compared_values == (index ? capacity * vocabulary : 0), "phase values/proofs");
            const qwen::SessionAttentionStats expected = index == 0 ? qwen::SessionAttentionStats{} :
                index == 1 ? qwen::SessionAttentionStats{3084,24672,3084,24672,0,8} :
                    qwen::SessionAttentionStats{3096,24612,3084,24600,12,8};
            attention_equal(phase.final.attention, expected);
            attention_equal(phase.final.attention, phase.teacher.attention); // ALL continuation remains old N1.
            std::ostringstream out; records.begin(out, "prefill_attention_phase", index);
            out << ",\"accepted_rows\":" << phase.accepted << ",\"completed_calls\":" << phase.calls << ",\"windows\":" << phase.windows
                << ",\"finite_values\":" << phase.finite_values << ",\"compared_values\":" << phase.compared_values
                << ",\"bit_mismatches\":" << phase.bit_mismatches << ",\"argmax_agree\":" << phase.argmax_agree
                << ",\"accepted_rows_expected_visible_mod4\":"; array(out, phase.visible_mod4);
            out << ",\"accepted_rows_logical2047_through2056\":" << phase.boundary_rows
                << ",\"rejections\":" << phase.rejections << ",\"toggle_checks\":" << phase.toggle_checks
                << ",\"toggle_preserved_full_span_rows\":" << (index == 1 ? 1024 : index == 2 ? 997 : 0)
                << ",\"teacher_state\":"; state_json(out, phase.teacher); out << ",\"final_state\":"; state_json(out, phase.final);
            out << ",\"cache_retained_by_reset\":true,\"cold_or_warm_claim\":\"unknown\",\"passed\":true}"; records.add(out);
            totals.calls += phase.calls; totals.windows += phase.windows; totals.accepted += phase.accepted;
            totals.finite_values += phase.finite_values; totals.compared_values += phase.compared_values;
            totals.rejections += phase.rejections; totals.toggle_checks += phase.toggle_checks; totals.boundary_rows += phase.boundary_rows;
            const auto observed = phase.final.attention;
            totals.attention.batch_calls += observed.batch_calls; totals.attention.query_rows += observed.query_rows;
            totals.attention.multiquery_calls += observed.multiquery_calls; totals.attention.multiquery_rows += observed.multiquery_rows;
            totals.attention.singleton_tail_calls += observed.singleton_tail_calls;
            totals.attention.max_query_rows = std::max(totals.attention.max_query_rows, observed.max_query_rows);
            // `active` is the last live span only, dropped before the next reset.
            active = {};
        }
        session.reset(); require(session.attention_batch(), "final reset mode retention");
        require(state_key(state(session), true) == state_key({{}, {}, {}, false}), "final reset counters");
        audit.observe(session, ref, {}); memory_record(records, "final_reset", -1, session, audit);
        memory_snapshots = audit.snapshots; preserved_values = audit.preserved_values; minimum_free = audit.minimum_free;
        ref.check();
    }
    // Session has unwound; both upfront CPU buffers unwind on return.
    cleaned = true; return totals;
}
} // namespace

int main(int argc, char** argv) {
    Records records; bool constructed = false, cleaned = false;
    try {
        require(argc == 2, "usage: core-prefill-attention-test MODEL");
        const std::string_view revision(CORE_REVISION);
        for (std::size_t i = 0; i < revision.size(); i += 8) {
            std::uint32_t value = 0; const char* begin = revision.data() + i;
            const auto result = std::from_chars(begin, begin + 8, value, 16);
            require(result.ec == std::errc{} && result.ptr == begin + 8, "CORE_REVISION not 40-hex");
        }
        const std::string model(argv[1]);
        require(!model.empty() && std::filesystem::is_regular_file(model), "MODEL not regular file");
        const auto bytes = std::filesystem::file_size(model);
        require(bytes > 0 && bytes <= static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()), "MODEL file size");
        source_record(records, model, static_cast<std::uint64_t>(bytes));
        std::uint64_t snapshots = 0, preserved = 0; std::array<std::uint64_t, 2> minimum_free{};
        const auto totals = run(records, model, constructed, cleaned, snapshots, preserved, minimum_free);
        require(cleaned && records.count == success_records - 1 && totals.calls == 2163 && totals.windows == 13 &&
            totals.accepted == 6264 && totals.finite_values == 1555476480ULL && totals.compared_values == 1036984320ULL &&
            totals.rejections == 8 && totals.toggle_checks == 6 && totals.boundary_rows == 30 && snapshots == 2190,
            "complete fixture counts/cleanup");
        attention_equal(totals.attention, {6180,49284,6168,49272,12,8});
        require(totals.attention.multiquery_calls > 0 && totals.attention.multiquery_rows > 0, "no actual reported batched-query calls");
        std::ostringstream out; records.begin(out, "prefill_attention_complete");
        out << ",\"records\":31,\"phases\":3,\"windows\":" << totals.windows << ",\"completed_calls\":" << totals.calls
            << ",\"accepted_rows\":" << totals.accepted << ",\"finite_values\":" << totals.finite_values
            << ",\"compared_values\":" << totals.compared_values << ",\"rejections\":" << totals.rejections
            << ",\"toggle_checks\":" << totals.toggle_checks << ",\"accepted_rows_logical2047_through2056\":" << totals.boundary_rows
            << ",\"attention_stats\":"; attention_json(out, totals.attention);
        out << ",\"memory_snapshots\":" << snapshots << ",\"preserved_values\":" << preserved << ",\"minimum_free_vram\":"; array(out, minimum_free);
        out << ",\"session_constructed\":true,\"all_owners_unwound\":true,\"reference\":\"same-session old N1\","
            "\"physical_kernel_count_claim\":false,\"selected_id_trace_claim\":false,\"hf_oracle_claim\":false,"
            "\"performance_qualification\":false,\"passed\":true}"; records.add(out);
        records.flush(); return 0;
    } catch (const std::exception& e) {
        // Any runtime/fixture exception has already unwound all Session/float
        // owners. A destructor's completed RAII scope is not a GPU success claim.
        std::ostringstream out; records.begin(out, "prefill_attention_failure");
        out << ",\"passed\":false,\"session_constructed\":" << constructed << ",\"all_owners_unwound\":true,\"failure\":";
        text(out, std::string_view(e.what()).substr(0, 2048)); out << '}';
        try { records.add(out); records.flush(); } catch (...) { }
        return 1;
    }
}
