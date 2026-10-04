#include "session.hpp"
#include "ple.hpp"
#include "quant.hpp"
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifndef CORE_REVISION
#error "core-session-restore-test requires CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-session-restore-test requires CORE_DIRTY"
#endif

namespace {
constexpr int capacity = 40, batch = 3, width = 10240, vocabulary = 248320;
constexpr double absolute_gate = .02, relative_gate = .002;
constexpr std::uint64_t expert_payload_bytes = 68262297600ULL, expert_payload_reads = 144;
constexpr int protocol = 1;
std::uint64_t records_emitted = 0;
static_assert(sizeof(float) == 4 && sizeof(double) == 8);
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
static_assert(std::string_view(CORE_REVISION).size() == 40);
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void json_string(std::ostream& out, std::string_view value) {
    constexpr char hex[] = "0123456789abcdef";
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
        else if (c < 0x20) out << "\\u00" << hex[c >> 4] << hex[c & 15];
        else out << static_cast<char>(c);
    }
    out << '"';
}
void record_begin(std::string_view kind) {
    ++records_emitted;
    std::cout << "{\"protocol\":" << protocol << ",\"kind\":"; json_string(std::cout, kind);
}
template<class T, std::size_t N> void json_array(std::ostream& out, const std::array<T, N>& values) {
    out << '[';
    for (std::size_t i = 0; i < N; ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}
void hip_ok(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void exact(std::span<const float> a, std::span<const float> b) {
    require(a.size() == b.size(), "bit-preservation extent");
    require(std::memcmp(a.data(), b.data(), a.size_bytes()) == 0, "bits changed");
}
struct Comparisons {
    std::uint64_t values = 0, bits_different = 0;
    double max_error = 0, max_bound_ratio = 0;
    void compare(std::span<const float> actual, std::span<const float> reference) {
        require(!actual.empty() && actual.size() == reference.size(), "full vocabulary extent");
        for (std::size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]) && std::isfinite(reference[i]), "nonfinite full vocabulary");
            const auto error = std::abs(double(actual[i]) - double(reference[i]));
            const auto bound = absolute_gate + relative_gate * std::abs(double(reference[i]));
            require(error <= bound, "frozen full-vocabulary numerical gate");
            max_error = std::max(max_error, error); max_bound_ratio = std::max(max_bound_ratio, error / bound);
            bits_different += std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(reference[i]);
        }
        values += actual.size();
    }
};
bool same_stats(qwen::SessionStats a, qwen::SessionStats b, bool cursor = true) {
    return (!cursor || a.consumed_tokens == b.consumed_tokens) && a.expert_hits == b.expert_hits &&
        a.expert_misses == b.expert_misses && a.expert_upload_bytes == b.expert_upload_bytes &&
        std::bit_cast<std::uint64_t>(a.last_completed_ms) == std::bit_cast<std::uint64_t>(b.last_completed_ms);
}
template<class T> bool word_equal(const T& a, const T& b) {
    static_assert(sizeof(T) % sizeof(std::uint64_t) == 0 && std::is_trivially_copyable_v<T>);
    return std::bit_cast<std::array<std::uint64_t, sizeof(T) / sizeof(std::uint64_t)>>(a) ==
           std::bit_cast<std::array<std::uint64_t, sizeof(T) / sizeof(std::uint64_t)>>(b);
}
// Value-only snapshots: pointer addresses are evidence of borrowed-view identity,
// never dereferenced here, including after a successful forward expires a view.
struct PublicSnapshot {
    qwen::SessionStats stats;
    qwen::SessionRouteStats route;
    qwen::SessionHybridStats hybrid;
    qwen::SessionAttentionStats attention;
    qwen::SessionSpeculativeStats speculative;
    qwen::SessionTargetTap tap;
    bool checkpoint_available;
    qwen::SessionCheckpointState checkpoint;
};
PublicSnapshot public_snapshot(const qwen::Session& s, bool checkpoint_available = true) {
    return {s.stats(), s.route_stats(), s.hybrid_stats(), s.attention_stats(), s.speculative_stats(), s.target_tap(),
        checkpoint_available, checkpoint_available ? s.checkpoint_state() : qwen::SessionCheckpointState{}};
}
void write_public(std::ostream& out, const PublicSnapshot& p) {
    out << "{\"stats\":{\"consumed_tokens\":" << p.stats.consumed_tokens;
#define FIELD(object, name) out << ",\"" #name "\":" << object.name
    FIELD(p.stats, expert_hits); FIELD(p.stats, expert_misses); FIELD(p.stats, expert_upload_bytes);
    out << ",\"last_completed_ms_bits\":" << std::bit_cast<std::uint64_t>(p.stats.last_completed_ms) << "},\"route\":{"
        << "\"last_max_expert_group_assignments\":" << p.route.last_max_expert_group_assignments;
    FIELD(p.route, expert_groups_gt128);
    out << "},\"hybrid\":{\"short_layers\":" << p.hybrid.short_layers;
    FIELD(p.hybrid, gpu_only_wide_layers); FIELD(p.hybrid, ready_hit_assignments); FIELD(p.hybrid, physical_miss_assignments);
    FIELD(p.hybrid, group_reuse_assignments); FIELD(p.hybrid, cpu_groups); FIELD(p.hybrid, cpu_assignments);
    FIELD(p.hybrid, gpu_hit_groups); FIELD(p.hybrid, gpu_hit_assignments); FIELD(p.hybrid, gpu_miss_groups);
    FIELD(p.hybrid, gpu_miss_assignments); FIELD(p.hybrid, admitted_groups); FIELD(p.hybrid, evicted_ready_slots);
    FIELD(p.hybrid, input_extractions); FIELD(p.hybrid, input_bytes); FIELD(p.hybrid, cpu_return_bytes);
    FIELD(p.hybrid, cpu_input_bytes_checked); FIELD(p.hybrid, all_hit_layers); FIELD(p.hybrid, forced_cpu_layers);
    FIELD(p.hybrid, forced_gpu_layers); FIELD(p.hybrid, cpu_gate_up_jobs); FIELD(p.hybrid, cpu_down_jobs);
    FIELD(p.hybrid, gpu_middle_columns); FIELD(p.hybrid, gpu_middle_batches); FIELD(p.hybrid, paired_gate_up_bytes);
    FIELD(p.hybrid, middle_q8_bytes);
    out << "},\"attention\":{\"batch_calls\":" << p.attention.batch_calls;
    FIELD(p.attention, query_rows); FIELD(p.attention, multiquery_calls); FIELD(p.attention, multiquery_rows);
    FIELD(p.attention, singleton_tail_calls); FIELD(p.attention, max_query_rows);
    out << "},\"speculative\":{\"target_forward_rows\":" << p.speculative.target_forward_rows;
    FIELD(p.speculative, verify_windows); FIELD(p.speculative, verify_rows); FIELD(p.speculative, restore_calls);
    FIELD(p.speculative, retained_inputs); FIELD(p.speculative, gdn_prefix_calls); FIELD(p.speculative, ple_prefix_calls);
    FIELD(p.speculative, qsa_tail_prefix_calls);
    out << ",\"restored_prefixes\":"; json_array(out, p.speculative.restored_prefixes);
    out << "},\"target_tap\":{\"device\":" << p.tap.device;
    FIELD(p.tap, rows); FIELD(p.tap, width); FIELD(p.tap, first_position);
    out << ",\"pointer_address\":" << static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(p.tap.pointer))
        << ",\"pointer_present\":" << (p.tap.pointer ? "true" : "false") << "},\"checkpoint_available\":"
        << (p.checkpoint_available ? "true" : "false") << ",\"pending_window\":";
    if (p.checkpoint_available) {
        out << "{\"enabled\":" << (p.checkpoint.enabled ? "true" : "false")
            << ",\"pending\":" << (p.checkpoint.pending ? "true" : "false");
        FIELD(p.checkpoint, start_position); FIELD(p.checkpoint, inputs); FIELD(p.checkpoint, valid_slots); out << '}';
    } else out << "null,\"checkpoint_unavailable_reason\":\"execution_failure_requires_reset\"";
    out << '}';
#undef FIELD
}
struct PhaseEvidence {
    std::uint64_t compared_full_vocab_values = 0, tap_exact_values = 0;
    std::uint64_t successful_restores = 0, argument_rejections = 0, steady_memory_observations = 0, sticky_rejections = 0;
};
void write_phase(std::ostream& out, const PhaseEvidence& p) {
    out << "{\"compared_full_vocab_values\":" << p.compared_full_vocab_values
        << ",\"compared_full_vocab_rows\":" << p.compared_full_vocab_values / vocabulary
        << ",\"tap_exact_values\":" << p.tap_exact_values << ",\"successful_restores\":" << p.successful_restores
        << ",\"argument_rejections\":" << p.argument_rejections << ",\"steady_memory_observations\":" << p.steady_memory_observations
        << ",\"sticky_rejections\":" << p.sticky_rejections << '}';
}
struct RejectionEvidence {
    std::string_view group;
    std::uint64_t count;
    PublicSnapshot before, after;
};
void write_rejection(std::ostream& out, const RejectionEvidence& e) {
    out << "{\"group\":"; json_string(out, e.group);
    out << ",\"count\":" << e.count << ",\"before\":"; write_public(out, e.before);
    out << ",\"after\":"; write_public(out, e.after); out << '}';
}
void steady(const qwen::SessionMemory& a, const qwen::SessionMemory& b) {
    require(a.ownership_verified && b.ownership_verified, "independent live-Buffer ledger");
    require(a.capacity == b.capacity && a.expert_slots == b.expert_slots && a.cpu_workers == b.cpu_workers &&
        a.attention_query_tile == b.attention_query_tile && a.speculative_checkpoints == b.speculative_checkpoints &&
        a.host_routing_capacity == b.host_routing_capacity && a.host_route_group_payload == b.host_route_group_payload &&
        a.pinned_route_metadata == b.pinned_route_metadata && a.route_metadata_bytes == b.route_metadata_bytes &&
        a.pinned_expert_staging == b.pinned_expert_staging && a.host_hybrid_probe == b.host_hybrid_probe &&
        a.host_hybrid_input_probe == b.host_hybrid_input_probe &&
        a.pinned_hybrid_input == b.pinned_hybrid_input && a.pinned_hybrid_output == b.pinned_hybrid_output &&
        a.pinned_hybrid_error == b.pinned_hybrid_error && a.host_hybrid_plans == b.host_hybrid_plans &&
        a.host_cpu_expert_views == b.host_cpu_expert_views && a.cpu_pool_metadata == b.cpu_pool_metadata &&
        a.cpu_pool_scratch == b.cpu_pool_scratch && a.pinned_hybrid_middle == b.pinned_hybrid_middle &&
        a.hybrid_contribution_bytes == b.hybrid_contribution_bytes &&
        a.expert_stage_capacity_bytes == b.expert_stage_capacity_bytes &&
        a.hybrid_gate_up_bytes == b.hybrid_gate_up_bytes && a.hybrid_middle_float_bytes == b.hybrid_middle_float_bytes &&
        a.hybrid_middle_q8_bytes == b.hybrid_middle_q8_bytes && a.hybrid_middle_error_bytes == b.hybrid_middle_error_bytes,
        "constructor policy/base host resources changed");
    require(a.ram_expert_capacity == b.ram_expert_capacity && a.ram_expert_payload == b.ram_expert_payload &&
        a.expert_payload_reads == b.expert_payload_reads && a.expert_payload_bytes_read == b.expert_payload_bytes_read &&
        a.host_logit_capacity == b.host_logit_capacity && a.host_embedding_capacity == b.host_embedding_capacity &&
        a.pinned_handoff == b.pinned_handoff && a.host_session_impl_bytes == b.host_session_impl_bytes &&
        a.host_session_config_bytes == b.host_session_config_bytes &&
        a.host_speculative_owner_bytes == b.host_speculative_owner_bytes &&
        a.host_speculative_hash_payload_bytes == b.host_speculative_hash_payload_bytes &&
        a.checkpoint_recurrent_bytes == b.checkpoint_recurrent_bytes &&
        a.checkpoint_history_bytes == b.checkpoint_history_bytes && a.checkpoint_qsa_tail_bytes == b.checkpoint_qsa_tail_bytes &&
        a.checkpoint_ple_history_bytes == b.checkpoint_ple_history_bytes && a.target_tap_bytes == b.target_tap_bytes &&
        a.target_tap_staging_bytes == b.target_tap_staging_bytes, "host/snapshot memory changed");
    for (int id = 0; id < 2; ++id) {
        const auto& x = a.devices[id]; const auto& y = b.devices[id];
        require(x.weights == y.weights && x.expert_slots == y.expert_slots && x.qsa_kv == y.qsa_kv &&
            x.qsa_index == y.qsa_index && x.gdn_state == y.gdn_state && x.ple_state == y.ple_state &&
            x.workspace == y.workspace && x.owned_bytes == y.owned_bytes && x.owned_buffers == y.owned_buffers &&
            x.owned_peak_bytes == y.owned_peak_bytes, "GPU owned capacities changed");
        require(x.weights + x.expert_slots + x.qsa_kv + x.qsa_index + x.gdn_state + x.ple_state + x.workspace ==
            x.owned_bytes, "category traversal sum");
    }
}
void geometry(const qwen::SessionMemory& m, int rows, int context) {
    require(m.speculative_checkpoints && m.host_speculative_owner_bytes > 0 &&
        m.host_speculative_hash_payload_bytes > 0 && m.host_session_impl_bytes > 0 &&
        m.host_session_config_bytes == sizeof(qwen::SessionConfig), "snapshot host accounting");
    for (int id = 0; id < 2; ++id) {
        require(m.checkpoint_recurrent_bytes[id] == 18ULL * 4 * 786432 * sizeof(float) &&
            m.checkpoint_history_bytes[id] == 18ULL * 4 * 30720 * sizeof(float) &&
            m.checkpoint_qsa_tail_bytes[id] == 6ULL * 4 * 384 * sizeof(float) &&
            m.checkpoint_ple_history_bytes[id] == (id == 0 ? 4ULL * 92160 * sizeof(float) : 0) &&
            m.target_tap_bytes[id] == (id == 1 ? static_cast<std::uint64_t>(rows) * width * sizeof(float) : 0) &&
            m.target_tap_staging_bytes[id] == m.target_tap_bytes[id], "four-slot/device/double-tap byte geometry");
        require(m.devices[id].qsa_kv == 6ULL * context * 2 * 16 * 18 &&
            m.devices[id].qsa_index == 6ULL * (context / 4 * 128 + 384) * sizeof(float), "Q4_0 KV/FP32 index geometry");
    }
}
void write_memory(std::ostream& out, const qwen::SessionMemory& m) {
    out << "{\"capacity\":" << m.capacity;
#define FIELD(name) out << ",\"" #name "\":" << m.name
#define ARRAY(name) out << ",\"" #name "\":"; json_array(out, m.name)
    FIELD(expert_slots); FIELD(cpu_workers); FIELD(attention_query_tile);
    out << ",\"speculative_checkpoints\":" << (m.speculative_checkpoints ? "true" : "false")
        << ",\"ownership_verified\":" << (m.ownership_verified ? "true" : "false");
    FIELD(ram_expert_capacity); FIELD(ram_expert_payload); FIELD(host_embedding_capacity); FIELD(host_logit_capacity);
    FIELD(pinned_handoff); FIELD(expert_payload_reads); FIELD(expert_payload_bytes_read); FIELD(pinned_expert_staging);
    FIELD(pinned_hybrid_input); FIELD(pinned_hybrid_output); FIELD(pinned_hybrid_error); FIELD(pinned_hybrid_middle);
    FIELD(host_hybrid_plans); FIELD(host_cpu_expert_views); FIELD(cpu_pool_metadata); FIELD(cpu_pool_scratch);
    FIELD(host_hybrid_probe); FIELD(host_routing_capacity); FIELD(host_route_group_payload); FIELD(pinned_route_metadata);
    FIELD(host_hybrid_input_probe); FIELD(host_session_impl_bytes); FIELD(host_session_config_bytes);
    FIELD(host_speculative_owner_bytes); FIELD(host_speculative_hash_payload_bytes);
    ARRAY(hybrid_contribution_bytes); ARRAY(expert_stage_capacity_bytes); ARRAY(route_metadata_bytes);
    ARRAY(hybrid_gate_up_bytes); ARRAY(hybrid_middle_float_bytes); ARRAY(hybrid_middle_q8_bytes); ARRAY(hybrid_middle_error_bytes);
    ARRAY(checkpoint_recurrent_bytes); ARRAY(checkpoint_history_bytes); ARRAY(checkpoint_qsa_tail_bytes);
    ARRAY(checkpoint_ple_history_bytes); ARRAY(target_tap_bytes); ARRAY(target_tap_staging_bytes);
#undef ARRAY
#undef FIELD
    out << ",\"attention\":[";
    for (int id = 0; id < 2; ++id) {
        if (id) out << ',';
        const auto& a = m.attention[id];
        out << "{\"gathered_key_bytes\":" << a.gathered_key_bytes;
#define FIELD(object, name) out << ",\"" #name "\":" << object.name
        FIELD(a, gathered_value_bytes); FIELD(a, partial_output_bytes); FIELD(a, partial_max_sum_bytes);
        FIELD(a, staged_buffer_bytes); FIELD(a, output_buffer_bytes); FIELD(a, selected_id_bytes);
        FIELD(a, selected_block_bytes); FIELD(a, selected_count_bytes); FIELD(a, private_workspace_bytes); out << '}';
    }
    out << "],\"devices\":[";
    for (int id = 0; id < 2; ++id) {
        if (id) out << ',';
        const auto& d = m.devices[id];
        out << "{\"device\":" << d.device;
        FIELD(d, first_layer); FIELD(d, last_layer); FIELD(d, gdn_layers); FIELD(d, qsa_layers);
        FIELD(d, weights); FIELD(d, expert_slots); FIELD(d, qsa_kv); FIELD(d, qsa_index);
        FIELD(d, gdn_state); FIELD(d, ple_state); FIELD(d, workspace); FIELD(d, owned_bytes);
        FIELD(d, owned_peak_bytes); FIELD(d, owned_buffers); FIELD(d, total_vram); FIELD(d, free_vram);
        out << ",\"category_sum_bytes\":" << d.weights + d.expert_slots + d.qsa_kv + d.qsa_index +
            d.gdn_state + d.ple_state + d.workspace << '}';
    }
#undef FIELD
    out << "]}";
}
void loaded_memory(std::string_view owner, int owner_index, const qwen::SessionConfig& config,
                   const qwen::SessionMemory& m, const PublicSnapshot& initial) {
    // One constructor snapshot per sequential scope, taken by the existing
    // memory() call. No payload re-read, RSS sampling, or release measurement.
    require(m.capacity == config.capacity && m.expert_slots == config.expert_slots && m.cpu_workers == config.cpu_workers &&
        m.expert_payload_reads == expert_payload_reads && m.expert_payload_bytes_read == expert_payload_bytes &&
        m.ram_expert_payload == expert_payload_bytes && m.ram_expert_capacity >= m.ram_expert_payload &&
        m.host_logit_capacity >= 2ULL * config.max_batch_tokens * vocabulary * sizeof(float), "loaded owner/payload/capacity evidence");
    steady(m, m);
    const std::array<std::uint64_t, 2> prefix_buffers{
        static_cast<std::uint64_t>(2 * m.devices[0].gdn_layers + m.devices[0].qsa_layers + 1),
        static_cast<std::uint64_t>(2 * m.devices[1].gdn_layers + m.devices[1].qsa_layers + 2)};
    require(prefix_buffers == std::array<std::uint64_t, 2>{43, 44}, "validated snapshot Buffer geometry");
    record_begin("session_restore_loaded_memory");
    std::cout << ",\"owner\":"; json_string(std::cout, owner);
    std::cout << ",\"owner_index\":" << owner_index << ",\"live_session_owners\":1,\"config\":{\"capacity\":" << config.capacity
        << ",\"expert_slots\":" << config.expert_slots << ",\"max_batch_tokens\":" << config.max_batch_tokens
        << ",\"cpu_workers\":" << config.cpu_workers << ",\"hybrid_probe\":" << (config.hybrid_probe ? "true" : "false")
        << ",\"attention_query_tile\":" << config.attention_query_tile << ",\"trace_first_token\":" << config.trace_first_token
        << ",\"trace_directory\":"; json_string(std::cout, config.trace_directory);
    std::cout << ",\"speculative_checkpoints\":" << (config.speculative_checkpoints ? "true" : "false")
        << "},\"tap_capacity\":{\"device\":1,\"rows\":" << config.max_batch_tokens << ",\"width\":" << width
        << ",\"published_bytes\":" << m.target_tap_bytes[1] << ",\"staging_bytes\":" << m.target_tap_staging_bytes[1]
        << "},\"speculative_buffer_counts_from_validated_geometry\":"; json_array(std::cout, prefix_buffers);
    std::cout << ",\"speculative_buffer_count_total_from_geometry\":" << prefix_buffers[0] + prefix_buffers[1]
        << ",\"speculative_buffer_bytes_by_device\":[";
    for (int id = 0; id < 2; ++id) {
        if (id) std::cout << ',';
        std::cout << m.checkpoint_recurrent_bytes[id] + m.checkpoint_history_bytes[id] + m.checkpoint_qsa_tail_bytes[id] +
            m.checkpoint_ple_history_bytes[id] + m.target_tap_bytes[id] + m.target_tap_staging_bytes[id];
    }
    std::cout << "],\"host_logit_buffer_count_required\":2,\"host_each_logit_requested_bytes\":"
        << static_cast<std::uint64_t>(config.max_batch_tokens) * vocabulary * sizeof(float)
        << ",\"host_logit_combined_capacity_floor_bytes\":" << 2ULL * config.max_batch_tokens * vocabulary * sizeof(float)
        << ",\"memory\":"; write_memory(std::cout, m);
    std::cout << ",\"initial_public\":"; write_public(std::cout, initial);
    std::cout << ",\"ledger_scope\":\"all_live_Session_Buffer_pointer_tree_and_reported_host_capacities\""
        << ",\"host_config_included_in_impl\":true,\"host_input_probe_included_in_hybrid_probe\":true"
        << ",\"speculative_GPU_bytes_included_in_workspace\":true"
        << ",\"constructor_read_scope\":\"expert_payload_API_reads_not_physical_SSD_trace\""
        << ",\"peak_scope\":\"Buffer_process_peak_not_reset_between_sequential_owners\""
        << ",\"evidence_limits\":[\"separate_working_logit_capacity_not_exposed\",\"snapshot_Buffer_count_geometry_derived\","
           "\"private_hash_history_and_route_allocator_slack_excluded\",\"allocator_overhead_and_thread_stacks_excluded\","
           "\"HIP_rocBLAS_context_allocations_not_owned_ledger\"]"
        << ",\"full_RSS_measured\":false,\"owned_buffer_release_measured\":false,\"passed\":true}\n";
}
void read_tap(qwen::SessionTargetTap tap, std::span<float> destination) {
    require(tap.device == 1 && tap.pointer && tap.width == width && tap.rows > 0 && tap.rows <= 1024 &&
        destination.size() == static_cast<std::size_t>(tap.rows) * width, "typed GPU tap extent");
    int previous = 0; hip_ok(hipGetDevice(&previous));
    struct RestoreDevice {
        int device;
        ~RestoreDevice() { if (hipSetDevice(device) != hipSuccess) std::terminate(); }
    } restore{previous};
    hip_ok(hipSetDevice(tap.device));
    hip_ok(hipMemcpy(destination.data(), tap.pointer, destination.size_bytes(), hipMemcpyDeviceToHost));
    for (float v : destination) require(std::isfinite(v), "nonfinite widened tap");
}
struct Fixture {
    // Bounded, once-allocated references, BEFORE the restoration Session.
    std::vector<float> logits = std::vector<float>(capacity * vocabulary);
    std::vector<float> taps = std::vector<float>(capacity * width);
    std::vector<float> saved_logits = std::vector<float>(batch * vocabulary);
    std::vector<float> saved_tap = std::vector<float>(batch * width);
    std::vector<float> readback = std::vector<float>(batch * width);
    std::vector<float> alternate_logits = std::vector<float>(batch * vocabulary);
    std::vector<float> alternate_taps = std::vector<float>(batch * width);
    std::vector<float> branch_logits = std::vector<float>(4 * batch * vocabulary);
    std::vector<float> branch_taps = std::vector<float>(4 * batch * width);
    std::array<std::int32_t, capacity> ids{};
    Comparisons comparison;
    std::uint64_t rejects = 0, restores = 0, tap_exact_values = 0, observations = 0;
    std::span<const float> logit_reference(int start, int rows) const {
        return std::span(logits).subspan(static_cast<std::size_t>(start) * vocabulary,
                                       static_cast<std::size_t>(rows) * vocabulary);
    }
    void tap(qwen::Session& s, int start, int rows, bool reference = false) {
        const auto view = s.target_tap();
        require(view.first_position == static_cast<std::uint64_t>(start) && view.rows == rows, "latest tap rows/position");
        auto observed = std::span(readback).first(static_cast<std::size_t>(rows) * width);
        read_tap(view, observed);
        auto retained = std::span(taps).subspan(static_cast<std::size_t>(start) * width, observed.size());
        if (reference) std::copy(observed.begin(), observed.end(), retained.begin());
        else { exact(observed, retained); tap_exact_values += observed.size(); }
    }
    void memory(qwen::Session& s, const qwen::SessionMemory& loaded) {
        const auto stats = s.stats(); const auto speculative = s.speculative_stats();
        const auto state = s.checkpoint_state(); const auto view = s.target_tap();
        steady(s.memory(), loaded);
        require(same_stats(s.stats(), stats) && s.speculative_stats() == speculative && s.checkpoint_state() == state &&
            s.target_tap() == view, "inspection changed publication");
        ++observations;
    }
    template<class Call> void reject(qwen::Session& s, std::span<const float> published,
                                    const qwen::SessionMemory& loaded, Call&& call) {
        require(!published.empty() && published.size() <= saved_logits.size(), "rejection full-span storage");
        std::copy(published.begin(), published.end(), saved_logits.begin());
        const auto stats = s.stats(); const auto spec = s.speculative_stats(); const auto checkpoint = s.checkpoint_state();
        const auto routes = s.route_stats(); const auto attention = s.attention_stats(); const auto hybrid = s.hybrid_stats();
        const auto view = s.target_tap();
        auto saved = std::span(saved_tap).first(static_cast<std::size_t>(view.rows) * width);
        if (view.rows) read_tap(view, saved);
        bool rejected = false;
        try { call(); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "expected argument rejection");
        require(same_stats(s.stats(), stats) && s.speculative_stats() == spec && s.checkpoint_state() == checkpoint &&
            word_equal(s.route_stats(), routes) && word_equal(s.attention_stats(), attention) &&
            word_equal(s.hybrid_stats(), hybrid) && s.target_tap() == view, "rejection changed stats/window/tap");
        exact(published, std::span(saved_logits).first(published.size()));
        if (view.rows) {
            auto again = std::span(readback).first(saved.size()); read_tap(view, again); exact(again, saved);
        }
        require(s.hybrid_intermediates().routed_down.empty() && s.hybrid_intermediates().ffn_output.empty() &&
            s.hybrid_inputs().original_q8.empty() && s.hybrid_inputs().expert_ids.empty(), "disabled CPU published extents");
        memory(s, loaded); ++rejects;
    }
    void invalid_pending(qwen::Session& s, std::span<const float> logits_view, const qwen::SessionMemory& loaded, int n) {
        const std::array<std::int32_t, 4> oversized{100, 101, 102, 103};
        const std::array<std::int32_t, 3> negative{100, 101, -1}, late_id{100, 101, vocabulary};
        reject(s, logits_view, loaded, [&] { (void)s.verify_window({}); });
        reject(s, logits_view, loaded, [&] { (void)s.step_batch({}); });
        reject(s, logits_view, loaded, [&] { (void)s.verify_window(oversized); });
        reject(s, logits_view, loaded, [&] { (void)s.step_batch(oversized); });
        reject(s, logits_view, loaded, [&] { (void)s.verify_window(negative); });
        reject(s, logits_view, loaded, [&] { (void)s.step_batch(negative); });
        reject(s, logits_view, loaded, [&] { (void)s.verify_window(late_id); });
        reject(s, logits_view, loaded, [&] { (void)s.step_batch(late_id); });
        reject(s, logits_view, loaded, [&] { s.restore_prefix(-1); });
        reject(s, logits_view, loaded, [&] { s.restore_prefix(n + 1); });
        reject(s, logits_view, loaded, [&] { (void)s.verify_window(std::span(oversized).first(1)); });
    }
    void reset(qwen::Session& s) {
        const auto policy = s.hybrid_policy(); const bool attention = s.attention_batch();
        s.reset();
        require(same_stats(s.stats(), {}) && s.speculative_stats() == qwen::SessionSpeculativeStats{} &&
            s.checkpoint_state() == qwen::SessionCheckpointState{true, false, 0, 0, 0} &&
            s.target_tap().rows == 0 && !s.target_tap().pointer && s.attention_batch() == attention &&
            s.hybrid_policy().mode == policy.mode && s.hybrid_policy().gpu_miss_groups == policy.gpu_miss_groups &&
            s.hybrid_policy().diagnostic_fail_after_admission == policy.diagnostic_fail_after_admission,
            "reset clear/policy retention");
    }
    void prefix(qwen::Session& s, int count) {
        for (int i = 0; i < count; ++i) {
            const auto output = s.step(ids[i]); comparison.compare(output, logit_reference(i, 1)); tap(s, i, 1);
        }
    }
};

PhaseEvidence wide_pp(const char* path, Fixture& f, int rows) {
    // Separate sequential instance: no concurrent resident models. Default32
    // is bounded; --wide-1024 explicitly selects the expensive maximum gate.
    std::vector<std::int32_t> ids(rows);
    std::vector<float> reference(static_cast<std::size_t>(rows) * vocabulary);
    std::vector<float> tap_reference(static_cast<std::size_t>(rows) * width);
    std::vector<float> saved_tap(tap_reference.size()), observed(tap_reference.size());
    std::vector<float> saved_logits(reference.size());
    for (int t = 0; t < rows; ++t) ids[t] = f.ids[t % capacity];
    // Enough remaining capacity to reach the actual late-ID validation after
    // a full-width publication; rejection must not merely fail its length gate.
    qwen::SessionConfig config; config.capacity = std::max(40, 2 * rows); config.expert_slots = 1;
    config.max_batch_tokens = rows; config.speculative_checkpoints = true;
    PhaseEvidence evidence;
    const auto values_before = f.comparison.values, taps_before = f.tap_exact_values;
    {
        qwen::Session s(path, config); const auto loaded = s.memory(); geometry(loaded, rows, config.capacity);
        loaded_memory(rows == 1024 ? "wide_PP_1024" : "wide_PP_32", 1, config, loaded, public_snapshot(s));
        // Same Session supplies every retained N1 vocabulary/tap reference row.
        for (int t = 0; t < rows; ++t) {
            const auto output = s.step(ids[t]);
            for (float v : output) require(std::isfinite(v), "wide N1 reference finite");
            std::copy(output.begin(), output.end(), reference.begin() + static_cast<std::size_t>(t) * vocabulary);
            read_tap(s.target_tap(), std::span(tap_reference).subspan(static_cast<std::size_t>(t) * width, width));
        }
        f.reset(s);
        const auto preverify = public_snapshot(s);
        (void)s.verify_window(std::span(ids).first(1));
        const auto postverify = public_snapshot(s);
        s.restore_prefix(0); ++evidence.successful_restores;
        const auto postrestore = public_snapshot(s);
        const auto before = s.speculative_stats();
        const auto output = s.step_batch(ids); f.comparison.compare(output, reference);
        require(!s.checkpoint_state().pending && s.target_tap().rows == rows && s.target_tap().first_position == 0 &&
            s.speculative_stats().target_forward_rows == before.target_forward_rows + rows &&
            s.speculative_stats().gdn_prefix_calls == before.gdn_prefix_calls &&
            s.speculative_stats().ple_prefix_calls == before.ple_prefix_calls &&
            s.speculative_stats().qsa_tail_prefix_calls == before.qsa_tail_prefix_calls, "actual wide PP tap/count geometry");
        read_tap(s.target_tap(), saved_tap);
        // Wide MMQ may change bounded reduction order; use the SAME frozen
        // gate for N1 hidden values, and exact bytes for publication/replay.
        Comparisons hidden; hidden.compare(saved_tap, tap_reference);
        std::copy(output.begin(), output.end(), saved_logits.begin());
        const auto tap = s.target_tap(); const auto stats = s.stats(); const auto spec = s.speculative_stats();
        const auto state = s.checkpoint_state();
        const auto route = s.route_stats(); const auto attention = s.attention_stats(); const auto hybrid = s.hybrid_stats();
        const auto prereject = public_snapshot(s);
        ids.back() = vocabulary;
        bool rejected = false; try { (void)s.step_batch(ids); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && same_stats(s.stats(), stats) && s.speculative_stats() == spec && s.checkpoint_state() == state &&
            word_equal(s.route_stats(), route) && word_equal(s.attention_stats(), attention) &&
            word_equal(s.hybrid_stats(), hybrid) && s.target_tap() == tap, "wide rejection publication");
        exact(output, saved_logits); read_tap(tap, observed); exact(observed, saved_tap); steady(s.memory(), loaded);
        ++evidence.argument_rejections; ++evidence.steady_memory_observations;
        const auto postreject = public_snapshot(s);
        ids.back() = f.ids[(rows - 1) % capacity];
        f.reset(s); // All borrowed views above expire here.
        const auto replay = s.step_batch(ids); f.comparison.compare(replay, reference); exact(replay, saved_logits);
        read_tap(s.target_tap(), observed); exact(observed, saved_tap); steady(s.memory(), loaded);
        ++evidence.steady_memory_observations;
        f.tap_exact_values += 2ULL * rows * width;
        evidence.compared_full_vocab_values = f.comparison.values - values_before;
        evidence.tap_exact_values = f.tap_exact_values - taps_before;
        record_begin("session_restore_wide_PP");
        std::cout << ",\"owner\":"; json_string(std::cout, rows == 1024 ? "wide_PP_1024" : "wide_PP_32");
        std::cout << ",\"rows\":" << rows << ",\"public_snapshots\":{\"preverify\":"; write_public(std::cout, preverify);
        std::cout << ",\"postverify\":"; write_public(std::cout, postverify);
        std::cout << ",\"postrestore\":"; write_public(std::cout, postrestore);
        std::cout << ",\"pre_late_ID_reject\":"; write_public(std::cout, prereject);
        std::cout << ",\"post_late_ID_reject\":"; write_public(std::cout, postreject);
        std::cout << ",\"post_reset_replay\":"; write_public(std::cout, public_snapshot(s));
        std::cout << "},\"phase_totals\":"; write_phase(std::cout, evidence);
        std::cout << ",\"hidden_N1_numerical_comparison\":{\"values\":" << hidden.values
            << ",\"max_error\":" << hidden.max_error << ",\"max_bound_ratio\":" << hidden.max_bound_ratio
            << "},\"all_rows_D2H_before_reuse\":true,\"full_vocabulary_N1_reference\":true,\"tap_replay_exact\":true,\"passed\":true}\n";
    }
    return evidence;
}

struct CpuPublished {
    std::vector<float> logits = std::vector<float>(batch * vocabulary);
    std::vector<float> tap = std::vector<float>(batch * width), observed = std::vector<float>(batch * width);
    std::vector<float> down = std::vector<float>(batch * 48 * 10 * 2560), ffn = std::vector<float>(batch * 48 * 2560);
    std::vector<qwen::Q8_1> original = std::vector<qwen::Q8_1>(batch * 48 * 80);
    std::vector<std::int32_t> ids = std::vector<std::int32_t>(batch * 48 * 10);
    std::vector<float> weights = std::vector<float>(batch * 48 * 10);
    std::vector<std::uint8_t> available = std::vector<std::uint8_t>(batch * 48), cpu = std::vector<std::uint8_t>(batch * 48 * 10);
    template<class T> static void copy(std::span<const T> source, std::vector<T>& target) {
        require(source.size() == target.size(), "CPU published probe shape");
        std::copy(source.begin(), source.end(), target.begin());
    }
    template<class T> static void check(std::span<const T> source, const std::vector<T>& target) {
        require(source.size() == target.size() && std::memcmp(source.data(), target.data(), source.size_bytes()) == 0,
            "CPU published probe bits changed");
    }
    void save(qwen::Session& s, std::span<const float> output) {
        copy(output, logits); read_tap(s.target_tap(), tap);
        const auto p = s.hybrid_intermediates(); copy(p.routed_down, down); copy(p.ffn_output, ffn);
        const auto i = s.hybrid_inputs();
        // Initial hybrid-disabled publication has unavailable original bytes;
        // do NOT copy/read those unspecified rows, only route/flag payloads.
        require(i.original_q8.size() == original.size(), "CPU original probe extent");
        copy(i.expert_ids, ids); copy(i.route_weights, weights); copy(i.input_available, available); copy(i.cpu_assignment, cpu);
        for (std::size_t row = 0; row < available.size(); ++row) if (available[row])
            std::copy_n(i.original_q8.begin() + row * 80, 80, original.begin() + row * 80);
    }
    void check(qwen::Session& s, std::span<const float> output, qwen::SessionTargetTap view) {
        exact(output, logits); require(s.target_tap() == view, "failed forward changed tap metadata");
        read_tap(view, observed); exact(observed, tap); // Physical OLD pointer, before reset.
        const auto p = s.hybrid_intermediates(); check(p.routed_down, down); check(p.ffn_output, ffn);
        const auto i = s.hybrid_inputs();
        require(i.original_q8.size() == original.size(), "failed original probe extent");
        check(i.expert_ids, ids); check(i.route_weights, weights); check(i.input_available, available); check(i.cpu_assignment, cpu);
        for (std::size_t row = 0; row < available.size(); ++row) if (available[row])
            require(std::memcmp(i.original_q8.data() + row * 80, original.data() + row * 80, 80 * sizeof(qwen::Q8_1)) == 0,
                "available original CPU probe bits changed");
    }
};
void write_fault(std::ostream& out, const qwen::SessionHybridProbeDiagnostics& d) {
    out << "{\"accepted_cpu_jobs\":" << d.accepted_cpu_jobs << ",\"queued_admission_copies\":" << d.queued_admission_copies
        << ",\"pending_slots_at_failure\":" << d.pending_slots_at_failure
        << ",\"cpu_return_bytes_before_failure\":" << d.cpu_return_bytes_before_failure
        << ",\"ready_cache_ids\":" << d.ready_cache_ids << ",\"pending_cache_ids\":" << d.pending_cache_ids
        << ",\"failure_layer\":" << d.failure_layer << ",\"synthetic_failure\":" << (d.synthetic_failure ? "true" : "false")
        << ",\"cpu_pool_drained\":" << (d.cpu_pool_drained ? "true" : "false")
        << ",\"gpu_streams_drained\":" << (d.gpu_streams_drained ? "true" : "false") << '}';
}
PhaseEvidence cpu_failure_recovery(const char* path, Fixture& f) {
    CpuPublished saved; // All host storage before Session/calls.
    std::vector<float> reference(6ULL * vocabulary);
    qwen::SessionConfig config; config.capacity = capacity; config.expert_slots = 1; config.max_batch_tokens = batch;
    config.speculative_checkpoints = true; config.cpu_workers = 1; config.hybrid_probe = true;
    PhaseEvidence evidence;
    const auto values_before = f.comparison.values, taps_before = f.tap_exact_values;
    {
        qwen::Session s(path, config); const auto loaded = s.memory(); geometry(loaded, batch, capacity);
        loaded_memory("CPU_failure_recovery", 2, config, loaded, public_snapshot(s));
        require(loaded.cpu_workers == 1 && loaded.cpu_pool_scratch && loaded.pinned_hybrid_middle &&
            s.hybrid_policy().mode == qwen::SessionHybridMode::disabled, "CPU initial disabled/resources");
        for (int t = 0; t < 6; ++t) {
            const auto output = s.step(f.ids[t]);
            for (float v : output) require(std::isfinite(v), "CPU resource N1 reference finite");
            std::copy(output.begin(), output.end(), reference.begin() + t * vocabulary);
        }
        f.reset(s);
        const auto output = s.step_batch(std::span(f.ids).first(batch));
        f.comparison.compare(output, std::span(reference).first(batch * vocabulary)); saved.save(s, output);
        const auto tap = s.target_tap(); const auto stats = s.stats(); const auto spec = s.speculative_stats();
        const auto route = s.route_stats(); const auto attention = s.attention_stats(); const auto hybrid = s.hybrid_stats();
        s.set_hybrid_policy({qwen::SessionHybridMode::mixed, 2, true});
        saved.check(s, output, tap);
        const auto prefault = public_snapshot(s);
        bool failed = false;
        try { (void)s.verify_window(std::span(f.ids).subspan(3, batch)); }
        catch (const std::runtime_error& e) {
            require(std::string_view(e.what()) == "synthetic hybrid failure after CPU acceptance and GPU admission enqueue",
                "unexpected failure instead of post-gate/up admission injection");
            failed = true;
        }
        require(failed, "post-gate/up injection not reached");
        auto preserved = [&] {
            require(same_stats(s.stats(), stats) && s.speculative_stats() == spec && word_equal(s.route_stats(), route) &&
                word_equal(s.attention_stats(), attention) && word_equal(s.hybrid_stats(), hybrid), "failure published counters changed");
            saved.check(s, output, tap);
        };
        preserved();
        const auto postfault = public_snapshot(s, false); // Window getter intentionally rejects while invalidated.
        const auto fault = s.hybrid_probe_diagnostics();
        require(fault.synthetic_failure && fault.failure_layer == 0 && fault.accepted_cpu_jobs > 0 &&
            fault.queued_admission_copies > 0 && fault.pending_slots_at_failure > 0 && fault.cpu_return_bytes_before_failure == 0 &&
            fault.cpu_pool_drained && fault.gpu_streams_drained && fault.ready_cache_ids == 0 && fault.pending_cache_ids == 0,
            "submitted failure pool/streams/cache drain");
        auto sticky = [&](auto&& call) {
            bool rejected = false; try { call(); } catch (const std::runtime_error&) { rejected = true; }
            require(rejected, "execution failure bypassed required reset"); preserved(); ++evidence.sticky_rejections;
        };
        sticky([&] { (void)s.step(f.ids[3]); });
        sticky([&] { (void)s.verify_window(std::span(f.ids).subspan(3, batch)); });
        sticky([&] { s.restore_prefix(0); });
        sticky([&] { (void)s.checkpoint_state(); });
        sticky([&] { s.set_hybrid_policy({qwen::SessionHybridMode::disabled, 2}); });
        sticky([&] { s.set_attention_batch(true); });
        steady(s.memory(), loaded); preserved(); ++evidence.steady_memory_observations;
        const auto poststicky = public_snapshot(s, false);
        f.reset(s); // Consumed one-shot retained; old borrowed views expire here.
        require(s.hybrid_probe_diagnostics().synthetic_failure && s.hybrid_probe_diagnostics().accepted_cpu_jobs == fault.accepted_cpu_jobs &&
            s.hybrid_probe_diagnostics().queued_admission_copies == fault.queued_admission_copies &&
            s.hybrid_intermediates().routed_down.empty() && s.hybrid_inputs().expert_ids.empty(), "reset recovery evidence/extents");
        const auto postreset = public_snapshot(s);
        s.set_hybrid_policy({qwen::SessionHybridMode::mixed, 2, false});
        for (int t = 0; t < 6; ++t) {
            const auto recovered = s.step(f.ids[t]);
            f.comparison.compare(recovered, std::span(reference).subspan(static_cast<std::size_t>(t) * vocabulary, vocabulary));
            require(s.target_tap().rows == 1 && s.target_tap().first_position == static_cast<std::uint64_t>(t), "recovered tap rows");
            read_tap(s.target_tap(), std::span(saved.observed).first(width));
        }
        require(s.hybrid_stats().cpu_gate_up_jobs > 0 && s.hybrid_stats().cpu_down_jobs > 0 &&
            s.hybrid_stats().gpu_middle_columns > 0 && s.speculative_stats().target_forward_rows == 6,
            "recovered CPUlinear_GPUmiddle actual completed work");
        steady(s.memory(), loaded); ++evidence.steady_memory_observations;
        require(evidence.sticky_rejections == 6, "CPU sticky rejection count");
        evidence.compared_full_vocab_values = f.comparison.values - values_before;
        evidence.tap_exact_values = f.tap_exact_values - taps_before;
        record_begin("session_restore_CPU_failure_recovery");
        std::cout << ",\"owner\":\"CPU_failure_recovery\",\"cpu_workers\":1,\"initial_policy\":\"disabled\""
            << ",\"fault\":\"after_gate_up_submission_and_GPU_admission_before_join_middle_down\",\"fault_diagnostics\":";
        write_fault(std::cout, fault);
        std::cout << ",\"public_snapshots\":{\"prefault\":"; write_public(std::cout, prefault);
        std::cout << ",\"postfault\":"; write_public(std::cout, postfault);
        std::cout << ",\"poststicky\":"; write_public(std::cout, poststicky);
        std::cout << ",\"postreset\":"; write_public(std::cout, postreset);
        std::cout << ",\"postrecovery\":"; write_public(std::cout, public_snapshot(s));
        std::cout << "},\"phase_totals\":"; write_phase(std::cout, evidence);
        std::cout << ",\"old_tap_physical_bytes_preserved\":true,\"sticky_rejections\":" << evidence.sticky_rejections
            << ",\"recovery_full_vocab_rows\":6,\"inflight_timing_claim\":false,\"passed\":true}\n";
    }
    return evidence;
}
void write_expert_inventory(std::ostream& out, const qwen::Model& model) {
    // Tensor inventory only: these descriptors own no weight payload.
    std::uint64_t matrices = 0, bytes = 0, q4_0_down_layers = 0, q4_1_down_layers = 0;
    for (int l = 0; l < 48; ++l) {
        const auto prefix = "blk." + std::to_string(l) + '.';
        for (auto suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}) {
            const auto& tensor = model.tensor(prefix + suffix);
            ++matrices; bytes += tensor.byte_size;
            if (std::string_view(suffix) == "ffn_down_exps.weight") {
                q4_0_down_layers += tensor.type == qwen::TensorType::Q4_0;
                q4_1_down_layers += tensor.type == qwen::TensorType::Q4_1;
            }
        }
    }
    out << "{\"scope\":\"tensor_inventory_no_payload_reads\",\"matrices\":" << matrices << ",\"bytes\":" << bytes
        << ",\"q4_0_down_layers\":" << q4_0_down_layers << ",\"q4_1_down_layers\":" << q4_1_down_layers << '}';
}
}

int main(int argc, char** argv) {
    const bool maximum_wide = argc == 3 && std::string_view(argv[2]) == "--wide-1024";
    if (argc != 2 && !maximum_wide) { std::cerr << "usage: core-session-restore-test MODEL.gguf [--wide-1024]\n"; return 2; }
    try {
        for (char c : std::string_view(CORE_REVISION))
            require((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'), "revision hex");
        Fixture f;
        qwen::Model metadata(argv[1]); // Inventory only; no second weight load.
        const auto hash = qwen::ple_hash_config(metadata, 1);
        require(hash.eos_token_id >= 0 && hash.eos_token_id < vocabulary, "actual PLE EOS ID");
        for (int i = 0; i < capacity; ++i) f.ids[i] = 100 + i;
        for (int i : {0, 4, 8, 10, 37}) f.ids[i] = static_cast<std::int32_t>(hash.eos_token_id);
        qwen::SessionConfig config; config.capacity = capacity; config.expert_slots = 1; config.max_batch_tokens = batch;
        config.speculative_checkpoints = true; // CPU workers=0 isolates restoration.
        std::array<RejectionEvidence, 3> extra_rejections{};
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
        record_begin("session_restore_source");
        std::cout << ",\"revision\":\"" << CORE_REVISION
            << "\",\"dirty\":" << (CORE_DIRTY ? "true" : "false") << ",\"capacity\":40,\"slots\":1,\"max_batch\":3"
            << ",\"ple_eos\":" << hash.eos_token_id << ",\"cpu_workers\":0,\"absolute_gate\":0.02,\"relative_gate\":0.002"
            << ",\"speculative_checkpoints\":true,\"model_path\":";
        json_string(std::cout, std::filesystem::absolute(argv[1]).lexically_normal().string());
        std::cout << ",\"model_argument\":"; json_string(std::cout, argv[1]);
        std::cout << ",\"expert_inventory\":"; write_expert_inventory(std::cout, metadata);
        std::cout << ",\"runtime_geometry\":{\"vocabulary\":" << vocabulary << ",\"layers\":48,\"gdn_layers\":36,\"qsa_layers\":12"
            << ",\"kv_type\":\"Q4_0\",\"kv_width\":512,\"kv_blocks_per_position\":16,\"kv_block_elements\":32,\"kv_block_bytes\":" << sizeof(qwen::Q4_0)
            << ",\"pooled_index_type\":\"F32\",\"pooled_index_width\":128,\"qsa_block_tokens\":4,\"qsa_raw_tail_elements\":384"
            << ",\"gdn_recurrent_elements\":786432,\"gdn_history_elements\":30720,\"ple_conv_history_elements\":92160"
            << ",\"checkpoint_slots\":4,\"max_verify_inputs\":3,\"tap_width\":" << width << ",\"tap_device\":1"
            << ",\"tap_location\":\"layer47_combine_before_root_HC\"},\"source_required\":{\"records\":56,\"loaded_memory_owners\":3"
            << ",\"primary\":{\"successful_restores\":50,\"argument_rejections\":83,\"steady_memory_observations\":133}"
            << ",\"CPU_sticky_rejections\":6,\"expert_payload_reads_per_owner\":" << expert_payload_reads
            << ",\"expert_payload_bytes_per_owner\":" << expert_payload_bytes
            << ",\"expert_geometry\":{\"layers\":48,\"experts_per_layer\":512,\"top_k\":10,\"input\":2560,\"middle\":640"
            << ",\"gate_up_type\":\"Q4_0\",\"q4_0_down_layers\":42,\"q4_1_down_layers\":6"
            << ",\"q4_0_triplet_bytes_per_expert\":2764800,\"q4_1_triplet_bytes_per_expert\":2867200}}"
            << ",\"wide_PP_mode\":\"" << (maximum_wide ? "explicit_1024" : "default_32") << "\",\"wide_PP_rows\":" << (maximum_wide ? 1024 : 32)
            << ",\"reference\":\"retained_N1_full_vocabulary_same_Session\",\"performance_claim\":false}\n";
        {
            qwen::Session s(argv[1], config);
            const auto loaded = s.memory(); geometry(loaded, batch, capacity);
            loaded_memory("primary_restoration", 0, config, loaded, public_snapshot(s));
            // Full N1 reference all 40 rows. Each GPU tap copied BEFORE reuse.
            for (int i = 0; i < capacity; ++i) {
                const auto output = s.step(f.ids[i]);
                for (float v : output) require(std::isfinite(v), "reference logits finite");
                std::copy(output.begin(), output.end(), f.logits.begin() + i * vocabulary);
                f.tap(s, i, 1, true);
            }
            // An actually DIFFERENT rejected suffix. The same Session supplies
            // all four branch oracles: k alternate IDs, then 3 original IDs.
            constexpr std::array<std::int32_t, 3> alternate_ids{150, 151, 152};
            f.reset(s); f.prefix(s, 3);
            for (int t = 0; t < 3; ++t) {
                const auto result = s.step(alternate_ids[t]);
                for (float v : result) require(std::isfinite(v), "alternate reference finite");
                std::copy(result.begin(), result.end(), f.alternate_logits.begin() + t * vocabulary);
                read_tap(s.target_tap(), std::span(f.alternate_taps).subspan(static_cast<std::size_t>(t) * width, width));
            }
            for (int k = 0; k <= 3; ++k) {
                f.reset(s); f.prefix(s, 3);
                for (int t = 0; t < k; ++t) {
                    const auto result = s.step(alternate_ids[t]);
                    f.comparison.compare(result, std::span(f.alternate_logits).subspan(static_cast<std::size_t>(t) * vocabulary, vocabulary));
                }
                for (int t = 0; t < 3; ++t) {
                    const auto result = s.step(f.ids[3 + k + t]);
                    for (float v : result) require(std::isfinite(v), "branch reference finite");
                    const auto row = static_cast<std::size_t>(k * 3 + t);
                    std::copy(result.begin(), result.end(), f.branch_logits.begin() + row * vocabulary);
                    read_tap(s.target_tap(), std::span(f.branch_taps).subspan(row * width, width));
                }
            }
            // All starting mod4 phases, plus EOS-crossing occupied windows.
            // EACH N/EVERY k recreates its one-shot window, then continues N1.
            for (int start : {2, 3, 4, 5, 8}) for (int n = 1; n <= 3; ++n) for (int k = 0; k <= n; ++k) {
                f.reset(s);
                const bool attention_case = start == 5 && n == 3 && k == 1;
                if (attention_case) {
                    s.set_attention_batch(true); f.reset(s); // Policy must survive reset.
                    require(s.attention_batch(), "enabled attention policy reset retention");
                }
                f.prefix(s, start);
                const auto preverify = public_snapshot(s);
                const auto before = s.speculative_stats();
                const auto output = s.verify_window(std::span(f.ids).subspan(start, n));
                f.comparison.compare(output, f.logit_reference(start, n)); f.tap(s, start, n);
                require(s.checkpoint_state() == qwen::SessionCheckpointState{true, true, static_cast<std::uint64_t>(start), n, n + 1},
                    "verify chronological publication");
                const auto verified = s.speculative_stats();
                require(verified.target_forward_rows == before.target_forward_rows + n && verified.verify_windows == before.verify_windows + 1 &&
                    verified.verify_rows == before.verify_rows + n && verified.gdn_prefix_calls == before.gdn_prefix_calls + 36 &&
                    verified.ple_prefix_calls == before.ple_prefix_calls + 1 && verified.qsa_tail_prefix_calls == before.qsa_tail_prefix_calls + 12,
                    "scoped prefix API counts (not physical launch observations)");
                const auto postverify = public_snapshot(s);
                const bool pending_rejects = start == 3 && k == 0;
                const auto rejects_before_pending = f.rejects;
                if (pending_rejects) f.invalid_pending(s, output, loaded, n);
                const auto post_pending_rejects = public_snapshot(s);
                const auto pending_reject_count = f.rejects - rejects_before_pending;
                std::copy(output.begin(), output.end(), f.saved_logits.begin());
                const auto tap_before = s.target_tap();
                auto tap_saved = std::span(f.saved_tap).first(static_cast<std::size_t>(n) * width); read_tap(tap_before, tap_saved);
                const auto stats = s.stats(); const auto routes = s.route_stats();
                const auto attention = s.attention_stats(); const auto hybrid = s.hybrid_stats();
                if (attention_case) require(attention.batch_calls > 0 && attention.query_rows == 12 * 3 &&
                    attention.singleton_tail_calls > 0, "nonzero physical attention diagnostics before rollback");
                s.restore_prefix(k); ++f.restores;
                require(s.stats().consumed_tokens == static_cast<std::uint64_t>(start + k) && same_stats(s.stats(), stats, false) &&
                    word_equal(s.route_stats(), routes) && word_equal(s.attention_stats(), attention) && word_equal(s.hybrid_stats(), hybrid),
                    "restore rewound physical statistics");
                auto expected = verified; ++expected.restore_calls; ++expected.restored_prefixes[k]; expected.retained_inputs += k;
                require(s.speculative_stats() == expected && !s.checkpoint_state().pending && s.target_tap() == tap_before,
                    "restore diagnostics/tap ownership");
                exact(output, std::span(f.saved_logits).first(output.size()));
                auto tap_after = std::span(f.readback).first(tap_saved.size()); read_tap(s.target_tap(), tap_after); exact(tap_after, tap_saved);
                // k==0 reads NO carry; otherwise row k-1 is this actual output.
                if (k) exact(tap_after.subspan(static_cast<std::size_t>(k - 1) * width, width),
                    std::span(f.taps).subspan(static_cast<std::size_t>(start + k - 1) * width, width));
                const auto postrestore = public_snapshot(s);
                const auto rejects_before_consumed = f.rejects;
                f.reject(s, output, loaded, [&] { s.restore_prefix(k); });
                const auto post_consumed_reject = public_snapshot(s);
                const auto consumed_reject_count = f.rejects - rejects_before_consumed;
                f.memory(s, loaded);
                // Do not inspect output/tap_before after this successful forward.
                for (int t = 0; t < 3; ++t) {
                    const int position = start + k + t;
                    const auto next = s.step(f.ids[position]); f.comparison.compare(next, f.logit_reference(position, 1)); f.tap(s, position, 1);
                    require(s.speculative_stats().target_forward_rows == verified.target_forward_rows + t + 1,
                        "logical rollback decreased monotonic forward rows");
                }
                if (attention_case)
                    require(word_equal(s.attention_stats(), attention), "N1 continuation changed physical batch diagnostics");
                record_begin("session_restore_case");
                std::cout << ",\"owner\":\"primary_restoration\",\"start\":" << start << ",\"inputs\":" << n
                    << ",\"retained\":" << k << ",\"restored_mod4\":" << (start + k) % 4
                    << ",\"restored_blocks\":" << (start + k) / 4 << ",\"attention_batch_enabled\":"
                    << (attention_case ? "true" : "false") << ",\"continuation_rows\":3,\"public_snapshots\":{\"preverify\":";
                write_public(std::cout, preverify);
                std::cout << ",\"postverify\":"; write_public(std::cout, postverify);
                std::cout << ",\"postrestore\":"; write_public(std::cout, postrestore);
                std::cout << ",\"postcontinuation\":"; write_public(std::cout, public_snapshot(s));
                std::cout << "},\"argument_rejections\":[";
                if (pending_rejects) {
                    write_rejection(std::cout, {"invalid_pending_arguments", pending_reject_count, postverify, post_pending_rejects});
                    std::cout << ',';
                }
                write_rejection(std::cout, {"consumed_window_restore", consumed_reject_count, postrestore, post_consumed_reject});
                std::cout << "],\"published_full_span_and_tap_bits_preserved\":true,\"passed\":true}\n";
                if (attention_case) s.set_attention_batch(false);
            }
            f.reset(s); f.prefix(s, 3);
            (void)s.verify_window(std::span(f.ids).subspan(3, 3));
            const auto ordinary = s.step_batch(std::span(f.ids).subspan(6, 3));
            f.comparison.compare(ordinary, f.logit_reference(6, 3)); f.tap(s, 6, 3);
            require(!s.checkpoint_state().pending && s.target_tap().rows == 3, "ordinary PP pending invalidation");
            auto rejects_before = f.rejects;
            extra_rejections[0] = {"ordinary_PP_invalidated_pending", 0, public_snapshot(s), {}};
            f.reject(s, ordinary, loaded, [&] { s.restore_prefix(0); });
            extra_rejections[0].count = f.rejects - rejects_before; extra_rejections[0].after = public_snapshot(s);
            for (int k = 0; k <= 3; ++k) {
                f.reset(s); f.prefix(s, 3);
                const auto preverify = public_snapshot(s);
                const auto verified = s.verify_window(alternate_ids);
                f.comparison.compare(verified, f.alternate_logits);
                read_tap(s.target_tap(), std::span(f.readback)); exact(f.readback, f.alternate_taps);
                f.tap_exact_values += f.readback.size();
                const auto postverify = public_snapshot(s);
                s.restore_prefix(k); ++f.restores;
                const auto postrestore = public_snapshot(s);
                for (int t = 0; t < 3; ++t) {
                    const auto next = s.step(f.ids[3 + k + t]); const auto row = static_cast<std::size_t>(k * 3 + t);
                    f.comparison.compare(next, std::span(f.branch_logits).subspan(row * vocabulary, vocabulary));
                    auto observed = std::span(f.readback).first(width); read_tap(s.target_tap(), observed);
                    exact(observed, std::span(f.branch_taps).subspan(row * width, width)); f.tap_exact_values += width;
                }
                f.memory(s, loaded);
                record_begin("session_restore_divergent_suffix");
                std::cout << ",\"owner\":\"primary_restoration\",\"start\":3,\"inputs\":3,\"retained\":" << k
                    << ",\"continuation_rows\":3,\"public_snapshots\":{\"preverify\":"; write_public(std::cout, preverify);
                std::cout << ",\"postverify\":"; write_public(std::cout, postverify);
                std::cout << ",\"postrestore\":"; write_public(std::cout, postrestore);
                std::cout << ",\"postcontinuation\":"; write_public(std::cout, public_snapshot(s));
                std::cout << "},\"passed\":true}\n";
            }
            f.reset(s); f.prefix(s, 39);
            const auto full = s.verify_window(std::span(f.ids).last(1));
            f.comparison.compare(full, f.logit_reference(39, 1)); f.tap(s, 39, 1);
            rejects_before = f.rejects;
            extra_rejections[1] = {"capacity_and_prefix_arguments", 0, public_snapshot(s), {}};
            f.reject(s, full, loaded, [&] { (void)s.step(f.ids[0]); });
            f.reject(s, full, loaded, [&] { (void)s.verify_window(std::span(f.ids).first(1)); });
            f.reject(s, full, loaded, [&] { s.restore_prefix(2); });
            extra_rejections[1].count = f.rejects - rejects_before; extra_rejections[1].after = public_snapshot(s);
            s.restore_prefix(0); ++f.restores;
            const auto reuse = s.verify_window(std::span(f.ids).last(1));
            f.comparison.compare(reuse, f.logit_reference(39, 1)); f.tap(s, 39, 1);
            f.reset(s); f.reset(s); f.memory(s, loaded);
            const auto replay = s.step(f.ids[0]); f.comparison.compare(replay, f.logit_reference(0, 1)); f.tap(s, 0, 1);
            rejects_before = f.rejects;
            extra_rejections[2] = {"restore_after_reset_without_window", 0, public_snapshot(s), {}};
            f.reject(s, replay, loaded, [&] { s.restore_prefix(0); });
            extra_rejections[2].count = f.rejects - rejects_before; extra_rejections[2].after = public_snapshot(s);
        }
        require(f.restores == 50 && f.rejects == 83 && f.observations == 133, "restoration fixture branch/count coverage");
        const PhaseEvidence primary{f.comparison.values, f.tap_exact_values, f.restores, f.rejects, f.observations, 0};
        const auto wide = wide_pp(argv[1], f, maximum_wide ? 1024 : 32);
        const auto cpu = cpu_failure_recovery(argv[1], f);
        // Every Session scope has completed RAII before this completion record.
        // Owned geometry is audited, but returned free VRAM is not measured.
        require(records_emitted == 55, "protocol record coverage before completion");
        record_begin("session_restore_complete");
        std::cout << ",\"restores\":" << f.restores << ",\"rejections\":" << f.rejects
            << ",\"compared_full_vocab_values\":" << f.comparison.values << ",\"diagnostic_bit_differences\":" << f.comparison.bits_different
            << ",\"max_error\":" << f.comparison.max_error << ",\"max_bound_ratio\":" << f.comparison.max_bound_ratio
            << ",\"tap_exact_values\":" << f.tap_exact_values << ",\"memory_observations\":" << f.observations
            << ",\"counts_scope\":\"primary_restoration_fixture\",\"wide_PP_rows\":" << (maximum_wide ? 1024 : 32)
            << ",\"comparison_counts_scope\":\"all_three_owners\",\"compared_full_vocab_rows\":" << f.comparison.values / vocabulary
            << ",\"phase_totals\":{\"primary_restoration\":"; write_phase(std::cout, primary);
        std::cout << ",\"wide_PP\":"; write_phase(std::cout, wide);
        std::cout << ",\"CPU_failure_recovery\":"; write_phase(std::cout, cpu);
        std::cout << "},\"primary_extra_argument_rejections\":[";
        for (std::size_t i = 0; i < extra_rejections.size(); ++i) { if (i) std::cout << ','; write_rejection(std::cout, extra_rejections[i]); }
        std::cout << "],\"aggregate_counts\":{\"successful_restores\":" << primary.successful_restores + wide.successful_restores + cpu.successful_restores
            << ",\"argument_rejections\":" << primary.argument_rejections + wide.argument_rejections + cpu.argument_rejections
            << ",\"sticky_rejections\":" << primary.sticky_rejections + wide.sticky_rejections + cpu.sticky_rejections
            << ",\"steady_memory_observations\":" << primary.steady_memory_observations + wide.steady_memory_observations + cpu.steady_memory_observations
            << "},\"records\":" << records_emitted << ",\"loaded_memory_records\":3"
            << ",\"snapshot_counters_scope\":\"completed_public_work_since_last_reset_not_kernel_counts\""
            << ",\"tap_exact_count_scope\":\"numerical_reference_and_replay_comparisons_excludes_rejection_and_failure_preservation_checks\""
            << ",\"CPU_failure_recovery_passed\":true,\"session_instances\":3,\"raii_cleanup_completed\":true,\"owned_buffer_release_measured\":false"
            << ",\"trained_MTP_qualified\":false,\"HF_qualified\":false,\"R6_complete_claim\":false,\"performance_claim\":false,\"passed\":true}\n";
        require(static_cast<bool>(std::cout), "stdout failure");
        return 0;
    } catch (const std::exception& e) { std::cerr << "core-session-restore-test: " << e.what() << '\n'; return 1; }
}
