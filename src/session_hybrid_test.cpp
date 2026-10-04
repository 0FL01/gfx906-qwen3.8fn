#include "session.hpp"
#include "cpu_expert.hpp"
#include "routes.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <system_error>
#include <unistd.h>
#include <vector>

#ifndef CORE_REVISION
#error "core-session-hybrid-test requires compiled CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-session-hybrid-test requires compiled CORE_DIRTY"
#endif

// Correctness only. Parent wires qwen-session + CpuExpertPool with strict FP and
// runs the independent actual-GPU CPU-expert shadow first. No timing/speed gate.
namespace {
using namespace qwen;
constexpr std::size_t vocab = 248320, hidden = 2560, layers = 48;
constexpr std::size_t down_row = layers * 10 * hidden, ffn_row = layers * hidden;
constexpr int rows = 40, teacher_rows = 32, max_batch = 3;
constexpr std::uint64_t q40_bytes = 2764800, q41_bytes = 2867200;
constexpr std::uint64_t input_probe_bytes = 2 * 3 * 48 * (80 * sizeof(Q8_1) + 10 * (sizeof(std::int32_t) + sizeof(float) + 1) + 1);
constexpr auto ids = [] {
    std::array<std::int32_t, rows> result{};
    result[0] = 248044;
    for (std::size_t i = 1; i < result.size(); ++i) result[i] = 99 + static_cast<std::int32_t>(i);
    return result;
}();
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little && input_probe_bytes == 855648);
static_assert(std::has_unique_object_representations_v<SessionHybridStats>);

void require(bool value, std::string_view label) {
    if (!value) throw std::runtime_error(std::string(label));
}
bool same(SessionStats a, SessionStats b) {
    return a.consumed_tokens == b.consumed_tokens && a.expert_hits == b.expert_hits &&
        a.expert_misses == b.expert_misses && a.expert_upload_bytes == b.expert_upload_bytes &&
        std::bit_cast<std::uint64_t>(a.last_completed_ms) == std::bit_cast<std::uint64_t>(b.last_completed_ms);
}
bool same(SessionRouteStats a, SessionRouteStats b) {
    return a.last_max_expert_group_assignments == b.last_max_expert_group_assignments &&
        a.expert_groups_gt128 == b.expert_groups_gt128;
}
bool same(SessionHybridStats a, SessionHybridStats b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}
bool same(SessionHybridProbeDiagnostics a, SessionHybridProbeDiagnostics b) {
    return a.accepted_cpu_jobs == b.accepted_cpu_jobs && a.queued_admission_copies == b.queued_admission_copies &&
        a.pending_slots_at_failure == b.pending_slots_at_failure && a.cpu_return_bytes_before_failure == b.cpu_return_bytes_before_failure &&
        a.ready_cache_ids == b.ready_cache_ids && a.pending_cache_ids == b.pending_cache_ids && a.failure_layer == b.failure_layer &&
        a.synthetic_failure == b.synthetic_failure && a.cpu_pool_drained == b.cpu_pool_drained && a.gpu_streams_drained == b.gpu_streams_drained;
}
bool exact(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size_bytes()) == 0);
}
struct Metrics {
    std::uint64_t values = 0, violations = 0, bit_mismatches = 0;
    double max_abs = 0, max_bound_ratio = 0;
    std::uint64_t nonfinite = 0;
    std::size_t first_bit = std::numeric_limits<std::size_t>::max(), first_gate = std::numeric_limits<std::size_t>::max();
};
struct CheckedCounts {
    std::uint64_t windows = 0, rows = 0, short_rows = 0, wide_windows = 0, wide_rows = 0;
} checked;
Metrics compare(std::span<const float> actual, std::span<const float> reference, double absolute) {
    require(actual.size() == reference.size() && !actual.empty(), "comparison extent");
    Metrics result;
    result.values = actual.size();
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const bool bits = std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(reference[i]);
        if (bits && result.bit_mismatches++ == 0) result.first_bit = i;
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) {
            ++result.nonfinite;
            if (result.violations++ == 0) result.first_gate = i;
            continue;
        }
        const double delta = std::fabs(static_cast<double>(actual[i]) - reference[i]);
        const double bound = absolute + .002 * std::fabs(static_cast<double>(reference[i]));
        result.max_abs = std::max(result.max_abs, delta);
        result.max_bound_ratio = std::max(result.max_bound_ratio, delta / bound);
        if (delta > bound && result.violations++ == 0) result.first_gate = i;
    }
    return result;
}
void emit(const Metrics& m) {
    std::cout << "{\"values\":" << m.values << ",\"violations\":" << m.violations
        << ",\"max_abs\":" << m.max_abs << ",\"max_bound_ratio\":" << m.max_bound_ratio
        << ",\"diagnostic_bit_mismatches\":" << m.bit_mismatches << '}';
}
void provenance() {
    const std::string_view revision = CORE_REVISION;
    require(revision.size() == 40, "compiled revision extent");
    for (std::size_t i = 0; i < revision.size(); i += 8) {
        std::uint32_t v = 0;
        const auto parsed = std::from_chars(revision.data() + i, revision.data() + i + 8, v, 16);
        require(parsed.ec == std::errc{} && parsed.ptr == revision.data() + i + 8, "compiled revision format");
    }
}
struct WitnessConfig {
    std::filesystem::path directory;
    std::string model;
    bool written = false;
} witness;
std::string json_quote(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c < 32 || c >= 127) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += char(c);
    }
    return result + '"';
}
void exclusive_write(const char* name, std::span<const std::byte> bytes) {
    const auto path = witness.directory / name;
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) throw std::system_error(errno, std::generic_category(), "create witness file");
    struct Close { int fd; ~Close() { if (fd >= 0) ::close(fd); } } close{fd};
    std::size_t done = 0;
    while (done < bytes.size()) {
        const auto count = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (count < 0 && errno == EINTR) continue;
        if (count < 0) throw std::system_error(errno, std::generic_category(), "write witness file");
        require(count > 0, "zero witness write"); done += static_cast<std::size_t>(count);
    }
    close.fd = -1;
    if (::close(fd) != 0) throw std::system_error(errno, std::generic_category(), "close witness file");
}
void coordinate(std::ostream& out, std::size_t index, std::size_t row_width, const char* kind) {
    if (index == std::numeric_limits<std::size_t>::max()) { out << "null"; return; }
    const auto token = index / row_width, row = index % row_width;
    out << "{\"flat\":" << index << ",\"window_row\":" << token;
    if (row_width == down_row) out << ",\"layer\":" << row / (10 * hidden) << ",\"rank\":" << (row / hidden) % 10 << ",\"element\":" << row % hidden;
    else if (row_width == ffn_row) out << ",\"layer\":" << row / hidden << ",\"element\":" << row % hidden;
    else out << ",\"vocab_id\":" << row;
    out << ",\"tensor\":" << json_quote(kind) << '}';
}
void dump_witness(Session& s, std::span<const float> reference_down, int offset, int n, std::string_view phase,
                  const Metrics& logits, const Metrics& routed, const Metrics& ffn,
                  std::span<const std::int32_t> reference_ids, std::span<const float> reference_weights) {
    if (witness.directory.empty() || witness.written) return;
    witness.written = true; // FIRST numerical comparison failure only, before any further accepted call.
    require(offset >= 0 && n >= 1 && offset <= rows - n && reference_down.size() == static_cast<std::size_t>(n) * down_row &&
        reference_ids.size() == static_cast<std::size_t>(n) * layers * 10 && reference_weights.size() == reference_ids.size(),
        "witness retained reference/window ranges");
    const auto p = s.hybrid_intermediates(); const auto input = s.hybrid_inputs();
    const auto index = routed.first_bit;
    const bool coordinate_valid = n <= 3 && index < static_cast<std::size_t>(n) * down_row;
    const auto token = coordinate_valid ? index / down_row : 0;
    const auto layer = coordinate_valid ? (index % down_row) / (10 * hidden) : 0;
    const auto rank = coordinate_valid ? (index / hidden) % 10 : 0;
    const auto route = (token * layers + layer) * 10 + rank;
    const auto input_row = token * layers + layer;
    const bool metadata_valid = coordinate_valid && input.expert_ids.size() == static_cast<std::size_t>(n) * 480 &&
        input.route_weights.size() == input.expert_ids.size() && input.cpu_assignment.size() == input.expert_ids.size() &&
        input.input_available.size() == static_cast<std::size_t>(n) * layers;
    const bool available = metadata_valid && input.input_available[input_row] == 1 && input.cpu_assignment[route] == 1 &&
        input.original_q8.size() == static_cast<std::size_t>(n) * layers * 80;
    std::uint64_t payload_bytes = 0, available_input_rows = 0;
    for (auto flag : input.input_available) { require(flag <= 1, "input availability flag"); available_input_rows += flag; }
    if (available) {
        require(input.expert_ids[route] >= 0 && input.expert_ids[route] < 512 && std::isfinite(input.route_weights[route]), "witness route value range");
        const auto q8 = input.original_q8.subspan(input_row * 80, 80);
        for (const auto& b : q8)
            require((b.d & 0x7c00U) != 0x7c00U && (b.s & 0x7c00U) != 0x7c00U && !(b.d & 0x8000U), "witness Q8 finite/nonnegative headers");
        const auto gpu = reference_down.subspan((token * layers * 10 + layer * 10 + rank) * hidden, hidden);
        const auto cpu = p.routed_down.subspan((token * layers * 10 + layer * 10 + rank) * hidden, hidden);
        for (const auto span : {gpu, cpu}) for (float v : span) require(std::isfinite(v), "witness down nonfinite");
        exclusive_write("original-input.q8.bin", std::as_bytes(q8));
        exclusive_write("gpu-reference-down.f32.bin", std::as_bytes(gpu));
        exclusive_write("cpu-candidate-down.f32.bin", std::as_bytes(cpu));
        payload_bytes = q8.size_bytes() + gpu.size_bytes() + cpu.size_bytes();
        require(payload_bytes == 23360, "bounded witness payload bytes");
    }
    std::ostringstream out; out.imbue(std::locale::classic()); out << std::setprecision(17);
    out << "{\"kind\":\"session_hybrid_failure_witness\",\"protocol\":1,\"revision\":" << json_quote(CORE_REVISION)
        << ",\"runtime_scope\":\"CPUlinear_GPUmiddle\",\"pure_CPU_only_claim\":false"
        << ",\"dirty\":" << (CORE_DIRTY ? "true" : "false") << ",\"model_path\":" << json_quote(witness.model)
        << ",\"phase\":" << json_quote(phase) << ",\"window_offset\":" << offset << ",\"window_rows\":" << n
        << ",\"input_available\":" << (available ? "true" : "false") << ",\"selected_by\":\"first_unweighted_down_bit_difference_not_first_gate\""
        << ",\"payload_bytes\":" << payload_bytes << ",\"full_logit_gate\":{\"absolute\":0.02,\"relative\":0.002}"
        << ",\"intermediate_gate\":{\"absolute\":0.002,\"relative\":0.002},\"coordinates\":{";
    bool first = true;
    for (auto item : {std::pair{"logits", &logits}, std::pair{"routed_down", &routed}, std::pair{"ffn_output", &ffn}}) {
        if (!first) out << ',';
        first = false;
        const auto width = item.second == &logits ? vocab : item.second == &routed ? down_row : ffn_row;
        out << json_quote(item.first) << ":{\"first_bit\":"; coordinate(out, item.second->first_bit, width, item.first);
        out << ",\"first_gate\":"; coordinate(out, item.second->first_gate, width, item.first);
        out << ",\"nonfinite_pairs\":" << item.second->nonfinite << '}';
    }
    out << '}';
    if (metadata_valid) {
        out << ",\"selected\":{\"window_row\":" << token << ",\"token_offset\":" << offset + token << ",\"token_id\":" << ids[offset + token]
            << ",\"layer\":" << layer << ",\"rank\":" << rank << ",\"expert_id\":" << input.expert_ids[route]
            << ",\"raw_route_weight\":" << input.route_weights[route] << ",\"raw_route_weight_bits\":" << std::bit_cast<std::uint32_t>(input.route_weights[route])
            << ",\"reference_expert_id\":" << reference_ids[route] << ",\"reference_raw_weight_bits\":" << std::bit_cast<std::uint32_t>(reference_weights[route])
            << ",\"same_expert_as_reference\":" << (reference_ids[route] == input.expert_ids[route] ? "true" : "false")
            << ",\"actual_CPU_assignment\":" << (input.cpu_assignment[route] ? "true" : "false") << '}';
    } else out << ",\"selected\":null";
    const auto h = s.hybrid_stats();
    out << ",\"source\":{\"Q8_producer\":\"current_window_GPU_f21_completed_existing_CPU_input_D2H\",\"requantized\":false"
        << ",\"CPU_scope\":\"gate_up_and_down_projections\",\"middle_producer\":\"canonical_GPU_SiLU_Q8\""
        << ",\"Q8_block_bytes\":36,\"Q8_blocks\":80,\"Q8_file\":" << (available ? "\"original-input.q8.bin\"" : "null")
        << ",\"Q8_file_bytes\":" << (available ? "2880" : "0")
        << ",\"down_dtype\":\"little_endian_FP32\",\"down_elements_each\":2560,\"down_file_bytes_each\":10240"
        << ",\"gpu_reference_file\":" << (available ? "\"gpu-reference-down.f32.bin\"" : "null")
        << ",\"cpu_candidate_file\":" << (available ? "\"cpu-candidate-down.f32.bin\"" : "null")
        << ",\"reference_scope\":\"retained_same_Session_GPU_N1_not_replayed_on_candidate_input\""
        << ",\"available_current_input_rows\":" << available_input_rows << ",\"current_opaque_host_copy_bytes\":" << available_input_rows * 2880
        << ",\"cumulative_CPU_input_bytes\":" << h.input_bytes << ",\"cumulative_CPU_checked_input_bytes\":" << h.cpu_input_bytes_checked
        << ",\"cumulative_CPU_return_bytes\":" << h.cpu_return_bytes << ",\"cumulative_CPU_assignments\":" << h.cpu_assignments
        << ",\"cumulative_CPU_gate_up_jobs\":" << h.cpu_gate_up_jobs << ",\"cumulative_CPU_down_jobs\":" << h.cpu_down_jobs
        << ",\"cumulative_GPU_middle_columns\":" << h.gpu_middle_columns << ",\"cumulative_GPU_middle_batches\":" << h.gpu_middle_batches
        << ",\"cumulative_paired_gate_up_H2D_bytes\":" << h.paired_gate_up_bytes << ",\"cumulative_middle_Q8_D2H_bytes\":" << h.middle_q8_bytes
        << ",\"copied_payload_allfinite\":" << (available ? "true" : "null") << ",\"stored_half_zero_keeps_codes_and_raw_sum\":true}"
        << ",\"numerical_parity_passed\":false,\"HF_claim\":false,\"performance_claim\":false,\"R5_complete_claim\":false}\n";
    const auto text = out.str(); require(text.size() <= 16384, "bounded witness metadata");
    exclusive_write("witness.json", std::as_bytes(std::span(text.data(), text.size())));
    std::cout << "{\"kind\":\"session_hybrid_witness_written\",\"directory\":" << json_quote(witness.directory.string())
        << ",\"payload_bytes\":" << payload_bytes << ",\"input_available\":" << (available ? "true" : "false") << ",\"numerical_parity_passed\":false}\n";
}
struct TemporaryTrace {
    std::filesystem::path path;
    TemporaryTrace() {
        const auto parent = std::filesystem::temp_directory_path();
        require(std::filesystem::is_directory(parent), "trace temporary parent");
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        for (int i = 0; i < 8; ++i) {
            auto root = parent / ("core-session-hybrid-" + std::to_string(nonce) + "-" + std::to_string(i));
            if (std::filesystem::create_directory(root)) { path = root / "trace"; return; }
        }
        throw std::runtime_error("cannot reserve fixture trace parent");
    }
    ~TemporaryTrace() {
        if (!path.empty()) { std::error_code ec; std::filesystem::remove_all(path.parent_path(), ec); }
    }
    void fail_io() {
        require(std::filesystem::is_directory(path), "fixture trace directory not owned/live");
        std::filesystem::remove_all(path);
        require(!std::filesystem::exists(path), "fixture trace fault did not remove owned directory");
    }
};
void geometry(const SessionMemory& m, int slots, int batch = max_batch, int workers = 1) {
    require(batch == 1 || batch == 3 || batch == 4, "fixture logical capacity");
    const std::uint64_t stage_capacity = q41_bytes * (batch > 3 ? 16 : 1);
    require(m.capacity == rows && m.expert_slots == slots && m.ownership_verified, "memory owner/capacity");
    require(m.ram_expert_payload == (6 * q41_bytes + 42 * q40_bytes) * 512 &&
        m.ram_expert_capacity >= m.ram_expert_payload && m.expert_payload_reads == 144 &&
        m.expert_payload_bytes_read == m.ram_expert_payload, "canonical RAM/read ledger");
    require(m.cpu_workers == workers && m.cpu_pool_metadata > 0 &&
        m.cpu_pool_scratch >= workers * sizeof(CpuExpertScratch), "pool capacity ledger");
    require(m.pinned_hybrid_input == 2 * 8640 && m.pinned_hybrid_output == 2 * 307200 &&
        m.pinned_hybrid_error == 2 * sizeof(int) && m.pinned_hybrid_middle == 2 * 21600, "bounded pinned hybrid frames");
    require(m.pinned_handoff == static_cast<std::uint64_t>(batch) * 10240 * sizeof(float) &&
        m.pinned_expert_staging == 4 * stage_capacity, "short-triplet or wide-band16 double stages");
    require(m.host_logit_capacity >= 2 * batch * vocab * sizeof(float) &&
        m.host_hybrid_input_probe == input_probe_bytes &&
        m.host_hybrid_probe == 2 * 3 * (down_row + ffn_row) * sizeof(float) + input_probe_bytes &&
        m.host_hybrid_plans > 0 && m.host_cpu_expert_views >= 48 * 512 * sizeof(CpuExpert), "known host frames/views");
    require(m.host_routing_capacity >= 2 * batch * (512 * sizeof(float) + 10 * sizeof(float) + 10 * sizeof(std::int32_t)) &&
        m.host_route_group_payload == 2 * (sizeof(RouteGroups) + batch * 10 * sizeof(RouteAssignment)), "known routing payloads");
    require(m.pinned_route_metadata == static_cast<std::uint64_t>(2 * batch * 10 * 8), "pinned 8-byte route DTO capacity");
    for (std::size_t i = 0; i < m.devices.size(); ++i) {
        const auto& d = m.devices[i];
        require(d.device == static_cast<int>(i) && d.first_layer == static_cast<int>(24 * i) &&
            d.last_layer == static_cast<int>(24 * i + 23) && d.gdn_layers == 18 && d.qsa_layers == 6, "layer owners");
        require(d.weights + d.expert_slots + d.qsa_kv + d.qsa_index + d.gdn_state + d.ple_state + d.workspace == d.owned_bytes &&
            d.owned_buffers > 0 && d.owned_peak_bytes >= d.owned_bytes, "GPU categories/independent Buffer ledger");
        require(m.hybrid_contribution_bytes[i] == static_cast<std::uint64_t>(std::max(3, batch)) * 10 * hidden * sizeof(float) &&
            m.expert_stage_capacity_bytes[i] == stage_capacity, "individual added Buffer capacities");
        require(m.route_metadata_bytes[i] == static_cast<std::uint64_t>(batch * 10 * 8), "individual GPU route metadata Buffer capacity");
        require(m.hybrid_gate_up_bytes[i] == 153600 && m.hybrid_middle_float_bytes[i] == 76800 &&
            m.hybrid_middle_q8_bytes[i] == 21600 && m.hybrid_middle_error_bytes[i] == 4,
            "four independent CPUlinear_GPUmiddle GPU Buffer capacities");
        require(d.free_vram > 0 && d.free_vram <= d.total_vram && d.owned_bytes <= d.total_vram, "VRAM snapshot");
    }
}
void steady(const SessionMemory& a, const SessionMemory& b) {
    geometry(a, b.expert_slots, static_cast<int>(b.pinned_handoff / (10240 * sizeof(float))), b.cpu_workers);
    require(a.ram_expert_capacity == b.ram_expert_capacity && a.ram_expert_payload == b.ram_expert_payload &&
        a.host_embedding_capacity == b.host_embedding_capacity && a.host_logit_capacity == b.host_logit_capacity &&
        a.pinned_handoff == b.pinned_handoff && a.pinned_expert_staging == b.pinned_expert_staging &&
        a.pinned_hybrid_input == b.pinned_hybrid_input && a.pinned_hybrid_output == b.pinned_hybrid_output &&
        a.pinned_hybrid_error == b.pinned_hybrid_error && a.host_hybrid_plans == b.host_hybrid_plans &&
        a.host_cpu_expert_views == b.host_cpu_expert_views && a.cpu_pool_metadata == b.cpu_pool_metadata &&
        a.cpu_pool_scratch == b.cpu_pool_scratch && a.cpu_workers == b.cpu_workers && a.host_hybrid_probe == b.host_hybrid_probe &&
        a.host_routing_capacity == b.host_routing_capacity && a.host_route_group_payload == b.host_route_group_payload &&
        a.pinned_route_metadata == b.pinned_route_metadata && a.route_metadata_bytes == b.route_metadata_bytes &&
        a.host_hybrid_input_probe == b.host_hybrid_input_probe &&
        a.pinned_hybrid_middle == b.pinned_hybrid_middle && a.hybrid_gate_up_bytes == b.hybrid_gate_up_bytes &&
        a.hybrid_middle_float_bytes == b.hybrid_middle_float_bytes && a.hybrid_middle_q8_bytes == b.hybrid_middle_q8_bytes &&
        a.hybrid_middle_error_bytes == b.hybrid_middle_error_bytes &&
        a.hybrid_contribution_bytes == b.hybrid_contribution_bytes && a.expert_stage_capacity_bytes == b.expert_stage_capacity_bytes,
        "host/known capacity changed");
    for (std::size_t i = 0; i < a.devices.size(); ++i) {
        const auto& x = a.devices[i]; const auto& y = b.devices[i];
        require(x.weights == y.weights && x.expert_slots == y.expert_slots && x.qsa_kv == y.qsa_kv &&
            x.qsa_index == y.qsa_index && x.gdn_state == y.gdn_state && x.ple_state == y.ple_state &&
            x.workspace == y.workspace && x.owned_bytes == y.owned_bytes && x.owned_buffers == y.owned_buffers &&
            x.owned_peak_bytes == y.owned_peak_bytes && x.total_vram == y.total_vram, "steady GPU allocation ledger changed");
    }
}
struct Preserved {
    SessionStats stats;
    SessionRouteStats routes;
    SessionHybridStats hybrid;
    SessionHybridPolicy policy;
    std::vector<float> logits, down, ffn;
    std::vector<Q8_1> original;
    std::vector<std::int32_t> expert_ids;
    std::vector<float> route_weights;
    std::vector<std::uint8_t> available, cpu;
    std::span<const float> live_logits;
    explicit Preserved(Session& s, std::span<const float> active) : stats(s.stats()), routes(s.route_stats()),
        hybrid(s.hybrid_stats()), policy(s.hybrid_policy()), logits(active.begin(), active.end()), live_logits(active) {
        const auto p = s.hybrid_intermediates(); down.assign(p.routed_down.begin(), p.routed_down.end());
        ffn.assign(p.ffn_output.begin(), p.ffn_output.end());
        const auto input = s.hybrid_inputs();
        original.resize(input.original_q8.size()); // Do not read unavailable rows.
        expert_ids.assign(input.expert_ids.begin(), input.expert_ids.end());
        route_weights.assign(input.route_weights.begin(), input.route_weights.end());
        available.assign(input.input_available.begin(), input.input_available.end());
        cpu.assign(input.cpu_assignment.begin(), input.cpu_assignment.end());
        for (std::size_t i = 0; i < available.size(); ++i) if (available[i])
            std::memcpy(original.data() + i * 80, input.original_q8.data() + i * 80, 2880);
    }
    void check(Session& s) const {
        require(same(stats, s.stats()) && same(routes, s.route_stats()) && same(hybrid, s.hybrid_stats()), "rejection/inspection changed published counters");
        require(policy.mode == s.hybrid_policy().mode && policy.gpu_miss_groups == s.hybrid_policy().gpu_miss_groups &&
            policy.diagnostic_fail_after_admission == s.hybrid_policy().diagnostic_fail_after_admission, "rejection changed policy");
        require(exact(live_logits, logits), "rejection/inspection changed full ACTIVE logit span");
        const auto p = s.hybrid_intermediates();
        require(exact(p.routed_down, down) && exact(p.ffn_output, ffn), "rejection/inspection changed intermediate publication");
        const auto input = s.hybrid_inputs();
        require(input.original_q8.size() == original.size() && std::ranges::equal(input.expert_ids, expert_ids) &&
            exact(input.route_weights, route_weights) && std::ranges::equal(input.input_available, available) &&
            std::ranges::equal(input.cpu_assignment, cpu), "rejection/inspection changed input/route publication");
        for (std::size_t i = 0; i < available.size(); ++i) if (available[i])
            require(std::memcmp(original.data() + i * 80, input.original_q8.data() + i * 80, 2880) == 0, "rejection changed opaque input bytes");
    }
};
template<class F> void reject(Session& s, const Preserved& before, F&& f) {
    const auto diagnostics = s.hybrid_probe_diagnostics();
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "expected atomic argument rejection"); before.check(s);
    require(same(diagnostics, s.hybrid_probe_diagnostics()), "argument rejection changed fault evidence/cache map IDs");
}
void rejects(Session& s, std::span<const float> active, int batch = max_batch) {
    const Preserved before(s, active);
    const std::array<std::int32_t, 5> long_window{100, 101, 102, 103, 104};
    auto negative_last = long_window, vocabulary_last = long_window;
    negative_last[batch - 1] = -1; vocabulary_last[batch - 1] = 248320;
    reject(s, before, [&] { (void)s.step_batch({}); });
    reject(s, before, [&] { (void)s.step_batch(std::span(long_window).first(batch + 1)); });
    reject(s, before, [&] { (void)s.step_batch(std::span(negative_last).first(batch)); });
    reject(s, before, [&] { (void)s.step_batch(std::span(vocabulary_last).first(batch)); });
    reject(s, before, [&] { s.set_hybrid_policy({static_cast<SessionHybridMode>(99), 2}); });
    reject(s, before, [&] { s.set_hybrid_policy({SessionHybridMode::mixed, 0}); });
    reject(s, before, [&] { s.set_hybrid_policy({SessionHybridMode::mixed, 3}); });
    reject(s, before, [&] { s.set_hybrid_policy({SessionHybridMode::disabled, -1}); });
    reject(s, before, [&] { s.set_hybrid_policy({SessionHybridMode::force_cpu, 0, true}); });
}
void zero(Session& s) {
    require(same(s.stats(), SessionStats{}) && same(s.route_stats(), SessionRouteStats{}) &&
        same(s.hybrid_stats(), SessionHybridStats{}), "successful reset did not clear statistics");
    const auto p = s.hybrid_intermediates(); require(p.routed_down.empty() && p.ffn_output.empty(), "reset probe visibility");
    const auto input = s.hybrid_inputs();
    require(input.original_q8.empty() && input.expert_ids.empty() && input.route_weights.empty() &&
        input.input_available.empty() && input.cpu_assignment.empty(), "reset input probe visibility");
}
struct Reference {
    std::vector<float> logits = std::vector<float>(rows * vocab);
    std::vector<float> down = std::vector<float>(rows * down_row);
    std::vector<float> ffn = std::vector<float>(rows * ffn_row);
    std::vector<std::int32_t> expert_ids = std::vector<std::int32_t>(rows * layers * 10);
    std::vector<float> route_weights = std::vector<float>(rows * layers * 10);
    void retain(Session& s, std::span<const float> result, std::size_t offset) {
        require(result.size() == vocab, "N1 reference full vocabulary");
        const auto p = s.hybrid_intermediates();
        require(p.routed_down.size() == down_row && p.ffn_output.size() == ffn_row, "N1 intermediate reference extent");
        for (const auto span : {result, p.routed_down, p.ffn_output})
            for (float x : span) require(std::isfinite(x), "nonfinite retained reference");
        std::copy(result.begin(), result.end(), logits.data() + offset * vocab);
        std::copy(p.routed_down.begin(), p.routed_down.end(), down.data() + offset * down_row);
        std::copy(p.ffn_output.begin(), p.ffn_output.end(), ffn.data() + offset * ffn_row);
        const auto input = s.hybrid_inputs();
        require(input.expert_ids.size() == 480 && input.route_weights.size() == 480 &&
            std::ranges::all_of(input.input_available, [](auto flag) { return flag == 0; }), "GPU reference input unavailable/route extent");
        std::copy(input.expert_ids.begin(), input.expert_ids.end(), expert_ids.data() + offset * 480);
        std::copy(input.route_weights.begin(), input.route_weights.end(), route_weights.data() + offset * 480);
    }
    void check(Session& s, std::span<const float> result, int offset, int n, std::string_view phase) const {
        require(result.size() == static_cast<std::size_t>(n) * vocab, "current window full vocabulary");
        const auto p = s.hybrid_intermediates();
        const auto logit = compare(result, std::span(logits).subspan(offset * vocab, n * vocab), .02);
        Metrics routed, output;
        if (n <= 3) {
            routed = compare(p.routed_down, std::span(down).subspan(offset * down_row, n * down_row), .002);
            output = compare(p.ffn_output, std::span(ffn).subspan(offset * ffn_row, n * ffn_row), .002);
        } else require(p.routed_down.empty() && p.ffn_output.empty(), "wide call published stale/out-of-bounds short probe");
        const auto h = s.hybrid_stats();
        std::cout << "{\"kind\":\"session_hybrid_window\",\"phase\":" << std::quoted(std::string(phase))
            << ",\"offset\":" << offset << ",\"rows\":" << n << ",\"continuation\":" << (offset >= teacher_rows ? "true" : "false")
            << ",\"logits\":"; emit(logit);
        std::cout << ",\"intermediate_probe_available\":" << (n <= 3 ? "true" : "false") << ",\"unweighted_down\":";
        if (n <= 3) emit(routed); else std::cout << "null";
        std::cout << ",\"ffn_output\":";
        if (n <= 3) emit(output); else std::cout << "null";
        std::cout << ",\"cumulative_cpu_assignments\":" << h.cpu_assignments << ",\"cumulative_gpu_hit_assignments\":" << h.gpu_hit_assignments
            << ",\"cumulative_gpu_miss_assignments\":" << h.gpu_miss_assignments
            << ",\"cumulative_cpu_gate_up_jobs\":" << h.cpu_gate_up_jobs << ",\"cumulative_cpu_down_jobs\":" << h.cpu_down_jobs
            << ",\"cumulative_GPU_middle_columns\":" << h.gpu_middle_columns << ",\"cumulative_GPU_middle_batches\":" << h.gpu_middle_batches
            << ",\"cumulative_paired_gate_up_H2D_bytes\":" << h.paired_gate_up_bytes
            << ",\"cumulative_middle_Q8_D2H_bytes\":" << h.middle_q8_bytes << "}\n";
        if (logit.violations || routed.violations || output.violations) {
            dump_witness(s, std::span(down).subspan(offset * down_row, n * down_row), offset, n, phase, logit, routed, output,
                std::span(expert_ids).subspan(offset * 480, n * 480), std::span(route_weights).subspan(offset * 480, n * 480));
            throw std::runtime_error("FROZEN numerical gate failed");
        }
        ++checked.windows; checked.rows += n;
        if (n <= 3) checked.short_rows += n;
        else { ++checked.wide_windows; checked.wide_rows += n; }
    }
};
void account(Session& s, SessionStats before, SessionHybridStats hb, int offset, int n, SessionHybridMode mode, int slots) {
    const auto a = s.stats(); const auto h = s.hybrid_stats(); const auto r = s.route_stats();
    const auto expected = static_cast<std::uint64_t>(48 * 10 * n);
    require(before.consumed_tokens == static_cast<std::uint64_t>(offset) && a.consumed_tokens == before.consumed_tokens + n &&
        std::isfinite(a.last_completed_ms) && a.last_completed_ms >= 0, "consumed/completed statistics");
    require(a.expert_hits >= before.expert_hits && a.expert_misses >= before.expert_misses &&
        a.expert_upload_bytes >= before.expert_upload_bytes &&
        a.expert_hits - before.expert_hits + a.expert_misses - before.expert_misses == expected, "complete route accounting");
    require(r.last_max_expert_group_assignments >= 1 && r.last_max_expert_group_assignments <= static_cast<std::uint64_t>(n) &&
        r.expert_groups_gt128 == 0, "logical RouteGroups ledger");
    if (mode == SessionHybridMode::disabled) {
        require(same(h, hb), "disabled path used hybrid scheduler"); return;
    }
    if (n > 3) {
        auto wide_expected = hb; wide_expected.gpu_only_wide_layers += 48;
        // Equality covers every CPU job/assignment/extraction/input/return byte
        // and all short residency/admission diagnostics, not just job count.
        require(same(h, wide_expected), "wide GPU fallback changed short/CPU scheduler diagnostics");
        const auto p = s.hybrid_intermediates();
        require(p.routed_down.empty() && p.ffn_output.empty(), "wide probe visibility");
        const auto input = s.hybrid_inputs(); require(input.original_q8.empty() && input.expert_ids.empty(), "wide input probe visibility");
        return;
    }
    require(h.short_layers - hb.short_layers == 48 && h.gpu_only_wide_layers == hb.gpu_only_wide_layers &&
        h.ready_hit_assignments - hb.ready_hit_assignments + h.physical_miss_assignments - hb.physical_miss_assignments == expected &&
        h.cpu_assignments - hb.cpu_assignments + h.gpu_hit_assignments - hb.gpu_hit_assignments + h.gpu_miss_assignments - hb.gpu_miss_assignments == expected,
        "physical residency/dispatch conservation");
    const auto groups = h.cpu_groups - hb.cpu_groups + h.gpu_hit_groups - hb.gpu_hit_groups + h.gpu_miss_groups - hb.gpu_miss_groups;
    const auto reuse = h.group_reuse_assignments - hb.group_reuse_assignments;
    const auto misses = a.expert_misses - before.expert_misses, hits = a.expert_hits - before.expert_hits;
    const auto ready_assignments = h.ready_hit_assignments - hb.ready_hit_assignments;
    const auto missing_assignments = h.physical_miss_assignments - hb.physical_miss_assignments;
    require(groups + reuse == expected && hits >= reuse && misses <= groups, "historical group/reuse accounting");
    const auto ready_groups = hits - reuse; // One first-acquisition hit per READY group.
    require(ready_groups + misses == groups && ready_groups <= ready_assignments && ready_assignments <= ready_groups * n &&
        misses <= missing_assignments && missing_assignments <= misses * n, "historical first acquisitions versus physical assignment residency");
    if (mode == SessionHybridMode::mixed)
        require(ready_groups == h.gpu_hit_groups - hb.gpu_hit_groups, "mixed READY group hits were relabeled as reuse/dispatch");
    require(h.cpu_return_bytes - hb.cpu_return_bytes == (h.cpu_assignments - hb.cpu_assignments) * hidden * sizeof(float), "CPU return row byte ledger");
    const auto cpu_groups = h.cpu_groups - hb.cpu_groups, cpu_columns = h.cpu_assignments - hb.cpu_assignments;
    require(h.cpu_gate_up_jobs - hb.cpu_gate_up_jobs == cpu_groups && h.cpu_down_jobs - hb.cpu_down_jobs == cpu_groups &&
        h.gpu_middle_columns - hb.gpu_middle_columns == cpu_columns &&
        h.gpu_middle_batches - hb.gpu_middle_batches == h.input_extractions - hb.input_extractions &&
        h.paired_gate_up_bytes - hb.paired_gate_up_bytes == cpu_columns * 1280 * sizeof(float) &&
        h.middle_q8_bytes - hb.middle_q8_bytes == cpu_columns * 20 * sizeof(Q8_1),
        "CPUlinear_GPUmiddle two-phase job/single-batch transfer conservation");
    require(h.cpu_input_bytes_checked - hb.cpu_input_bytes_checked == (h.cpu_assignments - hb.cpu_assignments) * 2880,
        "every CPU assignment must match GPU route DTO token/rank and exact original Q8 descriptor bytes");
    const auto input = s.hybrid_inputs();
    require(input.original_q8.size() == static_cast<std::size_t>(n) * 48 * 80 && input.expert_ids.size() == static_cast<std::size_t>(n) * 480 &&
        input.route_weights.size() == input.expert_ids.size() && input.cpu_assignment.size() == input.expert_ids.size() &&
        input.input_available.size() == static_cast<std::size_t>(n) * 48, "short input probe extent");
    require(std::count(input.cpu_assignment.begin(), input.cpu_assignment.end(), std::uint8_t{1}) ==
        static_cast<std::int64_t>(h.cpu_assignments - hb.cpu_assignments), "probe actual CPU dispatch ledger");
    require(static_cast<std::uint64_t>(std::count(input.input_available.begin(), input.input_available.end(), std::uint8_t{1})) * 2880 ==
        h.input_bytes - hb.input_bytes, "probe copied opaque original input byte ledger");
    const auto gpu_groups = h.gpu_miss_groups - hb.gpu_miss_groups;
    const auto uploads = a.expert_upload_bytes - before.expert_upload_bytes;
    require(uploads >= gpu_groups * q40_bytes && uploads <= gpu_groups * q41_bytes, "actual GPU staged upload byte ledger");
    require(h.admitted_groups - hb.admitted_groups <= 96, "experimental quota exceeded");
    if (mode == SessionHybridMode::force_cpu) {
        require(h.cpu_assignments - hb.cpu_assignments == expected && gpu_groups == 0 && uploads == 0 &&
            h.gpu_hit_assignments == hb.gpu_hit_assignments && h.input_extractions - hb.input_extractions == 48 &&
            h.input_bytes - hb.input_bytes == static_cast<std::uint64_t>(48 * n * 2880) &&
             h.forced_cpu_layers - hb.forced_cpu_layers == 48, "forced CPU LINEAR dispatch mislabeled as physical residency");
    } else if (mode == SessionHybridMode::mixed && slots == 1) {
        require(h.cpu_assignments - hb.cpu_assignments >= static_cast<std::uint64_t>(48 * 7 * n) && gpu_groups > 0 &&
            h.admitted_groups - hb.admitted_groups > 0 && h.input_extractions - hb.input_extractions == 48,
            "slot1 natural misses did not exercise mixed CPU/GPU admission");
    }
}
void phase(Session& s, const Reference& ref, const SessionMemory& loaded, int n, SessionHybridPolicy policy, std::string_view name,
           bool fresh_cache = false) {
    s.reset(); zero(s); s.set_hybrid_policy(policy); steady(s.memory(), loaded);
    int offset = 0;
    std::span<const float> active;
    while (offset < rows) {
        const int count = offset < teacher_rows ? std::min(n, teacher_rows - offset) : 1;
        const auto before = s.stats(); const auto hybrid_before = s.hybrid_stats();
        active = s.step_batch(std::span(ids).subspan(offset, count));
        // Compare immediately, BEFORE the next accepted call invalidates spans.
        ref.check(s, active, offset, count, name); account(s, before, hybrid_before, offset, count, policy.mode, loaded.expert_slots);
        const Preserved saved(s, active); steady(s.memory(), loaded); saved.check(s);
        offset += count;
        if (offset == count) {
            if (fresh_cache) {
                const auto h = s.hybrid_stats();
                require(h.ready_hit_assignments == 0 && h.physical_miss_assignments == static_cast<std::uint64_t>(48 * 10 * count) &&
                    s.stats().expert_hits == h.group_reuse_assignments &&
                    s.stats().expert_misses + h.group_reuse_assignments == static_cast<std::uint64_t>(48 * 10 * count),
                    "failure reset replay did not preserve fresh-miss/within-window-reuse accounting");
            }
            rejects(s, active);
            s.set_hybrid_policy(policy); saved.check(s); steady(s.memory(), loaded);
        }
    }
    const Preserved last(s, active);
    reject(s, last, [&] { (void)s.step(ids[0]); });
    s.reset(); zero(s); steady(s.memory(), loaded);
    const auto h = last.hybrid;
    if (policy.mode == SessionHybridMode::mixed)
        require(h.evicted_ready_slots > 0 && h.admitted_groups > 0, "slot1 fixture did not reuse/evict a READY physical slot");
    std::cout << "{\"kind\":\"session_hybrid_phase\",\"phase\":" << std::quoted(std::string(name))
        << ",\"teacher_rows\":32,\"continuation_rows\":8,\"cpu_groups\":" << h.cpu_groups
        << ",\"gpu_hit_groups\":" << h.gpu_hit_groups << ",\"gpu_miss_groups\":" << h.gpu_miss_groups
        << ",\"group_reuse_assignments\":" << h.group_reuse_assignments
        << ",\"admission_copies\":" << h.admitted_groups << ",\"evicted_ready_slots\":" << h.evicted_ready_slots
        << ",\"input_extractions\":" << h.input_extractions << ",\"CPU_input_bytes_checked\":" << h.cpu_input_bytes_checked
        << ",\"cpu_gate_up_jobs\":" << h.cpu_gate_up_jobs << ",\"cpu_down_jobs\":" << h.cpu_down_jobs
        << ",\"GPU_middle_columns\":" << h.gpu_middle_columns << ",\"GPU_middle_batches\":" << h.gpu_middle_batches
        << ",\"paired_gate_up_H2D_bytes\":" << h.paired_gate_up_bytes << ",\"middle_Q8_D2H_bytes\":" << h.middle_q8_bytes
        << ",\"fresh_cache_first_call_checked\":" << (fresh_cache ? "true" : "false") << ",\"passed\":true}\n";
}
void mixed_captured_reader_reuse(Session& s, const Reference& ref, const SessionMemory& loaded) {
    s.reset(); s.set_hybrid_policy({SessionHybridMode::mixed, 1});
    ref.check(s, s.step(ids[0]), 0, 1, "mixed_slot1_warm_BOS_quota1");
    s.reset(); zero(s);
    const auto current = s.step(ids[0]); ref.check(s, current, 0, 1, "mixed_slot1_captured_reader_then_eviction_BOS");
    const auto h = s.hybrid_stats();
    // At least layer0's identical GPU-produced FFN input/router reselects the
    // admitted BOS expert. Later routes are OBSERVED, never assumed unchanged.
    require(h.gpu_hit_groups > 0 && h.cpu_groups > 0 && h.gpu_miss_groups > 0 &&
        h.evicted_ready_slots > 0 && h.admitted_groups > 0, "mixed capture/read/evict seam not exercised");
    require(h.cpu_input_bytes_checked == h.cpu_assignments * 2880, "mixed exact original Q8 proof missing");
    const Preserved before(s, current); steady(s.memory(), loaded); before.check(s);
    std::cout << "{\"kind\":\"session_hybrid_captured_reader_reuse\",\"slots\":1,\"GPU_hit_groups\":" << h.gpu_hit_groups
        << ",\"CPU_groups\":" << h.cpu_groups << ",\"GPU_miss_groups\":" << h.gpu_miss_groups
        << ",\"evicted_READY_slots\":" << h.evicted_ready_slots << ",\"quota\":1,\"passed\":true}\n";
}
void sticky_io_failure(Session& s, TemporaryTrace& trace, const Reference& ref, const SessionMemory& loaded) {
    s.reset(); s.set_hybrid_policy({SessionHybridMode::mixed, 2});
    std::span<const float> active;
    for (int offset = 0; offset < 39; offset += 3) {
        active = s.step_batch(std::span(ids).subspan(offset, 3));
        ref.check(s, active, offset, 3, "sticky_failure_setup");
    }
    const Preserved before(s, active);
    const std::array<std::int32_t, 2> capacity_suffix{137, 138};
    reject(s, before, [&] { (void)s.step_batch(capacity_suffix); });
    trace.fail_io(); // Existing capture write failure, BEFORE CPU submission.
    bool failed = false;
    try { (void)s.step(ids[39]); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "accepted capture I/O failure was not reported"); before.check(s);
    failed = false;
    try { (void)s.step(ids[39]); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "failed Session was not sticky"); before.check(s);
    failed = false;
    try { s.set_hybrid_policy({SessionHybridMode::disabled, 2}); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "policy change bypassed required reset"); before.check(s);
    steady(s.memory(), loaded); before.check(s);
    s.reset(); zero(s);
    const auto recovered = s.step(ids[0]); ref.check(s, recovered, 0, 1, "reset_after_sticky_failure");
    steady(s.memory(), loaded);
    std::cout << "{\"kind\":\"session_hybrid_sticky_failure\",\"injection\":\"owned_trace_IO_before_CPU_submission\","
        "\"published_stats_logits_intermediates_preserved\":true,\"reset_recovery\":true,\"passed\":true}\n";
}
void sticky_submitted_admission_failure(Session& s, const Reference& ref, const SessionMemory& loaded) {
    s.reset(); zero(s); s.set_hybrid_policy({SessionHybridMode::mixed, 2});
    const auto active = s.step_batch(std::span(ids).first(3));
    ref.check(s, active, 0, 3, "synthetic_failure_setup_N3");
    require(s.hybrid_probe_diagnostics().ready_cache_ids == 48, "fault setup needs READY slot in every layer");
    s.set_hybrid_policy({SessionHybridMode::mixed, 2, true});
    const Preserved before(s, active);
    rejects(s, active); // Bad suffix/policy must not submit or consume the one-shot.
    bool failed = false;
    try { (void)s.step_batch(std::span(ids).subspan(3, 3)); }
    catch (const std::runtime_error& e) {
        require(std::string_view(e.what()) == "synthetic hybrid failure after CPU acceptance and GPU admission enqueue",
            "unexpected failure instead of the controlled post-admission fault");
        failed = true;
    }
    require(failed, "synthetic post-admission failure not reached"); before.check(s);
    const auto fault = s.hybrid_probe_diagnostics();
    require(fault.synthetic_failure && fault.failure_layer == 0 && fault.accepted_cpu_jobs > 0 &&
        fault.queued_admission_copies > 0 && fault.pending_slots_at_failure > 0 &&
        fault.cpu_return_bytes_before_failure == 0 && fault.cpu_pool_drained && fault.gpu_streams_drained &&
        fault.ready_cache_ids == 0 && fault.pending_cache_ids == 0, "post-admission drain/publication/cache failure contract");
    failed = false;
    try { (void)s.step(ids[3]); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "post-admission failure did not require reset"); before.check(s);
    failed = false;
    try { s.set_hybrid_policy({SessionHybridMode::mixed, 2}); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "post-admission policy change bypassed reset"); before.check(s);
    steady(s.memory(), loaded); before.check(s);
    require(same(fault, s.hybrid_probe_diagnostics()), "inspection/sticky rejection changed fault/cache evidence");
    s.reset(); zero(s); // The consumed one-shot cannot fire again during recovery.
    require(same(fault, s.hybrid_probe_diagnostics()), "reset recreated invalidated cache IDs/lost fault evidence");
    s.set_hybrid_policy({SessionHybridMode::mixed, 2, false});
    // All forty retained full-vocabulary rows, same frozen gates. First replay
    // call must see zero READY hits in ALL layers, not merely a miss at layer0.
    phase(s, ref, loaded, 3, {SessionHybridMode::mixed, 2}, "full_replay_after_submitted_admission_failure", true);
    std::cout << "{\"kind\":\"session_hybrid_submitted_admission_failure\",\"synthetic\":true,\"failure_layer\":" << fault.failure_layer
        << ",\"accepted_CPU_job_stage\":\"gate_up_before_GPU_middle_and_down\""
        << ",\"accepted_CPU_jobs\":" << fault.accepted_cpu_jobs << ",\"queued_admission_copies\":" << fault.queued_admission_copies
        << ",\"pending_host_slots_at_throw\":" << fault.pending_slots_at_failure
        << ",\"CPU_return_bytes_before_throw\":" << fault.cpu_return_bytes_before_failure << ','
        << "\"pool_and_both_GPU_streams_drained\":true,\"all_READY_and_pending_IDs_invalidated\":true,"
        "\"published_stats_route_hybrid_logits_probes_preserved\":true,\"reset_full_replay_rows\":40,"
        "\"fresh_physical_misses_first_replay_assignments\":1440,\"inflight_timing_claim\":false,\"passed\":true}\n";
}
void short_wide_short(Session& s, const Reference& ref, const SessionMemory& loaded) {
    s.reset(); zero(s); s.set_hybrid_policy({SessionHybridMode::mixed, 2}); steady(s.memory(), loaded);
    static constexpr std::array<int, 11> teacher_windows{1, 4, 2, 4, 3, 4, 1, 4, 2, 4, 3};
    static_assert([] { int sum = 0; for (int n : teacher_windows) sum += n; return sum; }() == teacher_rows);
    int offset = 0, short_calls = 0, wide_calls = 0;
    std::span<const float> active;
    auto call = [&](int n) {
        const auto before = s.stats(); const auto hb = s.hybrid_stats();
        active = s.step_batch(std::span(ids).subspan(offset, n));
        ref.check(s, active, offset, n, n == 4 ? "max4_GPU_fallback_N4" : "max4_short_hybrid_N" + std::to_string(n));
        account(s, before, hb, offset, n, SessionHybridMode::mixed, 1);
        const Preserved saved(s, active); steady(s.memory(), loaded); saved.check(s);
        if (n == 4) {
            ++wide_calls;
            // Test all four active vocabulary rows and late invalid fourth ID.
            rejects(s, active, 4);
        } else { ++short_calls; require(s.hybrid_intermediates().routed_down.size() == n * down_row, "short probe did not resume after wide call"); }
        offset += n;
    };
    for (int n : teacher_windows) call(n);
    require(offset == teacher_rows && short_calls == 6 && wide_calls == 5, "teacher short/wide timeline counts");
    for (int i = 0; i < 8; ++i) call(1);
    require(offset == rows && short_calls == 14 && wide_calls == 5, "timeline continuation counts");
    const auto h = s.hybrid_stats();
    require(h.short_layers == 14 * 48 && h.gpu_only_wide_layers == 5 * 48 && h.cpu_groups > 0 &&
        h.cpu_input_bytes_checked == h.cpu_assignments * 2880 && h.evicted_ready_slots > 0 && h.admitted_groups > 0,
        "short/wide scheduler/count/slot reuse evidence");
    const Preserved last(s, active); reject(s, last, [&] { (void)s.step(ids[0]); });
    s.reset(); zero(s); steady(s.memory(), loaded);
    std::cout << "{\"kind\":\"session_hybrid_short_wide_short\",\"session_instance\":3,\"max_batch_tokens\":4,\"cpu_workers\":4,"
        "\"teacher_window_sizes\":[1,4,2,4,3,4,1,4,2,4,3],\"teacher_rows\":32,\"continuation_N1_rows\":8,"
        "\"short_calls\":14,\"wide_N4_calls\":5,\"short_layers\":672,\"gpu_only_wide_layers\":240,"
        "\"wide_CPU_counters_and_transfer_bytes_unchanged\":true,\"wide_intermediate_probe_available\":false,"
        "\"stage_capacity_bytes_each\":45875200,\"total_pinned_stage_bytes\":183500800,"
        "\"contribution_bytes_each_GPU\":409600,\"full_40_row_replay\":true,\"passed\":true}\n";
}
void memory_record(const SessionMemory& m, int instance) {
    std::cout << "{\"kind\":\"session_hybrid_memory\",\"session_instance\":" << instance << ",\"slots\":" << m.expert_slots
        << ",\"max_batch_tokens\":" << m.pinned_handoff / (10240 * sizeof(float))
        << ",\"cpu_workers\":" << m.cpu_workers << ",\"pinned_input_bytes\":" << m.pinned_hybrid_input
        << ",\"pinned_output_bytes\":" << m.pinned_hybrid_output << ",\"pinned_error_bytes\":" << m.pinned_hybrid_error
        << ",\"pinned_middle_Q8_bytes\":" << m.pinned_hybrid_middle
        << ",\"pinned_stage_bytes\":" << m.pinned_expert_staging << ",\"host_plan_bytes\":" << m.host_hybrid_plans
        << ",\"host_cpu_views_and_pending_bytes\":" << m.host_cpu_expert_views << ",\"host_probe_bytes\":" << m.host_hybrid_probe
        << ",\"host_input_probe_bytes\":" << m.host_hybrid_input_probe
        << ",\"pool_metadata_bytes\":" << m.cpu_pool_metadata << ",\"pool_scratch_bytes\":" << m.cpu_pool_scratch
        << ",\"host_routing_capacity_bytes\":" << m.host_routing_capacity << ",\"RouteGroups_requested_payload_bytes\":" << m.host_route_group_payload
        << ",\"pinned_route_metadata_bytes\":" << m.pinned_route_metadata
        << ",\"route_metadata_bytes_per_GPU\":[" << m.route_metadata_bytes[0] << ',' << m.route_metadata_bytes[1] << ']'
        << ",\"paired_gate_up_bytes_per_GPU\":[" << m.hybrid_gate_up_bytes[0] << ',' << m.hybrid_gate_up_bytes[1] << ']'
        << ",\"middle_float_bytes_per_GPU\":[" << m.hybrid_middle_float_bytes[0] << ',' << m.hybrid_middle_float_bytes[1] << ']'
        << ",\"middle_Q8_bytes_per_GPU\":[" << m.hybrid_middle_q8_bytes[0] << ',' << m.hybrid_middle_q8_bytes[1] << ']'
        << ",\"middle_error_bytes_per_GPU\":[" << m.hybrid_middle_error_bytes[0] << ',' << m.hybrid_middle_error_bytes[1] << ']'
        << ",\"contribution_bytes_per_GPU\":[" << m.hybrid_contribution_bytes[0] << ',' << m.hybrid_contribution_bytes[1]
        << "],\"stage_buffer_bytes_per_GPU\":[" << m.expert_stage_capacity_bytes[0] << ',' << m.expert_stage_capacity_bytes[1]
        << "],\"full_RSS_or_stacks_claim\":false,\"ownership_verified\":true}\n";
}
} // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic()); std::cout << std::setprecision(17);
    try {
        require(argc == 2 || (argc == 4 && std::string_view(argv[2]) == "--failure-witness"),
            "usage: core-session-hybrid-test MODEL [--failure-witness NEW_DIRECTORY]"); provenance();
        require(std::filesystem::is_regular_file(argv[1]), "model must be a regular file");
        witness.model = argv[1];
        if (argc == 4) {
            const std::filesystem::path path(argv[3]);
            const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
            require(!path.empty() && std::filesystem::is_directory(parent) && !std::filesystem::exists(path) &&
                std::filesystem::create_directory(path), "failure witness directory must be new with existing parent");
            witness.directory = path;
        }
        std::cout << "{\"kind\":\"session_hybrid_header\",\"protocol\":1,\"revision\":" << std::quoted(CORE_REVISION)
            << ",\"dirty\":" << (CORE_DIRTY ? "true" : "false") << ",\"model_variant\":\"qwen38-keep1-Q4_0\","
            "\"runtime_scope\":\"CPUlinear_GPUmiddle\",\"force_cpu_enum_scope\":\"all_routed_linear_CPU_with_canonical_GPU_middle\",\"pure_CPU_only_claim\":false,"
            "\"sequential_Sessions\":3,\"simultaneous_Sessions\":1,\"max_batch_tokens_by_Session\":[3,1,4],\"cpu_workers_by_Session\":[1,1,4],"
            "\"reference_rows\":40,\"teacher\":\"BOS248044,100..130\","
            "\"continuation\":\"131..138\",\"full_logit_gate\":\".02+.002*abs(ref)\",\"intermediate_gate\":\".002+.002*abs(ref)\","
            "\"reference\":\"retained_full_vocabulary_GPU_N1_self_reference\",\"independent_HF_claim\":false,"
            "\"performance_claim\":false,\"R5_complete_claim\":false,\"MTP_claim\":false}\n";
        Reference reference;
        TemporaryTrace trace;
        {
            Session s(argv[1], {rows, 1, trace.path.string(), max_batch, 39, 1, true});
            const auto loaded = s.memory(); geometry(loaded, 1); memory_record(loaded, 1); zero(s);
            for (int i = 0; i < rows; ++i) {
                const auto current = s.step(ids[i]); reference.retain(s, current, i);
                const Preserved before(s, current); steady(s.memory(), loaded); before.check(s);
            }
            std::cout << "{\"kind\":\"session_hybrid_reference\",\"rows\":40,\"full_vocab_values\":" << rows * vocab
                << ",\"routed_down_values\":" << rows * down_row << ",\"ffn_output_values\":" << rows * ffn_row << ",\"all_finite\":true}\n";
            for (int n = 1; n <= 3; ++n)
                phase(s, reference, loaded, n, {SessionHybridMode::disabled, 2}, "hybrid_OFF_GPU_only_N" + std::to_string(n));
            for (int n = 1; n <= 3; ++n)
                phase(s, reference, loaded, n, {SessionHybridMode::force_cpu, 0}, "diagnostic_force_CPU_LINEAR_GPU_middle_N" + std::to_string(n));
            for (int n = 1; n <= 3; ++n)
                phase(s, reference, loaded, n, {SessionHybridMode::mixed, 2}, "mixed_slot1_N" + std::to_string(n));
            phase(s, reference, loaded, 3, {SessionHybridMode::force_gpu_misses, 2}, "diagnostic_force_GPU_misses_N3");
            mixed_captured_reader_reuse(s, reference, loaded);
            sticky_submitted_admission_failure(s, reference, loaded);
            // Destroying the owned trace directory is deliberately LAST for this
            // owner: later position39 captures would otherwise fail again.
            sticky_io_failure(s, trace, reference, loaded);
        } // All RAM/GPU/pool owners destroyed before constructing slots112.
        {
            Session s(argv[1], {rows, 112, {}, 1, 0, 1, true});
            const auto loaded = s.memory(); geometry(loaded, 112, 1); memory_record(loaded, 2);
            reference.check(s, s.step(ids[0]), 0, 1, "all_hit_warm_GPU_BOS");
            s.reset(); zero(s); s.set_hybrid_policy({SessionHybridMode::mixed, 2});
            const auto current = s.step(ids[0]); reference.check(s, current, 0, 1, "real_all_hit_reset_BOS");
            const auto h = s.hybrid_stats(); const auto a = s.stats();
            require(h.short_layers == 48 && h.all_hit_layers == 48 && h.ready_hit_assignments == 480 &&
                h.physical_miss_assignments == 0 && h.group_reuse_assignments == 0 && h.cpu_groups == 0 && h.cpu_assignments == 0 &&
                h.input_extractions == 0 && h.input_bytes == 0 && h.cpu_return_bytes == 0 &&
                h.cpu_input_bytes_checked == 0 &&
                h.cpu_gate_up_jobs == 0 && h.cpu_down_jobs == 0 && h.gpu_middle_columns == 0 && h.gpu_middle_batches == 0 &&
                h.paired_gate_up_bytes == 0 && h.middle_q8_bytes == 0 &&
                h.gpu_hit_assignments == 480 && h.gpu_miss_groups == 0 && h.admitted_groups == 0 &&
                a.expert_hits == 480 && a.expert_misses == 0 && a.expert_upload_bytes == 0,
                "real READY all-hit fast path extracted/dispatched/uploaded/admitted");
            rejects(s, current, 1); const Preserved before(s, current); steady(s.memory(), loaded); before.check(s);
            s.set_hybrid_policy({SessionHybridMode::mixed, 1});
            require(same(before.stats, s.stats()) && same(before.hybrid, s.hybrid_stats()) && exact(current, before.logits), "bounded policy update mutated results/counters");
            s.set_hybrid_policy({SessionHybridMode::mixed, 2}); before.check(s); steady(s.memory(), loaded);
            reference.check(s, s.step(ids[1]), 1, 1, "all_hit_session_continuation");
            std::cout << "{\"kind\":\"session_hybrid_real_all_hit\",\"warm_reset_replay\":\"BOS\",\"ready_assignments\":480,"
                "\"CPU_extractions\":0,\"CPU_jobs\":0,\"GPU_uploads\":0,\"passed\":true}\n";
        }
        { // Third owner starts only after both earlier model/pool owners died.
            Session s(argv[1], {rows, 1, {}, 4, 0, 4, true});
            const auto loaded = s.memory(); geometry(loaded, 1, 4, 4); memory_record(loaded, 3);
            short_wide_short(s, reference, loaded);
        }
        require(checked.windows == 326 && checked.rows == 528 && checked.short_rows == 508 &&
            checked.wide_windows == 5 && checked.wide_rows == 20, "fixture exact checked window/row counts");
        std::cout << "{\"kind\":\"session_hybrid_footer\",\"passed\":true,\"sequential_Sessions\":3,\"simultaneous_Sessions\":1,"
            "\"runtime_scope\":\"CPUlinear_GPUmiddle\",\"pure_CPU_only_claim\":false,"
            "\"bounded_GPU_quota\":2,\"physical_residency_not_dispatch\":true,\"sticky_failure_scopes\":[\"trace_IO_before_CPU\",\"synthetic_accepted_CPU_plus_queued_GPU_admission\"],"
            "\"retained_reference_rows\":40,\"compared_windows\":" << checked.windows << ",\"compared_full_vocab_rows\":" << checked.rows
            << ",\"compared_full_vocab_values\":" << checked.rows * vocab << ",\"intermediate_compared_rows\":" << checked.short_rows
            << ",\"compared_routed_down_values\":" << checked.short_rows * down_row << ",\"compared_ffn_output_values\":" << checked.short_rows * ffn_row
            << ",\"wide_N4_windows\":" << checked.wide_windows << ",\"wide_N4_rows\":" << checked.wide_rows
            << ",\"expected_execution_failures\":2,"
            "\"CPU_descriptor_exact_original_GPU_Q8_bytes\":true,\"CPU_descriptor_matches_GPU_route_DTO\":true,"
             "\"numeric_failure_fixture\":\"separate_CPU_stages_canonical_GPU_middle_and_GPU_shadow\","
            "\"individual_added_GPU_buffers_observable\":true,\"physical_128_tile_claim\":false,\"performance_claim\":false,\"R5_complete_claim\":false}\n";
        require(static_cast<bool>(std::cout), "JSONL output write failure");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "core-session-hybrid-test: " << e.what() << '\n';
        return 1;
    }
}
