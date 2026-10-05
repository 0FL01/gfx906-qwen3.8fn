#include "session.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
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
#error "core-prefill-wide-test requires the compiled 40-hex CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-prefill-wide-test requires the compiled CORE_DIRTY (0 or 1)"
#endif

namespace {
constexpr int capacity = 1056, expert_slots = 1, max_batch_tokens = 1024;
constexpr std::size_t vocabulary = 248320, teacher_rows = 1024, continuation_rows = 32;
constexpr double absolute_gate = .02, relative_gate = .002;
constexpr std::uint64_t routes_per_row = 480;
constexpr std::uint64_t q4_0_expert_bytes = 2764800, q4_1_expert_bytes = 2867200;
constexpr std::uint64_t ram_expert_payload = (6 * q4_1_expert_bytes + 42 * q4_0_expert_bytes) * 512;
constexpr std::uint64_t payload_reads = 144;
constexpr std::uint64_t reference_bytes = capacity * vocabulary * sizeof(float);
constexpr std::uint64_t host_logit_bytes = max_batch_tokens * vocabulary * sizeof(float);
constexpr std::uint64_t pinned_handoff_bytes = max_batch_tokens * 4 * 2560 * sizeof(float);
constexpr std::uint64_t stage_bytes = 16 * q4_1_expert_bytes;
constexpr std::uint64_t pinned_expert_staging_bytes = 2 * 2 * stage_bytes;
// Necessary aggregate floor, not evidence of individual Buffer capacities.
constexpr std::uint64_t workspace_floor = 24ULL * max_batch_tokens * 12288 * sizeof(float) +
    max_batch_tokens * 320ULL * 36 + 128ULL * vocabulary * sizeof(float) +
    max_batch_tokens * 10ULL * 2560 * sizeof(float) + 2 * stage_bytes;
constexpr std::array<std::uint64_t, 2> slot_bytes{
    6 * q4_1_expert_bytes + 18 * q4_0_expert_bytes, 24 * q4_0_expert_bytes};
constexpr std::uint64_t qsa_kv_bytes = 6ULL * capacity * 32 * 18;
constexpr std::uint64_t qsa_index_bytes = 6ULL * (capacity / 4 * 128 + 384) * sizeof(float);
constexpr std::uint64_t gdn_state_bytes = 18ULL * (786432 + 30720) * sizeof(float);
constexpr std::array<std::string_view, 4> phase_names{"reference_n1", "chunk128", "chunk129", "single1024"};
constexpr std::array<std::size_t, 4> expected_windows{1056, 40, 40, 33};
constexpr std::size_t window_count = 1169, success_record_count = 1181, memory_snapshot_count = 1205;
constexpr std::size_t guard_elements = 16;
constexpr std::uint32_t logit_guard_bits = 0x4b71abcd;
constexpr std::int32_t input_guard = -123456789;
constexpr std::array<std::int32_t, teacher_rows> teacher_ids = [] {
    std::array<std::int32_t, teacher_rows> ids{};
    ids[0] = 248044;
    for (std::size_t i = 1; i < ids.size(); ++i) ids[i] = 99 + static_cast<std::int32_t>(i);
    return ids;
}();
constexpr std::array<std::int32_t, continuation_rows> continuation_ids = [] {
    std::array<std::int32_t, continuation_rows> ids{};
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = 1123 + static_cast<std::int32_t>(i);
    return ids;
}();
constexpr std::array<std::int32_t, 2> capacity_ids{1153, 1154};
constexpr bool exact_source_ids() {
    if (teacher_ids.size() + continuation_ids.size() != 1056 || teacher_ids[0] != 248044) return false;
    // One BOS plus every integer 100..1154, split after 1122: check every
    // position, not just the two ranges' endpoints or their array extents.
    for (std::size_t position = 1; position < capacity; ++position) {
        const auto token = position < teacher_rows ? teacher_ids[position] : continuation_ids[position - teacher_rows];
        if (token != 99 + static_cast<std::int32_t>(position) || token < 0 || token >= static_cast<std::int32_t>(vocabulary))
            return false;
    }
    return true;
}
static_assert(sizeof(float) == 4 && sizeof(double) == 8);
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
static_assert(std::string_view(CORE_REVISION).size() == 40);
static_assert(teacher_ids.size() == 1 + (1122 - 100 + 1) && teacher_ids[1] == 100 && teacher_ids.back() == 1122);
static_assert(continuation_ids.size() == 1154 - 1123 + 1 && continuation_ids.front() == 1123 && continuation_ids.back() == 1154);
static_assert(teacher_rows + continuation_rows == capacity && 7 * 129 + 121 == teacher_rows);
static_assert(exact_source_ids() && 1 + (1154 - 100 + 1) == capacity);
static_assert(ram_expert_payload == 68262297600ULL && reference_bytes == 1048903680ULL);
static_assert(host_logit_bytes == 1017118720ULL && pinned_handoff_bytes == 41943040ULL);
static_assert(pinned_expert_staging_bytes == 183500800ULL && workspace_floor == 1543503872ULL);
static_assert(expected_windows[0] + expected_windows[1] + expected_windows[2] + expected_windows[3] == window_count);
// source + loaded + four (reset, windows, phase) + final_reset + complete.
static_assert(success_record_count == window_count + 12);
static_assert(memory_snapshot_count == 1 + 4 + window_count + 2 * 15 + 1);

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_revision() {
    const std::string_view revision(CORE_REVISION);
    for (std::size_t i = 0; i < revision.size(); i += 8) {
        std::uint32_t value = 0;
        const char* begin = revision.data() + i;
        const auto result = std::from_chars(begin, begin + 8, value, 16);
        require(result.ec == std::errc{} && result.ptr == begin + 8, "CORE_REVISION is not 40 hexadecimal digits");
    }
}

std::uint64_t check_model_file(const std::string& path) {
    require(!path.empty() && std::filesystem::is_regular_file(path), "model is not a regular file");
    const auto bytes = std::filesystem::file_size(path);
    require(bytes > 0 && bytes <= static_cast<std::uintmax_t>(std::numeric_limits<std::int64_t>::max()),
            "invalid model file size");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file.is_open() && static_cast<bool>(file), "cannot open model file");
    const auto end = file.tellg();
    require(end >= 0 && static_cast<std::uintmax_t>(end) == bytes, "model file size changed or seek failed");
    file.close();
    require(!file.fail(), "model file close failed");
    return static_cast<std::uint64_t>(bytes);
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

void json_number(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value; else out << "null";
}

template<class T, std::size_t N>
void json_array(std::ostream& out, const std::array<T, N>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
    out << ']';
}

void format(std::ostringstream& out) {
    out.imbue(std::locale::classic());
    out << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10);
}

// JSONL protocol 2 (bounded head logits), all kinds prefixed "prefill_wide_":
// source: provenance/config, exact IDs/schedules, frozen gate and evidence scope.
// memory: event=loaded/reset/final_reset, nullable phase_index, stats/route_stats, memory.
// window: phase_index/name, chronological window_index/offset/rows, segment,
//         completed_call/time, stats_before/after, expected_routes, route_delta,
//         retained_n1_route_delta, route_stats_before/after, expert_groups_gt128_delta,
//         observed max/proof, errors, memory_snapshot, passed/failure_stage/failure.
// phase: window/value counts, teacher/final stats/route_stats, observed maximum
//        and threshold count, invalid_proofs (including preserved route_stats), passed.
// complete: exact totals, memory counts/minimum free VRAM, cleanup/evidence flags.
// failure: passed=false, construction/cleanup flags, bounded error (no complete).
// Nested stats/route_stats/memory enumerate every current public field. Routes give actual
// hits/misses/upload bytes plus per-uploaded-group payload bounds; errors give
// finite/nonfinite counts, full-logit gate/bit/argmax diagnostics and coordinates.
// The footer requires observed max>128 AND positive threshold count in single1024;
// logical group cardinality does not prove a physical 128-column kernel/stage.
// No per-element dump. At most 1181 success lines; exceptions add one bounded
// failure line. Emit records chronologically only after the exclusive Session
// and both guarded float allocations have unwound, including on numeric failure.
struct Records {
    std::array<std::string, success_record_count + 2> lines{};
    std::size_t count = 0, emitted = 0;

    void add(std::ostringstream& out) {
        require(count < lines.size() && static_cast<bool>(out), "JSONL record bound/formatting failed");
        lines[count++] = out.str() + '\n';
    }
    void flush() {
        while (emitted < count) {
            std::cout << lines[emitted++];
            std::cout.flush();
            require(static_cast<bool>(std::cout), "JSONL stdout write failed");
        }
    }
};

void json_stats(std::ostream& out, const qwen::SessionStats& stats) {
    out << "{\"consumed_tokens\":" << stats.consumed_tokens << ",\"expert_hits\":" << stats.expert_hits
        << ",\"expert_misses\":" << stats.expert_misses << ",\"expert_upload_bytes\":" << stats.expert_upload_bytes
        << ",\"last_completed_ms\":";
    json_number(out, stats.last_completed_ms);
    out << ",\"last_completed_ms_bits\":" << std::bit_cast<std::uint64_t>(stats.last_completed_ms) << '}';
}

void check_same_stats(const qwen::SessionStats& a, const qwen::SessionStats& b, const std::string& where) {
    require(a.consumed_tokens == b.consumed_tokens && a.expert_hits == b.expert_hits && a.expert_misses == b.expert_misses &&
            a.expert_upload_bytes == b.expert_upload_bytes &&
            std::bit_cast<std::uint64_t>(a.last_completed_ms) == std::bit_cast<std::uint64_t>(b.last_completed_ms),
            where + ": public statistics changed");
}

void json_route_stats(std::ostream& out, const qwen::SessionRouteStats& stats) {
    out << "{\"last_max_expert_group_assignments\":" << stats.last_max_expert_group_assignments
        << ",\"expert_groups_gt128\":" << stats.expert_groups_gt128 << '}';
}

bool same_route_stats(const qwen::SessionRouteStats& a, const qwen::SessionRouteStats& b) {
    return a.last_max_expert_group_assignments == b.last_max_expert_group_assignments &&
           a.expert_groups_gt128 == b.expert_groups_gt128;
}

void check_same_route_stats(const qwen::SessionRouteStats& a, const qwen::SessionRouteStats& b, const std::string& where) {
    require(same_route_stats(a, b), where + ": completed-call routing diagnostics changed");
}

std::uint64_t check_completed_routes(const qwen::SessionRouteStats& after, const qwen::SessionRouteStats& before,
                                     std::size_t rows, const std::string& where) {
    require(after.last_max_expert_group_assignments >= 1 && after.last_max_expert_group_assignments <= rows &&
            after.expert_groups_gt128 >= before.expert_groups_gt128, where + ": completed-call group bounds");
    const auto delta = after.expert_groups_gt128 - before.expert_groups_gt128;
    require(delta <= routes_per_row * rows / 129 &&
            (delta > 0) == (after.last_max_expert_group_assignments > 128), where + ": threshold count/max disagree");
    if (rows <= 128) require(delta == 0, where + ": N<=128 cannot contain group>128");
    if (rows == 1) require(after.last_max_expert_group_assignments == 1, where + ": N1 group maximum is not one");
    return delta;
}

struct Routes {
    std::uint64_t hits = 0, misses = 0, upload_bytes = 0;
};

bool monotone_routes(const qwen::SessionStats& a, const qwen::SessionStats& b) {
    return a.expert_hits >= b.expert_hits && a.expert_misses >= b.expert_misses &&
           a.expert_upload_bytes >= b.expert_upload_bytes;
}

Routes route_delta(const qwen::SessionStats& a, const qwen::SessionStats& b) {
    require(monotone_routes(a, b), "expert counters decreased");
    return {a.expert_hits - b.expert_hits, a.expert_misses - b.expert_misses, a.expert_upload_bytes - b.expert_upload_bytes};
}

void json_routes(std::ostream& out, const Routes& routes) {
    out << "{\"hits\":" << routes.hits << ",\"misses\":" << routes.misses << ",\"upload_bytes\":" << routes.upload_bytes
        << ",\"miss_payload_min_bytes\":";
    // A failed counter gate can still reach this diagnostic serializer. Do not
    // wrap the payload bounds if the failed call reported impossible counters.
    if (routes.misses <= std::numeric_limits<std::uint64_t>::max() / q4_0_expert_bytes)
        out << routes.misses * q4_0_expert_bytes;
    else out << "null";
    out << ",\"miss_payload_max_bytes\":";
    if (routes.misses <= std::numeric_limits<std::uint64_t>::max() / q4_1_expert_bytes)
        out << routes.misses * q4_1_expert_bytes;
    else out << "null";
    out << '}';
}

void check_completed(const qwen::SessionStats& after, const qwen::SessionStats& before,
                     std::size_t offset, std::size_t rows, const std::string& where) {
    require(before.consumed_tokens == offset && after.consumed_tokens == offset + rows, where + ": consumed increment is not N");
    require(std::isfinite(after.last_completed_ms) && after.last_completed_ms > 0, where + ": completed time is not positive finite");
    const auto delta = route_delta(after, before);
    const auto routes = routes_per_row * rows;
    require(delta.hits <= routes && delta.misses <= routes && delta.hits + delta.misses == routes,
            where + ": hits+misses is not 480*N");
    require(delta.upload_bytes >= delta.misses * q4_0_expert_bytes && delta.upload_bytes <= delta.misses * q4_1_expert_bytes,
            where + ": upload bytes outside unchanged per-miss-group payload range");
}

void check_exact(std::span<const float> actual, std::span<const float> saved, const std::string& where) {
    require(actual.size() == saved.size(), where + ": preservation span extent");
    for (std::size_t i = 0; i < actual.size(); ++i)
        if (std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(saved[i]))
            throw std::runtime_error(where + ": active logit bits changed at element " + std::to_string(i));
}

void check_memory(const qwen::SessionMemory& m, const std::string& where) {
    require(m.capacity == capacity && m.expert_slots == expert_slots && m.ownership_verified, where + ": ownership/config ledger");
    require(m.ram_expert_payload == ram_expert_payload && m.ram_expert_capacity >= ram_expert_payload &&
            m.expert_payload_reads == payload_reads && m.expert_payload_bytes_read == ram_expert_payload,
            where + ": expert RAM/read accounting");
    require(m.host_embedding_capacity > 0 && m.host_logit_capacity >= host_logit_bytes &&
            m.pinned_handoff == pinned_handoff_bytes && m.pinned_expert_staging == pinned_expert_staging_bytes,
            where + ": host capacities or pinned handoff/staging geometry");
    for (std::size_t i = 0; i < m.devices.size(); ++i) {
        const auto& d = m.devices[i];
        require(d.device == static_cast<int>(i) && d.first_layer == static_cast<int>(i) * 24 &&
                d.last_layer == static_cast<int>(i) * 24 + 23 && d.gdn_layers == 18 && d.qsa_layers == 6,
                where + ": static 24/24 owners");
        require(d.expert_slots == slot_bytes[i] && d.qsa_kv == qsa_kv_bytes && d.qsa_index == qsa_index_bytes &&
                d.gdn_state == gdn_state_bytes && d.ple_state == (i == 0 ? 92160ULL * sizeof(float) : 0),
                where + ": expert/Q4 KV/index/GDN/PLE categories");
        std::uint64_t sum = 0;
        for (const auto bytes : {d.weights, d.expert_slots, d.qsa_kv, d.qsa_index, d.gdn_state, d.ple_state, d.workspace}) {
            require(bytes <= std::numeric_limits<std::uint64_t>::max() - sum, where + ": category sum overflow");
            sum += bytes;
        }
        require(d.head_logits_bytes == 128ULL * vocabulary * sizeof(float) && d.weights > 0 && d.workspace >= workspace_floor && d.owned_bytes == sum && d.owned_buffers > 0 &&
                d.owned_peak_bytes >= d.owned_bytes, where + ": allocation category/count/peak ledger");
        require(d.total_vram > 0 && d.free_vram > 0 && d.free_vram <= d.total_vram &&
                d.owned_bytes <= d.total_vram - d.free_vram, where + ": owned bytes/observed VRAM");
    }
}

void check_steady_memory(const qwen::SessionMemory& a, const qwen::SessionMemory& b, const std::string& where) {
    check_memory(a, where);
    require(a.capacity == b.capacity && a.expert_slots == b.expert_slots && a.ownership_verified == b.ownership_verified &&
            a.ram_expert_capacity == b.ram_expert_capacity && a.ram_expert_payload == b.ram_expert_payload &&
            a.host_embedding_capacity == b.host_embedding_capacity && a.host_logit_capacity == b.host_logit_capacity &&
            a.pinned_handoff == b.pinned_handoff && a.pinned_expert_staging == b.pinned_expert_staging &&
            a.expert_payload_reads == b.expert_payload_reads && a.expert_payload_bytes_read == b.expert_payload_bytes_read,
            where + ": host/pinned/payload-read ledger changed");
    for (std::size_t i = 0; i < a.devices.size(); ++i) {
        const auto& x = a.devices[i];
        const auto& y = b.devices[i];
        require(x.device == y.device && x.first_layer == y.first_layer && x.last_layer == y.last_layer &&
                x.gdn_layers == y.gdn_layers && x.qsa_layers == y.qsa_layers && x.weights == y.weights &&
                x.expert_slots == y.expert_slots && x.qsa_kv == y.qsa_kv && x.qsa_index == y.qsa_index &&
                x.gdn_state == y.gdn_state && x.ple_state == y.ple_state && x.workspace == y.workspace &&
                x.head_logits_bytes == y.head_logits_bytes &&
                x.owned_bytes == y.owned_bytes && x.owned_peak_bytes == y.owned_peak_bytes &&
                x.owned_buffers == y.owned_buffers && x.total_vram == y.total_vram, where + ": device ledger changed");
        // Free VRAM includes unowned HIP/context/rocBLAS state; equality is not a ledger gate.
    }
}

void json_memory(std::ostream& out, const qwen::SessionMemory& m) {
    out << "{\"capacity\":" << m.capacity << ",\"expert_slots\":" << m.expert_slots << ",\"ownership_verified\":" << m.ownership_verified
        << ",\"ram_expert_capacity\":" << m.ram_expert_capacity << ",\"ram_expert_payload\":" << m.ram_expert_payload
        << ",\"host_embedding_capacity\":" << m.host_embedding_capacity << ",\"host_logit_capacity\":" << m.host_logit_capacity
        << ",\"pinned_handoff\":" << m.pinned_handoff << ",\"pinned_expert_staging\":" << m.pinned_expert_staging
        << ",\"expert_payload_reads\":" << m.expert_payload_reads << ",\"expert_payload_bytes_read\":" << m.expert_payload_bytes_read
        << ",\"devices\":[";
    for (std::size_t i = 0; i < m.devices.size(); ++i) {
        if (i) out << ',';
        const auto& d = m.devices[i];
        out << "{\"device\":" << d.device << ",\"first_layer\":" << d.first_layer << ",\"last_layer\":" << d.last_layer
            << ",\"gdn_layers\":" << d.gdn_layers << ",\"qsa_layers\":" << d.qsa_layers << ",\"weights\":" << d.weights
            << ",\"expert_slots\":" << d.expert_slots << ",\"qsa_kv\":" << d.qsa_kv << ",\"qsa_index\":" << d.qsa_index
            << ",\"gdn_state\":" << d.gdn_state << ",\"ple_state\":" << d.ple_state << ",\"workspace\":" << d.workspace << ",\"head_logits_bytes\":" << d.head_logits_bytes
            << ",\"owned_bytes\":" << d.owned_bytes << ",\"owned_peak_bytes\":" << d.owned_peak_bytes
            << ",\"owned_buffers\":" << d.owned_buffers << ",\"total_vram\":" << d.total_vram << ",\"free_vram\":" << d.free_vram << '}';
    }
    out << "]}";
}

class GuardedLogits {
public:
    explicit GuardedLogits(std::size_t values) : values_(values), storage_(checked_elements(values)), capacity_(storage_.capacity()) {
        const float guard = std::bit_cast<float>(logit_guard_bits);
        std::fill_n(storage_.begin(), guard_elements, guard);
        std::fill_n(storage_.end() - guard_elements, guard_elements, guard);
    }
    std::span<float> values() { return std::span<float>(storage_).subspan(guard_elements, values_); }
    std::span<const float> values() const { return std::span<const float>(storage_).subspan(guard_elements, values_); }
    void check(const std::string& where) const {
        require(storage_.size() == values_ + 2 * guard_elements && storage_.capacity() == capacity_, where + ": fixture capacity changed");
        for (std::size_t i = 0; i < guard_elements; ++i)
            require(std::bit_cast<std::uint32_t>(storage_[i]) == logit_guard_bits &&
                    std::bit_cast<std::uint32_t>(storage_[guard_elements + values_ + i]) == logit_guard_bits,
                    where + ": fixture logit guard changed");
    }
private:
    static std::size_t checked_elements(std::size_t values) {
        const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
        require(values > 0 && values <= limit - 2 * guard_elements, "fixture logit allocation overflow");
        return values + 2 * guard_elements;
    }
    std::size_t values_;
    std::vector<float> storage_;
    std::size_t capacity_;
};

struct References {
    // Only two large host allocations, both guarded and fixed before the Session:
    // retained [1056][248320] N1 logits and a [1024][248320] preservation snapshot.
    GuardedLogits logits{capacity * vocabulary}, saved{max_batch_tokens * vocabulary};
    std::array<qwen::SessionStats, capacity> stats{};
    // One guarded input workspace serves oversized and both late-ID probes.
    std::array<std::int32_t, max_batch_tokens + 1 + 2 * guard_elements> invalid_ids{};

    References() {
        invalid_ids.fill(100);
        std::fill_n(invalid_ids.begin(), guard_elements, input_guard);
        std::fill_n(invalid_ids.end() - guard_elements, guard_elements, input_guard);
        for (const auto token : teacher_ids) require(token >= 0 && token < static_cast<std::int32_t>(vocabulary), "teacher ID preflight");
        for (const auto token : continuation_ids) require(token >= 0 && token < static_cast<std::int32_t>(vocabulary), "continuation ID preflight");
        check("preallocated workspaces");
    }
    void check(const std::string& where) const {
        logits.check(where + ": reference");
        saved.check(where + ": preservation");
        for (std::size_t i = 0; i < guard_elements; ++i)
            require(invalid_ids[i] == input_guard && invalid_ids[invalid_ids.size() - guard_elements + i] == input_guard,
                    where + ": fixture input guard changed");
    }
    std::span<float> slice(std::size_t offset, std::size_t rows) {
        require(rows > 0 && offset < capacity && rows <= capacity - offset, "reference slice outside timeline");
        return logits.values().subspan(offset * vocabulary, rows * vocabulary);
    }
    std::span<const float> snapshot(std::span<const float> active) {
        require(active.size() <= saved.values().size(), "active snapshot exceeds preallocated capacity");
        std::copy(active.begin(), active.end(), saved.values().begin());
        return saved.values().first(active.size());
    }
    std::span<const std::int32_t> invalid_window(std::size_t rows, std::int32_t last) {
        require(rows > 0 && rows <= max_batch_tokens + 1, "invalid input workspace extent");
        auto ids = std::span<std::int32_t>(invalid_ids).subspan(guard_elements, rows);
        std::fill(ids.begin(), ids.end(), 100);
        ids.back() = last;
        return ids;
    }
    Routes routes(std::size_t offset, std::size_t rows) const {
        require(rows > 0 && offset < capacity && rows <= capacity - offset, "reference routes outside timeline");
        return route_delta(stats[offset + rows - 1], offset == 0 ? qwen::SessionStats{} : stats[offset - 1]);
    }
};

struct MemoryAudit {
    qwen::SessionMemory loaded{}, last{};
    std::array<std::uint64_t, 2> minimum_free{};
    std::uint64_t snapshots = 0, preserved_values = 0;

    qwen::SessionMemory observe(qwen::Session& session, References& reference, std::span<const float> active,
                                const std::string& where) {
        reference.check(where + " before memory()");
        const auto saved = reference.snapshot(active);
        const auto before = session.stats();
        const auto routes_before = session.route_stats();
        const auto m = session.memory();
        check_same_stats(session.stats(), before, where + ": memory()");
        check_same_route_stats(session.route_stats(), routes_before, where + ": memory()");
        check_exact(active, saved, where + ": memory()");
        reference.check(where + " after memory()");
        if (snapshots == 0) {
            check_memory(m, where);
            loaded = m;
            minimum_free = {m.devices[0].free_vram, m.devices[1].free_vram};
        } else check_steady_memory(m, loaded, where);
        for (std::size_t i = 0; i < minimum_free.size(); ++i)
            minimum_free[i] = std::min(minimum_free[i], m.devices[i].free_vram);
        last = m;
        ++snapshots;
        preserved_values += active.size();
        return m;
    }
};

struct Errors {
    std::uint64_t values = 0, finite_values = 0, nonfinite_actual = 0, nonfinite_reference = 0;
    std::uint64_t compared = 0, finite_pairs = 0, violations = 0, bit_mismatches = 0;
    std::size_t maxabs_index = 0, maxratio_index = 0, first_violation = 0;
    std::uint64_t argmax_agree = 0, argmax_rows = 0;
    double actual_maxabs = 0, actual_rms = 0, maxabs = 0, rms = 0, maxboundratio = 0;
};

Errors inspect_logits(std::span<const float> actual, std::span<const float> expected) {
    require(!actual.empty() && actual.size() % vocabulary == 0 &&
            (expected.empty() || actual.size() == expected.size()), "logit inspection extent");
    Errors e;
    e.values = actual.size();
    e.compared = expected.size();
    double actual_squares = 0, error_squares = 0;
    for (std::size_t row = 0; row < actual.size() / vocabulary; ++row) {
        std::size_t actual_argmax = 0, reference_argmax = 0;
        bool finite_row = true;
        for (std::size_t column = 0; column < vocabulary; ++column) {
            const auto i = row * vocabulary + column;
            const double a = actual[i];
            if (std::isfinite(a)) {
                ++e.finite_values;
                e.actual_maxabs = std::max(e.actual_maxabs, std::abs(a));
                actual_squares += a * a;
            } else { ++e.nonfinite_actual; finite_row = false; }
            if (expected.empty()) continue;
            const double b = expected[i];
            if (!std::isfinite(b)) { ++e.nonfinite_reference; finite_row = false; }
            e.bit_mismatches += std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]);
            if (!std::isfinite(a) || !std::isfinite(b)) {
                if (e.violations++ == 0) e.first_violation = i;
                continue;
            }
            ++e.finite_pairs;
            const double error = std::abs(a - b);
            const double bound = absolute_gate + relative_gate * std::abs(b);
            const double ratio = error / bound;
            error_squares += error * error;
            if (e.finite_pairs == 1 || error > e.maxabs) { e.maxabs = error; e.maxabs_index = i; }
            if (e.finite_pairs == 1 || ratio > e.maxboundratio) { e.maxboundratio = ratio; e.maxratio_index = i; }
            if (error > bound) { if (e.violations++ == 0) e.first_violation = i; }
            if (actual[i] > actual[row * vocabulary + actual_argmax]) actual_argmax = column;
            if (expected[i] > expected[row * vocabulary + reference_argmax]) reference_argmax = column;
        }
        if (!expected.empty() && finite_row) {
            ++e.argmax_rows;
            e.argmax_agree += actual_argmax == reference_argmax;
        }
    }
    if (e.finite_values) e.actual_rms = std::sqrt(actual_squares / static_cast<double>(e.finite_values));
    if (e.finite_pairs) e.rms = std::sqrt(error_squares / static_cast<double>(e.finite_pairs));
    return e;
}

void json_errors(std::ostream& out, const Errors& e, std::size_t offset) {
    out << "{\"values\":" << e.values << ",\"finite_values\":" << e.finite_values
        << ",\"nonfinite_actual\":" << e.nonfinite_actual << ",\"nonfinite_reference\":" << e.nonfinite_reference
        << ",\"actual_maxabs\":" << e.actual_maxabs << ",\"actual_rms\":" << e.actual_rms
        << ",\"compared_values\":" << e.compared << ",\"finite_pairs\":" << e.finite_pairs
        << ",\"metric_scope\":\"finite_pairs_only_nonfinite_pairs_also_violate\",\"maxabs\":";
    if (e.finite_pairs) out << e.maxabs; else out << "null";
    out << ",\"rms\":"; if (e.finite_pairs) out << e.rms; else out << "null";
    out << ",\"maxboundratio\":"; if (e.finite_pairs) out << e.maxboundratio; else out << "null";
    out << ",\"violations\":" << e.violations << ",\"bit_mismatches\":" << e.bit_mismatches
        << ",\"bit_equality_required\":false,\"argmax_agree_rows\":" << e.argmax_agree
        << ",\"argmax_compared_rows\":" << e.argmax_rows << ",\"argmax_equality_required\":false,\"maxabs_coordinate\":";
    if (e.finite_pairs) out << "{\"position\":" << offset + e.maxabs_index / vocabulary << ",\"vocabulary_index\":" << e.maxabs_index % vocabulary << '}';
    else out << "null";
    out << ",\"maxboundratio_coordinate\":";
    if (e.finite_pairs) out << "{\"position\":" << offset + e.maxratio_index / vocabulary << ",\"vocabulary_index\":" << e.maxratio_index % vocabulary << '}';
    else out << "null";
    out << ",\"first_violation_coordinate\":";
    if (e.violations) out << "{\"position\":" << offset + e.first_violation / vocabulary << ",\"vocabulary_index\":" << e.first_violation % vocabulary << '}';
    else out << "null";
    out << '}';
}

void source_record(Records& records, const std::string& model, std::uint64_t model_bytes) {
    std::ostringstream out;
    format(out);
    out << "{\"kind\":\"prefill_wide_source\",\"protocol\":2,\"revision\":";
    json_string(out, CORE_REVISION);
    out << ",\"dirty\":" << (CORE_DIRTY != 0) << ",\"model\":";
    json_string(out, model);
    out << ",\"model_bytes\":" << model_bytes
        << ",\"model_source\":\"caller_supplied_GGUF_path_no_checksum_attestation\",\"runtime\":\"own_48_layer_HIP\""
           ",\"config\":{\"capacity\":1056,\"expert_slots\":1,\"max_batch_tokens\":1024,\"trace\":false},\"teacher_ids\":";
    json_array(out, teacher_ids);
    out << ",\"continuation_ids\":";
    json_array(out, continuation_ids);
    out << ",\"teacher_id_count\":" << teacher_ids.size() << ",\"continuation_id_count\":" << continuation_ids.size()
        << ",\"source_id_count\":" << teacher_ids.size() + continuation_ids.size()
        << ",\"all_source_ids_exact_compile_time_checked\":true";
    out << ",\"expected_windows_per_phase\":"; json_array(out, expected_windows);
    out << ",\"phase_order\":[";
    for (std::size_t i = 0; i < phase_names.size(); ++i) { if (i) out << ','; json_string(out, phase_names[i]); }
    out << "],\"teacher_schedules\":[\"1024_N1\",\"8_N128\",\"7_N129_then_N121\",\"1_N1024\"]"
           ",\"continuation_schedule\":\"32_N1_each_phase\",\"occupied_prefix_scope\":\"later_chunk128_and_chunk129_calls_have_real_prior_history\""
           ",\"gate\":{\"absolute\":" << absolute_gate << ",\"relative\":" << relative_gate
        << ",\"formula\":\"abs(actual-reference)<=.02+.002*abs(reference)\",\"all_finite_required\":true,\"allowed_violations\":0"
           ",\"bit_equality_required\":false,\"argmax_equality_required\":false,\"cli_adjustable\":false}"
           ",\"reference_scope\":\"all_1056_full_vocabulary_rows_retained_sequential_N1_same_Session\""
           ",\"weight_values\":\"unchanged_loaded_GGUF\",\"weight_precision\":\"unchanged_loaded_tensor_types\",\"kv\":\"Q4_0_K_and_V\""
           ",\"sampling\":\"teacher_forced\",\"session_instances\":1,\"reset_clears_logical_state_and_statistics\":true"
           ",\"reset_retains_expert_cache\":true,\"cold_cache_equality_claim\":false,\"speedup_claim\":false"
           ",\"independent_HF_reference\":false,\"MTP_qualification_claim\":false,\"qualification_4K_16K_claim\":false,\"R4_complete_claim\":false"
           ",\"timing_scope\":\"correctness_only_completed_call_wall_time_excludes_comparison_memory_reset_and_rejection_not_PP_throughput\""
           ",\"counter_scope\":\"480_assignments_per_row_hits_include_within_call_group_reuse_misses_count_uploaded_payloads\""
           ",\"uploaded_group_payload_bytes\":{\"min\":" << q4_0_expert_bytes << ",\"max\":" << q4_1_expert_bytes << '}'
         << ",\"individual_group_counts_observable\":false,\"completed_call_group_diagnostics_observable\":true"
            ",\"max_expert_group_count\":null,\"group_gt128_proven\":false"
            ",\"group_gt128_evidence\":\"requires_observed_completed_call_max_and_gt128_counter_not_chunk_length\""
            ",\"group_diagnostic_scope\":\"logical_assignments_to_one_expert_in_one_layer_max_across48_layers_last_successful_call_cumulative_groups_strictly_gt128_reset_zero\""
           ",\"read_counter_scope\":\"constructor_expert_payload_reads_not_physical_SSD_syscall_trace\""
           ",\"geometry\":{\"logical_chunk_limit\":1024,\"projection_expert_chronological_microtile_limit\":8,\"gdn_slice_limit\":128"
           ",\"expert_band\":16,\"stages_per_device\":2,\"pinned_expert_staging_bytes\":" << pinned_expert_staging_bytes
        << ",\"pinned_handoff_bytes\":" << pinned_handoff_bytes << ",\"host_logit_min_bytes\":" << host_logit_bytes
        << ",\"workspace_aggregate_min_bytes_per_device\":" << workspace_floor
        << ",\"individual_buffer_capacities_observable\":false,\"individual_buffer_capacity_qualification\":\"head_logits_reported_other_buffers_aggregate_floor\""
           ",\"component_128_staging_repack_qualified\":false,\"component_128_staging_repack_evidence\":\"not_inferred_from_N1024_or_aggregate_memory_floor\""
           ",\"free_vram_equality_required\":false,\"memory_scope\":\"Session_owned_Buffer_ledger_and_reported_host_capacities_excludes_allocator_metadata\"}"
           ",\"preallocated_reference_bytes\":" << reference_bytes << ",\"preallocated_preservation_snapshot_bytes\":" << host_logit_bytes
        << ",\"fixture_guard_elements_each_end\":" << guard_elements << ",\"fixture_float_guard_bytes\":" << 4 * guard_elements * sizeof(float)
        << ",\"preallocated_guarded_input_elements\":" << max_batch_tokens + 1 + 2 * guard_elements
        << ",\"expected_success_windows\":" << window_count << ",\"expected_success_records\":" << success_record_count
        << ",\"jsonl_after_session_cleanup\":true}";
    records.add(out);
}

void memory_record(Records& records, std::string_view event, std::size_t phase,
                   const qwen::SessionStats& stats, const qwen::SessionRouteStats& route_stats, const qwen::SessionMemory& memory) {
    std::ostringstream out;
    format(out);
    out << "{\"kind\":\"prefill_wide_memory\",\"protocol\":2,\"event\":"; json_string(out, event);
    out << ",\"phase_index\":";
    if (phase < phase_names.size()) out << phase; else out << "null";
    out << ",\"stats\":"; json_stats(out, stats);
    out << ",\"route_stats\":"; json_route_stats(out, route_stats);
    out << ",\"memory\":"; json_memory(out, memory);
    out << ",\"passed\":true}";
    records.add(out);
}

struct InvalidProof {
    std::string_view name;
    std::size_t rows = 0, prior_rows = 0, continuation_offset = 0, continuation_rows = 0;
    qwen::SessionStats stats{};
    qwen::SessionRouteStats route_stats{};
    bool input_fits_remaining_capacity = false, continued = false;
};

struct Totals {
    std::uint64_t windows = 0, finite_values = 0, compared_values = 0, bit_mismatches = 0, violations = 0;
    std::uint64_t invalid_windows = 0, rejection_preserved_values = 0, completed_phases = 0;
    std::uint64_t max_group = 0, groups_gt128 = 0, single1024_max_group = 0, single1024_groups_gt128 = 0;
    double completed_call_wall_ms = 0;
};

void window_record(Records& records, std::size_t phase, std::size_t window, std::size_t offset, std::size_t rows,
                    bool completed, double wall_ms, const qwen::SessionStats& before, const qwen::SessionStats& after,
                    const qwen::SessionRouteStats& routes_before, const qwen::SessionRouteStats& routes_after,
                   const Errors* errors, const Routes* reference_routes, const qwen::SessionMemory* memory,
                   bool passed, std::string_view failure_stage, std::string_view failure) {
    std::ostringstream out;
    format(out);
    out << "{\"kind\":\"prefill_wide_window\",\"protocol\":2,\"phase_index\":" << phase << ",\"phase\":";
    json_string(out, phase_names[phase]);
    out << ",\"window_index\":" << window << ",\"offset\":" << offset << ",\"rows\":" << rows
        << ",\"segment\":\"" << (offset < teacher_rows ? "teacher" : "continuation") << "\",\"reference\":" << (phase == 0)
        << ",\"completed_call\":" << completed << ",\"correctness_completed_call_wall_ms\":";
    if (completed) json_number(out, wall_ms); else out << "null";
    out << ",\"stats_before\":"; json_stats(out, before);
    out << ",\"stats_after\":"; json_stats(out, after);
    out << ",\"expected_routes\":" << routes_per_row * rows << ",\"route_delta\":";
    if (monotone_routes(after, before)) json_routes(out, route_delta(after, before)); else out << "null";
    out << ",\"retained_n1_route_delta\":";
    if (reference_routes) json_routes(out, *reference_routes); else out << "null";
    out << ",\"route_stats_before\":"; json_route_stats(out, routes_before);
    out << ",\"route_stats_after\":"; json_route_stats(out, routes_after);
    out << ",\"expert_groups_gt128_delta\":";
    if (routes_after.expert_groups_gt128 >= routes_before.expert_groups_gt128)
        out << routes_after.expert_groups_gt128 - routes_before.expert_groups_gt128;
    else out << "null";
    out << ",\"max_expert_group_count\":" << routes_after.last_max_expert_group_assignments
        << ",\"group_gt128_proven\":" << (completed && routes_after.last_max_expert_group_assignments > 128 &&
            routes_after.expert_groups_gt128 > routes_before.expert_groups_gt128)
        << ",\"errors\":";
    if (errors) json_errors(out, *errors, offset); else out << "null";
    out << ",\"memory_snapshot\":";
    if (memory) json_memory(out, *memory); else out << "null";
    out << ",\"memory_stats_and_full_active_span_preserved\":" << (memory != nullptr)
        << ",\"first_input_n1_smoke\":" << (phase == 0 && offset == 0) << ",\"passed\":" << passed << ",\"failure_stage\":";
    if (passed) out << "null"; else json_string(out, failure_stage);
    out << ",\"failure\":";
    if (passed) out << "null"; else json_string(out, failure.substr(0, 4096));
    out << '}';
    records.add(out);
}

void expect_rejection(qwen::Session& session, References& reference, MemoryAudit& audit,
                      std::array<InvalidProof, 5>& proofs, std::size_t& count, Totals& totals,
                      std::span<const std::int32_t> ids, std::span<const float> prior,
                      std::string_view name, const std::string& where) {
    require(!prior.empty() && count < proofs.size(), where + ": rejection fixture bound/prior span");
    const auto saved = reference.snapshot(prior);
    const auto before = session.stats();
    const auto routes_before = session.route_stats();
    const auto memory_before = audit.observe(session, reference, prior, where + " before rejection");
    bool rejected = false;
    try { static_cast<void>(session.step_batch(ids)); }
    catch (const std::invalid_argument&) { rejected = true; }
    catch (const std::exception& error) { throw std::runtime_error(where + ": expected invalid_argument, got " + error.what()); }
    require(rejected, where + ": invalid input accepted");
    check_same_stats(session.stats(), before, where + ": rejected window");
    check_same_route_stats(session.route_stats(), routes_before, where + ": rejected window");
    check_exact(prior, saved, where + ": rejected full active span");
    reference.check(where + ": rejected guards");
    const auto memory_after = audit.observe(session, reference, prior, where + " after rejection");
    check_steady_memory(memory_after, memory_before, where + ": rejected ledger");
    proofs[count++] = {name, ids.size(), prior.size() / vocabulary, 0, 0, before, routes_before,
                       ids.size() <= capacity - before.consumed_tokens, false};
    ++totals.invalid_windows;
    totals.rejection_preserved_values += prior.size();
}

void phase_record(Records& records, std::size_t phase, std::size_t windows, const Totals& before, const Totals& after,
                   const qwen::SessionStats& teacher_stats, const qwen::SessionStats& final_stats,
                   const qwen::SessionRouteStats& teacher_routes, const qwen::SessionRouteStats& final_routes,
                   std::uint64_t phase_max_group,
                  const std::array<InvalidProof, 5>& proofs, std::size_t proof_count) {
    std::ostringstream out;
    format(out);
    out << "{\"kind\":\"prefill_wide_phase\",\"protocol\":2,\"phase_index\":" << phase << ",\"phase\":";
    json_string(out, phase_names[phase]);
    out << ",\"windows\":" << windows << ",\"teacher_rows\":1024,\"continuation_rows\":32,\"expected_routes\":" << routes_per_row * capacity
        << ",\"finite_logit_values\":" << after.finite_values - before.finite_values
        << ",\"compared_logit_values\":" << after.compared_values - before.compared_values
        << ",\"bit_mismatches_diagnostic\":" << after.bit_mismatches - before.bit_mismatches
        << ",\"violations\":" << after.violations - before.violations
        << ",\"correctness_completed_call_wall_ms_sum\":" << after.completed_call_wall_ms - before.completed_call_wall_ms
        << ",\"timing_includes_rejection_reset_comparison_memory\":false,\"teacher_stats\":";
    json_stats(out, teacher_stats);
    out << ",\"final_stats\":"; json_stats(out, final_stats);
    out << ",\"teacher_route_stats\":"; json_route_stats(out, teacher_routes);
    out << ",\"final_route_stats\":"; json_route_stats(out, final_routes);
    out << ",\"max_expert_group_count\":" << phase_max_group
        << ",\"expert_groups_gt128\":" << final_routes.expert_groups_gt128
        << ",\"group_gt128_proven\":" << (phase_max_group > 128 && final_routes.expert_groups_gt128 > 0)
        << ",\"invalid_proofs\":[";
    for (std::size_t i = 0; i < proof_count; ++i) {
        if (i) out << ',';
        const auto& p = proofs[i];
        require(p.continued, "rejection missing correct same-history continuation");
        out << "{\"case\":"; json_string(out, p.name);
        out << ",\"input_rows\":" << p.rows << ",\"offset\":" << p.stats.consumed_tokens << ",\"prior_rows\":" << p.prior_rows
            << ",\"input_fits_remaining_capacity\":" << p.input_fits_remaining_capacity
            << ",\"preserved_fullspan_values\":" << p.prior_rows * vocabulary << ",\"stats_before_and_after\":";
        json_stats(out, p.stats);
        out << ",\"route_stats_before_and_after\":"; json_route_stats(out, p.route_stats);
        out << ",\"exception\":\"invalid_argument\",\"all_public_stats_bitwise_preserved\":true,\"fullspan_bits_preserved\":true"
                ",\"route_stats_preserved\":true"
               ",\"memory_ledger_preserved\":true,\"pinned_expert_staging_preserved\":true,\"fixture_guards_preserved\":true"
               ",\"free_vram_equality_required\":false,\"consumed_increment\":0,\"continued_without_reset\":true,\"continuation_offset\":" << p.continuation_offset
            << ",\"continuation_rows\":" << p.continuation_rows << ",\"continuation_compared_values\":" << p.continuation_rows * vocabulary
            << ",\"continuation_violations\":0}";
    }
    out << "],\"passed\":true}";
    records.add(out);
}

void run_phase(qwen::Session& session, References& reference, MemoryAudit& audit, Records& records,
               Totals& totals, std::size_t phase) {
    require(phase < phase_names.size(), "phase index outside schedule");
    const auto phase_start = totals;
    const std::string name(phase_names[phase]);
    session.reset();
    check_same_stats(session.stats(), {}, name + ": reset");
    check_same_route_stats(session.route_stats(), {}, name + ": reset");
    const auto reset_memory = audit.observe(session, reference, {}, name + ": reset");
    memory_record(records, "reset", phase, session.stats(), session.route_stats(), reset_memory);
    std::size_t offset = 0, windows = 0, proof_count = 0;
    std::array<InvalidProof, 5> proofs{};
    std::span<const float> prior;
    qwen::SessionStats teacher_stats{};
    qwen::SessionRouteStats teacher_routes{};
    std::uint64_t phase_max_group = 0;
    bool ordinary_rejections = false;

    const auto accept = [&](std::size_t rows) {
        const auto where = name + " offset " + std::to_string(offset) + " N" + std::to_string(rows);
        require(rows > 0 && rows <= max_batch_tokens && offset < capacity && rows <= capacity - offset &&
                (offset >= teacher_rows || rows <= teacher_rows - offset), where + ": input window preflight");
        const auto ids = offset < teacher_rows ? std::span<const std::int32_t>(teacher_ids).subspan(offset, rows) :
            std::span<const std::int32_t>(continuation_ids).subspan(offset - teacher_rows, rows);
        reference.check(where + " before execution");
        const auto before = session.stats();
        const auto routes_before = session.route_stats();
        qwen::SessionStats after = before;
        auto routes_after = routes_before;
        Errors errors;
        bool completed = false, inspected = false, memory_checked = false;
        double wall_ms = 0;
        qwen::SessionMemory memory;
        Routes n1_routes;
        bool have_n1_routes = phase != 0;
        if (have_n1_routes) n1_routes = reference.routes(offset, rows);
        std::string_view stage = "API";
        try {
            const auto begin = std::chrono::steady_clock::now();
            prior = rows == 1 ? session.step(ids[0]) : session.step_batch(ids);
            wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            completed = true;
            after = session.stats();
            routes_after = session.route_stats();
            stage = "returned_extent";
            require(prior.size() == rows * vocabulary, where + ": token-major full logit extent");
            // Immediately inspect every returned element before memory(), another
            // window, or a rejection probe. Numeric failure stops this fixture.
            stage = "frozen_full_logit_gate";
            errors = inspect_logits(prior, phase == 0 ? std::span<const float>{} : reference.slice(offset, rows));
            inspected = true;
            require(errors.nonfinite_actual == 0 && errors.nonfinite_reference == 0 && errors.violations == 0,
                    where + ": frozen full-logit gate failed");
            stage = "completed_stats";
            require(std::isfinite(wall_ms) && wall_ms > 0, where + ": completed caller wall time is not positive finite");
            check_completed(after, before, offset, rows, where);
            static_cast<void>(check_completed_routes(routes_after, routes_before, rows, where));
            if (phase == 0 && offset == 0)
                require(errors.actual_maxabs > 0 && errors.actual_rms > 0, "first reference N1 smoke has no positive finite logits");
            stage = "memory_and_fixture_guards";
            memory = audit.observe(session, reference, prior, where);
            memory_checked = true;
            if (phase == 0) {
                require(rows == 1, "reference must use sequential N1");
                std::copy(prior.begin(), prior.end(), reference.slice(offset, rows).begin());
                reference.stats[offset] = after;
                reference.check(where + " after retaining N1");
                n1_routes = reference.routes(offset, rows);
                have_n1_routes = true;
            }
            for (std::size_t i = 0; i < proof_count; ++i) {
                auto& p = proofs[i];
                if (p.continued) continue;
                require(phase != 0 && p.stats.consumed_tokens == offset, where + ": rejection continuation offset");
                p.continued = true;
                p.continuation_offset = offset;
                p.continuation_rows = rows;
            }
            stage = "window_record";
            window_record(records, phase, windows, offset, rows, true, wall_ms, before, after, routes_before, routes_after, &errors,
                          &n1_routes, &memory, true, {}, {});
        } catch (const std::exception& error) {
            after = session.stats();
            routes_after = session.route_stats();
            window_record(records, phase, windows, offset, rows, completed, wall_ms, before, after, routes_before, routes_after,
                          inspected ? &errors : nullptr, have_n1_routes ? &n1_routes : nullptr,
                          memory_checked ? &memory : nullptr, false, stage, error.what());
            if (!completed) check_same_route_stats(routes_after, routes_before, where + ": failed execution");
            throw;
        }
        ++windows;
        ++totals.windows;
        totals.finite_values += errors.finite_values;
        totals.compared_values += errors.compared;
        totals.bit_mismatches += errors.bit_mismatches;
        totals.violations += errors.violations;
        totals.completed_call_wall_ms += wall_ms;
        phase_max_group = std::max(phase_max_group, routes_after.last_max_expert_group_assignments);
        totals.max_group = std::max(totals.max_group, routes_after.last_max_expert_group_assignments);
        totals.groups_gt128 += routes_after.expert_groups_gt128 - routes_before.expert_groups_gt128;
        offset += rows;
    };

    while (offset < teacher_rows) {
        const std::size_t width = phase == 0 ? 1 : phase == 1 ? 128 : phase == 2 ? 129 : 1024;
        accept(std::min(width, teacher_rows - offset));
        if (phase != 0 && !ordinary_rejections) {
            expect_rejection(session, reference, audit, proofs, proof_count, totals, {}, prior, "empty", name + ": empty");
            expect_rejection(session, reference, audit, proofs, proof_count, totals,
                             reference.invalid_window(max_batch_tokens + 1, 100), prior, "length1025", name + ": length1025");
            // All valid-prefix IDs fit in the actual remaining capacity. In the
            // chunk phases the bad LAST ID is after hundreds of valid IDs; the
            // single1024 phase has exactly 32 positions left. Do not truncate the
            // returned span: even its 1024-row result must survive each rejection.
            const auto invalid_rows = std::min<std::size_t>(max_batch_tokens, capacity - offset);
            expect_rejection(session, reference, audit, proofs, proof_count, totals,
                             reference.invalid_window(invalid_rows, -1), prior, "negative_last", name + ": negative_last");
            expect_rejection(session, reference, audit, proofs, proof_count, totals,
                             reference.invalid_window(invalid_rows, static_cast<std::int32_t>(vocabulary)), prior,
                             "oov_last", name + ": oov_last");
            ordinary_rejections = true;
        }
    }
    teacher_stats = session.stats();
    teacher_routes = session.route_stats();
    while (offset < capacity - 1) accept(1);
    if (phase != 0)
        expect_rejection(session, reference, audit, proofs, proof_count, totals, capacity_ids, prior,
                         "capacity1055_length2", name + ": capacity1055_length2");
    accept(1);
    require(offset == capacity && (phase == 0 ? proof_count == 0 : ordinary_rejections && proof_count == proofs.size()),
            name + ": timeline/invalid proof coverage");
    require(windows == expected_windows[phase] && totals.finite_values - phase_start.finite_values == capacity * vocabulary &&
            totals.compared_values - phase_start.compared_values == (phase == 0 ? 0 : capacity * vocabulary), name + ": phase counts");
    phase_record(records, phase, windows, phase_start, totals, teacher_stats, session.stats(), teacher_routes,
                 session.route_stats(), phase_max_group, proofs, proof_count);
    if (phase == 3) {
        totals.single1024_max_group = phase_max_group;
        totals.single1024_groups_gt128 = session.route_stats().expert_groups_gt128;
    }
    ++totals.completed_phases;
}

void execute(const std::string& model, Records& records, Totals& totals, MemoryAudit& audit,
             bool& constructed, bool& cleanup_completed) {
    try {
        {
            References reference;
            qwen::SessionConfig config;
            config.capacity = capacity;
            config.expert_slots = expert_slots;
            config.max_batch_tokens = max_batch_tokens;
            config.trace_directory.clear();
            qwen::Session session(model, config);
            constructed = true;
            check_same_stats(session.stats(), {}, "construction");
            check_same_route_stats(session.route_stats(), {}, "construction");
            const auto loaded = audit.observe(session, reference, {}, "loaded");
            memory_record(records, "loaded", phase_names.size(), session.stats(), session.route_stats(), loaded);
            for (std::size_t phase = 0; phase < phase_names.size(); ++phase)
                run_phase(session, reference, audit, records, totals, phase);
            session.reset();
            check_same_stats(session.stats(), {}, "final reset");
            check_same_route_stats(session.route_stats(), {}, "final reset");
            const auto final_memory = audit.observe(session, reference, {}, "final reset");
            memory_record(records, "final_reset", phase_names.size(), session.stats(), session.route_stats(), final_memory);
            reference.check("before fixture destruction");
        } // The only Session and both guarded float allocations are destroyed.
        cleanup_completed = true;
    } catch (...) {
        cleanup_completed = constructed;
        records.flush();
        throw;
    }
}

void complete_record(Records& records, const Totals& totals, const MemoryAudit& audit, bool cleanup_completed) {
    constexpr std::uint64_t rejection_values = (4ULL * (128 + 129 + 1024) + 3) * vocabulary;
    require(cleanup_completed && totals.completed_phases == phase_names.size() && totals.windows == window_count &&
            totals.finite_values == 4ULL * capacity * vocabulary && totals.compared_values == 3ULL * capacity * vocabulary &&
            totals.violations == 0 && totals.invalid_windows == 15 && totals.rejection_preserved_values == rejection_values &&
            audit.snapshots == memory_snapshot_count &&
            audit.preserved_values == totals.finite_values + 2 * rejection_values && records.count + 1 == success_record_count,
            "final protocol counters/cleanup");
    require(totals.single1024_max_group > 128 && totals.single1024_groups_gt128 > 0 &&
            totals.max_group >= totals.single1024_max_group && totals.groups_gt128 >= totals.single1024_groups_gt128,
            "single1024 has not proven actual logical expert group>128");
    std::ostringstream out;
    format(out);
    out << "{\"kind\":\"prefill_wide_complete\",\"protocol\":2,\"phase_count\":4,\"record_count\":" << records.count + 1
        << ",\"timeline_rows\":4224,\"teacher_rows\":4096,\"continuation_rows\":128,\"timeline_routes\":" << 4ULL * capacity * routes_per_row
        << ",\"window_count\":" << totals.windows << ",\"finite_logit_values\":" << totals.finite_values
        << ",\"compared_logit_values\":" << totals.compared_values << ",\"violations\":" << totals.violations
        << ",\"bit_mismatches_diagnostic\":" << totals.bit_mismatches
        << ",\"bit_equality_required\":false,\"argmax_equality_required\":false,\"correctness_completed_call_wall_ms_sum\":" << totals.completed_call_wall_ms
        << ",\"invalid_window_rejections\":" << totals.invalid_windows << ",\"rejection_preserved_fullspan_values\":" << totals.rejection_preserved_values
        << ",\"memory_snapshot_count\":" << audit.snapshots << ",\"memory_preserved_fullspan_values\":" << audit.preserved_values
        << ",\"minimum_free_vram_bytes\":";
    json_array(out, audit.minimum_free);
    out << ",\"steady_categories_owners_allocated_bytes_buffer_counts_peak_host_pinned_and_payload_reads\":true"
           ",\"fixture_guards_and_capacities_preserved\":true,\"individual_buffer_capacities_qualified\":false,\"component_128_staging_repack_qualified\":false"
           ",\"session_instances\":1,\"raii_session_and_fixture_cleanup_completed\":true,\"owned_buffer_release_measured\":false"
           ",\"reference_scope\":\"same_session_N1_self_parity\",\"wide1024_full_logit_gate_passed\":true"
        << ",\"max_expert_group_count\":" << totals.max_group << ",\"expert_groups_gt128\":" << totals.groups_gt128
        << ",\"single1024_max_expert_group_count\":" << totals.single1024_max_group
        << ",\"single1024_expert_groups_gt128\":" << totals.single1024_groups_gt128
        << ",\"group_gt128_proven\":true,\"group_gt128_evidence\":\"observed_completed_call_route_stats_single1024_max_gt128_and_positive_threshold_count\""
           ",\"cold_cache_equality_claim\":false,\"speedup_claim\":false,\"R4_complete_claim\":false,\"qualification_4K_16K_claim\":false"
           ",\"independent_HF_reference\":false,\"MTP_qualification_claim\":false,\"performance_claim\":false,\"passed\":true}";
    records.add(out);
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: core-prefill-wide-test MODEL.gguf\n";
        return 2;
    }
    Records records;
    bool constructed = false, cleanup_completed = false;
    try {
        check_revision();
        const std::string model(argv[1]);
        const auto model_bytes = check_model_file(model);
        source_record(records, model, model_bytes);
        Totals totals;
        MemoryAudit audit;
        execute(model, records, totals, audit, constructed, cleanup_completed);
        complete_record(records, totals, audit, cleanup_completed);
        records.flush();
        return 0;
    } catch (const std::exception& error) {
        try {
            std::ostringstream out;
            format(out);
            out << "{\"kind\":\"prefill_wide_failure\",\"protocol\":2,\"passed\":false,\"session_constructed\":" << constructed
                << ",\"raii_session_and_fixture_cleanup_completed\":" << cleanup_completed << ",\"error\":";
            json_string(out, std::string_view(error.what()).substr(0, 4096));
            out << '}';
            records.add(out);
            records.flush();
        } catch (...) { /* Preserve exit1 even if diagnostic output itself fails. */ }
        std::cerr << "core-prefill-wide-test: " << error.what() << '\n';
        return 1;
    } catch (...) {
        try {
            std::ostringstream out;
            format(out);
            out << "{\"kind\":\"prefill_wide_failure\",\"protocol\":2,\"passed\":false,\"session_constructed\":" << constructed
                << ",\"raii_session_and_fixture_cleanup_completed\":" << cleanup_completed << ",\"error\":\"non-standard exception\"}";
            records.add(out);
            records.flush();
        } catch (...) {}
        std::cerr << "core-prefill-wide-test: non-standard exception\n";
        return 1;
    }
}
