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
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#ifndef CORE_REVISION
#error "core-prefill-long-test requires compiled 40-hex CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-prefill-long-test requires compiled CORE_DIRTY (0 or 1)"
#endif

namespace {
constexpr std::size_t vocabulary = 248320, continuation = 32, batch = 1024, guards = 16;
constexpr int slots = 112;
constexpr double absolute_gate = .02, relative_gate = .002;
constexpr std::uint32_t guard_bits = 0x4b71abcd;
constexpr std::int32_t input_guard = -123456789;
constexpr std::uint64_t routes_per_row = 480, q40 = 2764800, q41 = 2867200;
constexpr std::uint64_t payload = (6 * q41 + 42 * q40) * 512;
constexpr std::uint64_t logit_bytes = batch * vocabulary * sizeof(float);
constexpr std::uint64_t handoff_bytes = batch * 4 * 2560 * sizeof(float), stage_bytes = 16 * q41;
constexpr std::uint64_t staging_bytes = 4 * stage_bytes;
// Necessary aggregate floor only: the public ledger does not expose individual scratch capacities.
constexpr std::uint64_t workspace_floor = 24ULL * batch * 12288 * sizeof(float) + batch * 320ULL * 36 +
    logit_bytes + batch * 10ULL * 2560 * sizeof(float) + 2 * stage_bytes;
constexpr std::array<std::string_view, 3> phases{"reference_n1", "canonical1024", "occupied5_then997"};
constexpr std::int32_t token_at(std::size_t position) {
    return position == 0 ? 248044 : 99 + static_cast<std::int32_t>(position);
}
template<std::size_t Teacher> constexpr auto source_ids() {
    std::array<std::int32_t, Teacher + continuation> ids{};
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = token_at(i);
    return ids;
}
template<std::size_t Teacher> constexpr bool exact_ids() {
    constexpr auto ids = source_ids<Teacher>();
    if (ids.size() != Teacher + 32 || ids[0] != 248044 || ids[1] != 100 ||
        ids[Teacher - 1] != static_cast<std::int32_t>(Teacher + 98) ||
        ids[Teacher] != static_cast<std::int32_t>(Teacher + 99) ||
        ids.back() != static_cast<std::int32_t>(ids.size() + 98)) return false;
    for (std::size_t i = 0; i < ids.size(); ++i)
        if (ids[i] != (i == 0 ? 248044 : static_cast<std::int32_t>(99 + i)) ||
            ids[i] < 0 || ids[i] >= static_cast<std::int32_t>(vocabulary)) return false;
    return 1 + (ids.size() + 98 - 100 + 1) == ids.size();
}
constexpr std::size_t mixed_chunks(std::size_t teacher) { return (teacher - 5 + 996) / 997; }
constexpr std::size_t candidate_windows(std::size_t teacher) { return teacher / batch + 32 + 5 + mixed_chunks(teacher) + 32; }
constexpr std::size_t success_records(std::size_t teacher) { return 10 + teacher / batch + 1 + candidate_windows(teacher); }
static_assert(sizeof(float) == 4 && sizeof(double) == 8 && std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
static_assert(std::string_view(CORE_REVISION).size() == 40);
static_assert(exact_ids<4096>() && exact_ids<16384>());
static_assert((4096 + 32) % 4 == 0 && (16384 + 32) % 4 == 0);
static_assert(payload == 68262297600ULL && logit_bytes == 1017118720ULL && workspace_floor == 2433482752ULL);
static_assert(mixed_chunks(4096) == 5 && mixed_chunks(16384) == 17);
static_assert(success_records(4096) == 93 && success_records(16384) == 129);

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}
std::size_t checked_values(std::size_t rows) {
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(float);
    require(rows > 0 && rows <= (limit - 2 * guards) / vocabulary, "guarded logit allocation overflow");
    return rows * vocabulary;
}
void quote(std::ostream& out, std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    out << '"';
    for (const char c : text) {
        const auto b = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (b < 32) out << "\\u00" << hex[b >> 4] << hex[b & 15];
        else out << c;
    }
    out << '"';
}
void number(std::ostream& out, double value) { if (std::isfinite(value)) out << value; else out << "null"; }
struct Writer {
    std::uint64_t count = 0;
    template<class F> void emit(std::string_view kind, F fields) {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10)
            << "{\"kind\":\"prefill_long_" << kind << "\",\"protocol\":1";
        fields(out);
        out << "}\n";
        require(static_cast<bool>(out), "JSONL formatting failed");
        std::cout << out.str();
        std::cout.flush();
        require(static_cast<bool>(std::cout), "JSONL output failed");
        ++count;
    }
};
auto stats_key(const qwen::SessionStats& s) {
    return std::tuple{s.consumed_tokens, s.expert_hits, s.expert_misses, s.expert_upload_bytes,
                      std::bit_cast<std::uint64_t>(s.last_completed_ms)};
}
auto route_key(const qwen::SessionRouteStats& s) { return std::tuple{s.last_max_expert_group_assignments, s.expert_groups_gt128}; }
auto host_key(const qwen::SessionMemory& m) {
    return std::tuple{m.capacity, m.expert_slots, m.ram_expert_capacity, m.ram_expert_payload,
        m.host_embedding_capacity, m.host_logit_capacity, m.pinned_handoff, m.pinned_expert_staging,
        m.expert_payload_reads, m.expert_payload_bytes_read, m.ownership_verified};
}
auto device_key(const qwen::SessionDeviceMemory& d) {
    return std::tuple{d.device, d.first_layer, d.last_layer, d.gdn_layers, d.qsa_layers, d.weights,
        d.expert_slots, d.qsa_kv, d.qsa_index, d.gdn_state, d.ple_state, d.workspace, d.owned_bytes,
        d.owned_peak_bytes, d.owned_buffers, d.total_vram}; // Unowned runtime free VRAM may fluctuate.
}
void json_stats(std::ostream& out, const qwen::SessionStats& s) {
    out << "{\"consumed_tokens\":" << s.consumed_tokens << ",\"expert_hits\":" << s.expert_hits
        << ",\"expert_misses\":" << s.expert_misses << ",\"expert_upload_bytes\":" << s.expert_upload_bytes
        << ",\"last_completed_ms\":"; number(out, s.last_completed_ms);
    out << ",\"last_completed_ms_bits\":" << std::bit_cast<std::uint64_t>(s.last_completed_ms) << '}';
}
void json_routes(std::ostream& out, const qwen::SessionRouteStats& s) {
    out << "{\"last_max_expert_group_assignments\":" << s.last_max_expert_group_assignments
        << ",\"expert_groups_gt128\":" << s.expert_groups_gt128 << '}';
}
void json_memory(std::ostream& out, const qwen::SessionMemory& m) {
    out << "{\"capacity\":" << m.capacity << ",\"expert_slots\":" << m.expert_slots << ",\"ownership_verified\":" << m.ownership_verified
        << ",\"ram_expert_capacity\":" << m.ram_expert_capacity << ",\"ram_expert_payload\":" << m.ram_expert_payload
        << ",\"host_embedding_capacity\":" << m.host_embedding_capacity << ",\"host_logit_capacity\":" << m.host_logit_capacity
        << ",\"pinned_handoff\":" << m.pinned_handoff << ",\"pinned_expert_staging\":" << m.pinned_expert_staging
        << ",\"expert_payload_reads\":" << m.expert_payload_reads << ",\"expert_payload_bytes_read\":" << m.expert_payload_bytes_read << ",\"devices\":[";
    for (std::size_t i = 0; i < 2; ++i) {
        const auto& d = m.devices[i];
        if (i) out << ',';
        out << "{\"device\":" << d.device << ",\"first_layer\":" << d.first_layer << ",\"last_layer\":" << d.last_layer
            << ",\"gdn_layers\":" << d.gdn_layers << ",\"qsa_layers\":" << d.qsa_layers << ",\"weights\":" << d.weights
            << ",\"expert_slots\":" << d.expert_slots << ",\"qsa_kv\":" << d.qsa_kv << ",\"qsa_index\":" << d.qsa_index
            << ",\"gdn_state\":" << d.gdn_state << ",\"ple_state\":" << d.ple_state << ",\"workspace\":" << d.workspace
            << ",\"owned_bytes\":" << d.owned_bytes << ",\"owned_peak_bytes\":" << d.owned_peak_bytes
            << ",\"owned_buffers\":" << d.owned_buffers << ",\"total_vram\":" << d.total_vram << ",\"free_vram\":" << d.free_vram << '}';
    }
    out << "]}";
}
struct GuardedFloat {
    std::size_t values, allocated;
    std::vector<float> storage;
    explicit GuardedFloat(std::size_t rows) : values(checked_values(rows)), allocated(0), storage(values + 2 * guards) {
        allocated = storage.capacity();
        std::fill_n(storage.begin(), guards, std::bit_cast<float>(guard_bits));
        std::fill_n(storage.end() - guards, guards, std::bit_cast<float>(guard_bits));
    }
    std::span<float> span() { return std::span<float>(storage).subspan(guards, values); }
    void check() const {
        require(storage.size() == values + 2 * guards && storage.capacity() == allocated, "fixture float capacity changed");
        for (std::size_t i = 0; i < guards; ++i)
            require(std::bit_cast<std::uint32_t>(storage[i]) == guard_bits &&
                    std::bit_cast<std::uint32_t>(storage[guards + values + i]) == guard_bits, "fixture float guard changed");
    }
};
struct Reference {
    GuardedFloat logits, saved;
    std::array<std::int32_t, batch + 1 + 2 * guards> invalid{};
    explicit Reference(std::size_t total) : logits(total), saved(batch) {
        invalid.fill(100);
        std::fill_n(invalid.begin(), guards, input_guard);
        std::fill_n(invalid.end() - guards, guards, input_guard);
        check();
    }
    void check() const {
        logits.check(); saved.check();
        for (std::size_t i = 0; i < guards; ++i)
            require(invalid[i] == input_guard && invalid[invalid.size() - guards + i] == input_guard, "fixture input guard changed");
    }
    std::span<float> slice(std::size_t offset, std::size_t rows) {
        require(rows > 0 && offset <= logits.values / vocabulary && rows <= logits.values / vocabulary - offset, "reference slice extent");
        return logits.span().subspan(offset * vocabulary, rows * vocabulary);
    }
    std::span<const float> snapshot(std::span<const float> active) {
        require(active.size() <= saved.values, "preservation workspace extent");
        std::copy(active.begin(), active.end(), saved.span().begin());
        return saved.span().first(active.size());
    }
    std::span<const std::int32_t> bad_ids(std::size_t rows, std::int32_t last) {
        require(rows > 0 && rows <= batch + 1, "invalid input extent");
        auto ids = std::span<std::int32_t>(invalid).subspan(guards, rows);
        std::fill(ids.begin(), ids.end(), 100); ids.back() = last;
        return ids;
    }
};
void exact_span(std::span<const float> a, std::span<const float> b) {
    require(a.size() == b.size(), "preserved span extent");
    for (std::size_t i = 0; i < a.size(); ++i)
        require(std::bit_cast<std::uint32_t>(a[i]) == std::bit_cast<std::uint32_t>(b[i]), "live returned logit bits changed");
}
struct Audit {
    qwen::SessionMemory loaded{}, last{};
    std::array<std::uint64_t, 2> minimum_free{};
    std::uint64_t observations = 0, preserved_values = 0;
    void take(qwen::Session& session, Reference& reference, std::span<const float> active,
              std::span<const float> saved, std::size_t total) {
        reference.check();
        const auto stats = session.stats(); const auto routes = session.route_stats();
        const auto m = session.memory();
        require(stats_key(stats) == stats_key(session.stats()) && route_key(routes) == route_key(session.route_stats()), "memory() mutated public stats");
        exact_span(active, saved); reference.check();
        if (observations == 0) {
            require(m.capacity == static_cast<int>(total) && m.expert_slots == slots && m.ownership_verified, "memory owner/config ledger");
            require(m.ram_expert_payload == payload && m.ram_expert_capacity >= payload &&
                    m.expert_payload_reads == 144 && m.expert_payload_bytes_read == payload, "expert RAM/read accounting");
            require(m.host_embedding_capacity > 0 && m.host_logit_capacity >= logit_bytes &&
                    m.pinned_handoff == handoff_bytes && m.pinned_expert_staging == staging_bytes, "host/pinned geometry");
            for (std::size_t i = 0; i < 2; ++i) {
                const auto& d = m.devices[i];
                require(d.device == static_cast<int>(i) && d.first_layer == static_cast<int>(24 * i) &&
                        d.last_layer == static_cast<int>(24 * i + 23) && d.gdn_layers == 18 && d.qsa_layers == 6, "static 24/24 ownership");
                require(d.expert_slots == slots * (i == 0 ? 6 * q41 + 18 * q40 : 24 * q40) &&
                        d.qsa_kv == 6ULL * total * 32 * 18 && d.qsa_index == 6ULL * (total / 4 * 128 + 384) * sizeof(float) &&
                        d.gdn_state == 18ULL * (786432 + 30720) * sizeof(float) &&
                        d.ple_state == (i == 0 ? 92160ULL * sizeof(float) : 0), "expert/Q4 KV/index/GDN/PLE category geometry");
                std::uint64_t sum = 0;
                for (const auto bytes : {d.weights, d.expert_slots, d.qsa_kv, d.qsa_index, d.gdn_state, d.ple_state, d.workspace}) {
                    require(bytes <= std::numeric_limits<std::uint64_t>::max() - sum, "category sum overflow"); sum += bytes;
                }
                require(d.weights > 0 && d.workspace >= workspace_floor && d.owned_bytes == sum &&
                        d.owned_buffers > 0 && d.owned_peak_bytes >= sum, "Buffer category/count/peak accounting");
            }
            loaded = m; minimum_free = {m.devices[0].free_vram, m.devices[1].free_vram};
        }
        require(host_key(m) == host_key(loaded), "host/pinned/read ledger growth");
        for (std::size_t i = 0; i < 2; ++i) {
            const auto& d = m.devices[i];
            require(device_key(d) == device_key(loaded.devices[i]), "device category/Buffer count/peak growth");
            require(d.total_vram > 0 && d.free_vram > 0 && d.free_vram <= d.total_vram &&
                    d.owned_bytes <= d.total_vram - d.free_vram, "observed VRAM/owned bytes");
            minimum_free[i] = std::min(minimum_free[i], d.free_vram);
        }
        last = m; ++observations; preserved_values += active.size();
    }
};
struct Errors {
    std::uint64_t values = 0, finite = 0, compared = 0, finite_pairs = 0, nonfinite_actual = 0, nonfinite_reference = 0;
    std::uint64_t violations = 0, bits = 0, argmax_rows = 0, argmax_agree = 0;
    double maxabs = 0, maxratio = 0, squared_error = 0, bad_actual = 0, bad_reference = 0, bad_bound = 0;
    std::size_t abs_position = 0, abs_column = 0, ratio_position = 0, ratio_column = 0, bad_position = 0, bad_column = 0;
    void add(const Errors& e) {
        if (e.finite_pairs && (!finite_pairs || e.maxabs > maxabs)) { maxabs = e.maxabs; abs_position = e.abs_position; abs_column = e.abs_column; }
        if (e.finite_pairs && (!finite_pairs || e.maxratio > maxratio)) { maxratio = e.maxratio; ratio_position = e.ratio_position; ratio_column = e.ratio_column; }
        if (e.violations && !violations) {
            bad_position = e.bad_position; bad_column = e.bad_column;
            bad_actual = e.bad_actual; bad_reference = e.bad_reference; bad_bound = e.bad_bound;
        }
        values += e.values; finite += e.finite; compared += e.compared; finite_pairs += e.finite_pairs;
        nonfinite_actual += e.nonfinite_actual; nonfinite_reference += e.nonfinite_reference; violations += e.violations;
        bits += e.bits; argmax_rows += e.argmax_rows; argmax_agree += e.argmax_agree; squared_error += e.squared_error;
    }
};
Errors inspect(std::span<const float> actual, std::span<const float> expected, std::size_t offset) {
    require(!actual.empty() && actual.size() % vocabulary == 0 && (expected.empty() || actual.size() == expected.size()), "full logit inspection extent");
    Errors e; e.values = actual.size(); e.compared = expected.size();
    for (std::size_t row = 0; row < actual.size() / vocabulary; ++row) {
        std::size_t aa = 0, ba = 0; bool finite_row = true;
        for (std::size_t column = 0; column < vocabulary; ++column) {
            const auto i = row * vocabulary + column;
            const double a = actual[i], b = expected.empty() ? 0 : expected[i];
            const bool fa = std::isfinite(a), fb = std::isfinite(b);
            e.finite += fa; e.nonfinite_actual += !fa;
            if (!expected.empty()) { e.nonfinite_reference += !fb; e.bits += std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]); }
            const double error = std::abs(a - b), bound = absolute_gate + relative_gate * std::abs(b);
            if (!fa || !fb || (!expected.empty() && error > bound)) {
                if (e.violations++ == 0) { e.bad_position = offset + row; e.bad_column = column; e.bad_actual = a; e.bad_reference = b; e.bad_bound = bound; }
            }
            if (!fa || !fb) { finite_row = false; continue; }
            if (expected.empty()) continue;
            ++e.finite_pairs; e.squared_error += error * error;
            if (e.finite_pairs == 1 || error > e.maxabs) { e.maxabs = error; e.abs_position = offset + row; e.abs_column = column; }
            if (e.finite_pairs == 1 || error / bound > e.maxratio) { e.maxratio = error / bound; e.ratio_position = offset + row; e.ratio_column = column; }
            if (actual[i] > actual[row * vocabulary + aa]) aa = column;
            if (expected[i] > expected[row * vocabulary + ba]) ba = column;
        }
        if (!expected.empty() && finite_row) { ++e.argmax_rows; e.argmax_agree += aa == ba; }
    }
    return e;
}
void coordinate(std::ostream& out, std::size_t position, std::size_t column) {
    out << "{\"position\":" << position << ",\"vocabulary_index\":" << column << '}';
}
void json_errors(std::ostream& out, const Errors& e) {
    out << "{\"values\":" << e.values << ",\"finite_values\":" << e.finite << ",\"compared_values\":" << e.compared
        << ",\"finite_pairs\":" << e.finite_pairs << ",\"nonfinite_actual\":" << e.nonfinite_actual
        << ",\"nonfinite_reference\":" << e.nonfinite_reference << ",\"violations\":" << e.violations
        << ",\"bit_mismatches_diagnostic\":" << e.bits << ",\"argmax_rows_diagnostic\":" << e.argmax_rows
        << ",\"argmax_agree_diagnostic\":" << e.argmax_agree << ",\"maxabs\":";
    if (e.finite_pairs) number(out, e.maxabs); else out << "null";
    out << ",\"rms\":"; if (e.finite_pairs) number(out, std::sqrt(e.squared_error / static_cast<double>(e.finite_pairs))); else out << "null";
    out << ",\"maxboundratio\":"; if (e.finite_pairs) number(out, e.maxratio); else out << "null";
    out << ",\"maxabs_coordinate\":"; if (e.finite_pairs) coordinate(out, e.abs_position, e.abs_column); else out << "null";
    out << ",\"maxboundratio_coordinate\":"; if (e.finite_pairs) coordinate(out, e.ratio_position, e.ratio_column); else out << "null";
    out << ",\"first_violation_coordinate\":"; if (e.violations) coordinate(out, e.bad_position, e.bad_column); else out << "null";
    out << ",\"first_violation_actual\":"; if (e.violations) number(out, e.bad_actual); else out << "null";
    out << ",\"first_violation_reference\":"; if (e.violations && e.compared) number(out, e.bad_reference); else out << "null";
    out << ",\"first_violation_bound\":"; if (e.violations && e.compared) number(out, e.bad_bound); else out << "null";
    out << '}';
}
struct Coverage {
    unsigned visible_mask = 0, multirow_visible_mask = 0, mod4_mask = 0, occupied_start_mod4_mask = 0;
    std::uint64_t occupied_chunks = 0, completed_block_crossings = 0;
    bool occupied_budget_crossing = false;
    void accept(std::size_t offset, std::size_t rows, std::size_t teacher) {
        if (offset >= teacher) return;
        for (std::size_t j = 0; j < rows; ++j) {
            const auto visible = offset + j + 1;
            if (visible >= 2047 && visible <= 2056) {
                visible_mask |= 1U << (visible - 2047); mod4_mask |= 1U << (visible % 4);
                if (rows > 1) multirow_visible_mask |= 1U << (visible - 2047);
            }
        }
        completed_block_crossings += offset / 4 != (offset + rows) / 4;
        if (offset && rows > 1) {
            ++occupied_chunks; occupied_start_mod4_mask |= 1U << (offset % 4);
            occupied_budget_crossing |= offset < 2052 && offset + rows >= 2052;
        }
    }
};
void json_coverage(std::ostream& out, const Coverage& c) {
    out << "{\"visible2047_2056_mask\":" << c.visible_mask << ",\"visible_mod4_mask\":" << c.mod4_mask
        << ",\"multirow_visible2047_2056_mask\":" << c.multirow_visible_mask
        << ",\"occupied_chunk_start_mod4_mask\":" << c.occupied_start_mod4_mask
        << ",\"occupied_chunks\":" << c.occupied_chunks << ",\"completed_block_crossing_windows\":" << c.completed_block_crossings
        << ",\"occupied_budget2052_crossing\":" << c.occupied_budget_crossing
        << ",\"all_boundary_visibilities_observed\":" << (c.visible_mask == 1023 && c.mod4_mask == 15)
        << ",\"scope\":\"actual_accepted_teacher_rows_logical_causal_visibility_not_GPU_selected_ID_trace\"}";
}
struct Summary {
    Errors errors;
    std::uint64_t windows = 0, max_group = 0, groups_gt128 = 0;
    double wall_ms = 0;
    void add(const Errors& e, const qwen::SessionRouteStats& a, const qwen::SessionRouteStats& b, double ms) {
        errors.add(e); ++windows; max_group = std::max(max_group, a.last_max_expert_group_assignments);
        groups_gt128 += a.expert_groups_gt128 - b.expert_groups_gt128; wall_ms += ms;
    }
};
struct Context {
    std::size_t phase = 0, window = 0, offset = 0, rows = 0;
    std::string_view stage = "arguments";
    qwen::SessionStats before{}, after{};
    qwen::SessionRouteStats routes_before{}, routes_after{};
    Errors errors;
    bool completed = false, inspected = false, constructed = false, cleanup = false;
};
void context_fields(std::ostream& out, const Context& c) {
    out << ",\"phase_index\":" << c.phase << ",\"phase\":"; quote(out, phases[c.phase]);
    out << ",\"window_index\":" << c.window << ",\"offset\":" << c.offset << ",\"rows\":" << c.rows
        << ",\"stage\":"; quote(out, c.stage);
    out << ",\"completed_call\":" << c.completed << ",\"stats_before\":"; json_stats(out, c.before);
    out << ",\"stats_after\":"; json_stats(out, c.after);
    out << ",\"route_stats_before\":"; json_routes(out, c.routes_before);
    out << ",\"route_stats_after\":"; json_routes(out, c.routes_after);
}
void check_completed(const Context& c, double ms) {
    const auto& a = c.after; const auto& b = c.before;
    require(b.consumed_tokens == c.offset && a.consumed_tokens == c.offset + c.rows, "consumed increment is not N");
    require(std::isfinite(ms) && ms > 0 && std::isfinite(a.last_completed_ms) && a.last_completed_ms > 0, "completed time not positive finite");
    require(a.expert_hits >= b.expert_hits && a.expert_misses >= b.expert_misses && a.expert_upload_bytes >= b.expert_upload_bytes, "expert counters decreased");
    const auto hits = a.expert_hits - b.expert_hits, misses = a.expert_misses - b.expert_misses;
    const auto uploads = a.expert_upload_bytes - b.expert_upload_bytes, routes = routes_per_row * c.rows;
    require(hits <= routes && misses <= routes && hits + misses == routes && uploads >= misses * q40 && uploads <= misses * q41, "480*N routing/payload accounting");
    const auto& r = c.routes_after; const auto& p = c.routes_before;
    require(r.last_max_expert_group_assignments >= 1 && r.last_max_expert_group_assignments <= c.rows && r.expert_groups_gt128 >= p.expert_groups_gt128, "logical group diagnostic bounds");
    const auto delta = r.expert_groups_gt128 - p.expert_groups_gt128;
    require(delta <= routes / 129 && (delta > 0) == (r.last_max_expert_group_assignments > 128), "group maximum/threshold counter disagree");
}
struct Proof {
    std::string_view name;
    std::size_t input_rows = 0, offset = 0, prior_rows = 0, continued_rows = 0;
    qwen::SessionStats stats{};
    qwen::SessionRouteStats routes{};
};
void json_proofs(std::ostream& out, const std::array<Proof, 4>& proofs, std::size_t count, std::size_t total) {
    out << '[';
    for (std::size_t i = 0; i < count; ++i) {
        const auto& p = proofs[i]; require(p.continued_rows > 0, "rejection missing same-history continuation");
        if (i) out << ',';
        out << "{\"case\":"; quote(out, p.name);
        out << ",\"input_rows\":" << p.input_rows << ",\"offset\":" << p.offset << ",\"prior_rows\":" << p.prior_rows
            << ",\"input_fits_remaining_capacity\":" << (p.input_rows <= total - p.offset)
            << ",\"preserved_fullspan_values\":" << p.prior_rows * vocabulary << ",\"stats_before_and_after\":"; json_stats(out, p.stats);
        out << ",\"route_stats_before_and_after\":"; json_routes(out, p.routes);
        out << ",\"exception\":\"invalid_argument\",\"all_public_stats_and_route_stats_bitwise_preserved\":true"
               ",\"fullspan_bits_preserved\":true,\"memory_ledger_preserved\":true,\"guards_preserved\":true"
               ",\"continued_without_reset\":true,\"continuation_offset\":" << p.offset << ",\"continuation_rows\":" << p.continued_rows
            << ",\"continuation_compared_values\":" << p.continued_rows * vocabulary << ",\"continuation_violations\":0}";
    }
    out << ']';
}

// JSONL protocol 1: source; memory(loaded, reset x3, final_reset); baseline_checkpoint
// (every 1024 N1 teacher calls, then 32 N1 continuation calls); window (every candidate
// call); phase x3; complete OR bounded failure. Streaming records use one temporary
// string, never one retained record/memory snapshot per baseline row. Check every call
// in-process. Checkpoint errors/time/max/group counts aggregate just that checkpoint;
// stats_before/after delimit it. Phase errors aggregate its entire timeline. All
// coordinates are zero-based timeline position and vocabulary index. Complete is
// emitted only after the exclusive Session and both guarded allocations have unwound.
template<std::size_t Teacher> void run(const std::string& model, Writer& writer, Context& context) {
    constexpr std::size_t total = Teacher + continuation;
    constexpr auto ids = source_ids<Teacher>();
    constexpr std::array<std::size_t, 3> expected_windows{total, Teacher / batch + 32, 5 + mixed_chunks(Teacher) + 32};
    constexpr std::uint64_t all_windows = expected_windows[0] + expected_windows[1] + expected_windows[2];
    constexpr std::uint64_t rejection_values = (3ULL * (1024 + 997) + 2) * vocabulary;
    Summary overall; Audit audit;
    std::uint64_t rejection_count = 0, rejected_values = 0, completed_phases = 0;
    try {
        {
            context.stage = "guarded_preallocation_before_model";
            Reference reference(total); // Both large, touched, checked float allocations precede Model/Session.
            for (std::size_t i = 0; i < ids.size(); ++i)
                require(ids[i] == token_at(i) && ids[i] >= 0 && ids[i] < static_cast<std::int32_t>(vocabulary), "every source ID runtime preflight");
            context.stage = "Session_construction";
            qwen::SessionConfig config;
            config.capacity = static_cast<int>(total); config.expert_slots = slots; config.max_batch_tokens = static_cast<int>(batch);
            config.trace_directory.clear();
            qwen::Session session(model, config); context.constructed = true;
            const auto memory_event = [&](std::string_view event) {
                context.stage = event;
                context.before = context.after = session.stats(); context.routes_before = context.routes_after = session.route_stats();
                require(stats_key(context.after) == stats_key({}) && route_key(context.routes_after) == route_key({}), "construction/reset statistics not zero");
                audit.take(session, reference, {}, {}, total);
                writer.emit("memory", [&](auto& out) {
                    out << ",\"event\":"; quote(out, event); out << ",\"phase_index\":" << context.phase << ",\"stats\":";
                    json_stats(out, session.stats()); out << ",\"route_stats\":"; json_routes(out, session.route_stats());
                    out << ",\"memory\":"; json_memory(out, audit.last);
                    out << ",\"reported_known_host_and_fixture_bytes\":" << audit.last.ram_expert_capacity + audit.last.host_embedding_capacity +
                        audit.last.host_logit_capacity + audit.last.pinned_handoff + audit.last.pinned_expert_staging +
                        (reference.logits.allocated + reference.saved.allocated) * sizeof(float) + sizeof(reference.invalid)
                        << ",\"host_accounting_excludes_metadata_allocator_runtime_overhead\":true,\"passed\":true";
                });
            };
            memory_event("loaded");
            for (std::size_t phase = 0; phase < phases.size(); ++phase) {
                context.phase = phase; context.window = 0; context.offset = 0; context.rows = 0;
                context.completed = false; context.inspected = false; context.stage = "reset";
                session.reset(); memory_event("reset");
                Summary summary, block; Coverage coverage;
                std::size_t offset = 0, proof_count = 0, checkpoint = 0, block_offset = 0;
                std::array<Proof, 4> proofs{};
                std::span<const float> prior; // Used only until the NEXT accepted call; never reread afterwards.
                qwen::SessionStats block_stats{}, teacher_stats{};
                qwen::SessionRouteStats block_routes{}, teacher_routes{};
                const auto accept = [&](std::size_t rows) {
                    context.window = static_cast<std::size_t>(summary.windows); context.offset = offset; context.rows = rows;
                    context.completed = false; context.inspected = false; context.errors = {}; context.stage = "input_preflight";
                    require(rows > 0 && rows <= batch && rows <= total - offset && (offset >= Teacher || rows <= Teacher - offset), "accepted input extent");
                    reference.check(); context.before = session.stats(); context.routes_before = session.route_stats();
                    context.after = context.before; context.routes_after = context.routes_before;
                    double ms = 0;
                    try {
                        context.stage = "API";
                        const auto begin = std::chrono::steady_clock::now();
                        // Assignment expires the previous span; nothing reads its old value afterwards.
                        prior = rows == 1 ? session.step(ids[offset]) : session.step_batch(std::span<const std::int32_t>(ids).subspan(offset, rows));
                        ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                        context.completed = true; context.after = session.stats(); context.routes_after = session.route_stats();
                        context.stage = "returned_extent"; require(prior.size() == rows * vocabulary, "returned full vocabulary extent");
                        context.stage = "frozen_full_logit_gate";
                        context.errors = inspect(prior, phase == 0 ? std::span<const float>{} : reference.slice(offset, rows), offset);
                        context.inspected = true;
                        require(context.errors.finite == prior.size() && context.errors.nonfinite_reference == 0 && context.errors.violations == 0, "full-logit finite/numerical gate failed");
                        context.stage = "completed_stats"; check_completed(context, ms);
                        if (phase == 0) {
                            require(rows == 1, "reference is not sequential N1");
                            std::copy(prior.begin(), prior.end(), reference.slice(offset, rows).begin());
                        }
                        context.stage = "memory_and_full_live_span_preservation";
                        const auto saved = reference.snapshot(prior); // Preserves ALL 1024 rows, not a selected-logit subset.
                        audit.take(session, reference, prior, saved, total);
                    } catch (...) {
                        context.after = session.stats(); context.routes_after = session.route_stats(); throw;
                    }
                    for (std::size_t i = 0; i < proof_count; ++i) if (!proofs[i].continued_rows) {
                        require(phase != 0 && proofs[i].offset == offset, "rejected window continuation offset"); proofs[i].continued_rows = rows;
                    }
                    summary.add(context.errors, context.routes_after, context.routes_before, ms);
                    overall.add(context.errors, context.routes_after, context.routes_before, ms);
                    coverage.accept(offset, rows, Teacher);
                    if (phase == 0) {
                        block.add(context.errors, context.routes_after, context.routes_before, ms);
                    } else writer.emit("window", [&](auto& out) {
                        context_fields(out, context);
                        out << ",\"segment\":\"" << (offset < Teacher ? "teacher" : "continuation") << "\",\"expected_routes\":" << routes_per_row * rows
                            << ",\"expert_groups_gt128_delta\":" << context.routes_after.expert_groups_gt128 - context.routes_before.expert_groups_gt128
                            << ",\"correctness_completed_call_wall_ms\":" << ms << ",\"errors\":"; json_errors(out, context.errors);
                        out << ",\"memory\":"; json_memory(out, audit.last);
                        out << ",\"completed_blocks_before\":" << offset / 4 << ",\"completed_blocks_after\":" << (offset + rows) / 4
                            << ",\"tail_before\":" << offset % 4 << ",\"tail_after\":" << (offset + rows) % 4
                            << ",\"stats_and_full_live_span_preserved\":true,\"passed\":true";
                    });
                    offset += rows;
                    if (phase == 0 && (offset % batch == 0 || offset == total)) {
                        writer.emit("baseline_checkpoint", [&](auto& out) {
                            out << ",\"phase_index\":0,\"checkpoint_index\":" << checkpoint << ",\"offset\":" << block_offset
                                << ",\"rows\":" << offset - block_offset << ",\"windows\":" << block.windows
                                << ",\"segment\":\"" << (block_offset < Teacher ? "teacher" : "continuation") << "\",\"stats_before\":";
                            json_stats(out, block_stats); out << ",\"stats_after\":"; json_stats(out, context.after);
                            out << ",\"route_stats_before\":"; json_routes(out, block_routes); out << ",\"route_stats_after\":"; json_routes(out, context.routes_after);
                            out << ",\"max_expert_group_count\":" << block.max_group << ",\"expert_groups_gt128\":" << block.groups_gt128
                                << ",\"correctness_completed_call_wall_ms_sum\":" << block.wall_ms << ",\"errors\":"; json_errors(out, block.errors);
                            out << ",\"memory\":"; json_memory(out, audit.last);
                            out << ",\"every_N1_call_finite_stats_routes_memory_guards_checked\":true,\"passed\":true";
                        });
                        ++checkpoint; block_offset = offset; block_stats = context.after; block_routes = context.routes_after; block = {};
                    }
                };
                const auto reject = [&](std::string_view name, std::span<const std::int32_t> input) {
                    context.stage = name; context.offset = offset; context.rows = input.size(); context.window = static_cast<std::size_t>(summary.windows);
                    context.completed = false; context.inspected = false; context.errors = {};
                    require(proof_count < proofs.size() && !prior.empty(), "rejection proof extent");
                    context.before = session.stats(); context.routes_before = session.route_stats();
                    context.after = context.before; context.routes_after = context.routes_before;
                    const auto saved = reference.snapshot(prior);
                    audit.take(session, reference, prior, saved, total); const auto before_memory = audit.last;
                    bool rejected = false;
                    try { static_cast<void>(session.step_batch(input)); } catch (const std::invalid_argument&) { rejected = true; }
                    catch (...) { context.after = session.stats(); context.routes_after = session.route_stats(); throw; }
                    context.after = session.stats(); context.routes_after = session.route_stats();
                    require(rejected, "invalid input accepted");
                    require(stats_key(context.after) == stats_key(context.before) && route_key(context.routes_after) == route_key(context.routes_before), "rejection mutated public stats");
                    exact_span(prior, saved); reference.check();
                    audit.take(session, reference, prior, saved, total);
                    require(host_key(audit.last) == host_key(before_memory) && device_key(audit.last.devices[0]) == device_key(before_memory.devices[0]) &&
                            device_key(audit.last.devices[1]) == device_key(before_memory.devices[1]), "rejection mutated memory ledger");
                    proofs[proof_count++] = {name, input.size(), offset, prior.size() / vocabulary, 0, context.before, context.routes_before};
                    ++rejection_count; rejected_values += prior.size();
                };
                bool ordinary_rejections = false;
                while (offset < Teacher) {
                    const std::size_t width = phase == 0 ? 1 : phase == 1 ? batch : offset < 5 ? 1 : 997;
                    accept(std::min(width, Teacher - offset));
                    if (phase && width > 1 && !ordinary_rejections) {
                        require(total - offset >= batch + 1, "ordinary invalid probes do not fit capacity");
                        reject("oversized1025", reference.bad_ids(batch + 1, 100));
                        reject("negative_last1024", reference.bad_ids(batch, -1));
                        reject("oov_last1024", reference.bad_ids(batch, static_cast<std::int32_t>(vocabulary)));
                        ordinary_rejections = true;
                    }
                }
                teacher_stats = session.stats(); teacher_routes = session.route_stats();
                while (offset < total - 1) accept(1);
                if (phase) {
                    const std::array<std::int32_t, 2> capacity_ids{token_at(total - 1), token_at(total)};
                    require(capacity_ids[0] >= 0 && capacity_ids[1] < static_cast<std::int32_t>(vocabulary), "capacity probe IDs");
                    reject("capacity_remaining1_length2", capacity_ids);
                }
                accept(1); context.stage = "phase_footer";
                require(offset == total && summary.windows == expected_windows[phase] && summary.errors.values == total * vocabulary &&
                        summary.errors.finite == total * vocabulary && summary.errors.compared == (phase ? total * vocabulary : 0) &&
                        summary.errors.finite_pairs == summary.errors.compared && summary.errors.violations == 0, "phase timeline/value counts");
                require(teacher_stats.consumed_tokens == Teacher && session.stats().consumed_tokens == total &&
                        session.route_stats().expert_groups_gt128 == summary.groups_gt128 &&
                        coverage.visible_mask == 1023 && coverage.mod4_mask == 15 && proof_count == (phase ? 4U : 0U), "phase teacher/coverage/rejection counts");
                if (phase) require(ordinary_rejections && coverage.multirow_visible_mask == 1023 && coverage.occupied_budget_crossing &&
                                   coverage.occupied_chunks > 0 && summary.max_group > 128 && summary.groups_gt128 > 0, "candidate occupied-boundary/logical group>128 coverage");
                if (phase == 2) require(coverage.occupied_start_mod4_mask == 15, "997 schedule missing occupied start tail phase");
                writer.emit("phase", [&](auto& out) {
                    out << ",\"phase_index\":" << phase << ",\"phase\":"; quote(out, phases[phase]);
                    out << ",\"windows\":" << summary.windows << ",\"teacher_rows\":" << Teacher << ",\"continuation_rows\":32,\"errors\":"; json_errors(out, summary.errors);
                    out << ",\"teacher_stats\":"; json_stats(out, teacher_stats); out << ",\"final_stats\":"; json_stats(out, session.stats());
                    out << ",\"teacher_route_stats\":"; json_routes(out, teacher_routes); out << ",\"final_route_stats\":"; json_routes(out, session.route_stats());
                    out << ",\"max_expert_group_count\":" << summary.max_group << ",\"expert_groups_gt128\":" << summary.groups_gt128
                        << ",\"correctness_completed_call_wall_ms_sum\":" << summary.wall_ms << ",\"coverage\":"; json_coverage(out, coverage);
                    out << ",\"invalid_proofs\":"; json_proofs(out, proofs, proof_count, total); out << ",\"passed\":true";
                });
                ++completed_phases;
            }
            context.stage = "final_reset"; context.rows = 0; context.completed = false; context.inspected = false;
            session.reset(); memory_event("final_reset"); reference.check();
        } // Session first, then guarded reference/preservation: RAII before any complete/failure.
        context.cleanup = true;
    } catch (...) { context.cleanup = true; throw; }
    context.stage = "complete_gates";
    require(completed_phases == 3 && overall.windows == all_windows && overall.errors.finite == 3ULL * total * vocabulary &&
            overall.errors.compared == 2ULL * total * vocabulary && overall.errors.finite_pairs == overall.errors.compared &&
            overall.errors.violations == 0 && rejection_count == 8 && rejected_values == rejection_values &&
            audit.observations == all_windows + 21 && audit.preserved_values == overall.errors.finite + 2 * rejection_values &&
            writer.count + 1 == success_records(Teacher), "complete evidence counts");
    writer.emit("complete", [&](auto& out) {
        out << ",\"teacher_rows_per_phase\":" << Teacher << ",\"continuation_rows_per_phase\":32,\"phase_count\":3,\"timeline_rows\":" << 3 * total
            << ",\"window_count\":" << overall.windows << ",\"record_count\":" << writer.count + 1 << ",\"errors\":"; json_errors(out, overall.errors);
        out << ",\"max_expert_group_count\":" << overall.max_group << ",\"expert_groups_gt128\":" << overall.groups_gt128
            << ",\"invalid_window_rejections\":" << rejection_count << ",\"rejection_preserved_fullspan_values\":" << rejected_values
            << ",\"memory_observation_count\":" << audit.observations << ",\"memory_preserved_fullspan_values\":" << audit.preserved_values
            << ",\"minimum_free_vram_bytes\":[" << audit.minimum_free[0] << ',' << audit.minimum_free[1] << ']'
            << ",\"correctness_completed_call_wall_ms_sum\":" << overall.wall_ms
            << ",\"steady_owners_categories_Buffer_counts_peaks_and_host_read_ledger\":true,\"fixture_guards_preserved\":true"
               ",\"raii_session_and_fixture_cleanup_completed\":true,\"owned_buffer_release_measured\":false,\"full_prefill_self_parity_passed\":true"
               ",\"logical_group_gt128_proven\":true,\"physical_128_column_tile_proven\":false,\"independent_HF_reference\":false"
               ",\"cold_cache_equality_claim\":false,\"performance_claim\":false,\"peak_VRAM_qualification_claim\":false,\"R4_complete_claim\":false,\"passed\":true";
    });
}

void source_record(Writer& writer, const std::string& model, std::size_t teacher) {
    const auto total = teacher + continuation;
    const auto reference_bytes = checked_values(total) * sizeof(float);
    const auto known_host_min = payload + reference_bytes + 2 * logit_bytes + handoff_bytes + staging_bytes +
        4 * guards * sizeof(float) + (batch + 1 + 2 * guards) * sizeof(std::int32_t);
    writer.emit("source", [&](auto& out) {
        out << ",\"revision\":"; quote(out, CORE_REVISION); out << ",\"dirty\":" << (CORE_DIRTY != 0) << ",\"model\":"; quote(out, model);
        out << ",\"model_bytes\":" << std::filesystem::file_size(model) << ",\"runtime\":\"own_48_layer_HIP\",\"session_instances\":1"
            << ",\"config\":{\"capacity\":" << total << ",\"expert_slots\":112,\"max_batch_tokens\":1024,\"trace\":false}"
            << ",\"large_RAM_opt_in\":\"explicit_ROWS_4096_or_16384\",\"teacher_family\":\"new_BOS_then_monotone_teacher_IDs_not_original_user_prompt_or_baseline_parity\""
            << ",\"source_ID_formula\":\"id[p]=248044 if p==0 else 99+p; 0<=p<ROWS+32\",\"source_id_count\":" << total
            << ",\"teacher_id_count\":" << teacher << ",\"teacher_BOS\":248044,\"teacher_non_BOS_first\":100,\"teacher_last\":" << teacher + 98
            << ",\"continuation_id_count\":32,\"continuation_first\":" << teacher + 99 << ",\"continuation_last\":" << total + 98
            << ",\"every_source_ID_compile_time_and_runtime_checked\":true,\"expected_windows_per_phase\":[" << total << ',' << teacher / batch + 32 << ',' << 5 + mixed_chunks(teacher) + 32 << ']'
            << ",\"phase_order\":[\"reference_n1\",\"canonical1024\",\"occupied5_then997\"]"
            << ",\"teacher_schedules\":[\"ROWS_N1\",\"ROWS/1024_N1024\",\"5_N1_then_ceil((ROWS-5)/997)_chunks_final_remainder\"]"
            << ",\"mixed_final_remainder\":" << teacher - 5 - (mixed_chunks(teacher) - 1) * 997
            << ",\"continuation_schedule\":\"32_N1_each_phase\",\"baseline_checkpoint_count\":" << teacher / batch + 1
            << ",\"expected_success_records\":" << success_records(teacher)
            << ",\"expected_success_windows\":" << total + candidate_windows(teacher)
            << ",\"expected_finite_values\":" << 3ULL * total * vocabulary << ",\"expected_compared_values\":" << 2ULL * total * vocabulary
            << ",\"expected_memory_observations\":" << total + candidate_windows(teacher) + 21 << ",\"expected_invalid_rejections\":8"
            << ",\"gate\":{\"absolute\":0.02,\"relative\":0.002,\"formula\":\"abs(actual-reference)<=.02+.002*abs(reference)\",\"all_finite_required\":true,\"allowed_violations\":0,\"bit_equality_required\":false,\"argmax_equality_required\":false,\"cli_adjustable\":false}"
            << ",\"reference_scope\":\"all_ROWS_plus32_full_vocabulary_logits_retained_sequential_N1_same_Session\",\"vocabulary\":248320"
               ",\"weights\":\"unchanged_loaded_GGUF_values_and_tensor_types\",\"kv\":\"Q4_0_K_and_V\",\"sampling\":\"teacher_forced\""
               ",\"reset_retains_expert_cache\":true,\"cold_cache_equality_claim\":false,\"independent_HF_reference\":false"
               ",\"timing_scope\":\"diagnostic_correctness_completed_call_wall_excludes_comparison_memory_reset_rejection_not_PP_or_full_request_performance\""
               ",\"counter_scope\":\"480_assignments_per_row_hits_include_within_call_reuse_misses_count_uploaded_groups\""
            << ",\"uploaded_group_payload_min_bytes\":" << q40 << ",\"uploaded_group_payload_max_bytes\":" << q41
            << ",\"expert_payload_reads_required\":144,\"expert_payload_bytes_read_required\":" << payload
            << ",\"read_counter_scope\":\"constructor_payload_reads_not_physical_SSD_trace\",\"preallocated_reference_bytes\":" << reference_bytes
            << ",\"preallocated_preservation_bytes\":" << logit_bytes << ",\"float_guard_elements_each_end\":16,\"float_guard_total_bytes\":" << 4 * guards * sizeof(float)
            << ",\"guarded_input_workspace_bytes\":" << (batch + 1 + 2 * guards) * sizeof(std::int32_t)
            << ",\"known_host_min_bytes_before_embedding_metadata_runtime\":" << known_host_min
            << ",\"host_accounting\":\"expert_RAM_plus_reference_plus_preservation_plus_Session_logits_plus_pinned_handoff_and_expert_staging_no_weight_copy\""
            << ",\"workspace_aggregate_min_bytes_per_device\":" << workspace_floor
            << ",\"individual_buffer_capacities_observable\":false,\"physical_128_column_tile_proven\":false"
               ",\"coverage_scope\":\"actual_teacher_rows_and_call_offsets_cover_visible2047..2056_each_mod4_not_selected_ID_trace_or_synthetic_QKV\""
               ",\"performance_claim\":false,\"peak_VRAM_qualification_claim\":false,\"R4_complete_claim\":false"
               ",\"remaining_R4_evidence\":\"separate_full_request_512_output_benchmark_and_peak_VRAM_qualification\"";
    });
}
} // namespace

int main(int argc, char** argv) {
    Writer writer; Context context;
    try {
        require(argc == 3, "usage: core-prefill-long-test MODEL.gguf ROWS (exactly 4096 or 16384; explicit large-RAM opt-in)");
        const std::string_view rows_text(argv[2]); std::size_t teacher = 0;
        const auto parsed = std::from_chars(rows_text.data(), rows_text.data() + rows_text.size(), teacher);
        require(parsed.ec == std::errc{} && parsed.ptr == rows_text.data() + rows_text.size() &&
                (rows_text == "4096" || rows_text == "16384"), "ROWS must be exactly 4096 or 16384");
        const std::string_view revision(CORE_REVISION);
        require(std::all_of(revision.begin(), revision.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }), "CORE_REVISION not 40 hex digits");
        const std::string model(argv[1]); context.stage = "source_preflight";
        require(!model.empty() && std::filesystem::is_regular_file(model) && std::filesystem::file_size(model) > 0, "model is not a nonempty regular file");
        source_record(writer, model, teacher);
        if (teacher == 4096) run<4096>(model, writer, context); else run<16384>(model, writer, context);
        return 0;
    } catch (const std::exception& error) {
        try {
            writer.emit("failure", [&](auto& out) {
                context_fields(out, context); out << ",\"errors\":";
                if (context.inspected) json_errors(out, context.errors); else out << "null";
                out << ",\"session_constructed\":" << context.constructed << ",\"raii_session_and_fixture_cleanup_completed\":" << context.cleanup
                    << ",\"error\":"; quote(out, std::string_view(error.what()).substr(0, 2048)); out << ",\"passed\":false";
            });
        } catch (...) {}
        std::cerr << "core-prefill-long-test: " << std::string_view(error.what()).substr(0, 2048) << '\n';
        return 1;
    } catch (...) {
        try {
            writer.emit("failure", [&](auto& out) {
                context_fields(out, context);
                out << ",\"session_constructed\":" << context.constructed << ",\"raii_session_and_fixture_cleanup_completed\":" << context.cleanup
                    << ",\"error\":\"non-standard exception\",\"passed\":false";
            });
        } catch (...) {}
        return 1;
    }
}
