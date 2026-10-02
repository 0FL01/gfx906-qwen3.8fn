#include "session.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
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
#error "core-session-batch-test requires the compiled 40-hex CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-session-batch-test requires the compiled CORE_DIRTY (0 or 1)"
#endif

namespace {
constexpr int capacity = 40, expert_slots = 1, max_batch_tokens = 3;
constexpr std::size_t vocabulary = 248320, teacher_rows = 32, tail_rows = 8;
constexpr std::uint64_t routes_per_row = 48 * 10;
constexpr std::uint64_t inherited_hits_per_window = 48;
constexpr std::uint64_t q4_0_expert_bytes = 2764800, q4_1_expert_bytes = 2867200;
constexpr std::uint64_t ram_expert_payload = (6 * q4_1_expert_bytes + 42 * q4_0_expert_bytes) * 512;
constexpr std::uint64_t payload_reads = 48 * 3;
constexpr std::uint64_t min_q8_blocks = 960, q8_block_bytes = 36;
constexpr std::uint64_t scratch_buffers = 24, min_scratch_floats = 36864;
constexpr std::uint64_t host_logit_bytes = max_batch_tokens * vocabulary * sizeof(float);
constexpr std::uint64_t pinned_bytes = max_batch_tokens * 4 * 2560 * sizeof(float);
// SessionMemory does not expose individual workspace Buffer sizes. This is a
// necessary aggregate floor, NOT proof of each Q8/scratch Buffer's geometry.
constexpr std::uint64_t workspace_floor = scratch_buffers * min_scratch_floats * sizeof(float) +
                                          min_q8_blocks * q8_block_bytes + host_logit_bytes;
constexpr std::array<std::int32_t, teacher_rows> teacher_ids = [] {
    std::array<std::int32_t, teacher_rows> ids{};
    ids[0] = 248044;
    for (std::size_t i = 1; i < ids.size(); ++i) ids[i] = 99 + static_cast<std::int32_t>(i);
    return ids;
}();
constexpr std::array<std::int32_t, tail_rows> tail_ids{131, 132, 133, 134, 135, 136, 137, 138};
constexpr std::array<std::string_view, 5> schedule_names{
    "reference_n1", "replay_n1", "replay_n2", "replay_n3", "replay_mixed_123_prefix5"};
constexpr std::array<std::int32_t, 4> too_many_ids{100, 101, 102, 103};
constexpr std::array<std::int32_t, 3> negative_last{100, 101, -1};
constexpr std::array<std::int32_t, 3> vocabulary_last{100, 101, 248320};
constexpr std::array<std::int32_t, 3> capacity_ids{131, 132, 133};
static_assert(sizeof(float) == 4 && sizeof(double) == 8);
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
static_assert(std::string_view(CORE_REVISION).size() == 40);
static_assert(pinned_bytes == 122880 && workspace_floor == 6553344);

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
    require(!path.empty(), "model path is empty");
    require(std::filesystem::is_regular_file(path), "model is not a regular file");
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

void check_zero_stats(const qwen::SessionStats& stats, const std::string& where) {
    require(stats.consumed_tokens == 0 && stats.expert_hits == 0 && stats.expert_misses == 0 &&
            stats.expert_upload_bytes == 0 && stats.last_completed_ms == 0,
            where + ": reset statistics are not zero");
}

void check_same_stats(const qwen::SessionStats& actual, const qwen::SessionStats& expected,
                      const std::string& where) {
    require(actual.consumed_tokens == expected.consumed_tokens && actual.expert_hits == expected.expert_hits &&
            actual.expert_misses == expected.expert_misses && actual.expert_upload_bytes == expected.expert_upload_bytes &&
            std::bit_cast<std::uint64_t>(actual.last_completed_ms) ==
                std::bit_cast<std::uint64_t>(expected.last_completed_ms), where + ": statistics changed");
}

struct Routes {
    std::uint64_t hits = 0, misses = 0, upload_bytes = 0;
};

Routes route_delta(const qwen::SessionStats& after, const qwen::SessionStats& before) {
    require(after.expert_hits >= before.expert_hits && after.expert_misses >= before.expert_misses &&
            after.expert_upload_bytes >= before.expert_upload_bytes, "expert counters decreased");
    return {after.expert_hits - before.expert_hits, after.expert_misses - before.expert_misses,
            after.expert_upload_bytes - before.expert_upload_bytes};
}

Routes check_completed_stats(const qwen::SessionStats& after, const qwen::SessionStats& before,
                             std::size_t offset, std::size_t rows, const std::string& where) {
    require(before.consumed_tokens == offset && after.consumed_tokens == offset + rows,
            where + ": consumed count/advancement");
    require(std::isfinite(after.last_completed_ms) && after.last_completed_ms >= 0,
            where + ": invalid completed time");
    const auto delta = route_delta(after, before);
    const auto routes = routes_per_row * rows;
    require(delta.hits <= routes && delta.misses <= routes && delta.hits + delta.misses == routes,
            where + ": incomplete 480*N expert route accounting");
    require(delta.upload_bytes >= delta.misses * q4_0_expert_bytes &&
            delta.upload_bytes <= delta.misses * q4_1_expert_bytes,
            where + ": upload bytes inconsistent with expert miss payloads");
    if (rows == 1)
        require(delta.hits <= inherited_hits_per_window && delta.misses >= 48 * 9,
                where + ": one-slot N1 route bound violated");
    return delta;
}

void check_finite_logits(std::span<const float> logits, std::size_t rows, const std::string& where) {
    require(rows > 0 && rows <= max_batch_tokens && logits.size() == rows * vocabulary,
            where + ": token-major logit extent");
    for (std::size_t i = 0; i < logits.size(); ++i)
        if (!std::isfinite(logits[i]))
            throw std::runtime_error(where + ": nonfinite logit at row " + std::to_string(i / vocabulary) +
                                     ", vocabulary index " + std::to_string(i % vocabulary));
}

void check_exact_logits(std::span<const float> actual, std::span<const float> expected,
                        const std::string& where) {
    require(!actual.empty() && actual.size() == expected.size(), where + ": comparison extent");
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const auto a = std::bit_cast<std::uint32_t>(actual[i]);
        const auto b = std::bit_cast<std::uint32_t>(expected[i]);
        if (a != b)
            throw std::runtime_error(where + ": logit bits differ at row " + std::to_string(i / vocabulary) +
                                     ", vocabulary index " + std::to_string(i % vocabulary) +
                                     " (actual=" + std::to_string(a) + ", expected=" + std::to_string(b) + ")");
    }
}

void check_memory_geometry(const qwen::SessionMemory& memory, const std::string& where) {
    require(memory.capacity == capacity && memory.expert_slots == expert_slots && memory.ownership_verified,
            where + ": capacity/slots/ownership ledger");
    require(memory.ram_expert_payload == ram_expert_payload && memory.ram_expert_capacity >= memory.ram_expert_payload &&
            memory.expert_payload_reads == payload_reads && memory.expert_payload_bytes_read == memory.ram_expert_payload,
            where + ": RAM expert payload/read accounting");
    require(memory.host_embedding_capacity > 0 && memory.host_logit_capacity >= host_logit_bytes &&
            memory.pinned_handoff == pinned_bytes, where + ": host capacities/pinned geometry");
    for (std::size_t id = 0; id < memory.devices.size(); ++id) {
        const auto& d = memory.devices[id];
        require(d.device == static_cast<int>(id) && d.first_layer == static_cast<int>(id) * 24 &&
                d.last_layer == static_cast<int>(id) * 24 + 23 && d.gdn_layers == 18 && d.qsa_layers == 6,
                where + ": static 24/24 layer owners");
        const auto categories = d.weights + d.expert_slots + d.qsa_kv + d.qsa_index + d.gdn_state + d.ple_state + d.workspace;
        require(categories == d.owned_bytes && d.owned_bytes > 0 && d.owned_buffers > 0 &&
                d.owned_peak_bytes >= d.owned_bytes, where + ": category/self ledger");
        require(d.weights > 0 && d.expert_slots > 0 && d.qsa_kv > 0 && d.qsa_index > 0 && d.gdn_state > 0 &&
                d.workspace >= workspace_floor, where + ": missing category or aggregate workspace floor");
        require((id == 0 && d.ple_state > 0) || (id == 1 && d.ple_state == 0), where + ": PLE owner");
        require(d.total_vram > 0 && d.free_vram > 0 && d.free_vram <= d.total_vram && d.owned_bytes <= d.total_vram,
                where + ": invalid or exhausted VRAM");
    }
}

void check_steady_memory(const qwen::SessionMemory& actual, const qwen::SessionMemory& loaded,
                         const std::string& where) {
    check_memory_geometry(actual, where);
    require(actual.capacity == loaded.capacity && actual.expert_slots == loaded.expert_slots &&
            actual.ram_expert_capacity == loaded.ram_expert_capacity && actual.ram_expert_payload == loaded.ram_expert_payload &&
            actual.host_embedding_capacity == loaded.host_embedding_capacity && actual.host_logit_capacity == loaded.host_logit_capacity &&
            actual.pinned_handoff == loaded.pinned_handoff && actual.expert_payload_reads == loaded.expert_payload_reads &&
            actual.expert_payload_bytes_read == loaded.expert_payload_bytes_read && actual.ownership_verified == loaded.ownership_verified,
            where + ": host ledger or payload reads changed");
    for (std::size_t i = 0; i < actual.devices.size(); ++i) {
        const auto& a = actual.devices[i];
        const auto& b = loaded.devices[i];
        require(a.device == b.device && a.first_layer == b.first_layer && a.last_layer == b.last_layer &&
                a.gdn_layers == b.gdn_layers && a.qsa_layers == b.qsa_layers && a.weights == b.weights &&
                a.expert_slots == b.expert_slots && a.qsa_kv == b.qsa_kv && a.qsa_index == b.qsa_index &&
                a.gdn_state == b.gdn_state && a.ple_state == b.ple_state && a.workspace == b.workspace &&
                a.owned_bytes == b.owned_bytes && a.owned_peak_bytes == b.owned_peak_bytes &&
                a.owned_buffers == b.owned_buffers && a.total_vram == b.total_vram,
                where + ": GPU categories/count/peak ledger changed");
        // HIP/context/rocBLAS free VRAM may vary; it is not an owned Buffer ledger.
    }
}

struct MemoryAudit {
    const qwen::SessionMemory& loaded;
    qwen::SessionMemory last{};
    std::array<std::uint64_t, 2> minimum_free{};
    std::uint64_t snapshots = 0, logit_values_preserved = 0;

    explicit MemoryAudit(const qwen::SessionMemory& anchor) : loaded(anchor), last(anchor),
        minimum_free{anchor.devices[0].free_vram, anchor.devices[1].free_vram} {}

    void observe(qwen::Session& session, std::span<const float> prior, std::span<const float> saved,
                 const std::string& where) {
        const auto before = session.stats();
        last = session.memory();
        check_same_stats(session.stats(), before, where + ": memory()");
        if (!prior.empty()) {
            check_exact_logits(prior, saved, where + ": memory() logits");
            logit_values_preserved += prior.size();
        }
        check_steady_memory(last, loaded, where);
        for (std::size_t i = 0; i < minimum_free.size(); ++i)
            minimum_free[i] = std::min(minimum_free[i], last.devices[i].free_vram);
        ++snapshots;
    }
};

struct References {
    // Allocate once, before the only Session: 31,784,960 teacher bytes, plus
    // 7,946,240 tail bytes for exact continuation through the capacity boundary.
    std::vector<float> teacher = std::vector<float>(teacher_rows * vocabulary);
    std::vector<float> tail = std::vector<float>(tail_rows * vocabulary);
    std::vector<float> saved = std::vector<float>(max_batch_tokens * vocabulary);
    std::array<qwen::SessionStats, capacity> stats{};

    std::span<float> slice(std::size_t offset, std::size_t rows) {
        require(rows > 0 && offset < capacity && rows <= static_cast<std::size_t>(capacity) - offset,
                "reference range outside timeline");
        if (offset < teacher_rows) {
            require(rows <= teacher_rows - offset, "reference window crosses teacher/tail boundary");
            return std::span<float>(teacher).subspan(offset * vocabulary, rows * vocabulary);
        }
        return std::span<float>(tail).subspan((offset - teacher_rows) * vocabulary, rows * vocabulary);
    }

    std::span<const float> snapshot(std::span<const float> logits) {
        require(logits.size() <= saved.size(), "snapshot extent exceeds preallocated storage");
        std::copy(logits.begin(), logits.end(), saved.begin());
        return std::span<const float>(saved).first(logits.size());
    }

    Routes routes(std::size_t offset, std::size_t rows) const {
        require(rows > 0 && offset + rows <= stats.size(), "reference route range");
        return route_delta(stats[offset + rows - 1], offset == 0 ? qwen::SessionStats{} : stats[offset - 1]);
    }
};

struct Window {
    std::size_t offset = 0, rows = 0, requested_rows = 0;
    Routes delta{}, reference{};
    qwen::SessionStats stats{};
};

struct InvalidWindow {
    std::string_view name;
    std::size_t rows = 0, prior_rows = 0, continuation_rows = 0;
    qwen::SessionStats stats{};
    bool continued = false, reset_before_continuation = false;
};

struct Reuse {
    bool applicable = false;
    std::size_t first_row = 0, rows = 0;
    Routes reference{}, candidate{};
    std::int64_t hits_gained = 0, misses_saved = 0;
    std::uint64_t within_window_hit_lower_bound = 0;
};

struct Schedule {
    std::string_view name;
    std::array<Window, capacity> windows{};
    std::array<InvalidWindow, 7> invalid{};
    std::size_t window_count = 0, teacher_window_count = 0, invalid_count = 0, partial_windows = 0;
    std::uint64_t finite_values = 0, compared_values = 0, rejection_values_preserved = 0;
    qwen::SessionStats initial{}, teacher_stats{}, timeline_stats{}, final_reset{};
    Window reset_probe{};
    bool has_reset_probe = false;
    Reuse reuse{};
};

void expect_rejection(qwen::Session& session, References& reference, MemoryAudit& audit, Schedule& result,
                      std::span<const std::int32_t> tokens, std::span<const float> prior,
                      std::string_view name) {
    require(!prior.empty() && result.invalid_count < result.invalid.size(), "invalid fixture storage/previous result");
    const auto where = std::string(result.name) + ": " + std::string(name);
    const auto saved = reference.snapshot(prior);
    const auto before = session.stats();
    audit.observe(session, prior, saved, where + " before");
    bool rejected = false;
    try {
        static_cast<void>(session.step_batch(tokens));
    } catch (const std::invalid_argument&) {
        rejected = true;
    } catch (const std::exception& error) {
        throw std::runtime_error(where + ": expected std::invalid_argument, got " + error.what());
    }
    require(rejected, where + ": rejected input was accepted");
    check_same_stats(session.stats(), before, where);
    check_exact_logits(prior, saved, where + ": prior full logits");
    audit.observe(session, prior, saved, where + " after");
    result.invalid[result.invalid_count++] = {name, tokens.size(), prior.size() / vocabulary, 0, before, false, false};
    result.rejection_values_preserved += prior.size();
}

void prove_continuation(Schedule& result, std::size_t offset, std::size_t rows, bool reset) {
    for (std::size_t i = 0; i < result.invalid_count; ++i) {
        auto& rejected = result.invalid[i];
        if (rejected.continued) continue;
        require((!reset && rejected.stats.consumed_tokens == offset) ||
                (reset && rejected.stats.consumed_tokens == capacity && offset == 0),
                "continuation did not start at the unconsumed rejected position");
        rejected.continued = true;
        rejected.reset_before_continuation = reset;
        rejected.continuation_rows = rows;
    }
}

Schedule run_schedule(qwen::Session& session, References& reference, MemoryAudit& audit, std::size_t variant) {
    require(variant < schedule_names.size(), "unknown schedule");
    const bool baseline = variant == 0;
    Schedule result;
    result.name = schedule_names[variant];
    result.initial = session.stats();
    check_zero_stats(result.initial, std::string(result.name) + ": start");
    audit.observe(session, {}, {}, std::string(result.name) + ": start");
    std::span<const float> prior;
    std::size_t offset = 0, mixed_index = 0;
    bool ordinary_rejections = false;
    const auto requested_width = [&]() -> std::size_t {
        if (variant <= 1) return 1;
        if (variant == 2) return 2;
        if (variant == 3) return 3;
        if (offset < 5) return 1;
        constexpr std::array<std::size_t, 3> cycle{2, 3, 1};
        return cycle[mixed_index++ % cycle.size()];
    };
    const auto accept = [&](std::size_t rows, std::size_t requested) {
        require(result.window_count < result.windows.size(), "window record storage exhausted");
        const auto where = std::string(result.name) + " row " + std::to_string(offset);
        const auto ids = offset < teacher_rows ? std::span<const std::int32_t>(teacher_ids).subspan(offset, rows) :
            std::span<const std::int32_t>(tail_ids).subspan(offset - teacher_rows, rows);
        const auto before = session.stats();
        // Both sequential passes exercise the public N1 adapter, all other
        // passes exercise step_batch, including their partial N1 windows.
        prior = variant <= 1 ? session.step(ids[0]) : session.step_batch(ids);
        check_finite_logits(prior, rows, where);
        const auto after = session.stats();
        const auto delta = check_completed_stats(after, before, offset, rows, where);
        result.finite_values += prior.size();
        if (baseline) {
            require(rows == 1, "reference must be sequential N1");
            const auto destination = reference.slice(offset, rows);
            std::copy(prior.begin(), prior.end(), destination.begin());
            reference.stats[offset] = after;
        } else {
            check_exact_logits(prior, reference.slice(offset, rows), where);
            result.compared_values += prior.size();
            prove_continuation(result, offset, rows, false);
        }
        const auto saved = reference.snapshot(prior);
        audit.observe(session, prior, saved, where);
        result.windows[result.window_count++] = {offset, rows, requested, delta, reference.routes(offset, rows), after};
        if (rows < requested) ++result.partial_windows;
        if (offset < teacher_rows) {
            ++result.teacher_window_count;
            if (rows > 1 && delta.hits > inherited_hits_per_window)
                result.reuse.within_window_hit_lower_bound += delta.hits - inherited_hits_per_window;
            if (rows > 1 && result.reuse.first_row == 0) result.reuse.first_row = offset + rows;
        }
        offset += rows;
    };

    while (offset < teacher_rows) {
        const auto requested = requested_width();
        accept(std::min(requested, teacher_rows - offset), requested);
        if (!baseline && !ordinary_rejections && (variant != 4 || (offset >= 5 && prior.size() == 3 * vocabulary))) {
            // All valid prefixes fit and IDs are checked in the LAST element.
            expect_rejection(session, reference, audit, result, {}, prior, "empty");
            expect_rejection(session, reference, audit, result, too_many_ids, prior, "length4");
            expect_rejection(session, reference, audit, result, negative_last, prior, "negative_last");
            expect_rejection(session, reference, audit, result, vocabulary_last, prior, "vocabulary_last");
            ordinary_rejections = true;
        }
    }
    require(baseline || ordinary_rejections, "ordinary invalid-window fixture was not exercised");
    result.teacher_stats = session.stats();
    if (variant >= 2) {
        // Every layer has >=10 distinct groups in the first grouped window,
        // overwriting its one retained slot. Compare the matching occupied
        // suffix, not reset-retained, possibly different initial slot contents.
        // Later slot differences are actual consequences of the two schedules.
        auto& reuse = result.reuse;
        require(reuse.first_row > 0 && reuse.first_row < teacher_rows, "reuse comparison has no occupied suffix");
        reuse.applicable = true;
        reuse.rows = teacher_rows - reuse.first_row;
        reuse.reference = reference.routes(reuse.first_row, reuse.rows);
        for (std::size_t i = 0; i < result.teacher_window_count; ++i) {
            const auto& window = result.windows[i];
            if (window.offset < reuse.first_row) continue;
            reuse.candidate.hits += window.delta.hits;
            reuse.candidate.misses += window.delta.misses;
            reuse.candidate.upload_bytes += window.delta.upload_bytes;
        }
        require(reuse.reference.hits + reuse.reference.misses == routes_per_row * reuse.rows &&
                reuse.candidate.hits + reuse.candidate.misses == routes_per_row * reuse.rows, "reuse route scope mismatch");
        reuse.hits_gained = static_cast<std::int64_t>(reuse.candidate.hits) - static_cast<std::int64_t>(reuse.reference.hits);
        reuse.misses_saved = static_cast<std::int64_t>(reuse.reference.misses) - static_cast<std::int64_t>(reuse.candidate.misses);
        require(reuse.within_window_hit_lower_bound > 0 && reuse.hits_gained > 0 && reuse.misses_saved > 0,
                std::string(result.name) + ": no positive observed one-slot grouped reuse against matching N1 suffix");
    }

    // Stop at 39 to reject both N2 and N3 with exactly ONE position remaining,
    // then consume that position without reset and compare it to the clean N1.
    while (offset < capacity - 1) {
        const auto requested = requested_width();
        accept(std::min(requested, static_cast<std::size_t>(capacity - 1) - offset), requested);
    }
    if (!baseline) {
        expect_rejection(session, reference, audit, result, std::span<const std::int32_t>(capacity_ids).first(2),
                         prior, "capacity39_length2");
        expect_rejection(session, reference, audit, result, capacity_ids, prior, "capacity39_length3");
    }
    accept(1, requested_width());
    require(offset == capacity && result.finite_values == capacity * vocabulary &&
            result.compared_values == (baseline ? 0 : capacity * vocabulary), "timeline logit counters");
    result.timeline_stats = session.stats();
    if (!baseline)
        expect_rejection(session, reference, audit, result, std::span<const std::int32_t>(capacity_ids).first(2),
                         prior, "capacity40_length2");

    // Full capacity cannot have a same-history continuation. Its proof is an
    // explicitly labelled reset + exact first-token replay, on the SAME Session.
    prior = {};
    session.reset();
    check_zero_stats(session.stats(), std::string(result.name) + ": reset");
    audit.observe(session, {}, {}, std::string(result.name) + ": reset");
    if (!baseline) {
        const auto before = session.stats();
        prior = session.step(teacher_ids[0]);
        check_finite_logits(prior, 1, std::string(result.name) + ": reset continuation");
        check_exact_logits(prior, reference.slice(0, 1), std::string(result.name) + ": reset continuation");
        const auto after = session.stats();
        const auto delta = check_completed_stats(after, before, 0, 1, std::string(result.name) + ": reset continuation");
        prove_continuation(result, 0, 1, true);
        result.reset_probe = {0, 1, 1, delta, reference.routes(0, 1), after};
        result.has_reset_probe = true;
        audit.observe(session, prior, reference.snapshot(prior), std::string(result.name) + ": reset continuation");
        prior = {};
        session.reset();
        check_zero_stats(session.stats(), std::string(result.name) + ": final reset");
        audit.observe(session, {}, {}, std::string(result.name) + ": final reset");
        require(result.invalid_count == result.invalid.size(), "invalid proof count");
        for (std::size_t i = 0; i < result.invalid_count; ++i)
            require(result.invalid[i].continued, "invalid window has no checked continuation");
    }
    result.final_reset = session.stats();
    return result;
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

template<class T, std::size_t N>
void json_ids(std::ostream& out, const std::array<T, N>& ids) {
    out << '[';
    for (std::size_t i = 0; i < ids.size(); ++i) { if (i) out << ','; out << ids[i]; }
    out << ']';
}

void json_stats(std::ostream& out, const qwen::SessionStats& stats) {
    require(std::isfinite(stats.last_completed_ms) && stats.last_completed_ms >= 0, "nonfinite JSON statistics");
    out << "{\"consumed_tokens\":" << stats.consumed_tokens << ",\"expert_hits\":" << stats.expert_hits
        << ",\"expert_misses\":" << stats.expert_misses << ",\"expert_upload_bytes\":" << stats.expert_upload_bytes
        << ",\"last_completed_ms\":" << stats.last_completed_ms << '}';
}

void json_routes(std::ostream& out, const Routes& routes) {
    out << "{\"hits\":" << routes.hits << ",\"misses\":" << routes.misses
        << ",\"upload_bytes\":" << routes.upload_bytes << '}';
}

void json_window(std::ostream& out, const Window& window) {
    out << "{\"offset\":" << window.offset << ",\"rows\":" << window.rows
        << ",\"requested_rows\":" << window.requested_rows << ",\"routes\":" << routes_per_row * window.rows
        << ",\"delta\":";
    json_routes(out, window.delta);
    out << ",\"n1_reference_delta\":";
    json_routes(out, window.reference);
    out << ",\"stats\":";
    json_stats(out, window.stats);
    out << '}';
}

void json_memory(std::ostream& out, const qwen::SessionMemory& memory) {
    out << "{\"capacity\":" << memory.capacity << ",\"expert_slots\":" << memory.expert_slots
        << ",\"ownership_verified\":" << memory.ownership_verified
        << ",\"ram_expert_capacity\":" << memory.ram_expert_capacity << ",\"ram_expert_payload\":" << memory.ram_expert_payload
        << ",\"host_embedding_capacity\":" << memory.host_embedding_capacity << ",\"host_logit_capacity\":" << memory.host_logit_capacity
        << ",\"pinned_handoff\":" << memory.pinned_handoff << ",\"expert_payload_reads\":" << memory.expert_payload_reads
        << ",\"expert_payload_bytes_read\":" << memory.expert_payload_bytes_read << ",\"devices\":[";
    for (std::size_t i = 0; i < memory.devices.size(); ++i) {
        if (i) out << ',';
        const auto& d = memory.devices[i];
        out << "{\"device\":" << d.device << ",\"first_layer\":" << d.first_layer << ",\"last_layer\":" << d.last_layer
            << ",\"gdn_layers\":" << d.gdn_layers << ",\"qsa_layers\":" << d.qsa_layers
            << ",\"weights\":" << d.weights << ",\"expert_slots\":" << d.expert_slots << ",\"qsa_kv\":" << d.qsa_kv
            << ",\"qsa_index\":" << d.qsa_index << ",\"gdn_state\":" << d.gdn_state << ",\"ple_state\":" << d.ple_state
            << ",\"workspace\":" << d.workspace << ",\"owned_bytes\":" << d.owned_bytes
            << ",\"owned_peak_bytes\":" << d.owned_peak_bytes << ",\"owned_buffers\":" << d.owned_buffers
            << ",\"total_vram\":" << d.total_vram << ",\"free_vram\":" << d.free_vram << '}';
    }
    out << "]}";
}

void emit(std::ostringstream& record) {
    record << '\n';
    require(static_cast<bool>(record), "JSONL formatting failed");
    std::cout << record.str();
    std::cout.flush();
    require(static_cast<bool>(std::cout), "JSONL stdout write failed");
}

void format(std::ostringstream& record) {
    record.imbue(std::locale::classic());
    record << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10);
}

void emit_source(const std::string& model, std::uint64_t model_bytes) {
    std::ostringstream out;
    format(out);
    out << "{\"kind\":\"session_batch_source\",\"protocol\":1,\"revision\":";
    json_string(out, CORE_REVISION);
    out << ",\"dirty\":" << (CORE_DIRTY != 0) << ",\"model\":";
    json_string(out, model);
    out << ",\"model_bytes\":" << model_bytes
        << ",\"runtime\":\"own_48_layer_HIP\",\"config\":{\"capacity\":40,\"expert_slots\":1,\"max_batch_tokens\":3,\"trace\":false}"
           ",\"teacher_ids\":";
    json_ids(out, teacher_ids);
    out << ",\"capacity_tail_ids\":";
    json_ids(out, tail_ids);
    out << ",\"schedule_order\":[";
    for (std::size_t i = 0; i < schedule_names.size(); ++i) { if (i) out << ','; json_string(out, schedule_names[i]); }
    out << "],\"scope\":\"exact_causal_short_window_replay_before_PP\",\"comparison\":\"all_FP32_bits_no_tolerance\""
           ",\"reference_scope\":\"retained_sequential_N1_same_Session_unchanged_weights_and_gates\""
           ",\"sampling\":\"teacher_forced\",\"performance_claim\":false,\"session_instances\":1"
           ",\"reset_clears_weight_slots\":false,\"cache_start_equality_claim\":false"
           ",\"reuse_scope\":\"matching_teacher_suffix_after_first_grouped_window_initial_slots_overwritten\""
           ",\"grouping_scope\":\"within_call_only_all_48_layers_10_routes_per_row\""
           ",\"counter_definitions\":{\"routes_per_row\":480,\"max_inherited_slot_hits_per_window\":48"
           ",\"timeline_rows_per_schedule\":40,\"teacher_rows_per_schedule\":32,\"tail_rows_per_schedule\":8"
           ",\"logit_values_per_row\":248320,\"invalid_windows_per_replay\":7"
           ",\"bitwise_count_scope\":\"timeline_and_reset_probe_only_excludes_preservation_rechecks\""
           ",\"route_stats_scope\":\"timeline_cumulative_reset_probe_separate\""
           ",\"within_window_hit_lower_bound_scope\":\"sum_max_0_window_hits_minus_48_all_teacher_windows\"}"
           ",\"geometry\":{\"min_q8_blocks_per_device\":" << min_q8_blocks
        << ",\"q8_block_bytes\":" << q8_block_bytes << ",\"scratch_buffers_per_device\":" << scratch_buffers
        << ",\"min_scratch_floats_per_buffer\":" << min_scratch_floats << ",\"min_host_logit_bytes\":" << host_logit_bytes
        << ",\"pinned_handoff_bytes\":" << pinned_bytes << ",\"min_workspace_aggregate_bytes_per_device\":" << workspace_floor
        << ",\"individual_q8_scratch_geometry_observable\":false,\"workspace_check\":\"aggregate_floor_only\""
           ",\"free_vram_equality_required\":false"
           ",\"ledger_scope\":\"Session_owned_buffers_and_reported_host_capacities_excludes_allocator_metadata\"}"
           ",\"preallocated_reference_bytes\":" << teacher_rows * vocabulary * sizeof(float)
        << ",\"preallocated_tail_reference_bytes\":" << tail_rows * vocabulary * sizeof(float)
        << ",\"preallocated_snapshot_bytes\":" << host_logit_bytes << '}';
    emit(out);
}

void emit_schedule(const Schedule& result, const MemoryAudit& audit, std::size_t index) {
    std::ostringstream out;
    format(out);
    out << "{\"kind\":\"session_batch_schedule\",\"protocol\":1,\"index\":" << index << ",\"schedule\":";
    json_string(out, result.name);
    out << ",\"reference\":" << (index == 0) << ",\"occupied_prefix_n1_rows\":" << (index == 4 ? 5 : 0)
        << ",\"counts\":{\"teacher_rows\":32,\"tail_rows\":8,\"rows\":40,\"finite_rows\":40"
        << ",\"bitwise_compared_rows\":" << (index == 0 ? 0 : capacity)
        << ",\"teacher_finite_logit_values\":" << teacher_rows * vocabulary
        << ",\"teacher_bitwise_compared_logit_values\":" << (index == 0 ? 0 : teacher_rows * vocabulary)
        << ",\"tail_finite_logit_values\":" << tail_rows * vocabulary
        << ",\"tail_bitwise_compared_logit_values\":" << (index == 0 ? 0 : tail_rows * vocabulary)
        << ",\"finite_logit_values\":" << result.finite_values
        << ",\"bitwise_compared_logit_values\":" << result.compared_values << ",\"windows\":" << result.window_count
        << ",\"teacher_windows\":" << result.teacher_window_count << ",\"tail_windows\":" << result.window_count - result.teacher_window_count
        << ",\"partial_windows\":" << result.partial_windows << ",\"routes\":19200,\"invalid_windows\":" << result.invalid_count
        << ",\"rejection_preserved_logit_values\":" << result.rejection_values_preserved << '}'
        << ",\"initial_stats\":";
    json_stats(out, result.initial);
    out << ",\"teacher_stats\":";
    json_stats(out, result.teacher_stats);
    out << ",\"timeline_stats\":";
    json_stats(out, result.timeline_stats);
    out << ",\"windows\":[";
    for (std::size_t i = 0; i < result.window_count; ++i) { if (i) out << ','; json_window(out, result.windows[i]); }
    const auto& reuse = result.reuse;
    out << "],\"reuse\":{\"applicable\":" << reuse.applicable << ",\"first_row\":" << reuse.first_row
        << ",\"rows\":" << reuse.rows << ",\"n1_reference\":";
    json_routes(out, reuse.reference);
    out << ",\"candidate\":";
    json_routes(out, reuse.candidate);
    out << ",\"hits_gained\":" << reuse.hits_gained << ",\"misses_saved\":" << reuse.misses_saved
        << ",\"teacher_within_window_hit_lower_bound\":" << reuse.within_window_hit_lower_bound
        << ",\"initial_cache_contents_matched\":false,\"performance_claim\":false}"
           ",\"invalid_proofs\":[";
    for (std::size_t i = 0; i < result.invalid_count; ++i) {
        if (i) out << ',';
        const auto& rejected = result.invalid[i];
        out << "{\"case\":";
        json_string(out, rejected.name);
        out << ",\"rows\":" << rejected.rows << ",\"prior_rows\":" << rejected.prior_rows
            << ",\"preserved_logit_values\":" << rejected.prior_rows * vocabulary
            << ",\"stats_before_and_after\":";
        json_stats(out, rejected.stats);
        out << ",\"exception\":\"invalid_argument\",\"stats_bitwise_preserved\":true,\"logits_bitwise_preserved\":true"
               ",\"memory_ledger_preserved\":true,\"consumed_advance\":0,\"continued\":" << rejected.continued
            << ",\"reset_before_continuation\":" << rejected.reset_before_continuation
            << ",\"continuation_rows\":" << rejected.continuation_rows
            << ",\"continuation_bitwise_logit_values\":" << rejected.continuation_rows * vocabulary << '}';
    }
    out << "],\"reset_probe\":{\"rows\":" << (result.has_reset_probe ? 1 : 0)
        << ",\"finite_logit_values\":" << (result.has_reset_probe ? vocabulary : 0)
        << ",\"bitwise_compared_logit_values\":" << (result.has_reset_probe ? vocabulary : 0) << ",\"window\":";
    if (result.has_reset_probe) json_window(out, result.reset_probe); else out << "null";
    out << "},\"final_reset_stats\":";
    json_stats(out, result.final_reset);
    out << ",\"memory\":{\"snapshot_count\":" << audit.snapshots << ",\"steady_comparisons_to_loaded\":" << audit.snapshots
        << ",\"stats_preserved_comparisons\":" << audit.snapshots
        << ",\"bitwise_preserved_logit_values\":" << audit.logit_values_preserved
        << ",\"minimum_free_vram_bytes\":";
    json_ids(out, audit.minimum_free);
    out << ",\"steady_categories_counts_peak_host_and_payload_reads\":true,\"final_snapshot\":";
    json_memory(out, audit.last);
    out << ",\"loaded_anchor_snapshot\":";
    if (index == 0) json_memory(out, audit.loaded); else out << "null";
    out << "},\"passed\":true}";
    emit(out);
}

struct Totals {
    std::uint64_t windows = 0, finite_values = 0, compared_values = 0, invalid_windows = 0;
    std::uint64_t rejection_values = 0, memory_snapshots = 0, memory_preserved_values = 0;
    std::uint64_t reset_probe_rows = 0, reset_probe_finite_values = 0, reset_probe_compared_values = 0;
};

void emit_complete(const Totals& totals) {
    require(totals.finite_values == schedule_names.size() * capacity * vocabulary &&
            totals.compared_values == (schedule_names.size() - 1) * capacity * vocabulary &&
            totals.windows == 140 && totals.invalid_windows == 28 && totals.reset_probe_rows == 4 &&
            totals.reset_probe_finite_values == 4 * vocabulary && totals.reset_probe_compared_values == 4 * vocabulary,
            "final protocol counters");
    std::ostringstream out;
    format(out);
    out << "{\"kind\":\"session_batch_complete\",\"protocol\":1,\"schedule_count\":5,\"record_count\":7"
           ",\"timeline_rows\":200,\"teacher_rows\":160,\"tail_rows\":40,\"finite_rows\":200,\"bitwise_compared_rows\":160"
           ",\"timeline_routes\":96000"
           ",\"window_count\":" << totals.windows << ",\"finite_logit_values\":" << totals.finite_values
        << ",\"bitwise_compared_logit_values\":" << totals.compared_values
        << ",\"reset_probe_rows\":" << totals.reset_probe_rows << ",\"reset_probe_routes\":" << totals.reset_probe_rows * routes_per_row
        << ",\"reset_probe_finite_logit_values\":" << totals.reset_probe_finite_values
        << ",\"reset_probe_bitwise_compared_logit_values\":" << totals.reset_probe_compared_values
        << ",\"invalid_window_rejections\":" << totals.invalid_windows
        << ",\"rejection_preserved_logit_values\":" << totals.rejection_values
        << ",\"loaded_memory_snapshot_count\":1,\"schedule_memory_snapshot_count\":" << totals.memory_snapshots
        << ",\"memory_snapshot_count\":" << totals.memory_snapshots + 1
        << ",\"memory_bitwise_preserved_logit_values\":" << totals.memory_preserved_values
        << ",\"positive_grouped_reuse_schedules\":3,\"session_instances\":1"
           ",\"raii_session_cleanup_completed\":true,\"owned_buffer_release_measured\":false"
           ",\"performance_claim\":false,\"passed\":true}";
    emit(out);
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: core-session-batch-test MODEL.gguf\n";
        return 2;
    }
    try {
        check_revision();
        const std::string model(argv[1]);
        const auto model_bytes = check_model_file(model);
        qwen::SessionConfig config;
        config.capacity = capacity;
        config.expert_slots = expert_slots;
        config.max_batch_tokens = max_batch_tokens;
        config.trace_directory.clear();
        Totals totals;
        {
            References reference;
            emit_source(model, model_bytes);
            // Exactly one model/Session. reset() intentionally retains immutable
            // weight caches; never create a second ~68 GB RAM expert inventory.
            qwen::Session session(model, config);
            check_zero_stats(session.stats(), "construction");
            const auto before = session.stats();
            const auto loaded = session.memory();
            check_same_stats(session.stats(), before, "loaded memory()");
            check_memory_geometry(loaded, "loaded");
            for (std::size_t i = 0; i < schedule_names.size(); ++i) {
                MemoryAudit audit(loaded);
                const auto result = run_schedule(session, reference, audit, i);
                emit_schedule(result, audit, i);
                totals.windows += result.window_count;
                totals.finite_values += result.finite_values;
                totals.compared_values += result.compared_values;
                totals.invalid_windows += result.invalid_count;
                totals.rejection_values += result.rejection_values_preserved;
                totals.memory_snapshots += audit.snapshots;
                totals.memory_preserved_values += audit.logit_values_preserved;
                if (result.has_reset_probe) {
                    ++totals.reset_probe_rows;
                    totals.reset_probe_finite_values += vocabulary;
                    totals.reset_probe_compared_values += vocabulary;
                }
            }
        } // Session AND all retained host snapshots destroyed before success footer.
        emit_complete(totals);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core-session-batch-test: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "core-session-batch-test: non-standard exception\n";
        return 1;
    }
}
