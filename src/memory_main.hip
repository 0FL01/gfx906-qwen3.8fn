#include "session.hpp"

#include <hip/hip_runtime_api.h>
#include <sys/resource.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
constexpr int capacity = 131072, expert_slots = 112;
constexpr std::size_t vocabulary = 248320;
constexpr std::array<std::int32_t, 2> teacher_ids{248044, 100};
constexpr std::array<std::string_view, 6> phases{
    "loaded", "step0", "step1", "reset", "replay0", "replay1"};
constexpr std::uint64_t mib = 1024ULL * 1024;
// Frozen acceptance gates, not adjustable CLI options or measured speed targets.
constexpr std::uint64_t min_free_headroom = 1024 * mib;
constexpr std::uint64_t qsa_kv_bytes = 432 * mib;
constexpr std::uint64_t qsa_pooled_bytes = 96 * mib;
constexpr std::uint64_t qsa_tail_bytes = 6ULL * 384 * 4;
constexpr std::uint64_t gdn_state_bytes = 18ULL * (786432 + 30720) * 4;
constexpr std::array<std::uint64_t, 2> ple_state_bytes{92160 * 4, 0};
// Actual target types: Q4_0 gate/up; Q4_1 down on layers 0..5,
// Q4_0 down on layers 6..47. These are payload bytes without padding.
constexpr std::uint64_t q4_1_expert_bytes = 2867200, q4_0_expert_bytes = 2764800;
constexpr std::array<std::uint64_t, 2> slot_bytes{
    (6 * q4_1_expert_bytes + 18 * q4_0_expert_bytes) * expert_slots,
    24 * q4_0_expert_bytes * expert_slots};
constexpr std::uint64_t ram_expert_payload =
    (6 * q4_1_expert_bytes + 42 * q4_0_expert_bytes) * 512;
constexpr std::uint64_t payload_reads = 48 * 3;
constexpr std::uint64_t logit_payload = vocabulary * sizeof(float);
constexpr std::uint64_t pinned_handoff = 4 * 2560 * sizeof(float);
constexpr std::uint64_t routes_per_step = 48 * 10;
static_assert(sizeof(float) == 4 && sizeof(double) == 8);
static_assert(qsa_kv_bytes * 2 == 864 * mib && qsa_pooled_bytes * 2 == 192 * mib);
#ifdef CORE_DIRTY
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
#endif

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void hip_check(hipError_t status, const char* operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

void hip_cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) {
        std::fprintf(stderr, "core-memory: HIP device restoration: %s\n", hipGetErrorString(status));
        std::abort();
    }
}

int current_device() {
    int device = -1;
    hip_check(hipGetDevice(&device), "hipGetDevice");
    return device;
}

class RestoreDevice {
public:
    RestoreDevice() : device_(current_device()) {}
    ~RestoreDevice() { if (active_) hip_cleanup(hipSetDevice(device_)); }
    RestoreDevice(const RestoreDevice&) = delete;
    RestoreDevice& operator=(const RestoreDevice&) = delete;
    int device() const { return device_; }
    void restore_checked() {
        hip_check(hipSetDevice(device_), "hipSetDevice restore");
        require(current_device() == device_, "caller device restoration failed");
        active_ = false;
    }
private:
    int device_;
    bool active_ = true;
};

struct DeviceProbe {
    int device = 0, wave_size = 0, compute_units = 0, async_engines = 0;
    std::string name, arch;
    std::uint64_t properties_total_vram = 0, total_vram = 0, free_vram = 0;
};

template<std::size_t N>
std::string checked_c_string(const char (&text)[N]) {
    const auto end = std::find(text, text + N, '\0');
    require(end != text + N, "unterminated HIP property string");
    return std::string(text, end);
}

std::array<DeviceProbe, 2> probe_devices() {
    RestoreDevice restore;
    std::array<DeviceProbe, 2> result{};
    int count = 0;
    hip_check(hipGetDeviceCount(&count), "hipGetDeviceCount");
    require(count == 2, "core-memory requires exactly two HIP devices");
    for (int id = 0; id < 2; ++id) {
        hip_check(hipSetDevice(id), "hipSetDevice probe");
        // Also required after destruction: complete outstanding work on BOTH
        // devices before measuring the returned free bytes. Never reset a device.
        hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize probe");
        hipDeviceProp_t properties{};
        hip_check(hipGetDeviceProperties(&properties, id), "hipGetDeviceProperties");
        std::size_t free = 0, total = 0;
        hip_check(hipMemGetInfo(&free, &total), "hipMemGetInfo probe");
        auto& out = result[static_cast<std::size_t>(id)];
        out.device = id;
        out.name = checked_c_string(properties.name);
        out.arch = checked_c_string(properties.gcnArchName);
        out.wave_size = properties.warpSize;
        out.compute_units = properties.multiProcessorCount;
        out.async_engines = properties.asyncEngineCount;
        out.properties_total_vram = properties.totalGlobalMem;
        out.total_vram = total;
        out.free_vram = free;
        require(out.wave_size == 64 && (out.arch == "gfx906" || out.arch.starts_with("gfx906:")),
                "GPU " + std::to_string(id) + ": requires gfx906 wave64");
        require(total > 0 && free <= total && out.properties_total_vram == total,
                "GPU " + std::to_string(id) + ": inconsistent HIP memory properties");
        require(free >= min_free_headroom, "GPU " + std::to_string(id) + ": probe headroom below 1 GiB");
    }
    restore.restore_checked();
    return result;
}

std::uint64_t peak_rss_bytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        throw std::runtime_error(std::string("getrusage: ") + std::strerror(errno));
    require(usage.ru_maxrss >= 0, "getrusage returned negative peak RSS");
    const auto kib = static_cast<std::uint64_t>(usage.ru_maxrss);
    require(kib <= std::numeric_limits<std::uint64_t>::max() / 1024, "peak RSS byte overflow");
    // Linux ru_maxrss is KiB; RSS is observed process high-water usage, not a
    // fictional sum of the reported host capacities and other model storage.
    return kib * 1024;
}

void equal_bytes(std::uint64_t actual, std::uint64_t expected,
                 const std::string& where, const char* field) {
    require(actual == expected, where + ": " + field + " actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

void check_zero_stats(const qwen::SessionStats& stats, const std::string& where) {
    require(stats.consumed_tokens == 0 && stats.expert_hits == 0 && stats.expert_misses == 0 &&
            stats.expert_upload_bytes == 0 && stats.last_completed_ms == 0,
            where + ": consumed length/reset statistics are not zero");
}

void check_same_stats(const qwen::SessionStats& actual, const qwen::SessionStats& expected,
                      const std::string& where) {
    require(actual.consumed_tokens == expected.consumed_tokens && actual.expert_hits == expected.expert_hits &&
            actual.expert_misses == expected.expert_misses && actual.expert_upload_bytes == expected.expert_upload_bytes &&
            std::bit_cast<std::uint64_t>(actual.last_completed_ms) ==
                std::bit_cast<std::uint64_t>(expected.last_completed_ms),
            where + ": memory() mutated statistics");
}

void check_completed_stats(const qwen::SessionStats& stats, const qwen::SessionStats& previous,
                           std::size_t position, const std::string& where) {
    require(stats.consumed_tokens == position + 1, where + ": consumed length");
    require(std::isfinite(stats.last_completed_ms) && stats.last_completed_ms >= 0,
            where + ": invalid completed time");
    require(stats.expert_hits >= previous.expert_hits && stats.expert_misses >= previous.expert_misses &&
            stats.expert_upload_bytes >= previous.expert_upload_bytes, where + ": expert counters decreased");
    const auto hits = stats.expert_hits - previous.expert_hits;
    const auto misses = stats.expert_misses - previous.expert_misses;
    require(hits <= routes_per_step && misses <= routes_per_step && hits + misses == routes_per_step,
            where + ": incomplete expert route accounting");
    const auto uploaded = stats.expert_upload_bytes - previous.expert_upload_bytes;
    require(uploaded >= misses * q4_0_expert_bytes && uploaded <= misses * q4_1_expert_bytes,
            where + ": expert upload bytes inconsistent with miss payloads");
}

void check_finite_logits(std::span<const float> logits, const std::string& where) {
    require(logits.size() == vocabulary, where + ": logits size is not 248320");
    for (std::size_t i = 0; i < logits.size(); ++i)
        if (!std::isfinite(logits[i]))
            throw std::runtime_error(where + ": nonfinite logit at " + std::to_string(i));
}

void check_exact_logits(std::span<const float> actual, std::span<const float> expected,
                        const std::string& where) {
    require(actual.size() == vocabulary && expected.size() == vocabulary, where + ": logits size");
    for (std::size_t i = 0; i < vocabulary; ++i) {
        const auto a = std::bit_cast<std::uint32_t>(actual[i]);
        const auto b = std::bit_cast<std::uint32_t>(expected[i]);
        if (a != b)
            throw std::runtime_error(where + ": logit bits differ at " + std::to_string(i) +
                                     " (actual=" + std::to_string(a) + ", expected=" + std::to_string(b) + ")");
    }
}

std::span<float> reference_slice(std::vector<float>& reference, std::size_t position) {
    require(position < teacher_ids.size(), "reference position outside two-token fixture");
    const auto offset = position * vocabulary;
    require(offset <= reference.size() && vocabulary <= reference.size() - offset, "reference slice out of bounds");
    return std::span<float>(reference).subspan(offset, vocabulary);
}

void check_geometry(const qwen::SessionMemory& memory, const std::array<DeviceProbe, 2>& before,
                    const std::string& where) {
    require(memory.capacity == capacity && memory.expert_slots == expert_slots, where + ": capacity/slots");
    require(memory.ownership_verified, where + ": independent live-RAII ownership ledger not verified");
    equal_bytes(memory.ram_expert_payload, ram_expert_payload, where, "ram_expert_payload");
    require(memory.ram_expert_capacity >= memory.ram_expert_payload, where + ": RAM expert capacity below payload");
    require(memory.host_embedding_capacity > 0 && memory.host_logit_capacity >= logit_payload,
            where + ": missing embedding capacity or logit capacity below payload");
    equal_bytes(memory.pinned_handoff, pinned_handoff, where, "pinned_handoff");
    equal_bytes(memory.expert_payload_reads, payload_reads, where, "expert_payload_reads");
    equal_bytes(memory.expert_payload_bytes_read, ram_expert_payload, where, "expert_payload_bytes_read");
    for (std::size_t id = 0; id < memory.devices.size(); ++id) {
        const auto& d = memory.devices[id];
        const auto label = where + " GPU " + std::to_string(id);
        require(d.device == static_cast<int>(id) && d.first_layer == static_cast<int>(id) * 24 &&
                d.last_layer == static_cast<int>(id) * 24 + 23 && d.gdn_layers == 18 && d.qsa_layers == 6,
                label + ": owner range/layer counts");
        equal_bytes(d.expert_slots, slot_bytes[id], label, "expert_slots");
        equal_bytes(d.qsa_kv, qsa_kv_bytes, label, "qsa_kv");
        equal_bytes(d.qsa_index, qsa_pooled_bytes + qsa_tail_bytes, label, "qsa_index");
        equal_bytes(d.gdn_state, gdn_state_bytes, label, "gdn_state");
        equal_bytes(d.ple_state, ple_state_bytes[id], label, "ple_state");
        require(d.weights > 0 && d.workspace > 0 && d.owned_buffers > 0, label + ": empty allocation category/count");
        std::uint64_t sum = 0;
        for (const auto bytes : {d.weights, d.expert_slots, d.qsa_kv, d.qsa_index, d.gdn_state, d.ple_state, d.workspace}) {
            require(bytes <= std::numeric_limits<std::uint64_t>::max() - sum, label + ": category sum overflow");
            sum += bytes;
        }
        equal_bytes(d.owned_bytes, sum, label, "owned_bytes/category sum");
        require(d.owned_peak_bytes >= d.owned_bytes, label + ": peak below live owned bytes");
        equal_bytes(d.total_vram, before[id].total_vram, label, "total_vram");
        require(d.free_vram <= d.total_vram && d.owned_bytes <= d.total_vram - d.free_vram,
                label + ": owned bytes exceed observed used VRAM");
        require(d.free_vram >= min_free_headroom,
                label + ": headroom below frozen 1 GiB (free=" + std::to_string(d.free_vram) + ")");
    }
}

void check_steady_memory(const qwen::SessionMemory& memory, const qwen::SessionMemory& loaded,
                         const std::string& where) {
    for (std::size_t id = 0; id < memory.devices.size(); ++id) {
        const auto& a = memory.devices[id];
        const auto& b = loaded.devices[id];
        const auto label = where + " GPU " + std::to_string(id);
        equal_bytes(a.weights, b.weights, label, "steady weights");
        equal_bytes(a.expert_slots, b.expert_slots, label, "steady expert_slots");
        equal_bytes(a.qsa_kv, b.qsa_kv, label, "steady qsa_kv");
        equal_bytes(a.qsa_index, b.qsa_index, label, "steady qsa_index");
        equal_bytes(a.gdn_state, b.gdn_state, label, "steady gdn_state");
        equal_bytes(a.ple_state, b.ple_state, label, "steady ple_state");
        equal_bytes(a.workspace, b.workspace, label, "steady workspace");
        equal_bytes(a.owned_bytes, b.owned_bytes, label, "steady owned_bytes");
        equal_bytes(a.owned_peak_bytes, b.owned_peak_bytes, label, "steady owned_peak_bytes");
        equal_bytes(a.owned_buffers, b.owned_buffers, label, "steady owned_buffers");
    }
    equal_bytes(memory.ram_expert_capacity, loaded.ram_expert_capacity, where, "steady ram_expert_capacity");
    equal_bytes(memory.ram_expert_payload, loaded.ram_expert_payload, where, "steady ram_expert_payload");
    equal_bytes(memory.host_embedding_capacity, loaded.host_embedding_capacity, where, "steady host_embedding_capacity");
    equal_bytes(memory.host_logit_capacity, loaded.host_logit_capacity, where, "steady host_logit_capacity");
    equal_bytes(memory.pinned_handoff, loaded.pinned_handoff, where, "steady pinned_handoff");
    equal_bytes(memory.expert_payload_reads, loaded.expert_payload_reads, where, "steady expert_payload_reads");
    equal_bytes(memory.expert_payload_bytes_read, loaded.expert_payload_bytes_read, where, "steady expert_payload_bytes_read");
}

struct Snapshot {
    qwen::SessionMemory memory;
    qwen::SessionStats stats;
    std::uint64_t peak_rss = 0;
    int caller_device = 0;
};

Snapshot capture(qwen::Session& session, std::size_t index, std::uint64_t consumed,
                 const std::array<DeviceProbe, 2>& before, const qwen::SessionMemory* loaded,
                 std::span<const float> logits = {}, std::span<const float> saved = {}) {
    require(index < phases.size(), "snapshot phase out of bounds");
    const std::string where(phases[index]);
    require(logits.empty() == saved.empty(), where + ": incomplete snapshot logit reference");
    if (!logits.empty())
        require(logits.size() == vocabulary && saved.size() == vocabulary, where + ": snapshot logits size");
    RestoreDevice restore;
    Snapshot result;
    // Exercise preservation for callers on BOTH GPUs, including device 0 even
    // though memory() probes device 1 last. There is one API call per JSON row.
    result.caller_device = static_cast<int>(index % 2);
    hip_check(hipSetDevice(result.caller_device), "hipSetDevice snapshot caller");
    result.stats = session.stats();
    require(result.stats.consumed_tokens == consumed, where + ": unexpected consumed length");
    if (consumed == 0) check_zero_stats(result.stats, where);
    result.memory = session.memory();
    require(current_device() == result.caller_device, where + ": memory() changed caller device");
    check_same_stats(session.stats(), result.stats, where);
    if (!logits.empty()) check_exact_logits(logits, saved, where + " after memory()");
    check_geometry(result.memory, before, where);
    if (loaded) check_steady_memory(result.memory, *loaded, where);
    result.peak_rss = peak_rss_bytes();
    restore.restore_checked();
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

void json_stats(std::ostream& out, const qwen::SessionStats& stats) {
    out << "{\"consumed_tokens\":" << stats.consumed_tokens << ",\"expert_hits\":" << stats.expert_hits
        << ",\"expert_misses\":" << stats.expert_misses << ",\"expert_upload_bytes\":" << stats.expert_upload_bytes
        << ",\"last_completed_ms\":" << stats.last_completed_ms << '}';
}

void json_probe(std::ostream& out, const DeviceProbe& d) {
    out << "{\"device\":" << d.device << ",\"name\":"; json_string(out, d.name);
    out << ",\"arch\":"; json_string(out, d.arch);
    out << ",\"wave_size\":" << d.wave_size << ",\"compute_units\":" << d.compute_units
        << ",\"async_engines\":" << d.async_engines << ",\"properties_total_vram\":" << d.properties_total_vram
        << ",\"total_vram\":" << d.total_vram << ",\"free_vram\":" << d.free_vram << '}';
}

void emit(const std::ostringstream& record) {
    require(static_cast<bool>(record), "JSON record formatting failed");
    std::cout << record.str() << '\n';
    std::cout.flush();
    require(static_cast<bool>(std::cout), "stdout failed");
}

void source_json(std::string_view model, const std::array<DeviceProbe, 2>& before, int caller_device) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"kind\":\"memory_source\",\"protocol\":1,\"revision\":";
#ifdef CORE_REVISION
    json_string(out, CORE_REVISION);
#else
    out << "null";
#endif
    out << ",\"dirty\":";
#ifdef CORE_DIRTY
    out << (CORE_DIRTY ? "true" : "false");
#else
    out << "null";
#endif
    out << ",\"build_provenance_complete\":";
#if defined(CORE_REVISION) && defined(CORE_DIRTY)
    out << "true";
#else
    out << "false";
#endif
    out << ",\"model_path\":"; json_string(out, model);
    out << ",\"runtime\":\"own_48_layer_HIP\",\"scope\":\"r3_allocation_owner_headroom_reset_replay\""
        << ",\"capacity\":" << capacity << ",\"expert_slots\":" << expert_slots
        << ",\"session_instances\":1,\"trace\":false,\"sampling\":\"teacher_forced\",\"token_ids\":[248044,100]"
        << ",\"vocabulary\":" << vocabulary << ",\"min_free_headroom_bytes\":" << min_free_headroom
        << ",\"gates_frozen_before_execution\":true,\"occupied_128k_qualified\":false,\"performance_claim\":false"
        << ",\"limits\":{\"gpu_categories\":\"live_Session_Buffer_allocations_only\","
           "\"gpu_private_allocations\":\"HIP_context_rocBLAS_visible_only_in_total_free_VRAM\","
           "\"workspace\":\"includes_staged_recurrent_conv_PLE_state\","
           "\"host_capacities\":\"reported_payload_buffers_only_excludes_metadata_allocator_overhead\","
           "\"host_embedding_payload\":\"not_exposed_by_SessionMemory_capacity_checked_nonzero_and_steady\","
           "\"rss\":\"getrusage_RUSAGE_SELF_ru_maxrss_Linux_KiB_times_1024\","
           "\"context\":\"capacity_allocation_only_two_consumed_tokens_per_pass\",\"mtp\":false}"
        << ",\"expected_geometry\":{\"qsa_kv_bytes_per_gpu\":" << qsa_kv_bytes
        << ",\"qsa_pooled_index_bytes_per_gpu\":" << qsa_pooled_bytes
        << ",\"qsa_index_tail_bytes_per_gpu\":" << qsa_tail_bytes
        << ",\"qsa_index_bytes_per_gpu\":" << qsa_pooled_bytes + qsa_tail_bytes
        << ",\"gdn_state_bytes_per_gpu\":" << gdn_state_bytes
        << ",\"ple_state_bytes\":[" << ple_state_bytes[0] << ',' << ple_state_bytes[1]
        << "],\"q4_1_down_expert_bytes\":" << q4_1_expert_bytes
        << ",\"q4_0_down_expert_bytes\":" << q4_0_expert_bytes
        << ",\"expert_slots_bytes\":[" << slot_bytes[0] << ',' << slot_bytes[1]
        << "],\"ram_expert_payload_bytes\":" << ram_expert_payload
        << ",\"expert_payload_reads\":" << payload_reads << ",\"host_logit_payload_bytes\":" << logit_payload
        << ",\"pinned_handoff_bytes\":" << pinned_handoff << "},\"before_load\":[";
    json_probe(out, before[0]); out << ','; json_probe(out, before[1]);
    out << "],\"caller_device\":" << caller_device << ",\"peak_rss_bytes\":" << peak_rss_bytes() << '}';
    emit(out);
}

void snapshot_json(const Snapshot& snapshot, std::size_t index) {
    require(index < phases.size(), "JSON snapshot phase out of bounds");
    const auto& m = snapshot.memory;
    const bool has_logits = index == 1 || index == 2 || index == 4 || index == 5;
    const bool replay = index >= 4;
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\"kind\":\"memory_snapshot\",\"phase\":"; json_string(out, phases[index]);
    out << ",\"capacity\":" << m.capacity << ",\"expert_slots\":" << m.expert_slots
        << ",\"consumed_tokens\":" << snapshot.stats.consumed_tokens << ",\"token_id\":";
    if (has_logits) out << teacher_ids[index == 1 || index == 4 ? 0 : 1]; else out << "null";
    out << ",\"stats\":"; json_stats(out, snapshot.stats);
    out << ",\"ownership_verified\":" << (m.ownership_verified ? "true" : "false") << ",\"devices\":[";
    for (std::size_t id = 0; id < m.devices.size(); ++id) {
        if (id) out << ',';
        const auto& d = m.devices[id];
        out << "{\"device\":" << d.device << ",\"first_layer\":" << d.first_layer << ",\"last_layer\":" << d.last_layer
            << ",\"gdn_layers\":" << d.gdn_layers << ",\"qsa_layers\":" << d.qsa_layers
            << ",\"weights\":" << d.weights << ",\"expert_slots\":" << d.expert_slots << ",\"qsa_kv\":" << d.qsa_kv
            << ",\"qsa_index\":" << d.qsa_index << ",\"gdn_state\":" << d.gdn_state << ",\"ple_state\":" << d.ple_state
            << ",\"workspace\":" << d.workspace << ",\"owned_bytes\":" << d.owned_bytes
            << ",\"owned_peak_bytes\":" << d.owned_peak_bytes << ",\"owned_buffers\":" << d.owned_buffers
            << ",\"total_vram\":" << d.total_vram << ",\"free_vram\":" << d.free_vram << '}';
    }
    out << "],\"host\":{\"ram_expert_capacity\":" << m.ram_expert_capacity << ",\"ram_expert_payload\":" << m.ram_expert_payload
        << ",\"host_embedding_capacity\":" << m.host_embedding_capacity << ",\"host_logit_capacity\":" << m.host_logit_capacity
        << ",\"pinned_handoff\":" << m.pinned_handoff << ",\"expert_payload_reads\":" << m.expert_payload_reads
        << ",\"expert_payload_bytes_read\":" << m.expert_payload_bytes_read << "},\"peak_rss_bytes\":" << snapshot.peak_rss
        << ",\"checks\":{\"geometry\":true,\"headroom\":true,\"steady_allocations\":" << (index ? "true" : "null")
        << ",\"zero_stats\":" << (has_logits ? "null" : "true")
        << ",\"finite_logits\":" << (has_logits ? "true" : "null")
        << ",\"finite_logit_values\":" << (has_logits ? vocabulary : 0)
        << ",\"bitwise_replay\":" << (replay ? "true" : "null")
        << ",\"bitwise_replay_compared_values\":" << (replay ? vocabulary : 0)
        << ",\"snapshot_stats_preserved\":true,\"snapshot_logits_preserved\":" << (has_logits ? "true" : "null")
        << ",\"snapshot_logit_compared_values\":" << (has_logits ? vocabulary : 0)
        << ",\"snapshot_caller_device_preserved\":true,\"snapshot_caller_device\":" << snapshot.caller_device << "}}";
    emit(out);
}

void complete_json(const std::array<DeviceProbe, 2>& before, const std::array<DeviceProbe, 2>& after,
                   const std::array<Snapshot, 6>& snapshots, int caller_device) {
    const auto& last = snapshots.back();
    std::array<std::uint64_t, 2> recovered{}, minimum_free{};
    for (std::size_t id = 0; id < after.size(); ++id) {
        const auto& d = last.memory.devices[id];
        const auto label = "GPU " + std::to_string(id) + ": destruction recovery";
        equal_bytes(after[id].total_vram, before[id].total_vram, label, "total_vram");
        require(after[id].arch == before[id].arch && after[id].wave_size == before[id].wave_size,
                label + ": device properties changed");
        const auto detail = " before_load_free=" + std::to_string(before[id].free_vram) +
            " last_snapshot_free=" + std::to_string(d.free_vram) +
            " after_destroy_free=" + std::to_string(after[id].free_vram) +
            " required_owned_bytes=" + std::to_string(d.owned_bytes);
        require(after[id].free_vram >= d.free_vram, label + ": free bytes decreased;" + detail);
        recovered[id] = after[id].free_vram - d.free_vram;
        // No allocator tolerance: unexpectedly retained bytes FAIL this fixture.
        require(recovered[id] >= d.owned_bytes, label + ": free-byte recovery below owned bytes;" + detail);
        minimum_free[id] = d.free_vram;
        for (const auto& snapshot : snapshots)
            minimum_free[id] = std::min(minimum_free[id], snapshot.memory.devices[id].free_vram);
    }
    require(current_device() == caller_device, "final caller device differs from before load");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "{\"kind\":\"memory_complete\",\"protocol\":1,\"scope\":\"r3_allocation_owner_headroom_reset_replay\""
        << ",\"capacity\":" << capacity << ",\"expert_slots\":" << expert_slots
        << ",\"session_instances\":1,\"consumed_tokens\":" << last.stats.consumed_tokens
        << ",\"baseline_consumed_tokens\":" << snapshots[2].stats.consumed_tokens
        << ",\"replay_consumed_tokens\":" << last.stats.consumed_tokens
        << ",\"occupied_128k_qualified\":false,\"performance_claim\":false,\"after_destruction\":[";
    json_probe(out, after[0]); out << ','; json_probe(out, after[1]);
    out << "],\"cleanup\":{\"both_devices_synchronized\":true,\"device_reset_used\":false,\"devices\":[";
    for (std::size_t id = 0; id < after.size(); ++id) {
        if (id) out << ',';
        const auto& d = last.memory.devices[id];
        out << "{\"device\":" << id << ",\"before_load_free_vram\":" << before[id].free_vram
            << ",\"last_snapshot_free_vram\":" << d.free_vram << ",\"after_destroy_free_vram\":" << after[id].free_vram
            << ",\"required_owned_bytes\":" << d.owned_bytes << ",\"free_recovered_bytes\":" << recovered[id]
            << ",\"recovered_minus_owned_bytes\":" << recovered[id] - d.owned_bytes
            << ",\"min_snapshot_free_vram\":" << minimum_free[id] << ",\"owned_bytes_recovered\":true}";
    }
    out << "],\"caller_device_before\":" << caller_device << ",\"caller_device_after\":" << current_device()
        << "},\"peak_rss_bytes\":" << peak_rss_bytes()
        << ",\"checks\":{\"completed_steps\":4,\"snapshot_count\":6,\"steady_memory_comparisons\":5"
        << ",\"initial_stats_zero\":true,\"reset_consumed_and_stats_zero\":true,\"all_outputs_finite\":true"
        << ",\"finite_logit_values\":" << 4 * vocabulary << ",\"bitwise_replay\":true"
        << ",\"bitwise_replay_compared_values\":" << 2 * vocabulary
        << ",\"snapshot_stats_checks\":6,\"snapshot_device_checks\":6,\"snapshot_logit_compared_values\":" << 4 * vocabulary
        << ",\"ownership_verified\":true,\"steady_categories_counts_peak_host_and_payload_reads\":true"
        << ",\"min_free_headroom_bytes\":" << min_free_headroom
        << ",\"expert_payload_reads\":" << last.memory.expert_payload_reads
        << ",\"expert_payload_bytes_read\":" << last.memory.expert_payload_bytes_read
        << ",\"caller_device_restored\":true,\"owned_bytes_recovered_both_devices\":true},\"passed\":true}";
    emit(out);
}
}

int main(int argc, char** argv) {
    if (argc != 2 || argv[1][0] == '\0' || argv[1][0] == '-') {
        std::cerr << "usage: core-memory MODEL.gguf\n";
        return 2;
    }
    try {
        RestoreDevice restore;
        const int caller_device = restore.device();
        const auto before = probe_devices();
        source_json(argv[1], before, caller_device);
        qwen::SessionConfig config;
        config.capacity = capacity;
        config.expert_slots = expert_slots;
        // Empty trace_directory: all measurements here are correctness-only.
        std::vector<float> reference(teacher_ids.size() * vocabulary);
        std::array<Snapshot, 6> snapshots{};
        {
            // Exactly ONE Session. memory() independently cross-checks the
            // category traversal against its live Buffer RAII counters.
            qwen::Session session(argv[1], config);
            snapshots[0] = capture(session, 0, 0, before, nullptr);
            snapshot_json(snapshots[0], 0);
            auto previous = snapshots[0].stats;
            for (std::size_t position = 0; position < teacher_ids.size(); ++position) {
                const auto logits = session.step(teacher_ids[position]);
                const std::string where(phases[position + 1]);
                check_finite_logits(logits, where);
                check_completed_stats(session.stats(), previous, position, where);
                const auto saved = reference_slice(reference, position);
                std::copy(logits.begin(), logits.end(), saved.begin());
                snapshots[position + 1] = capture(session, position + 1, position + 1, before,
                    &snapshots[0].memory, logits, saved);
                previous = snapshots[position + 1].stats;
                snapshot_json(snapshots[position + 1], position + 1);
            }
            // Do not use a returned logit span across reset(): only the retained
            // full two-vector reference outlives subsequent Session calls.
            session.reset();
            snapshots[3] = capture(session, 3, 0, before, &snapshots[0].memory);
            snapshot_json(snapshots[3], 3);
            previous = snapshots[3].stats;
            for (std::size_t position = 0; position < teacher_ids.size(); ++position) {
                const auto logits = session.step(teacher_ids[position]);
                const std::string where(phases[position + 4]);
                check_finite_logits(logits, where);
                check_completed_stats(session.stats(), previous, position, where);
                const auto saved = reference_slice(reference, position);
                check_exact_logits(logits, saved, where + " replay");
                snapshots[position + 4] = capture(session, position + 4, position + 1, before,
                    &snapshots[0].memory, logits, saved);
                previous = snapshots[position + 4].stats;
                snapshot_json(snapshots[position + 4], position + 4);
            }
        } // Session's checked Buffer hipFree / handle / stream / pinned cleanup.
        restore.restore_checked();
        const auto after = probe_devices();
        // Only this footer contains passed:true, after strict recovery checks.
        complete_json(before, after, snapshots, caller_device);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "core-memory: " << error.what() << '\n';
        return 1;
    } catch (...) {
        std::cerr << "core-memory: non-standard exception\n";
        return 1;
    }
}
