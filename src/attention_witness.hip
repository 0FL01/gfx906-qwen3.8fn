// Atomic common-input attention diagnostic. No kernel/math/packing duplication.
// CLI: core-attention-witness VISIBLE QUERY.f32 K.q4_0 V.q4_0 CORRECTED.f32
// Link qwen-qsa-gpu (and its qwen-core dependency); C++/HIP20, gfx906,
// -ffp-contract=off, NO fast/finite-only math in this file or linked attention.
// Define CORE_REVISION (quoted string) and CORE_DIRTY (0/1) as for other witnesses.
// Normal stdout: source, device0, device1, footer JSONL, AFTER checked cleanup.
// Raw files are little-endian, immutable, exact length; no output file is opened.
// Reference provenance is supplied by the parent/original-oracle capture, not
// inferred from filenames. This gates only the captured attention operation.
#include "attention.hpp"
#include "hip/attention.cuh"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Attention witness requires finite checks and no fast-math"
#endif
#ifdef __clang__
#pragma clang fp contract(off)
#endif

namespace {
constexpr std::size_t width = qwen::attention_query_elements;
constexpr std::size_t blocks_per_position = qwen::attention_kv_heads * qwen::attention_q4_blocks_per_head;
constexpr std::size_t f32_bytes = width * sizeof(float), guard_bytes = 64;
constexpr unsigned char canary = 0xa7;
// FROZEN: every pair uses the second (reference) vector's magnitude.
constexpr double absolute_gate = 2.0e-4, relative_gate = 2.0e-4;
constexpr std::string_view scope = "same_captured_FLASH_query_and_canonical_selected_Q4_KV_attention_only";
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && sizeof(std::int32_t) == 4);
static_assert(std::numeric_limits<float>::is_iec559 && std::endian::native == std::endian::little);
static_assert(sizeof(qwen::Q4_0) == 18 && width == 6144 && blocks_per_position == 16);
static_assert(sizeof(float2) == 8 && guard_bytes % 16 == 0);

void require(bool value, std::string_view message) {
    if (!value) throw std::runtime_error(std::string(message));
}
void check(hipError_t status, std::string_view operation) {
    if (status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}
std::string json_string(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 32 || c >= 127) {
            result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15];
        } else result += static_cast<char>(c);
    }
    return result + '"';
}
void common_fields() {
    std::cout << ",\"protocol\":1,\"diagnostic_only\":true,\"common_input_scope\":" << json_string(scope)
              << ",\"revision\":";
#ifdef CORE_REVISION
    std::cout << json_string(CORE_REVISION);
#else
    std::cout << "null";
#endif
    std::cout << ",\"dirty\":";
#ifdef CORE_DIRTY
    static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
    std::cout << (CORE_DIRTY ? "true" : "false");
#else
    std::cout << "null";
#endif
    std::cout << ",\"absolute_gate\":" << absolute_gate << ",\"relative_gate\":" << relative_gate
              << ",\"bound\":\"abs(actual-reference)<=2e-4+2e-4*abs(reference)\"";
}

// Typed storage supplies real T lifetimes/alignment, including canonical Q4_0.
// Guards are whole elements: Q4 guards are 72 bytes, other guards 64 bytes.
template<class T> struct HostBuffer {
    static constexpr std::size_t guard_elements = (guard_bytes + sizeof(T) - 1) / sizeof(T);
    std::size_t count;
    std::vector<T> storage;
    std::vector<std::byte> original;
    explicit HostBuffer(std::size_t n) : count(n), storage(n + 2 * guard_elements) {
        std::memset(storage.data(), canary, storage.size() * sizeof(T));
        std::memset(data(), 0xff, count * sizeof(T));
    }
    T* data() { return storage.data() + guard_elements; }
    const T* data() const { return storage.data() + guard_elements; }
    std::span<T> values() { return {data(), count}; }
    std::span<const T> values() const { return {data(), count}; }
    std::span<std::byte> bytes() { return std::as_writable_bytes(values()); }
    std::span<const std::byte> bytes() const { return std::as_bytes(values()); }
    bool guards_intact() const {
        const auto all = std::as_bytes(std::span(storage));
        const auto test = [](std::span<const std::byte> bytes) {
            return std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte(canary); });
        };
        return test(all.first(guard_elements * sizeof(T))) && test(all.last(guard_elements * sizeof(T)));
    }
    void freeze() { const auto b = bytes(); original.assign(b.begin(), b.end()); }
    bool immutable() const {
        return original.size() == count * sizeof(T) &&
               std::memcmp(original.data(), data(), original.size()) == 0;
    }
};

struct File {
    int fd = -1;
    File() = default;
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { if (fd >= 0) (void)::close(fd); } // Successful path always checks close.
    void open(const char* path) {
        fd = ::open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), std::string("open ") + path);
    }
    struct stat status() const {
        struct stat s{};
        if (::fstat(fd, &s) != 0) throw std::system_error(errno, std::generic_category(), "fstat input");
        return s;
    }
    void close_checked() {
        const int descriptor = std::exchange(fd, -1);
        if (::close(descriptor) != 0) throw std::system_error(errno, std::generic_category(), "close input");
    }
};
bool aliases(const struct stat& a, const struct stat& b) { return a.st_dev == b.st_dev && a.st_ino == b.st_ino; }
bool unchanged_file(const struct stat& a, const struct stat& b) {
    return aliases(a, b) && a.st_size == b.st_size && a.st_mode == b.st_mode &&
           a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
           a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
void read_exact(File& file, std::span<std::byte> bytes, const struct stat& initial) {
    require(S_ISREG(initial.st_mode) && initial.st_size >= 0, "inputs must be regular files");
    require(initial.st_size == static_cast<off_t>(bytes.size()), "input file has incorrect exact byte length");
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto n = ::pread(file.fd, bytes.data() + offset, bytes.size() - offset, static_cast<off_t>(offset));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "pread input");
        }
        require(n > 0, "input file truncated during read");
        offset += static_cast<std::size_t>(n);
    }
    require(unchanged_file(initial, file.status()), "input file changed during read");
}
void finite_values(std::span<const float> values, std::string_view name) {
    for (std::size_t i = 0; i < values.size(); ++i)
        require(std::isfinite(values[i]), std::string(name) + ": nonfinite element " + std::to_string(i));
}
int parse_visible(const char* text) {
    const std::string_view s(text);
    require(!s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }),
            "VISIBLE must be a decimal integer in 1..2048");
    int value = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), value);
    require(r.ec == std::errc{} && r.ptr == s.data() + s.size() && value >= 1 && value <= 2048,
            "VISIBLE must be in 1..2048");
    return value;
}

struct Metrics {
    std::size_t count = 0, nonfinite_elements = 0, violating_elements = 0, bit_mismatches = 0;
    double max_abs = 0, rms = 0, max_bound_ratio = 0;
    bool passed() const { return count == width && nonfinite_elements == 0 && violating_elements == 0; }
};
Metrics compare(std::span<const float> actual, std::span<const float> reference) {
    require(actual.size() == width && reference.size() == width, "attention comparison requires all 6144 elements");
    Metrics m;
    m.count = width;
    double squared = 0;
    for (std::size_t i = 0; i < width; ++i) {
        m.bit_mismatches += std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(reference[i]);
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) {
            ++m.nonfinite_elements; ++m.violating_elements;
            continue;
        }
        const double error = std::abs(double(actual[i]) - double(reference[i]));
        const double bound = absolute_gate + relative_gate * std::abs(double(reference[i]));
        m.max_abs = std::max(m.max_abs, error);
        m.max_bound_ratio = std::max(m.max_bound_ratio, error / bound);
        m.violating_elements += error > bound;
        squared += error * error;
    }
    m.rms = std::sqrt(squared / double(width));
    return m;
}
void print_metrics(const Metrics& m) {
    std::cout << "{\"count\":" << m.count << ",\"all_finite\":" << (m.nonfinite_elements == 0)
              << ",\"nonfinite_elements\":" << m.nonfinite_elements << ",\"max_abs\":";
    if (m.nonfinite_elements) std::cout << "null"; else std::cout << m.max_abs;
    std::cout << ",\"rms\":";
    if (m.nonfinite_elements) std::cout << "null"; else std::cout << m.rms;
    std::cout << ",\"max_bound_ratio\":";
    if (m.nonfinite_elements) std::cout << "null"; else std::cout << m.max_bound_ratio;
    std::cout << ",\"violating_elements\":" << m.violating_elements << ",\"bit_mismatches\":" << m.bit_mismatches
              << ",\"byte_exact\":" << (m.bit_mismatches == 0) << ",\"within_frozen_gate\":" << m.passed() << '}';
}

struct Inputs {
    int visible;
    HostBuffer<float> query{width}, reference{width};
    HostBuffer<qwen::Q4_0> keys, values;
    HostBuffer<std::int32_t> ids{qwen::attention_max_selected};
    HostBuffer<int> counts{2};
    HostBuffer<float> cpu{width}, scores, staged{width};
    explicit Inputs(int n) : visible(n), keys(std::size_t(n) * blocks_per_position),
        values(std::size_t(n) * blocks_per_position), scores(std::size_t(n)) {
        std::fill(ids.values().begin(), ids.values().end(), -1);
        for (int i = 0; i < n; ++i) ids.data()[i] = i;
        counts.data()[0] = n; counts.data()[1] = n / 4;
        ids.freeze(); counts.freeze();
    }
    bool guards_intact() const {
        return query.guards_intact() && reference.guards_intact() && keys.guards_intact() && values.guards_intact() &&
               ids.guards_intact() && counts.guards_intact() && cpu.guards_intact() && scores.guards_intact() && staged.guards_intact();
    }
    bool immutable() const {
        return query.immutable() && reference.immutable() && keys.immutable() && values.immutable() && ids.immutable() && counts.immutable();
    }
};
struct SourceResult {
    int visible = 0;
    bool files_validated = false, cpu_completed = false, host_canaries = false, host_immutable = false;
    std::array<struct stat, 4> files{};
    std::array<std::string, 4> paths{};
    Metrics cpu_reference;
    std::string error;
    bool passed() const {
        return error.empty() && files_validated && cpu_completed && cpu_reference.passed() && host_canaries && host_immutable;
    }
};

// All asynchronous copy storage belongs to this owner. It drains BEFORE either
// device allocations or transfer storage can die, including exception unwinding.
enum Buffer : std::size_t { query, keys, values, ids, counts, output, gathered_keys,
    gathered_values, partial_output, partial_max_sum, staged_output, error_flag, buffer_count };
struct CleanupReport {
    bool attempted = false, complete = false, drained = false;
    std::size_t failure_count = 0;
    struct Failure { const char* operation = ""; hipError_t status = hipSuccess; };
    std::array<Failure, 20> failures{};
    void record(hipError_t status, const char* operation) noexcept {
        if (status == hipSuccess) return;
        if (failure_count < failures.size()) failures[failure_count] = {operation, status};
        ++failure_count;
    }
};
struct Allocation {
    void* raw = nullptr;
    std::size_t bytes = 0;
    std::unique_ptr<std::vector<std::byte>> upload, snapshot;
    void* data() const { return static_cast<std::byte*>(raw) + guard_bytes; }
};
struct Resources {
    int device;
    CleanupReport& report;
    bool closed = false;
    hipStream_t stream = nullptr;
    std::array<Allocation, buffer_count> buffers{};
    Resources(int d, CleanupReport& r) : device(d), report(r) {}
    Resources(const Resources&) = delete;
    Resources& operator=(const Resources&) = delete;
    ~Resources() { if (!closed) release(); }
    void initialize() {
        check(hipSetDevice(device), "hipSetDevice");
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "hipStreamCreateWithFlags");
    }
    void allocate(Buffer index, std::size_t bytes, std::size_t alignment) {
        auto& b = buffers[index];
        require(!b.raw && bytes > 0 && bytes <= std::numeric_limits<std::size_t>::max() - 2 * guard_bytes,
                "invalid guarded allocation size");
        b.bytes = bytes;
        b.snapshot = std::make_unique<std::vector<std::byte>>(bytes + 2 * guard_bytes);
        check(hipMalloc(&b.raw, bytes + 2 * guard_bytes), "hipMalloc guarded attention buffer");
        const auto begin = reinterpret_cast<std::uintptr_t>(b.raw);
        require(begin != 0 && reinterpret_cast<std::uintptr_t>(b.data()) % alignment == 0 &&
                bytes + 2 * guard_bytes <= std::numeric_limits<std::uintptr_t>::max() - begin,
                "invalid guarded device alignment/address span");
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            if (i == index || !buffers[i].raw) continue;
            const auto other = reinterpret_cast<std::uintptr_t>(buffers[i].raw);
            require(begin + bytes + 2 * guard_bytes <= other || other + buffers[i].bytes + 2 * guard_bytes <= begin,
                    "guarded device allocations overlap");
        }
        check(hipMemsetAsync(b.raw, canary, bytes + 2 * guard_bytes, stream), "initialize device canaries");
        check(hipMemsetAsync(b.data(), 0xff, bytes, stream), "poison provisional attention payload");
    }
    void put(Buffer index, std::span<const std::byte> bytes) {
        auto& b = buffers[index];
        require(bytes.size() == b.bytes && !b.upload, "attention upload shape/ownership mismatch");
        b.upload = std::make_unique<std::vector<std::byte>>(bytes.begin(), bytes.end());
        check(hipMemcpyAsync(b.data(), b.upload->data(), b.bytes, hipMemcpyHostToDevice, stream), "upload owned captured bytes");
    }
    void synchronize() {
        check(hipStreamSynchronize(stream), "hipStreamSynchronize attention completion");
        check(hipGetLastError(), "hipGetLastError attention completion");
    }
    void snapshot_all() {
        for (auto& b : buffers)
            check(hipMemcpyAsync(b.snapshot->data(), b.raw, b.snapshot->size(), hipMemcpyDeviceToHost, stream),
                  "read owned guarded attention buffer");
        synchronize();
    }
    std::span<const std::byte> payload(Buffer index) const {
        const auto& b = buffers[index];
        return std::span<const std::byte>(*b.snapshot).subspan(guard_bytes, b.bytes);
    }
    bool immutable(Buffer index) const {
        const auto& b = buffers[index];
        return b.upload && std::memcmp(payload(index).data(), b.upload->data(), b.bytes) == 0;
    }
    bool guards_intact() const {
        for (const auto& b : buffers) {
            for (std::size_t i = 0; i < guard_bytes; ++i)
                if ((*b.snapshot)[i] != std::byte(canary) || (*b.snapshot)[guard_bytes + b.bytes + i] != std::byte(canary))
                    return false;
        }
        return true;
    }
    void release() noexcept {
        report.attempted = true;
        const auto selected = hipSetDevice(device);
        report.record(selected, "hipSetDevice cleanup");
        bool drained = !stream;
        if (selected == hipSuccess && stream) {
            const auto synced = hipStreamSynchronize(stream);
            report.record(synced, "hipStreamSynchronize cleanup");
            drained = synced == hipSuccess;
            if (!drained) {
                const auto fallback = hipDeviceSynchronize();
                report.record(fallback, "hipDeviceSynchronize cleanup fallback");
                drained = fallback == hipSuccess;
            }
        }
        report.drained = drained;
        if (selected == hipSuccess && drained) {
            for (auto& b : buffers) if (b.raw) {
                report.record(hipFree(b.raw), "hipFree attention buffer");
                b.raw = nullptr;
            }
            if (stream) report.record(hipStreamDestroy(stream), "hipStreamDestroy attention stream");
            stream = nullptr;
        } else {
            // No confirmed drain: retain every borrowed device/host copy address
            // until process exit. Never free a possibly in-flight copy operand.
            for (auto& b : buffers) { (void)b.upload.release(); (void)b.snapshot.release(); }
        }
        closed = true;
        report.complete = selected == hipSuccess && drained && report.failure_count == 0;
    }
    void close_checked() {
        release();
        require(report.complete, "checked HIP cleanup failed (see cleanup_failures)");
    }
};

struct DeviceResult {
    bool attempted = false, completed = false, comparisons_complete = false, snapshots_complete = false;
    bool canaries = false, query_immutable = false, cache_immutable = false, ids_immutable = false, counts_immutable = false;
    int error_flag = -1;
    Metrics gpu_cpu, gpu_reference;
    CleanupReport cleanup;
    std::string error;
    bool passed() const {
        return error.empty() && completed && comparisons_complete && gpu_cpu.passed() && gpu_reference.passed() &&
               error_flag == 0 && snapshots_complete && canaries && query_immutable && cache_immutable &&
               ids_immutable && counts_immutable && cleanup.complete && cleanup.drained;
    }
};
void run_device(int device, const Inputs& in, DeviceResult& result) {
    result.attempted = true;
    Resources r(device, result.cleanup);
    r.initialize();
    r.allocate(query, f32_bytes, 16);
    r.allocate(keys, in.keys.bytes().size(), 2);
    r.allocate(values, in.values.bytes().size(), 2);
    r.allocate(ids, qwen::attention_max_selected * sizeof(std::int32_t), 4);
    r.allocate(counts, 2 * sizeof(int), 4);
    r.allocate(output, f32_bytes, 4);
    r.allocate(gathered_keys, qwen::attention_gather_elements * sizeof(std::uint16_t), 2);
    r.allocate(gathered_values, qwen::attention_gather_elements * sizeof(std::uint16_t), 2);
    r.allocate(partial_output, qwen::attention_partial_elements * sizeof(float), 4);
    r.allocate(partial_max_sum, qwen::attention_partial_ml_elements * sizeof(float2), 8);
    r.allocate(staged_output, f32_bytes, 4);
    r.allocate(error_flag, sizeof(int), 4);
    r.put(query, in.query.bytes()); r.put(keys, in.keys.bytes()); r.put(values, in.values.bytes());
    r.put(ids, in.ids.bytes()); r.put(counts, in.counts.bytes());
    check(hipMemsetAsync(r.buffers[error_flag].data(), 0, sizeof(int), r.stream), "reset attention sticky error");
    const qwen::AttentionQ4DeviceWorkspace workspace{
        static_cast<std::uint16_t*>(r.buffers[gathered_keys].data()),
        static_cast<std::uint16_t*>(r.buffers[gathered_values].data()),
        static_cast<float*>(r.buffers[partial_output].data()),
        static_cast<float2*>(r.buffers[partial_max_sum].data()),
        static_cast<float*>(r.buffers[staged_output].data()),
        static_cast<int*>(r.buffers[error_flag].data())};
    check(qwen::launch_attention_q4(static_cast<const float*>(r.buffers[query].data()),
        static_cast<const qwen::Q4_0*>(r.buffers[keys].data()), static_cast<const qwen::Q4_0*>(r.buffers[values].data()),
        in.visible, static_cast<const std::int32_t*>(r.buffers[ids].data()),
        static_cast<const int*>(r.buffers[counts].data()), in.visible,
        static_cast<float*>(r.buffers[output].data()), workspace, r.stream), "launch_attention_q4");
    r.synchronize();
    result.completed = true;
    r.snapshot_all();
    result.snapshots_complete = true;
    result.canaries = r.guards_intact();
    result.query_immutable = r.immutable(query);
    result.cache_immutable = r.immutable(keys) && r.immutable(values);
    result.ids_immutable = r.immutable(ids);
    result.counts_immutable = r.immutable(counts);
    std::memcpy(&result.error_flag, r.payload(error_flag).data(), sizeof(int));
    if (result.error_flag == 0) {
        std::vector<float> actual(width);
        std::memcpy(actual.data(), r.payload(output).data(), f32_bytes);
        result.gpu_cpu = compare(actual, in.cpu.values());
        result.gpu_reference = compare(actual, in.reference.values());
        result.comparisons_complete = true;
    }
    // Integrity and numerical gate failures still finish every check/cleanup.
    r.close_checked();
    require(result.error_flag == 0, "attention sticky error is nonzero; provisional output not consumed");
}

void print_source(const SourceResult& s) {
    std::cout << "{\"kind\":\"attention_witness_source\"";
    common_fields();
    std::cout << ",\"visible\":" << s.visible << ",\"capacity\":" << s.visible
              << ",\"query_elements\":6144,\"output_elements\":6144,\"query_heads\":24,\"kv_heads\":2,\"head_dim\":256"
              << ",\"q4_block_bytes\":18,\"cache_layout\":\"[visible_position][2][8_Q4_0_blocks]\""
              << ",\"file_endianness\":\"little\",\"selected_ids\":\"exactly_0_through_visible_minus_1_in_order\""
              << ",\"ids_buffer_capacity\":" << qwen::attention_max_selected << ",\"counts\":[" << s.visible << ',' << s.visible / 4 << ']'
              << ",\"cpu_api\":\"qwen::attention_q4\",\"cpu_mode\":\"gathered_fp16\""
              << ",\"gpu_api\":\"qwen::launch_attention_q4\",\"reference\":\"caller_supplied_independent_corrected_FP32_QSA\""
              << ",\"reference_lineage_verified_by_binary\":false,\"files\":[";
    constexpr std::array<const char*, 4> roles{"FLASH_query", "canonical_selected_K", "canonical_selected_V", "corrected_attention_output"};
    for (std::size_t i = 0; i < s.files.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"role\":" << json_string(roles[i]) << ",\"path\":" << json_string(s.paths[i])
                  << ",\"expected_bytes\":" << ((i == 1 || i == 2) ? std::size_t(s.visible) * blocks_per_position * sizeof(qwen::Q4_0) : f32_bytes)
                  << ",\"file_device\":" << static_cast<std::uint64_t>(s.files[i].st_dev)
                  << ",\"file_inode\":" << static_cast<std::uint64_t>(s.files[i].st_ino) << ",\"file_bytes\":" << s.files[i].st_size << '}';
    }
    std::cout << "],\"files_validated_and_closed\":" << s.files_validated
              << ",\"cpu_completed\":" << s.cpu_completed << ",\"cpu_vs_reference\":";
    if (s.cpu_completed) print_metrics(s.cpu_reference); else std::cout << "null";
    std::cout << ",\"host_inputs_immutable\":" << s.host_immutable << ",\"host_canaries_intact\":" << s.host_canaries
              << ",\"host_canary_checks\":" << (s.host_canaries ? 18 : 0)
              << ",\"error\":" << (s.error.empty() ? "null" : json_string(s.error)) << ",\"passed\":" << s.passed() << "}\n";
}
void print_device(int device, const hipDeviceProp_t& p, const DeviceResult& d, bool source_passed) {
    std::cout << "{\"kind\":\"attention_witness_device\",\"device\":" << device;
    common_fields();
    std::cout << ",\"device_name\":" << json_string(p.name) << ",\"gcn_arch\":" << json_string(p.gcnArchName)
              << ",\"wave_size\":" << p.warpSize << ",\"vram_bytes\":" << p.totalGlobalMem
              << ",\"attempted\":" << d.attempted << ",\"completed\":" << d.completed << ",\"gpu_vs_cpu\":";
    if (d.comparisons_complete) print_metrics(d.gpu_cpu); else std::cout << "null";
    std::cout << ",\"gpu_vs_reference\":";
    if (d.comparisons_complete) print_metrics(d.gpu_reference); else std::cout << "null";
    std::cout << ",\"sticky_error\":" << d.error_flag << ",\"query_immutable\":" << d.query_immutable
              << ",\"cache_immutable\":" << d.cache_immutable << ",\"ids_immutable\":" << d.ids_immutable
              << ",\"counts_immutable\":" << d.counts_immutable << ",\"canaries_intact\":" << d.canaries
              << ",\"canary_checks\":" << (d.canaries ? 2 * buffer_count : 0)
              << ",\"allocation_aliases_and_capacities_checked\":" << d.snapshots_complete
              << ",\"cleanup_attempted\":" << d.cleanup.attempted << ",\"owner_drained\":" << d.cleanup.drained
              << ",\"cleanup_complete\":" << d.cleanup.complete << ",\"cleanup_failures\":[";
    for (std::size_t i = 0; i < std::min(d.cleanup.failure_count, d.cleanup.failures.size()); ++i) {
        if (i) std::cout << ',';
        const auto& f = d.cleanup.failures[i];
        std::cout << "{\"operation\":" << json_string(f.operation) << ",\"status\":" << static_cast<int>(f.status)
                  << ",\"message\":" << json_string(hipGetErrorString(f.status)) << '}';
    }
    std::cout << "],\"error\":" << (d.error.empty() ? "null" : json_string(d.error))
              << ",\"device_gates_passed\":" << d.passed() << ",\"passed\":" << (source_passed && d.passed()) << "}\n";
}

struct Results {
    SourceResult source;
    std::array<DeviceResult, 2> devices{};
    std::array<hipDeviceProp_t, 2> properties{};
    int discovered_devices = -1;
    bool topology_validated = false, stdout_safe = true, stderr_safe = true;
    std::string error;
};
void execute(int argc, char** argv, Results& results) {
    std::unique_ptr<Inputs> in;
    auto& s = results.source;
    try {
        require(argc == 6, "usage: core-attention-witness VISIBLE QUERY.f32 K.q4_0 V.q4_0 CORRECTED.f32");
        std::array<File, 4> files;
        struct stat out{}, err{};
        results.stdout_safe = ::fstat(STDOUT_FILENO, &out) == 0;
        require(results.stdout_safe, "cannot validate stdout identity");
        results.stderr_safe = ::fstat(STDERR_FILENO, &err) == 0;
        for (std::size_t i = 0; i < files.size(); ++i) {
            s.paths[i] = argv[i + 2];
            files[i].open(argv[i + 2]);
            s.files[i] = files[i].status();
            if (aliases(s.files[i], out)) results.stdout_safe = false;
            if (aliases(s.files[i], err)) results.stderr_safe = false;
        }
        require(results.stdout_safe && results.stderr_safe, "input aliases stdout/stderr; refusing output to an input file");
        for (std::size_t i = 0; i < files.size(); ++i)
            for (std::size_t j = 0; j < i; ++j)
                require(!aliases(s.files[i], s.files[j]), "input files alias, including hard links");
        s.visible = parse_visible(argv[1]);
        in = std::make_unique<Inputs>(s.visible);
        read_exact(files[0], in->query.bytes(), s.files[0]);
        read_exact(files[1], in->keys.bytes(), s.files[1]);
        read_exact(files[2], in->values.bytes(), s.files[2]);
        read_exact(files[3], in->reference.bytes(), s.files[3]);
        for (std::size_t i = 0; i < files.size(); ++i)
            require(unchanged_file(s.files[i], files[i].status()), "input changed while reading the common-input set");
        for (auto& file : files) file.close_checked();
        finite_values(in->query.values(), "captured FLASH query");
        finite_values(in->reference.values(), "independent corrected output");
        in->query.freeze(); in->keys.freeze(); in->values.freeze(); in->reference.freeze();
        s.files_validated = true;
        qwen::attention_q4({in->keys.values(), in->values.values(), std::size_t(s.visible)}, in->query.values(),
            in->ids.values().first(std::size_t(s.visible)), std::size_t(s.visible), qwen::AttentionQ4Mode::gathered_fp16,
            {in->scores.values(), in->staged.values()}, in->cpu.values());
        s.cpu_completed = true;
        s.cpu_reference = compare(in->cpu.values(), in->reference.values());
        s.host_canaries = in->guards_intact();
        s.host_immutable = in->immutable();
        require(s.host_canaries && s.host_immutable, "CPU common-input integrity/canary failure");
    } catch (const std::exception& e) {
        s.error = e.what();
        if (in) { s.host_canaries = in->guards_intact(); s.host_immutable = in->immutable(); }
    }
    // A bounded CPU/reference mismatch is diagnostic evidence, not a reason to
    // omit either GPU comparison. Rejected/failed CPU inputs are never launched.
    if (s.error.empty() && s.cpu_completed) {
        try {
            check(hipGetDeviceCount(&results.discovered_devices), "hipGetDeviceCount");
            require(results.discovered_devices == 2, "attention witness requires exactly two gfx906 wave64 devices");
            for (int d = 0; d < 2; ++d) {
                check(hipGetDeviceProperties(&results.properties[d], d), "hipGetDeviceProperties");
                const auto& p = results.properties[d];
                const std::string_view arch(p.gcnArchName);
                require((arch == "gfx906" || arch.starts_with("gfx906:")) && p.warpSize == 64 && p.maxThreadsPerBlock >= 256,
                        "expected exact gfx906, wave64, and support for the qualified 256-thread attention topology");
            }
            results.topology_validated = true;
        } catch (const std::exception& e) { results.error = e.what(); }
    }
    for (int d = 0; d < 2; ++d) {
        auto& r = results.devices[d];
        if (!results.topology_validated) {
            r.error = s.error.empty() ? results.error : "source rejected: " + s.error;
            if (r.error.empty()) r.error = "source CPU computation incomplete";
            continue;
        }
        try { run_device(d, *in, r); }
        catch (const std::exception& e) { r.error = e.what(); }
        s.host_canaries = in->guards_intact(); s.host_immutable = in->immutable();
        if (!s.host_canaries || !s.host_immutable) {
            s.error = "host common-input integrity/canary failure after GPU diagnostic";
            results.topology_validated = false; // Do not launch with corrupted shared inputs.
        }
        if (r.cleanup.attempted && !r.cleanup.drained) {
            results.error = "HIP owner drain unconfirmed; further device work skipped";
            results.topology_validated = false;
        }
    }
} // File and host owners die here; every attempted HIP owner has already closed.
} // namespace

int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic());
    std::cout << std::boolalpha << std::setprecision(17);
    Results results;
    try { execute(argc, argv, results); }
    catch (const std::exception& e) { results.error = e.what(); }
    if (!results.stdout_safe) {
        return 1; // Emitting JSON to an input would itself violate immutability.
    }
    print_source(results.source);
    for (int d = 0; d < 2; ++d) print_device(d, results.properties[d], results.devices[d], results.source.passed());
    const bool passed = results.error.empty() && results.topology_validated && results.source.passed() &&
                        results.devices[0].passed() && results.devices[1].passed();
    const bool cleanup_complete = results.devices[0].cleanup.complete && results.devices[1].cleanup.complete;
    std::cout << "{\"kind\":\"attention_witness_complete\"";
    common_fields();
    std::cout << ",\"source_records\":1,\"device_records\":2,\"discovered_devices\":" << results.discovered_devices
              << ",\"topology_validated\":" << results.topology_validated
              << ",\"primary_gate\":\"full6144_CPU_reference_and_both_GPU_CPU_reference_plus_finite_integrity_error0_cleanup\""
              << ",\"all_required_comparisons\":5,\"source_passed\":" << results.source.passed()
              << ",\"cleanup_complete\":" << cleanup_complete
              << ",\"error\":" << (results.error.empty() ? "null" : json_string(results.error))
              << ",\"passed\":" << passed << "}\n";
    std::cout.flush();
    return passed && std::cout ? 0 : 1;
}
