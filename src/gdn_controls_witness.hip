// SAME-input correctness-only diagnostic; no runtime backend, CPU math oracle or speed claim.
// Link qwen-gdn-gpu/PUBLIC qwen-core; HIP/C++20 gfx906, -ffp-contract=off, no fast-math.
// Define CORE_REVISION/CORE_DIRTY. stdout: source/device0/device1/footer AFTER cleanup.
// Controls are helper-recomputed, NOT captured recurrence registers. CLI usage below.
#include "dense.hpp"
#include "model.hpp"
#include "hip/gdn.cuh"
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
#include <vector>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "GDN controls witness requires finite checks and no fast-math"
#endif
#if !defined(CORE_REVISION) || !defined(CORE_DIRTY)
#error "Define compiled CORE_REVISION string and CORE_DIRTY 0/1"
#endif
#ifdef __clang__
#pragma clang fp contract(off)
#endif
namespace {
constexpr std::size_t heads = 48, bytes = 192, guard = 64;
constexpr double absolute_gate = 2e-4, relative_gate = 2e-4;
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::endian::native == std::endian::little && qwen::gdn_device_value_heads == heads);
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);
void require(bool ok, std::string_view text) { if (!ok) throw std::runtime_error(std::string(text)); }
void check(hipError_t s, const char* op) { if (s != hipSuccess) throw std::runtime_error(std::string(op) + ": " + hipGetErrorString(s)); }
std::string quote(std::string_view s) {
    std::string r = "\""; constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { r += '\\'; r += char(c); }
        else if (c < 32 || c >= 127) { r += "\\u00"; r += hex[c >> 4]; r += hex[c & 15]; } else r += char(c);
    } return r + '"'; }
void value(float x) { if (std::isfinite(x)) std::cout << double(x); else std::cout << "null"; }
struct Host {
    std::array<float, heads + 2 * guard / 4> storage{};
    std::array<std::byte, sizeof(storage)> frozen{};
    Host() { std::memset(storage.data(), 0xa7, sizeof(storage)); }
    std::span<float> data() { return std::span(storage).subspan(guard / 4, heads); }
    void freeze() {
        const auto all = std::as_bytes(std::span(storage));
        for (std::size_t i = 0; i < guard; ++i) require(all[i] == std::byte{0xa7} && all[sizeof(storage) - guard + i] == std::byte{0xa7}, "host canary changed during source read");
        std::memcpy(frozen.data(), storage.data(), sizeof(storage));
    }
    bool immutable() const { return std::memcmp(frozen.data(), storage.data(), sizeof(storage)) == 0; }
};
struct Files {
    std::array<int, 5> fd{-1,-1,-1,-1,-1}; int& failures;
    explicit Files(int& f) : failures(f) {}
    ~Files() { for (int d : fd) if (d >= 0 && ::close(d) != 0) ++failures; } };
struct stat status(int fd) { struct stat s{}; if (::fstat(fd, &s)) throw std::system_error(errno, std::generic_category(), "fstat"); return s; }
bool same(const struct stat& a, const struct stat& b) {
    return a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size && a.st_mode == b.st_mode &&
        a.st_mtim.tv_sec == b.st_mtim.tv_sec && a.st_mtim.tv_nsec == b.st_mtim.tv_nsec &&
        a.st_ctim.tv_sec == b.st_ctim.tv_sec && a.st_ctim.tv_nsec == b.st_ctim.tv_nsec;
}
void read_f32(int fd, Host& h) {
    const auto s = status(fd); require(S_ISREG(s.st_mode) && s.st_size == off_t(bytes), "expected regular exact 192-byte little-endian F32 file");
    std::array<std::byte, bytes> raw{}; std::size_t offset = 0;
    while (offset < bytes) {
        const auto n = ::pread(fd, raw.data() + offset, bytes - offset, off_t(offset));
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) throw std::system_error(errno, std::generic_category(), "pread F32");
        require(n > 0, "truncated F32 file"); offset += std::size_t(n);
    }
    require(same(s, status(fd)), "F32 file changed during read");
    for (std::size_t i = 0; i < heads; ++i) { h.data()[i] = qwen::tensor_element(qwen::TensorType::F32, raw, i); require(std::isfinite(h.data()[i]), "nonfinite F32 input/reference"); } }
void parameter(const qwen::Model& m, const qwen::TensorView& t, Host& h, std::uint64_t& read_bytes) {
    require(t.type == qwen::TensorType::F32 && t.rank == 1 && t.dimensions[0] == heads && t.elements == heads &&
        t.byte_size == bytes && t.strides[0] == 4, "controls weight must be contiguous rank1 F32[48]");
    require(t.relative_offset % m.alignment() == 0 && t.file_offset >= m.data_offset() &&
        t.file_offset - m.data_offset() == t.relative_offset && t.file_offset <= m.file_size() && bytes <= m.file_size() - t.file_offset, "invalid controls tensor range");
    std::array<std::byte, bytes> raw{}; m.read_tensor(t.name, raw); read_bytes += bytes;
    for (std::size_t i = 0; i < heads; ++i) { h.data()[i] = qwen::tensor_element(t.type, raw, i); require(std::isfinite(h.data()[i]), "nonfinite controls weight"); } }
struct Metrics { int nonfinite = 0, violations = 0, bit_mismatches = 0; double max_abs = 0, rms = 0, max_bound_ratio = 0; };
Metrics compare(const float* a, std::span<const float> ref) { Metrics m; double squared = 0;
    for (std::size_t i = 0; i < heads; ++i) {
        m.bit_mismatches += std::bit_cast<std::uint32_t>(a[i]) != std::bit_cast<std::uint32_t>(ref[i]);
        if (!std::isfinite(a[i]) || !std::isfinite(ref[i])) { ++m.nonfinite; ++m.violations; continue; }
        const double e = std::abs(double(a[i]) - double(ref[i])), b = absolute_gate + relative_gate * std::abs(double(ref[i]));
        m.max_abs = std::max(m.max_abs, e); m.max_bound_ratio = std::max(m.max_bound_ratio, e / b); squared += e * e; m.violations += e > b;
    } m.rms = std::sqrt(squared / heads); return m;
}
void metrics(const Metrics& m) {
    std::cout << "{\"count\":48,\"nonfinite\":" << m.nonfinite << ",\"violations\":" << m.violations << ",\"bit_mismatches\":" << m.bit_mismatches << ",\"byte_exact\":" << (m.bit_mismatches == 0);
    std::cout << ",\"max_abs\":"; if (m.nonfinite) std::cout << "null"; else std::cout << m.max_abs;
    std::cout << ",\"rms\":"; if (m.nonfinite) std::cout << "null"; else std::cout << m.rms;
    std::cout << ",\"max_bound_ratio\":"; if (m.nonfinite) std::cout << "null"; else std::cout << m.max_bound_ratio;
    std::cout << ",\"within_frozen_gate\":" << (m.violations == 0) << '}'; }
struct Device {
    bool attempted = false, completed = false, guards = false, immutable = false, cleanup_attempted = false, drained = false, cleaned = false;
    int flag = -1, guard_checks = 0, immutable_checks = 0, cleanup_failures = 0, decay_nonfinite = 0, allocations = 0;
    std::size_t allocated_bytes = 0;
    std::array<float, 3 * heads> output{}; std::array<Metrics, 2> comparisons{}; std::string error;
    bool passed() const { return error.empty() && completed && guards && immutable && flag == 0 && cleaned && drained &&
        comparisons[0].violations == 0 && comparisons[1].violations == 0 && decay_nonfinite == 0; }
};
struct Buffer {
    void* raw = nullptr; std::size_t size = 0; std::unique_ptr<std::vector<std::byte>> upload, snapshot;
    void* data() const { return static_cast<std::byte*>(raw) + guard; }
};
struct Owner {
    int device; Device& result; hipStream_t stream = nullptr; bool closed = false; std::array<Buffer, 6> buffers{};
    Owner(int d, Device& r) : device(d), result(r) {}
    Owner(const Owner&) = delete; Owner& operator=(const Owner&) = delete; ~Owner() { if (!closed) release(); }
    void release() noexcept {
        result.cleanup_attempted = true;
        auto record = [&](hipError_t s, const char* op) { if (s != hipSuccess) { ++result.cleanup_failures; std::cerr << op << ": " << hipGetErrorString(s) << '\n'; } };
        const auto selected = hipSetDevice(device); record(selected, "cleanup select owner"); result.drained = !stream;
        if (selected == hipSuccess && stream) {
            const auto s = hipStreamSynchronize(stream); record(s, "cleanup drain"); result.drained = s == hipSuccess;
            if (!result.drained) { const auto f = hipDeviceSynchronize(); record(f, "cleanup fallback drain"); result.drained = f == hipSuccess; }
        }
        if (selected == hipSuccess && result.drained) {
            for (auto& b : buffers) if (b.raw) { record(hipFree(b.raw), "cleanup free"); b.raw = nullptr; }
            if (stream) { record(hipStreamDestroy(stream), "cleanup stream destroy"); stream = nullptr; }
        } else { // Unconfirmed drain: retain every asynchronous operand until process exit.
            for (auto& b : buffers) { (void)b.upload.release(); (void)b.snapshot.release(); }
        }
        closed = true; result.cleaned = selected == hipSuccess && result.drained && result.cleanup_failures == 0;
    }
    void allocate(std::size_t i, std::span<const std::byte> source, std::size_t n) {
        auto& b = buffers[i]; b.size = n; b.upload = std::make_unique<std::vector<std::byte>>(n + 2 * guard, std::byte{0xa7});
        b.snapshot = std::make_unique<std::vector<std::byte>>(n + 2 * guard);
        if (!source.empty()) { require(source.size() == n, "upload capacity mismatch"); std::memcpy(b.upload->data() + guard, source.data(), n); }
        else std::memset(b.upload->data() + guard, i == 5 ? 0 : 0xff, n);
        check(hipMalloc(&b.raw, n + 2 * guard), "hipMalloc"); ++result.allocations; result.allocated_bytes += n + 2 * guard;
        const auto start = reinterpret_cast<std::uintptr_t>(b.raw);
        require(start && start % 4 == 0 && n + 2 * guard <= std::numeric_limits<std::uintptr_t>::max() - start, "bad allocation alignment/range");
        for (std::size_t j = 0; j < i; ++j) { const auto other = reinterpret_cast<std::uintptr_t>(buffers[j].raw);
            require(start + n + 2 * guard <= other || other + buffers[j].size + 2 * guard <= start, "device allocations overlap"); }
        check(hipMemcpyAsync(b.raw, b.upload->data(), n + 2 * guard, hipMemcpyHostToDevice, stream), "upload guarded owner storage");
    }
};
void run(int d, std::array<Host, 6>& host, Device& out) {
    out.attempted = true; Owner r(d, out); check(hipSetDevice(d), "select owner"); int current = -1;
    check(hipGetDevice(&current), "query owner"); require(current == d, "wrong HIP owner");
    check(hipStreamCreateWithFlags(&r.stream, hipStreamNonBlocking), "nonblocking stream");
    for (std::size_t i = 0; i < 4; ++i) r.allocate(i, std::as_bytes(host[i].data()), bytes);
    r.allocate(4, {}, 3 * bytes); r.allocate(5, {}, sizeof(int));
    qwen::GdnDeviceParameters p{}; p.dt_bias = static_cast<float*>(r.buffers[0].data()); p.ssm_a = static_cast<float*>(r.buffers[1].data());
    qwen::GdnDeviceInput in{}; in.alpha = static_cast<float*>(r.buffers[2].data()); in.beta = static_cast<float*>(r.buffers[3].data());
    qwen::launch_gdn_control_witness(p, in, static_cast<float*>(r.buffers[4].data()), static_cast<int*>(r.buffers[5].data()), r.stream);
    for (auto& b : r.buffers) check(hipMemcpyAsync(b.snapshot->data(), b.raw, b.size + 2 * guard, hipMemcpyDeviceToHost, r.stream), "snapshot guarded owner storage");
    check(hipStreamSynchronize(r.stream), "complete controls and snapshots"); check(hipGetLastError(), "completed launch status");
    out.guards = true; out.immutable = true;
    for (std::size_t i = 0; i < r.buffers.size(); ++i) { const auto& b = r.buffers[i];
        for (std::size_t side : {std::size_t(0), guard + b.size}) { ++out.guard_checks;
            out.guards &= std::all_of(b.snapshot->begin() + side, b.snapshot->begin() + side + guard, [](std::byte x) { return x == std::byte{0xa7}; }); }
        if (i < 4) { ++out.immutable_checks; out.immutable &= std::memcmp(b.snapshot->data() + guard, b.upload->data() + guard, b.size) == 0; }
    }
    std::memcpy(&out.flag, r.buffers[5].snapshot->data() + guard, sizeof(int));
    std::memcpy(out.output.data(), r.buffers[4].snapshot->data() + guard, sizeof(out.output));
    out.comparisons = {compare(out.output.data(), host[4].data()), compare(out.output.data() + heads, host[5].data())};
    for (std::size_t i = 2 * heads; i < 3 * heads; ++i) out.decay_nonfinite += !std::isfinite(out.output[i]);
    out.completed = true; r.release(); }
struct Results {
    std::array<Host, 6> host{}; std::array<Device, 2> devices{}; std::array<hipDeviceProp_t, 2> properties{};
    std::array<std::string, 5> paths{}; std::array<qwen::TensorView, 2> weights{};
    int layer = -1, count = -1, file_cleanup_failures = 0; std::uint64_t model_bytes = 0, payload_bytes_read = 0;
    bool source_valid = false, topology = false, host_immutable = false, stdout_safe = true, stderr_safe = true; std::string error; };
void execute(int argc, char** argv, Results& r) {
    require(argc == 7, "usage: core-gdn-controls-witness MODEL LAYER ALPHA.f32 BETA.f32 ORIGINAL_LOG_DECAY.f32 ORIGINAL_BETA.f32");
    r.paths = {argv[1], argv[3], argv[4], argv[5], argv[6]}; Files files(r.file_cleanup_failures); std::array<struct stat, 5> initial{};
    const auto stdout_stat = status(STDOUT_FILENO), stderr_stat = status(STDERR_FILENO);
    for (std::size_t i = 0; i < 5; ++i) {
        files.fd[i] = ::open(r.paths[i].c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
        if (files.fd[i] < 0) throw std::system_error(errno, std::generic_category(), "open " + r.paths[i]);
        initial[i] = status(files.fd[i]); require(S_ISREG(initial[i].st_mode) && initial[i].st_size >= 0, "source must be regular file");
        const auto alias = [&](const struct stat& s) { return s.st_dev == initial[i].st_dev && s.st_ino == initial[i].st_ino; };
        r.stdout_safe &= !alias(stdout_stat); r.stderr_safe &= !alias(stderr_stat);
    }
    require(r.stdout_safe && r.stderr_safe, "source aliases stdout/stderr");
    const std::string_view text(argv[2]); const auto parsed = std::from_chars(text.data(), text.data() + text.size(), r.layer);
    require(parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size() && r.layer >= 0 && r.layer < 48 && r.layer % 4 != 3, "LAYER must be GDN 0..47 except 3 mod 4");
    { const qwen::Model m(r.paths[0]); const auto& arch = m.metadata_value("general.architecture"); const auto& blocks = m.metadata_value("qwen4exp.block_count");
        require(arch.type == qwen::MetadataType::STRING && arch.get<std::string>() == "qwen4exp" && blocks.type == qwen::MetadataType::UINT32 && blocks.get<std::uint32_t>() == 48, "expected qwen4exp/block48");
        r.model_bytes = m.file_size(); require(r.model_bytes == std::uint64_t(initial[0].st_size), "model size changed");
        const auto prefix = "blk." + std::to_string(r.layer) + "."; r.weights = {m.tensor(prefix + "ssm_dt.bias"), m.tensor(prefix + "ssm_a")};
        for (std::size_t i = 0; i < 2; ++i) parameter(m, r.weights[i], r.host[i], r.payload_bytes_read);
    }
    for (std::size_t i = 0; i < heads; ++i) require(r.host[1].data()[i] <= 0, "ssm_a must already be -exp(A_log), <=0");
    for (std::size_t i = 1; i < 5; ++i) read_f32(files.fd[i], r.host[i + 1]);
    for (std::size_t i = 0; i < 5; ++i) require(same(initial[i], status(files.fd[i])), "source file changed");
    for (auto& h : r.host) { h.freeze(); } r.source_valid = true; r.host_immutable = true;
    check(hipGetDeviceCount(&r.count), "device count"); require(r.count == 2, "requires exactly two gfx906/wave64 GPUs");
    for (int d = 0; d < 2; ++d) { check(hipGetDeviceProperties(&r.properties[d], d), "device properties"); const auto& p = r.properties[d];
        const std::string_view arch(p.gcnArchName); require((arch == "gfx906" || arch.starts_with("gfx906:")) && p.warpSize == 64 && p.maxThreadsPerBlock >= 64, "requires gfx906/wave64"); }
    r.topology = true;
    for (int d = 0; d < 2; ++d) {
        try { run(d, r.host, r.devices[d]); } catch (const std::exception& e) { r.devices[d].error = e.what(); }
        r.host_immutable = std::all_of(r.host.begin(), r.host.end(), [](const Host& h) { return h.immutable(); });
        require(r.host_immutable, "host inputs/parameters/references/canaries changed");
        require(r.devices[d].drained, "HIP drain unconfirmed; skipping further work");
    }
    for (std::size_t i = 0; i < 5; ++i) require(same(initial[i], status(files.fd[i])), "source file changed during GPU diagnostic");
}
void common() {
    std::cout << ",\"protocol\":1,\"revision\":" << quote(CORE_REVISION) << ",\"dirty\":" << bool(CORE_DIRTY)
        << ",\"diagnostic_only\":true,\"correctness_only\":true,\"performance_claim\":false,\"whole_logits_gate_waived\":false,\"bit_exact_required\":false"
        << ",\"control_semantics\":\"same_private_helper_recomputed_not_recurrence_register_capture\",\"api\":\"qwen::launch_gdn_control_witness\""
        << ",\"witness_source\":\"src/gdn_controls_witness.hip\",\"helper_source\":\"src/hip/gdn.hip\",\"absolute_gate\":" << absolute_gate << ",\"relative_gate\":" << relative_gate << ",\"bound\":\"abs(actual-reference)<=2e-4+2e-4*abs(reference)\""; }
void pair(float a, float b) { std::cout << "{\"actual\":"; value(a); std::cout << ",\"reference\":"; value(b); std::cout << ",\"actual_bits\":" << std::bit_cast<std::uint32_t>(a) << ",\"reference_bits\":" << std::bit_cast<std::uint32_t>(b) << '}'; }
} // namespace
int main(int argc, char** argv) {
    std::cout.imbue(std::locale::classic()); std::cout << std::boolalpha << std::setprecision(17); Results r;
    try { execute(argc, argv, r); } catch (const std::exception& e) { r.error = e.what(); }
    if (!r.stdout_safe || !r.stderr_safe) return 1;
    const bool source_pass = r.source_valid && r.host_immutable && r.file_cleanup_failures == 0;
    std::cout << "{\"kind\":\"gdn_controls_witness_source\""; common(); std::cout << ",\"layer\":" << r.layer << ",\"model_bytes\":" << r.model_bytes << ",\"file_endianness\":\"little\",\"paths\":[";
    for (std::size_t i = 0; i < 5; ++i) { if (i) std::cout << ','; std::cout << quote(r.paths[i]); }
    std::cout << "],\"path_roles\":[\"model\",\"alpha\",\"beta\",\"original_log_decay\",\"original_beta\"],\"direct_original_weights\":true,\"weights\":[";
    for (std::size_t i = 0; i < 2; ++i) { if (i) std::cout << ','; const auto& t = r.weights[i]; std::cout << "{\"name\":" << quote(t.name) << ",\"type_id\":" << std::uint32_t(t.type) << ",\"rank\":" << t.rank << ",\"dimensions\":[" << t.dimensions[0] << "],\"elements\":" << t.elements << ",\"file_offset\":" << t.file_offset << ",\"bytes\":" << t.byte_size << '}'; }
    std::cout << "],\"tensor_payload_bytes_read\":" << r.payload_bytes_read << ",\"raw_file_bytes_each\":192,\"reference_lineage_verified_by_binary\":false,\"validated\":" << r.source_valid << ",\"host_guard_regions\":12,\"host_inputs_parameters_references_and_canaries_immutable\":" << r.host_immutable << ",\"file_cleanup_failures\":" << r.file_cleanup_failures << ",\"passed\":" << source_pass << "}\n";
    for (int d = 0; d < 2; ++d) { const auto& x = r.devices[d]; const auto& p = r.properties[d];
        std::cout << "{\"kind\":\"gdn_controls_witness_device\",\"device\":" << d; common();
        std::cout << ",\"gcn_arch\":" << quote(p.gcnArchName) << ",\"wave_size\":" << p.warpSize << ",\"attempted\":" << x.attempted << ",\"completed\":" << x.completed << ",\"log_decay\":";
        if (x.completed) metrics(x.comparisons[0]); else std::cout << "null"; std::cout << ",\"sigmoid_beta\":"; if (x.completed) metrics(x.comparisons[1]); else std::cout << "null";
        std::cout << ",\"exp_log_decay\":{\"count\":48,\"all_finite\":" << (x.completed && x.decay_nonfinite == 0) << ",\"nonfinite\":"; if (x.completed) std::cout << x.decay_nonfinite; else std::cout << "null";
        std::cout << ",\"original_device_exp_comparator\":null},\"heads\":[";
        if (x.completed) for (std::size_t h = 0; h < heads; ++h) { if (h) std::cout << ','; std::cout << "{\"head\":" << h << ",\"log_decay\":"; pair(x.output[h], r.host[4].data()[h]);
            std::cout << ",\"sigmoid_beta\":"; pair(x.output[heads + h], r.host[5].data()[h]); std::cout << ",\"exp_log_decay\":{\"actual\":"; value(x.output[2 * heads + h]); std::cout << ",\"actual_bits\":" << std::bit_cast<std::uint32_t>(x.output[2 * heads + h]) << "}}"; }
        std::cout << "],\"sticky_error\":" << x.flag << ",\"canaries_intact\":" << x.guards << ",\"guard_checks\":" << x.guard_checks << ",\"inputs_parameters_immutable\":" << x.immutable << ",\"immutable_checks\":" << x.immutable_checks
            << ",\"allocations\":" << x.allocations << ",\"allocated_bytes_including_guards\":" << x.allocated_bytes << ",\"cleanup_attempted\":" << x.cleanup_attempted << ",\"owner_drained\":" << x.drained << ",\"cleanup_complete\":" << x.cleaned << ",\"cleanup_failures\":" << x.cleanup_failures << ",\"error\":" << (x.error.empty() ? "null" : quote(x.error)) << ",\"passed\":" << (source_pass && x.passed()) << "}\n";
    }
    const bool pass = r.error.empty() && source_pass && r.topology && r.devices[0].passed() && r.devices[1].passed();
    std::cout << "{\"kind\":\"gdn_controls_witness_complete\""; common(); std::cout << ",\"source_records\":1,\"device_records\":2,\"discovered_devices\":" << r.count << ",\"topology_validated\":" << r.topology
        << ",\"required_comparisons\":4,\"compared_elements\":" << (96 * (int(r.devices[0].completed) + int(r.devices[1].completed))) << ",\"cleanup_complete\":" << (r.file_cleanup_failures == 0 && r.devices[0].cleaned && r.devices[1].cleaned) << ",\"primary_gate\":\"finite_full48_log_decay_and_beta_frozen_bound_plus_decay_finite_guards_immutable_error0_checked_cleanup\",\"error\":" << (r.error.empty() ? "null" : quote(r.error)) << ",\"passed\":" << pass << "}\n";
    std::cout.flush(); return pass && std::cout ? 0 : 1;
}
