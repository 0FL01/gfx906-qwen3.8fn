#include "hip/attention.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Standalone model-free fixture. Compile this, src/hip/attention.hip,
// src/attention.cpp and src/quant.cpp with C++20/HIP20, gfx906,
// -ffp-contract=off, no fast/finite-only math, and caller-supplied
// -DCORE_REVISION='"<compiled revision>"' -DCORE_DIRTY=0|1. No guessed provenance.
#ifndef CORE_REVISION
#error "Supply the compiled source revision"
#endif
#ifndef CORE_DIRTY
#error "Supply the compiled source dirty flag"
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Attention fixtures require strict finite checks and unfused arithmetic"
#endif

namespace {
using namespace qwen;
constexpr std::size_t guard_bytes = 256;
constexpr float sentinel = -777.0f;
// EXACT existing src/qsa_main.hip gate, frozen before any execution.
constexpr double absolute_gate = 2e-4, relative_gate = 2e-4;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// Optional cross-build witness, separate from the frozen CPU/numeric gates.
// Native little-endian uint64 headers and FP32 values on this same gfx906 host.
// A footer is written only after both GPUs and all failure/cleanup tests pass.
class Snapshot {
    std::ofstream output;
    std::ifstream input;
public:
    bool compare;
    std::uint64_t cases = 0, values = 0;
    Snapshot(bool read, const char* path) : compare(read) {
        if (compare) { input.open(path, std::ios::binary); require(bool(input), "snapshot open failed"); }
        else {
            require(!std::filesystem::exists(path), "snapshot capture requires fresh path");
            output.open(path, std::ios::binary); require(bool(output), "snapshot create failed");
        }
        const std::array<std::uint64_t, 2> header{0x5153414254303031ULL, sizeof(float)};
        transfer(header.data(), sizeof(header));
    }
    void transfer(const void* data, std::size_t bytes) {
        if (compare) {
            std::vector<char> reference(bytes);
            input.read(reference.data(), static_cast<std::streamsize>(bytes));
            require(bool(input), "snapshot truncated");
            require(std::memcmp(reference.data(), data, bytes) == 0, "cross-build attention bits differ");
        } else {
            output.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
            require(bool(output), "snapshot write failed");
        }
    }
    void observe(int device, int batch, int first, int capacity, const std::vector<float>& data) {
        const std::array<std::uint64_t, 5> header{static_cast<std::uint64_t>(device),
            static_cast<std::uint64_t>(batch), static_cast<std::uint64_t>(first),
            static_cast<std::uint64_t>(capacity), data.size()};
        transfer(header.data(), sizeof(header)); transfer(data.data(), data.size() * sizeof(float));
        ++cases; values += data.size();
    }
    void finish() {
        const std::array<std::uint64_t, 3> footer{0x515341454e443031ULL, cases, values};
        transfer(footer.data(), sizeof(footer));
        if (compare) require(input.peek() == std::char_traits<char>::eof(), "snapshot trailing data");
        else { output.close(); require(bool(output), "snapshot close failed"); }
    }
};
Snapshot* cross_build = nullptr;
void check(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
void cleanup(hipError_t e) noexcept {
    if (e != hipSuccess) std::abort();
}
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { cleanup(hipStreamSynchronize(value)); cleanup(hipStreamDestroy(value)); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Graph {
    hipGraph_t value = nullptr;
    ~Graph() { if (value) cleanup(hipGraphDestroy(value)); }
};
struct Capture {
    hipStream_t stream;
    Graph graph;
    bool active = false;
    explicit Capture(hipStream_t s) : stream(s) {
        check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal)); active = true;
    }
    ~Capture() { if (active) cleanup(hipStreamEndCapture(stream, &graph.value)); }
    void finish() { check(hipStreamEndCapture(stream, &graph.value)); active = false; }
};

template<class T> struct Buffer {
    void* base = nullptr;
    T* data = nullptr;
    std::size_t elements;
    hipStream_t stream;
    Buffer(std::size_t n, hipStream_t s) : elements(n), stream(s) {
        check(hipMalloc(&base, bytes() + 2 * guard_bytes));
        data = reinterpret_cast<T*>(static_cast<std::byte*>(base) + guard_bytes);
        try {
            check(hipMemsetAsync(base, 0xa5, bytes() + 2 * guard_bytes, stream));
            check(hipStreamSynchronize(stream));
        } catch (...) { cleanup(hipFree(base)); throw; }
    }
    ~Buffer() { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(base)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    std::size_t bytes() const { return elements * sizeof(T); }
    void put(std::span<const T> host) {
        require(host.size() == elements, "upload shape");
        check(hipMemcpyAsync(data, host.data(), bytes(), hipMemcpyHostToDevice, stream));
        check(hipStreamSynchronize(stream));
    }
    std::vector<T> get() const {
        std::vector<T> host(elements);
        check(hipMemcpyAsync(host.data(), data, bytes(), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        return host;
    }
    void guards() const {
        std::array<std::byte, guard_bytes> a{}, b{};
        check(hipMemcpyAsync(a.data(), base, guard_bytes, hipMemcpyDeviceToHost, stream));
        check(hipMemcpyAsync(b.data(), static_cast<std::byte*>(base) + guard_bytes + bytes(),
                             guard_bytes, hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        for (auto x : a) require(x == std::byte{0xa5}, "leading guard");
        for (auto x : b) require(x == std::byte{0xa5}, "trailing guard");
    }
    AttentionQ4BatchView<T> view(std::size_t stride) { return {data, elements, stride}; }
    AttentionQ4BatchView<const T> read(std::size_t stride) const { return {data, elements, stride}; }
};
template<class T> void exact(std::span<const T> a, std::span<const T> b, const char* message) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0, message);
}
template<class T> void unchanged(const Buffer<T>& buffer, const std::vector<T>& before) {
    const auto after = buffer.get();
    exact<T>(after, before, "read-only or rejected buffer changed");
}
template<class T> std::vector<T> poison(std::size_t n) {
    std::vector<T> v(n);
    std::memset(v.data(), 0xa5, n * sizeof(T));
    return v;
}
template<class T> void gaps(const Buffer<T>& buffer, int batch, std::size_t stride, std::size_t width) {
    const auto host = buffer.get();
    for (int q = 0; q < batch; ++q) {
        const auto bytes = std::as_bytes(std::span(host).subspan(std::size_t(q) * stride + width, stride - width));
        for (auto x : bytes) require(x == std::byte{0xa5}, "row padding/scratch neighbor changed");
    }
    buffer.guards();
}

struct Timing { int batch, first; float completed_ms; };
struct Event {
    hipEvent_t value = nullptr;
    Event() { check(hipEventCreate(&value)); }
    ~Event() { if (value) cleanup(hipEventDestroy(value)); }
};
bool measure_performance = false;
struct Summary {
    std::vector<Timing> timings;
    int device = -1;
    std::size_t cases = 0, queries = 0, bit_elements = 0, cpu_elements = 0;
    std::size_t device_rejects = 0, host_rejects = 0, sticky_checks = 0;
    double max_abs = 0, max_ratio = 0;
    void gate(std::span<const float> got, std::span<const float> ref) {
        require(got.size() == ref.size(), "oracle shape");
        for (std::size_t i = 0; i < got.size(); ++i) {
            require(std::isfinite(got[i]) && std::isfinite(ref[i]), "nonfinite oracle/output");
            const double error = std::abs(double(got[i]) - double(ref[i]));
            const double bound = absolute_gate + relative_gate * std::abs(double(ref[i]));
            max_abs = std::max(max_abs, error); max_ratio = std::max(max_ratio, error / bound);
            require(error <= bound, "frozen gathered-FP16 oracle gate");
            ++cpu_elements;
        }
    }
};

// Deliberately padded rows, including IDs/counts and EVERY private workspace.
struct Fixture {
    int batch, first, capacity;
    std::size_t qs = 6160, is = 2060, cs = 4;
    std::size_t gs = attention_gather_elements + 8, ps = attention_partial_elements + 8;
    std::size_t ms = attention_partial_ml_elements + 4;
    Stream stream;
    Buffer<float> query, output;
    Buffer<Q4_0> keys, values;
    Buffer<std::int32_t> ids;
    Buffer<int> counts, error;
    Buffer<std::uint16_t> gk, gv;
    Buffer<float> partial, staged;
    Buffer<float2> ml;
    std::vector<float> hq, ho;
    std::vector<Q4_0> hk, hv;
    std::vector<std::int32_t> hi;
    std::vector<int> hc;
    Fixture(int b, int f, int cap = 0) : batch(b), first(f), capacity(cap ? cap : f + b + 8),
        query(std::size_t(b) * qs, stream.value), output(std::size_t(b) * qs, stream.value),
        keys(std::size_t(capacity) * 16, stream.value), values(std::size_t(capacity) * 16, stream.value),
        ids(std::size_t(b) * is, stream.value), counts(std::size_t(b) * cs, stream.value), error(1, stream.value),
        gk(std::size_t(b) * gs, stream.value), gv(std::size_t(b) * gs, stream.value),
        partial(std::size_t(b) * ps, stream.value), staged(std::size_t(b) * qs, stream.value),
        ml(std::size_t(b) * ms, stream.value), hq(poison<float>(query.elements)), ho(poison<float>(output.elements)),
        hk(keys.elements), hv(values.elements), hi(poison<std::int32_t>(ids.elements)), hc(poison<int>(counts.elements)) {
        for (int q = 0; q < batch; ++q) {
            std::fill_n(ho.data() + std::size_t(q) * qs, attention_query_elements, sentinel);
            for (std::size_t d = 0; d < attention_query_elements; ++d)
                hq[std::size_t(q) * qs + d] = .7f * std::sin(float(d + q * 37) * .173f) +
                                            .1f * std::cos(float(d + q * 11) * .037f);
        }
    }
    AttentionQ4BatchCache cache() const {
        return {keys.data, keys.elements, values.data, values.elements, capacity, first + batch - 1};
    }
    AttentionQ4BatchWorkspace workspace() {
        return {gk.view(gs), gv.view(gs), partial.view(ps), ml.view(ms), staged.view(qs), error.data};
    }
    hipError_t call() {
        return launch_attention_q4_batch(query.read(qs), cache(), batch, first,
            ids.read(is), counts.read(cs), output.view(qs), workspace(), stream.value);
    }
    void flag(int bits) { const std::array<int, 1> h{bits}; error.put(h); }
    int flag() { return error.get()[0]; }
    void upload() { query.put(hq); output.put(ho); keys.put(hk); values.put(hv); ids.put(hi); counts.put(hc); }
    void sentinel_output() { output.put(ho); }
    // Whole blocks plus actual tail, budget 512 blocks, ranked unsorted input.
    // This is test-owned selector input, not an alternative production selector.
    void selection(bool short_members = false, bool duplicates = false) {
        for (int q = 0; q < batch; ++q) {
            const int visible = first + q, complete = visible / 4, tail = visible % 4;
            const int blocks = std::min(complete, 512);
            std::vector<std::int32_t> row;
            // Rotate which complete block is omitted beyond the budget, then
            // reverse block rank. The full batch has distinct query selections.
            const int start = complete > blocks ? q % (complete - blocks + 1) : 0;
            for (int k = blocks - 1; k >= 0; --k)
                for (int j = 0; j < 4; ++j) row.push_back((start + k) * 4 + j);
            for (int j = 0; j < tail; ++j) row.push_back(complete * 4 + j);
            if (short_members && q + 1 < batch) {
                constexpr std::array<int, 6> sizes{1, 2, 3, 63, 64, 65};
                row.resize(std::min<std::size_t>(row.size(), std::size_t(sizes[std::size_t(q) % sizes.size()])));
            }
            if (duplicates && row.size() >= 2) row[1] = row[0]; // valid repeated entry
            std::reverse(row.begin(), row.end());
            hc[std::size_t(q) * cs] = static_cast<int>(row.size());
            hc[std::size_t(q) * cs + 1] = short_members || duplicates ? 0 : blocks;
            std::copy(row.begin(), row.end(), hi.begin() + static_cast<std::ptrdiff_t>(std::size_t(q) * is));
        }
        std::vector<bool> selected(std::size_t(capacity), false);
        for (int q = 0; q < batch; ++q) for (int j = 0; j < hc[std::size_t(q) * cs]; ++j)
            selected[std::size_t(hi[std::size_t(q) * is + std::size_t(j)])] = true;
        for (int t = 0; t < capacity; ++t) for (int b = 0; b < 16; ++b) {
            auto& k = hk[std::size_t(t) * 16 + std::size_t(b)];
            auto& v = hv[std::size_t(t) * 16 + std::size_t(b)];
            // Poison every future/unselected row. Finite rows chosen only by a
            // later query still vary with t, making earlier-query leakage visible.
            k.d = selected[std::size_t(t)] ? std::uint16_t(0x2803 + (t * 19 + b * 31) % 511 + ((t + b) % 2 ? 0x8000 : 0)) : 0x7e01;
            v.d = selected[std::size_t(t)] ? std::uint16_t(0x2c01 + (t * 7 + b * 43) % 701 + ((t + 2 * b) % 3 ? 0x8000 : 0)) : 0x7c00;
            for (int j = 0; j < 16; ++j) {
                k.qs[j] = std::uint8_t(((t * 5 + b + j * 3) % 16) | (((t + b * 7 + j * 11) % 16) << 4));
                v.qs[j] = std::uint8_t(((t + b * 13 + j * 7) % 16) | (((t * 3 + b * 5 + j) % 16) << 4));
            }
        }
    }
    void sources() {
        unchanged(query, hq); unchanged(keys, hk); unchanged(values, hv); unchanged(ids, hi); unchanged(counts, hc);
        query.guards(); keys.guards(); values.guards(); ids.guards(); counts.guards(); error.guards();
    }
    void neighbors() {
        gaps(output, batch, qs, attention_query_elements); gaps(staged, batch, qs, attention_query_elements);
        gaps(gk, batch, gs, attention_gather_elements); gaps(gv, batch, gs, attention_gather_elements);
        gaps(partial, batch, ps, attention_partial_elements); gaps(ml, batch, ms, attention_partial_ml_elements);
    }
    void poison_workspace() {
        auto gathered = poison<std::uint16_t>(gk.elements);
        for (int q = 0; q < batch; ++q)
            std::fill_n(gathered.data() + std::size_t(q) * gs, attention_gather_elements, std::uint16_t{0x7e01});
        gk.put(gathered); gv.put(gathered);
        auto p = poison<float>(partial.elements);
        for (int q = 0; q < batch; ++q)
            std::fill_n(p.data() + std::size_t(q) * ps, attention_partial_elements, std::numeric_limits<float>::quiet_NaN());
        partial.put(p);
        auto m = poison<float2>(ml.elements);
        for (int q = 0; q < batch; ++q)
            std::fill_n(m.data() + std::size_t(q) * ms, attention_partial_ml_elements,
                        make_float2(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
        ml.put(m);
    }
    void unused_workspace() {
        for (const auto* buffer : {&gk, &gv}) {
            const auto h = buffer->get();
            for (int q = 0; q < batch; ++q)
                for (std::size_t i = std::size_t(hc[std::size_t(q) * cs]) * 512; i < attention_gather_elements; ++i)
                    require(h[std::size_t(q) * gs + i] == 0x7e01, "unselected gather capacity was written");
        }
        const auto p = partial.get(); const auto m = ml.get();
        for (int q = 0; q < batch; ++q) {
            const auto chunks = std::size_t((hc[std::size_t(q) * cs] + 63) / 64);
            for (std::size_t i = chunks * attention_query_elements; i < attention_partial_elements; ++i)
                require(std::isnan(p[std::size_t(q) * ps + i]), "inactive partial chunk was written");
            for (std::size_t i = chunks * attention_query_heads; i < attention_partial_ml_elements; ++i)
                require(std::isnan(m[std::size_t(q) * ms + i].x) && std::isnan(m[std::size_t(q) * ms + i].y), "inactive ml chunk was written");
        }
    }
};

void parity(Fixture& f, Summary& s) {
    f.flag(0); f.upload(); f.poison_workspace(); check(f.call()); check(hipStreamSynchronize(f.stream.value));
    require(f.flag() == 0, "valid batch error");
    f.unused_workspace();
    const auto batch_output = f.output.get();
    // Sequential CURRENT N1 on the same GPU/cache/Q/IDs/count/visibility and
    // the same private workspace rows. It cannot alter the saved batch result.
    f.sentinel_output();
    for (int q = 0; q < f.batch; ++q) {
        const auto row = std::size_t(q);
        AttentionQ4DeviceWorkspace w{f.gk.data + row * f.gs, f.gv.data + row * f.gs,
            f.partial.data + row * f.ps, f.ml.data + row * f.ms, f.staged.data + row * f.qs, f.error.data};
        check(launch_attention_q4(f.query.data + row * f.qs, f.keys.data, f.values.data, f.capacity,
            f.ids.data + row * f.is, f.counts.data + row * f.cs, f.first + q, f.output.data + row * f.qs, w, f.stream.value));
    }
    check(hipStreamSynchronize(f.stream.value)); require(f.flag() == 0, "valid N1 error");
    const auto scalar_output = f.output.get();
    exact<float>(batch_output, scalar_output, "batch versus CURRENT N1 must be bitwise identical including padding");
    for (int q = 0; q < f.batch; ++q) {
        const auto row = std::size_t(q);
        const auto n = std::size_t(f.hc[row * f.cs]);
        std::vector<std::int32_t> ordered(f.hi.begin() + static_cast<std::ptrdiff_t>(row * f.is),
                                        f.hi.begin() + static_cast<std::ptrdiff_t>(row * f.is + n));
        std::sort(ordered.begin(), ordered.end());
        std::vector<float> scores(n), staged(attention_query_elements), ref(attention_query_elements);
        attention_q4({f.hk, f.hv, std::size_t(f.capacity)},
            std::span<const float>(f.hq).subspan(row * f.qs, attention_query_elements), ordered,
            std::size_t(f.first + q), AttentionQ4Mode::gathered_fp16, {scores, staged}, ref);
        s.gate(std::span<const float>(batch_output).subspan(row * f.qs, attention_query_elements), ref);
        s.bit_elements += attention_query_elements; ++s.queries;
    }
    f.sources(); f.neighbors();
    if (cross_build) cross_build->observe(s.device, f.batch, f.first, f.capacity, batch_output);
    ++s.cases;
}

// Rejected device inputs: bad LAST query must not publish even valid neighbors.
// All non-gather preflight failures leave both gathers, ml and staged untouched;
// sorted-ID temporary partial words are explicitly allowed to be provisional.
void failed(Fixture& f, Summary& s, int expected, bool preflight) {
    f.flag(0); f.upload();
    const auto k = f.gk.get(), v = f.gv.get();
    const auto ml = f.ml.get(); const auto staged = f.staged.get();
    check(f.call()); check(hipStreamSynchronize(f.stream.value));
    require(f.flag() == expected, "device error class"); unchanged(f.output, f.ho);
    if (preflight) { unchanged(f.gk, k); unchanged(f.gv, v); unchanged(f.ml, ml); unchanged(f.staged, staged); }
    f.sources(); f.neighbors(); ++s.device_rejects;
}

void device_failures(Summary& s) {
    Fixture f(3, 65); f.selection(true);
    const auto base_q = f.hq; const auto base_ids = f.hi; const auto base_counts = f.hc;
    const auto base_k = f.hk, base_v = f.hv;
    const auto last = std::size_t(f.batch - 1), at = last * f.cs;
    for (int n : {-1, 0, 2052, std::numeric_limits<int>::max()}) {
        f.hc[at] = n; failed(f, s, attention_error_invalid, true); f.hc = base_counts;
    }
    for (int b : {-1, 513, std::numeric_limits<int>::max(), 17}) {
        f.hc[at + 1] = b; failed(f, s, attention_error_invalid, true); f.hc = base_counts;
    }
    // Separately tokens/4 constraint while visible/4 is large enough.
    f.hc[at] = 1; f.hc[at + 1] = 1; failed(f, s, attention_error_invalid, true); f.hc = base_counts;
    for (auto id : {-1, f.first + f.batch - 1, std::numeric_limits<int>::max()}) {
        f.hi[last * f.is] = id; failed(f, s, attention_error_invalid, true); f.hi = base_ids;
    }
    for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        f.hq[last * f.qs + attention_query_elements - 1] = bad;
        failed(f, s, attention_error_invalid, true); f.hq = base_q;
    }
    const auto selected = std::size_t(f.hi[last * f.is]);
    for (auto bits : {std::uint16_t{0x7c00}, std::uint16_t{0x7e01}, std::uint16_t{0xfc00}}) {
        f.hk[selected * 16 + 15].d = bits; failed(f, s, attention_error_invalid, true); f.hk = base_k;
        f.hv[selected * 16 + 15].d = bits; failed(f, s, attention_error_invalid, true); f.hv = base_v;
    }
    for (bool value : {false, true}) {
        auto& block = value ? f.hv[selected * 16 + 15] : f.hk[selected * 16 + 15];
        block.d = 0x7bff; std::fill(std::begin(block.qs), std::end(block.qs), std::uint8_t{0xff});
        failed(f, s, attention_error_arithmetic, false); f.hk = base_k; f.hv = base_v;
    }
    // Finite Q with finite gathered operands can still overflow FP32 dots.
    std::fill_n(f.hq.data() + last * f.qs, attention_query_elements, std::numeric_limits<float>::max());
    // Positive >1 K operands force multiplication overflow before cancellation.
    for (int j = 0; j < f.hc[at]; ++j) {
        const auto id = std::size_t(f.hi[last * f.is + std::size_t(j)]);
        for (int b = 0; b < 16; ++b) {
            auto& k = f.hk[id * 16 + std::size_t(b)]; k.d = 0x3c00;
            std::fill(std::begin(k.qs), std::end(k.qs), std::uint8_t{0xff});
        }
    }
    failed(f, s, attention_error_arithmetic, false);
    // Overflow followed by opposite-sign products must not escape detection.
    for (std::size_t d = 0; d < attention_query_elements; ++d)
        f.hq[last * f.qs + d] = d % 2 ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
    failed(f, s, attention_error_arithmetic, false);
    // Each product is finite, but the ascending addition chain overflows.
    std::fill_n(f.hq.data() + last * f.qs, attention_query_elements, std::numeric_limits<float>::max() / 16.0f);
    failed(f, s, attention_error_arithmetic, false);
    f.hq = base_q; f.hk = base_k;

    // Valid inputs under sticky 1/2/3/high-bit flags: no implicit clear and NO
    // scratch/output change at all. Reuse after EXPLICIT caller reset succeeds.
    f.upload();
    auto sticky = [&](int bits) {
        f.sentinel_output();
        const auto k = f.gk.get(), v = f.gv.get(); const auto p = f.partial.get();
        const auto m = f.ml.get(); const auto o = f.staged.get();
        check(f.call()); check(hipStreamSynchronize(f.stream.value));
        require(f.flag() == bits, "sticky flag cleared"); unchanged(f.output, f.ho);
        unchanged(f.gk, k); unchanged(f.gv, v); unchanged(f.partial, p); unchanged(f.ml, m); unchanged(f.staged, o);
        ++s.sticky_checks;
    };
    sticky(attention_error_arithmetic); // actual previous failure, NO clear/set
    for (int bits : {1, 2, 3, 8}) { f.flag(bits); sticky(bits); }
    // Invalid count is reported even with arithmetic already sticky (N1 ready).
    f.hc[at] = 0; f.counts.put(f.hc); f.flag(attention_error_arithmetic);
    check(f.call()); check(hipStreamSynchronize(f.stream.value)); require(f.flag() == 3, "sticky atomic OR");
    unchanged(f.output, f.ho); ++s.sticky_checks; f.hc = base_counts;
    parity(f, s); // valid reuse of the SAME previously provisional workspace
}

struct Args {
    AttentionQ4BatchView<const float> query;
    AttentionQ4BatchCache cache;
    int batch, first;
    AttentionQ4BatchView<const std::int32_t> ids;
    AttentionQ4BatchView<const int> counts;
    AttentionQ4BatchView<float> output;
    AttentionQ4BatchWorkspace w;
    hipStream_t stream;
    hipError_t call() const {
        return launch_attention_q4_batch(query, cache, batch, first, ids, counts, output, w, stream);
    }
    void* pointer(int slot) const {
        const std::array<const void*, 12> p{query.data, cache.keys, cache.values, ids.data, counts.data,
            output.data, w.gathered_keys.data, w.gathered_values.data, w.partial_output.data,
            w.partial_max_sum.data, w.staged_output.data, w.error};
        return const_cast<void*>(p[std::size_t(slot)]);
    }
    void pointer(int slot, void* p) {
        switch (slot) {
        case 0: query.data = static_cast<const float*>(p); break;
        case 1: cache.keys = static_cast<const Q4_0*>(p); break;
        case 2: cache.values = static_cast<const Q4_0*>(p); break;
        case 3: ids.data = static_cast<const std::int32_t*>(p); break;
        case 4: counts.data = static_cast<const int*>(p); break;
        case 5: output.data = static_cast<float*>(p); break;
        case 6: w.gathered_keys.data = static_cast<std::uint16_t*>(p); break;
        case 7: w.gathered_values.data = static_cast<std::uint16_t*>(p); break;
        case 8: w.partial_output.data = static_cast<float*>(p); break;
        case 9: w.partial_max_sum.data = static_cast<float2*>(p); break;
        case 10: w.staged_output.data = static_cast<float*>(p); break;
        case 11: w.error = static_cast<int*>(p); break;
        default: throw std::runtime_error("slot");
        }
    }
};

void host_failures(Summary& s) {
    Fixture f(3, 65); f.selection(true); f.upload(); f.flag(8);
    const Args good{f.query.read(f.qs), f.cache(), f.batch, f.first, f.ids.read(f.is), f.counts.read(f.cs),
                    f.output.view(f.qs), f.workspace(), f.stream.value};
    const auto k = f.gk.get(), v = f.gv.get(); const auto p = f.partial.get();
    const auto m = f.ml.get(); const auto o = f.staged.get();
    // Graph capture supplies a real no-enqueue witness; zero captured nodes is
    // required. All calls below must reject by host metadata without using HIP.
    Capture capture(f.stream.value);
    auto reject = [&](const Args& a) { require(a.call() == hipErrorInvalidValue, "host accepted bad metadata"); ++s.host_rejects; };
    for (int slot = 0; slot < 12; ++slot) {
        auto a = good; a.pointer(slot, nullptr); reject(a);
        a = good; a.pointer(slot, reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(a.pointer(slot)) + 1)); reject(a);
        constexpr std::array<std::uintptr_t, 12> alignments{16, 2, 2, 4, 4, 4, 2, 2, 4, 8, 4, 4};
        a = good; a.pointer(slot, reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - (alignments[std::size_t(slot)] - 1))); reject(a);
    }
    for (int b : {-1, 0, 129, std::numeric_limits<int>::max()}) { auto a = good; a.batch = b; reject(a); }
    for (int first : {-1, 0, good.cache.valid_rows, std::numeric_limits<int>::max()}) { auto a = good; a.first = first; reject(a); }
    for (int cap : {-1, 0, 131073, std::numeric_limits<int>::max()}) { auto a = good; a.cache.capacity = cap; reject(a); }
    for (int rows : {-1, 0, good.cache.capacity + 1, std::numeric_limits<int>::max()}) { auto a = good; a.cache.valid_rows = rows; reject(a); }
    // Every row view: too-short stride, insufficient final row, stride wrap,
    // overflowing declared capacity, and a valid-width but misaligned Q stride.
    auto bad_view = [&](auto member, std::size_t width) {
        auto a = good; auto& x = member(a); x.stride = 0; reject(a);
        a = good; auto& y = member(a); y.elements = 1; reject(a);
        a = good; auto& z = member(a); z.stride = std::numeric_limits<std::size_t>::max(); reject(a);
        a = good; auto& t = member(a); t.elements = std::numeric_limits<std::size_t>::max(); reject(a);
        a = good; auto& u = member(a); u.elements = std::size_t(a.batch - 1) * u.stride + width - 1; reject(a);
    };
    bad_view([](Args& a) -> auto& { return a.query; }, attention_query_elements);
    bad_view([](Args& a) -> auto& { return a.ids; }, attention_max_selected);
    bad_view([](Args& a) -> auto& { return a.counts; }, 2);
    bad_view([](Args& a) -> auto& { return a.output; }, attention_query_elements);
    bad_view([](Args& a) -> auto& { return a.w.gathered_keys; }, attention_gather_elements);
    bad_view([](Args& a) -> auto& { return a.w.gathered_values; }, attention_gather_elements);
    bad_view([](Args& a) -> auto& { return a.w.partial_output; }, attention_partial_elements);
    bad_view([](Args& a) -> auto& { return a.w.partial_max_sum; }, attention_partial_ml_elements);
    bad_view([](Args& a) -> auto& { return a.w.staged_output; }, attention_query_elements);
    { auto a = good; ++a.query.stride; reject(a); }
    { auto a = good; --a.cache.key_elements; reject(a); }
    { auto a = good; --a.cache.value_elements; reject(a); }
    { auto a = good; a.cache.key_elements = std::numeric_limits<std::size_t>::max(); reject(a); }
    { auto a = good; a.cache.value_elements = std::numeric_limits<std::size_t>::max(); reject(a); }
    // ALL writable-vs-any range pairs, including the single shared error.
    for (int writable = 5; writable < 12; ++writable) for (int other = 0; other < 12; ++other) {
        if (writable == other) continue;
        auto a = good; a.pointer(writable, a.pointer(other)); reject(a);
    }
    // Alias occurring ONLY in the unused declared tail must also be rejected.
    { auto a = good; a.w.error = reinterpret_cast<int*>(a.output.data + std::size_t(a.batch - 1) * a.output.stride + attention_query_elements); reject(a); }
    capture.finish();
    std::size_t nodes = 0; check(hipGraphGetNodes(capture.graph.value, nullptr, &nodes)); require(nodes == 0, "host rejection enqueued work");
    require(f.flag() == 8, "host rejection changed sticky bits"); unchanged(f.output, f.ho);
    unchanged(f.gk, k); unchanged(f.gv, v); unchanged(f.partial, p); unchanged(f.ml, m); unchanged(f.staged, o);
    f.sources(); f.neighbors();
}

Summary device(int index) {
    check(hipSetDevice(index)); hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, index));
    require(std::string_view(prop.gcnArchName).starts_with("gfx906") && prop.warpSize == 64, "fixture requires gfx906 wave64");
    Summary s; s.device = index;
    // Each 2047..2056 case includes its actual complete-block/tail count (2051
    // only at 2051/2055); visibility never implies reading unused capacity.
    std::vector<int> lengths{1, 2, 3, 63, 64, 65};
    for (int n = 2047; n <= 2056; ++n) lengths.push_back(n);
    for (int n : lengths) { Fixture f(1, n); f.selection(); parity(f, s); }
    for (int b : {2, 3, 8}) { Fixture f(b, 2049); f.selection(true); parity(f, s); }
    for (int b : {5, 7, 127, 128}) { Fixture f(b, 1); f.selection(true); parity(f, s); }
    { Fixture f(3, 65); f.selection(true, true); parity(f, s); }
    { Fixture f(2, 131071, attention_max_capacity); f.selection(true); parity(f, s); }
    if (measure_performance) for (int batch : {1, 3, 8}) for (int first : {128, 2051, 4096, 16384}) {
        Fixture f(batch, first); f.selection(); parity(f, s);
        const auto expected = f.output.get();
        f.flag(0);
        for (int i = 0; i < 4; ++i) check(f.call());
        check(hipStreamSynchronize(f.stream.value));
        Event begin, end;
        check(hipEventRecord(begin.value, f.stream.value));
        for (int i = 0; i < 64; ++i) check(f.call());
        check(hipEventRecord(end.value, f.stream.value));
        check(hipEventSynchronize(end.value));
        float milliseconds = 0;
        check(hipEventElapsedTime(&milliseconds, begin.value, end.value));
        require(milliseconds > 0 && std::isfinite(milliseconds) && f.flag() == 0, "completed resident timing");
        const auto repeated = f.output.get();
        exact<float>(repeated, expected, "resident timing repeats changed output");
        f.sources(); f.neighbors();
        s.timings.push_back({batch, first, milliseconds / 64});
    }
    device_failures(s); host_failures(s);
    return s; // every fixture/stream/graph/buffer destroyed BEFORE serialization
}

void json_string(std::string_view text) {
    std::cout << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32) {
            constexpr char hex[] = "0123456789abcdef";
            std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else std::cout << char(c);
    }
    std::cout << '"';
}
} // namespace

int main(int argc, char** argv) {
    try {
        std::unique_ptr<Snapshot> snapshot;
        if (argc != 1) {
            require((argc == 3 || argc == 4) && (std::string_view(argv[1]) == "--capture" ||
                std::string_view(argv[1]) == "--compare"), "use --capture FRESH or --compare SNAPSHOT [--benchmark]");
            if (argc == 4) {
                require(std::string_view(argv[3]) == "--benchmark", "expected --benchmark");
                measure_performance = true;
            }
            snapshot = std::make_unique<Snapshot>(std::string_view(argv[1]) == "--compare", argv[2]);
            cross_build = snapshot.get();
        }
        int count = 0; check(hipGetDeviceCount(&count)); require(count == 2, "fixture requires both target GPUs");
        const std::array<Summary, 2> results{device(0), device(1)};
        if (snapshot) snapshot->finish();
        // Bounded JSONL only after BOTH GPUs and all RAII cleanup succeed.
        std::cout << std::setprecision(17) << "{\"kind\":\"attention_batch_source\",\"protocol\":1,\"revision\":";
        json_string(CORE_REVISION);
        std::cout << ",\"dirty\":" << (CORE_DIRTY ? "true" : "false")
                  << ",\"model_free\":true,\"domain\":\"qsa_selected_2051\",\"absolute_gate\":0.0002,\"relative_gate\":0.0002"
                     ",\"workspace_bytes_per_query\":5042368,\"workspace_B128_bytes\":645423104,\"performance_claim\":false}\n";
        for (const auto& s : results)
            std::cout << "{\"kind\":\"attention_batch_correctness\",\"protocol\":1,\"device\":" << s.device
                      << ",\"cases\":" << s.cases << ",\"queries\":" << s.queries << ",\"bit_elements\":" << s.bit_elements
                      << ",\"cpu_elements\":" << s.cpu_elements << ",\"max_abs\":" << s.max_abs << ",\"max_bound_ratio\":" << s.max_ratio
                      << ",\"device_rejects\":" << s.device_rejects << ",\"host_rejects\":" << s.host_rejects
                      << ",\"sticky_checks\":" << s.sticky_checks << ",\"passed\":true,\"cleanup_complete\":true}\n";
        for (const auto& s : results) for (const auto& t : s.timings)
            std::cout << "{\"kind\":\"attention_batch_timing\",\"device\":" << s.device
                      << ",\"batch\":" << t.batch << ",\"first_visible\":" << t.first
                      << ",\"completed_ms\":" << t.completed_ms
                      << ",\"iterations\":64,\"scope\":\"resident completed stream events; no transfers/model; not full-request speed\"}\n";
        if (snapshot)
            std::cout << "{\"kind\":\"attention_batch_cross_build\",\"mode\":\""
                      << (snapshot->compare ? "compare" : "capture") << "\",\"cases\":" << snapshot->cases
                      << ",\"values_including_padding\":" << snapshot->values << ",\"passed\":true}\n";
        return 0;
    } catch (const std::exception& e) {
        // Stack unwinding finishes RAII cleanup before the bounded failure row.
        std::cout << "{\"kind\":\"attention_batch_failure\",\"protocol\":1,\"passed\":false,\"error\":";
        json_string(std::string_view(e.what()).substr(0, 512)); std::cout << ",\"cleanup_complete\":true}\n";
        return 1;
    }
}
