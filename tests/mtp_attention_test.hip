#include "hip/mtp_attention.cuh"
#include "hip/attention.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

// Standalone, no model/sidecar, no runtime wiring or speed qualification.
// Target: compile THIS + hip/mtp_attention.hip + hip/attention.hip +
// attention.cpp + quant.cpp, HIP20/gfx906/-O3/-Wall/-Wextra/-Wpedantic/-Werror,
// -ffp-contract=off, no fast/finite-only math. Supply actual compiled
// -DCORE_REVISION='"<40hex>"' -DCORE_DIRTY=0|1. Both target GPUs are mandatory.
// CPU-only: g++ -x c++ -DMTP_ATTENTION_CPU_ONLY [same strict C++20 flags]
// this + attention.cpp + quant.cpp. Declaration-only HIP header is sufficient;
// this mode executes REAL host metadata/CPU arithmetic checks, no fake HIP gate.
#ifndef CORE_REVISION
#error "Supply compiled source revision"
#endif
#ifndef CORE_DIRTY
#error "Supply compiled dirty flag"
#endif
#if CORE_DIRTY != 0 && CORE_DIRTY != 1
#error "CORE_DIRTY must be 0 or 1"
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Fixture requires strict finite/unfused arithmetic"
#endif

namespace {
using namespace qwen;
constexpr std::size_t width = mtp_attention_query_elements, guard_bytes = 256;
constexpr float sentinel = -777.0f;
constexpr double absolute_gate = 2e-4, relative_gate = 2e-4; // frozen existing attention gate
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> std::vector<T> poison(std::size_t n) {
    std::vector<T> v(n); std::memset(v.data(), 0xa5, n * sizeof(T)); return v;
}
template<class T> void exact(std::span<const T> a, std::span<const T> b, const char* message) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0, message);
}
void provenance() {
    constexpr std::string_view revision = CORE_REVISION;
    require(revision.size() == 40, "revision must be actual 40hex");
    for (char c : revision) require((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'), "revision hex");
}
float finite(float x) { require(std::isfinite(x), "CPU nonfinite arithmetic"); return x; }
float decoded(const Q4_0& b, int d) {
    require((b.d & 0x7c00) != 0x7c00, "CPU nonfinite scale");
    const int code = d < 16 ? b.qs[d] & 15 : b.qs[d - 16] >> 4;
    return finite(half_to_float(float_to_half(half_to_float(b.d) * float(code - 8))));
}
void cache_data(std::vector<Q4_0>& k, std::vector<Q4_0>& v, int last, bool constant = false) {
    for (std::size_t b = 0; b < k.size(); ++b) {
        const auto t = b / 16, h = b % 16, pos = constant ? 0 : t;
        k[b].d = t < std::size_t(last) ? std::uint16_t(0x2803 + (pos * 19 + h * 31) % 511 + ((pos + h) % 2 ? 0x8000 : 0)) : 0x7e01;
        v[b].d = t < std::size_t(last) ? std::uint16_t(0x2c01 + (pos * 7 + h * 43) % 701 + ((pos + h * 2) % 3 ? 0x8000 : 0)) : 0x7c00;
        for (int j = 0; j < 16; ++j) {
            k[b].qs[j] = std::uint8_t(((pos * 5 + h + std::size_t(j) * 3) % 16) | (((pos + h * 7 + std::size_t(j) * 11) % 16) << 4));
            v[b].qs[j] = std::uint8_t(((pos + h * 13 + std::size_t(j) * 7) % 16) | (((pos * 3 + h * 5 + std::size_t(j)) % 16) << 4));
        }
    }
}
std::vector<float> query_data(int batch, std::size_t stride) {
    auto q = poison<float>(std::size_t(batch) * stride);
    for (int b = 0; b < batch; ++b) for (std::size_t d = 0; d < width; ++d)
        q[std::size_t(b) * stride + d] = .7f * std::sin(float(d + std::size_t(b) * 37) * .173f) +
                                       .1f * std::cos(float(d + std::size_t(b) * 11) * .037f);
    return q;
}

// Independent fixture-only unrestricted FULL causal gathered-F16 CPU oracle.
// Same equations/order as attention.cpp, without its QSA2051 shape restriction.
// Lookup merely caches EXACT Q4->FP32->FP16RNE->FP32 values for scale/code, not
// a precision change. Value loop interleaving preserves each dim's ascending
// token addition order and avoids repeated scalar half conversion at128K.
std::vector<float> dense_oracle(std::span<const Q4_0> keys, std::span<const Q4_0> values,
                               std::span<const float> query, int visible) {
    require(visible > 0 && visible <= 131072 && keys.size() >= std::size_t(visible) * 16 &&
            values.size() >= std::size_t(visible) * 16 && query.size() == width, "CPU oracle shape");
    for (float x : query) require(std::isfinite(x), "CPU nonfinite Q");
    for (int b = 0; b < visible * 16; ++b)
        require((keys[std::size_t(b)].d & 0x7c00) != 0x7c00 &&
                (values[std::size_t(b)].d & 0x7c00) != 0x7c00, "CPU full-prefix scale preflight");
    static const auto table = [] {
        std::vector<float> t(65536 * 16);
        for (unsigned bits = 0; bits < 65536; ++bits) for (int code = 0; code < 16; ++code)
            t[std::size_t(bits) * 16 + std::size_t(code)] = half_to_float(float_to_half(half_to_float(std::uint16_t(bits)) * float(code - 8)));
        return t;
    }();
    auto load = [&](const Q4_0& b, int d) {
        const auto code = d < 16 ? b.qs[d] & 15 : b.qs[d - 16] >> 4;
        return table[std::size_t(b.d) * 16 + std::size_t(code)];
    };
    for (int b = 0; b < visible * 16; ++b) for (int d = 0; d < 32; ++d)
        require(std::isfinite(load(keys[std::size_t(b)], d)) && std::isfinite(load(values[std::size_t(b)], d)), "CPU full-prefix half overflow");
    std::vector<float> scores(std::size_t(visible), 0.0f), output(width);
    for (int h = 0; h < 24; ++h) {
        float maximum = -std::numeric_limits<float>::infinity();
        const auto kv = std::size_t(h / 12) * 8;
        for (int j = 0; j < visible; ++j) {
            const auto base = std::size_t(j) * 16 + kv;
            float s = 0.0f;
            for (int d = 0; d < 256; ++d)
                s = finite(s + finite(query[std::size_t(h) * 256 + std::size_t(d)] * load(keys[base + std::size_t(d / 32)], d % 32)));
            scores[std::size_t(j)] = finite(s * .0625f);
            maximum = std::max(maximum, scores[std::size_t(j)]);
        }
        float sum = 0.0f;
        for (float& s : scores) { s = finite(std::exp(finite(s - maximum))); sum = finite(sum + s); }
        require(sum > 0, "CPU denominator"); const float inverse = finite(1.0f / sum);
        std::array<float, 256> numerator{};
        for (int j = 0; j < visible; ++j) {
            const auto base = std::size_t(j) * 16 + kv;
            for (int d = 0; d < 256; ++d)
                numerator[std::size_t(d)] = finite(numerator[std::size_t(d)] +
                    finite(scores[std::size_t(j)] * load(values[base + std::size_t(d / 32)], d % 32)));
        }
        for (int d = 0; d < 256; ++d) output[std::size_t(h) * 256 + std::size_t(d)] = finite(numerator[std::size_t(d)] * inverse);
    }
    return output;
}
// Constant-in-time K/V: the independent mathematical answer is the decoded V
// at every dim/head for ANY finite Q/visibility. Numeric gate still checks ALL
// 6144 outputs/query. This cheaply covers extra B2/B3 128K windows, while one
// NONconstant 128K B1 case runs the entire serial gathered-F16 oracle/24 heads.
std::vector<float> constant_oracle(std::span<const Q4_0> values) {
    std::vector<float> out(width);
    for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d)
        out[std::size_t(h) * 256 + std::size_t(d)] = decoded(values[std::size_t(h / 12) * 8 + std::size_t(d / 32)], d % 32);
    return out;
}
struct Args {
    MtpAttentionQ4View<const float> query;
    MtpAttentionQ4Cache cache;
    int batch, first;
    MtpAttentionQ4View<float> output;
    MtpAttentionQ4Workspace w;
    hipStream_t stream;
    void* pointer(int slot) const {
        const std::array<const void*, 8> p{query.data, cache.keys, cache.values, output.data,
            w.partial_output.data, w.partial_max_sum.data, w.staged_output.data, w.error};
        return const_cast<void*>(p[std::size_t(slot)]);
    }
    void pointer(int slot, void* p) {
        switch (slot) {
        case 0: query.data = static_cast<const float*>(p); break;
        case 1: cache.keys = static_cast<const Q4_0*>(p); break;
        case 2: cache.values = static_cast<const Q4_0*>(p); break;
        case 3: output.data = static_cast<float*>(p); break;
        case 4: w.partial_output.data = static_cast<float*>(p); break;
        case 5: w.partial_max_sum.data = static_cast<float2*>(p); break;
        case 6: w.staged_output.data = static_cast<float*>(p); break;
        case 7: w.error = static_cast<int*>(p); break;
        default: throw std::runtime_error("pointer slot");
        }
    }
};
template<class Reject> void bad_metadata(const Args& good, Reject reject) {
    for (int slot = 0; slot < 8; ++slot) {
        auto a = good; a.pointer(slot, nullptr); reject(a);
        a = good; a.pointer(slot, reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(a.pointer(slot)) + 1)); reject(a);
        constexpr std::array<std::uintptr_t, 8> alignments{16, 2, 2, 4, 4, 8, 4, 4};
        a = good; a.pointer(slot, reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - (alignments[std::size_t(slot)] - 1))); reject(a);
    }
    for (int b : {-1, 0, 4, std::numeric_limits<int>::max()}) { auto a = good; a.batch = b; reject(a); }
    for (int f : {-1, 0, good.cache.valid_rows, std::numeric_limits<int>::max()}) { auto a = good; a.first = f; reject(a); }
    for (int c : {-1, 0, 131073, std::numeric_limits<int>::max()}) { auto a = good; a.cache.capacity = c; reject(a); }
    for (int r : {-1, 0, good.cache.capacity + 1, std::numeric_limits<int>::max()}) { auto a = good; a.cache.valid_rows = r; reject(a); }
    { auto a = good; a.cache.valid_rows = a.first + a.batch - 2; reject(a); } // last query exceeds LOADED prefix only
    auto bad_view = [&](auto member, std::size_t n) {
        auto a = good; member(a).stride = n - 1; reject(a);
        a = good; member(a).stride = std::numeric_limits<std::size_t>::max(); reject(a);
        a = good; member(a).elements = 0; reject(a);
        a = good; member(a).elements = std::numeric_limits<std::size_t>::max(); reject(a);
        a = good; member(a).elements = std::size_t(a.batch - 1) * member(a).stride + n - 1; reject(a);
        a = good; member(a).elements = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()); reject(a);
    };
    const auto l = mtp_attention_q4_layout(good.cache.capacity);
    bad_view([](Args& a) -> auto& { return a.query; }, width);
    bad_view([](Args& a) -> auto& { return a.output; }, width);
    bad_view([](Args& a) -> auto& { return a.w.partial_output; }, l.partial_elements);
    bad_view([](Args& a) -> auto& { return a.w.partial_max_sum; }, l.max_sum_elements);
    bad_view([](Args& a) -> auto& { return a.w.staged_output; }, width);
    { auto a = good; ++a.query.stride; reject(a); }
    { auto a = good; a.cache.key_elements = l.cache_elements - 1; reject(a); }
    { auto a = good; a.cache.value_elements = l.cache_elements - 1; reject(a); }
    { auto a = good; a.cache.key_elements = std::numeric_limits<std::size_t>::max(); reject(a); }
    { auto a = good; a.cache.value_elements = std::numeric_limits<std::size_t>::max(); reject(a); }
    // Dynamic layout MUST be actual capacity even with a very small visibility.
    { auto a = good; a.cache.capacity = 131072; a.cache.key_elements = 131072 * 16; a.cache.value_elements = 131072 * 16; reject(a); }
    for (int w = 3; w < 8; ++w) for (int o = 0; o < 8; ++o) if (w != o) {
        auto a = good; a.pointer(w, a.pointer(o)); reject(a);
    }
    // Numeric offsets also keep synthetic CPU addresses free of pointer
    // arithmetic outside actual C++ objects. These addresses are never read.
    auto at = [](const void* p, std::size_t bytes) { return reinterpret_cast<int*>(reinterpret_cast<std::uintptr_t>(p) + bytes); };
    { auto a = good; a.w.error = at(a.output.data, (std::size_t(a.batch - 1) * a.output.stride + width) * sizeof(float)); reject(a); }
    { auto a = good; a.w.error = at(a.query.data, width * sizeof(float)); reject(a); }
    { auto a = good; a.w.error = at(a.w.partial_output.data, l.partial_elements * sizeof(float)); reject(a); }
    { auto a = good; a.w.error = at(a.w.partial_max_sum.data, l.max_sum_elements * sizeof(float2)); reject(a); }
    { auto a = good; a.w.error = at(a.w.staged_output.data, width * sizeof(float)); reject(a); }
    { auto a = good; a.w.error = at(a.cache.keys, l.cache_elements * sizeof(Q4_0)); reject(a); }
    { auto a = good; a.w.error = at(a.cache.values, l.cache_elements * sizeof(Q4_0)); reject(a); }
    { auto a = good; a.stream = nullptr; reject(a); }
}

struct Summary {
    int device = -1;
    std::size_t cases = 0, queries = 0, finite_elements = 0, bit_elements = 0;
    std::size_t cpu_full_elements = 0, cpu_constant_elements = 0, cpu_full_128k_cases = 0, cpu_full_128k_builds = 0;
    std::size_t device_rejects = 0, host_rejects = 0, sticky_checks = 0, peak_workspace_bytes = 0;
    double max_abs = 0, max_ratio = 0;
    void gate(std::span<const float> got, std::span<const float> ref, bool constant) {
        require(got.size() == width && ref.size() == width, "oracle shape");
        for (std::size_t i = 0; i < width; ++i) {
            require(std::isfinite(got[i]) && std::isfinite(ref[i]), "nonfinite full output/oracle");
            const double error = std::abs(double(got[i]) - double(ref[i]));
            const double bound = absolute_gate + relative_gate * std::abs(double(ref[i]));
            max_abs = std::max(max_abs, error); max_ratio = std::max(max_ratio, error / bound);
            require(error <= bound, "frozen gathered-F16 attention gate");
            ++finite_elements;
            if (constant) ++cpu_constant_elements; else ++cpu_full_elements;
        }
    }
};

#ifndef MTP_ATTENTION_CPU_ONLY
std::size_t live_resources = 0;
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void cleanup(hipError_t e) noexcept { if (e != hipSuccess) std::abort(); }
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); ++live_resources; }
    ~Stream() { cleanup(hipStreamSynchronize(value)); cleanup(hipStreamDestroy(value)); --live_resources; }
    Stream(const Stream&) = delete; Stream& operator=(const Stream&) = delete;
};
template<class T> struct Buffer {
    void* base = nullptr;
    T* data = nullptr;
    std::size_t elements;
    hipStream_t stream;
    Buffer(std::size_t n, hipStream_t s) : elements(n), stream(s) {
        check(hipMalloc(&base, bytes() + 2 * guard_bytes)); ++live_resources;
        data = reinterpret_cast<T*>(static_cast<std::byte*>(base) + guard_bytes);
        try { check(hipMemsetAsync(base, 0xa5, bytes() + 2 * guard_bytes, stream)); check(hipStreamSynchronize(stream)); }
        catch (...) { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(base)); --live_resources; throw; }
    }
    ~Buffer() { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(base)); --live_resources; }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    std::size_t bytes() const { return elements * sizeof(T); }
    void put(std::span<const T> h) {
        require(h.size() == elements, "upload shape");
        check(hipMemcpyAsync(data, h.data(), bytes(), hipMemcpyHostToDevice, stream)); check(hipStreamSynchronize(stream));
    }
    std::vector<T> get() const {
        std::vector<T> h(elements);
        check(hipMemcpyAsync(h.data(), data, bytes(), hipMemcpyDeviceToHost, stream)); check(hipStreamSynchronize(stream)); return h;
    }
    void guards() const {
        std::array<std::byte, guard_bytes> a{}, b{};
        check(hipMemcpyAsync(a.data(), base, guard_bytes, hipMemcpyDeviceToHost, stream));
        check(hipMemcpyAsync(b.data(), static_cast<std::byte*>(base) + guard_bytes + bytes(), guard_bytes, hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        for (auto x : a) require(x == std::byte{0xa5}, "leading guard");
        for (auto x : b) require(x == std::byte{0xa5}, "trailing guard");
    }
    MtpAttentionQ4View<T> view(std::size_t stride) { return {data, elements, stride}; }
    MtpAttentionQ4View<const T> read(std::size_t stride) const { return {data, elements, stride}; }
};
template<class T> void unchanged(const Buffer<T>& b, const std::vector<T>& before) { exact<T>(b.get(), before, "readonly/rejected mutation"); }
template<class T> void gaps(const Buffer<T>& b, int batch, std::size_t stride, std::size_t n) {
    const auto h = b.get();
    for (int q = 0; q < batch; ++q) {
        const auto gap = std::as_bytes(std::span(h).subspan(std::size_t(q) * stride + n, stride - n));
        for (auto x : gap) require(x == std::byte{0xa5}, "row padding changed");
    }
    b.guards();
}
struct Capture {
    hipStream_t stream;
    hipGraph_t graph = nullptr;
    bool active = false;
    explicit Capture(hipStream_t s) : stream(s) { check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal)); active = true; }
    ~Capture() {
        if (active) { cleanup(hipStreamEndCapture(stream, &graph)); if (graph) ++live_resources; }
        if (graph) { cleanup(hipGraphDestroy(graph)); --live_resources; }
    }
    void finish() {
        const auto status = hipStreamEndCapture(stream, &graph); active = false;
        if (graph) ++live_resources;
        check(status);
        std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes)); require(nodes == 0, "host rejection enqueued GPU work");
    }
};
struct Fixture {
    int batch, first, capacity, valid_rows;
    MtpAttentionQ4Layout layout;
    std::size_t qs = 6160, ps, ms;
    bool constant;
    Stream stream;
    Buffer<float> query, output, partial, staged;
    Buffer<Q4_0> keys, values;
    Buffer<float2> ml;
    Buffer<int> error;
    std::vector<float> hq, ho;
    std::vector<Q4_0> hk, hv;
    Fixture(int b, int f, bool c = false, int cap = 0) : batch(b), first(f),
        capacity(cap ? cap : std::min(131072, f + b + 8)), valid_rows(capacity),
        layout(mtp_attention_q4_layout(capacity)), ps(layout.partial_elements + 8), ms(layout.max_sum_elements + 4), constant(c),
        query(std::size_t(b) * qs, stream.value), output(std::size_t(b) * qs, stream.value),
        partial(std::size_t(b) * ps, stream.value), staged(std::size_t(b) * qs, stream.value),
        keys(layout.cache_elements + 16, stream.value), values(layout.cache_elements + 16, stream.value),
        ml(std::size_t(b) * ms, stream.value), error(1, stream.value), hq(query_data(b, qs)),
        ho(poison<float>(output.elements)), hk(keys.elements), hv(values.elements) {
        for (int q = 0; q < batch; ++q) std::fill_n(ho.data() + std::size_t(q) * qs, width, sentinel);
        cache_data(hk, hv, first + batch - 1, constant);
    }
    MtpAttentionQ4Cache cache() const { return {keys.data, keys.elements, values.data, values.elements, capacity, valid_rows}; }
    MtpAttentionQ4Workspace workspace() { return {partial.view(ps), ml.view(ms), staged.view(qs), error.data}; }
    Args args() { return {query.read(qs), cache(), batch, first, output.view(qs), workspace(), stream.value}; }
    hipError_t call() { const auto a = args(); return launch_mtp_attention_q4(a.query, a.cache, a.batch, a.first, a.output, a.w, a.stream); }
    void flag(int bits) { const std::array<int, 1> h{bits}; error.put(h); }
    int flag() { return error.get()[0]; }
    void upload() { query.put(hq); output.put(ho); keys.put(hk); values.put(hv); }
    void poisoned_scratch() {
        auto p = poison<float>(partial.elements); auto m = poison<float2>(ml.elements);
        for (int q = 0; q < batch; ++q) {
            std::fill_n(p.data() + std::size_t(q) * ps, layout.partial_elements, std::numeric_limits<float>::quiet_NaN());
            std::fill_n(m.data() + std::size_t(q) * ms, layout.max_sum_elements,
                        make_float2(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
        }
        partial.put(p); ml.put(m);
    }
    void sources() { unchanged(query, hq); unchanged(keys, hk); unchanged(values, hv); query.guards(); keys.guards(); values.guards(); error.guards(); }
    void neighbors() {
        gaps(output, batch, qs, width); gaps(staged, batch, qs, width);
        gaps(partial, batch, ps, layout.partial_elements); gaps(ml, batch, ms, layout.max_sum_elements);
    }
    void inactive() {
        const auto p = partial.get(); const auto m = ml.get();
        for (int q = 0; q < batch; ++q) {
            const auto chunks = std::size_t((first + q + 63) / 64);
            for (std::size_t i = chunks * width; i < layout.partial_elements; ++i)
                require(std::isnan(p[std::size_t(q) * ps + i]), "inactive partial written/read");
            for (std::size_t i = chunks * 24; i < layout.max_sum_elements; ++i)
                require(std::isnan(m[std::size_t(q) * ms + i].x) && std::isnan(m[std::size_t(q) * ms + i].y), "inactive ml written/read");
        }
    }
};
void parity(Fixture& f, Summary& s) {
    f.flag(0); f.upload(); f.poisoned_scratch(); check(f.call()); check(hipStreamSynchronize(f.stream.value));
    require(f.flag() == 0, "valid dense flag"); f.inactive();
    const auto got = f.output.get();
    // Small cases use the EXACT existing N1 API with COMPLETE contiguous IDs.
    // Separately owned old gather buffers, bounded at5MiB, never a dense gather.
    if (f.first <= 2051) {
        Buffer<std::int32_t> ids(2051, f.stream.value); Buffer<int> counts(2, f.stream.value);
        Buffer<std::uint16_t> gk(attention_gather_elements, f.stream.value), gv(attention_gather_elements, f.stream.value);
        Buffer<float> p(attention_partial_elements, f.stream.value); Buffer<float2> m(attention_partial_ml_elements, f.stream.value);
        Buffer<float> stage(width, f.stream.value);
        std::vector<std::int32_t> hi(2051); for (int i = 0; i < 2051; ++i) hi[std::size_t(i)] = i;
        ids.put(hi); f.output.put(f.ho);
        for (int q = 0; q < f.batch && f.first + q <= 2051; ++q) {
            const int visible = f.first + q; const std::array<int, 2> hc{visible, visible / 4}; counts.put(hc);
            const AttentionQ4DeviceWorkspace w{gk.data, gv.data, p.data, m.data, stage.data, f.error.data};
            check(launch_attention_q4(f.query.data + std::size_t(q) * f.qs, f.keys.data, f.values.data, f.capacity,
                ids.data, counts.data, visible, f.output.data + std::size_t(q) * f.qs, w, f.stream.value));
        }
        check(hipStreamSynchronize(f.stream.value)); require(f.flag() == 0, "old complete N1 flag");
        const auto old = f.output.get();
        for (int q = 0; q < f.batch; ++q) {
            const auto prior = std::span<const float>(old).subspan(std::size_t(q) * f.qs, width);
            if (f.first + q <= 2051) {
                exact<float>(std::span<const float>(got).subspan(std::size_t(q) * f.qs, width), prior,
                             "dense versus complete contiguous old N1 bit mismatch");
                s.bit_elements += width;
            } else exact<float>(prior, std::span<const float>(f.ho).subspan(std::size_t(q) * f.qs, width), "old N1 touched unlaunched query");
        }
        unchanged(ids, hi); ids.guards(); counts.guards(); gk.guards(); gv.guards(); p.guards(); m.guards(); stage.guards();
    }
    for (int q = 0; q < f.batch; ++q) {
        const auto hq = std::span<const float>(f.hq).subspan(std::size_t(q) * f.qs, width);
        std::vector<float> ref;
        if (f.constant) ref = constant_oracle(f.hv);
        else if (f.first + q == 131072) {
            // The sole nonconstant 128K B1 fixture has identical source bytes
            // on both devices: build its ALL-head oracle ONCE, reuse the full
            // 6144-value reference, not sampled heads/dims, on the second GPU.
            static std::vector<float> reference;
            if (reference.empty()) { reference = dense_oracle(f.hk, f.hv, hq, 131072); ++s.cpu_full_128k_builds; }
            ref = reference;
        } else ref = dense_oracle(f.hk, f.hv, hq, f.first + q);
        s.gate(std::span<const float>(got).subspan(std::size_t(q) * f.qs, width), ref, f.constant);
        if (!f.constant && f.first + q == 131072) ++s.cpu_full_128k_cases;
        ++s.queries;
    }
    f.sources(); f.neighbors(); ++s.cases;
    s.peak_workspace_bytes = std::max(s.peak_workspace_bytes,
        f.partial.bytes() + f.ml.bytes() + f.staged.bytes() + f.error.bytes() + 8 * guard_bytes);
    require(s.peak_workspace_bytes < 200 * 1024 * 1024, "dense scratch budget exceeds200MiB");
}
void failed(Fixture& f, Summary& s, int expected, bool preflight) {
    f.flag(0); f.upload();
    const auto p = f.partial.get(), o = f.staged.get(); const auto m = f.ml.get();
    check(f.call()); check(hipStreamSynchronize(f.stream.value)); require(f.flag() == expected, "dense error class");
    unchanged(f.output, f.ho);
    if (preflight) { unchanged(f.partial, p); unchanged(f.ml, m); unchanged(f.staged, o); }
    f.sources(); f.neighbors(); ++s.device_rejects;
}
void device_failures(Summary& s) {
    Fixture f(3, 65); const auto q = f.hq; const auto k = f.hk, v = f.hv;
    const auto last_row = std::size_t(f.first + f.batch - 2), late = last_row * 16 + 15;
    for (float bad : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        f.hq[std::size_t(f.batch - 1) * f.qs + width - 1] = bad; failed(f, s, 1, true); f.hq = q;
    }
    for (auto bits : {std::uint16_t{0x7c00}, std::uint16_t{0x7e01}, std::uint16_t{0xfc00}}) {
        f.hk[late].d = bits; failed(f, s, 1, true); f.hk = k;
        f.hv[late].d = bits; failed(f, s, 1, true); f.hv = v;
    }
    for (bool value : {false, true}) for (auto bits : {std::uint16_t{0x7bff}, std::uint16_t{0xfbff}}) {
        auto& block = value ? f.hv[late] : f.hk[late]; block.d = bits;
        std::fill(std::begin(block.qs), std::end(block.qs), std::uint8_t{0}); // -8*65504 = -524032
        failed(f, s, 2, true); f.hk = k; f.hv = v;
    }
    std::fill_n(f.hq.data() + std::size_t(f.batch - 1) * f.qs, width, std::numeric_limits<float>::max());
    for (int i = 0; i < (f.first + f.batch - 1) * 16; ++i) {
        f.hk[std::size_t(i)].d = 0x3c00;
        std::fill(std::begin(f.hk[std::size_t(i)].qs), std::end(f.hk[std::size_t(i)].qs), std::uint8_t{0xff});
    }
    failed(f, s, 2, false); f.hq = q; f.hk = k;
    // Sticky REUSE: preserve the actual failure flag without resetting it.
    f.upload();
    auto sticky = [&](int bits) {
        const auto p = f.partial.get(), o = f.staged.get(); const auto m = f.ml.get();
        check(f.call()); check(hipStreamSynchronize(f.stream.value)); require(f.flag() == bits, "sticky flag cleared");
        unchanged(f.output, f.ho); unchanged(f.partial, p); unchanged(f.ml, m); unchanged(f.staged, o); ++s.sticky_checks;
    };
    sticky(2);
    for (int bits : {1, 2, 3, 8}) { f.flag(bits); sticky(bits); }
    parity(f, s); // explicit flag reset inside parity, same formerly provisional scratch

    // Each earlier query must ignore newly loaded future rows: singleton dense
    // replay uses the SAME valid_rows, with ALL positions >=its visibility poison.
    for (int earlier = 0; earlier < 2; ++earlier) {
        Fixture future(1, 65 + earlier, false, 80); parity(future, s);
    }
    // Near-end invalid source at >2051 verifies FULL prefix validation, including
    // rows a QSA selector would omit. Late query failure suppresses ALL B output.
    Fixture long_bad(3, 4094); const auto lk = long_bad.hk, lv = long_bad.hv;
    const auto end = std::size_t(4095) * 16 + 15;
    long_bad.hk[end].d = 0x7e01; failed(long_bad, s, 1, true); long_bad.hk = lk;
    long_bad.hv[end].d = 0x7bff;
    std::fill(std::begin(long_bad.hv[end].qs), std::end(long_bad.hv[end].qs), std::uint8_t{0});
    failed(long_bad, s, 2, true); long_bad.hv = lv;
    Fixture max_bad(3, 131070, true);
    const auto max_k = max_bad.hk, max_v = max_bad.hv;
    const auto max_end = std::size_t(131071) * 16 + 15;
    max_bad.hk[max_end].d = 0x7e01; failed(max_bad, s, 1, true); max_bad.hk = max_k;
    max_bad.hv[max_end].d = 0x7bff;
    std::fill(std::begin(max_bad.hv[max_end].qs), std::end(max_bad.hv[max_end].qs), std::uint8_t{0});
    failed(max_bad, s, 2, true); max_bad.hv = max_v;
}
void host_failures(Summary& s) {
    Fixture f(3, 65); f.upload(); f.flag(8); const auto good = f.args();
    const auto p = f.partial.get(), o = f.staged.get(); const auto m = f.ml.get();
    {
        Capture capture(f.stream.value);
        bad_metadata(good, [&](const Args& a) {
            require(launch_mtp_attention_q4(a.query, a.cache, a.batch, a.first, a.output, a.w, a.stream) == hipErrorInvalidValue, "bad host metadata accepted");
            ++s.host_rejects;
        });
        capture.finish();
    }
    require(f.flag() == 8, "host rejection flag mutation"); unchanged(f.output, f.ho);
    unchanged(f.partial, p); unchanged(f.ml, m); unchanged(f.staged, o); f.sources(); f.neighbors();
}
Summary device(int index) {
    check(hipSetDevice(index)); hipDeviceProp_t prop{}; check(hipGetDeviceProperties(&prop, index));
    require(std::string_view(prop.gcnArchName).starts_with("gfx906") && prop.warpSize == 64, "requires gfx906 wave64");
    Summary s; s.device = index;
    // B1/B2/B3 EACH cover all requested boundaries; window end is capped at128K.
    std::vector<int> lengths{1, 2, 3, 63, 64, 65};
    for (int n = 2047; n <= 2056; ++n) lengths.push_back(n);
    lengths.insert(lengths.end(), {4096, 16384, 131072});
    for (int b : {1, 2, 3}) for (int n : lengths) {
        const int first = n == 131072 ? n - b + 1 : n;
        Fixture f(b, first, n == 131072 && b > 1); parity(f, s);
    }
    // Capacity rather than visibility determines scratch widths: huge inactive
    // region stays poisoned/unread; full loaded future prefix is also poisoned.
    { Fixture f(3, 1, false, 131072); parity(f, s); }
    // Strong dense-vs-QSA and per-query visibility witness. Only position4095
    // contributes: first two queries must output ZERO in every dim, third sees
    // it and returns448/4096. Any2051 clipping or future leakage fails the gate.
    { Fixture f(3, 4094); f.valid_rows = 4096;
      for (int q = 0; q < f.batch; ++q) std::fill_n(f.hq.data() + std::size_t(q) * f.qs, width, 0.0f);
      for (int b = 0; b < 4096 * 16; ++b) { f.hk[std::size_t(b)] = {}; f.hv[std::size_t(b)] = {}; }
      for (int b = 4095 * 16; b < 4096 * 16; ++b) {
          f.hv[std::size_t(b)].d = 0x5400;
          std::fill(std::begin(f.hv[std::size_t(b)].qs), std::end(f.hv[std::size_t(b)].qs), std::uint8_t{0xff});
      }
      parity(f, s); }
    // Valid tiny half scales, signed scale and signed zero stay valid.
    { Fixture f(3, 63);
      for (int b = 0; b < (f.first + f.batch - 1) * 16; ++b) {
          f.hk[std::size_t(b)].d = std::uint16_t((b % 2 ? 0x8000 : 0) | 1);
          f.hv[std::size_t(b)].d = std::uint16_t((b % 2 ? 0x8000 : 0) | (b % 3 ? 1 : 0));
      }
      parity(f, s); }
    device_failures(s); host_failures(s);
    require(live_resources == 0, "resources not released before GPU serialization");
    return s;
}
#else
// Genuine CPU-only checks of the SAME pure metadata preflight. Synthetic
// addresses are NEVER dereferenced and no HIP API or fake GPU kernel is called.
void cpu_checks() {
    std::size_t rejects = 0;
    const auto l = mtp_attention_q4_layout(80);
    auto p = [](std::uintptr_t n) { return reinterpret_cast<void*>(n); };
    Args a{{static_cast<const float*>(p(0x10000000)), 3 * 6160, 6160},
           {static_cast<const Q4_0*>(p(0x20000000)), 80 * 16 + 16, static_cast<const Q4_0*>(p(0x30000000)), 80 * 16 + 16, 80, 80},
           3, 65, {static_cast<float*>(p(0x40000000)), 3 * 6160, 6160},
           {{static_cast<float*>(p(0x50000000)), 3 * (l.partial_elements + 8), l.partial_elements + 8},
            {static_cast<float2*>(p(0x60000000)), 3 * (l.max_sum_elements + 4), l.max_sum_elements + 4},
            {static_cast<float*>(p(0x70000000)), 3 * 6160, 6160}, static_cast<int*>(p(0x80000000))},
           reinterpret_cast<hipStream_t>(p(0x90000000))};
    auto valid = [](const Args& x) { return x.stream && mtp_attention_detail::valid_arguments(x.query, x.cache, x.batch, x.first, x.output, x.w); };
    require(valid(a), "CPU valid metadata");
    bad_metadata(a, [&](const Args& bad) { require(!valid(bad), "CPU invalid metadata accepted"); ++rejects; });
    for (int b : {1, 2, 3}) { auto x = a; x.batch = b; require(valid(x), "CPU B1..3 metadata"); }
    { auto x = a; x.cache.valid_rows = x.first + x.batch - 1; require(valid(x), "CPU loaded prefix below capacity"); }
    { auto x = a; x.cache.values = x.cache.keys; require(valid(x), "CPU readonly alias allowed"); }
    std::size_t layouts = 0;
    for (int cap = 1; cap <= 131072; ++cap) {
        const auto x = mtp_attention_q4_layout(cap);
        require(x.chunks == std::size_t((cap + 63) / 64) && x.cache_elements == std::size_t(cap) * 16 &&
                x.partial_elements == x.chunks * width && x.max_sum_elements == x.chunks * 24, "CPU dynamic layout"); ++layouts;
    }
    for (int cap : {-1, 0, 131073, std::numeric_limits<int>::max()}) require(mtp_attention_q4_layout(cap).chunks == 0, "CPU invalid layout");
    std::vector<Q4_0> k(80 * 16), v(80 * 16); cache_data(k, v, 65);
    const auto q = query_data(1, width); const auto dense = dense_oracle(k, v, q, 65);
    std::vector<std::int32_t> ids(65); for (int i = 0; i < 65; ++i) ids[std::size_t(i)] = i;
    std::vector<float> scores(65), stage(width), reference(width);
    attention_q4({k, v, 80}, q, ids, 65, AttentionQ4Mode::gathered_fp16, {scores, stage}, reference);
    exact<float>(dense, reference, "CPU dense oracle versus existing gathered-F16 must be bitexact");
    // FULL 128K CPU loop/all heads, independent constant-cache analytical gate.
    k.resize(131072 * 16); v.resize(131072 * 16); cache_data(k, v, 131072, true);
    // Dyadic constant values keep this independent CPU mathematical check exact
    // across131072 serial additions; do not loosen the gate for accumulation.
    for (std::size_t b = 0; b < k.size(); ++b) {
        k[b].d = std::uint16_t(0x1c00 | (b % 2 ? 0x8000 : 0));
        v[b].d = std::uint16_t(0x1c00 | ((b % 16) % 3 ? 0x8000 : 0));
    }
    Summary s; s.gate(dense_oracle(k, v, q, 131072), constant_oracle(v), true);
    std::size_t source_rejects = 0;
    auto reject = [&] { bool threw = false; try { (void)dense_oracle(k, v, q, 131072); }
                       catch (const std::runtime_error&) { threw = true; } require(threw, "CPU late source accepted"); ++source_rejects; };
    const auto last = v.back(); v.back().d = 0x7e01; reject(); v.back() = last;
    v.back().d = 0x7bff; std::fill(std::begin(v.back().qs), std::end(v.back().qs), std::uint8_t{0}); reject(); v.back() = last;
    Q4_0 tiny{}; tiny.d = 0x8001;
    require(std::isfinite(decoded(tiny, 31)), "CPU signed tiny scale");
    std::cout << "{\"kind\":\"mtp_attention_cpu_only\",\"hip_qualified\":false,\"revision\":\"" << CORE_REVISION
              << "\",\"dirty\":" << (CORE_DIRTY ? "true" : "false") << ",\"metadata_rejects\":" << rejects
              << ",\"layouts\":" << layouts << ",\"short_bit_elements\":6144,\"full128k_allhead_elements\":" << s.cpu_constant_elements
              << ",\"late_source_rejects\":" << source_rejects << ",\"max_bound_ratio\":" << s.max_ratio << ",\"passed\":true}\n";
}
#endif
void json_string(std::string_view text) {
    std::cout << '"';
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32) { constexpr char hex[] = "0123456789abcdef"; std::cout << "\\u00" << hex[c >> 4] << hex[c & 15]; }
        else std::cout << char(c);
    }
    std::cout << '"';
}
} // namespace
int main() {
    try {
        provenance(); std::cout << std::setprecision(17);
#ifdef MTP_ATTENTION_CPU_ONLY
        cpu_checks();
#else
        int count = 0; check(hipGetDeviceCount(&count)); require(count == 2, "requires exactly both target GPUs");
        const std::array<Summary, 2> results{device(0), device(1)};
        require(live_resources == 0, "cleanup incomplete");
        // Bounded FOUR-record JSONL, only AFTER both devices/all cleanup pass.
        std::cout << "{\"kind\":\"mtp_attention_source\",\"protocol\":1,\"revision\":"; json_string(CORE_REVISION);
        std::cout << ",\"dirty\":" << (CORE_DIRTY ? "true" : "false")
                  << ",\"domain\":\"dense_full_causal_prefix\",\"model_free\":true,\"absolute_gate\":0.0002,\"relative_gate\":0.0002"
                     ",\"max_workspace_bytes_per_query\":50749440,\"max_B3_workspace_bytes_with_flag\":152248324,\"performance_claim\":false}\n";
        for (const auto& s : results)
            std::cout << "{\"kind\":\"mtp_attention_correctness\",\"protocol\":1,\"device\":" << s.device
                      << ",\"cases\":" << s.cases << ",\"queries\":" << s.queries << ",\"finite_elements\":" << s.finite_elements
                      << ",\"bit_elements\":" << s.bit_elements << ",\"cpu_full_elements\":" << s.cpu_full_elements
                      << ",\"cpu_constant_elements\":" << s.cpu_constant_elements << ",\"cpu_full_128k_cases\":" << s.cpu_full_128k_cases
                      << ",\"cpu_full_128k_oracle_builds\":" << s.cpu_full_128k_builds
                      << ",\"device_rejects\":" << s.device_rejects << ",\"host_rejects\":" << s.host_rejects
                      << ",\"sticky_checks\":" << s.sticky_checks << ",\"peak_private_workspace_bytes\":" << s.peak_workspace_bytes
                      << ",\"max_abs\":" << s.max_abs << ",\"max_bound_ratio\":" << s.max_ratio << ",\"passed\":true}\n";
        std::cout << "{\"kind\":\"mtp_attention_footer\",\"protocol\":1,\"devices\":2,\"passed\":true,\"cleanup_complete\":true,\"live_resources\":0}\n";
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cout << "{\"kind\":\"mtp_attention_failure\",\"passed\":false,\"hip_qualified\":false,\"error\":";
        json_string(std::string_view(e.what()).substr(0, 512));
#ifndef MTP_ATTENTION_CPU_ONLY
        std::cout << ",\"cleanup_complete\":" << (live_resources == 0 ? "true" : "false");
#endif
        std::cout << "}\n"; return 1;
    }
}
