#include "qsa.hpp"
#include "hip/qsa_select.cuh"
#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {
void need(bool x, const char* why) { if (!x) throw std::runtime_error(why); }
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
template<class T> struct Buffer {
    T* raw = nullptr;
    std::size_t n;
    explicit Buffer(std::size_t count) : n(count) {
        check(hipMalloc(reinterpret_cast<void**>(&raw), (n + 2) * sizeof(T)));
        fill();
    }
    ~Buffer() { if (raw && hipFree(raw) != hipSuccess) std::abort(); }
    Buffer(const Buffer&) = delete;
    T* data() const { return raw + 1; }
    qwen::QsaSelectView<T> view(std::size_t stride) const { return {data(), n, stride}; }
    void fill(T value = T{73}) {
        std::vector<T> x(n + 2, T{73});
        std::fill(x.begin() + 1, x.end() - 1, value);
        check(hipMemcpy(raw, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    std::vector<T> read() const {
        std::vector<T> x(n + 2);
        check(hipMemcpy(x.data(), raw, x.size() * sizeof(T), hipMemcpyDeviceToHost));
        need(x.front() == T{73} && x.back() == T{73}, "outer guard");
        return {x.begin() + 1, x.end() - 1};
    }
    void write(const std::vector<T>& x) {
        need(x.size() == n, "upload size");
        check(hipMemcpy(data(), x.data(), n * sizeof(T), hipMemcpyHostToDevice));
    }
};
constexpr int maximum = 8;
constexpr std::size_t ss = 32768 + 7, ts = 2051 + 5, bs = 512 + 3;
constexpr std::size_t cs = 5, hs = 8192 + 9, ws = 16 + 3, ks = 512 + 5;
constexpr int sentinel = -31337;
struct Fixture {
    hipStream_t stream{};
    Buffer<float> scores{maximum * ss};
    Buffer<int> tokens{maximum * ts}, blocks{maximum * bs}, counts{maximum * cs};
    Buffer<int> old_tokens{maximum * ts}, old_blocks{maximum * bs}, old_counts{maximum * cs};
    Buffer<int> histogram{maximum * hs}, error{1};
    Buffer<std::uint64_t> state{maximum * ws}, candidates{maximum * ks};
    std::vector<float> input;
    std::vector<int> expected_t, expected_b, expected_c;
    int queries = 0, first = 0;
    std::uint64_t compared = 0;
    Fixture() { check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); }
    ~Fixture() { if (hipStreamDestroy(stream) != hipSuccess) std::abort(); }
    qwen::QsaSelectBatchWorkspace workspace() const {
        return {histogram.view(hs), state.view(ws), candidates.view(ks), error.data()};
    }
    qwen::QsaSelectView<const float> score_view() const { return {scores.data(), scores.n, ss}; }
    void reset_outputs() {
        tokens.fill(sentinel); blocks.fill(sentinel); counts.fill(sentinel);
        old_tokens.fill(sentinel); old_blocks.fill(sentinel); old_counts.fill(sentinel);
        error.fill(0);
    }
    void prepare(int n, int visible, int pattern) {
        queries = n; first = visible;
        input.assign(scores.n, std::numeric_limits<float>::infinity());
        expected_t.assign(tokens.n, sentinel); expected_b.assign(blocks.n, sentinel);
        expected_c.assign(counts.n, sentinel);
        std::uint32_t rng = 0x1234567;
        for (int q = 0; q < n; ++q) {
            const int complete = (visible + q) / 4;
            for (int i = 0; i < complete; ++i) {
                rng = rng * 1664525u + 1013904223u;
                float value = 0;
                if (pattern == 0) value = float(int(rng % 100003) - 50001) / 8192.0f;
                if (pattern == 1) value = float((i + q) % 7 - 3);
                if (pattern == 2) value = std::bit_cast<float>(i % 2 ? 0x80000000u : 0u);
                if (pattern == 3) value = std::bit_cast<float>((i % 2 ? 0x80000000u : 0u) | (1u + rng % 0x7ffffeu));
                if (pattern == 4) value = std::bit_cast<float>((rng & 0x807fffffu) | (0x7eu << 24));
                input[std::size_t(q) * ss + i] = value;
            }
            const auto count = qwen::qsa_select(
                std::span<const float>(input.data() + std::size_t(q) * ss, complete), visible + q,
                std::span<int>(expected_t.data() + std::size_t(q) * ts, 2051),
                std::span<int>(expected_b.data() + std::size_t(q) * bs, 512));
            expected_c[std::size_t(q) * cs] = static_cast<int>(count.token_count);
            expected_c[std::size_t(q) * cs + 1] = static_cast<int>(count.block_count);
        }
        scores.write(input); reset_outputs();
        histogram.fill(); state.fill(); candidates.fill();
    }
    hipError_t batch() {
        return qwen::launch_qsa_select_batch(score_view(), queries, first,
            tokens.view(ts), blocks.view(bs), counts.view(cs), workspace(), stream);
    }
    void serial() {
        for (int q = 0; q < queries; ++q)
            check(qwen::launch_qsa_select(scores.data() + std::size_t(q) * ss, first + q,
                old_tokens.data() + std::size_t(q) * ts, old_blocks.data() + std::size_t(q) * bs,
                old_counts.data() + std::size_t(q) * cs,
                {histogram.data(), state.data(), candidates.data(), error.data()}, stream));
    }
    void verify(bool old = false) {
        check(hipStreamSynchronize(stream)); need(error.read()[0] == 0, "device flags");
        const auto t = (old ? old_tokens : tokens).read();
        const auto b = (old ? old_blocks : blocks).read();
        const auto c = (old ? old_counts : counts).read();
        need(t == expected_t && b == expected_b && c == expected_c, "CPU/serial exact IDs/counts/padding");
        compared += t.size() + b.size() + c.size();
        const auto x = scores.read();
        need(std::memcmp(x.data(), input.data(), x.size() * sizeof(float)) == 0, "scores readonly");
        auto pad = [](const auto& buffer, std::size_t stride, std::size_t width) {
            const auto x = buffer.read();
            for (int q = 0; q < maximum; ++q)
                for (std::size_t j = width; j < stride; ++j)
                    need(x[std::size_t(q) * stride + j] == 73, "workspace row padding");
        };
        pad(histogram, hs, 8192); pad(state, ws, 16); pad(candidates, ks, 512);
    }
    void unchanged() {
        check(hipStreamSynchronize(stream));
        for (const auto* b : {&tokens, &blocks, &counts}) {
            const auto x = b->read();
            need(std::all_of(x.begin(), x.end(), [](int v) { return v == sentinel; }),
                 "failure batch publication");
        }
    }
    void host_rejections() {
        prepare(8, 2048, 0);
        for (int scenario = 0; scenario < 23; ++scenario) {
            auto sv = score_view(); auto tv = tokens.view(ts); auto bv = blocks.view(bs);
            auto cv = counts.view(cs); auto w = workspace(); int n = 8, v = 2048;
            if (scenario == 0) n = 0;
            if (scenario == 1) n = 9;
            if (scenario == 2) v = 0;
            if (scenario == 3) v = 131066;
            if (scenario == 4) sv.data = nullptr;
            if (scenario == 5) sv.stride = 1;
            if (scenario == 6) sv.elements = 511;
            if (scenario == 7) sv.elements = std::numeric_limits<std::size_t>::max();
            if (scenario == 8) sv.stride = std::numeric_limits<std::size_t>::max();
            if (scenario == 9) tv.stride = 2050;
            if (scenario == 10) bv.elements = 512;
            if (scenario == 11) cv.stride = 1;
            if (scenario == 12) w.histogram.stride = 8191;
            if (scenario == 13) w.state.stride = 15;
            if (scenario == 14) w.candidates.stride = 511;
            if (scenario == 15) w.error = nullptr;
            if (scenario == 16) w.error = tokens.data();
            if (scenario == 17) bv.data = tokens.data();
            if (scenario == 18) sv.data = reinterpret_cast<const float*>(tokens.data());
            if (scenario == 19) w.state.data = candidates.data();
            if (scenario == 20) tv.data = reinterpret_cast<int*>(reinterpret_cast<std::uintptr_t>(tokens.data()) + 1);
            if (scenario == 21) w.histogram.elements = 0;
            if (scenario == 22) cv.elements = 0;
            check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
            const auto status = qwen::launch_qsa_select_batch(sv, n, v, tv, bv, cv, w, stream);
            hipGraph_t graph{}; check(hipStreamEndCapture(stream, &graph));
            std::size_t nodes = 0; check(hipGraphGetNodes(graph, nullptr, &nodes));
            check(hipGraphDestroy(graph));
            need(status == hipErrorInvalidValue && nodes == 0, "host rejection must enqueue nothing");
            unchanged();
        }
    }
    void failures() {
        for (int visible : {8, 2050, 16377, 131065})
            for (int row : {0, 7})
                for (std::uint32_t bits : {0x7fc00000u, 0x7f800000u, 0xff800000u}) {
                    prepare(8, visible, 0);
                    input[std::size_t(row) * ss + (visible + row) / 4 - 1] = std::bit_cast<float>(bits);
                    scores.write(input); check(batch()); unchanged(); need(error.read()[0] & 1, "nonfinite flag");
                }
        for (int visible : {1, 2050, 16377}) {
            prepare(8, visible, 1); error.fill(4); check(batch()); unchanged();
            need(error.read()[0] == 4, "sticky flag preserved");
        }
        prepare(8, 16377, 0); check(batch()); verify();
    }
    void graph_reuse() {
        for (int visible : {1, 2050, 4093, 131065}) {
            prepare(8, visible, 1);
            check(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
            check(batch());
            hipGraph_t graph{}; check(hipStreamEndCapture(stream, &graph));
            hipGraphExec_t exec{}; check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            for (int repeat = 0; repeat < 3; ++repeat) {
                reset_outputs(); check(hipGraphLaunch(exec, stream)); verify();
            }
            check(hipGraphExecDestroy(exec)); check(hipGraphDestroy(graph));
        }
    }
    double timing(bool old, int repeats) {
        hipEvent_t start{}, end{}; check(hipEventCreate(&start)); check(hipEventCreate(&end));
        double total = 0;
        for (int i = 0; i < repeats; ++i) {
            error.fill(0);
            check(hipEventRecord(start, stream));
            if (old) serial(); else check(batch());
            check(hipEventRecord(end, stream)); check(hipEventSynchronize(end));
            float ms = 0; check(hipEventElapsedTime(&ms, start, end)); total += ms;
            verify(old);
        }
        check(hipEventDestroy(start)); check(hipEventDestroy(end));
        return total / repeats;
    }
};
}
int main() {
    try {
        std::cout << std::setprecision(17);
        int devices = 0; check(hipGetDeviceCount(&devices)); need(devices == 2, "two GPUs required");
        for (int device = 0; device < devices; ++device) {
            check(hipSetDevice(device)); Fixture f; int cases = 0;
            for (int first : {1,2,3,4,5,7,2045,2046,2048,2050,2051,2052,2053,4093,4096,8191,16377,16384,32765,65529,131065})
                for (int n : {1,2,3,4,8})
                    for (int pattern = 0; pattern < 5; ++pattern) {
                        f.prepare(n, first, pattern); check(f.batch()); f.verify();
                        f.serial(); f.verify(true); ++cases;
                    }
            f.host_rejections(); f.failures(); f.graph_reuse();
            std::cout << "{\"kind\":\"qsa_batch_correctness\",\"device\":" << device
                << ",\"cases\":" << cases << ",\"compared_ids_counts_with_padding\":" << f.compared
                << ",\"host_rejects\":23,\"device_nonfinite_cases\":24,\"sticky_cases\":3,"
                   "\"graph_replays\":12,\"passed\":true}\n" << std::flush;
            for (int first : {2040,2050,4093,16377,32761,65529,131065})
                for (int n : {2,8}) {
                    f.prepare(n, first, 0); f.serial(); f.verify(true); check(f.batch()); f.verify();
                    const double a = f.timing(true, 20), b = f.timing(false, 20), z = f.timing(true, 20);
                    std::cout << "{\"kind\":\"qsa_batch_timing\",\"device\":" << device
                        << ",\"first_visible\":" << first << ",\"queries\":" << n
                        << ",\"serial_a_ms\":" << a << ",\"batch_ms\":" << b << ",\"serial_z_ms\":" << z
                        << ",\"ratio\":" << (a + z) / (2 * b)
                        << ",\"beats_both\":" << (b < std::min(a,z) ? "true" : "false")
                        << ",\"scope\":\"completed_GPU_events_including_launch_gaps;20_repeats_each;resident_scores;no_model_speed_claim\"}\n" << std::flush;
                }
        }
        std::cout << "{\"kind\":\"qsa_batch_complete\",\"revision\":\"" << CORE_REVISION
            << "\",\"dirty\":" << (CORE_DIRTY ? "true" : "false") << ",\"passed\":true}\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
