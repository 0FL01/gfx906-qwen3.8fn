#include "hip/session_ops.cuh"
#include "routes.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

// Parent build: C++20/gfx906, link qwen-session (or session_ops.hip object)
// and qwen-core for the real RouteGroups. Supply quoted 40-hex CORE_REVISION
// and numeric CORE_DIRTY. Run core-moe-route-copy on both real gfx906 devices.
// Component BYTE correctness only: NaN/half-zero patterns are opaque-copy
// probes, NOT numeric linear-validation inputs or inference/performance proof.
// All four bounded JSONL records are emitted only after both devices' cleanup.
namespace {
using qwen::MoeCopyView;
using qwen::MoeRouteIndex;
using qwen::Q8_1;
constexpr int tokens = 1024, ranks = 10, blocks = 80, width = 2560;
constexpr int columns = 129; // count<=128 plus an untouched neighbor column.
constexpr int entries = tokens * ranks;
constexpr std::size_t q8_row_bytes = blocks * sizeof(Q8_1);
constexpr std::size_t down_row_bytes = width * sizeof(float);
constexpr std::size_t contribution_bytes = std::size_t(entries) * down_row_bytes;
constexpr std::array<int, 14> counts{1,2,3,4,7,8,9,16,31,32,63,64,127,128};
constexpr std::byte poison{0xa5};
static_assert(sizeof(float) == 4 && sizeof(int) == 4 && std::endian::native == std::endian::little);
static_assert(q8_row_bytes == 2880 && down_row_bytes == 10240 && contribution_bytes == 104857600);
static_assert(counts.front() == 1 && counts.back() == qwen::session_ops_max_tokens);
static_assert(std::all_of(counts.begin(), counts.end(), [](int n) { return n >= 1 && n <= 128; }));
static_assert(qwen::moe_copy_max_input_tokens == tokens && qwen::moe_copy_ranks == ranks &&
    qwen::moe_copy_q8_blocks == blocks && qwen::moe_copy_width == width);
static_assert(sizeof(qwen::RouteAssignment) == 16 && sizeof(MoeRouteIndex) == 8);

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) { std::cerr << "MoE copy cleanup: " << hipGetErrorString(status) << '\n'; std::abort(); }
}
struct Owners {
    int buffers = 0, streams = 0;
    void empty() const { require(buffers == 0 && streams == 0, "live HIP owners after cleanup"); }
} owners;
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); ++owners.streams; }
    ~Stream() { cleanup(hipStreamSynchronize(value)); cleanup(hipStreamDestroy(value)); --owners.streams; }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct CopyCompletion {
    hipStream_t stream;
    bool finished = false;
    ~CopyCompletion() { if (!finished) cleanup(hipStreamSynchronize(stream)); }
    void wait() { check(hipStreamSynchronize(stream)); finished = true; }
};
struct Buffer {
    hipStream_t stream;
    std::size_t bytes, prefix;
    void* allocation = nullptr;
    std::vector<std::byte> expected;
    Buffer(hipStream_t s, std::size_t n, std::size_t guard)
        : stream(s), bytes(n), prefix(guard), expected(n + 2 * guard, poison) {
        require(n > 0 && n <= contribution_bytes && (guard == 64 || guard == 68), "bounded buffer geometry");
        check(hipMalloc(&allocation, expected.size())); ++owners.buffers;
        try { upload(); }
        catch (...) {
            cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation));
            allocation = nullptr; --owners.buffers; throw;
        }
    }
    ~Buffer() {
        if (allocation) { cleanup(hipStreamSynchronize(stream)); cleanup(hipFree(allocation)); --owners.buffers; }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    template<class T> T* as() const { return reinterpret_cast<T*>(static_cast<std::byte*>(allocation) + prefix); }
    template<class T> MoeCopyView<T> view() const {
        require(bytes % sizeof(T) == 0, "typed buffer extent"); return {as<T>(), bytes / sizeof(T)};
    }
    std::byte* cpu() { return expected.data() + prefix; }
    const std::byte* cpu() const { return expected.data() + prefix; }
    void copy(void* destination, const void* source, std::size_t n, hipMemcpyKind kind) const {
        CopyCompletion completion{stream};
        check(hipMemcpyAsync(destination, source, n, kind, stream)); completion.wait();
    }
    void upload() const { copy(allocation, expected.data(), expected.size(), hipMemcpyHostToDevice); }
    void fill() {
        std::fill(expected.begin(), expected.end(), poison); upload();
    }
    void verify() const {
        std::vector<std::byte> got(expected.size());
        copy(got.data(), allocation, got.size(), hipMemcpyDeviceToHost);
        require(got == expected, "exact byte oracle/untouched neighbor/read-only/canary failure");
    }
    void flag(int value) {
        require(bytes == sizeof(int), "flag buffer shape"); std::memcpy(cpu(), &value, sizeof(value)); upload();
    }
    void expect_flag(int value) { std::memcpy(cpu(), &value, sizeof(value)); }
};

// Frozen deterministic integer-only object representations, no production
// quantizer, half conversion, FP evaluation or copy kernel reused as an oracle.
void fill_original(Buffer& original) {
    constexpr std::array<std::uint16_t, 5> headers{0x0000,0x8000,0x0001,0x3c00,0x7e55};
    for (int token = 0; token < tokens; ++token) for (int block = 0; block < blocks; ++block) {
        auto* p = original.cpu() + (std::size_t(token) * blocks + block) * sizeof(Q8_1);
        const std::uint16_t d = headers[block % headers.size()];
        // 0x5948 is half(169): RAW sum, not scale-times-sum. Nonzero codes
        // with d=0 are intentionally retained, along with signed zero/NaNs.
        const std::uint16_t s = block % 5 == 0 ? 0x5948 : std::uint16_t(0x3100 + (token + block) % 511);
        std::memcpy(p, &d, 2); std::memcpy(p + 2, &s, 2);
        for (unsigned lane = 0; lane < 32; ++lane)
            p[4 + lane] = std::byte((unsigned(token) * 17 + unsigned(block) * 29 + lane * 13 + 1) & 255u);
    }
    original.upload();
}
void fill_down(Buffer& down) {
    constexpr std::array<std::uint32_t, 10> bits{
        0x00000000,0x80000000,0x3f800000,0xbf800000,0x00000001,
        0x7fc01234,0x7fa01234,0xffc05678,0x7f800000,0xff800000};
    for (int column = 0; column < columns; ++column) for (int lane = 0; lane < width; ++lane) {
        const auto value = lane % 11 == 10 ?
            (0x3f000000u | ((std::uint32_t(column) * 65537u + std::uint32_t(lane) * 31u) & 0x7fffffu)) :
            bits[(unsigned(column) + unsigned(lane)) % bits.size()];
        std::memcpy(down.cpu() + (std::size_t(column) * width + lane) * sizeof(float), &value, sizeof(value));
    }
    down.upload();
}
std::vector<MoeRouteIndex> make_routes() {
    std::vector<std::int32_t> ids(entries);
    std::vector<float> weights(entries, 0.125f);
    for (int token = 0; token < tokens; ++token) for (int rank = 0; rank < ranks; ++rank)
        ids[token * ranks + rank] = (token * 37 + rank * 53) % 512;
    qwen::RouteGroups groups(tokens); groups.prepare(ids, weights, tokens);
    std::vector<MoeRouteIndex> result;
    result.reserve(entries);
    std::array<bool, entries> seen{};
    for (const qwen::RouteAssignment& assignment : groups.assignments()) {
        const auto token = assignment.token, rank = assignment.rank;
        require(token >= 0 && token < tokens && rank >= 0 && rank < ranks, "upstream route index");
        require(!seen[token * ranks + rank], "upstream duplicate token/rank"); seen[token * ranks + rank] = true;
        result.push_back({std::int32_t(token), std::int32_t(rank)});
    }
    require(result.size() == entries && std::all_of(seen.begin(), seen.end(), [](bool v) { return v; }),
            "full 0..1023/token,0..9/rank coverage");
    return result;
}
void upload_routes(Buffer& metadata, std::span<const MoeRouteIndex> routes) {
    require(routes.size_bytes() == metadata.bytes, "whole sorted metadata upload");
    std::memcpy(metadata.cpu(), routes.data(), routes.size_bytes()); metadata.upload();
}
void oracle_gather(Buffer& gathered, const Buffer& original, std::span<const MoeRouteIndex> routes,
                   int actual_tokens, int offset, int count) {
    for (int column = 0; column < count; ++column) {
        const auto r = routes[offset + column];
        auto* destination = gathered.cpu() + std::size_t(column) * q8_row_bytes;
        if (r.token < 0 || r.token >= actual_tokens || r.rank < 0 || r.rank >= ranks)
            std::memset(destination, 0, q8_row_bytes);
        else std::memcpy(destination, original.cpu() + std::size_t(r.token) * q8_row_bytes, q8_row_bytes);
    }
}
void oracle_scatter(Buffer& contributions, const Buffer& down, std::span<const MoeRouteIndex> routes,
                    int actual_tokens, int offset, int count) {
    for (int column = 0; column < count; ++column) {
        const auto r = routes[offset + column];
        if (r.token < 0 || r.token >= actual_tokens || r.rank < 0 || r.rank >= ranks) continue;
        const auto row = std::size_t(r.token) * ranks + std::size_t(r.rank);
        std::memcpy(contributions.cpu() + row * down_row_bytes,
                    down.cpu() + std::size_t(column) * down_row_bytes, down_row_bytes);
    }
}
struct Counts {
    int gather_calls = 0, scatter_calls = 0, metadata_uploads = 0, full_pairs = 0;
    int partial_tiles = 0, completed_reuses = 0, invalid_calls = 0, sticky_reuses = 0, cleared_reuses = 0;
    int host_rejects = 0;
};
struct Fixture {
    Stream stream;
    Buffer original, gathered, down, contributions, metadata, flags;
    std::vector<MoeRouteIndex> routes;
    Counts& tally;
    Fixture(std::size_t guard, Counts& c)
        : original(stream.value, std::size_t(tokens) * q8_row_bytes, guard),
          gathered(stream.value, std::size_t(columns) * q8_row_bytes, guard),
          down(stream.value, std::size_t(columns) * down_row_bytes, guard),
          contributions(stream.value, contribution_bytes, guard),
          metadata(stream.value, std::size_t(entries) * sizeof(MoeRouteIndex), guard),
          flags(stream.value, sizeof(int), guard), routes(make_routes()), tally(c) {
        require(reinterpret_cast<std::uintptr_t>(original.as<void>()) % 16 == guard % 16 &&
            reinterpret_cast<std::uintptr_t>(gathered.as<void>()) % 16 == guard % 16 &&
            reinterpret_cast<std::uintptr_t>(down.as<void>()) % 16 == guard % 16 &&
            reinterpret_cast<std::uintptr_t>(contributions.as<void>()) % 16 == guard % 16, "copy alignment path");
        fill_original(original); fill_down(down); upload_routes(metadata, routes); ++tally.metadata_uploads;
        flags.flag(0);
    }
    hipError_t gather(int offset, int count, int actual_tokens = tokens, int valid_entries = entries) {
        return qwen::launch_moe_gather_q8(original.view<const Q8_1>(), actual_tokens,
            metadata.view<const MoeRouteIndex>(), valid_entries, offset, count, blocks,
            gathered.view<Q8_1>(), columns, flags.as<int>(), stream.value);
    }
    hipError_t scatter(int offset, int count, int actual_tokens = tokens, int valid_entries = entries) {
        return qwen::launch_moe_scatter(down.view<const float>(), actual_tokens,
            metadata.view<const MoeRouteIndex>(), valid_entries, offset, count, width, ranks,
            contributions.view<float>(), columns, flags.as<int>(), stream.value);
    }
    void copies(int offset, int count, int actual_tokens = tokens, int valid_entries = entries) {
        check(gather(offset, count, actual_tokens, valid_entries)); ++tally.gather_calls;
        check(scatter(offset, count, actual_tokens, valid_entries)); ++tally.scatter_calls;
        check(hipStreamSynchronize(stream.value));
        oracle_gather(gathered, original, routes, actual_tokens, offset, count);
        oracle_scatter(contributions, down, routes, actual_tokens, offset, count);
    }
    void readonly() { original.verify(); down.verify(); metadata.verify(); flags.verify(); }
    void all() { readonly(); gathered.verify(); contributions.verify(); }
    void valid() {
        // Full sorted metadata reused for every tile; all 10240 pairs actually
        // copied, with varying counts/offsets and an explicit partial last tile.
        int offset = 0;
        std::size_t step = 0;
        while (offset < entries) {
            const int requested = counts[step++ % counts.size()];
            const int count = std::min(requested, entries - offset);
            gathered.fill(); copies(offset, count); gathered.verify();
            flags.verify(); offset += count; tally.full_pairs += count;
            if (count < requested) ++tally.partial_tiles;
            if (offset < entries) ++tally.completed_reuses;
        }
        all();
        // Isolated tiles prove every untouched contribution and column remains
        // poisoned, including late-offset slices and both metadata endpoints.
        for (std::size_t j = 0; j < counts.size(); ++j) {
            const int count = counts[j];
            const int start = j % 3 == 0 ? 0 : j % 3 == 1 ? 37 : entries - count;
            gathered.fill(); contributions.fill(); copies(start, count); all(); ++tally.completed_reuses;
        }
        // Reuse the same uploaded metadata AFTER stream completion; a sticky
        // earlier arithmetic bit does not suppress valid payloads or acquire1.
        flags.flag(qwen::session_ops_error_arithmetic); gathered.fill(); contributions.fill();
        copies(entries - 3, 3); all(); ++tally.sticky_reuses; ++tally.completed_reuses;
        flags.flag(0); gathered.fill(); contributions.fill(); copies(5, 2); all(); ++tally.cleared_reuses;
    }
    void invalid() {
        // Actual count7 with much larger backing buffers: allocation size must
        // NEVER substitute for actual token visibility. Metadata prefix70 is
        // likewise smaller than its allocation; offset54,count16 is exact end.
        constexpr int actual_tokens = 7, valid_entries = 70, offset = 54, count = 16;
        constexpr std::array<MoeRouteIndex, count> mixed{{
            {0,0},{-1,0},{7,1},{6,9},{1,-1},{2,10},{3,3},{1024,2},
            {std::numeric_limits<std::int32_t>::max(),4},{4,std::numeric_limits<std::int32_t>::min()},
            {-9,-2},{5,8},{0,1},{1,2},{2,std::numeric_limits<std::int32_t>::max()},{6,0}}};
        std::copy(mixed.begin(), mixed.end(), routes.begin() + offset);
        upload_routes(metadata, routes); ++tally.metadata_uploads;
        // Each kernel independently establishes bit1; combined earlierbit2|1
        // and already-sticky3 are checked without clearing or neighbor loss.
        for (int prior : {0, qwen::session_ops_error_arithmetic, 3}) {
            flags.flag(prior); gathered.fill(); contributions.fill();
            check(gather(offset, count, actual_tokens, valid_entries)); ++tally.gather_calls; ++tally.invalid_calls;
            check(hipStreamSynchronize(stream.value));
            flags.expect_flag(prior | qwen::session_ops_error_invalid);
            oracle_gather(gathered, original, routes, actual_tokens, offset, count);
            gathered.verify(); contributions.verify(); readonly();
            // Reset flag only, so scatter must independently detect invalids.
            flags.flag(prior);
            check(scatter(offset, count, actual_tokens, valid_entries)); ++tally.scatter_calls; ++tally.invalid_calls;
            check(hipStreamSynchronize(stream.value));
            flags.expect_flag(prior | qwen::session_ops_error_invalid);
            oracle_scatter(contributions, down, routes, actual_tokens, offset, count); all();
        }
        routes = make_routes(); upload_routes(metadata, routes); ++tally.metadata_uploads;
        // A valid route after a previous invalid stage retains sticky3, then
        // caller-cleared reuse succeeds with0 and unchanged payload semantics.
        for (int prior : {3, 0}) {
            flags.flag(prior); gathered.fill(); contributions.fill(); copies(entries - 8, 8); all();
            if (prior) ++tally.sticky_reuses; else ++tally.cleared_reuses;
            ++tally.completed_reuses;
        }
    }
    void host_rejections();
};

void Fixture::host_rejections() {
    // Earlier bit2 detects an accidental OR1 as well as an accidental clear.
    flags.flag(qwen::session_ops_error_arithmetic); gathered.fill(); contributions.fill();
    struct GatherCall {
        MoeCopyView<const Q8_1> input;
        MoeCopyView<const MoeRouteIndex> routes;
        MoeCopyView<Q8_1> output;
        int actual_tokens = tokens, valid_entries = entries, offset = 3, count = 8, row_blocks = blocks, capacity = columns;
        int* error = nullptr;
    };
    struct ScatterCall {
        MoeCopyView<const float> input;
        MoeCopyView<const MoeRouteIndex> routes;
        MoeCopyView<float> output;
        int actual_tokens = tokens, valid_entries = entries, offset = 3, count = 8;
        int row_width = width, top_k = ranks, capacity = columns;
        int* error = nullptr;
    };
    const GatherCall g{original.view<const Q8_1>(), metadata.view<const MoeRouteIndex>(), gathered.view<Q8_1>(),
        tokens, entries, 3, 8, blocks, columns, flags.as<int>()};
    const ScatterCall s{down.view<const float>(), metadata.view<const MoeRouteIndex>(), contributions.view<float>(),
        tokens, entries, 3, 8, width, ranks, columns, flags.as<int>()};
    auto reject_gather = [&](const GatherCall& a) {
        require(qwen::launch_moe_gather_q8(a.input, a.actual_tokens, a.routes, a.valid_entries,
            a.offset, a.count, a.row_blocks, a.output, a.capacity, a.error, stream.value) == hipErrorInvalidValue,
            "host gather rejection"); ++tally.host_rejects;
    };
    auto reject_scatter = [&](const ScatterCall& a) {
        require(qwen::launch_moe_scatter(a.input, a.actual_tokens, a.routes, a.valid_entries,
            a.offset, a.count, a.row_width, a.top_k, a.output, a.capacity, a.error, stream.value) == hipErrorInvalidValue,
            "host scatter rejection"); ++tally.host_rejects;
    };
    auto both = [&](auto change) { auto a = g; auto b = s; change(a); change(b); reject_gather(a); reject_scatter(b); };
    for (int v : {0, -1, 1025, std::numeric_limits<int>::max()}) both([&](auto& a) { a.actual_tokens = v; });
    for (int v : {0, -1, entries + 1, std::numeric_limits<int>::max()}) both([&](auto& a) { a.valid_entries = v; });
    both([](auto& a) { a.actual_tokens = 7; a.valid_entries = 71; });
    for (int v : {0, -1, 129, std::numeric_limits<int>::max()}) both([&](auto& a) { a.count = v; });
    for (int v : {-1, entries, entries - 7, std::numeric_limits<int>::max()}) both([&](auto& a) { a.offset = v; });
    both([](auto& a) { a.valid_entries = 3; a.count = 4; a.offset = 0; });
    for (int v : {0, -1, 7, std::numeric_limits<int>::max()}) both([&](auto& a) { a.capacity = v; });
    for (int v : {0, -1, blocks - 1, blocks + 1, std::numeric_limits<int>::max()}) {
        auto a = g; a.row_blocks = v; reject_gather(a);
    }
    for (int v : {0, -1, width - 1, width + 1, std::numeric_limits<int>::max()}) {
        auto a = s; a.row_width = v; reject_scatter(a);
    }
    for (int v : {0, -1, ranks - 1, ranks + 1, std::numeric_limits<int>::max()}) {
        auto a = s; a.top_k = v; reject_scatter(a);
    }
    both([](auto& a) { a.input.data = nullptr; });
    both([](auto& a) { a.output.data = nullptr; });
    both([](auto& a) { a.routes.data = nullptr; });
    both([](auto& a) { a.error = nullptr; });
    for (auto n : {std::size_t(0), std::size_t(1), std::numeric_limits<std::size_t>::max()}) {
        both([&](auto& a) { a.input.capacity = n; });
        both([&](auto& a) { a.output.capacity = n; });
        both([&](auto& a) { a.routes.capacity = n; });
    }
    { auto a = g; a.input.capacity = std::size_t(tokens) * blocks - 1; reject_gather(a); }
    { auto a = g; a.output.capacity = std::size_t(columns) * blocks - 1; reject_gather(a); }
    { auto a = s; a.input.capacity = std::size_t(columns) * width - 1; reject_scatter(a); }
    { auto a = s; a.output.capacity = std::size_t(entries) * width - 1; reject_scatter(a); }
    both([](auto& a) { a.routes.capacity = entries - 1; });
    // Late offsets cannot conceal a bad full metadata capacity or base range.
    both([](auto& a) { a.offset = entries - 8; a.routes.capacity = entries - 1; });
    const auto ptrmax = std::numeric_limits<std::uintptr_t>::max();
    both([&](auto& a) { a.routes.data = reinterpret_cast<const MoeRouteIndex*>(ptrmax & ~std::uintptr_t(3)); });
    both([&](auto& a) { a.offset = entries - 8;
        a.routes.data = reinterpret_cast<const MoeRouteIndex*>(ptrmax & ~std::uintptr_t(3)); });
    { auto a = g; a.input.data = reinterpret_cast<const Q8_1*>(ptrmax & ~std::uintptr_t(3)); reject_gather(a); }
    { auto a = g; a.output.data = reinterpret_cast<Q8_1*>(ptrmax & ~std::uintptr_t(3)); reject_gather(a); }
    { auto a = s; a.input.data = reinterpret_cast<const float*>(ptrmax & ~std::uintptr_t(3)); reject_scatter(a); }
    { auto a = s; a.output.data = reinterpret_cast<float*>(ptrmax & ~std::uintptr_t(3)); reject_scatter(a); }
    both([&](auto& a) { a.error = reinterpret_cast<int*>(ptrmax & ~std::uintptr_t(3)); });
    // Full span PTRDIFF bound, independently of address wrapping and of small
    // accessed shapes. Exact >max/sizeof(T) extent with a small aligned base.
    const auto diffmax = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    { auto a = g; a.input = {reinterpret_cast<const Q8_1*>(16), diffmax / sizeof(Q8_1) + 1}; reject_gather(a); }
    { auto a = g; a.output = {reinterpret_cast<Q8_1*>(16), diffmax / sizeof(Q8_1) + 1}; reject_gather(a); }
    { auto a = s; a.input = {reinterpret_cast<const float*>(16), diffmax / sizeof(float) + 1}; reject_scatter(a); }
    { auto a = s; a.output = {reinterpret_cast<float*>(16), diffmax / sizeof(float) + 1}; reject_scatter(a); }
    both([&](auto& a) { a.routes = {reinterpret_cast<const MoeRouteIndex*>(16), diffmax / sizeof(MoeRouteIndex) + 1}; });
    // Q8 is natively 2-aligned, but copying requires 4-aligned source/dest.
    for (std::uintptr_t delta : {1u, 2u}) {
        auto a = g; a.input.data = reinterpret_cast<const Q8_1*>(reinterpret_cast<std::uintptr_t>(a.input.data) + delta);
        reject_gather(a); a = g; a.output.data = reinterpret_cast<Q8_1*>(reinterpret_cast<std::uintptr_t>(a.output.data) + delta);
        reject_gather(a);
    }
    { auto a = s; a.input.data = reinterpret_cast<const float*>(reinterpret_cast<std::uintptr_t>(a.input.data) + 1); reject_scatter(a); }
    { auto a = s; a.output.data = reinterpret_cast<float*>(reinterpret_cast<std::uintptr_t>(a.output.data) + 1); reject_scatter(a); }
    both([](auto& a) { a.routes.data = reinterpret_cast<const MoeRouteIndex*>(reinterpret_cast<std::uintptr_t>(a.routes.data) + 1); });
    both([](auto& a) { a.error = reinterpret_cast<int*>(reinterpret_cast<std::uintptr_t>(a.error) + 1); });
    // Real exact/partial data overlap; declared WHOLE spans also cover unused
    // columns and route entries, not merely count8 at offset3.
    { auto a = g; a.output.data = const_cast<Q8_1*>(a.input.data); reject_gather(a);
      a.output.data = const_cast<Q8_1*>(a.input.data + 1); reject_gather(a); }
    { auto a = s; a.output.data = const_cast<float*>(a.input.data); reject_scatter(a);
      a.output.data = const_cast<float*>(a.input.data + 1); reject_scatter(a); }
    { auto a = g; a.input = {gathered.as<const Q8_1>() + 8 * blocks, std::size_t(tokens) * blocks}; reject_gather(a); }
    { auto a = s; a.input = {contributions.as<const float>() + 128 * width, std::size_t(columns) * width}; reject_scatter(a); }
    { auto a = g; a.routes = {gathered.as<const MoeRouteIndex>(), std::size_t(entries)}; reject_gather(a); }
    { auto a = s; a.routes = {contributions.as<const MoeRouteIndex>(), std::size_t(entries)}; reject_scatter(a); }
    { auto a = g; a.output = {metadata.as<Q8_1>(), std::size_t(columns) * blocks}; reject_gather(a); }
    { auto a = s; a.output = {metadata.as<float>(), std::size_t(entries) * width}; reject_scatter(a); }
    { auto a = g; a.error = reinterpret_cast<int*>(const_cast<Q8_1*>(a.input.data)); reject_gather(a);
      a.error = reinterpret_cast<int*>(a.output.data + 128 * blocks); reject_gather(a); }
    { auto a = s; a.error = reinterpret_cast<int*>(const_cast<float*>(a.input.data)); reject_scatter(a);
      a.error = reinterpret_cast<int*>(a.output.data + (entries - 1) * width); reject_scatter(a); }
    both([](auto& a) { a.error = reinterpret_cast<int*>(const_cast<MoeRouteIndex*>(a.routes.data) + entries - 1); });
    check(hipStreamSynchronize(stream.value)); all(); // No flag/data/canary mutation from any rejection.
}

Counts qualify(int device) {
    check(hipSetDevice(device)); Counts result;
    for (std::size_t guard : {64u, 68u}) {
        Fixture fixture(guard, result); fixture.valid(); fixture.invalid(); fixture.host_rejections();
    }
    owners.empty(); return result;
}
void run() {
    constexpr std::string_view revision = CORE_REVISION;
    require(CORE_DIRTY == 0 || CORE_DIRTY == 1, "CORE_DIRTY must be numeric0/1");
    require(revision.size() == 40 && std::all_of(revision.begin(), revision.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }), "requires quoted 40-lowercase-hex CORE_REVISION");
    int device_count = 0; check(hipGetDeviceCount(&device_count)); require(device_count == 2, "requires both real gfx906 devices");
    for (int device = 0; device < 2; ++device) {
        hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p, device)); const std::string_view arch = p.gcnArchName;
        require(p.warpSize == 64 && (arch == "gfx906" || arch.starts_with("gfx906:")), "requires gfx906 wave64");
    }
    const std::array<Counts, 2> results{qualify(0), qualify(1)}; owners.empty();
    // Bounded four-record output AFTER every GPU owner has been destroyed.
    std::cout << "{\"kind\":\"moe_route_copy_source\",\"protocol\":1,\"revision\":\"" << revision
        << "\",\"dirty\":" << CORE_DIRTY
        << ",\"scope\":\"opaque byte copy only\",\"route_abi_bytes\":8,\"q8_abi_bytes\":36,\"q8_row_bytes\":2880,\"down_row_bytes\":10240"
           ",\"tokens\":1024,\"ranks\":10,\"column_capacity\":129,\"counts\":[";
    for (std::size_t i = 0; i < counts.size(); ++i) { if (i) std::cout << ','; std::cout << counts[i]; }
    std::cout << "],\"payload_alignments\":[16,4],\"tolerance\":0,\"jsonl_records\":4}\n";
    for (int device = 0; device < 2; ++device) {
        const auto& c = results[device];
        std::cout << "{\"kind\":\"moe_route_copy_correctness\",\"protocol\":1,\"device\":" << device
            << ",\"gather_calls\":" << c.gather_calls << ",\"scatter_calls\":" << c.scatter_calls
            << ",\"metadata_uploads\":" << c.metadata_uploads << ",\"full_token_rank_pairs\":" << c.full_pairs
            << ",\"partial_tiles\":" << c.partial_tiles << ",\"completed_metadata_reuses\":" << c.completed_reuses
            << ",\"invalid_calls\":" << c.invalid_calls << ",\"sticky_valid_reuses\":" << c.sticky_reuses
            << ",\"cleared_valid_reuses\":" << c.cleared_reuses << ",\"host_rejects\":" << c.host_rejects
            << ",\"exact_bytes\":true,\"sticky_or_3\":true,\"source_metadata_readonly\":true,\"untouched_poison\":true"
               ",\"canaries\":true,\"cleanup\":true,\"passed\":true}\n";
    }
    std::cout << "{\"kind\":\"moe_route_copy_complete\",\"protocol\":1,\"devices\":2,\"jsonl_records\":4,\"live_owners\":0,\"cleanup\":true,\"passed\":true}\n";
    std::cout.flush(); require(bool(std::cout), "MoE copy JSONL output failure");
}
} // namespace

int main(int argc, char**) {
    try { require(argc == 1, "usage: core-moe-route-copy"); run(); owners.empty(); return 0; }
    catch (const std::exception& e) { std::cerr << "moe-route-copy: " << e.what() << '\n'; return 1; }
}
