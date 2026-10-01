#include "qsa_select.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "qsa_select.hip is specialized for gfx906 wave64"
#endif

// Donor-first, one query, FP32 scores[visible/4], compress4/budget2048:
// mx dcd685463d597d31f5ca759d32c94592a2740fa4, src/models/qwen4exp.cpp
// ::build_qsa_top_k / ggml/src/ggml-cuda/top-k.cu is the historical diagnostic
// baseline, not the whole-block/actual-tail specification. The qualified oracle
// is src/qsa.cpp::qsa_select: numeric descending, lower-ID ties, +0 == -0.
// ADAPT furnace 905021dbad71c5056ef51f9fd45d545403fc989c,
// ggml/src/ggml-cuda/top-k.cu::top_k_float_to_ordered, top_k_radix_histogram,
// top_k_radix_select, top_k_radix_gather and top_k_radix_cuda (full source read
// in runs/donors/furnace-top-k.cu). Retain 256-bin partial LDS histograms,
// four independent partial reduction sums and parallel suffix-rank selection.
// Modifications: unique uint64 (ordered score, complemented chronological ID)
// keys, signed-zero normalization, eight radix bytes, at most 32 partial blocks,
// checked bounded gather, a final <=512-key LDS sort, whole-block expansion and
// actual counts/tail. No atomic tie membership, full-history sort/workspace,
// ggml pool/runtime, allocations, device queries or host synchronization.
// Checked reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8 in RECON.md
// section 16: its dense Q8 attention is not this QSA block-selection contract.
// WRITE OURS borrowed-buffer validation, sticky flags and conditional publication.
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

namespace qwen {
namespace {

constexpr int threads = 256;
constexpr int bins = 256;
constexpr int max_partials = 32;
constexpr int max_candidates = 512;
constexpr int token_capacity = 2051;
constexpr int ratio = 4;
constexpr int state_words = 16;
constexpr int error_invalid = 1;
constexpr int error_internal = 2;
constexpr std::uint64_t all_bits = ~std::uint64_t{0};
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(int) == 4 && sizeof(std::int32_t) == 4);
static_assert(sizeof(std::uint64_t) == 8 && threads == bins);

// Native uint64 words, including the atomic counters: no struct/int type punning
// over the frozen workspace. Other words are reserved and initialized to zero.
enum StateWord : int {
    prefix_word, mask_word, rank_word, gathered_word, equal_word
};

struct Range {
    std::uintptr_t begin, end;
    bool writable, valid;
};

template<class T>
Range range(const T* pointer, std::size_t count, bool writable = false) noexcept {
    // For visible=1..3 neither scores nor block_ids has an active element.
    if (count == 0) return {0, 0, writable, true};
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (pointer == nullptr || begin % alignof(T) != 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(T))
        return {0, 0, writable, false};
    const auto bytes = count * sizeof(T);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin)
        return {0, 0, writable, false};
    return {begin, begin + bytes, writable, true};
}

template<std::size_t Count>
bool valid_ranges(const std::array<Range, Count>& buffers) noexcept {
    for (const auto& r : buffers) if (!r.valid) return false;
    for (std::size_t i = 0; i < Count; ++i) {
        const auto& a = buffers[i];
        if (a.begin == a.end) continue;
        for (std::size_t j = i + 1; j < Count; ++j) {
            const auto& b = buffers[j];
            if (b.begin != b.end && (a.writable || b.writable) &&
                a.begin < b.end && b.begin < a.end) return false;
        }
    }
    return true;
}

__device__ __forceinline__ void report(int flags, int* error) {
    if (flags != 0) atomicOr(error, flags);
}

// ADAPT furnace's numeric float ordering without any FP arithmetic/comparison.
// This also preserves finite subnormal order on devices configured to flush FP
// arithmetic to zero. Sentinel zero is below EVERY finite composite rank key.
__device__ __forceinline__ std::uint64_t rank_key(float score, int id, int& flags) {
    std::uint32_t bits = __float_as_uint(score);
    if ((bits & 0x7f800000U) == 0x7f800000U) {
        flags |= error_invalid;
        return 0;
    }
    if ((bits & 0x7fffffffU) == 0) bits = 0; // +0/-0 are numeric ties
    const std::uint32_t ordered = bits ^ ((bits >> 31) ? 0xffffffffU : 0x80000000U);
    return (std::uint64_t{ordered} << 32) | ~static_cast<std::uint32_t>(id);
}

template<bool Direct>
__global__ __launch_bounds__(threads)
void initialize(const float* scores, int blocks, int k, QsaSelectWorkspace w) {
    const int tid = threadIdx.x;
    if (tid < state_words) {
        std::uint64_t value = 0;
        if (tid == rank_word) value = static_cast<std::uint64_t>(k);
        if constexpr (Direct) {
            if (tid == gathered_word) value = static_cast<std::uint64_t>(k);
        }
        w.state[tid] = value;
    }
    int flags = 0;
    for (int i = tid; i < max_candidates; i += threads) {
        std::uint64_t key = 0;
        if constexpr (Direct) {
            if (i < blocks) key = rank_key(scores[i], i, flags);
        }
        w.candidates[i] = key;
    }
    report(flags, w.error); // Caller-owned sticky flag is NEVER reset here.
}

// ADAPT furnace top_k_radix_histogram: only the exactly visible completed-block
// prefix is read, in a bounded strided grid (ceil(blocks/1024) <= 32 groups).
__global__ __launch_bounds__(threads)
void radix_histogram(const float* scores, int blocks, int partials, int shift,
                     QsaSelectWorkspace w) {
    const int tid = threadIdx.x;
    const int partial = blockIdx.x;
    __shared__ int histogram[bins];
    histogram[tid] = 0;
    __syncthreads();

    const std::uint64_t prefix = w.state[prefix_word];
    const std::uint64_t mask = w.state[mask_word];
    int flags = 0;
    for (int id = partial * threads + tid; id < blocks; id += partials * threads) {
        const auto key = rank_key(scores[id], id, flags);
        if (key != 0 && (key & mask) == prefix)
            atomicAdd(&histogram[static_cast<int>((key >> shift) & 255U)], 1);
    }
    __syncthreads();
    w.histogram[partial * bins + tid] = histogram[tid];
    report(flags, w.error);
}

// ADAPT furnace top_k_radix_select: suffix sums find the highest bin reaching
// the one-based remaining rank. No scores/keys are stored for discarded blocks.
__global__ __launch_bounds__(threads)
void radix_select(int blocks, int partials, int shift, QsaSelectWorkspace w) {
    const int tid = threadIdx.x;
    __shared__ int histogram[bins];
    __shared__ int active;
    const auto prefix = w.state[prefix_word];
    const auto mask = w.state[mask_word];
    const auto rank = w.state[rank_word];
    if (tid == 0) {
        active = *w.error == 0;
        const auto expected_mask = shift == 56 ? std::uint64_t{0} : all_bits << (shift + 8);
        if (active && (mask != expected_mask || (prefix & ~mask) != 0 ||
                       rank == 0 || rank > static_cast<std::uint64_t>(blocks))) {
            report(error_internal, w.error);
            active = 0;
        }
    }
    __syncthreads();
    if (!active) return; // Uniform: producer flags are complete in stream order.

    const int* h = w.histogram + tid;
    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int partial = 0;
    for (; partial + 4 <= partials; partial += 4) {
        c0 += h[(partial + 0) * bins];
        c1 += h[(partial + 1) * bins];
        c2 += h[(partial + 2) * bins];
        c3 += h[(partial + 3) * bins];
    }
    for (; partial < partials; ++partial) c0 += h[partial * bins];
    histogram[tid] = (c0 + c1) + (c2 + c3);
    __syncthreads();

    for (int off = 1; off < bins; off *= 2) {
        const int add = tid + off < bins ? histogram[tid + off] : 0;
        __syncthreads();
        histogram[tid] += add;
        __syncthreads();
    }
    if (tid == 0 && (histogram[0] < static_cast<int>(rank) || histogram[0] > blocks))
        report(error_internal, w.error);
    const int above = tid + 1 < bins ? histogram[tid + 1] : 0;
    if (histogram[tid] >= static_cast<int>(rank) && above < static_cast<int>(rank)) {
        w.state[rank_word] = rank - static_cast<std::uint64_t>(above);
        w.state[prefix_word] = prefix | (static_cast<std::uint64_t>(tid) << shift);
        w.state[mask_word] = mask | (std::uint64_t{255} << shift);
    }
}

// ADAPT furnace top_k_radix_gather: a fully resolved UNIQUE key determines
// membership, including the cutoff's lower-ID ties. Atomic arrival order only
// determines temporary slots; it cannot decide which tied IDs are selected.
__global__ __launch_bounds__(threads)
void radix_gather(const float* scores, int blocks, int partials, QsaSelectWorkspace w) {
    // This kernel can report errors in other groups, so use an atomic flag load.
    if (__atomic_load_n(w.error, __ATOMIC_RELAXED) != 0) return;
    const auto threshold = w.state[prefix_word];
    int flags = 0;
    for (int id = blockIdx.x * threads + threadIdx.x; id < blocks; id += partials * threads) {
        const auto key = rank_key(scores[id], id, flags);
        if (key != 0 && key >= threshold) {
            // Use the exact uint64 workspace type (uint64_t may be unsigned
            // long, rather than HIP atomicAdd's unsigned long long overload).
            const auto slot = __atomic_fetch_add(w.state + gathered_word, std::uint64_t{1},
                                                 __ATOMIC_RELAXED);
            if (slot < max_candidates) w.candidates[slot] = key;
            else flags |= error_internal; // Never write past the 512-key capacity.
            if (key == threshold)
                (void) __atomic_fetch_add(w.state + equal_word, std::uint64_t{1}, __ATOMIC_RELAXED);
        }
    }
    report(flags, w.error);
}

// WRITE OURS bounded bitonic network: 256 disjoint comparator pairs/pass over
// 512 LDS keys (4 KiB), 45 compare/barrier passes. Never sort the full history.
// One group owns all validation AND publication, so after its final barrier no
// thread can discover a late numeric error while another is publishing IDs.
template<bool Radix>
__global__ __launch_bounds__(threads)
void sort_publish(int blocks, int k, int visible, std::int32_t* token_ids,
                  std::int32_t* block_ids, int* counts, QsaSelectWorkspace w) {
    const int tid = threadIdx.x;
    __shared__ std::uint64_t keys[max_candidates];
    __shared__ int publish;
    if (tid == 0) publish = *w.error == 0;
    __syncthreads();
    if (!publish) return;

    int flags = 0;
    if (tid == 0) {
        if (w.state[gathered_word] != static_cast<std::uint64_t>(k)) flags |= error_internal;
        if constexpr (Radix) {
            if (w.state[mask_word] != all_bits || w.state[rank_word] != 1 ||
                w.state[equal_word] != 1 || w.state[prefix_word] == 0) flags |= error_internal;
        }
    }
    keys[tid] = w.candidates[tid];
    keys[tid + threads] = w.candidates[tid + threads];
    __syncthreads();
    for (int size = 2; size <= max_candidates; size *= 2) {
        for (int distance = size / 2; distance != 0; distance /= 2) {
            const int left = (tid / distance) * (2 * distance) + tid % distance;
            const int right = left + distance;
            const auto a = keys[left], b = keys[right];
            const bool descending = (left & size) == 0;
            if ((descending && a < b) || (!descending && a > b)) {
                keys[left] = b;
                keys[right] = a;
            }
            __syncthreads();
        }
    }

    for (int rank = tid; rank < max_candidates; rank += threads) {
        const auto key = keys[rank];
        if (rank < k) {
            const auto ordered = static_cast<std::uint32_t>(key >> 32);
            const auto id = ~static_cast<std::uint32_t>(key);
            if (ordered < 0x00800000U || ordered > 0xff7fffffU || ordered == 0x7fffffffU ||
                id >= static_cast<std::uint32_t>(blocks) ||
                (rank != 0 && keys[rank - 1] <= key)) flags |= error_internal;
            if constexpr (Radix) {
                if (key < w.state[prefix_word] ||
                    (rank == k - 1 && key != w.state[prefix_word])) flags |= error_internal;
            }
        } else if (key != 0) flags |= error_internal;
    }
    report(flags, w.error);
    __syncthreads(); // All group-local checks/atomic flags precede publication.
    if (tid == 0) publish = *w.error == 0;
    __syncthreads();
    if (!publish) return;

    for (int rank = tid; rank < k; rank += threads) {
        const auto id = static_cast<std::int32_t>(~static_cast<std::uint32_t>(keys[rank]));
        block_ids[rank] = id;
        for (int member = 0; member < ratio; ++member)
            token_ids[rank * ratio + member] = id * ratio + member;
    }
    const int tail = visible % ratio;
    if (tid < tail) token_ids[k * ratio + tid] = blocks * ratio + tid;
    if (tid == 0) {
        counts[0] = k * ratio + tail;
        counts[1] = k;
    }
}

} // namespace

hipError_t launch_qsa_select(const float* scores, int visible_tokens,
        std::int32_t* token_ids, std::int32_t* block_ids, int* counts,
        QsaSelectWorkspace w, hipStream_t stream) noexcept {
    if (visible_tokens < 1 || visible_tokens > 131072) return hipErrorInvalidValue;
    const int blocks = visible_tokens / ratio;
    const int k = blocks < max_candidates ? blocks : max_candidates;
    // Validate full DECLARED storage, even if a launch does not use all of it:
    // tokens[2051], blocks[k], counts[2], hist[8192], state[16], candidates[512].
    // Every writable pair and every writable/input alias is rejected before any
    // enqueue, including aliases with the sticky flag or unused scratch/capacity.
    // The raw-pointer API cannot inspect allocation extents/device residency;
    // the caller guarantees those and immutable scores through stream completion.
    if (!valid_ranges(std::array{
            range(scores, blocks), range(token_ids, token_capacity, true),
            range(block_ids, k, true), range(counts, 2, true),
            range(w.histogram, max_partials * bins, true), range(w.state, state_words, true),
            range(w.candidates, max_candidates, true), range(w.error, 1, true)}))
        return hipErrorInvalidValue;

    // No default-stream substitution. Empty completed-block prefixes need no
    // scores/block_ids, but still publish their actual 1..3 chronological tokens.
    if (blocks <= max_candidates) {
        initialize<true><<<1, threads, 0, stream>>>(scores, blocks, k, w);
        const auto status = hipGetLastError();
        if (status != hipSuccess) return status;
        sort_publish<false><<<1, threads, 0, stream>>>(blocks, k, visible_tokens, token_ids, block_ids, counts, w);
        return hipGetLastError();
    }

    initialize<false><<<1, threads, 0, stream>>>(scores, blocks, k, w);
    auto status = hipGetLastError();
    if (status != hipSuccess) return status;
    const int partials = (blocks + 1023) / 1024; // 1..32 after bounded host validation
    for (int shift = 56; shift >= 0; shift -= 8) {
        radix_histogram<<<partials, threads, 0, stream>>>(scores, blocks, partials, shift, w);
        status = hipGetLastError();
        if (status != hipSuccess) return status;
        radix_select<<<1, threads, 0, stream>>>(blocks, partials, shift, w);
        status = hipGetLastError();
        if (status != hipSuccess) return status;
    }
    radix_gather<<<partials, threads, 0, stream>>>(scores, blocks, partials, w);
    status = hipGetLastError();
    if (status != hipSuccess) return status;
    sort_publish<true><<<1, threads, 0, stream>>>(blocks, k, visible_tokens, token_ids, block_ids, counts, w);
    return hipGetLastError();
}

} // namespace qwen
