#include "session.hpp"
#include "cpu_expert.hpp"
#include "dense.hpp"
#include "ple.hpp"
#include "routes.hpp"
#include "prefill_pipeline.hpp"
#include "hip/attention.cuh"
#include "hip/blocks.cuh"
#include "hip/dense.cuh"
#include "hip/gdn.cuh"
#include "hip/kv.cuh"
#include "hip/linear.cuh"
#include "hip/mmq_wide.cuh"
#include "hip/qsa_index.cuh"
#include "hip/qsa_select.cuh"
#include "hip/session_ops.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

namespace qwen {
namespace {
constexpr int H = 2560, W = 10240, V = 248320;
constexpr int microtile_tokens = 128, expert_band = 16;
constexpr std::size_t expert_stage_bytes = 45875200; // 16 canonical Q4_1-down triplets
constexpr std::size_t short_stage_bytes = expert_stage_bytes / expert_band;
static_assert(microtile_tokens == blocks_device_max_tokens && microtile_tokens == gdn_device_max_tokens &&
              microtile_tokens == session_ops_max_tokens && microtile_tokens == mmq_device_max_columns);
static_assert(H == moe_copy_width && H / 32 == moe_copy_q8_blocks && moe_copy_ranks == 10 &&
              sizeof(RouteAssignment) == 16 && sizeof(MoeRouteIndex) == 8);
static_assert(silu_pairs_width == 640 && silu_pairs_max_columns == cpu_expert_max_jobs);
static_assert(attention_query_heads == 24 && attention_kv_heads == 2 && attention_gqa == 12 &&
    attention_head_dim == 256 && attention_query_elements == 6144 &&
    attention_max_selected == 2051 && attention_chunk_keys == 64 && attention_max_chunks == 33);
static_assert(attention_batch_workspace_bytes_per_query == 5042368 &&
    attention_batch_workspace_bytes_per_query * 8 == 40338944 &&
    (attention_max_selected + 512 + 2) * sizeof(int) == 10260);
std::size_t checked_size(std::size_t count, std::size_t width) {
    if (width && count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / width)
        throw std::overflow_error("Session allocation size");
    return count * width;
}
template<class F> void tiles(int tokens, F&& f) {
    for (int first = 0; first < tokens; first += microtile_tokens)
        f(first, std::min(microtile_tokens, tokens - first));
}
SessionConfig checked_config(SessionConfig config) {
    if (config.capacity < 4 || config.capacity > 131072 || config.capacity % 4 ||
        config.expert_slots < 1 || config.expert_slots > 512 ||
        config.max_batch_tokens < 1 || config.max_batch_tokens > 1024 ||
        config.cpu_workers < 0 || config.cpu_workers > static_cast<int>(cpu_expert_max_workers))
        throw std::invalid_argument("Session capacity/slots/max_batch_tokens");
    if (config.trace_first_token < 0 || config.trace_first_token > config.capacity)
        throw std::invalid_argument("Session trace_first_token");
    if (config.hybrid_probe && !config.cpu_workers) throw std::invalid_argument("Session hybrid probe resources");
    if (config.attention_query_tile < 1 || config.attention_query_tile > attention_max_batch)
        throw std::invalid_argument("Session attention_query_tile");
    if (config.speculative_checkpoints && config.max_batch_tokens < 3)
        throw std::invalid_argument("Session speculative checkpoints require max_batch_tokens>=3");
    if (config.layerwise_prefill_capacity < 0 || config.layerwise_prefill_capacity > 16384 ||
        (config.layerwise_prefill_capacity &&
         (config.max_batch_tokens < 4 || !config.trace_directory.empty())))
        throw std::invalid_argument("Session layerwise capacity/resources/trace");
    if (config.prefill_pipeline_tokens < 0 || config.prefill_pipeline_tokens > 4096 ||
        (config.prefill_pipeline_tokens &&
         (config.layerwise_prefill_capacity < 1 ||
          config.prefill_pipeline_tokens > config.layerwise_prefill_capacity || config.cpu_workers ||
          (config.layerwise_prefill_capacity + config.prefill_pipeline_tokens - 1) /
              config.prefill_pipeline_tokens > PrefillPipeline::max_windows)))
        throw std::invalid_argument("Session prefill pipeline resources/window");
    if (config.prefill_residual_device && !config.prefill_pipeline_tokens)
        throw std::invalid_argument("Session device residual requires bounded pipeline");
    const bool fault = config.prefill_pipeline_fail_stage != -1 || config.prefill_pipeline_fail_window != -1;
    if (fault && (!config.prefill_pipeline_tokens ||
        config.prefill_pipeline_fail_stage < 0 || config.prefill_pipeline_fail_stage > 1 ||
        config.prefill_pipeline_fail_window < 0 || config.prefill_pipeline_fail_window >= PrefillPipeline::max_windows))
        throw std::invalid_argument("Session prefill pipeline fault bound");
    return config;
}
void hip(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
void blas(rocblas_status e) {
    if (e != rocblas_status_success) throw std::runtime_error("rocBLAS status " + std::to_string(e));
}
void cleanup(hipError_t e) noexcept {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP cleanup: %s\n", hipGetErrorString(e)); std::abort(); }
}
void cpu_ok(CpuExpertStatus status) {
    if (status != CpuExpertStatus::ok)
        throw std::runtime_error("Session CPU expert: " + std::string(cpu_expert_status_name(status)));
}
struct CoordinatorAffinity {
#if defined(__linux__)
    cpu_set_t previous{};
    bool active = false;
    explicit CoordinatorAffinity(int cpu) {
        if (cpu < 0) return;
        if (sched_getaffinity(0, sizeof(previous), &previous)) throw std::runtime_error("Session coordinator affinity read");
        cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(cpu, &mask);
        if (sched_setaffinity(0, sizeof(mask), &mask)) throw std::runtime_error("Session coordinator pin");
        active = true;
    }
    ~CoordinatorAffinity() {
        if (active && sched_setaffinity(0, sizeof(previous), &previous)) std::abort();
    }
#else
    explicit CoordinatorAffinity(int) {}
#endif
};

class Buffer {
public:
    inline static std::array<std::uint64_t, 2> live_bytes{}, peak_bytes{}, live_buffers{};
    Buffer() = default;
    explicit Buffer(std::size_t bytes) : bytes_(bytes) {
        hip(hipGetDevice(&device_));
        if (device_ < 0 || device_ >= 2) throw std::runtime_error("Session buffer owner");
        hip(hipMalloc(&pointer_, bytes));
        live_bytes[device_] += bytes_; ++live_buffers[device_];
        peak_bytes[device_] = std::max(peak_bytes[device_], live_bytes[device_]);
    }
    ~Buffer() { release(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& b) noexcept : pointer_(std::exchange(b.pointer_, nullptr)), bytes_(std::exchange(b.bytes_, 0)), device_(b.device_) {}
    Buffer& operator=(Buffer&& b) noexcept {
        if (this != &b) {
            release();
            pointer_ = std::exchange(b.pointer_, nullptr); bytes_ = std::exchange(b.bytes_, 0); device_ = b.device_;
        }
        return *this;
    }
    template<class T = float> T* get() const { return static_cast<T*>(pointer_); }
    std::size_t bytes() const { return bytes_; }
    int owner() const { return device_; }
private:
    void release() noexcept {
        if (!pointer_) return;
        cleanup(hipSetDevice(device_)); cleanup(hipFree(pointer_));
        live_bytes[device_] -= bytes_; --live_buffers[device_];
        pointer_ = nullptr; bytes_ = 0;
    }
    void* pointer_ = nullptr;
    std::size_t bytes_ = 0;
    int device_ = 0;
};

// WRITE OURS bounded orchestration over KEEP canonical MMQ/DPP primitives.
// RECON16: mx dcd6854 staging/layout-aware PP D2D, furnace 905021d pipeline,
// reinstinct 0b79e32 route sort/gather/scatter. No donor scheduler or GPU slab.
// Exactly two reusable band16 payloads per device, only for explicit max>3.
struct ExpertStage {
    int device;
    Buffer data;
    std::byte* host = nullptr;
    hipEvent_t copy_ready = nullptr, consumer_done = nullptr;
    bool used = false;
    explicit ExpertStage(int id, std::size_t bytes = expert_stage_bytes) : device(id), data(bytes) {
        try {
            hip(hipHostMalloc(&host, bytes, hipHostMallocDefault));
            if (bytes == short_stage_bytes) std::memset(host, 0, bytes); // New short frames first-touch once.
            hip(hipEventCreateWithFlags(&copy_ready, hipEventDisableTiming));
            hip(hipEventCreateWithFlags(&consumer_done, hipEventDisableTiming));
        } catch (...) { release(); throw; }
    }
    ~ExpertStage() { release(); }
    void release() noexcept {
        cleanup(hipSetDevice(device));
        if (consumer_done) { cleanup(hipEventDestroy(consumer_done)); consumer_done = nullptr; }
        if (copy_ready) { cleanup(hipEventDestroy(copy_ready)); copy_ready = nullptr; }
        if (host) { cleanup(hipHostFree(host)); host = nullptr; }
    }
};

// WRITE OURS bounded CPUlinear_GPUmiddle orchestration. KEEP mx
// dcd685463d597d31f5ca759d32c94592a2740fa4 common.cuh/vecdotq.cuh canonical/DPP,
// furnace 905021dbad71c5056ef51f9fd45d545403fc989c ggml-cuda.cu event pattern,
// reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8 moe_expert_sort.cpp and
// src/runtime/pipeline.rs grouped gather/scatter/lifetime pattern. ADAPT the
// existing CpuExpert/Pool projections; canonical GPU SiLU/quantization between
// CPU phases preserves the nonlinear producer contract (witness759). Paired
// middle addressing alone is new; whole CPU host-libm is only a component oracle.
struct Device;
struct HybridInputProbeFrame {
    std::vector<Q8_1> original;
    std::vector<std::int32_t> ids;
    std::vector<float> weights;
    std::vector<std::uint8_t> available, cpu;
    void allocate() {
        original.resize(3 * 48 * (H / 32)); ids.resize(3 * 48 * 10);
        weights.resize(3 * 48 * 10); available.resize(3 * 48); cpu.resize(3 * 48 * 10);
    }
    void begin(int tokens) {
        std::fill_n(available.data(), tokens * 48, std::uint8_t{0});
        std::fill_n(cpu.data(), tokens * 48 * 10, std::uint8_t{0});
    }
    void capture(const Device&, int layer, int tokens, bool hybrid);
    std::uint64_t capacity_bytes() const {
        return original.capacity() * sizeof(Q8_1) + ids.capacity() * sizeof(std::int32_t) +
            weights.capacity() * sizeof(float) + available.capacity() + cpu.capacity();
    }
};
struct HybridDevice {
    enum class Work { hit, cpu, gpu_miss };
    struct Plan {
        int expert = -1;
        const std::byte* hit = nullptr; // Immutable physical snapshot, never a map lookup later.
        Work work = Work::cpu;
        std::size_t output_column = 0, ticket = 0;
        CpuExpertJob job{};
    };
    int device;
    Q8_1* input = nullptr;
    Q8_1* middle = nullptr;
    float* output = nullptr;
    int* error = nullptr;
    Buffer gate_up, middle_float, middle_q8, middle_error;
    hipEvent_t input_copied = nullptr, middle_ready = nullptr, cpu_returned = nullptr;
    std::array<Plan, cpu_expert_max_jobs> plans{};
    // Fixed pageable diagnostic bytes, independent of the pinned dispatch input.
    // No transfer/read in production or on an all-hit layer, even with probe on.
    std::array<Q8_1, 3 * (H / 32)> original_probe{};
    // Only device0 allocates these payloads, only with constructor hybrid_probe.
    // Keeping this hybrid metadata in the already-opt-in owner adds no default
    // hybrid payload. The separate checkpoint handle changes base host Impl
    // sizeof even with snapshots off, and memory() reports the actual sizeof.
    std::array<HybridInputProbeFrame, 2> probe_inputs;
    std::array<std::uint64_t, 6> projection_statistics{}; // Published tail on device0 only.
    int count = 0, cpu_columns = 0;
    bool cpu_active = false;
    static constexpr std::size_t input_bytes = 3 * (H / 32) * sizeof(Q8_1);
    static constexpr std::size_t output_bytes = 3 * 10 * H * sizeof(float);
    static constexpr std::size_t gate_up_bytes = 30 * 2 * 640 * sizeof(float);
    static constexpr std::size_t middle_float_bytes = 30 * 640 * sizeof(float);
    static constexpr std::size_t middle_bytes = 30 * 20 * sizeof(Q8_1);
    explicit HybridDevice(int id) : device(id) {
        try {
            gate_up = Buffer(gate_up_bytes); middle_float = Buffer(middle_float_bytes);
            middle_q8 = Buffer(middle_bytes); middle_error = Buffer(sizeof(int));
            hip(hipHostMalloc(&input, input_bytes, hipHostMallocDefault));
            hip(hipHostMalloc(&middle, middle_bytes, hipHostMallocDefault));
            hip(hipHostMalloc(&output, output_bytes, hipHostMallocDefault));
            hip(hipHostMalloc(&error, sizeof(int), hipHostMallocDefault));
            hip(hipEventCreateWithFlags(&input_copied, hipEventDisableTiming));
            hip(hipEventCreateWithFlags(&middle_ready, hipEventDisableTiming));
            hip(hipEventCreateWithFlags(&cpu_returned, hipEventDisableTiming));
            std::memset(input, 0, input_bytes); std::memset(middle, 0, middle_bytes);
            std::memset(output, 0, output_bytes); *error = 0;
        } catch (...) { release(); throw; }
    }
    ~HybridDevice() { release(); }
    void release() noexcept {
        cleanup(hipSetDevice(device));
        if (input_copied) { cleanup(hipEventDestroy(input_copied)); input_copied = nullptr; }
        if (middle_ready) { cleanup(hipEventDestroy(middle_ready)); middle_ready = nullptr; }
        if (cpu_returned) { cleanup(hipEventDestroy(cpu_returned)); cpu_returned = nullptr; }
        if (error) { cleanup(hipHostFree(error)); error = nullptr; }
        if (output) { cleanup(hipHostFree(output)); output = nullptr; }
        if (input) { cleanup(hipHostFree(input)); input = nullptr; }
        if (middle) { cleanup(hipHostFree(middle)); middle = nullptr; }
    }
};
static_assert(HybridDevice::input_bytes == 8640 && HybridDevice::output_bytes == 307200);
static_assert(HybridDevice::gate_up_bytes == 153600 && HybridDevice::middle_float_bytes == 76800 &&
              HybridDevice::middle_bytes == 21600 &&
              HybridDevice::gate_up_bytes + HybridDevice::middle_float_bytes + HybridDevice::middle_bytes + sizeof(int) == 252004);

struct Device {
    int id;
    hipStream_t stream = nullptr;
    hipStream_t copy_stream = nullptr;
    rocblas_handle handle = nullptr;
    std::vector<Buffer> scratch;
    Buffer q8, error, logits, gathered_k, gathered_v, partial_o, partial_ms;
    Buffer histogram, select_state, candidates, ids, block_ids, counts;
    Buffer routed_contributions, route_metadata;
    MoeRouteIndex* route_dto = nullptr;
    int route_tokens = 0, route_entries = 0;
    Buffer ds4, append_packed, append_work;
    std::array<std::unique_ptr<ExpertStage>, 2> stages;
    std::unique_ptr<HybridDevice> hybrid;
    std::unique_ptr<RouteGroups> routes;
    std::vector<float> route_probabilities, route_weights;
    std::vector<std::int32_t> route_ids;
    explicit Device(int d, int max_tokens, bool prepare_hybrid, int attention_tile) : id(d) {
        hip(hipSetDevice(id));
        hipDeviceProp_t p{}; hip(hipGetDeviceProperties(&p, id));
        const std::string arch(p.gcnArchName);
        if (p.warpSize != 64 || !(arch == "gfx906" || arch.starts_with("gfx906:")))
            throw std::runtime_error("Session requires gfx906 wave64");
        hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        try {
            blas(rocblas_create_handle(&handle));
            route_probabilities.resize(checked_size(max_tokens, 512));
            route_weights.resize(checked_size(max_tokens, 10));
            route_ids.resize(checked_size(max_tokens, 10));
            scratch.reserve(24);
            // f(16) also holds 32768 completed-block scores at capacity 128K.
            // N3 QSA projections require 36864 floats. N1 sizes/counts stay exact.
            const auto elements = std::max<std::size_t>(32768, checked_size(max_tokens, 12288));
            for (int i = 0; i < 24; ++i) {
                // Score temporary dies before staged_output f(15); append's
                // f(17) dies before attention output. Only these prefixes grow
                // if tile exceeds the existing logical chunk scratch capacity.
                const auto n = (i == 15 || i == 17) ?
                    std::max(elements, checked_size(attention_tile, attention_query_elements)) :
                    (i == 16 ? std::max(elements, checked_size(std::min(attention_tile, 8), 32768)) : elements);
                scratch.emplace_back(checked_size(n, sizeof(float)));
            }
            q8 = Buffer(checked_size(std::max<std::size_t>(512, checked_size(max_tokens, W / 32)), sizeof(Q8_1)));
            error = Buffer(sizeof(int));
            // The head already projects in qualified logical tiles <=128.
            // Bound device output independently of the full PP frame; host
            // publication still owns every requested vocabulary row.
            logits = Buffer(checked_size(checked_size(std::min(max_tokens, microtile_tokens), V), sizeof(float)));
            gathered_k = Buffer(checked_size(checked_size(attention_tile, attention_gather_elements), sizeof(std::uint16_t)));
            gathered_v = Buffer(checked_size(checked_size(attention_tile, attention_gather_elements), sizeof(std::uint16_t)));
            partial_o = Buffer(checked_size(checked_size(attention_tile, attention_partial_elements), sizeof(float)));
            partial_ms = Buffer(checked_size(checked_size(attention_tile, attention_partial_ml_elements), sizeof(float2)));
            histogram = Buffer(checked_size(checked_size(std::min(attention_tile, 8), 8192), sizeof(int)));
            select_state = Buffer(checked_size(checked_size(std::min(attention_tile, 8), 16), sizeof(std::uint64_t)));
            candidates = Buffer(checked_size(checked_size(std::min(attention_tile, 8), 512), sizeof(std::uint64_t)));
            ids = Buffer(checked_size(checked_size(attention_tile, attention_max_selected), sizeof(std::int32_t)));
            block_ids = Buffer(checked_size(checked_size(attention_tile, 512), sizeof(std::int32_t)));
            counts = Buffer(checked_size(checked_size(attention_tile, 2), sizeof(int)));
            if (max_tokens > 1 || prepare_hybrid) {
                routed_contributions = Buffer(checked_size(checked_size(prepare_hybrid ? std::max(3, max_tokens) : max_tokens, 10 * H), sizeof(float)));
                routes = std::make_unique<RouteGroups>(max_tokens);
                route_metadata = Buffer(checked_size(checked_size(max_tokens, 10), sizeof(MoeRouteIndex)));
                hip(hipHostMalloc(&route_dto, route_metadata.bytes(), hipHostMallocDefault));
                std::memset(route_dto, 0, route_metadata.bytes());
            }
            if (max_tokens > 3) {
                const int bounded = std::min(max_tokens, microtile_tokens);
                ds4 = Buffer(checked_size(checked_size(bounded, W / 128), sizeof(Q8MmqBlock)));
                append_packed = Buffer(microtile_tokens * 4 * sizeof(Q4_0));
                // Roundtrip128*128 + at most33 new blocks*128 + staged tail384.
                append_work = Buffer((microtile_tokens * 128 + 33 * 128 + 384) * sizeof(float));
                hip(hipStreamCreateWithFlags(&copy_stream, hipStreamNonBlocking));
                for (auto& stage : stages) stage = std::make_unique<ExpertStage>(id);
            }
            if (prepare_hybrid) {
                if (!copy_stream) hip(hipStreamCreateWithFlags(&copy_stream, hipStreamNonBlocking));
                if (max_tokens <= 3)
                    for (auto& stage : stages) stage = std::make_unique<ExpertStage>(id, short_stage_bytes);
                hybrid = std::make_unique<HybridDevice>(id);
            }
        } catch (...) {
            drain_cleanup();
            if (route_dto) { cleanup(hipHostFree(route_dto)); route_dto = nullptr; }
            if (handle) { if (rocblas_destroy_handle(handle) != rocblas_status_success) std::abort(); }
            if (copy_stream) { cleanup(hipStreamDestroy(copy_stream)); copy_stream = nullptr; }
            cleanup(hipStreamDestroy(stream)); stream = nullptr; throw;
        }
    }
    ~Device() {
        drain_cleanup();
        if (route_dto) cleanup(hipHostFree(route_dto));
        if (rocblas_destroy_handle(handle) != rocblas_status_success) std::abort();
        if (copy_stream) cleanup(hipStreamDestroy(copy_stream));
        cleanup(hipStreamDestroy(stream));
    }
    void drain() {
        hip(hipSetDevice(id));
        if (copy_stream) hip(hipStreamSynchronize(copy_stream));
        hip(hipStreamSynchronize(stream));
    }
    void drain_cleanup() noexcept {
        cleanup(hipSetDevice(id));
        if (copy_stream) cleanup(hipStreamSynchronize(copy_stream));
        if (stream) cleanup(hipStreamSynchronize(stream));
    }
    float* f(int i) { return scratch.at(static_cast<std::size_t>(i)).get(); }
    int* flag() { return error.get<int>(); }
    void upload_routes(int tokens) {
        if (!routes || !route_dto || tokens < 1 || tokens > moe_copy_max_input_tokens)
            throw std::runtime_error("Session route metadata resources/tokens");
        const auto assignments = routes->assignments();
        if (assignments.size() != static_cast<std::size_t>(tokens) * 10 ||
                assignments.size() > route_metadata.bytes() / sizeof(MoeRouteIndex))
            throw std::runtime_error("Session route metadata extent");
        // Convert the WHOLE prepared, stable expert/token/rank order, never the
        // 16-byte RouteAssignment ABI. One pinned H2D per grouped layer. The
        // prior layer's completed check (or failure drain) precedes this refill;
        // pinned bytes remain immutable through all current layer consumers.
        // This transfer is metadata, not expert weight upload_bytes.
        for (std::size_t i = 0; i < assignments.size(); ++i)
            route_dto[i] = {assignments[i].token, assignments[i].rank};
        route_tokens = tokens; route_entries = static_cast<int>(assignments.size());
        hip(hipMemcpyAsync(route_metadata.get<MoeRouteIndex>(), route_dto,
            assignments.size() * sizeof(MoeRouteIndex), hipMemcpyHostToDevice, stream));
    }
    int route_offset(std::span<const RouteAssignment> group, int begin, int count) const {
        // Every caller borrows group() from this same RouteGroups vector. Its
        // pointer difference is the global metadata offset, including microtiles.
        if (group.empty() || begin < 0 || count < 1 || count > microtile_tokens ||
                static_cast<std::size_t>(begin) > group.size() ||
                static_cast<std::size_t>(count) > group.size() - static_cast<std::size_t>(begin))
            throw std::runtime_error("Session route metadata group tile");
        const auto offset = group.data() - routes->assignments().data() + begin;
        if (offset < 0 || offset > route_entries || count > route_entries - offset)
            throw std::runtime_error("Session route metadata group offset");
        return static_cast<int>(offset);
    }
    void gather_routes(const Q8_1* original, std::span<const RouteAssignment> group, int begin, int count) {
        hip(launch_moe_gather_q8({original, scratch[21].bytes() / sizeof(Q8_1)}, route_tokens,
            {route_metadata.get<MoeRouteIndex>(), route_metadata.bytes() / sizeof(MoeRouteIndex)}, route_entries,
            route_offset(group, begin, count), count, H / 32,
            {reinterpret_cast<Q8_1*>(f(23)), scratch[23].bytes() / sizeof(Q8_1)}, count, flag(), stream));
    }
    void scatter_routes(std::span<const RouteAssignment> group, int begin, int count) {
        hip(launch_moe_scatter({f(19), scratch[19].bytes() / sizeof(float)}, route_tokens,
            {route_metadata.get<MoeRouteIndex>(), route_metadata.bytes() / sizeof(MoeRouteIndex)}, route_entries,
            route_offset(group, begin, count), count, H, 10,
            {routed_contributions.get(), routed_contributions.bytes() / sizeof(float)}, count, flag(), stream));
    }
    void clear() { hip(hipMemsetAsync(flag(), 0, sizeof(int), stream)); }
    void check(const std::string& where) {
        int e = 0; hip(hipMemcpyAsync(&e, flag(), sizeof(e), hipMemcpyDeviceToHost, stream));
        hip(hipStreamSynchronize(stream));
        if (e) throw std::runtime_error(where + ": numeric flag " + std::to_string(e));
    }
};

// R4 donor decision (RECON16): KEEP mx dcd6854 canonical Q4/Q8 short
// projections, band16 copy_ready/consumer_done staging; KEEP furnace 905021d
// chronological GDN/selected-Q4 B8 attention. ADAPT reinstinct 0b79e32's
// token/rank gather/scatter orchestration only: our qualified opaque 36B Q8
// copies now have a separate checked <=16384 entry point. No GEMM/packer/math
// replacement, scheduler import, PP cache admission, or shortened QSA horizon.
struct LayerwiseDevice {
    Buffer original, contributions, metadata, residual;
    explicit LayerwiseDevice(Device& d, int capacity, bool resident) {
        hip(hipSetDevice(d.id));
        original = Buffer(checked_size(checked_size(capacity, H / 32), sizeof(Q8_1)));
        contributions = Buffer(checked_size(checked_size(capacity, 10 * H), sizeof(float)));
        metadata = Buffer(checked_size(checked_size(capacity, 10), sizeof(MoeRouteIndex)));
        if (resident) residual = Buffer(checked_size(checked_size(capacity, W), sizeof(float)));
    }
    static int offset(const RouteGroups& groups, std::span<const RouteAssignment> group,
                      int begin, int count, int tokens) {
        if (tokens < 1 || tokens > moe_layerwise_max_input_tokens || group.empty() ||
            begin < 0 || count < 1 || count > linear_device_tiled_max_columns ||
            static_cast<std::size_t>(begin) > group.size() ||
            static_cast<std::size_t>(count) > group.size() - static_cast<std::size_t>(begin) ||
            groups.assignments().size() != static_cast<std::size_t>(tokens) * 10)
            throw std::runtime_error("Session layerwise route tile extent");
        // Borrowed from this exact owner, as in the original chunk route_offset.
        const auto first = group.data() - groups.assignments().data() + begin;
        const auto entries = static_cast<std::ptrdiff_t>(groups.assignments().size());
        if (first < 0 || first > entries || count > entries - first || first > INT32_MAX)
            throw std::runtime_error("Session layerwise route source offset");
        return static_cast<int>(first);
    }
    void gather(Device& d, const RouteGroups& groups, std::span<const RouteAssignment> group,
                int begin, int count, int tokens) {
        hip(launch_moe_gather_q8_layerwise({original.get<Q8_1>(), original.bytes() / sizeof(Q8_1)}, tokens,
            {metadata.get<MoeRouteIndex>(), metadata.bytes() / sizeof(MoeRouteIndex)}, tokens * 10,
            offset(groups, group, begin, count, tokens), count, H / 32,
            {reinterpret_cast<Q8_1*>(d.f(23)), d.scratch[23].bytes() / sizeof(Q8_1)}, count, d.flag(), d.stream));
    }
    void scatter(Device& d, const RouteGroups& groups, std::span<const RouteAssignment> group,
                 int begin, int count, int tokens) {
        hip(launch_moe_scatter_layerwise({d.f(19), d.scratch[19].bytes() / sizeof(float)}, tokens,
            {metadata.get<MoeRouteIndex>(), metadata.bytes() / sizeof(MoeRouteIndex)}, tokens * 10,
            offset(groups, group, begin, count, tokens), count, H, 10,
            {contributions.get(), contributions.bytes() / sizeof(float)}, count, d.flag(), d.stream));
    }
};

// Mutable pageable and pinned state belongs to one GPU stage. The parent
// drains all borrowing streams before destroying this host owner.
struct LayerwiseHost {
    std::array<std::vector<float>, 2> activations;
    std::vector<float> mixed, injection, probabilities, weights;
    std::vector<std::int32_t> ids;
    RouteGroups groups;
    MoeRouteIndex* dto = nullptr;
    std::size_t dto_bytes;
    explicit LayerwiseHost(int capacity)
        : mixed(checked_size(capacity, H)), injection(checked_size(capacity, 4)),
          probabilities(checked_size(capacity, 512)), weights(checked_size(capacity, 10)),
          ids(checked_size(capacity, 10)), groups(capacity, RouteGroups::Bound::layerwise16384),
          dto_bytes(checked_size(checked_size(capacity, 10), sizeof(MoeRouteIndex))) {
        for (auto& a : activations) a.resize(checked_size(capacity, W));
        hip(hipHostMalloc(&dto, dto_bytes, hipHostMallocDefault));
    }
    ~LayerwiseHost() { if (dto) cleanup(hipHostFree(dto)); }
    LayerwiseHost(const LayerwiseHost&) = delete;
    LayerwiseHost& operator=(const LayerwiseHost&) = delete;
    void upload_routes(Device& d, LayerwiseDevice& gpu, int tokens) {
        const auto assignments = groups.assignments();
        if (assignments.size() != static_cast<std::size_t>(tokens) * 10 ||
            assignments.size() > dto_bytes / sizeof(MoeRouteIndex))
            throw std::runtime_error("Session layerwise metadata extent");
        for (std::size_t i = 0; i < assignments.size(); ++i)
            dto[i] = {assignments[i].token, assignments[i].rank};
        hip(hipMemcpyAsync(gpu.metadata.get<MoeRouteIndex>(), dto,
            assignments.size() * sizeof(MoeRouteIndex), hipMemcpyHostToDevice, d.stream));
    }
};

struct LayerwisePrefill : LayerwiseHost {
    const int stage_capacity;
    std::array<Device*, 2> owners;
    std::array<std::unique_ptr<LayerwiseDevice>, 2> gpu;
    std::unique_ptr<LayerwiseHost> stage1;
    std::array<std::vector<float>, 2> ring;
    float* stage1_transfer = nullptr;
    std::size_t stage1_transfer_bytes = 0;
    void enable_pipeline(int capacity, int frame) {
        stage1 = std::make_unique<LayerwiseHost>(capacity);
        for (auto& slot : ring) slot.resize(checked_size(capacity, W));
        stage1_transfer_bytes = checked_size(checked_size(frame, W), sizeof(float));
        hip(hipHostMalloc(&stage1_transfer, stage1_transfer_bytes, hipHostMallocDefault));
    }
    explicit LayerwisePrefill(std::array<Device*, 2> devices, int capacity, bool resident)
        : LayerwiseHost(capacity), stage_capacity(capacity), owners(devices) {
        try {
            for (int id = 0; id < 2; ++id) gpu[id] = std::make_unique<LayerwiseDevice>(*owners[id], capacity, resident);
        } catch (...) { drain(); throw; }
    }
    ~LayerwisePrefill() {
        drain();
        if (stage1_transfer) cleanup(hipHostFree(stage1_transfer));
    }
    void drain() noexcept {
        for (auto* d : owners) if (d) d->drain_cleanup();
    }
};

Buffer upload(Device& d, std::span<const std::byte> bytes) {
    hip(hipSetDevice(d.id)); Buffer result(bytes.size());
    hip(hipMemcpyAsync(result.get<void>(), bytes.data(), bytes.size(), hipMemcpyHostToDevice, d.stream));
    hip(hipStreamSynchronize(d.stream)); return result;
}
Buffer upload(Device& d, const std::vector<float>& x) { return upload(d, std::as_bytes(std::span(x))); }

struct Matrix {
    TensorType type = TensorType::F32;
    int input = 0, output = 0;
    Buffer data;
    void apply(Device& d, const float* x, float* y, int columns) const {
        // KEEP canonical physical N<=8 arithmetic. Component A/B/A supports
        // tiled quantized projections and small-output dense; large dense stays
        // on the original N8 launches (its tiled-grid case regressed).
        const bool dense = type == TensorType::F32 || type == TensorType::BF16 || type == TensorType::F16;
        for (int first = 0; first < columns;) {
            const int remaining = columns - first;
            const bool tiled = remaining >= 16 && (!dense || output <= 512);
            const int count = std::min(tiled ? linear_device_tiled_max_columns : 8, remaining);
            const auto* source = x + first * input;
            auto* destination = y + first * output;
            if (dense) {
                if (tiled)
                    blas(launch_dense_linear_tiled(d.handle, {data.get(), input, output}, source, count, destination, d.stream));
                else
                    blas(launch_dense_linear(d.handle, {data.get(), input, output}, source, count, destination, d.stream));
            } else {
                hip(launch_quantize_q8_1(source, d.q8.get<Q8_1>(), input, count, d.flag(), d.stream));
                if (count <= 3)
                    hip(launch_quantized_linear({type, data.get<void>(), input, output}, d.q8.get<Q8_1>(), count, destination, d.stream));
                else if (count <= linear_device_short_max_columns)
                    hip(launch_quantized_linear_short({type, data.get<void>(), input, output}, d.q8.get<Q8_1>(), count, destination, d.stream));
                else {
                    hip(launch_quantized_linear_tiled({type, data.get<void>(), input, output},
                        d.q8.get<Q8_1>(), count, destination, d.stream));
                }
            }
            first += count;
        }
    }
};

const TensorView& require(const Model& model, const std::string& name, std::initializer_list<std::uint64_t> dims) {
    const auto& t = model.tensor(name);
    if (t.rank != dims.size() || !std::equal(dims.begin(), dims.end(), t.dimensions.begin()))
        throw std::runtime_error("tensor geometry: " + name);
    return t;
}
Matrix matrix(const Model& model, Device& d, const std::string& name, int input, int output) {
    const auto& t = require(model, name, {static_cast<std::uint64_t>(input), static_cast<std::uint64_t>(output)});
    std::vector<std::byte> bytes(static_cast<std::size_t>(t.byte_size)); model.read_tensor(name, bytes);
    Matrix result; result.type = t.type; result.input = input; result.output = output;
    if (t.type == TensorType::F32 || t.type == TensorType::F16 || t.type == TensorType::BF16) {
        std::vector<float> decoded(static_cast<std::size_t>(input) * output);
        for (std::size_t i = 0; i < decoded.size(); ++i) {
            decoded[i] = tensor_element(t.type, bytes, i);
            if (!std::isfinite(decoded[i])) throw std::runtime_error("nonfinite weight: " + name);
        }
        result.data = upload(d, decoded);
    } else result.data = upload(d, bytes);
    return result;
}
Buffer parameter(const Model& model, Device& d, const std::string& name, std::initializer_list<std::uint64_t> dims) {
    const auto& t = require(model, name, dims);
    if (t.type != TensorType::F32 && t.type != TensorType::F16) throw std::runtime_error("parameter dtype: " + name);
    std::vector<std::byte> bytes(static_cast<std::size_t>(t.byte_size)); model.read_tensor(name, bytes);
    std::vector<float> x(static_cast<std::size_t>(t.elements));
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = tensor_element(t.type, bytes, i);
        if (!std::isfinite(x[i])) throw std::runtime_error("nonfinite parameter: " + name);
    }
    return upload(d, x);
}

struct Hc {
    Matrix down, up, inject;
    Buffer norm;
    Hc(const Model& m, Device& d, const std::string& prefix, bool head = false)
        : down(matrix(m, d, prefix + "down.weight", W, 320)),
          up(matrix(m, d, prefix + "up.weight", 320, W)),
          norm(parameter(m, d, prefix + "norm.weight", {W})) {
        if (!head) inject = matrix(m, d, prefix + "inject.weight", W, 4);
    }
    void mix(Device& d, const float* residual, float epsilon, int tokens, bool head = false) const {
        tiles(tokens, [&](int first, int count) {
            hip(launch_group_rms_norm(residual + first * W, norm.get(), count, W, H, epsilon, false,
                d.f(2) + first * W, d.flag(), d.stream));
        });
        down.apply(d, d.f(2), d.f(3), tokens);
        tiles(tokens, [&](int first, int count) {
            hip(launch_hc_lowrank_activation(d.f(3) + first * 320, count, d.f(4) + first * 320, d.flag(), d.stream));
        });
        up.apply(d, d.f(4), d.f(5), tokens);
        tiles(tokens, [&](int first, int count) {
            hip(launch_hc_mix(d.f(2) + first * W, d.f(5) + first * W, count, d.f(6) + first * H, d.flag(), d.stream));
        });
        if (!head) {
            inject.apply(d, d.f(2), d.f(7), tokens);
            tiles(tokens, [&](int first, int count) {
                hip(launch_hc_injection(d.f(7) + first * 4, count, d.f(8) + first * 4, d.flag(), d.stream));
            });
        }
    }
};

void combine(Device& d, const float* residual, const float* block, int tokens, float* output) {
    tiles(tokens, [&](int first, int count) {
        hip(launch_hc_combine(residual + first * W, block + first * H, d.f(8) + first * 4,
            count, output + first * W, d.flag(), d.stream));
    });
}

// Extracted unchanged from Layer::ffn: identical ID tie-breaks, bucket order,
// ascending XOR reduction, floor, reciprocal and separately rounded multiply.
// Shared by old chunks and new global routing; no extra normalization.
void select_routes(const float* probabilities, std::int32_t* ids, float* weights, int tokens) {
    for (int t = 0; t < tokens; ++t) {
        const float* row = probabilities + t * 512;
        std::array<int, 512> order{}; std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 10, order.end(), [&](int i, int j) {
            return row[i] > row[j] || (row[i] == row[j] && i < j);
        });
        std::array<float, 32> buckets{};
        for (int i = 0; i < 10; ++i) buckets[order[i] % 32] += row[order[i]];
        for (int distance = 1; distance < 32; distance *= 2) {
            const auto previous = buckets;
            for (int lane = 0; lane < 32; ++lane)
                buckets[lane] = previous[lane] + previous[lane ^ distance];
        }
        const float denominator = std::max(buckets[0], 1.0f / 16384.0f);
        if (!(denominator > 0) || !std::isfinite(denominator)) throw std::runtime_error("invalid route denominator");
        const float inverse = 1.0f / denominator;
        for (int i = 0; i < 10; ++i) {
            ids[t * 10 + i] = order[i]; weights[t * 10 + i] = row[order[i]] * inverse;
        }
    }
}

struct Experts {
    struct Slot { int id = -1; std::uint64_t stamp = 0; };
    TensorType down_type;
    std::array<std::vector<std::byte>, 3> ram;
    std::array<std::size_t, 3> bytes{};
    std::vector<CpuExpert> cpu_views; // Immutable borrowed RAM, constructor only.
    Buffer slots;
    std::vector<Slot> map;
    std::vector<int> pending; // Opt-in only; LOADING is invisible to resident().
    std::uint64_t clock = 0;
    std::uint64_t payload_reads = 0, payload_bytes_read = 0;
    Experts(const Model& m, Device& d, const std::string& prefix, int nslots, bool prepare_cpu) : down_type(TensorType::Q4_0) {
        const std::array<std::string, 3> names{prefix + "ffn_gate_exps.weight", prefix + "ffn_up_exps.weight", prefix + "ffn_down_exps.weight"};
        for (int i = 0; i < 3; ++i) {
            const auto& t = require(m, names[i], {static_cast<std::uint64_t>(i == 2 ? 640 : H),
                                               static_cast<std::uint64_t>(i == 2 ? H : 640), 512});
            if (i < 2 && t.type != TensorType::Q4_0) throw std::runtime_error("expert gate/up dtype");
            if (i == 2) {
                down_type = t.type;
                if (down_type != TensorType::Q4_0 && down_type != TensorType::Q4_1) throw std::runtime_error("expert down dtype");
            }
            bytes[i] = static_cast<std::size_t>(t.strides[2]);
            const auto layout = type_layout(t.type);
            const auto expected = checked_size(H * 640 / layout.block_elements, layout.block_bytes);
            if (t.strides[2] != expected || t.byte_size != checked_size(expected, 512))
                throw std::runtime_error("expert canonical span: " + names[i]);
            ram[i].resize(static_cast<std::size_t>(t.byte_size)); m.read_tensor(names[i], ram[i]);
            ++payload_reads; payload_bytes_read += ram[i].size();
        }
        if (checked_size(total(), expert_band) > expert_stage_bytes) throw std::runtime_error("expert staging capacity");
        if (prepare_cpu) {
            cpu_views.reserve(512);
            for (int id = 0; id < 512; ++id)
                cpu_views.emplace_back(QMatrix{TensorType::Q4_0, H, 640, std::span(ram[0]).subspan(bytes[0] * id, bytes[0])},
                    QMatrix{TensorType::Q4_0, H, 640, std::span(ram[1]).subspan(bytes[1] * id, bytes[1])},
                    QMatrix{down_type, 640, H, std::span(ram[2]).subspan(bytes[2] * id, bytes[2])});
        }
        hip(hipSetDevice(d.id)); slots = Buffer(checked_size(total(), nslots)); map.resize(nslots);
        if (prepare_cpu) pending.assign(nslots, -1);
    }
    std::size_t total() const { return bytes[0] + bytes[1] + bytes[2]; }
    int resident(int id) const {
        for (std::size_t i = 0; i < map.size(); ++i) if (map[i].id == id) return static_cast<int>(i);
        return -1;
    }
    void finish_admissions(bool success) noexcept {
        for (std::size_t i = 0; i < pending.size(); ++i) if (pending[i] >= 0) {
            map[i].id = success ? pending[i] : -1;
            pending[i] = -1;
        }
    }
    void invalidate_cache() noexcept {
        // Even the disabled acquire() can fail midway through a triplet upload
        // before replacing the old map ID. Only a completed successful reset
        // may retain READY maps; execution failure conservatively frees ALL IDs.
        for (auto& slot : map) slot.id = -1;
        std::fill(pending.begin(), pending.end(), -1);
    }
    void gpu_group(Device& d, const HybridDevice::Plan& plan, const std::byte* base, const Q8_1* original) {
        const auto group = d.routes->group(plan.expert);
        auto* gathered = reinterpret_cast<Q8_1*>(d.f(23));
        const int columns = static_cast<int>(group.size());
        d.gather_routes(original, group, 0, columns);
        project(d, base, gathered, columns);
        d.scatter_routes(group, 0, columns);
    }
    void run_hybrid(Device& d, CpuExpertPool& pool, SessionHybridPolicy policy, SessionStats& stats,
                    SessionHybridStats& diagnostics, int tokens, const Q8_1* original, bool verify_input,
                    SessionHybridProbeDiagnostics* failure_probe, std::uint64_t cpu_return_baseline, int layer) {
        auto& frame = *d.hybrid;
        frame.count = 0; frame.cpu_columns = 0; frame.cpu_active = false;
        ++diagnostics.short_layers;
        if (policy.mode == SessionHybridMode::force_cpu) ++diagnostics.forced_cpu_layers;
        if (policy.mode == SessionHybridMode::force_gpu_misses) ++diagnostics.forced_gpu_layers;
        // Snapshot EVERY READY physical reader before any admission/slot write.
        bool all_hit = true;
        for (int id = 0; id < 512; ++id) {
            const auto group = d.routes->group(id);
            if (group.empty()) continue;
            if (frame.count >= static_cast<int>(frame.plans.size())) throw std::runtime_error("hybrid plan capacity");
            auto& plan = frame.plans[frame.count++]; plan = {}; plan.expert = id;
            const int slot = resident(id);
            const auto columns = static_cast<std::uint64_t>(group.size());
            diagnostics.group_reuse_assignments += columns - 1;
            if (slot >= 0) {
                plan.hit = slots.get<std::byte>() + total() * slot;
                map[slot].stamp = ++clock;
                diagnostics.ready_hit_assignments += columns; stats.expert_hits += columns;
            } else {
                all_hit = false; diagnostics.physical_miss_assignments += columns;
                // Preserve historical first-acquisition + within-window reuse
                // accounting independently of the forced/mixed dispatch below.
                ++stats.expert_misses; stats.expert_hits += columns - 1;
            }
            plan.work = policy.mode == SessionHybridMode::force_cpu ? HybridDevice::Work::cpu :
                policy.mode == SessionHybridMode::force_gpu_misses ? HybridDevice::Work::gpu_miss :
                slot >= 0 ? HybridDevice::Work::hit : HybridDevice::Work::cpu;
        }
        if (all_hit) ++diagnostics.all_hit_layers;
        if (policy.mode == SessionHybridMode::mixed) {
            // Known current-window reuse count, ties by expert ID. No prediction.
            for (int i = 0; i < policy.gpu_miss_groups; ++i) {
                HybridDevice::Plan* best = nullptr;
                for (int j = 0; j < frame.count; ++j) {
                    auto& plan = frame.plans[j];
                    if (plan.hit || plan.work != HybridDevice::Work::cpu) continue;
                    if (!best || d.routes->group(plan.expert).size() > d.routes->group(best->expert).size()) best = &plan;
                }
                if (best) best->work = HybridDevice::Work::gpu_miss;
            }
        }
        std::size_t output_columns = 0;
        for (int j = 0; j < frame.count; ++j) {
            auto& plan = frame.plans[j];
            const auto group = d.routes->group(plan.expert);
            const auto columns = group.size();
            if (plan.work == HybridDevice::Work::cpu) {
                plan.output_column = output_columns; output_columns += columns; frame.cpu_active = true;
                ++diagnostics.cpu_groups; diagnostics.cpu_assignments += columns;
            } else if (plan.work == HybridDevice::Work::hit) {
                ++diagnostics.gpu_hit_groups; diagnostics.gpu_hit_assignments += columns;
            } else { ++diagnostics.gpu_miss_groups; diagnostics.gpu_miss_assignments += columns; }
        }
        if (output_columns > 30) throw std::runtime_error("hybrid result capacity");
        frame.cpu_columns = static_cast<int>(output_columns);
        if (frame.cpu_active) {
            cpu_ok(pool.reset());
            // compute owns f21 and the sticky quant flag. A copy-stream D2H
            // without this dependency could race quantization and/or later jobs.
            hip(hipMemcpyAsync(frame.input, original, tokens * (H / 32) * sizeof(Q8_1), hipMemcpyDeviceToHost, d.stream));
            if (verify_input)
                hip(hipMemcpyAsync(frame.original_probe.data(), original, tokens * (H / 32) * sizeof(Q8_1), hipMemcpyDeviceToHost, d.stream));
            hip(hipMemcpyAsync(frame.error, d.flag(), sizeof(int), hipMemcpyDeviceToHost, d.stream));
            hip(hipEventRecord(frame.input_copied, d.stream));
            ++diagnostics.input_extractions; diagnostics.input_bytes += tokens * (H / 32) * sizeof(Q8_1);
        }
        // Captured hit readers and scatters ALL precede permanent-slot writes.
        for (int j = 0; j < frame.count; ++j)
            if (frame.plans[j].work == HybridDevice::Work::hit)
                gpu_group(d, frame.plans[j], frame.plans[j].hit, original);
        int accepted_jobs = 0;
        if (frame.cpu_active) {
            hip(hipEventSynchronize(frame.input_copied));
            if (*frame.error) throw std::runtime_error("hybrid original input numeric flag");
            for (int j = 0; j < frame.count; ++j) {
                auto& plan = frame.plans[j];
                if (plan.work != HybridDevice::Work::cpu) continue;
                const auto group = d.routes->group(plan.expert);
                plan.job.expert = &cpu_views[plan.expert]; plan.job.columns = static_cast<int>(group.size());
                plan.job.stage = CpuExpertStage::gate_up;
                for (std::size_t c = 0; c < group.size(); ++c)
                    plan.job.input[c] = {frame.input + group[c].token * (H / 32), H / 32};
                // TEMP paired projections use only the packed prefix of the
                // existing 307200B pinned down frame. Not H-strided yet.
                plan.job.output = {frame.output + plan.output_column * (2 * 640), group.size() * (2 * 640)};
                if (verify_input) for (std::size_t c = 0; c < group.size(); ++c) {
                    const auto& dto = d.route_dto[d.route_offset(group, 0, static_cast<int>(group.size())) + c];
                    if (dto.token != group[c].token || dto.rank != group[c].rank)
                        throw std::runtime_error("hybrid CPU descriptor differs from GPU route metadata");
                    if (std::memcmp(plan.job.input[c].data(), frame.original_probe.data() + dto.token * (H / 32),
                            (H / 32) * sizeof(Q8_1))) throw std::runtime_error("hybrid CPU input differs from original GPU Q8 bytes");
                    diagnostics.cpu_input_bytes_checked += (H / 32) * sizeof(Q8_1);
                }
                cpu_ok(pool.submit(plan.job, plan.ticket));
                ++accepted_jobs; ++diagnostics.cpu_gate_up_jobs;
            }
        }
        int band = 0, admitted = 0;
        for (int j = 0; j < frame.count; ++j) {
            const auto& plan = frame.plans[j];
            if (plan.work != HybridDevice::Work::gpu_miss) continue;
            auto& stage = *d.stages[band++ % 2];
            if (stage.used) {
                hip(hipEventSynchronize(stage.copy_ready));
                hip(hipStreamWaitEvent(d.copy_stream, stage.consumer_done, 0));
            }
            std::size_t offset = 0;
            for (int part = 0; part < 3; ++part) {
                std::memcpy(stage.host + offset, ram[part].data() + bytes[part] * plan.expert, bytes[part]);
                hip(hipMemcpyAsync(stage.data.get<std::byte>() + offset, stage.host + offset, bytes[part], hipMemcpyHostToDevice, d.copy_stream));
                offset += bytes[part];
            }
            stats.expert_upload_bytes += total();
            hip(hipEventRecord(stage.copy_ready, d.copy_stream));
            hip(hipStreamWaitEvent(d.stream, stage.copy_ready, 0));
            gpu_group(d, plan, stage.data.get<std::byte>(), original);
            if (admitted < policy.gpu_miss_groups && !plan.hit) {
                auto chosen = std::min_element(map.begin(), map.end(), [](const auto& a, const auto& b) { return a.stamp < b.stamp; });
                if (chosen->id >= 0) ++diagnostics.evicted_ready_slots;
                const auto slot = static_cast<std::size_t>(chosen - map.begin());
                chosen->id = -1; pending[slot] = plan.expert; chosen->stamp = ++clock;
                // After ALL captured readers; compute owns D2D admission too.
                hip(hipMemcpyAsync(slots.get<std::byte>() + slot * total(), stage.data.get<std::byte>(), total(), hipMemcpyDeviceToDevice, d.stream));
                ++admitted; ++diagnostics.admitted_groups;
            }
            hip(hipEventRecord(stage.consumer_done, d.stream)); // Includes admission's read.
            stage.used = true;
        }
        if (policy.diagnostic_fail_after_admission && failure_probe && !failure_probe->synthetic_failure &&
                accepted_jobs > 0 && admitted > 0) {
            const auto pending_slots = static_cast<std::uint64_t>(std::count_if(pending.begin(), pending.end(),
                [](int id) { return id >= 0; }));
            if (!pending_slots) throw std::runtime_error("hybrid diagnostic admission invariant");
            failure_probe->accepted_cpu_jobs = static_cast<std::uint64_t>(accepted_jobs);
            failure_probe->queued_admission_copies = static_cast<std::uint64_t>(admitted);
            failure_probe->pending_slots_at_failure = pending_slots;
            // No join_hybrid(), GPU middle or CPU-result H2D has run here. Gate/up jobs
            // may already have completed: this is NOT an in-flight timing proof.
            failure_probe->cpu_return_bytes_before_failure = diagnostics.cpu_return_bytes - cpu_return_baseline;
            failure_probe->failure_layer = layer;
            failure_probe->synthetic_failure = true;
            throw std::runtime_error("synthetic hybrid failure after CPU acceptance and GPU admission enqueue");
        }
    }
    void join_hybrid(Device& d, CpuExpertPool& pool, SessionHybridStats& diagnostics) {
        auto& frame = *d.hybrid;
        if (!frame.cpu_active) return; // All-hit has no extraction/pool/events.
        cpu_ok(pool.drain()); // Gate ALL gate/up jobs before the single middle H2D.
        const int columns = frame.cpu_columns;
        if (columns < 1 || columns > silu_pairs_max_columns) throw std::runtime_error("hybrid GPU middle capacity");
        const auto paired_bytes = static_cast<std::size_t>(columns) * (2 * 640) * sizeof(float);
        const auto middle_bytes = static_cast<std::size_t>(columns) * 20 * sizeof(Q8_1);
        // Dedicated buffers/flag on copy_stream: GPU hit/miss/shared compute can
        // overlap safely. NEVER borrow d.q8/f16/f19/d.flag, all live on compute.
        hip(hipMemsetAsync(frame.middle_error.get<int>(), 0, sizeof(int), d.copy_stream));
        hip(hipMemcpyAsync(frame.gate_up.get(), frame.output, paired_bytes, hipMemcpyHostToDevice, d.copy_stream));
        diagnostics.paired_gate_up_bytes += paired_bytes;
        hip(launch_silu_up_pairs(frame.gate_up.get(), columns, frame.middle_float.get(), frame.middle_error.get<int>(), d.copy_stream));
        hip(launch_quantize_q8_1(frame.middle_float.get(), frame.middle_q8.get<Q8_1>(), 640, columns,
            frame.middle_error.get<int>(), d.copy_stream));
        hip(hipMemcpyAsync(frame.middle, frame.middle_q8.get<Q8_1>(), middle_bytes, hipMemcpyDeviceToHost, d.copy_stream));
        // Original input flag was consumed after input_copied; reuse these 4B
        // pinned bytes only in this ordered chain. No extra pinned error frame.
        hip(hipMemcpyAsync(frame.error, frame.middle_error.get<int>(), sizeof(int), hipMemcpyDeviceToHost, d.copy_stream));
        hip(hipEventRecord(frame.middle_ready, d.copy_stream));
        hip(hipEventSynchronize(frame.middle_ready));
        if (*frame.error) throw std::runtime_error("hybrid GPU middle numeric flag");
        ++diagnostics.gpu_middle_batches; diagnostics.gpu_middle_columns += columns;
        diagnostics.middle_q8_bytes += middle_bytes;
        // This completion also protects the TEMP gate/up pinned H2D source.
        // Reset retires the first phase's ranges/tickets before reusing output.
        cpu_ok(pool.reset());
        for (int j = 0; j < frame.count; ++j) {
            auto& plan = frame.plans[j];
            if (plan.work != HybridDevice::Work::cpu) continue;
            const auto group = d.routes->group(plan.expert);
            plan.job = {}; plan.job.expert = &cpu_views[plan.expert];
            plan.job.stage = CpuExpertStage::down; plan.job.columns = static_cast<int>(group.size());
            for (std::size_t c = 0; c < group.size(); ++c)
                plan.job.input[c] = {frame.middle + (plan.output_column + c) * 20, 20};
            plan.job.output = {frame.output + plan.output_column * H, group.size() * H};
            cpu_ok(pool.submit(plan.job, plan.ticket)); ++diagnostics.cpu_down_jobs;
        }
        cpu_ok(pool.drain()); // Gate ALL down jobs before publishing even one CPU row.
        for (int j = 0; j < frame.count; ++j) {
            const auto& plan = frame.plans[j];
            if (plan.work != HybridDevice::Work::cpu) continue;
            const auto group = d.routes->group(plan.expert);
            for (std::size_t c = 0; c < group.size(); ++c) {
                hip(hipMemcpyAsync(d.routed_contributions.get() + (group[c].token * 10 + group[c].rank) * H,
                    frame.output + (plan.output_column + c) * H, H * sizeof(float), hipMemcpyHostToDevice, d.copy_stream));
                diagnostics.cpu_return_bytes += H * sizeof(float);
            }
        }
        hip(hipEventRecord(frame.cpu_returned, d.copy_stream));
        hip(hipStreamWaitEvent(d.stream, frame.cpu_returned, 0));
        // Host output is retained until layer completion, not reused at enqueue.
    }
    std::byte* acquire(Device& d, int id, int columns, SessionStats& stats) {
        if (id < 0 || id >= 512) throw std::runtime_error("expert ID");
        int chosen = -1;
        for (std::size_t i = 0; i < map.size(); ++i) if (map[i].id == id) { chosen = static_cast<int>(i); break; }
        if (chosen < 0) {
            chosen = static_cast<int>(std::min_element(map.begin(), map.end(), [](auto a, auto b) { return a.stamp < b.stamp; }) - map.begin());
            auto* destination = slots.get<std::byte>() + total() * chosen;
            for (int j = 0; j < 3; ++j) {
                hip(hipMemcpyAsync(destination, ram[j].data() + bytes[j] * id, bytes[j], hipMemcpyHostToDevice, d.stream));
                destination += bytes[j];
            }
            map[chosen].id = id; ++stats.expert_misses; stats.expert_upload_bytes += total();
            stats.expert_hits += static_cast<std::uint64_t>(columns - 1);
        } else stats.expert_hits += static_cast<std::uint64_t>(columns);
        map[chosen].stamp = ++clock;
        // Upload, prior readers, reuse and new readers are on the SAME stream.
        // Eviction cannot overwrite a slot before any previous down has finished.
        return slots.get<std::byte>() + total() * chosen;
    }
    void project(Device& d, const std::byte* base, const Q8_1* input, int columns) {
        // KEEP qualified canonical Q4xQ8 Columns1/2/3: the same weight registers
        // feed all columns. No padding/repack or per-token projection fallback.
        if (columns <= 3) {
            hip(launch_quantized_linear({TensorType::Q4_0, base, H, 640}, input, columns, d.f(16), d.stream));
            hip(launch_quantized_linear({TensorType::Q4_0, base + bytes[0], H, 640}, input, columns, d.f(17), d.stream));
        } else if (columns <= linear_device_short_max_columns) {
            hip(launch_quantized_linear_short({TensorType::Q4_0, base, H, 640}, input, columns, d.f(16), d.stream));
            hip(launch_quantized_linear_short({TensorType::Q4_0, base + bytes[0], H, 640}, input, columns, d.f(17), d.stream));
        } else {
            hip(launch_quantized_linear_tiled({TensorType::Q4_0, base, H, 640}, input, columns, d.f(16), d.stream));
            hip(launch_quantized_linear_tiled({TensorType::Q4_0, base + bytes[0], H, 640}, input, columns, d.f(17), d.stream));
        }
        hip(launch_silu_up_q8_1(d.f(16), d.f(17), 640, columns,
            d.f(18), d.q8.get<Q8_1>(), d.flag(), d.stream));
        if (columns <= 3)
            hip(launch_quantized_linear({down_type, base + bytes[0] + bytes[1], 640, H}, d.q8.get<Q8_1>(), columns, d.f(19), d.stream));
        else if (columns <= linear_device_short_max_columns)
            hip(launch_quantized_linear_short({down_type, base + bytes[0] + bytes[1], 640, H}, d.q8.get<Q8_1>(), columns, d.f(19), d.stream));
        else {
            hip(launch_quantized_linear_tiled({down_type, base + bytes[0] + bytes[1], 640, H},
                d.q8.get<Q8_1>(), columns, d.f(19), d.stream));
        }
    }
    void run(Device& d, int id, const Q8_1* input, float weight, SessionStats& stats) {
        project(d, acquire(d, id, 1, stats), input, 1);
        hip(launch_scaled_add(d.f(19), weight, H, d.f(20), d.flag(), d.stream));
    }
    void run_group(Device& d, int id, const Q8_1* input, int columns, SessionStats& stats) {
        project(d, acquire(d, id, columns, stats), input, columns);
    }
    void stage_band(Device& d, ExpertStage& stage, std::span<const int> ids, SessionStats& stats,
                    const RouteGroups& groups) const {
        if (ids.empty() || ids.size() > expert_band) throw std::runtime_error("expert band size");
        if (stage.used) {
            // CPU must not refill pinned bytes until the prior DMA read finishes.
            hip(hipEventSynchronize(stage.copy_ready));
            // GPU destination is reusable only after EVERY group microtile/scatter.
            hip(hipStreamWaitEvent(d.copy_stream, stage.consumer_done, 0));
        }
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const int expert = ids[i], slot = resident(expert);
            const auto reuse = static_cast<std::uint64_t>(groups.group(expert).size());
            std::size_t offset = i * total();
            std::size_t source_offset = slot < 0 ? 0 : static_cast<std::size_t>(slot) * total();
            for (int j = 0; j < 3; ++j) {
                auto* destination = stage.data.get<std::byte>() + offset;
                if (slot >= 0) {
                    const auto* source = slots.get<std::byte>() + source_offset;
                    hip(hipMemcpyAsync(destination, source, bytes[j], hipMemcpyDeviceToDevice, d.copy_stream));
                } else {
                    // Exact validated RAM spans only; never copy slot/K padding.
                    std::memcpy(stage.host + offset, ram[j].data() + bytes[j] * expert, bytes[j]);
                    hip(hipMemcpyAsync(destination, stage.host + offset, bytes[j], hipMemcpyHostToDevice, d.copy_stream));
                }
                offset += bytes[j];
                source_offset += bytes[j];
            }
            if (slot >= 0) stats.expert_hits += reuse;
            else { ++stats.expert_misses; stats.expert_hits += reuse - 1; stats.expert_upload_bytes += total(); }
        }
        // PP makes no admissions/LRU edits. Router's completed check ordered all
        // earlier source writers; this serial owner cannot evict any source slot
        // until the consuming compute chain has waited on this D2D completion.
        hip(hipEventRecord(stage.copy_ready, d.copy_stream));
        hip(hipStreamWaitEvent(d.stream, stage.copy_ready, 0));
    }
    void run_chunk(Device& d, const Q8_1* original, SessionStats& stats,
                   LayerwiseDevice* full = nullptr, const RouteGroups* global = nullptr, int tokens = 0) {
        const auto& groups = global ? *global : *d.routes;
        static_assert(microtile_tokens == linear_device_tiled_max_columns);
        std::array<int, 512> active{};
        int count = 0;
        for (int expert = 0; expert < 512; ++expert) if (!groups.group(expert).empty()) active[count++] = expert;
        auto* gathered = reinterpret_cast<Q8_1*>(d.f(23));
        for (int first = 0, band = 0; first < count; first += expert_band, ++band) {
            auto& stage = *d.stages[band % 2];
            const int length = std::min(expert_band, count - first);
            stage_band(d, stage, std::span(active).subspan(first, length), stats, groups);
            for (int i = 0; i < length; ++i) {
                const auto group = groups.group(active[first + i]);
                for (int begin = 0; begin < static_cast<int>(group.size());) {
                    const int remaining = static_cast<int>(group.size()) - begin;
                    const int columns = std::min(remaining >= 16 ? microtile_tokens : linear_device_short_max_columns, remaining);
                    if (full) full->gather(d, groups, group, begin, columns, tokens);
                    else d.gather_routes(original, group, begin, columns);
                    project(d, stage.data.get<std::byte>() + i * total(), gathered, columns);
                    if (full) full->scatter(d, groups, group, begin, columns, tokens);
                    else d.scatter_routes(group, begin, columns);
                    begin += columns;
                }
            }
            hip(hipEventRecord(stage.consumer_done, d.stream));
            stage.used = true;
        }
    }
};

// No instance on the default path; all arrays live inside the optional owner.
struct TargetLayerPrefixes {
    Buffer recurrent, history, tail;
    std::uint64_t* calls = nullptr; // Optional owner's provisional prefix-API enqueue count.
};

struct Layer {
    int index;
    Device& device;
    Hc attention_hc, ffn_hc;
    Matrix router, shared_gate, shared_up, shared_down;
    Buffer shared_weight;
    Experts experts;
    Matrix qkv, z, alpha, beta, attn_output, key, value, index_q, index_k;
    Buffer conv, dt, a, gdn_norm, recurrent, history;
    Buffer q_norm, k_norm, iq_norm, ik_norm, frequencies, kv_k, kv_v, pooled, tail;
    Buffer stage_recurrent, stage_history;
    bool failure_probe = false;
    bool state_probe = false;
    bool ffn_probe = false;
    Layer(const Model& m, Device& d, int l, const SessionConfig& cfg, float rope_base)
        : index(l), device(d), attention_hc(m, d, "blk." + std::to_string(l) + ".hc_attn_"),
          ffn_hc(m, d, "blk." + std::to_string(l) + ".hc_ffn_"),
          router(matrix(m, d, "blk." + std::to_string(l) + ".ffn_gate_inp.weight", H, 512)),
          shared_gate(matrix(m, d, "blk." + std::to_string(l) + ".ffn_gate_shexp.weight", H, 640)),
          shared_up(matrix(m, d, "blk." + std::to_string(l) + ".ffn_up_shexp.weight", H, 640)),
          shared_down(matrix(m, d, "blk." + std::to_string(l) + ".ffn_down_shexp.weight", 640, H)),
          shared_weight(parameter(m, d, "blk." + std::to_string(l) + ".ffn_gate_inp_shexp.weight", {H})),
          experts(m, d, "blk." + std::to_string(l) + ".", cfg.expert_slots, cfg.cpu_workers > 0) {
        const std::string p = "blk." + std::to_string(l) + ".";
        failure_probe = !cfg.trace_directory.empty() && cfg.trace_first_token > 0 && l == 8;
        state_probe = (!cfg.trace_directory.empty() && l == 22) || failure_probe;
        ffn_probe = !cfg.trace_directory.empty() && l == 0;
        hip(hipSetDevice(d.id));
        if ((l + 1) % 4) {
            qkv = matrix(m, d, p + "attn_qkv.weight", H, W);
            z = matrix(m, d, p + "attn_gate.weight", H, 6144);
            alpha = matrix(m, d, p + "ssm_alpha.weight", H, 48);
            beta = matrix(m, d, p + "ssm_beta.weight", H, 48);
            attn_output = matrix(m, d, p + "ssm_out.weight", 6144, H);
            conv = parameter(m, d, p + "ssm_conv1d.weight", {4, W});
            dt = parameter(m, d, p + "ssm_dt.bias", {48}); a = parameter(m, d, p + "ssm_a", {48});
            gdn_norm = parameter(m, d, p + "ssm_norm.weight", {128});
            recurrent = Buffer(786432 * sizeof(float)); history = Buffer(30720 * sizeof(float));
            stage_recurrent = Buffer(786432 * sizeof(float)); stage_history = Buffer(30720 * sizeof(float));
        } else {
            qkv = matrix(m, d, p + "attn_q.weight", H, 12288);
            key = matrix(m, d, p + "attn_k.weight", H, 512); value = matrix(m, d, p + "attn_v.weight", H, 512);
            attn_output = matrix(m, d, p + "attn_output.weight", 6144, H);
            index_q = matrix(m, d, p + "indexer.q_proj.weight", H, 512);
            index_k = matrix(m, d, p + "indexer.k_proj.weight", H, 128);
            q_norm = parameter(m, d, p + "attn_q_norm.weight", {256}); k_norm = parameter(m, d, p + "attn_k_norm.weight", {256});
            iq_norm = parameter(m, d, p + "indexer.q_norm.weight", {128}); ik_norm = parameter(m, d, p + "indexer.k_norm.weight", {128});
            frequencies = Buffer(32 * sizeof(float));
            d.clear();
            hip(launch_rope_frequencies(rope_base, frequencies.get(), d.flag(), d.stream));
            d.check("rotary frequency table");
            kv_k = Buffer(static_cast<std::size_t>(cfg.capacity) * 16 * sizeof(Q4_0));
            kv_v = Buffer(static_cast<std::size_t>(cfg.capacity) * 16 * sizeof(Q4_0));
            pooled = Buffer(static_cast<std::size_t>(cfg.capacity / 4) * 128 * sizeof(float)); tail = Buffer(384 * sizeof(float));
        }
    }
    void reset() {
        auto& d = device; hip(hipSetDevice(d.id));
        for (auto* b : {&recurrent, &history, &tail}) if (b->bytes()) hip(hipMemsetAsync(b->get<void>(), 0, b->bytes(), d.stream));
    }
    // ADAPT the existing furnace N1 chunk/merge API, pins/layout in attention.hip;
    // KEEP canonical index score/select, Q4 cache and chronological postprocessing.
    // No new attention arithmetic. All append work completes before this chain.
    template<class Observe> void attention_queries(QsaIndexDeviceParameters p, int position, int capacity,
        int tokens, SessionAttentionStats& calls, Observe&& observe) {
        auto& d = device;
        const int tile = static_cast<int>(d.counts.bytes() / (2 * sizeof(int)));
        for (int first = 0; first < tokens; first += tile) {
            const int count = std::min(tile, tokens - first);
            // Keep each score arithmetic/causal prefix; subgroup only the
            // selector (<=8 rows), without narrowing the attention tile<=128.
            // Previous subgroup selection completes before f(16) is reused.
            for (int selected = 0; selected < count; selected += 8) {
                const int rows = std::min(8, count - selected);
                for (int q = 0; q < rows; ++q) {
                    const int t = first + selected + q, visible = position + t + 1;
                    hip(launch_qsa_index_score(p, d.f(14) + t * 512, pooled.get(), visible,
                        d.f(15), d.f(16) + static_cast<std::size_t>(q) * 32768, d.flag(), d.stream));
                }
                const auto id_offset = static_cast<std::size_t>(selected) * attention_max_selected;
                const auto block_offset = static_cast<std::size_t>(selected) * 512;
                const auto count_offset = static_cast<std::size_t>(selected) * 2;
                if (rows == 1) {
                    hip(launch_qsa_select(d.f(16), position + first + selected + 1,
                        d.ids.get<std::int32_t>() + id_offset, d.block_ids.get<std::int32_t>() + block_offset,
                        d.counts.get<int>() + count_offset,
                        {d.histogram.get<int>(), d.select_state.get<std::uint64_t>(), d.candidates.get<std::uint64_t>(), d.flag()}, d.stream));
                } else {
                    hip(launch_qsa_select_batch(
                        {d.f(16), d.scratch[16].bytes() / sizeof(float), 32768}, rows, position + first + selected + 1,
                        {d.ids.get<std::int32_t>() + id_offset, d.ids.bytes() / sizeof(std::int32_t) - id_offset, attention_max_selected},
                        {d.block_ids.get<std::int32_t>() + block_offset, d.block_ids.bytes() / sizeof(std::int32_t) - block_offset, 512},
                        {d.counts.get<int>() + count_offset, d.counts.bytes() / sizeof(int) - count_offset, 2},
                        {{d.histogram.get<int>(), d.histogram.bytes() / sizeof(int), 8192},
                         {d.select_state.get<std::uint64_t>(), d.select_state.bytes() / sizeof(std::uint64_t), 16},
                         {d.candidates.get<std::uint64_t>(), d.candidates.bytes() / sizeof(std::uint64_t), 512},
                         d.flag()}, d.stream));
                }
            }
            const auto query_offset = static_cast<std::size_t>(first) * attention_query_elements;
            // Borrow the REMAINING query capacity after slicing, never base+offset
            // with the original full elements. Other views use their actual whole
            // allocations, tight private strides and unchanged row-major Q6144.
            hip(launch_attention_q4_batch(
                {d.f(12) + query_offset, d.scratch[12].bytes() / sizeof(float) - query_offset, attention_query_elements},
                {kv_k.get<Q4_0>(), kv_k.bytes() / sizeof(Q4_0), kv_v.get<Q4_0>(), kv_v.bytes() / sizeof(Q4_0),
                    capacity, position + tokens}, count, position + first + 1,
                {d.ids.get<std::int32_t>(), d.ids.bytes() / sizeof(std::int32_t), attention_max_selected},
                {d.counts.get<int>(), d.counts.bytes() / sizeof(int), 2},
                {d.f(17), d.scratch[17].bytes() / sizeof(float), attention_query_elements},
                {{d.gathered_k.get<std::uint16_t>(), d.gathered_k.bytes() / sizeof(std::uint16_t), attention_gather_elements},
                 {d.gathered_v.get<std::uint16_t>(), d.gathered_v.bytes() / sizeof(std::uint16_t), attention_gather_elements},
                 {d.partial_o.get(), d.partial_o.bytes() / sizeof(float), attention_partial_elements},
                 {d.partial_ms.get<float2>(), d.partial_ms.bytes() / sizeof(float2), attention_partial_ml_elements},
                 {d.f(15), d.scratch[15].bytes() / sizeof(float), attention_query_elements}, d.flag()}, d.stream));
            // Provisional host accounting; published only after full-call/head
            // completion and numeric checks. Not a physical launch-count proxy.
            ++calls.batch_calls; calls.query_rows += static_cast<std::uint64_t>(count);
            if (count > 1) { ++calls.multiquery_calls; calls.multiquery_rows += static_cast<std::uint64_t>(count); }
            else ++calls.singleton_tail_calls;
            calls.max_query_rows = std::max(calls.max_query_rows, static_cast<std::uint64_t>(count));
            for (int q = 0; q < count; ++q) {
                const int t = first + q;
                const auto* rotated = d.f(17) + q * attention_query_elements;
                observe("qsa_attention_rotated", rotated, 6144, t);
                launch_hadamard(rotated, d.f(18), 6144, 64, d.stream);
                observe("qsa_attention_inverse_h64", d.f(18), 6144, t);
                hip(launch_sigmoid_gate(d.f(18), d.f(13) + t * 6144, 6144, d.f(9) + t * 6144, d.flag(), d.stream));
                observe("qsa_attention_gated", d.f(9) + t * 6144, 6144, t);
            }
        }
    }
    template<class Observe> void attention_chunk(int position, int capacity, float epsilon, int tokens, bool trace_stages,
        bool batch_attention, SessionAttentionStats& calls, Observe&& observe) {
        auto& d = device;
        // All projections belong to the logical chunk; only their borrowed
        // launch geometry is tiled. Recurrent/append work retains active history
        // across chronological <=128 slices, never replaying a decode layer.
        qkv.apply(d, d.f(6), d.f(9), tokens);
        if ((index + 1) % 4) {
            z.apply(d, d.f(6), d.f(10), tokens); alpha.apply(d, d.f(6), d.f(11), tokens); beta.apply(d, d.f(6), d.f(12), tokens);
            observe("qkv", d.f(9), W); observe("z", d.f(10), 6144);
            observe("alpha", d.f(11), 48); observe("beta", d.f(12), 48);
            tiles(tokens, [&](int first, int count) {
                launch_gdn({conv.get(), dt.get(), a.get(), gdn_norm.get(), epsilon},
                    {d.f(9) + first * W, d.f(10) + first * 6144, d.f(11) + first * 48, d.f(12) + first * 48},
                    count, recurrent.get(), history.get(), d.f(13) + first * 6144,
                    {d.f(14) + first * W, stage_recurrent.get(), stage_history.get(), d.f(15) + first * 6144,
                     nullptr, nullptr, d.flag()}, d.stream);
            });
            // GDN publishes output/state only when the sticky flag is zero. In
            // the diagnostic layer, check before observing unpublished scratch.
            if (failure_probe && trace_stages) d.check("GDN layer 8");
            // Large recurrent/control witnesses are explicitly N1-only.
            observe("gdn_convolved_normalized", d.f(14), W);
            observe("final_output", d.f(13), 6144);
        } else {
            key.apply(d, d.f(6), d.f(10), tokens); value.apply(d, d.f(6), d.f(11), tokens);
            observe("qsa_q_gate_raw", d.f(9), 12288);
            observe("qsa_key_raw", d.f(10), 512); observe("qsa_value_raw", d.f(11), 512);
            tiles(tokens, [&](int first, int count) {
                hip(launch_qsa_split_q_gate(d.f(9) + first * 12288, count, d.f(12) + first * 6144,
                    d.f(13) + first * 6144, d.flag(), d.stream));
            });
            observe("qsa_gate_raw", d.f(13), 6144);
            tiles(tokens, [&](int first, int count) {
                hip(launch_group_rms_norm(d.f(12) + first * 6144, q_norm.get(), count, 6144, 256, epsilon, true,
                    d.f(14) + first * 6144, d.flag(), d.stream));
                hip(launch_group_rms_norm(d.f(10) + first * 512, k_norm.get(), count, 512, 256, epsilon, true,
                    d.f(15) + first * 512, d.flag(), d.stream));
                hip(launch_qsa_rope(d.f(14) + first * 6144, frequencies.get(), count, 24, position + first,
                    d.f(14) + first * 6144, d.flag(), d.stream));
                hip(launch_qsa_rope(d.f(15) + first * 512, frequencies.get(), count, 2, position + first,
                    d.f(15) + first * 512, d.flag(), d.stream));
                launch_hadamard(d.f(14) + first * 6144, d.f(12) + first * 6144, count * 6144, 256, d.stream);
                launch_hadamard(d.f(15) + first * 512, d.f(10) + first * 512, count * 512, 256, d.stream);
                launch_hadamard(d.f(11) + first * 512, d.f(15) + first * 512, count * 512, 64, d.stream);
            });
            observe("qsa_query_rotated", d.f(12), 6144);
            observe("qsa_key_rotated", d.f(10), 512); observe("qsa_value_rotated", d.f(15), 512);
            tiles(tokens, [&](int first, int count) {
                launch_quantize_q4(d.f(10) + first * 512, kv_k.get<Q4_0>() + (position + first) * 16,
                    count * 512, d.flag(), d.stream);
                launch_quantize_q4(d.f(15) + first * 512, kv_v.get<Q4_0>() + (position + first) * 16,
                    count * 512, d.flag(), d.stream);
            });
            index_q.apply(d, d.f(6), d.f(14), tokens); index_k.apply(d, d.f(6), d.f(15), tokens);
            QsaIndexConfig cfg; cfg.capacity = static_cast<std::size_t>(capacity); cfg.rms_epsilon = epsilon; cfg.rotary_dim = 64;
            QsaIndexDeviceParameters p{cfg, iq_norm.get(), ik_norm.get(), frequencies.get()};
            tiles(tokens, [&](int first, int count) {
                auto* work = d.append_work.get();
                hip(launch_qsa_index_append(p, d.f(15) + first * 128, position + first, count, pooled.get(), tail.get(),
                    {d.append_packed.get<Q4_0>(), work, work + microtile_tokens * 128,
                     work + microtile_tokens * 128 + 33 * 128, nullptr, d.flag()}, d.stream));
            });
            // Future main/pooled keys can be prepared, but NONE is visible to an
            // earlier query. Every query keeps its own <=2051 selected gather.
            // All append inputs die before f(15..18) becomes the query workspace.
            if (batch_attention) attention_queries(p, position, capacity, tokens, calls, observe);
            else for (int t = 0; t < tokens; ++t) {
                const int visible = position + t + 1;
                hip(launch_qsa_index_score(p, d.f(14) + t * 512, pooled.get(), visible, d.f(15), d.f(16), d.flag(), d.stream));
                hip(launch_qsa_select(d.f(16), visible, d.ids.get<std::int32_t>(), d.block_ids.get<std::int32_t>(), d.counts.get<int>(),
                    {d.histogram.get<int>(), d.select_state.get<std::uint64_t>(), d.candidates.get<std::uint64_t>(), d.flag()}, d.stream));
                hip(launch_attention_q4(d.f(12) + t * 6144, kv_k.get<Q4_0>(), kv_v.get<Q4_0>(), capacity, d.ids.get<std::int32_t>(),
                    d.counts.get<int>(), visible, d.f(17),
                    {d.gathered_k.get<std::uint16_t>(), d.gathered_v.get<std::uint16_t>(), d.partial_o.get(),
                     d.partial_ms.get<float2>(), d.f(15), d.flag()}, d.stream));
                observe("qsa_attention_rotated", d.f(17), 6144, t);
                launch_hadamard(d.f(17), d.f(18), 6144, 64, d.stream);
                observe("qsa_attention_inverse_h64", d.f(18), 6144, t);
                hip(launch_sigmoid_gate(d.f(18), d.f(13) + t * 6144, 6144, d.f(9) + t * 6144, d.flag(), d.stream));
                observe("qsa_attention_gated", d.f(9) + t * 6144, 6144, t);
            }
            hip(hipMemcpyAsync(d.f(13), d.f(9), tokens * 6144 * sizeof(float), hipMemcpyDeviceToDevice, d.stream));
        }
        attn_output.apply(d, d.f(13), d.f(9), tokens);
        if (failure_probe && trace_stages) d.check("attention output layer 8");
        observe("block_attention_out", d.f(9), H);
    }
    template<class Observe> void attention(int position, int capacity, float epsilon, int tokens, bool trace_stages,
        bool batch_attention, SessionAttentionStats& calls, TargetLayerPrefixes* prefixes, Observe&& observe) {
        if (tokens > 3) { attention_chunk(position, capacity, epsilon, tokens, trace_stages, batch_attention, calls, observe); return; }
        auto& d = device;
        qkv.apply(d, d.f(6), d.f(9), tokens);
        if ((index + 1) % 4) {
            z.apply(d, d.f(6), d.f(10), tokens); alpha.apply(d, d.f(6), d.f(11), tokens); beta.apply(d, d.f(6), d.f(12), tokens);
            observe("qkv", d.f(9), 10240); observe("z", d.f(10), 6144);
            observe("alpha", d.f(11), 48); observe("beta", d.f(12), 48);
            // The large state/control witness is N1-only. Do not label a whole
            // window's pre/post state as per-token chronological prefixes.
            if (state_probe && tokens == 1 && trace_stages) {
                observe("gdn_state_pre", recurrent.get(), 786432);
                // Recomputed through the same GPU helper, NOT a capture of
                // recurrence registers. f(22) is diagnostic-only scratch.
                launch_gdn_control_witness({conv.get(), dt.get(), a.get(), gdn_norm.get(), epsilon},
                    {d.f(9), d.f(10), d.f(11), d.f(12)}, d.f(22), d.flag(), d.stream);
                if (failure_probe && trace_stages) d.check("GDN controls layer 8");
                observe("gdn_control_log_decay_recomputed", d.f(22), 48);
                observe("gdn_control_beta_recomputed", d.f(22) + 48, 48);
                observe("gdn_control_decay_recomputed", d.f(22) + 96, 48);
            }
            launch_gdn({conv.get(), dt.get(), a.get(), gdn_norm.get(), epsilon},
                       {d.f(9), d.f(10), d.f(11), d.f(12)}, tokens, recurrent.get(), history.get(), d.f(13),
                       {d.f(14), stage_recurrent.get(), stage_history.get(), d.f(15),
                        prefixes ? prefixes->recurrent.get() : nullptr,
                        prefixes ? prefixes->history.get() : nullptr, d.flag()}, d.stream);
            if (prefixes) ++*prefixes->calls;
            if (failure_probe && trace_stages) d.check("GDN layer 8");
            if (state_probe && tokens == 1 && trace_stages) observe("gdn_state_post", recurrent.get(), 786432);
            observe("gdn_convolved_normalized", d.f(14), 10240);
            observe("final_output", d.f(13), 6144);
        } else {
            key.apply(d, d.f(6), d.f(10), tokens); value.apply(d, d.f(6), d.f(11), tokens);
            observe("qsa_q_gate_raw", d.f(9), 12288);
            observe("qsa_key_raw", d.f(10), 512); observe("qsa_value_raw", d.f(11), 512);
            hip(launch_qsa_split_q_gate(d.f(9), tokens, d.f(12), d.f(13), d.flag(), d.stream));
            observe("qsa_gate_raw", d.f(13), 6144);
            hip(launch_group_rms_norm(d.f(12), q_norm.get(), tokens, 6144, 256, epsilon, true, d.f(14), d.flag(), d.stream));
            hip(launch_group_rms_norm(d.f(10), k_norm.get(), tokens, 512, 256, epsilon, true, d.f(15), d.flag(), d.stream));
            hip(launch_qsa_rope(d.f(14), frequencies.get(), tokens, 24, position, d.f(14), d.flag(), d.stream));
            hip(launch_qsa_rope(d.f(15), frequencies.get(), tokens, 2, position, d.f(15), d.flag(), d.stream));
            launch_hadamard(d.f(14), d.f(12), tokens * 6144, 256, d.stream);
            launch_hadamard(d.f(15), d.f(10), tokens * 512, 256, d.stream);
            launch_hadamard(d.f(11), d.f(15), tokens * 512, 64, d.stream);
            observe("qsa_query_rotated", d.f(12), 6144);
            observe("qsa_key_rotated", d.f(10), 512); observe("qsa_value_rotated", d.f(15), 512);
            // Contiguous [position+t][2][8 Q4 blocks], with unchanged packing.
            launch_quantize_q4(d.f(10), kv_k.get<Q4_0>() + position * 16, tokens * 512, d.flag(), d.stream);
            launch_quantize_q4(d.f(15), kv_v.get<Q4_0>() + position * 16, tokens * 512, d.flag(), d.stream);
            index_q.apply(d, d.f(6), d.f(14), tokens); index_k.apply(d, d.f(6), d.f(15), tokens);
            QsaIndexConfig cfg; cfg.capacity = static_cast<std::size_t>(capacity); cfg.rms_epsilon = epsilon; cfg.rotary_dim = 64;
            QsaIndexDeviceParameters p{cfg, iq_norm.get(), ik_norm.get(), frequencies.get()};
            hip(launch_qsa_index_append(p, d.f(15), position, tokens, pooled.get(), tail.get(),
                {d.q8.get<Q4_0>(), d.f(16), d.f(17), d.f(18),
                 prefixes ? prefixes->tail.get() : nullptr, d.flag()}, d.stream));
            if (prefixes) ++*prefixes->calls;
            // All prepared keys may exist, but every score/select/gather receives
            // ONLY start+t+1 visibility (and thus the actual causal tail). f(12)
            // queries, f(13) raw gates and f(14) index queries survive the loop.
            // Append's f(16..18) die before the serialized bounded query workspace.
            if (batch_attention && tokens > 1) attention_queries(p, position, capacity, tokens, calls, observe);
            else for (int t = 0; t < tokens; ++t) {
                const int visible = position + t + 1;
                hip(launch_qsa_index_score(p, d.f(14) + t * 512, pooled.get(), visible, d.f(15), d.f(16), d.flag(), d.stream));
                hip(launch_qsa_select(d.f(16), visible, d.ids.get<std::int32_t>(), d.block_ids.get<std::int32_t>(), d.counts.get<int>(),
                    {d.histogram.get<int>(), d.select_state.get<std::uint64_t>(), d.candidates.get<std::uint64_t>(), d.flag()}, d.stream));
                hip(launch_attention_q4(d.f(12) + t * 6144, kv_k.get<Q4_0>(), kv_v.get<Q4_0>(), capacity, d.ids.get<std::int32_t>(),
                    d.counts.get<int>(), visible, d.f(17),
                    {d.gathered_k.get<std::uint16_t>(), d.gathered_v.get<std::uint16_t>(), d.partial_o.get(),
                     d.partial_ms.get<float2>(), d.f(15), d.flag()}, d.stream));
                observe("qsa_attention_rotated", d.f(17), 6144, t);
                launch_hadamard(d.f(17), d.f(18), 6144, 64, d.stream);
                observe("qsa_attention_inverse_h64", d.f(18), 6144, t);
                hip(launch_sigmoid_gate(d.f(18), d.f(13) + t * 6144, 6144, d.f(9) + t * 6144, d.flag(), d.stream));
                observe("qsa_attention_gated", d.f(9) + t * 6144, 6144, t);
            }
            // f(9) holds all gated rows. Only now are ALL raw gates dead.
            hip(hipMemcpyAsync(d.f(13), d.f(9), tokens * 6144 * sizeof(float), hipMemcpyDeviceToDevice, d.stream));
        }
        attn_output.apply(d, d.f(13), d.f(9), tokens);
        if (failure_probe && trace_stages) d.check("attention output layer 8");
        observe("block_attention_out", d.f(9), H);
    }
    template<class Observe> void shared_project(int tokens, bool probe, Observe&& observe) {
        auto& d = device;
        shared_gate.apply(d, d.f(6), d.f(16), tokens); shared_up.apply(d, d.f(6), d.f(17), tokens);
        if (probe) {
            observe("ffn_shared_gate_raw", d.f(16), 640);
            observe("ffn_shared_up_raw", d.f(17), 640);
        }
        tiles(tokens, [&](int first, int count) {
            hip(launch_silu_up(d.f(16) + first * 640, d.f(17) + first * 640, count * 640, d.f(18) + first * 640, d.flag(), d.stream));
        });
        shared_down.apply(d, d.f(18), d.f(19), tokens);
        if (probe) {
            observe("ffn_shared_silu_middle", d.f(18), 640);
            observe("ffn_shared_down", d.f(19), H);
        }
        for (int first = 0; first < tokens; first += 8) {
            const int count = std::min(8, tokens - first);
            blas(launch_dense_linear(d.handle, {shared_weight.get(), H, 1}, d.f(6) + first * H, count, d.f(16) + first, d.stream));
        }
        if (probe) observe("ffn_shared_scalar_gate_raw", d.f(16), 1);
    }
    void shared_add(int tokens) {
        auto& d = device;
        if (tokens == 1) hip(launch_shared_sigmoid_add(d.f(19), d.f(16), H, d.f(20), d.flag(), d.stream));
        else tiles(tokens, [&](int first, int count) {
            hip(launch_shared_sigmoid_add_rows(d.f(19) + first * H, d.f(16) + first, count, H, d.f(20) + first * H, d.flag(), d.stream));
        });
    }
    template<class Observe> void ffn(SessionStats& stats, SessionRouteStats& call_routes, int tokens, bool trace_observations,
                                   CpuExpertPool* pool, SessionHybridPolicy policy, SessionHybridStats& diagnostics,
                                   SessionHybridProbeDiagnostics* failure_probe, std::uint64_t cpu_return_baseline,
                                   float* probe_down, Observe&& observe) {
        auto& d = device;
        // Besides skipping captures, skip diagnostic H2D/host temporaries before
        // the range. selected_ids relies on a captured row draining its upload.
        const bool probe = ffn_probe && trace_observations;
        router.apply(d, d.f(6), d.f(9), tokens);
        tiles(tokens, [&](int first, int count) {
            hip(launch_router_softmax(d.f(9) + first * 512, count, d.f(10) + first * 512, d.flag(), d.stream));
        });
        auto& probabilities = d.route_probabilities;
        hip(hipMemcpyAsync(probabilities.data(), d.f(10), tokens * 512 * sizeof(float), hipMemcpyDeviceToHost, d.stream));
        d.check("router layer " + std::to_string(index));
        if (probe) {
            observe("ffn_router_logits_raw", d.f(9), 512);
            observe("ffn_router_probabilities", d.f(10), 512);
        }
        auto& ids = d.route_ids;
        auto& weights = d.route_weights;
        select_routes(probabilities.data(), ids.data(), weights.data(), tokens);
        if (probe) {
            // Exact semantic ID->F32 conversion (0..511), never reinterpret I32
            // bytes. f(22) is dead until the wide rank fold's weight upload.
            std::vector<float> selected_ids(static_cast<std::size_t>(tokens) * 10);
            std::transform(ids.begin(), ids.begin() + tokens * 10, selected_ids.begin(),
                [](std::int32_t id) { return static_cast<float>(id); });
            hip(hipMemcpyAsync(d.f(22), selected_ids.data(), selected_ids.size() * sizeof(float), hipMemcpyHostToDevice, d.stream));
            observe("ffn_selected_expert_ids_exact_f32", d.f(22), 10);
            // Observe completes the stream before host lifetime/scratch reuse.
            hip(hipMemcpyAsync(d.f(22), weights.data(), tokens * 10 * sizeof(float), hipMemcpyHostToDevice, d.stream));
            observe("ffn_selected_weights", d.f(22), 10);
        }
        // Dedicated input copy: the common quantization workspace is overwritten
        // by expert down inputs. f(21) reserves ALL original Q8 X columns until
        // every group finishes. f(23) is separate gathered Q8, not common d.q8.
        tiles(tokens, [&](int first, int count) {
            hip(launch_quantize_q8_1(d.f(6) + first * H, d.q8.get<Q8_1>() + first * (H / 32), H, count, d.flag(), d.stream));
        });
        hip(hipMemcpyAsync(d.f(21), d.q8.get<Q8_1>(), tokens * (H / 32) * sizeof(Q8_1), hipMemcpyDeviceToDevice, d.stream));
        hip(hipMemsetAsync(d.f(20), 0, tokens * H * sizeof(float), d.stream));
        const auto* original = reinterpret_cast<const Q8_1*>(d.f(21));
        if (policy.mode != SessionHybridMode::disabled && tokens <= 3) {
            d.routes->prepare(std::span(ids).first(tokens * 10), std::span(weights).first(tokens * 10), tokens);
            d.upload_routes(tokens);
            for (const int count : d.routes->counts())
                call_routes.last_max_expert_group_assignments = std::max(
                    call_routes.last_max_expert_group_assignments, static_cast<std::uint64_t>(count));
            experts.run_hybrid(d, *pool, policy, stats, diagnostics, tokens, original, probe_down != nullptr,
                failure_probe, cpu_return_baseline, index);
            // Shared GPU work overlaps CPU jobs. f19/f16 must stay live through
            // the CPU-only row returns and routed fold; neither uses these buffers.
            shared_project(tokens, probe, observe);
            experts.join_hybrid(d, *pool, diagnostics);
            if (probe_down)
                for (int t = 0; t < tokens; ++t)
                    hip(hipMemcpyAsync(probe_down + t * 48 * 10 * H, d.routed_contributions.get() + t * 10 * H,
                        10 * H * sizeof(float), hipMemcpyDeviceToHost, d.stream));
            for (int t = 0; t < tokens; ++t)
                for (int rank = 0; rank < 10; ++rank)
                    hip(launch_scaled_add(d.routed_contributions.get() + (t * 10 + rank) * H,
                        weights[t * 10 + rank], H, d.f(20) + t * H, d.flag(), d.stream));
            if (probe) {
                for (int rank = 0; rank < 10; ++rank)
                    observe("ffn_expert_down_unweighted_rank_" + std::to_string(rank),
                        d.routed_contributions.get() + rank * H, H, -1, 10 * H);
                observe("ffn_routed_weighted_aggregate", d.f(20), H);
            }
            shared_add(tokens);
            return;
        }
        if (policy.mode != SessionHybridMode::disabled) ++diagnostics.gpu_only_wide_layers;
        if (tokens == 1) {
            // partial_sort above preserves the iota permutation: ten distinct
            // routes, hence exactly one assignment per selected expert for N1.
            call_routes.last_max_expert_group_assignments = 1;
            // Retain N1 physical cache/LRU order and immediate rank-order adds.
            for (int rank = 0; rank < 10; ++rank) {
                experts.run(d, ids[rank], original, weights[rank], stats);
                if (probe_down) hip(hipMemcpyAsync(probe_down + rank * H, d.f(19), H * sizeof(float), hipMemcpyDeviceToHost, d.stream));
                // Read after scaled-add's consumer, before the next rank reuses
                // f(19). No change to physical cache or numerical add order.
                if (probe) observe("ffn_expert_down_unweighted_rank_" + std::to_string(rank), d.f(19), H);
            }
        } else {
            d.routes->prepare(std::span(ids).first(tokens * 10), std::span(weights).first(tokens * 10), tokens);
            d.upload_routes(tokens);
            // prepare validated uniqueness per token. Scan the existing logical
            // histogram once, before any expert microtile/staging execution.
            for (const int count : d.routes->counts()) {
                call_routes.last_max_expert_group_assignments = std::max(
                    call_routes.last_max_expert_group_assignments, static_cast<std::uint64_t>(count));
                if (count > 128) ++call_routes.expert_groups_gt128;
            }
            if (tokens > 3) experts.run_chunk(d, original, stats);
            else {
                auto* gathered = reinterpret_cast<Q8_1*>(d.f(23));
                for (int expert = 0; expert < 512; ++expert) {
                    const auto group = d.routes->group(expert);
                    if (group.empty()) continue;
                    const int columns = static_cast<int>(group.size()); // unique per token, <=3
                    d.gather_routes(original, group, 0, columns);
                    // One canonical triplet acquisition/upload per group. Every
                    // additional group column is a logical hit, even on a cold miss.
                    experts.run_group(d, expert, gathered, columns, stats);
                    d.scatter_routes(group, 0, columns);
                    // Scatters precede f(16..19)/f(23)/slot reuse on this stream.
                }
            }
            // Execution order may group experts, but numerical reduction cannot:
            // scatter UNWEIGHTED down results, then left-fold original ranks0..9
            // with separately rounded normalized-weight MUL and accumulator ADD.
            if (probe_down)
                for (int t = 0; t < tokens; ++t)
                    hip(hipMemcpyAsync(probe_down + t * 48 * 10 * H, d.routed_contributions.get() + t * 10 * H,
                        10 * H * sizeof(float), hipMemcpyDeviceToHost, d.stream));
            if (tokens <= 3) {
                for (int t = 0; t < tokens; ++t)
                    for (int rank = 0; rank < 10; ++rank)
                        hip(launch_scaled_add(d.routed_contributions.get() + (t * 10 + rank) * H, weights[t * 10 + rank], H,
                            d.f(20) + t * H, d.flag(), d.stream));
            } else {
                // f(22) is dead diagnostic scratch here. Upload raw normalized
                // FP32 bits, then one pointwise left-fold per <=128 row slice.
                hip(hipMemcpyAsync(d.f(22), weights.data(), tokens * 10 * sizeof(float), hipMemcpyHostToDevice, d.stream));
                tiles(tokens, [&](int first, int count) {
                    hip(launch_routed_fold(d.routed_contributions.get() + first * 10 * H, d.f(22) + first * 10,
                        count, d.f(20) + first * H, d.flag(), d.stream));
                });
            }
        }
        if (probe) {
            if (tokens > 1) {
                // ALL scatters/fold consumers are queued. Storage is immutable
                // [token][original-rank][H], not a contiguous rank-major matrix.
                for (int rank = 0; rank < 10; ++rank)
                    observe("ffn_expert_down_unweighted_rank_" + std::to_string(rank),
                        d.routed_contributions.get() + rank * H, H, -1, 10 * H);
            }
            observe("ffn_routed_weighted_aggregate", d.f(20), H);
        }
        shared_project(tokens, probe, observe);
        shared_add(tokens);
    }
};
}

namespace {
void HybridInputProbeFrame::capture(const Device& d, int layer, int tokens, bool hybrid) {
    // run_hybrid has synchronized input_copied BEFORE accepting CPU jobs.
    // No new D2H/quantization; all-hit leaves availability zero.
    for (int t = 0; t < tokens; ++t) {
        const auto row = static_cast<std::size_t>(t * 48 + layer);
        std::copy_n(d.route_ids.data() + t * 10, 10, ids.data() + row * 10);
        std::copy_n(d.route_weights.data() + t * 10, 10, weights.data() + row * 10);
        if (hybrid && d.hybrid->cpu_active) {
            std::memcpy(original.data() + row * (H / 32), d.hybrid->input + t * (H / 32), (H / 32) * sizeof(Q8_1));
            available[row] = 1;
        }
    }
    if (!hybrid) return;
    for (int j = 0; j < d.hybrid->count; ++j) {
        const auto& plan = d.hybrid->plans[j];
        if (plan.work != HybridDevice::Work::cpu) continue;
        for (const auto& a : d.routes->group(plan.expert))
            cpu[(a.token * 48 + layer) * 10 + a.rank] = 1;
    }
}
}

namespace {
// KEEP tested N+1 prefix producers (gdn/blocks/qsa_index), WRITE OURS bounded
// ownership/restore orchestration. No new kernel, KV precision or math path.
struct TargetCheckpoints {
    std::array<Device*, 2> devices;
    std::array<TargetLayerPrefixes, 48> layers;
    Buffer ple_history, tap, staged_tap;
    std::array<PleHashCheckpoint, 4> hash;
    SessionCheckpointState state{true, false, 0, 0, 0};
    SessionSpeculativeStats stats;
    std::array<std::uint64_t, 3> prefix_calls{}; // GDN, PLE conv, QSA tail; provisional until head success.
    int tap_rows = 0;
    std::uint64_t tap_position = 0;
    TargetCheckpoints(std::array<Device*, 2> owners, const PleHashConfig& config, int max_rows)
        : devices(owners), hash{PleHashCheckpoint(config), PleHashCheckpoint(config),
                               PleHashCheckpoint(config), PleHashCheckpoint(config)} {
        try {
            for (int l = 0; l < 48; ++l) {
                hip(hipSetDevice(l / 24));
                auto& p = layers[l];
                if ((l + 1) % 4) {
                    p.recurrent = Buffer(checked_size(checked_size(4, gdn_device_recurrent_elements), sizeof(float)));
                    p.history = Buffer(checked_size(checked_size(4, gdn_device_history_elements), sizeof(float)));
                    p.calls = &prefix_calls[0];
                } else {
                    p.tail = Buffer(checked_size(checked_size(4, 384), sizeof(float)));
                    p.calls = &prefix_calls[2];
                }
            }
            hip(hipSetDevice(0));
            ple_history = Buffer(checked_size(checked_size(4, ple_device_history_elements), sizeof(float)));
            hip(hipSetDevice(1));
            // Ordinary PP must expose every actual row, even max_batch=1024.
            const auto bytes = checked_size(checked_size(max_rows, W), sizeof(float));
            tap = Buffer(bytes); staged_tap = Buffer(bytes);
        } catch (...) { drain_cleanup(); throw; }
    }
    ~TargetCheckpoints() { drain_cleanup(); }
    void drain_cleanup() noexcept {
        for (auto* d : devices) if (d) d->drain_cleanup();
    }
    void clear() noexcept {
        state = {true, false, 0, 0, 0}; stats = {}; prefix_calls = {}; tap_rows = 0; tap_position = 0;
    }
    std::uint64_t hash_payload_bytes() const {
        std::uint64_t bytes = 0;
        for (const auto& s : hash) {
            const auto& c = s.config();
            bytes += (c.multipliers.capacity() + c.vocab_sizes.capacity() + c.offsets.capacity()) * sizeof(std::uint64_t);
            // Checkpoint API exposes fixed history size, not private capacity.
            bytes += s.history().size() * sizeof(std::int64_t);
        }
        return bytes;
    }
};
static_assert(gdn_device_recurrent_elements == 786432 && gdn_device_history_elements == 30720 &&
              ple_device_history_elements == 92160 && W == 10240);
}

struct Session::Impl {
    using LayerwiseRouteCounts = std::array<std::array<std::uint32_t, 512>, 48>;
    struct PipelineCall {
        std::span<const std::int32_t> window;
        int position;
        std::array<SessionStats, 2> stats{};
        std::array<SessionAttentionStats, 2> attention{};
        std::array<SessionLayerwiseTransfers, 2> transfers{};
        LayerwiseRouteCounts* routes;
    };
    Model model;
    SessionConfig config;
    SessionStats statistics;
    SessionRouteStats route_statistics;
    SessionAttentionStats attention_statistics;
    SessionLayerwiseTransfers transfer_statistics;
    bool batch_attention = false;
    SessionHybridPolicy hybrid_policy;
    // Preserve the original 20-counter member footprint/alignment; the new
    // six-counter tail is stored in opt-in HybridDevice only. Base Impl sizeof
    // includes the separate optional checkpoint handle even when disabled.
    std::array<std::uint64_t, 20> hybrid_statistics{};
    SessionHybridProbeDiagnostics probe_diagnostics;
    std::array<std::unique_ptr<Device>, 2> devices;
    std::vector<std::unique_ptr<Layer>> layers;
    std::unique_ptr<Hc> head_hc;
    Matrix head;
    std::vector<std::byte> embeddings;
    TensorType embedding_type;
    std::size_t embedding_stride;
    PleHashCpu ple_hash;
    PleLookup ple_lookup;
    Matrix ple_key, ple_value;
    Buffer ple_key_norm, ple_query_norm, ple_conv_norm, ple_conv, ple_history, ple_stage_history;
    std::array<std::uint64_t, 16> ple_rows{};
    std::array<float, H> ple_embedding{};
    std::vector<float> host_logits;
    std::vector<float> working_logits; // Opt-in transactional full-head staging.
    std::array<std::vector<float>, 2> probe_down, probe_ffn;
    int probe_tokens = 0;
    struct Pinned {
        float* pointer = nullptr;
        std::size_t bytes;
        explicit Pinned(int tokens) : bytes(checked_size(checked_size(tokens, W), sizeof(float))) {
            hip(hipHostMalloc(&pointer, bytes, hipHostMallocDefault));
        }
        ~Pinned() { if (pointer) cleanup(hipHostFree(pointer)); }
    };
    std::unique_ptr<Pinned> pinned;
    float* handoff = nullptr;
    float epsilon;
    bool usable = true;
    std::ofstream trace_metadata;
    std::uint64_t trace_occurrence = 0;
    int coordinator_cpu = -1;
    // One new base-host handle, honestly reported by sizeof(Impl). Declared
    // before pool: constructor unwind joins CPU, then drains GPU before snapshots
    // or any earlier live layer/PLE/pinned owner can release its buffers.
    std::unique_ptr<TargetCheckpoints> checkpoints;
    // Constructor-only full-prompt resources; destruction drains before DMA
    // staging or activation ownership is released, including constructor unwind.
    std::unique_ptr<LayerwisePrefill> layerwise;
    // Declared last: constructor unwind joins before borrowed RAM/frames release.
    std::unique_ptr<CpuExpertPool> cpu_pool;
    // Last owner joins before every borrowed device/host object during unwind.
    PipelineCall* pipeline_call = nullptr; // Borrowed only between begin and drain.
    bool pipeline_fault_pending = true;
    std::unique_ptr<PrefillPipeline> prefill_pipeline;

    static_assert(offsetof(SessionHybridStats, cpu_gate_up_jobs) == 20 * sizeof(std::uint64_t) &&
        sizeof(SessionHybridStats) == 26 * sizeof(std::uint64_t) &&
        sizeof(std::array<std::uint64_t, 20>) == 20 * sizeof(std::uint64_t) &&
        alignof(std::array<std::uint64_t, 20>) == alignof(SessionHybridStats) &&
        std::is_trivially_copyable_v<SessionHybridStats>);
    SessionHybridStats hybrid_stats() const {
        std::array<std::uint64_t, 26> words{};
        std::copy(hybrid_statistics.begin(), hybrid_statistics.end(), words.begin());
        if (devices[0] && devices[0]->hybrid)
            std::copy(devices[0]->hybrid->projection_statistics.begin(),
                devices[0]->hybrid->projection_statistics.end(), words.begin() + 20);
        return std::bit_cast<SessionHybridStats>(words);
    }
    void publish_hybrid(SessionHybridStats stats) {
        const auto words = std::bit_cast<std::array<std::uint64_t, 26>>(stats);
        std::copy_n(words.begin(), 20, hybrid_statistics.begin());
        if (devices[0] && devices[0]->hybrid)
            std::copy_n(words.begin() + 20, 6, devices[0]->hybrid->projection_statistics.begin());
    }

    Impl(const std::string& path, SessionConfig cfg)
        : model(path), config(checked_config(std::move(cfg))), ple_hash(ple_hash_config(model, 1)),
          ple_lookup(model, ple_hash.state().config()), host_logits(checked_size(config.max_batch_tokens, V)) {
        if (!config.trace_directory.empty()) {
            const std::filesystem::path path(config.trace_directory);
            const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
            if (!std::filesystem::is_directory(parent) || !std::filesystem::create_directory(path))
                throw std::runtime_error("trace directory must be new with an existing parent");
            trace_metadata.open(path / "tensors.jsonl");
            if (!trace_metadata) throw std::runtime_error("trace metadata open");
        }
        if (std::get<std::string>(model.metadata_value("general.architecture").value) != "qwen4exp" ||
            std::get<std::uint32_t>(model.metadata_value("qwen4exp.block_count").value) != 48 ||
            std::get<std::uint32_t>(model.metadata_value("qwen4exp.embedding_length").value) != H)
            throw std::runtime_error("unsupported Session architecture");
        epsilon = std::get<float>(model.metadata_value("qwen4exp.attention.layer_norm_rms_epsilon").value);
        const auto rope = std::get<std::uint32_t>(model.metadata_value("qwen4exp.rope.dimension_count").value);
        const auto base = std::get<float>(model.metadata_value("qwen4exp.rope.freq_base").value);
        if (rope != 64 || !std::isfinite(base) || base <= 0 || !std::isfinite(epsilon) || epsilon <= 0)
            throw std::runtime_error("Session RoPE/epsilon");
        const auto* embedding = &require(model, "token_embd.weight", {H, V});
        embedding_type = embedding->type; embedding_stride = static_cast<std::size_t>(embedding->strides[1]);
        embeddings.resize(static_cast<std::size_t>(embedding->byte_size)); model.read_tensor(embedding->name, embeddings);
        int n = 0; hip(hipGetDeviceCount(&n)); if (n != 2) throw std::runtime_error("Session requires two gfx906");
        devices[0] = std::make_unique<Device>(0, config.max_batch_tokens, config.cpu_workers > 0, config.attention_query_tile);
        devices[1] = std::make_unique<Device>(1, config.max_batch_tokens, config.cpu_workers > 0, config.attention_query_tile);
        layers.reserve(48);
        for (int l = 0; l < 48; ++l) {
            std::fprintf(stderr, "load layer %d/48\n", l + 1);
            layers.push_back(std::make_unique<Layer>(model, *devices[l / 24], l, config, base));
        }
        auto& d = *devices[0];
        ple_key = matrix(model, d, "blk.1.ple_key.weight", H, W);
        ple_value = matrix(model, d, "blk.1.ple_value.weight", H, H);
        ple_key_norm = parameter(model, d, "blk.1.ple_norm_key.weight", {W});
        ple_query_norm = parameter(model, d, "blk.1.ple_norm_query.weight", {W});
        ple_conv_norm = parameter(model, d, "blk.1.ple_norm_conv.weight", {W});
        ple_conv = parameter(model, d, "blk.1.ple_conv1d.weight", {4, W});
        hip(hipSetDevice(0)); ple_history = Buffer(92160 * sizeof(float)); ple_stage_history = Buffer(92160 * sizeof(float));
        head_hc = std::make_unique<Hc>(model, *devices[1], "output_hc_", true);
        head = matrix(model, *devices[1], "output.weight", H, V);
        pinned = std::make_unique<Pinned>(config.max_batch_tokens); handoff = pinned->pointer;
        if (config.cpu_workers || config.speculative_checkpoints || config.prefill_pipeline_tokens) working_logits.resize(host_logits.size());
        if (config.cpu_workers) {
            if (config.hybrid_probe) {
                for (auto& frame : probe_down) frame.resize(3 * 48 * 10 * H);
                for (auto& frame : probe_ffn) frame.resize(3 * 48 * H);
                for (auto& frame : devices[0]->hybrid->probe_inputs) frame.allocate();
            }
#if defined(__linux__)
            std::array<int, CPU_SETSIZE> cpus{};
            std::size_t count = 0;
            cpu_ok(cpu_expert_allowed_physical_cpus(cpus, count));
            if (count <= static_cast<std::size_t>(config.cpu_workers))
                throw std::runtime_error("Session CPU pool requires workers plus reserved coordinator physical core");
            coordinator_cpu = cpus[0];
            cpu_pool = std::make_unique<CpuExpertPool>(CpuExpertPoolConfig{
                static_cast<std::size_t>(config.cpu_workers), true,
                std::span(cpus).subspan(1, static_cast<std::size_t>(config.cpu_workers))});
#else
            throw std::runtime_error("Session CPU worker pinning requires Linux");
#endif
        }
        if (config.speculative_checkpoints)
            checkpoints = std::make_unique<TargetCheckpoints>(
                std::array<Device*, 2>{devices[0].get(), devices[1].get()}, ple_hash.state().config(),
                std::max(config.max_batch_tokens, config.layerwise_prefill_capacity));
        if (config.layerwise_prefill_capacity)
            layerwise = std::make_unique<LayerwisePrefill>(
                std::array<Device*, 2>{devices[0].get(), devices[1].get()},
                config.prefill_pipeline_tokens ? config.prefill_pipeline_tokens : config.layerwise_prefill_capacity,
                config.prefill_residual_device);
        if (config.prefill_pipeline_tokens)
            layerwise->enable_pipeline(layerwise->stage_capacity, config.max_batch_tokens);
        reset();
        if (config.prefill_pipeline_tokens)
            prefill_pipeline = std::make_unique<PrefillPipeline>(this, [](void* self, int window, int slot) {
                static_cast<Impl*>(self)->pipeline_produce(window, slot);
            });
    }
    ~Impl() {
        prefill_pipeline.reset();
        cpu_pool.reset(); // Drain/join BEFORE immutable RAM and pinned frames.
        for (auto& d : devices) if (d) d->drain_cleanup();
    }
    void set_attention_batch(bool enabled) {
        // ALL validation precedes any drain, device switch or policy mutation.
        // Reject even trace_first_token==capacity: the mode is truthful for the
        // entire Session lifetime and legacy manifest ordering stays unchanged.
        if (enabled && !config.trace_directory.empty())
            throw std::invalid_argument("Session attention batch requires trace off");
        if (!usable) throw std::runtime_error("Session invalidated; reset required");
        if (cpu_pool) cpu_ok(cpu_pool->drain());
        for (auto& d : devices) d->drain();
        batch_attention = enabled;
    }
    void set_hybrid_policy(SessionHybridPolicy policy) {
        switch (policy.mode) {
        case SessionHybridMode::disabled: case SessionHybridMode::mixed:
        case SessionHybridMode::force_cpu: case SessionHybridMode::force_gpu_misses: break;
        default: throw std::invalid_argument("Session hybrid mode");
        }
        if (policy.gpu_miss_groups < 0 || policy.gpu_miss_groups > 2 ||
             (policy.mode == SessionHybridMode::mixed && policy.gpu_miss_groups == 0) ||
            (policy.mode != SessionHybridMode::disabled && !cpu_pool) ||
            (policy.diagnostic_fail_after_admission && (!config.hybrid_probe || policy.mode != SessionHybridMode::mixed)))
            throw std::invalid_argument("Session hybrid policy/resources");
        if (!usable) throw std::runtime_error("Session invalidated; reset required");
        if (cpu_pool) cpu_ok(cpu_pool->drain());
        for (auto& d : devices) d->drain();
        if (policy.diagnostic_fail_after_admission && !hybrid_policy.diagnostic_fail_after_admission)
            probe_diagnostics = {};
        hybrid_policy = policy;
    }
    void reset() {
        // A HIP failure during incremental reset leaves state partially cleared.
        // Retain published diagnostics until the successful endpoint, but never
        // let such a session run without another completed reset.
        usable = false;
        if (cpu_pool) {
            (void)cpu_pool->drain(); // Previous numeric status must not poison recovery.
            cpu_ok(cpu_pool->reset()); // Drained reset clears old jobs/status/tickets.
        }
        // Also orders exceptional partial bands before any flag/state/host reuse.
        for (auto& d : devices) {
            d->drain();
            for (auto& stage : d->stages) if (stage) stage->used = false;
        }
        for (auto& l : layers) { l->experts.finish_admissions(false); l->reset(); }
        hip(hipSetDevice(0)); hip(hipMemsetAsync(ple_history.get<void>(), 0, ple_history.bytes(), devices[0]->stream));
        for (auto& d : devices) { hip(hipSetDevice(d->id)); d->clear(); hip(hipStreamSynchronize(d->stream)); }
        ple_hash.reset(); statistics = {}; route_statistics = {}; attention_statistics = {}; transfer_statistics = {}; publish_hybrid({});
        if (checkpoints) checkpoints->clear();
        probe_tokens = 0; usable = true;
    }
    SessionMemory memory() const {
        int previous = 0; hip(hipGetDevice(&previous));
        struct RestoreDevice {
            int device;
            ~RestoreDevice() { cleanup(hipSetDevice(device)); }
        } restore{previous};
        SessionMemory result;
        result.capacity = config.capacity; result.expert_slots = config.expert_slots;
        result.speculative_checkpoints = config.speculative_checkpoints;
        result.prefill_residual_device = config.prefill_residual_device;
        result.host_session_impl_bytes = sizeof(Impl); result.host_session_config_bytes = sizeof(SessionConfig);
        if (checkpoints) {
            result.host_speculative_owner_bytes = sizeof(TargetCheckpoints);
            result.host_speculative_hash_payload_bytes = checkpoints->hash_payload_bytes();
        }
        result.attention_query_tile = config.attention_query_tile;
        result.host_embedding_capacity = embeddings.capacity();
        result.host_logit_capacity = (host_logits.capacity() + working_logits.capacity()) * sizeof(float);
        result.pinned_handoff = pinned->bytes;
        if (layerwise) {
            const auto& p = *layerwise;
            result.layerwise_prefill_capacity = config.layerwise_prefill_capacity;
            result.layerwise_stage_capacity = p.stage_capacity;
            result.host_layerwise_owner = sizeof(LayerwisePrefill) + 2 * sizeof(LayerwiseDevice);
            for (const auto& a : p.activations) result.host_layerwise_activations += a.capacity() * sizeof(float);
            result.host_layerwise_ffn_input = p.mixed.capacity() * sizeof(float);
            result.host_layerwise_injection = p.injection.capacity() * sizeof(float);
            result.host_layerwise_probabilities = p.probabilities.capacity() * sizeof(float);
            result.host_layerwise_routes = p.weights.capacity() * sizeof(float) + p.ids.capacity() * sizeof(std::int32_t);
            // Inline RouteGroups metadata is included in host_layerwise_owner.
            result.host_layerwise_groups = static_cast<std::size_t>(p.stage_capacity) * 10 * sizeof(RouteAssignment);
            result.pinned_layerwise_metadata = p.dto_bytes;
            if (p.stage1) {
                const auto& h = *p.stage1;
                result.prefill_pipeline_tokens = config.prefill_pipeline_tokens;
                result.prefill_pipeline_workers = prefill_pipeline ? 1 : 0;
                result.host_pipeline_owner = sizeof(LayerwiseHost) +
                    (prefill_pipeline ? sizeof(PrefillPipeline) + prefill_pipeline->metadata_bytes() : 0);
                for (const auto& a : h.activations) result.host_pipeline_payload += a.capacity() * sizeof(float);
                for (const auto& a : p.ring) result.host_pipeline_payload += a.capacity() * sizeof(float);
                result.host_pipeline_payload += (h.mixed.capacity() + h.injection.capacity() +
                    h.probabilities.capacity() + h.weights.capacity()) * sizeof(float) +
                    h.ids.capacity() * sizeof(std::int32_t) +
                    static_cast<std::size_t>(p.stage_capacity) * 10 * sizeof(RouteAssignment);
                result.pinned_pipeline_metadata = h.dto_bytes;
                result.pinned_pipeline_handoff = p.stage1_transfer_bytes;
            }
        }
        for (const auto& frame : probe_down) result.host_hybrid_probe += frame.capacity() * sizeof(float);
        for (const auto& frame : probe_ffn) result.host_hybrid_probe += frame.capacity() * sizeof(float);
        if (config.hybrid_probe)
            for (const auto& frame : devices[0]->hybrid->probe_inputs) result.host_hybrid_input_probe += frame.capacity_bytes();
        result.host_hybrid_probe += result.host_hybrid_input_probe;
        if (cpu_pool) {
            (void)cpu_pool->drain(); // Inspection still works after failed jobs.
            result.cpu_workers = static_cast<int>(cpu_pool->workers());
            result.cpu_pool_metadata = cpu_pool->metadata_bytes(); result.cpu_pool_scratch = cpu_pool->scratch_bytes();
        }
        for (int id = 0; id < 2; ++id) {
            auto& out = result.devices[id];
            auto& d = *devices[id];
            d.drain();
            out.head_logits_bytes = d.logits.bytes();
            result.selection_score_buffer_bytes[id] = d.scratch[16].bytes();
            result.selection_histogram_bytes[id] = d.histogram.bytes();
            result.selection_state_bytes[id] = d.select_state.bytes();
            result.selection_candidate_bytes[id] = d.candidates.bytes();
            result.host_routing_capacity += d.route_probabilities.capacity() * sizeof(float) +
                d.route_weights.capacity() * sizeof(float) + d.route_ids.capacity() * sizeof(std::int32_t);
            if (d.routes) result.host_route_group_payload += sizeof(RouteGroups) +
                static_cast<std::size_t>(config.max_batch_tokens) * 10 * sizeof(RouteAssignment);
            if (d.route_dto) result.pinned_route_metadata += d.route_metadata.bytes();
            result.route_metadata_bytes[id] = d.route_metadata.bytes();
            result.attention[id] = {d.gathered_k.bytes(), d.gathered_v.bytes(), d.partial_o.bytes(), d.partial_ms.bytes(),
                d.scratch[15].bytes(), d.scratch[17].bytes(), d.ids.bytes(), d.block_ids.bytes(), d.counts.bytes(),
                d.gathered_k.bytes() + d.gathered_v.bytes() + d.partial_o.bytes() + d.partial_ms.bytes() +
                    checked_size(checked_size(config.attention_query_tile, attention_query_elements), sizeof(float))};
            out.device = id; out.first_layer = id * 24; out.last_layer = id * 24 + 23;
            auto add = [&](std::uint64_t& category, const Buffer& b) {
                if (!b.bytes()) {
                    if (b.get<void>()) throw std::runtime_error("zero-byte Session allocation");
                    return;
                }
                if (!b.get<void>() || b.owner() != id) throw std::runtime_error("Session allocation owner mismatch");
                category += b.bytes(); out.owned_bytes += b.bytes(); ++out.owned_buffers;
            };
            auto weight = [&](const Matrix& m) { add(out.weights, m.data); };
            auto hc = [&](const Hc& h) {
                weight(h.down); weight(h.up); weight(h.inject); add(out.weights, h.norm);
            };
            for (const auto& b : d.scratch) add(out.workspace, b);
            if (layerwise) {
                const auto& p = *layerwise->gpu[id];
                result.layerwise_original_q8_bytes[id] = p.original.bytes();
                result.layerwise_contribution_bytes[id] = p.contributions.bytes();
                result.layerwise_metadata_bytes[id] = p.metadata.bytes();
                result.layerwise_residual_bytes[id] = p.residual.bytes();
                for (const auto* b : {&p.original, &p.contributions, &p.metadata, &p.residual}) add(out.workspace, *b);
            }
            for (const auto* b : {&d.q8, &d.error, &d.logits, &d.gathered_k, &d.gathered_v,
                    &d.partial_o, &d.partial_ms, &d.histogram, &d.select_state, &d.candidates,
                    &d.ids, &d.block_ids, &d.counts, &d.routed_contributions, &d.route_metadata, &d.ds4,
                    &d.append_packed, &d.append_work}) add(out.workspace, *b);
            for (const auto& stage : d.stages) if (stage) {
                add(out.workspace, stage->data);
                result.pinned_expert_staging += stage->data.bytes();
                result.expert_stage_capacity_bytes[id] = stage->data.bytes();
            }
            if (d.hybrid) {
                result.pinned_hybrid_input += HybridDevice::input_bytes;
                result.pinned_hybrid_output += HybridDevice::output_bytes;
                result.pinned_hybrid_error += sizeof(int);
                result.pinned_hybrid_middle += HybridDevice::middle_bytes;
                result.hybrid_gate_up_bytes[id] = d.hybrid->gate_up.bytes();
                result.hybrid_middle_float_bytes[id] = d.hybrid->middle_float.bytes();
                result.hybrid_middle_q8_bytes[id] = d.hybrid->middle_q8.bytes();
                result.hybrid_middle_error_bytes[id] = d.hybrid->middle_error.bytes();
                for (const auto* b : {&d.hybrid->gate_up, &d.hybrid->middle_float,
                        &d.hybrid->middle_q8, &d.hybrid->middle_error}) add(out.workspace, *b);
                result.host_hybrid_plans += sizeof(HybridDevice);
                result.hybrid_contribution_bytes[id] = d.routed_contributions.bytes();
            }
            for (const auto& pointer : layers) {
                const auto& l = *pointer;
                if (l.index / 24 != id) continue;
                if (l.device.id != id) throw std::runtime_error("Session layer owner mismatch");
                hc(l.attention_hc); hc(l.ffn_hc);
                for (const auto* m : {&l.router, &l.shared_gate, &l.shared_up, &l.shared_down,
                        &l.qkv, &l.z, &l.alpha, &l.beta, &l.attn_output, &l.key, &l.value,
                        &l.index_q, &l.index_k}) weight(*m);
                for (const auto* b : {&l.shared_weight, &l.conv, &l.dt, &l.a, &l.gdn_norm,
                        &l.q_norm, &l.k_norm, &l.iq_norm, &l.ik_norm, &l.frequencies}) add(out.weights, *b);
                add(out.expert_slots, l.experts.slots);
                for (const auto& ram : l.experts.ram) {
                    result.ram_expert_capacity += ram.capacity(); result.ram_expert_payload += ram.size();
                }
                result.expert_payload_reads += l.experts.payload_reads;
                result.expert_payload_bytes_read += l.experts.payload_bytes_read;
                result.host_cpu_expert_views += l.experts.cpu_views.capacity() * sizeof(CpuExpert) +
                    l.experts.pending.capacity() * sizeof(int);
                add(out.qsa_kv, l.kv_k); add(out.qsa_kv, l.kv_v);
                add(out.qsa_index, l.pooled); add(out.qsa_index, l.tail);
                add(out.gdn_state, l.recurrent); add(out.gdn_state, l.history);
                add(out.workspace, l.stage_recurrent); add(out.workspace, l.stage_history);
                if ((l.index + 1) % 4) ++out.gdn_layers; else ++out.qsa_layers;
            }
            if (id == 0) {
                weight(ple_key); weight(ple_value);
                for (const auto* b : {&ple_key_norm, &ple_query_norm, &ple_conv_norm, &ple_conv}) add(out.weights, *b);
                add(out.ple_state, ple_history); add(out.workspace, ple_stage_history);
            } else { hc(*head_hc); weight(head); }
            if (checkpoints) {
                for (int l = id * 24; l < id * 24 + 24; ++l) {
                    const auto& p = checkpoints->layers[l];
                    result.checkpoint_recurrent_bytes[id] += p.recurrent.bytes();
                    result.checkpoint_history_bytes[id] += p.history.bytes();
                    result.checkpoint_qsa_tail_bytes[id] += p.tail.bytes();
                    add(out.workspace, p.recurrent); add(out.workspace, p.history); add(out.workspace, p.tail);
                }
                if (id == 0) {
                    result.checkpoint_ple_history_bytes[id] = checkpoints->ple_history.bytes();
                    add(out.workspace, checkpoints->ple_history);
                } else {
                    result.target_tap_bytes[id] = checkpoints->tap.bytes();
                    result.target_tap_staging_bytes[id] = checkpoints->staged_tap.bytes();
                    add(out.workspace, checkpoints->tap); add(out.workspace, checkpoints->staged_tap);
                }
            }
            if (out.owned_bytes != Buffer::live_bytes[id] || out.owned_buffers != Buffer::live_buffers[id])
                throw std::runtime_error("Session allocation ledger omitted a live Buffer");
            out.owned_peak_bytes = Buffer::peak_bytes[id];
            std::size_t free = 0, total = 0; hip(hipMemGetInfo(&free, &total));
            out.total_vram = total; out.free_vram = free;
        }
        result.ownership_verified = true;
        return result;
    }
    void capture(Device& d, const std::string& name, const float* x, int count, int layer = -1, int token_offset = 0) {
        if (!trace_metadata.is_open()) return;
        const auto position = statistics.consumed_tokens + static_cast<std::uint64_t>(token_offset);
        if (position < static_cast<std::uint64_t>(config.trace_first_token)) return;
        std::vector<float> values(static_cast<std::size_t>(count));
        hip(hipMemcpyAsync(values.data(), x, values.size() * sizeof(float), hipMemcpyDeviceToHost, d.stream));
        hip(hipStreamSynchronize(d.stream));
        for (float v : values) if (!std::isfinite(v)) throw std::runtime_error("capture nonfinite: " + name);
        const std::string file = "tensor-" + std::to_string(trace_occurrence++) + ".f32.bin";
        std::ofstream output(std::filesystem::path(config.trace_directory) / file, std::ios::binary);
        output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!output) throw std::runtime_error("trace write");
        // Additional diagnostic layer 2 does not change the frozen allowlist
        // of gate-required/optional named captures in compare_session.py.
        trace_metadata << "{\"name\":" << std::quoted((layer == 2 || layer == 6 || layer == 8 || layer == 19 || layer == 22) ? "debug_" + name : name) << ",\"layer\":" << layer << ",\"position\":" << position
                       << ",\"type\":\"F32\",\"elements\":" << count << ",\"file\":" << std::quoted(file) << "}\n";
        trace_metadata.flush(); if (!trace_metadata) throw std::runtime_error("trace metadata write");
    }
    void capture_batch(Device& d, const std::string& name, const float* x, int width, int tokens, int layer = -1) {
        if (!trace_metadata.is_open()) return;
        for (int t = 0; t < tokens; ++t) capture(d, name, x + t * width, width, layer, t);
    }
    void ple(Device& d, std::span<const std::int32_t> window, float*& residual, float*& alternate, bool verify) {
        const int tokens = static_cast<int>(window.size());
        // Initial embedding H2D has completed at layer0's check. Borrow the
        // preallocated pinned staging for chronological selected PLE lookups;
        // it remains immutable until layer1 finishes, before the split handoff.
        if (verify) ple_hash.save(checkpoints->hash[0]);
        for (int t = 0; t < tokens; ++t) {
            ple_hash.step(window[t], ple_rows);
            if (verify) ple_hash.save(checkpoints->hash[t + 1]);
            ple_lookup.lookup(ple_rows, ple_embedding);
            std::copy(ple_embedding.begin(), ple_embedding.end(), handoff + t * H);
        }
        hip(hipMemcpyAsync(d.f(9), handoff, tokens * H * sizeof(float), hipMemcpyHostToDevice, d.stream));
        capture_batch(d, "ple_embedding", d.f(9), H, tokens, 1);
        ple_key.apply(d, d.f(9), d.f(10), tokens); ple_value.apply(d, d.f(9), d.f(11), tokens);
        capture_batch(d, "ple_key_raw", d.f(10), W, tokens, 1); capture_batch(d, "ple_value_raw", d.f(11), H, tokens, 1);
        tiles(tokens, [&](int first, int count) {
            hip(launch_group_rms_norm(d.f(10) + first * W, ple_key_norm.get(), count, W, H, epsilon, false,
                d.f(12) + first * W, d.flag(), d.stream));
        });
        tiles(tokens, [&](int first, int count) {
            hip(launch_group_rms_norm(residual + first * W, ple_query_norm.get(), count, W, H, epsilon, false,
                d.f(13) + first * W, d.flag(), d.stream));
        });
        // Keep the qualified N1 PLE dot reduction geometry on identical rows.
        // Only gate launches are sliced; logical PLE/history work stays whole.
        for (int first = 0; first < tokens; first += 8) {
            const int count = std::min(8, tokens - first);
            hip(launch_ple_gate(d.f(12) + first * W, d.f(13) + first * W, d.f(11) + first * H,
                count, d.f(14) + first * W, d.flag(), d.stream));
        }
        capture_batch(d, "ple_gated_value", d.f(14), W, tokens, 1);
        tiles(tokens, [&](int first, int count) {
            hip(launch_group_rms_norm(d.f(14) + first * W, ple_conv_norm.get(), count, W, H, epsilon, false,
                d.f(15) + first * W, d.flag(), d.stream));
        });
        tiles(tokens, [&](int first, int count) {
            hip(launch_ple_causal_conv(d.f(14) + first * W, d.f(15) + first * W, ple_conv.get(), ple_history.get(), count, nullptr,
                {d.f(16) + first * W, ple_stage_history.get(),
                 verify ? checkpoints->ple_history.get() : nullptr}, d.flag(), d.stream));
            if (verify) ++checkpoints->prefix_calls[1];
            hip(launch_publish_ple_history(ple_stage_history.get(), ple_history.get(), d.flag(), d.stream));
        });
        tiles(tokens, [&](int first, int count) {
            hip(launch_add(residual + first * W, d.f(16) + first * W, count * W, alternate + first * W, d.flag(), d.stream));
        });
        std::swap(residual, alternate);
        capture_batch(d, "ple_residual", residual, W, tokens, 1);
    }
    std::span<const float> step_batch(std::span<const std::int32_t> window, bool verify = false, bool last_only = false) {
        if (!usable) throw std::runtime_error("Session invalidated; reset required");
        if (verify && (!checkpoints || window.empty() || window.size() > 3))
            throw std::invalid_argument("Session verify resources/length");
        // Entire argument preflight is outside the invalidating execution block:
        // no stream/flag/hash/cache/stat/logit/capture mutation on a bad suffix.
        if (window.empty() || window.size() > static_cast<std::size_t>(config.max_batch_tokens) || !window.data())
            throw std::invalid_argument("Session batch length");
        if (statistics.consumed_tokens > static_cast<std::uint64_t>(config.capacity) ||
            window.size() > static_cast<std::uint64_t>(config.capacity) - statistics.consumed_tokens)
            throw std::invalid_argument("Session batch capacity");
        for (const auto token : window) if (token < 0 || token >= V) throw std::invalid_argument("token ID");
        if (verify && checkpoints->state.pending) throw std::invalid_argument("Session verify already pending");
        if (checkpoints) {
            const auto& s = checkpoints->stats;
            // Rollback makes logical capacity insufficient as a cumulative work
            // bound. Reserve conservative room BEFORE accepted execution, so no
            // historical physical diagnostic can wrap after repeated rollback.
            const auto assignments = static_cast<std::uint64_t>(window.size()) * 48 * 10;
            const auto work_bytes = assignments * short_stage_bytes;
            auto room = [](std::uint64_t current, std::uint64_t increment) {
                if (increment > UINT64_MAX - current)
                    throw std::invalid_argument("Session completed-work diagnostic overflow");
            };
            room(statistics.expert_hits, assignments); room(statistics.expert_misses, assignments);
            room(statistics.expert_upload_bytes, work_bytes);
            room(route_statistics.expert_groups_gt128, assignments / 129);
            for (auto word : std::bit_cast<std::array<std::uint64_t, 26>>(hybrid_stats())) room(word, work_bytes);
            for (auto word : std::bit_cast<std::array<std::uint64_t, 6>>(attention_statistics)) room(word, assignments);
            if (window.size() > UINT64_MAX - s.target_forward_rows ||
                (verify && (s.verify_windows == UINT64_MAX || window.size() > UINT64_MAX - s.verify_rows ||
                 s.gdn_prefix_calls > UINT64_MAX - 36 || s.ple_prefix_calls == UINT64_MAX ||
                 s.qsa_tail_prefix_calls > UINT64_MAX - 12)))
                throw std::invalid_argument("Session speculative diagnostic overflow");
        }
        const int tokens = static_cast<int>(window.size());
        const int position = static_cast<int>(statistics.consumed_tokens);
        const int output_first = last_only ? tokens - 1 : 0;
        const int output_rows = last_only ? 1 : tokens;
        const bool trace_observations = trace_metadata.is_open() &&
            statistics.consumed_tokens + window.size() > static_cast<std::uint64_t>(config.trace_first_token);
        // Diagnostic checks never clear the sticky flag. Their labels identify
        // the first observed failing interval, not necessarily its last kernel.
        const bool trace_stages = trace_observations && position >= config.trace_first_token;
        const bool transactional = hybrid_policy.mode != SessionHybridMode::disabled;
        const bool publication_staging = transactional || checkpoints;
        // Reserve an actual allowed physical representative while coordinating
        // short CPU jobs. Restore caller affinity; never infer cores from threads.
        CoordinatorAffinity coordinator(transactional && tokens <= 3 ? coordinator_cpu : -1);
        SessionStats working_stats = statistics;
        auto& execution_stats = publication_staging ? working_stats : statistics;
        SessionHybridStats working_hybrid = hybrid_stats();
        SessionAttentionStats working_attention = attention_statistics;
        const auto cpu_return_baseline = working_hybrid.cpu_return_bytes;
        const auto begin = std::chrono::steady_clock::now();
        try {
            if (checkpoints) {
                if (cpu_pool) cpu_ok(cpu_pool->drain());
                for (auto& device : devices) device->drain();
                // Argument rejection above has not touched slots/publication.
                // Physical prefix storage is exclusive to this accepted attempt;
                // previous logits/tap remain published until full-call success.
                checkpoints->state = {true, false, 0, 0, 0};
                checkpoints->prefix_calls = {};
            }
            SessionRouteStats call_routes{};
            const bool capture_inputs = config.hybrid_probe && tokens <= 3;
            if (capture_inputs) devices[0]->hybrid->probe_inputs[1].begin(tokens);
            auto* d = devices[0].get(); hip(hipSetDevice(0)); d->clear();
            for (int t = 0; t < tokens; ++t) {
                const auto row = std::span(embeddings).subspan(embedding_stride * window[t], embedding_stride);
                for (int i = 0; i < H; ++i) {
                    const float x = tensor_element(embedding_type, row, static_cast<std::size_t>(i));
                    if (!std::isfinite(x)) throw std::runtime_error("embedding nonfinite");
                    for (int c = 0; c < 4; ++c) handoff[t * W + c * H + i] = x;
                }
            }
            float* residual = d->f(0); float* alternate = d->f(1);
            hip(hipMemcpyAsync(residual, handoff, tokens * W * sizeof(float), hipMemcpyHostToDevice, d->stream));
            capture_batch(*d, "hc_init", residual, W, tokens);
            for (int l = 0; l < 48; ++l) {
                if (l == 24) {
                    d->check("split producer");
                    hip(hipMemcpyAsync(handoff, residual, tokens * W * sizeof(float), hipMemcpyDeviceToHost, d->stream));
                    hip(hipStreamSynchronize(d->stream));
                    d = devices[1].get(); hip(hipSetDevice(1)); d->clear(); residual = d->f(0); alternate = d->f(1);
                    hip(hipMemcpyAsync(residual, handoff, tokens * W * sizeof(float), hipMemcpyHostToDevice, d->stream));
                }
                auto& layer = *layers[l];
                if (l == 1) ple(*d, window, residual, alternate, verify);
                layer.attention_hc.mix(*d, residual, epsilon, tokens);
                if (layer.failure_probe && trace_stages) d->check("attention HC layer 8");
                if (l == 0 || l == 1 || l == 2 || l == 3 || l == 6 || l == 8 || l == 19 || l == 22 || l == 47) {
                    capture_batch(*d, "hc_attn_mix", d->f(6), H, tokens, l);
                    capture_batch(*d, "hc_attn_norm", d->f(2), W, tokens, l);
                    capture_batch(*d, "hc_attn_gate_raw", d->f(5), W, tokens, l);
                    capture_batch(*d, "hc_attn_inject_raw", d->f(7), 4, tokens, l);
                }
                layer.attention(position, config.capacity, epsilon, tokens, trace_stages, batch_attention, working_attention,
                    verify ? &checkpoints->layers[l] : nullptr,
                    [&](const char* name, const float* x, int count, int token_offset = -1) {
                        if (l == 0 || l == 1 || l == 2 || l == 3 || l == 6 || l == 8 || l == 19 || l == 22 || l == 47) {
                            if (token_offset < 0) capture_batch(*d, name, x, count, tokens, l);
                            else capture(*d, name, x, count, l, token_offset);
                        }
                    });
                combine(*d, residual, d->f(9), tokens, alternate);
                if (layer.failure_probe && trace_stages) d->check("attention combine layer 8");
                std::swap(residual, alternate);
                layer.ffn_hc.mix(*d, residual, epsilon, tokens);
                if (layer.failure_probe && trace_stages) d->check("FFN HC layer 8");
                if (l == 0 || l == 1 || l == 2 || l == 3 || l == 6 || l == 8 || l == 19 || l == 22 || l == 47) {
                    capture_batch(*d, "hc_ffn_mix", d->f(6), H, tokens, l);
                    capture_batch(*d, "hc_ffn_norm", d->f(2), W, tokens, l);
                    capture_batch(*d, "hc_ffn_gate_raw", d->f(5), W, tokens, l);
                    capture_batch(*d, "hc_ffn_inject_raw", d->f(7), 4, tokens, l);
                }
                const bool capture_probe = config.hybrid_probe && tokens <= 3;
                layer.ffn(execution_stats, call_routes, tokens, trace_observations, cpu_pool.get(), hybrid_policy, working_hybrid,
                    config.hybrid_probe ? &probe_diagnostics : nullptr, cpu_return_baseline,
                    capture_probe ? probe_down[1].data() + l * 10 * H : nullptr,
                    [&](const std::string& name, const float* x, int width, int token_offset = -1, int stride = 0) {
                        if (l != 0 || !trace_metadata.is_open()) return;
                        if (token_offset >= 0) capture(*d, name, x, width, l, token_offset);
                        else for (int t = 0; t < tokens; ++t)
                            capture(*d, name, x + t * (stride ? stride : width), width, l, t);
                    });
                if (capture_inputs) devices[0]->hybrid->probe_inputs[1].capture(*d, l, tokens, transactional);
                if (capture_probe)
                    for (int t = 0; t < tokens; ++t)
                        hip(hipMemcpyAsync(probe_ffn[1].data() + (t * 48 + l) * H, d->f(20) + t * H,
                            H * sizeof(float), hipMemcpyDeviceToHost, d->stream));
                if (l == 0 || l == 1 || l == 2 || l == 3 || l == 6 || l == 8 || l == 19 || l == 22 || l == 47) capture_batch(*d, "ffn_out", d->f(20), H, tokens, l);
                combine(*d, residual, d->f(20), tokens, alternate);
                std::swap(residual, alternate);
                d->check("layer " + std::to_string(l));
                layer.experts.finish_admissions(true); // Actual completion + successful entire layer.
                capture_batch(*d, "l_last-" + std::to_string(l), residual, W, tokens, l);
            }
            if (checkpoints) {
                // Stage the actual layer47 widened combine BEFORE root/head HC.
                // Never overwrite the published tap on a failed network/head or
                // copy. Head check below synchronizes this same ordered stream.
                hip(hipMemcpyAsync(checkpoints->staged_tap.get(), residual,
                    checked_size(checked_size(tokens, W), sizeof(float)), hipMemcpyDeviceToDevice, d->stream));
            }
            head_hc->mix(*d, residual, epsilon, tokens, true); capture_batch(*d, "result_norm", d->f(6), H, tokens);
            auto& destination = publication_staging ? working_logits : host_logits;
            head_to_host(*d, d->f(6) + output_first * H, destination.data(), output_rows);
            d->check("head");
            for (float v : std::span(destination).first(output_rows * V)) if (!std::isfinite(v)) throw std::runtime_error("logits nonfinite");
            // Each counted group consumes >=129 of 48*10 assignments per token.
            // Default capacity bounds the cumulative count; opt-in rollback uses
            // the preflight completed-work bounds above. Publication cannot fail.
            call_routes.expert_groups_gt128 += route_statistics.expert_groups_gt128;
            execution_stats.consumed_tokens += static_cast<std::uint64_t>(tokens);
            execution_stats.last_completed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            if (publication_staging) { statistics = working_stats; host_logits.swap(working_logits); }
            route_statistics = call_routes;
            attention_statistics = working_attention;
            publish_hybrid(working_hybrid);
            if (config.hybrid_probe && tokens <= 3) {
                probe_down[0].swap(probe_down[1]); probe_ffn[0].swap(probe_ffn[1]); probe_tokens = tokens;
            } else probe_tokens = 0;
            if (capture_inputs) std::swap(devices[0]->hybrid->probe_inputs[0], devices[0]->hybrid->probe_inputs[1]);
            if (checkpoints) {
                std::swap(checkpoints->tap, checkpoints->staged_tap); // noexcept, after ALL execution gates.
                checkpoints->tap_rows = tokens; checkpoints->tap_position = static_cast<std::uint64_t>(position);
                auto& s = checkpoints->stats; s.target_forward_rows += static_cast<std::uint64_t>(tokens);
                if (verify) {
                    ++s.verify_windows; s.verify_rows += static_cast<std::uint64_t>(tokens);
                    s.gdn_prefix_calls += checkpoints->prefix_calls[0];
                    s.ple_prefix_calls += checkpoints->prefix_calls[1];
                    s.qsa_tail_prefix_calls += checkpoints->prefix_calls[2];
                    checkpoints->state = {true, true, static_cast<std::uint64_t>(position), tokens, tokens + 1};
                }
            }
            return {host_logits.data(), static_cast<std::size_t>(output_rows) * V};
        } catch (...) {
            usable = false;
            if (cpu_pool) (void)cpu_pool->drain(); // Stop owner submissions; gate failed batch globally.
            if (probe_diagnostics.synthetic_failure) probe_diagnostics.cpu_pool_drained = true;
            // In-flight DMA may still read pinned bands, RAM or resident slots.
            // Finish both streams before reset/destruction can release any owner.
            for (auto& d : devices) d->drain_cleanup();
            if (probe_diagnostics.synthetic_failure) probe_diagnostics.gpu_streams_drained = true;
            for (auto& l : layers) l->experts.invalidate_cache();
            throw;
        }
    }
    // One owner/stream per layer range, absolute chronology and full residual.
    // Return only after all borrowed pinned buffers and GPU work are complete.
    int layerwise_layers(std::span<const std::int32_t> window, int position,
                     int layer_first, int layer_end, int tap_offset,
                     LayerwiseHost& p, float* transfer, SessionStats& working_stats,
                     SessionAttentionStats& working_attention, LayerwiseRouteCounts& route_counts,
                     SessionLayerwiseTransfers& transfers) {
        const int tokens = static_cast<int>(window.size());
        auto finite = [](const float* x, std::size_t count, const char* label) {
            for (std::size_t i = 0; i < count; ++i)
                if (!std::isfinite(x[i])) throw std::runtime_error(label);
        };
        auto load = [&](Device& d, const float* cpu, int count, const float* device_input = nullptr) {
            const auto bytes = static_cast<std::size_t>(count) * W * sizeof(float);
            if (device_input) {
                hip(hipMemcpyAsync(d.f(0), device_input, bytes, hipMemcpyDeviceToDevice, d.stream));
                transfers.residual_d2d_bytes += bytes;
            } else {
                std::copy_n(cpu, static_cast<std::size_t>(count) * W, transfer);
                hip(hipMemcpyAsync(d.f(0), transfer, bytes, hipMemcpyHostToDevice, d.stream));
                transfers.residual_h2d_bytes += bytes;
            }
            // PLE and the next CPU staging refill both use transfer. Protect
            // its pinned H2D lifetime even when PLE starts this frame.
            ++transfers.loop_barriers; d.check("layerwise residual upload");
        };
        auto download = [&](Device& d, const float* gpu, float* cpu, int count, float* device_output = nullptr) {
            const auto bytes = static_cast<std::size_t>(count) * W * sizeof(float);
            if (device_output) {
                // Both attention and FFN residual producers are hc_combine:
                // it validates every operand and arithmetic result with the
                // same sticky flag. Keep the completed error check here.
                hip(hipMemcpyAsync(device_output, gpu, bytes, hipMemcpyDeviceToDevice, d.stream));
                transfers.residual_d2d_bytes += bytes;
                ++transfers.loop_barriers; d.check("layerwise device residual completion");
                return;
            }
            hip(hipMemcpyAsync(transfer, gpu, bytes, hipMemcpyDeviceToHost, d.stream));
            transfers.residual_d2h_bytes += bytes;
            ++transfers.loop_barriers; d.check("layerwise residual completion");
            finite(transfer, static_cast<std::size_t>(count) * W, "layerwise residual nonfinite");
            std::copy_n(transfer, static_cast<std::size_t>(count) * W, cpu);
        };
        float* resident = nullptr;
        if (config.prefill_residual_device) {
            if (layer_first < 0 || layer_end > 48 || layer_first >= layer_end ||
                layer_first / 24 != (layer_end - 1) / 24 || tokens > layerwise->stage_capacity)
                throw std::runtime_error("device residual stage ownership/extent");
            auto& d = *devices[layer_first / 24];
            hip(hipSetDevice(d.id)); d.clear();
            auto& buffer = layerwise->gpu[d.id]->residual;
            if (buffer.bytes() < static_cast<std::size_t>(tokens) * W * sizeof(float))
                throw std::runtime_error("device residual capacity");
            resident = buffer.get();
            for (int first = 0; first < tokens; first += config.max_batch_tokens) {
                const int count = std::min(config.max_batch_tokens, tokens - first);
                const auto offset = static_cast<std::size_t>(first) * W;
                const auto bytes = static_cast<std::size_t>(count) * W * sizeof(float);
                std::copy_n(p.activations[0].data() + offset, static_cast<std::size_t>(count) * W, transfer);
                hip(hipMemcpyAsync(resident + offset, transfer, bytes, hipMemcpyHostToDevice, d.stream));
                transfers.residual_h2d_bytes += bytes;
                ++transfers.loop_barriers; d.check("layerwise device stage entry");
            }
        }
        auto no_attention_observation = [](const char*, const float*, int, int = -1) {};
        auto no_ffn_observation = [](const std::string&, const float*, int, int = -1, int = 0) {};
        int current = 0;
        for (int l = layer_first; l < layer_end; ++l) {
            auto& layer = *layers[l]; auto& d = layer.device;
            auto& g = *layerwise->gpu[d.id];
            hip(hipSetDevice(d.id)); d.clear();
            auto& input = p.activations[current]; auto& output = p.activations[1 - current];
            // Phase 1: chronological HC/attention/PLE/router frames. GDN
            // mutates once in order; QSA retains its COMPLETE history and
            // uses position+first+t+1, never a frame-local causal horizon.
            // input becomes the POST-attention residual; output is untouched.
            for (int first = 0; first < tokens; first += config.max_batch_tokens) {
                const int count = std::min(config.max_batch_tokens, tokens - first);
                const auto row = static_cast<std::size_t>(first);
                load(d, input.data() + row * W, count, resident ? resident + row * W : nullptr);
                float* residual = d.f(0); float* alternate = d.f(1);
                if (l == 1) ple(d, window.subspan(row, count), residual, alternate, false);
                layer.attention_hc.mix(d, residual, epsilon, count);
                layer.attention(position + first, config.capacity, epsilon, count, false,
                    batch_attention, working_attention, nullptr, no_attention_observation);
                combine(d, residual, d.f(9), count, alternate); std::swap(residual, alternate);
                layer.ffn_hc.mix(d, residual, epsilon, count);
                layer.router.apply(d, d.f(6), d.f(9), count);
                tiles(count, [&](int local, int n) {
                    hip(launch_router_softmax(d.f(9) + local * 512, n, d.f(10) + local * 512, d.flag(), d.stream));
                    hip(launch_quantize_q8_1(d.f(6) + local * H, d.q8.get<Q8_1>() + local * (H / 32),
                        H, n, d.flag(), d.stream));
                });
                hip(hipMemcpyAsync(p.probabilities.data() + row * 512, d.f(10),
                    static_cast<std::size_t>(count) * 512 * sizeof(float), hipMemcpyDeviceToHost, d.stream));
                transfers.router_d2h_bytes += static_cast<std::size_t>(count) * 512 * sizeof(float);
                hip(hipMemcpyAsync(g.original.get<Q8_1>() + row * (H / 32), d.q8.get<Q8_1>(),
                    static_cast<std::size_t>(count) * (H / 32) * sizeof(Q8_1), hipMemcpyDeviceToDevice, d.stream));
                download(d, residual, input.data() + row * W, count, resident ? resident + row * W : nullptr);
                // Retain exact f6/f8 bytes; recomputing HC is unnecessary and
                // could conflate its arithmetic with expert traffic savings.
                hip(hipMemcpyAsync(transfer, d.f(6), static_cast<std::size_t>(count) * H * sizeof(float), hipMemcpyDeviceToHost, d.stream));
                hip(hipMemcpyAsync(transfer + count * H, d.f(8), static_cast<std::size_t>(count) * 4 * sizeof(float), hipMemcpyDeviceToHost, d.stream));
                transfers.ffn_d2h_bytes += static_cast<std::size_t>(count) * (H + 4) * sizeof(float);
                ++transfers.loop_barriers; d.check("layerwise FFN HC/router completion");
                finite(transfer, static_cast<std::size_t>(count) * (H + 4), "layerwise FFN HC nonfinite");
                finite(p.probabilities.data() + row * 512, static_cast<std::size_t>(count) * 512, "layerwise router nonfinite");
                std::copy_n(transfer, static_cast<std::size_t>(count) * H, p.mixed.data() + row * H);
                std::copy_n(transfer + count * H, static_cast<std::size_t>(count) * 4, p.injection.data() + row * 4);
            }
            select_routes(p.probabilities.data(), p.ids.data(), p.weights.data(), tokens);
            p.groups.prepare(std::span(p.ids).first(tokens * 10), std::span(p.weights).first(tokens * 10), tokens);
            for (int expert = 0; expert < 512; ++expert)
                route_counts[l][expert] += static_cast<std::uint32_t>(p.groups.counts()[expert]);
            p.upload_routes(d, g, tokens);
            transfers.route_h2d_bytes += static_cast<std::size_t>(tokens) * 10 * sizeof(MoeRouteIndex);
            // ONE physical acquisition for each active expert/layer, all
            // actual assignments (including >128) use qualified selective tiles<=128.
            // Original all-N Q8 stays immutable until every scatter ends.
            layer.experts.run_chunk(d, g.original.get<Q8_1>(), working_stats, &g, &p.groups, tokens);
            ++transfers.loop_barriers; d.check("layerwise global expert completion");
            // Phase 2: preserve original-rank separately rounded MUL+ADD,
            // shared mix, then the original widened HC FFN combine.
            for (int first = 0; first < tokens; first += config.max_batch_tokens) {
                const int count = std::min(config.max_batch_tokens, tokens - first);
                const auto row = static_cast<std::size_t>(first);
                load(d, input.data() + row * W, count, resident ? resident + row * W : nullptr);
                std::copy_n(p.mixed.data() + row * H, static_cast<std::size_t>(count) * H, transfer);
                std::copy_n(p.injection.data() + row * 4, static_cast<std::size_t>(count) * 4, transfer + count * H);
                hip(hipMemcpyAsync(d.f(6), transfer, static_cast<std::size_t>(count) * H * sizeof(float), hipMemcpyHostToDevice, d.stream));
                hip(hipMemcpyAsync(d.f(8), transfer + count * H, static_cast<std::size_t>(count) * 4 * sizeof(float), hipMemcpyHostToDevice, d.stream));
                hip(hipMemcpyAsync(d.f(22), p.weights.data() + row * 10,
                    static_cast<std::size_t>(count) * 10 * sizeof(float), hipMemcpyHostToDevice, d.stream));
                transfers.ffn_h2d_bytes += static_cast<std::size_t>(count) * (H + 4) * sizeof(float);
                transfers.route_h2d_bytes += static_cast<std::size_t>(count) * 10 * sizeof(float);
                tiles(count, [&](int local, int n) {
                    hip(launch_routed_fold(g.contributions.get() + (row + local) * 10 * H,
                        d.f(22) + local * 10, n, d.f(20) + local * H, d.flag(), d.stream));
                });
                layer.shared_project(count, false, no_ffn_observation);
                layer.shared_add(count);
                combine(d, d.f(0), d.f(20), count, d.f(1));
                if (l == 47 && checkpoints)
                    hip(hipMemcpyAsync(checkpoints->staged_tap.get() + (static_cast<std::size_t>(tap_offset) + row) * W, d.f(1),
                        static_cast<std::size_t>(count) * W * sizeof(float), hipMemcpyDeviceToDevice, d.stream));
                download(d, d.f(1), output.data() + row * W, count, resident ? resident + row * W : nullptr);
            }
            current = 1 - current;
        }
        if (resident) {
            auto& d = *devices[layer_first / 24];
            for (int first = 0; first < tokens; first += config.max_batch_tokens) {
                const int count = std::min(config.max_batch_tokens, tokens - first);
                const auto offset = static_cast<std::size_t>(first) * W;
                download(d, resident + offset, p.activations[current].data() + offset, count);
            }
        }
        return current;
    }

    void layerwise_embed(std::span<const std::int32_t> window, LayerwiseHost& p) {
        const int tokens = static_cast<int>(window.size());
        for (int t = 0; t < tokens; ++t) {
            const auto row = std::span(embeddings).subspan(embedding_stride * window[t], embedding_stride);
            for (int i = 0; i < H; ++i) {
                const float x = tensor_element(embedding_type, row, static_cast<std::size_t>(i));
                if (!std::isfinite(x)) throw std::runtime_error("embedding nonfinite");
                for (int c = 0; c < 4; ++c) p.activations[0][static_cast<std::size_t>(t) * W + c * H + i] = x;
            }
        }
    }
    void head_to_host(Device& d, const float* normalized, float* destination, int rows) {
        const auto capacity = d.logits.bytes() / (static_cast<std::size_t>(V) * sizeof(float));
        if (capacity < 1 || capacity > static_cast<std::size_t>(microtile_tokens) ||
            rows < 1 || rows > config.max_batch_tokens)
            throw std::runtime_error("Session head output tile extent");
        const int tile = static_cast<int>(capacity);
        for (int first = 0; first < rows; first += tile) {
            const int count = std::min(tile, rows - first);
            head.apply(d, normalized + static_cast<std::size_t>(first) * H, d.logits.get(), count);
            // Same stream: this copy finishes before the next tile overwrites
            // d.logits. Caller checks/drains before host validation/publication.
            hip(hipMemcpyAsync(destination + static_cast<std::size_t>(first) * V,
                d.logits.get(), static_cast<std::size_t>(count) * V * sizeof(float),
                hipMemcpyDeviceToHost, d.stream));
        }
    }
    void layerwise_head(LayerwiseHost& p, int current, float* transfer,
                        int first, int count, float* destination) {
        auto& d = *devices[1]; hip(hipSetDevice(1));
        std::copy_n(p.activations[current].data() + static_cast<std::size_t>(first) * W,
            static_cast<std::size_t>(count) * W, transfer);
        hip(hipMemcpyAsync(d.f(0), transfer, static_cast<std::size_t>(count) * W * sizeof(float), hipMemcpyHostToDevice, d.stream));
        d.check("layerwise head upload");
        head_hc->mix(d, d.f(0), epsilon, count, true);
        head_to_host(d, d.f(6), destination, count);
        d.check("layerwise head completion");
    }
    void pipeline_fault(int stage, int window) {
        // Only the configured stage thread reads/writes pending. The caller
        // observes it only after coordinator completion/cancel joins the callback.
        if (config.prefill_pipeline_fail_stage == stage && pipeline_fault_pending &&
            config.prefill_pipeline_fail_window == window) {
            pipeline_fault_pending = false;
            throw std::runtime_error("injected prefill pipeline stage failure");
        }
    }
    void pipeline_produce(int index, int slot) {
        try {
            auto& task = *pipeline_call;
            const int first = index * config.prefill_pipeline_tokens;
            const int count = std::min(config.prefill_pipeline_tokens, static_cast<int>(task.window.size()) - first);
            auto window = task.window.subspan(first, count);
            auto& p = *layerwise;
            layerwise_embed(window, p);
            const int current = layerwise_layers(window, task.position + first, 0, 24, first,
                p, handoff, task.stats[0], task.attention[0], *task.routes, task.transfers[0]);
            pipeline_fault(0, index);
            p.activations[current].swap(p.ring[slot]);
        } catch (...) {
            devices[0]->drain_cleanup();
            throw;
        }
    }

    std::span<const float> prefill_layerwise(std::span<const std::int32_t> window, Session::OutputMode mode) {
        if (!usable) throw std::runtime_error("Session invalidated; reset required");
        // ALL argument/resource/diagnostic preflight precedes owner or stream
        // mutation, including a bad FINAL token and pending verify ownership.
        if (!layerwise || window.empty() || !window.data() ||
            window.size() > static_cast<std::size_t>(config.layerwise_prefill_capacity) ||
            (mode != Session::OutputMode::all_rows && mode != Session::OutputMode::last_row) ||
            hybrid_policy.mode != SessionHybridMode::disabled ||
            (checkpoints && checkpoints->state.pending))
            throw std::invalid_argument("Session layerwise resources/length/mode/policy/pending verify");
        if (statistics.consumed_tokens > static_cast<std::uint64_t>(config.capacity) ||
            window.size() > static_cast<std::uint64_t>(config.capacity) - statistics.consumed_tokens)
            throw std::invalid_argument("Session layerwise capacity");
        for (const auto token : window) if (token < 0 || token >= V) throw std::invalid_argument("token ID");
        const auto assignments = static_cast<std::uint64_t>(window.size()) * 48 * 10;
        auto room = [](std::uint64_t current, std::uint64_t increment) {
            if (increment > UINT64_MAX - current) throw std::invalid_argument("Session completed-work diagnostic overflow");
        };
        room(statistics.expert_hits, assignments); room(statistics.expert_misses, assignments);
        room(statistics.expert_upload_bytes, assignments * short_stage_bytes);
        room(route_statistics.expert_groups_gt128, assignments / 129);
        for (auto word : std::bit_cast<std::array<std::uint64_t, 6>>(attention_statistics)) room(word, assignments);
        if (checkpoints) room(checkpoints->stats.target_forward_rows, window.size());
        const auto transfer_bound = static_cast<std::uint64_t>(window.size()) * 48 * 4 * W * sizeof(float);
        for (auto word : std::bit_cast<std::array<std::uint64_t, 8>>(transfer_statistics)) room(word, transfer_bound);
        const int tokens = static_cast<int>(window.size());
        const int position = static_cast<int>(statistics.consumed_tokens);
        const int result_rows = mode == Session::OutputMode::all_rows ? tokens : 1;
        const auto result_elements = checked_size(result_rows, V);
        auto& p = *layerwise;
        SessionStats working_stats = statistics;
        SessionAttentionStats working_attention = attention_statistics;
        SessionLayerwiseTransfers working_transfers = transfer_statistics;
        SessionRouteStats call_routes{};
        LayerwiseRouteCounts route_counts{};
        const auto begin = std::chrono::steady_clock::now();
        try {
            if (cpu_pool) cpu_ok(cpu_pool->drain());
            for (auto& d : devices) d->drain();
            if (checkpoints) {
                checkpoints->state = {true, false, 0, 0, 0};
                checkpoints->prefix_calls = {}; // Ordinary PP NEVER writes prefix slots.
            }
            // Allocation is an explicit OUTPUT decision, never a constructor N*V
            // default. Checkpoint users preserve the old borrowed publication in
            // host_logits until every gate succeeds (same two-owner swap as R6).
            auto& destination = (checkpoints || prefill_pipeline) ? working_logits : host_logits;
            if (destination.size() < result_elements) destination.resize(result_elements);
            auto finite = [](const float* x, std::size_t count, const char* label) {
                for (std::size_t i = 0; i < count; ++i)
                    if (!std::isfinite(x[i])) throw std::runtime_error(label);
            };
            if (prefill_pipeline) {
                PipelineCall task{window, position, {}, {}, {}, &route_counts};
                pipeline_call = &task;
                // Ensure task remains alive until even a failed callback has
                // drained. This scope owns the borrowed task, not GPU payload.
                try {
                    const int windows = (tokens + config.prefill_pipeline_tokens - 1) / config.prefill_pipeline_tokens;
                    prefill_pipeline->begin(windows);
                    for (int index = 0; index < windows; ++index) {
                        const int first = index * config.prefill_pipeline_tokens;
                        const int count = std::min(config.prefill_pipeline_tokens, tokens - first);
                        const int slot = prefill_pipeline->acquire(index);
                        auto& h = *p.stage1;
                        h.activations[0].swap(p.ring[slot]);
                        // Previous stage1 contents are fully drained. Swapping
                        // transfers vector ownership, never a live DMA address.
                        prefill_pipeline->release(index);
                        const int current = layerwise_layers(window.subspan(first, count), position + first,
                            24, 48, first, h, p.stage1_transfer, task.stats[1], task.attention[1], route_counts, task.transfers[1]);
                        pipeline_fault(1, index);
                        const bool all = mode == Session::OutputMode::all_rows;
                        if (all || first + count == tokens) {
                            const int head_first = all ? 0 : count - 1;
                            for (int row = head_first; row < count; row += config.max_batch_tokens) {
                                const int n = std::min(config.max_batch_tokens, count - row);
                                layerwise_head(h, current, p.stage1_transfer, row, n,
                                    destination.data() + (all ? static_cast<std::size_t>(first + row) * V : 0));
                            }
                        }
                    }
                    prefill_pipeline->finish();
                } catch (...) {
                    prefill_pipeline->cancel_and_drain();
                    pipeline_call = nullptr;
                    throw;
                }
                pipeline_call = nullptr;
                for (int stage = 0; stage < 2; ++stage) {
                    working_stats.expert_hits += task.stats[stage].expert_hits;
                    working_stats.expert_misses += task.stats[stage].expert_misses;
                    working_stats.expert_upload_bytes += task.stats[stage].expert_upload_bytes;
                    auto sum = std::bit_cast<std::array<std::uint64_t, 6>>(working_attention);
                    const auto added = std::bit_cast<std::array<std::uint64_t, 6>>(task.attention[stage]);
                    for (int i = 0; i < 5; ++i) sum[i] += added[i];
                    sum[5] = std::max(sum[5], added[5]);
                    working_attention = std::bit_cast<SessionAttentionStats>(sum);
                    auto copies = std::bit_cast<std::array<std::uint64_t, 8>>(working_transfers);
                    const auto extra = std::bit_cast<std::array<std::uint64_t, 8>>(task.transfers[stage]);
                    for (int i = 0; i < 8; ++i) copies[i] += extra[i];
                    working_transfers = std::bit_cast<SessionLayerwiseTransfers>(copies);
                }
            } else {
                layerwise_embed(window, p);
                const int current = layerwise_layers(window, position, 0, 48, 0,
                    p, handoff, working_stats, working_attention, route_counts, working_transfers);
                const int head_first = mode == Session::OutputMode::all_rows ? 0 : tokens - 1;
                for (int first = head_first; first < tokens; first += config.max_batch_tokens) {
                    const int count = std::min(config.max_batch_tokens, tokens - first);
                    layerwise_head(p, current, handoff, first, count,
                        destination.data() + static_cast<std::size_t>(first - head_first) * V);
                }
            }
            finite(destination.data(), result_elements, "logits nonfinite");
            for (const auto& layer_counts : route_counts) for (const auto count : layer_counts) {
                call_routes.last_max_expert_group_assignments = std::max(
                    call_routes.last_max_expert_group_assignments, static_cast<std::uint64_t>(count));
                if (count > 128) ++call_routes.expert_groups_gt128;
            }
            call_routes.expert_groups_gt128 += route_statistics.expert_groups_gt128;
            working_stats.consumed_tokens += static_cast<std::uint64_t>(tokens);
            working_stats.last_completed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            if (checkpoints || prefill_pipeline) host_logits.swap(working_logits);
            if (checkpoints) {
                std::swap(checkpoints->tap, checkpoints->staged_tap);
                checkpoints->tap_rows = tokens; checkpoints->tap_position = static_cast<std::uint64_t>(position);
                checkpoints->stats.target_forward_rows += static_cast<std::uint64_t>(tokens);
            }
            statistics = working_stats; route_statistics = call_routes; attention_statistics = working_attention;
            transfer_statistics = working_transfers;
            probe_tokens = 0; // Optional short-input probes must not expose stale rows.
            return {host_logits.data(), result_elements};
        } catch (...) {
            usable = false;
            if (cpu_pool) (void)cpu_pool->drain();
            for (auto& d : devices) d->drain_cleanup();
            for (auto& l : layers) l->experts.invalidate_cache();
            throw;
        }
    }

    void restore_prefix(int retained) {
        if (!usable) throw std::runtime_error("Session invalidated; reset required");
        if (!checkpoints || !checkpoints->state.pending || retained < 0 || retained > checkpoints->state.inputs)
            throw std::invalid_argument("Session restore pending window/prefix");
        auto& c = *checkpoints;
        if (c.stats.restore_calls == UINT64_MAX || c.stats.restored_prefixes[retained] == UINT64_MAX ||
            static_cast<std::uint64_t>(retained) > UINT64_MAX - c.stats.retained_inputs)
            throw std::invalid_argument("Session restore diagnostic overflow");
        usable = false;
        try {
            if (cpu_pool) cpu_ok(cpu_pool->drain());
            for (auto& d : devices) d->drain();
            for (int l = 0; l < 48; ++l) {
                auto& layer = *layers[l]; auto& d = layer.device;
                hip(hipSetDevice(d.id));
                const auto& p = c.layers[l];
                auto copy = [&](const Buffer& prefix, const Buffer& active) {
                    if (!active.bytes()) return;
                    // Slot stride is exactly the mutable state's allocation,
                    // not a reversed accepted-draft index or an entire KV copy.
                    const auto offset = checked_size(static_cast<std::size_t>(retained), active.bytes());
                    if (offset > prefix.bytes() || active.bytes() > prefix.bytes() - offset)
                        throw std::runtime_error("Session checkpoint slot extent");
                    hip(hipMemcpyAsync(active.get<void>(), prefix.get<std::byte>() + offset,
                        active.bytes(), hipMemcpyDeviceToDevice, d.stream));
                };
                copy(p.recurrent, layer.recurrent); copy(p.history, layer.history); copy(p.tail, layer.tail);
            }
            hip(hipSetDevice(0));
            hip(hipMemcpyAsync(ple_history.get(), c.ple_history.get() + retained * ple_device_history_elements,
                ple_history.bytes(), hipMemcpyDeviceToDevice, devices[0]->stream));
            for (auto& d : devices) d->drain();
            ple_hash.restore(c.hash[retained]);
            // Main K/V remain Q4_0. Only consumed cursor drives append addresses,
            // pooled completed-block count, per-query select and causal gather.
            // Hidden suffix is never read and is overwritten when consumed again.
            statistics.consumed_tokens = c.state.start_position + static_cast<std::uint64_t>(retained);
            ++c.stats.restore_calls; ++c.stats.restored_prefixes[retained];
            c.stats.retained_inputs += static_cast<std::uint64_t>(retained);
            c.state = {true, false, 0, 0, 0}; usable = true;
        } catch (...) {
            if (cpu_pool) (void)cpu_pool->drain();
            for (auto& d : devices) d->drain_cleanup();
            throw;
        }
    }
};

Session::Session(const std::string& path, SessionConfig config) : impl_(std::make_unique<Impl>(path, std::move(config))) {}
Session::~Session() = default;
std::span<const float> Session::step(std::int32_t token) { return step_batch(std::span(&token, 1)); }
std::span<const float> Session::step_batch(std::span<const std::int32_t> tokens) { return impl_->step_batch(tokens); }
std::span<const float> Session::prefill_layerwise(std::span<const std::int32_t> tokens, OutputMode mode) {
    return impl_->prefill_layerwise(tokens, mode);
}
std::span<const float> Session::prefill_last(std::span<const std::int32_t> tokens) { return impl_->step_batch(tokens, false, true); }
std::span<const float> Session::verify_window(std::span<const std::int32_t> tokens) { return impl_->step_batch(tokens, true); }
void Session::restore_prefix(int retained_inputs) { impl_->restore_prefix(retained_inputs); }
SessionCheckpointState Session::checkpoint_state() const {
    if (!impl_->usable) throw std::runtime_error("Session invalidated; reset required");
    return impl_->checkpoints ? impl_->checkpoints->state : SessionCheckpointState{};
}
SessionTargetTap Session::target_tap() const {
    if (!impl_->checkpoints || !impl_->checkpoints->tap_rows) return {};
    const auto& c = *impl_->checkpoints;
    return {1, c.tap.get(), c.tap_rows, W, c.tap_position};
}
SessionSharedWeights Session::borrow_mtp_weights() const {
    if (!impl_->usable) throw std::runtime_error("Session invalidated; reset required");
    if (impl_->head.type != TensorType::Q6_K || impl_->head.data.owner() != 1 ||
        impl_->head.input != H || impl_->head.output != V || impl_->embedding_type != TensorType::Q4_0)
        throw std::runtime_error("MTP requires actual Q6_K head/Q4_0 embedding");
    int previous = 0; hip(hipGetDevice(&previous));
    try { for (auto& d : impl_->devices) d->drain(); }
    catch (...) { cleanup(hipSetDevice(previous)); throw; }
    hip(hipSetDevice(previous));
    return {&impl_->model,
        {1, H, V, impl_->head.data.get<void>(), impl_->head.data.bytes()},
        {impl_->embeddings, impl_->embedding_stride, H, V}};
}
SessionSpeculativeStats Session::speculative_stats() const {
    return impl_->checkpoints ? impl_->checkpoints->stats : SessionSpeculativeStats{};
}
void Session::reset() { impl_->reset(); }
SessionStats Session::stats() const { return impl_->statistics; }
SessionRouteStats Session::route_stats() const { return impl_->route_statistics; }
void Session::set_attention_batch(bool enabled) { impl_->set_attention_batch(enabled); }
bool Session::attention_batch() const { return impl_->batch_attention; }
SessionAttentionStats Session::attention_stats() const { return impl_->attention_statistics; }
SessionLayerwiseTransfers Session::layerwise_transfers() const { return impl_->transfer_statistics; }
void Session::set_hybrid_policy(SessionHybridPolicy policy) { impl_->set_hybrid_policy(policy); }
SessionHybridPolicy Session::hybrid_policy() const { return impl_->hybrid_policy; }
SessionHybridStats Session::hybrid_stats() const { return impl_->hybrid_stats(); }
SessionHybridIntermediates Session::hybrid_intermediates() const {
    return {std::span(impl_->probe_down[0]).first(static_cast<std::size_t>(impl_->probe_tokens) * 48 * 10 * H),
        std::span(impl_->probe_ffn[0]).first(static_cast<std::size_t>(impl_->probe_tokens) * 48 * H)};
}
SessionHybridInputs Session::hybrid_inputs() const {
    if (!impl_->config.hybrid_probe || !impl_->probe_tokens) return {};
    const auto count = static_cast<std::size_t>(impl_->probe_tokens) * 48;
    const auto& p = impl_->devices[0]->hybrid->probe_inputs[0];
    return {std::span(p.original).first(count * (H / 32)), std::span(p.ids).first(count * 10),
        std::span(p.weights).first(count * 10), std::span(p.available).first(count), std::span(p.cpu).first(count * 10)};
}
SessionHybridProbeDiagnostics Session::hybrid_probe_diagnostics() const {
    if (!impl_->config.hybrid_probe) return {};
    auto result = impl_->probe_diagnostics;
    for (const auto& layer : impl_->layers) {
        for (const auto& slot : layer->experts.map) result.ready_cache_ids += slot.id >= 0;
        for (int id : layer->experts.pending) result.pending_cache_ids += id >= 0;
    }
    return result;
}
SessionMemory Session::memory() const { return impl_->memory(); }

}
