#include "mtp_session.hpp"
#include "mtp_model.hpp"
#include "dense.hpp"
#include "routes.hpp"
#include "hip/blocks.cuh"
#include "hip/dense.cuh"
#include "hip/kv.cuh"
#include "hip/linear.cuh"
#include "hip/mtp_attention.cuh"
#include "hip/session_ops.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Trained MTP requires finite checks, no fast/finite-only math"
#endif
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx906__)
#error "MTP is specialized for gfx906 wave64"
#endif

// KEEP/ADAPT existing qualified gfx906 primitives, not a donor runtime:
// mx dcd685463d597d31f5ca759d32c94592a2740fa4 common/vecdotq/norm/topk-moe;
// furnace 905021dbad71c5056ef51f9fd45d545403fc989c dense attention/HC patterns;
// reinstinct 0b79e326351d90d4554a1c18df92da5d0ab692e8 route sort/gather/scatter.
// Licenses already retained in third_party; no new imported code/dependency.
// WRITE OURS: single-block ownership/history/transaction boundary and narrow
// embedding-first branch packing. Matrix/Hc mirror current Session math only.
// Compile HIP20 gfx906 with -ffp-contract=off, no fast/finite-only math.
// nextn.hnorm follows pinned mx's per2560 branch reduction with four distinct
// gamma slices; see prepare_kv and header. Full trained qualification is pending.

namespace qwen {
namespace {
constexpr int H = 2560, W = 10240, V = 248320, B = 3, R = 320, Q = 6144;
constexpr int E = 512, K = 10, I = 640, scratch_count = 39;
constexpr std::size_t scratch_elements = B * 12288;
static_assert(sizeof(float) == 4 && sizeof(Q4_0) == 18 && sizeof(Q8_1) == 36);

void hip_ok(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
void blas_ok(rocblas_status e) {
    if (e != rocblas_status_success) throw std::runtime_error("MTP rocBLAS status " + std::to_string(e));
}
void cleanup(hipError_t e) noexcept {
    if (e != hipSuccess) { std::fprintf(stderr, "MTP HIP cleanup: %s\n", hipGetErrorString(e)); std::abort(); }
}
std::size_t bytes(std::size_t n, std::size_t width) {
    if (width && n > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / width)
        throw std::overflow_error("MTP allocation range");
    return n * width;
}
MtpSessionConfig checked_config(MtpSessionConfig c) {
    if (c.capacity < 1 || c.capacity > mtp_attention_max_capacity)
        throw std::invalid_argument("MTP capacity must be 1..131072");
    if (c.diagnostic_first_row && !c.diagnostic_probe)
        throw std::invalid_argument("MTP first-row diagnostics require diagnostic_probe");
    if (c.diagnostic_probe_position < 0 || c.diagnostic_probe_position >= c.capacity)
        throw std::invalid_argument("MTP diagnostic position must be 0..capacity-1");
    if (c.diagnostic_probe_position && !c.diagnostic_first_row)
        throw std::invalid_argument("MTP nonzero diagnostic position requires first-row diagnostics");
    return c;
}
int diagnostic_probe_row(std::uint64_t first,int rows,int position) noexcept {
    if (rows<1 || rows>B || position<0 || first>std::uint64_t(position)) return -1;
    const auto relative=std::uint64_t(position)-first;
    return relative<std::uint64_t(rows) ? int(relative) : -1;
}
bool valid_row(MtpHiddenRow row, bool allow_zero) noexcept {
    if (row.device != 1) return false;
    if (!row.pointer) return allow_zero && row.elements == 0;
    const auto address = reinterpret_cast<std::uintptr_t>(row.pointer);
    return row.elements == W && address % alignof(float) == 0 &&
        address <= std::numeric_limits<std::uintptr_t>::max() - W * sizeof(float);
}
struct DeviceGuard {
    int previous = 0;
    DeviceGuard() { hip_ok(hipGetDevice(&previous)); hip_ok(hipSetDevice(1)); }
    ~DeviceGuard() { cleanup(hipSetDevice(previous)); }
};
struct Ledger { std::uint64_t live = 0, peak = 0, buffers = 0; };
class Buffer {
    void* pointer_ = nullptr;
    std::size_t bytes_ = 0;
    Ledger* ledger_ = nullptr;
    void release() noexcept {
        if (!pointer_) return;
        int previous = 0; cleanup(hipGetDevice(&previous)); cleanup(hipSetDevice(1));
        cleanup(hipFree(pointer_)); cleanup(hipSetDevice(previous));
        ledger_->live -= bytes_; --ledger_->buffers;
        pointer_ = nullptr; bytes_ = 0;
    }
public:
    Buffer() = default;
    Buffer(Ledger& ledger, std::size_t size) : bytes_(size), ledger_(&ledger) {
        if (!size) throw std::invalid_argument("MTP empty allocation");
        DeviceGuard device; hip_ok(hipMalloc(&pointer_, size));
        ledger.live += size; ++ledger.buffers; ledger.peak = std::max(ledger.peak, ledger.live);
    }
    ~Buffer() { release(); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept { *this = std::move(other); }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            release(); pointer_ = std::exchange(other.pointer_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0); ledger_ = other.ledger_;
        }
        return *this;
    }
    template<class T = float> T* get() const noexcept { return static_cast<T*>(pointer_); }
    std::size_t size() const noexcept { return bytes_; }
};
struct Context {
    hipStream_t stream = nullptr;
    rocblas_handle handle = nullptr;
    Context() {
        DeviceGuard device;
        hipDeviceProp_t p{}; hip_ok(hipGetDeviceProperties(&p, 1));
        if (p.warpSize != 64 || std::strncmp(p.gcnArchName, "gfx906", 6))
            throw std::runtime_error("MTP requires device1 gfx906 wave64");
        try {
            hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
            blas_ok(rocblas_create_handle(&handle));
        } catch (...) { release(); throw; }
    }
    void drain() const { hip_ok(hipStreamSynchronize(stream)); }
    void drain_cleanup() const noexcept { if (stream) cleanup(hipStreamSynchronize(stream)); }
    void release() noexcept {
        int previous = 0; cleanup(hipGetDevice(&previous)); cleanup(hipSetDevice(1));
        drain_cleanup();
        if (handle && rocblas_destroy_handle(handle) != rocblas_status_success) std::abort();
        if (stream) cleanup(hipStreamDestroy(stream));
        handle = nullptr; stream = nullptr; cleanup(hipSetDevice(previous));
    }
    ~Context() { release(); }
};

__device__ __forceinline__ float source(float x, int& flags) {
    if (!isfinite(x)) { flags |= 1; return 0.0f; }
    return x;
}
__global__ void pack_hidden(const float* input, float* output, bool zero_first, int* error) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= W) return;
    int flags = 0;
    const float x = input ? source(input[j], flags) : 0.0f;
    if (zero_first && x != 0.0f) flags |= 1;
    output[j] = x;
    if (flags) atomicOr(error, flags);
}

__global__ void branch_concat(const float* embedding, const float* hidden,
                              int n, int branch, float* output) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n * 2 * H) return;
    const int token = j / (2 * H), feature = j % (2 * H);
    // EMBEDDING FIRST. Repeat the same normalized embedding for all four branches.
    output[j] = feature < H ? embedding[token * H + feature] :
        hidden[token * W + branch * H + feature - H];
}
__global__ void branch_store(const float* input, int n, int branch, float* output) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n * H) output[(j / H) * W + branch * H + j % H] = input[j];
}
__global__ void finite_check(const float* input, int count, int* error) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < count && !isfinite(input[j])) atomicOr(error, 2);
}
void check_finite(const float* input, int count, int* error, hipStream_t stream) {
    hipLaunchKernelGGL(finite_check, dim3((count + 255) / 256), dim3(256), 0, stream,
                      input, count, error);
    hip_ok(hipGetLastError());
}

struct Matrix {
    TensorType type = TensorType::Q8_0;
    int input = 0, output = 0;
    Buffer data;
    void apply(const Context& c, const float* x, int n, float* y, Q8_1* q8, int* error) const {
        if (type == TensorType::F32)
            blas_ok(launch_dense_linear(c.handle, {data.get(), input, output}, x, n, y, c.stream));
        else {
            hip_ok(launch_quantize_q8_1(x, q8, input, n, error, c.stream));
            hip_ok(launch_quantized_linear({type, data.get<void>(), input, output}, q8, n, y, c.stream));
        }
    }
};
struct Mixer { int norm, down, up, inject = -1; };
struct ProbeFrame {
    std::vector<float> gates, head;
    std::array<std::int32_t, B * K> ids{};
    std::array<float, B * K> weights{};
};
struct FirstRowFrame {
    std::vector<float> embedding, hnorm, enorm, concat, fused;
    std::vector<float> attention_mixed, pregate, attention_output, attention_combined;
    std::vector<float> ffn_mixed, ffn_output, D, head;
    void allocate() {
        embedding.resize(H); hnorm.resize(W); enorm.resize(W); concat.resize(2*W); fused.resize(W);
        attention_mixed.resize(H); pregate.resize(Q); attention_output.resize(H); attention_combined.resize(W);
        ffn_mixed.resize(H); ffn_output.resize(H); D.resize(W); head.resize(H);
    }
    std::uint64_t capacity_bytes() const {
        std::uint64_t count=0;
        for (const auto* v:{&embedding,&hnorm,&enorm,&concat,&fused,&attention_mixed,&pregate,
             &attention_output,&attention_combined,&ffn_mixed,&ffn_output,&D,&head}) count+=v->capacity();
        return count*sizeof(float);
    }
};

// COPY unchanged qualified Session stable selection/width32 renormalization.
void top10(const float* probabilities, int n, std::array<std::int32_t, B*K>& ids,
           std::array<float, B*K>& weights) {
    for (int t = 0; t < n; ++t) {
        const float* row = probabilities + t * E;
        std::array<int, E> order{}; std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](int a, int b) {
            return row[a] != row[b] ? row[a] > row[b] : a < b;
        });
        std::array<float, 32> buckets{};
        for (int r = 0; r < K; ++r) buckets[order[r] % 32] += row[order[r]];
        for (int mask = 1; mask < 32; mask *= 2) {
            const auto previous = buckets;
            for (int lane = 0; lane < 32; ++lane) buckets[lane] = previous[lane] + previous[lane ^ mask];
        }
        const float inverse = 1.0f / std::max(buckets[0], 1.0f / 16384.0f);
        for (int r = 0; r < K; ++r) {
            ids[t*K+r] = order[r]; weights[t*K+r] = row[order[r]] * inverse;
            if (!std::isfinite(weights[t*K+r])) throw std::runtime_error("MTP nonfinite routing");
        }
    }
}
void validate_payload(const TensorView& t, std::span<const std::byte> payload) {
    if (t.type == TensorType::F32) {
        for (std::size_t i = 0; i < payload.size(); i += 4) {
            float x; std::memcpy(&x, payload.data() + i, 4);
            if (!std::isfinite(x)) throw std::runtime_error("MTP nonfinite F32 weight: " + t.name);
        }
    } else if (t.type == TensorType::Q8_0) {
        // Q8_0 canonical block32 is HALF scale +32 signed codes, NOT Q8_1.
        for (std::size_t i = 0; i < payload.size(); i += 34) {
            std::uint16_t d; std::memcpy(&d, payload.data() + i, 2);
            if (!std::isfinite(half_to_float(d))) throw std::runtime_error("MTP nonfinite Q8 scale: " + t.name);
        }
    } else throw std::runtime_error("MTP executed role must be actual F32/Q8_0");
}
} // namespace

struct MtpSession::Impl {
    // Destruction order: all Buffers -> context drain/destroy -> ledger. Impl
    // destructor also drains BEFORE the first Buffer can be released.
    Session& target;
    MtpSessionConfig config;
    SessionSharedWeights shared;
    MtpModel model;
    Ledger ledger;
    Context context;
    std::vector<Matrix> matrices;
    std::array<std::vector<std::byte>, 3> ram_experts;
    std::array<Buffer, 3> resident_experts;
    std::array<std::size_t, 3> expert_stride{};
    std::array<Buffer, scratch_count> scratch;
    Buffer error, q8, original_q8, gathered_q8, contributions, routes, route_weights;
    Buffer keys, values, partial, max_sum, dense_staged, head_output, frequencies;
    std::array<Buffer, 2> taps, carry;
    std::array<std::vector<float>, 2> host_logits;
    std::array<ProbeFrame, 2> probes;
    std::array<FirstRowFrame, 2> first_rows;
    std::array<float, B * H> host_embedding{};
    std::array<float, B * E> probabilities{};
    std::array<std::int32_t, B * K> ids{};
    std::array<float, B * K> weights{};
    std::array<MoeRouteIndex, B * K> route_dto{};
    RouteGroups groups{B};
    MtpSessionStats statistics;
    MtpSessionMemory accounting;
    std::uint64_t cursor = 0, tap_position = 0;
    int tap_rows = 0, logit_rows = 0, probe_rows = 0;
    bool usable = true, carry_valid = false, fail_after_kv = false;
    bool first_row_active = false, first_row_published = false;
    int first_row_source_row = -1;
    std::uint64_t first_row_position = 0;
    int enorm = -1, hnorm = -1, eh = -1, query = -1, key = -1, value = -1;
    int attention_output = -1, qnorm = -1, knorm = -1;
    int router = -1, shared_router = -1, shared_gate = -1, shared_up = -1, shared_down = -1;
    Mixer attention_hc{}, ffn_hc{}, own_head{};

    explicit Impl(Session& t, const std::string& path, MtpSessionConfig c)
        : target(t), config(checked_config(c)), shared(t.borrow_mtp_weights()),
          model(path, *shared.descriptor) {
        DeviceGuard device;
        try {
        const auto& g = model.geometry();
        const auto& borrowed = model.shared_target();
        if (!shared.descriptor || shared.output.device != 1 || !shared.output.pointer ||
            shared.output.input != H || shared.output.output != V ||
            shared.output.bytes != borrowed.output.byte_size || shared.embedding.width != H ||
            shared.embedding.rows != V || shared.embedding.cpu_bytes.size() != borrowed.embedding.byte_size ||
            shared.embedding.row_bytes != borrowed.embedding.strides[1] ||
            shared.embedding.row_bytes != std::uint64_t(H/32*18) ||
            g.hidden_size != H || g.widened_size != W || g.hc_count != 4 || g.hc_rank != R ||
            g.query_heads != 24 || g.kv_heads != 2 || g.head_dim != 256 || g.experts != E ||
            g.intermediate_size != I || g.vocabulary_size != V || g.predict_layers != 1 ||
            g.layer != 48 || g.rotary_dim != 64 || g.rope_base != 1e7f ||
            g.rope_sections != std::array<std::int32_t,4>{11,11,10,0})
            throw std::runtime_error("MTP shared payload/geometry guard");

        // Build the exact executed role list BEFORE payload reads/allocations.
        // BF16 indexer Q/K and F32 indexer norms were validated by MtpModel but
        // do not execute at compression0 and are not payload-read here.
        const auto& w = model.weights();
        struct Role { const TensorView* tensor; int input, output; };
        std::vector<Role> roles; roles.reserve(25);
        const auto role = [&](const TensorView& tv, int in, int out) {
            const auto layout = type_layout(tv.type);
            const auto expected = bytes(bytes(in / layout.block_elements, out), layout.block_bytes);
            if ((tv.type != TensorType::F32 && tv.type != TensorType::Q8_0) ||
                in % layout.block_elements || tv.elements != std::uint64_t(in) * out ||
                tv.byte_size != expected || tv.dimensions[0] != std::uint64_t(in))
                throw std::runtime_error("MTP role layout: " + tv.name);
            roles.push_back({&tv, in, out}); return int(roles.size()) - 1;
        };
        const auto mixer = [&](const MtpHeadWeights& m, const TensorView* inject) {
            return Mixer{role(m.norm,W,1),role(m.down,W,R),role(m.up,R,W), inject ? role(*inject,W,4) : -1};
        };
        enorm=role(w.embedding_norm,H,1); hnorm=role(w.hidden_norm,W,1);
        eh=role(w.embedding_hidden_projection,2*H,H);
        attention_hc=mixer(w.attention_hc.mixer,&w.attention_hc.inject);
        ffn_hc=mixer(w.ffn_hc.mixer,&w.ffn_hc.inject); own_head=mixer(w.head,nullptr);
        query=role(w.attention.query,H,2*Q); key=role(w.attention.key,H,512);
        value=role(w.attention.value,H,512); attention_output=role(w.attention.output,Q,H);
        qnorm=role(w.attention.query_norm,256,1); knorm=role(w.attention.key_norm,256,1);
        router=role(w.experts.router,H,E); shared_router=role(w.experts.shared_router,H,1);
        shared_gate=role(w.experts.shared_gate,H,I); shared_up=role(w.experts.shared_up,H,I);
        shared_down=role(w.experts.shared_down,I,H);
        if (roles.size() != 25) throw std::logic_error("MTP executed role inventory");
        const std::array<const TensorView*,3> experts{&w.experts.gate,&w.experts.up,&w.experts.down};
        for (int i=0;i<3;++i) {
            const auto& tv=*experts[i]; const int in=i==2?I:H, out=i==2?H:I;
            const auto layout=type_layout(tv.type);
            expert_stride[i]=bytes(bytes(in/layout.block_elements,out),layout.block_bytes);
            if (tv.type!=TensorType::Q8_0 || tv.rank!=3 || tv.dimensions[0]!=std::uint64_t(in) ||
                tv.dimensions[1]!=std::uint64_t(out) || tv.dimensions[2]!=E ||
                tv.strides[2]!=expert_stride[i] || tv.byte_size!=bytes(E,expert_stride[i]))
                throw std::runtime_error("MTP canonical Q8 expert stride");
            accounting.gpu_experts+=tv.byte_size;
        }
        if (accounting.gpu_experts!=2673868800ULL) throw std::logic_error("MTP actual expert payload");
        for (const auto& r:roles) accounting.gpu_nonexperts+=r.tensor->byte_size;
        const auto l=mtp_attention_q4_layout(config.capacity);
        accounting.gpu_kv=2*bytes(l.cache_elements,sizeof(Q4_0));
        accounting.gpu_dense_scratch=B*l.workspace_bytes_per_query;
        accounting.gpu_workspace=scratch_count*bytes(scratch_elements,sizeof(float))+
            sizeof(int)+bytes(B*(W/32),sizeof(Q8_1))+2*bytes(B*(H/32),sizeof(Q8_1))+
            bytes(B*K*H,sizeof(float))+bytes(B*K,sizeof(MoeRouteIndex))+bytes(B*K,sizeof(float))+
            bytes(B*V,sizeof(float))+bytes(32,sizeof(float))+2*bytes(B*W,sizeof(float))+2*bytes(W,sizeof(float));
        const auto planned=accounting.gpu_experts+accounting.gpu_nonexperts+accounting.gpu_kv+
            accounting.gpu_dense_scratch+accounting.gpu_workspace;
        std::size_t free=0,total=0; hip_ok(hipMemGetInfo(&free,&total));
        if (planned>free) throw std::runtime_error("MTP all512 requires "+std::to_string(planned)+
            " device1 bytes; free="+std::to_string(free));

        // Canonical RAM arrays retained once; immutable GPU arrays loaded once.
        for (int i=0;i<3;++i) {
            ram_experts[i].resize(experts[i]->byte_size);
            read(*experts[i],ram_experts[i],true); validate_payload(*experts[i],ram_experts[i]);
            resident_experts[i]=Buffer(ledger,ram_experts[i].size());
            hip_ok(hipMemcpyAsync(resident_experts[i].get<void>(),ram_experts[i].data(),
                ram_experts[i].size(),hipMemcpyHostToDevice,context.stream));
            accounting.ram_expert_payload+=ram_experts[i].size();
            accounting.ram_expert_capacity+=ram_experts[i].capacity();
        }
        matrices.reserve(roles.size());
        for (const auto& r:roles) {
            std::vector<std::byte> payload(r.tensor->byte_size);
            read(*r.tensor,payload,false); validate_payload(*r.tensor,payload);
            Matrix m{r.tensor->type,r.input,r.output,Buffer(ledger,payload.size())};
            try {
                hip_ok(hipMemcpyAsync(m.data.get<void>(),payload.data(),payload.size(),hipMemcpyHostToDevice,context.stream));
                // Temporary payload and Matrix remain alive in THIS handler even
                // if enqueue/completion fails; drain before either local unwinds.
                context.drain();
            } catch (...) { context.drain_cleanup(); throw; }
            matrices.push_back(std::move(m));
        }
        for (auto& f:scratch) f=Buffer(ledger,bytes(scratch_elements,sizeof(float)));
        const auto alloc=[&](Buffer& b,std::size_t n,std::size_t size){b=Buffer(ledger,bytes(n,size));};
        alloc(error,1,sizeof(int)); alloc(q8,B*(W/32),sizeof(Q8_1));
        alloc(original_q8,B*(H/32),sizeof(Q8_1)); alloc(gathered_q8,B*(H/32),sizeof(Q8_1));
        alloc(contributions,B*K*H,sizeof(float)); alloc(routes,B*K,sizeof(MoeRouteIndex));
        alloc(route_weights,B*K,sizeof(float)); alloc(keys,l.cache_elements,sizeof(Q4_0));
        alloc(values,l.cache_elements,sizeof(Q4_0)); alloc(partial,B*l.partial_elements,sizeof(float));
        alloc(max_sum,B*l.max_sum_elements,sizeof(float2)); alloc(dense_staged,B*Q,sizeof(float));
        alloc(head_output,B*V,sizeof(float)); alloc(frequencies,32,sizeof(float));
        for (auto& b:taps) alloc(b,B*W,sizeof(float));
        for (auto& b:carry) alloc(b,W,sizeof(float));
        for (auto& v:host_logits) v.resize(B*V);
        if (config.diagnostic_probe) for (auto& p:probes) {p.gates.resize(B*Q);p.head.resize(B*H);}
        if (config.diagnostic_first_row) for (auto& p:first_rows) p.allocate();
        hip_ok(hipMemsetAsync(error.get<int>(),0,sizeof(int),context.stream));
        hip_ok(launch_rope_frequencies(g.rope_base,frequencies.get(),error.get<int>(),context.stream));
        numeric_ok(); accounting.immutable_ready_slots=E;
        accounting.borrowed_head_bytes=shared.output.bytes;
        accounting.borrowed_embedding_bytes=shared.embedding.cpu_bytes.size();
        if (accounting.model_payload_reads!=28 || accounting.expert_payload_reads!=3 || ledger.live!=planned)
            throw std::logic_error("MTP constructor payload/owned ledger mismatch");
        } catch (...) {
            // Constructor failures drain while ALL source/destination members
            // still exist, before C++ starts unwinding their Buffer/vector owners.
            context.drain_cleanup(); throw;
        }
    }
    ~Impl() { DeviceGuard device; context.drain_cleanup(); }
    float* f(int i) const { return scratch[i].get(); }
    int* flag() const { return error.get<int>(); }
    void read(const TensorView& t,std::span<std::byte> destination,bool expert) {
        model.sidecar().read_tensor(t.name,destination);
        ++accounting.model_payload_reads; accounting.model_payload_bytes+=destination.size();
        if (expert) {++accounting.expert_payload_reads;accounting.expert_payload_bytes+=destination.size();}
    }
    void numeric_ok() const {
        int flags=0;
        try {
            hip_ok(hipMemcpyAsync(&flags,flag(),sizeof(int),hipMemcpyDeviceToHost,context.stream));
            context.drain();
        } catch (...) {
            // Drain while the local DMA destination is still alive, rather than
            // relying on the caller's handler after this stack frame unwinds.
            context.drain_cleanup(); throw;
        }
        if (flags) throw std::runtime_error("MTP numeric failure flags="+std::to_string(flags)+"; reset required");
    }
    void apply(int matrix,const float* x,int n,float* y) {
        matrices.at(matrix).apply(context,x,n,y,q8.get<Q8_1>(),flag());
    }
    void first_copy(std::span<float> destination,const float* source,std::size_t input_row_stride=0) {
        if (!first_row_active) return;
        const auto stride=input_row_stride ? input_row_stride : destination.size();
        if (destination.empty() || destination.size()>2*W || stride<destination.size() ||
            stride>2*W || first_row_source_row<0 || first_row_source_row>=B)
            throw std::logic_error("MTP first-row diagnostic extent");
        // Destinations are constructor-owned staging members; every exit drains
        // before owners unwind. Stream ordering protects scratch before reuse.
        hip_ok(hipMemcpyAsync(destination.data(),source+first_row_source_row*stride,destination.size_bytes(),
            hipMemcpyDeviceToHost,context.stream));
    }
    void mix(const Mixer& hc,const float* input,int n) {
        hip_ok(launch_group_rms_norm(input,matrices[hc.norm].data.get(),n,W,H,
            model.geometry().rms_epsilon,false,f(7),flag(),context.stream));
        apply(hc.down,f(7),n,f(8));
        hip_ok(launch_hc_lowrank_activation(f(8),n,f(9),flag(),context.stream));
        apply(hc.up,f(9),n,f(10));
        hip_ok(launch_hc_mix(f(7),f(10),n,f(11),flag(),context.stream));
        if (hc.inject>=0) {
            apply(hc.inject,f(7),n,f(12));
            hip_ok(launch_hc_injection(f(12),n,f(13),flag(),context.stream));
        }
    }
    void prepare_kv(std::span<const MtpHiddenRow> hidden,std::span<const std::int32_t> tokens) {
        const int n=int(tokens.size()); const auto stream=context.stream;
        hip_ok(hipMemsetAsync(flag(),0,sizeof(int),stream));
        for (int t=0;t<n;++t) {
            hipLaunchKernelGGL(pack_hidden,dim3(40),dim3(256),0,stream,
                hidden[t].pointer,f(0)+t*W,cursor+t==0,flag()); hip_ok(hipGetLastError());
            const auto row=shared.embedding.cpu_bytes.subspan(tokens[t]*shared.embedding.row_bytes,
                                                             shared.embedding.row_bytes);
            for (int j=0;j<H;++j) {
                const float x=tensor_element(SessionEmbedding::type,row,j);
                if (!std::isfinite(x)) throw std::runtime_error("MTP nonfinite borrowed embedding");
                host_embedding[t*H+j]=x;
            }
        }
        hip_ok(hipMemcpyAsync(f(1),host_embedding.data(),bytes(n*H,sizeof(float)),hipMemcpyHostToDevice,stream));
        hip_ok(launch_group_rms_norm(f(1),matrices[enorm].data.get(),n,H,H,model.geometry().rms_epsilon,
                                    false,f(2),flag(),stream));
        // Pin: mx dcd685463d597d31f5ca759d32c94592a2740fa4,
        // src/models/qwen4exp.cpp:438-450. ggml [H,4,N] reduces axis0=H:
        // hidden[N][4][H] -> RMS per (token,branch) -> gamma[4][H]
        //                         -> normalized_hidden[N][4][H].
        // width=W, group=H gives 4*N independent columns, NOT RMS over W.
        // shared_gamma=false: gamma_base=((token*4+branch)*H)%W=branch*H;
        // gamma[branch*H+j] is reused across tokens, never across branches.
        // Reuse the qualified producer; HC mixers keep their existing path.
        hip_ok(launch_group_rms_norm(f(0),matrices[hnorm].data.get(),n,W,H,model.geometry().rms_epsilon,
                                    false,f(3),flag(),stream));
        first_copy(first_rows[1].embedding,f(1)); first_copy(first_rows[1].hnorm,f(3));
        for (int branch=0;branch<4;++branch) {
            hipLaunchKernelGGL(branch_concat,dim3((n*2*H+255)/256),dim3(256),0,stream,f(2),f(3),n,branch,f(4));
            hip_ok(hipGetLastError());
            if (first_row_active) {
                first_copy(std::span(first_rows[1].enorm).subspan(branch*H,H),f(4),2*H);
                first_copy(std::span(first_rows[1].concat).subspan(branch*2*H,2*H),f(4));
            }
            apply(eh,f(4),n,f(5));
            hipLaunchKernelGGL(branch_store,dim3((n*H+255)/256),dim3(256),0,stream,f(5),n,branch,f(6));
            hip_ok(hipGetLastError());
        }
        first_copy(first_rows[1].fused,f(6));
        mix(attention_hc,f(6),n);
        first_copy(first_rows[1].attention_mixed,f(11));
        // Hc inject consumes the normalized widened mixer input (f7), while
        // residual combination consumes ORIGINAL widened residual (f6/f30).
        apply(query,f(11),n,f(14)); apply(key,f(11),n,f(20)); apply(value,f(11),n,f(24));
        hip_ok(launch_qsa_split_q_gate(f(14),n,f(15),f(19),flag(),stream));
        hip_ok(launch_group_rms_norm(f(15),matrices[qnorm].data.get(),n,Q,256,
            model.geometry().rms_epsilon,true,f(16),flag(),stream));
        hip_ok(launch_group_rms_norm(f(20),matrices[knorm].data.get(),n,512,256,
            model.geometry().rms_epsilon,true,f(21),flag(),stream));
        hip_ok(launch_qsa_rope(f(16),frequencies.get(),n,24,int(cursor),f(17),flag(),stream));
        hip_ok(launch_qsa_rope(f(21),frequencies.get(),n,2,int(cursor),f(22),flag(),stream));
        // Pin check: mx dcd6854 llama-graph.cpp build_attn lines2983..3020
        // rotates Q/K and V before canonical cache store and inverts V after
        // attention, before qwen4exp's raw sigmoid gate. KEEP target Q4's H256/H64.
        launch_hadamard(f(17),f(18),n*Q,256,stream); hip_ok(hipGetLastError());
        launch_hadamard(f(22),f(23),n*512,256,stream); hip_ok(hipGetLastError());
        launch_hadamard(f(24),f(25),n*512,64,stream); hip_ok(hipGetLastError());
        // Canonical Q4_0 producer/types/scales unchanged; discarded slots are
        // overwritten in stream order BEFORE dense visibility is extended.
        launch_quantize_q4(f(23),keys.get<Q4_0>()+cursor*16,n*512,flag(),stream); hip_ok(hipGetLastError());
        launch_quantize_q4(f(25),values.get<Q4_0>()+cursor*16,n*512,flag(),stream); hip_ok(hipGetLastError());
    }
    void attention(int n) {
        const auto l=mtp_attention_q4_layout(config.capacity); const auto stream=context.stream;
        hip_ok(launch_mtp_attention_q4({f(18),scratch_elements,Q},
            {keys.get<Q4_0>(),l.cache_elements,values.get<Q4_0>(),l.cache_elements,config.capacity,int(cursor)+n},
            n,int(cursor)+1,{f(26),scratch_elements,Q},
            {{partial.get(),B*l.partial_elements,l.partial_elements},
             {max_sum.get<float2>(),B*l.max_sum_elements,l.max_sum_elements},
             {dense_staged.get(),B*Q,Q},flag()},stream));
        launch_hadamard(f(26),f(27),n*Q,64,stream); hip_ok(hipGetLastError());
        first_copy(first_rows[1].pregate,f(27));
        hip_ok(launch_sigmoid_gate(f(27),f(19),n*Q,f(28),flag(),stream));
        apply(attention_output,f(28),n,f(29));
        first_copy(first_rows[1].attention_output,f(29));
        hip_ok(launch_hc_combine(f(6),f(29),f(13),n,f(30),flag(),stream));
        first_copy(first_rows[1].attention_combined,f(30));
    }
    std::uint64_t ffn(int n) {
        const auto stream=context.stream; mix(ffn_hc,f(30),n);
        first_copy(first_rows[1].ffn_mixed,f(11));
        apply(router,f(11),n,f(31));
        hip_ok(launch_router_softmax(f(31),n,f(32),flag(),stream));
        hip_ok(hipMemcpyAsync(probabilities.data(),f(32),bytes(n*E,sizeof(float)),hipMemcpyDeviceToHost,stream));
        numeric_ok(); top10(probabilities.data(),n,ids,weights);
        groups.prepare(std::span(ids).first(n*K),std::span(weights).first(n*K),n);
        int count=0; for (const auto& a:groups.assignments()) route_dto[count++]={a.token,a.rank};
        hip_ok(hipMemcpyAsync(routes.get<void>(),route_dto.data(),bytes(n*K,sizeof(MoeRouteIndex)),hipMemcpyHostToDevice,stream));
        hip_ok(hipMemcpyAsync(route_weights.get<void>(),weights.data(),bytes(n*K,sizeof(float)),hipMemcpyHostToDevice,stream));
        hip_ok(launch_quantize_q8_1(f(11),original_q8.get<Q8_1>(),H,n,flag(),stream));
        int offset=0; std::uint64_t group_count=0;
        for (int expert=0;expert<E;++expert) {
            const int cols=int(groups.group(expert).size()); if (!cols) continue;
            if (cols>B || accounting.immutable_ready_slots!=E) throw std::logic_error("MTP immutable slot guard");
            hip_ok(launch_moe_gather_q8({original_q8.get<Q8_1>(),B*(H/32)},n,
                {routes.get<MoeRouteIndex>(),B*K},n*K,offset,cols,H/32,
                {gathered_q8.get<Q8_1>(),B*(H/32)},B,flag(),stream));
            const auto weights_for=[&](int role) {
                return resident_experts[role].get<std::byte>()+expert*expert_stride[role];
            };
            // One immutable triplet per group; original GPU Q8 input, GPU SiLU,
            // GPU Q8 middle, Q8_0 down. NO CPU Q4 maddubs or host-libm expert.
            hip_ok(launch_quantized_linear({TensorType::Q8_0,weights_for(0),H,I},gathered_q8.get<Q8_1>(),cols,f(33),stream));
            hip_ok(launch_quantized_linear({TensorType::Q8_0,weights_for(1),H,I},gathered_q8.get<Q8_1>(),cols,f(34),stream));
            hip_ok(launch_silu_up(f(33),f(34),cols*I,f(35),flag(),stream));
            hip_ok(launch_quantize_q8_1(f(35),q8.get<Q8_1>(),I,cols,flag(),stream));
            hip_ok(launch_quantized_linear({TensorType::Q8_0,weights_for(2),I,H},q8.get<Q8_1>(),cols,f(36),stream));
            hip_ok(launch_moe_scatter({f(36),scratch_elements},n,{routes.get<MoeRouteIndex>(),B*K},n*K,
                offset,cols,H,K,{contributions.get(),B*K*H},B,flag(),stream));
            offset+=cols; ++group_count;
        }
        if (offset!=n*K) throw std::logic_error("MTP routed assignment coverage");
        hip_ok(launch_routed_fold(contributions.get(),route_weights.get(),n,f(37),flag(),stream));
        apply(shared_gate,f(11),n,f(33)); apply(shared_up,f(11),n,f(34));
        hip_ok(launch_silu_up(f(33),f(34),n*I,f(35),flag(),stream));
        apply(shared_down,f(35),n,f(36)); apply(shared_router,f(11),n,f(38));
        hip_ok(launch_shared_sigmoid_add_rows(f(36),f(38),n,H,f(37),flag(),stream));
        first_copy(first_rows[1].ffn_output,f(37));
        hip_ok(launch_hc_combine(f(30),f(37),f(13),n,taps[1].get(),flag(),stream));
        return group_count;
    }
    std::span<const float> forward(std::span<const MtpHiddenRow> hidden,
        std::span<const std::int32_t> tokens,bool teacher,bool skip) {
        if (!usable) throw std::runtime_error("MTP invalidated; reset required");
        if (tokens.empty() || tokens.size()>B || hidden.size()!=tokens.size() ||
            tokens.size()>std::uint64_t(config.capacity)-cursor)
            throw std::invalid_argument("MTP rows/hidden/capacity");
        for (std::size_t t=0;t<tokens.size();++t)
            if (tokens[t]<0 || tokens[t]>=V || !valid_row(hidden[t],cursor+t==0))
                throw std::invalid_argument("MTP token/device/hidden range");
        const int n=int(tokens.size()); const auto begin=std::chrono::steady_clock::now();
        usable=false; DeviceGuard device;
        try {
            first_row_source_row=config.diagnostic_first_row && !skip ?
                diagnostic_probe_row(cursor,n,config.diagnostic_probe_position) : -1;
            first_row_active=first_row_source_row>=0;
            prepare_kv(hidden,tokens);
            if (config.diagnostic_probe && fail_after_kv) {
                context.drain(); throw std::runtime_error("MTP fixture failure after KV; reset required");
            }
            std::uint64_t group_count=0;
            if (!skip) {
                attention(n); group_count=ffn(n); // D is BEFORE the own nextn HC head
                first_copy(first_rows[1].D,taps[1].get());
                mix(own_head,taps[1].get(),n);
                first_copy(first_rows[1].head,f(11));
                hip_ok(launch_quantize_q8_1(f(11),q8.get<Q8_1>(),H,n,flag(),context.stream));
                hip_ok(launch_quantized_linear({SessionQ6Head::type,shared.output.pointer,H,V},
                    q8.get<Q8_1>(),n,head_output.get(),context.stream));
                check_finite(taps[1].get(),n*W,flag(),context.stream);
                check_finite(head_output.get(),n*V,flag(),context.stream);
                hip_ok(hipMemcpyAsync(host_logits[1].data(),head_output.get(),bytes(n*V,sizeof(float)),
                    hipMemcpyDeviceToHost,context.stream));
                if (config.diagnostic_probe) {
                    hip_ok(hipMemcpyAsync(probes[1].gates.data(),f(19),bytes(n*Q,sizeof(float)),hipMemcpyDeviceToHost,context.stream));
                    hip_ok(hipMemcpyAsync(probes[1].head.data(),f(11),bytes(n*H,sizeof(float)),hipMemcpyDeviceToHost,context.stream));
                    std::copy_n(ids.begin(),n*K,probes[1].ids.begin());
                    std::copy_n(weights.begin(),n*K,probes[1].weights.begin());
                }
            }
            numeric_ok(); // completion before ANY cursor/publication/counter change
            if (!skip) {
                for (int i=0;i<n*V;++i) if (!std::isfinite(host_logits[1][i]))
                    throw std::runtime_error("MTP full-logit finiteness guard");
                std::swap(taps[0],taps[1]); std::swap(host_logits[0],host_logits[1]);
                if (config.diagnostic_probe) {std::swap(probes[0],probes[1]);probe_rows=n;}
                if (first_row_active) {
                    std::swap(first_rows[0],first_rows[1]);first_row_published=true;
                    first_row_position=cursor+std::uint64_t(first_row_source_row);
                }
                tap_position=cursor; tap_rows=logit_rows=n;
                statistics.full_rows+=n; ++statistics.full_calls; statistics.head_rows+=n;
                statistics.routed_groups+=group_count; statistics.routed_assignments+=n*K;
                statistics.grouped_reuses+=n*K-group_count;
            } else statistics.kv_only_rows+=n;
            if (teacher) ++statistics.teacher_calls;
            cursor+=n; statistics.consumed_rows=cursor;
            statistics.last_completed_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-begin).count();
            usable=true;
            first_row_active=false;first_row_source_row=-1;
            return skip ? std::span<const float>{} : std::span<const float>(host_logits[0]).first(n*V);
        } catch (...) { context.drain_cleanup(); first_row_active=false;first_row_source_row=-1;throw; }
    }
    MtpSessionMemory memory() const {
        DeviceGuard device; context.drain(); auto result=accounting;
        std::uint64_t sum=0,count=0;
        const auto visit=[&](const Buffer& b){sum+=b.size();count+=b.size()!=0;};
        for (const auto& m:matrices) visit(m.data);
        for (const auto& b:resident_experts) visit(b);
        for (const auto& b:scratch) visit(b);
        for (const auto& b:taps) visit(b);
        for (const auto& b:carry) visit(b);
        for (const auto* b:{&error,&q8,&original_q8,&gathered_q8,&contributions,&routes,&route_weights,
            &keys,&values,&partial,&max_sum,&dense_staged,&head_output,&frequencies}) visit(*b);
        result.owned_bytes=sum; result.owned_buffers=count; result.peak_owned_bytes=ledger.peak;
        result.ownership_verified=sum==ledger.live && count==ledger.buffers &&
            sum==result.gpu_experts+result.gpu_nonexperts+result.gpu_kv+result.gpu_dense_scratch+result.gpu_workspace;
        for (const auto& l:host_logits) result.host_logits+=l.capacity()*sizeof(float);
        for (const auto& p:probes) result.host_probe+=(p.gates.capacity()+p.head.capacity())*sizeof(float)+
            sizeof(p.ids)+sizeof(p.weights); // fixed route frames exist even when capture is off
        for (const auto& p:first_rows) result.host_probe+=p.capacity_bytes();
        // Explicit fixed owner payloads, plus RouteGroups requested assignment
        // storage. Model/parser/vector allocator overhead is not RSS accounting.
        result.host_staging=sizeof(host_embedding)+sizeof(probabilities)+sizeof(ids)+sizeof(weights)+
            sizeof(route_dto)+sizeof(groups)+B*K*sizeof(RouteAssignment);
        std::size_t free=0,total=0; hip_ok(hipMemGetInfo(&free,&total));
        result.free_vram=free;result.total_vram=total;return result;
    }
};

MtpHiddenRow MtpTap::row(int index) const {
    if (device!=1 || width!=W || index<0 || index>=rows || !pointer ||
        rows<1 || rows>B || !valid_row({device,pointer,std::size_t(W)},false) ||
        reinterpret_cast<std::uintptr_t>(pointer)>std::numeric_limits<std::uintptr_t>::max()-bytes(rows*W,sizeof(float)))
        throw std::invalid_argument("MTP tap row");
    return {device,pointer+index*W,W};
}
MtpSession::MtpSession(Session& target,const std::string& path,MtpSessionConfig c)
    : impl_(std::make_unique<Impl>(target,path,c)) {}
MtpSession::~MtpSession()=default;
std::span<const float> MtpSession::step(std::span<const MtpHiddenRow> h,std::span<const std::int32_t> t) {
    return impl_->forward(h,t,false,false);
}
std::span<const float> MtpSession::teacher_append(std::span<const MtpHiddenRow> h,
    std::span<const std::int32_t> t,MtpTeacherHead head) {
    if (head!=MtpTeacherHead::skip && head!=MtpTeacherHead::publish)
        throw std::invalid_argument("MTP explicit teacher head mode");
    return impl_->forward(h,t,true,head==MtpTeacherHead::skip);
}
void MtpSession::restore_prefix(std::uint64_t prefix) {
    auto& p=*impl_;
    if (!p.usable) throw std::runtime_error("MTP invalidated; reset required");
    if (prefix>p.cursor) throw std::invalid_argument("MTP restore can only hide current suffix");
    p.usable=false; DeviceGuard device;
    try {p.context.drain();p.cursor=prefix;p.statistics.consumed_rows=prefix;++p.statistics.restore_calls;p.usable=true;}
    catch (...) {p.context.drain_cleanup();throw;}
}
void MtpSession::reset() {
    auto& p=*impl_; p.usable=false; DeviceGuard device;
    try {
        p.context.drain();hip_ok(hipMemsetAsync(p.flag(),0,sizeof(int),p.context.stream));p.context.drain();
        p.cursor=0;p.tap_rows=p.logit_rows=p.probe_rows=0;p.carry_valid=false;
        p.first_row_active=p.first_row_published=false;
        p.first_row_source_row=-1;p.first_row_position=0;
        p.tap_position=0;p.statistics={};p.fail_after_kv=false;p.usable=true;
    } catch (...) {p.context.drain_cleanup();throw;}
}
void MtpSession::save_target_carry(int row) {
    auto& p=*impl_;if (!p.usable) throw std::runtime_error("MTP invalidated; reset required");
    const auto tap=p.target.target_tap();
    // Session can now publish an opt-in full-window tap up to16384 rows.
    // The carry remains ONE checked row; no larger copy or new allocation.
    if (tap.device!=1 || tap.width!=W || row<0 || row>=tap.rows || tap.rows>16384 ||
        !valid_row({tap.device,tap.pointer,W},false) ||
        reinterpret_cast<std::uintptr_t>(tap.pointer)>std::numeric_limits<std::uintptr_t>::max()-bytes(tap.rows*W,sizeof(float)))
        throw std::invalid_argument("MTP target carry view/row");
    p.usable=false;DeviceGuard device;
    try {
        hip_ok(hipMemsetAsync(p.flag(),0,sizeof(int),p.context.stream));
        hip_ok(hipMemcpyAsync(p.carry[1].get(),tap.pointer+row*W,bytes(W,sizeof(float)),
            hipMemcpyDeviceToDevice,p.context.stream));
        check_finite(p.carry[1].get(),W,p.flag(),p.context.stream);p.numeric_ok();
        std::swap(p.carry[0],p.carry[1]);p.carry_valid=true;p.usable=true;
    } catch (...) {p.context.drain_cleanup();throw;}
}
MtpHiddenRow MtpSession::saved_target_carry() const {
    if (!impl_->carry_valid) throw std::runtime_error("MTP has no saved target carry");
    return {1,impl_->carry[0].get(),W};
}
MtpTap MtpSession::own_tap() const {
    if (!impl_->tap_rows) return {};
    return {1,impl_->taps[0].get(),impl_->tap_rows,W,impl_->tap_position};
}
std::span<const float> MtpSession::logits() const {return std::span<const float>(impl_->host_logits[0]).first(impl_->logit_rows*V);}
std::uint64_t MtpSession::prefix() const {return impl_->cursor;}
bool MtpSession::requires_reset() const {return !impl_->usable;}
MtpSessionStats MtpSession::stats() const {return impl_->statistics;}
MtpSessionMemory MtpSession::memory() const {return impl_->memory();}
MtpSessionProbe MtpSession::probe() const {
    const auto& p=impl_->probes[0];const int n=impl_->probe_rows;
    return {std::span(p.gates).first(n*Q),std::span(p.head).first(n*H),
        std::span(p.ids).first(n*K),std::span(p.weights).first(n*K)};
}
MtpFirstRowProbe MtpSession::first_row_probe() const {
    if (!impl_->first_row_published) return {};
    const auto& p=impl_->first_rows[0];
    return {true,p.embedding,p.hnorm,p.enorm,p.concat,p.fused,p.attention_mixed,p.pregate,
        p.attention_output,p.attention_combined,p.ffn_mixed,p.ffn_output,p.D,p.head,impl_->first_row_position};
}
void MtpSession::diagnostic_fail_after_kv(bool enabled) {
    if (!impl_->config.diagnostic_probe) throw std::invalid_argument("MTP diagnostic resources disabled");
    if (!impl_->usable) throw std::runtime_error("MTP invalidated; reset required");
    impl_->fail_after_kv=enabled;
}
void MtpSession::diagnostic_poison_future() {
    auto& p=*impl_;
    if (!p.config.diagnostic_probe) throw std::invalid_argument("MTP diagnostic resources disabled");
    if (!p.usable) throw std::runtime_error("MTP invalidated; reset required");
    p.usable=false;DeviceGuard device;
    try {
        p.context.drain();const auto count=bytes((p.config.capacity-p.cursor)*16,sizeof(Q4_0));
        if (count) {
            hip_ok(hipMemsetAsync(p.keys.get<Q4_0>()+p.cursor*16,0x7f,count,p.context.stream));
            hip_ok(hipMemsetAsync(p.values.get<Q4_0>()+p.cursor*16,0x7f,count,p.context.stream));
        }
        p.context.drain();p.usable=true;
    } catch (...) {p.context.drain_cleanup();throw;}
}
} // namespace qwen
