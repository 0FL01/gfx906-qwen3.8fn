#include "session.hpp"
#include "dense.hpp"
#include "ple.hpp"
#include "hip/attention.cuh"
#include "hip/blocks.cuh"
#include "hip/dense.cuh"
#include "hip/gdn.cuh"
#include "hip/kv.cuh"
#include "hip/linear.cuh"
#include "hip/qsa_index.cuh"
#include "hip/qsa_select.cuh"
#include "hip/session_ops.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qwen {
namespace {
constexpr int H = 2560, W = 10240, V = 248320;
void hip(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
void blas(rocblas_status e) {
    if (e != rocblas_status_success) throw std::runtime_error("rocBLAS status " + std::to_string(e));
}
void cleanup(hipError_t e) noexcept {
    if (e != hipSuccess) { std::fprintf(stderr, "HIP cleanup: %s\n", hipGetErrorString(e)); std::abort(); }
}

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

struct Device {
    int id;
    hipStream_t stream = nullptr;
    rocblas_handle handle = nullptr;
    std::vector<Buffer> scratch;
    Buffer q8, error, logits, gathered_k, gathered_v, partial_o, partial_ms;
    Buffer histogram, select_state, candidates, ids, block_ids, counts;
    explicit Device(int d) : id(d) {
        hip(hipSetDevice(id));
        hipDeviceProp_t p{}; hip(hipGetDeviceProperties(&p, id));
        const std::string arch(p.gcnArchName);
        if (p.warpSize != 64 || !(arch == "gfx906" || arch.starts_with("gfx906:")))
            throw std::runtime_error("Session requires gfx906 wave64");
        hip(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        try {
            blas(rocblas_create_handle(&handle));
            scratch.reserve(24);
            // f(16) also holds 32768 completed-block scores at capacity 128K.
            for (int i = 0; i < 24; ++i) scratch.emplace_back(32768 * sizeof(float));
            q8 = Buffer(512 * sizeof(Q8_1)); error = Buffer(sizeof(int)); logits = Buffer(V * sizeof(float));
            gathered_k = Buffer(1050112 * sizeof(std::uint16_t));
            gathered_v = Buffer(1050112 * sizeof(std::uint16_t));
            partial_o = Buffer(202752 * sizeof(float)); partial_ms = Buffer(792 * sizeof(float2));
            histogram = Buffer(8192 * sizeof(int)); select_state = Buffer(16 * sizeof(std::uint64_t));
            candidates = Buffer(512 * sizeof(std::uint64_t)); ids = Buffer(2051 * sizeof(std::int32_t));
            block_ids = Buffer(512 * sizeof(std::int32_t)); counts = Buffer(2 * sizeof(int));
        } catch (...) {
            if (handle) { if (rocblas_destroy_handle(handle) != rocblas_status_success) std::abort(); }
            cleanup(hipStreamDestroy(stream)); stream = nullptr; throw;
        }
    }
    ~Device() {
        cleanup(hipSetDevice(id)); cleanup(hipStreamSynchronize(stream));
        if (rocblas_destroy_handle(handle) != rocblas_status_success) std::abort();
        cleanup(hipStreamDestroy(stream));
    }
    float* f(int i) { return scratch.at(static_cast<std::size_t>(i)).get(); }
    int* flag() { return error.get<int>(); }
    void clear() { hip(hipMemsetAsync(flag(), 0, sizeof(int), stream)); }
    void check(const std::string& where) {
        int e = 0; hip(hipMemcpyAsync(&e, flag(), sizeof(e), hipMemcpyDeviceToHost, stream));
        hip(hipStreamSynchronize(stream));
        if (e) throw std::runtime_error(where + ": numeric flag " + std::to_string(e));
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
    void apply(Device& d, const float* x, float* y) const {
        if (type == TensorType::F32 || type == TensorType::BF16 || type == TensorType::F16) {
            blas(launch_dense_linear(d.handle, {data.get(), input, output}, x, 1, y, d.stream));
        } else {
            hip(launch_quantize_q8_1(x, d.q8.get<Q8_1>(), input, 1, d.flag(), d.stream));
            hip(launch_quantized_linear({type, data.get<void>(), input, output}, d.q8.get<Q8_1>(), 1, y, d.stream));
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
    void mix(Device& d, const float* residual, float epsilon, bool head = false) const {
        hip(launch_group_rms_norm(residual, norm.get(), 1, W, H, epsilon, false, d.f(2), d.flag(), d.stream));
        down.apply(d, d.f(2), d.f(3));
        hip(launch_hc_lowrank_activation(d.f(3), 1, d.f(4), d.flag(), d.stream));
        up.apply(d, d.f(4), d.f(5));
        hip(launch_hc_mix(d.f(2), d.f(5), 1, d.f(6), d.flag(), d.stream));
        if (!head) {
            inject.apply(d, d.f(2), d.f(7));
            hip(launch_hc_injection(d.f(7), 1, d.f(8), d.flag(), d.stream));
        }
    }
};

struct Experts {
    struct Slot { int id = -1; std::uint64_t stamp = 0; };
    TensorType down_type;
    std::array<std::vector<std::byte>, 3> ram;
    std::array<std::size_t, 3> bytes{};
    Buffer slots;
    std::vector<Slot> map;
    std::uint64_t clock = 0;
    std::uint64_t payload_reads = 0, payload_bytes_read = 0;
    Experts(const Model& m, Device& d, const std::string& prefix, int nslots) : down_type(TensorType::Q4_0) {
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
            ram[i].resize(static_cast<std::size_t>(t.byte_size)); m.read_tensor(names[i], ram[i]);
            ++payload_reads; payload_bytes_read += ram[i].size();
        }
        hip(hipSetDevice(d.id)); slots = Buffer(total() * static_cast<std::size_t>(nslots)); map.resize(nslots);
    }
    std::size_t total() const { return bytes[0] + bytes[1] + bytes[2]; }
    void run(Device& d, int id, const Q8_1* input, float weight, SessionStats& stats) {
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
        } else ++stats.expert_hits;
        map[chosen].stamp = ++clock;
        // Upload, prior readers, reuse and new readers are on the SAME stream.
        // Eviction cannot overwrite a slot before any previous down has finished.
        auto* base = slots.get<std::byte>() + total() * chosen;
        hip(launch_quantized_linear({TensorType::Q4_0, base, H, 640}, input, 1, d.f(16), d.stream));
        hip(launch_quantized_linear({TensorType::Q4_0, base + bytes[0], H, 640}, input, 1, d.f(17), d.stream));
        hip(launch_silu_up(d.f(16), d.f(17), 640, d.f(18), d.flag(), d.stream));
        hip(launch_quantize_q8_1(d.f(18), d.q8.get<Q8_1>(), 640, 1, d.flag(), d.stream));
        hip(launch_quantized_linear({down_type, base + bytes[0] + bytes[1], 640, H}, d.q8.get<Q8_1>(), 1, d.f(19), d.stream));
        hip(launch_scaled_add(d.f(19), weight, H, d.f(20), d.flag(), d.stream));
    }
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
    bool state_probe = false;
    Layer(const Model& m, Device& d, int l, const SessionConfig& cfg, float rope_base)
        : index(l), device(d), attention_hc(m, d, "blk." + std::to_string(l) + ".hc_attn_"),
          ffn_hc(m, d, "blk." + std::to_string(l) + ".hc_ffn_"),
          router(matrix(m, d, "blk." + std::to_string(l) + ".ffn_gate_inp.weight", H, 512)),
          shared_gate(matrix(m, d, "blk." + std::to_string(l) + ".ffn_gate_shexp.weight", H, 640)),
          shared_up(matrix(m, d, "blk." + std::to_string(l) + ".ffn_up_shexp.weight", H, 640)),
          shared_down(matrix(m, d, "blk." + std::to_string(l) + ".ffn_down_shexp.weight", 640, H)),
          shared_weight(parameter(m, d, "blk." + std::to_string(l) + ".ffn_gate_inp_shexp.weight", {H})),
          experts(m, d, "blk." + std::to_string(l) + ".", cfg.expert_slots) {
        const std::string p = "blk." + std::to_string(l) + ".";
        state_probe = !cfg.trace_directory.empty() && l == 22;
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
    template<class Observe> void attention(int position, int capacity, float epsilon, Observe&& observe) {
        auto& d = device;
        qkv.apply(d, d.f(6), d.f(9));
        if ((index + 1) % 4) {
            z.apply(d, d.f(6), d.f(10)); alpha.apply(d, d.f(6), d.f(11)); beta.apply(d, d.f(6), d.f(12));
            observe("qkv", d.f(9), 10240); observe("z", d.f(10), 6144);
            observe("alpha", d.f(11), 48); observe("beta", d.f(12), 48);
            if (state_probe) {
                observe("gdn_state_pre", recurrent.get(), 786432);
                // Recomputed through the same GPU helper, NOT a capture of
                // recurrence registers. f(22) is diagnostic-only scratch.
                launch_gdn_control_witness({conv.get(), dt.get(), a.get(), gdn_norm.get(), epsilon},
                    {d.f(9), d.f(10), d.f(11), d.f(12)}, d.f(22), d.flag(), d.stream);
                observe("gdn_control_log_decay_recomputed", d.f(22), 48);
                observe("gdn_control_beta_recomputed", d.f(22) + 48, 48);
                observe("gdn_control_decay_recomputed", d.f(22) + 96, 48);
            }
            launch_gdn({conv.get(), dt.get(), a.get(), gdn_norm.get(), epsilon},
                       {d.f(9), d.f(10), d.f(11), d.f(12)}, 1, recurrent.get(), history.get(), d.f(13),
                       {d.f(14), stage_recurrent.get(), stage_history.get(), d.f(15), nullptr, nullptr, d.flag()}, d.stream);
            if (state_probe) observe("gdn_state_post", recurrent.get(), 786432);
            observe("gdn_convolved_normalized", d.f(14), 10240);
            observe("final_output", d.f(13), 6144);
        } else {
            key.apply(d, d.f(6), d.f(10)); value.apply(d, d.f(6), d.f(11));
            observe("qsa_q_gate_raw", d.f(9), 12288);
            observe("qsa_key_raw", d.f(10), 512); observe("qsa_value_raw", d.f(11), 512);
            hip(launch_qsa_split_q_gate(d.f(9), 1, d.f(12), d.f(13), d.flag(), d.stream));
            observe("qsa_gate_raw", d.f(13), 6144);
            hip(launch_group_rms_norm(d.f(12), q_norm.get(), 1, 6144, 256, epsilon, true, d.f(14), d.flag(), d.stream));
            hip(launch_group_rms_norm(d.f(10), k_norm.get(), 1, 512, 256, epsilon, true, d.f(15), d.flag(), d.stream));
            hip(launch_qsa_rope(d.f(14), frequencies.get(), 1, 24, position, d.f(14), d.flag(), d.stream));
            hip(launch_qsa_rope(d.f(15), frequencies.get(), 1, 2, position, d.f(15), d.flag(), d.stream));
            launch_hadamard(d.f(14), d.f(12), 6144, 256, d.stream);
            launch_hadamard(d.f(15), d.f(10), 512, 256, d.stream);
            launch_hadamard(d.f(11), d.f(15), 512, 64, d.stream);
            observe("qsa_query_rotated", d.f(12), 6144);
            observe("qsa_key_rotated", d.f(10), 512); observe("qsa_value_rotated", d.f(15), 512);
            launch_quantize_q4(d.f(10), kv_k.get<Q4_0>() + position * 16, 512, d.flag(), d.stream);
            launch_quantize_q4(d.f(15), kv_v.get<Q4_0>() + position * 16, 512, d.flag(), d.stream);
            index_q.apply(d, d.f(6), d.f(14)); index_k.apply(d, d.f(6), d.f(15));
            QsaIndexConfig cfg; cfg.capacity = static_cast<std::size_t>(capacity); cfg.rms_epsilon = epsilon; cfg.rotary_dim = 64;
            QsaIndexDeviceParameters p{cfg, iq_norm.get(), ik_norm.get(), frequencies.get()};
            hip(launch_qsa_index_append(p, d.f(15), position, 1, pooled.get(), tail.get(),
                {d.q8.get<Q4_0>(), d.f(16), d.f(17), d.f(18), nullptr, d.flag()}, d.stream));
            hip(launch_qsa_index_score(p, d.f(14), pooled.get(), position + 1, d.f(15), d.f(16), d.flag(), d.stream));
            hip(launch_qsa_select(d.f(16), position + 1, d.ids.get<std::int32_t>(), d.block_ids.get<std::int32_t>(), d.counts.get<int>(),
                {d.histogram.get<int>(), d.select_state.get<std::uint64_t>(), d.candidates.get<std::uint64_t>(), d.flag()}, d.stream));
            hip(launch_attention_q4(d.f(12), kv_k.get<Q4_0>(), kv_v.get<Q4_0>(), capacity, d.ids.get<std::int32_t>(),
                d.counts.get<int>(), position + 1, d.f(14),
                {d.gathered_k.get<std::uint16_t>(), d.gathered_v.get<std::uint16_t>(), d.partial_o.get(),
                 d.partial_ms.get<float2>(), d.f(15), d.flag()}, d.stream));
            observe("qsa_attention_rotated", d.f(14), 6144);
            launch_hadamard(d.f(14), d.f(12), 6144, 64, d.stream);
            observe("qsa_attention_inverse_h64", d.f(12), 6144);
            hip(launch_sigmoid_gate(d.f(12), d.f(13), 6144, d.f(14), d.flag(), d.stream));
            observe("qsa_attention_gated", d.f(14), 6144);
            // The common post-attention output buffer is f(13); raw gate is now dead.
            hip(hipMemcpyAsync(d.f(13), d.f(14), 6144 * sizeof(float), hipMemcpyDeviceToDevice, d.stream));
        }
        attn_output.apply(d, d.f(13), d.f(9));
        observe("block_attention_out", d.f(9), H);
    }
    void ffn(SessionStats& stats) {
        auto& d = device;
        router.apply(d, d.f(6), d.f(9));
        hip(launch_router_softmax(d.f(9), 1, d.f(10), d.flag(), d.stream));
        std::array<float, 512> probabilities{};
        hip(hipMemcpyAsync(probabilities.data(), d.f(10), sizeof(probabilities), hipMemcpyDeviceToHost, d.stream));
        d.check("router layer " + std::to_string(index));
        std::array<int, 512> order{}; std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + 10, order.end(), [&](int i, int j) {
            return probabilities[i] > probabilities[j] || (probabilities[i] == probabilities[j] && i < j);
        });
        // Production fused topk-moe collects selected values by expert ID's
        // logical lane, then performs an ascending-XOR width32 reduction.
        std::array<float, 32> buckets{};
        for (int i = 0; i < 10; ++i) buckets[order[i] % 32] += probabilities[order[i]];
        for (int distance = 1; distance < 32; distance *= 2) {
            const auto previous = buckets;
            for (int lane = 0; lane < 32; ++lane)
                buckets[lane] = previous[lane] + previous[lane ^ distance];
        }
        const float denominator = std::max(buckets[0], 1.0f / 16384.0f);
        if (!(denominator > 0) || !std::isfinite(denominator)) throw std::runtime_error("invalid route denominator");
        const float inverse = 1.0f / denominator;
        // Dedicated input copy: the common quantization workspace is overwritten
        // by expert down inputs. Keep the original quantized X for all ten routes.
        hip(launch_quantize_q8_1(d.f(6), d.q8.get<Q8_1>(), H, 1, d.flag(), d.stream));
        hip(hipMemcpyAsync(d.f(21), d.q8.get<Q8_1>(), (H / 32) * sizeof(Q8_1), hipMemcpyDeviceToDevice, d.stream));
        hip(hipMemsetAsync(d.f(20), 0, H * sizeof(float), d.stream));
        for (int i = 0; i < 10; ++i)
            experts.run(d, order[i], reinterpret_cast<const Q8_1*>(d.f(21)), probabilities[order[i]] * inverse, stats);
        shared_gate.apply(d, d.f(6), d.f(16)); shared_up.apply(d, d.f(6), d.f(17));
        hip(launch_silu_up(d.f(16), d.f(17), 640, d.f(18), d.flag(), d.stream));
        shared_down.apply(d, d.f(18), d.f(19));
        blas(launch_dense_linear(d.handle, {shared_weight.get(), H, 1}, d.f(6), 1, d.f(16), d.stream));
        hip(launch_shared_sigmoid_add(d.f(19), d.f(16), H, d.f(20), d.flag(), d.stream));
    }
};
}

struct Session::Impl {
    Model model;
    SessionConfig config;
    SessionStats statistics;
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
    struct Pinned {
        float* pointer = nullptr;
        Pinned() { hip(hipHostMalloc(&pointer, W * sizeof(float), hipHostMallocDefault)); }
        ~Pinned() { if (pointer) cleanup(hipHostFree(pointer)); }
    };
    std::unique_ptr<Pinned> pinned;
    float* handoff = nullptr;
    float epsilon;
    bool usable = true;
    std::ofstream trace_metadata;
    std::uint64_t trace_occurrence = 0;

    Impl(const std::string& path, SessionConfig cfg)
        : model(path), config(std::move(cfg)), ple_hash(ple_hash_config(model, 1)),
          ple_lookup(model, ple_hash.state().config()), host_logits(V) {
        if (config.capacity < 4 || config.capacity > 131072 || config.capacity % 4 || config.expert_slots < 1 || config.expert_slots > 512)
            throw std::invalid_argument("Session capacity/slots");
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
        const auto& embedding = require(model, "token_embd.weight", {H, V});
        embedding_type = embedding.type; embedding_stride = static_cast<std::size_t>(embedding.strides[1]);
        embeddings.resize(static_cast<std::size_t>(embedding.byte_size)); model.read_tensor(embedding.name, embeddings);
        int n = 0; hip(hipGetDeviceCount(&n)); if (n != 2) throw std::runtime_error("Session requires two gfx906");
        devices[0] = std::make_unique<Device>(0); devices[1] = std::make_unique<Device>(1);
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
        pinned = std::make_unique<Pinned>(); handoff = pinned->pointer;
        reset();
    }
    ~Impl() {
        for (auto& d : devices) if (d) { cleanup(hipSetDevice(d->id)); cleanup(hipStreamSynchronize(d->stream)); }
    }
    void reset() {
        for (auto& l : layers) l->reset();
        hip(hipSetDevice(0)); hip(hipMemsetAsync(ple_history.get<void>(), 0, ple_history.bytes(), devices[0]->stream));
        for (auto& d : devices) { hip(hipSetDevice(d->id)); d->clear(); hip(hipStreamSynchronize(d->stream)); }
        ple_hash.reset(); statistics = {}; usable = true;
    }
    SessionMemory memory() const {
        int previous = 0; hip(hipGetDevice(&previous));
        struct RestoreDevice {
            int device;
            ~RestoreDevice() { cleanup(hipSetDevice(device)); }
        } restore{previous};
        SessionMemory result;
        result.capacity = config.capacity; result.expert_slots = config.expert_slots;
        result.host_embedding_capacity = embeddings.capacity();
        result.host_logit_capacity = host_logits.capacity() * sizeof(float);
        result.pinned_handoff = W * sizeof(float);
        for (int id = 0; id < 2; ++id) {
            auto& out = result.devices[id];
            auto& d = *devices[id];
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
            for (const auto* b : {&d.q8, &d.error, &d.logits, &d.gathered_k, &d.gathered_v,
                    &d.partial_o, &d.partial_ms, &d.histogram, &d.select_state, &d.candidates,
                    &d.ids, &d.block_ids, &d.counts}) add(out.workspace, *b);
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
            if (out.owned_bytes != Buffer::live_bytes[id] || out.owned_buffers != Buffer::live_buffers[id])
                throw std::runtime_error("Session allocation ledger omitted a live Buffer");
            out.owned_peak_bytes = Buffer::peak_bytes[id];
            hip(hipSetDevice(id)); hip(hipStreamSynchronize(d.stream));
            std::size_t free = 0, total = 0; hip(hipMemGetInfo(&free, &total));
            out.total_vram = total; out.free_vram = free;
        }
        result.ownership_verified = true;
        return result;
    }
    void capture(Device& d, const std::string& name, const float* x, int count, int layer = -1) {
        if (!trace_metadata.is_open()) return;
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
        trace_metadata << "{\"name\":" << std::quoted((layer == 2 || layer == 6 || layer == 8 || layer == 19 || layer == 22) ? "debug_" + name : name) << ",\"layer\":" << layer << ",\"position\":" << statistics.consumed_tokens
                       << ",\"type\":\"F32\",\"elements\":" << count << ",\"file\":" << std::quoted(file) << "}\n";
        trace_metadata.flush(); if (!trace_metadata) throw std::runtime_error("trace metadata write");
    }
    void ple(Device& d, std::int32_t token, float*& residual, float*& alternate) {
        ple_hash.step(token, ple_rows); ple_lookup.lookup(ple_rows, ple_embedding);
        hip(hipMemcpyAsync(d.f(9), ple_embedding.data(), sizeof(ple_embedding), hipMemcpyHostToDevice, d.stream));
        capture(d, "ple_embedding", d.f(9), H, 1);
        ple_key.apply(d, d.f(9), d.f(10)); ple_value.apply(d, d.f(9), d.f(11));
        capture(d, "ple_key_raw", d.f(10), W, 1); capture(d, "ple_value_raw", d.f(11), H, 1);
        hip(launch_group_rms_norm(d.f(10), ple_key_norm.get(), 1, W, H, epsilon, false, d.f(12), d.flag(), d.stream));
        hip(launch_group_rms_norm(residual, ple_query_norm.get(), 1, W, H, epsilon, false, d.f(13), d.flag(), d.stream));
        hip(launch_ple_gate(d.f(12), d.f(13), d.f(11), 1, d.f(14), d.flag(), d.stream));
        capture(d, "ple_gated_value", d.f(14), W, 1);
        hip(launch_group_rms_norm(d.f(14), ple_conv_norm.get(), 1, W, H, epsilon, false, d.f(15), d.flag(), d.stream));
        hip(launch_ple_causal_conv(d.f(14), d.f(15), ple_conv.get(), ple_history.get(), 1, nullptr,
            {d.f(16), ple_stage_history.get(), nullptr}, d.flag(), d.stream));
        hip(launch_publish_ple_history(ple_stage_history.get(), ple_history.get(), d.flag(), d.stream));
        hip(launch_add(residual, d.f(16), W, alternate, d.flag(), d.stream)); std::swap(residual, alternate);
        capture(d, "ple_residual", residual, W, 1);
    }
    std::span<const float> step(std::int32_t token) {
        if (!usable) throw std::runtime_error("Session invalidated; reset required");
        if (token < 0 || token >= V || statistics.consumed_tokens >= static_cast<std::uint64_t>(config.capacity))
            throw std::invalid_argument("token ID/capacity");
        const auto begin = std::chrono::steady_clock::now();
        try {
            auto* d = devices[0].get(); hip(hipSetDevice(0)); d->clear();
            const auto row = std::span(embeddings).subspan(embedding_stride * token, embedding_stride);
            for (int i = 0; i < H; ++i) {
                const float x = tensor_element(embedding_type, row, static_cast<std::size_t>(i));
                if (!std::isfinite(x)) throw std::runtime_error("embedding nonfinite");
                for (int c = 0; c < 4; ++c) handoff[c * H + i] = x;
            }
            float* residual = d->f(0); float* alternate = d->f(1);
            hip(hipMemcpyAsync(residual, handoff, W * sizeof(float), hipMemcpyHostToDevice, d->stream));
            capture(*d, "hc_init", residual, W);
            for (int l = 0; l < 48; ++l) {
                if (l == 24) {
                    d->check("split producer");
                    hip(hipMemcpyAsync(handoff, residual, W * sizeof(float), hipMemcpyDeviceToHost, d->stream));
                    hip(hipStreamSynchronize(d->stream));
                    d = devices[1].get(); hip(hipSetDevice(1)); d->clear(); residual = d->f(0); alternate = d->f(1);
                    hip(hipMemcpyAsync(residual, handoff, W * sizeof(float), hipMemcpyHostToDevice, d->stream));
                }
                auto& layer = *layers[l];
                if (l == 1) ple(*d, token, residual, alternate);
                layer.attention_hc.mix(*d, residual, epsilon);
                if (l == 0 || l == 1 || l == 2 || l == 3 || l == 6 || l == 8 || l == 19 || l == 22 || l == 47) {
                    capture(*d, "hc_attn_mix", d->f(6), H, l);
                    capture(*d, "hc_attn_norm", d->f(2), W, l);
                    capture(*d, "hc_attn_gate_raw", d->f(5), W, l);
                    capture(*d, "hc_attn_inject_raw", d->f(7), 4, l);
                }
                layer.attention(static_cast<int>(statistics.consumed_tokens), config.capacity, epsilon,
                    [&](const char* name, const float* x, int count) {
                        if (l == 0 || l == 1 || l == 2 || l == 3 || l == 6 || l == 8 || l == 19 || l == 22 || l == 47) capture(*d, name, x, count, l);
                    });
                hip(launch_hc_combine(residual, d->f(9), d->f(8), 1, alternate, d->flag(), d->stream));
                std::swap(residual, alternate);
                layer.ffn_hc.mix(*d, residual, epsilon);
                if (l == 0 || l == 1 || l == 2 || l == 3 || l == 6 || l == 8 || l == 19 || l == 22 || l == 47) {
                    capture(*d, "hc_ffn_mix", d->f(6), H, l);
                    capture(*d, "hc_ffn_norm", d->f(2), W, l);
                    capture(*d, "hc_ffn_gate_raw", d->f(5), W, l);
                    capture(*d, "hc_ffn_inject_raw", d->f(7), 4, l);
                }
                layer.ffn(statistics);
                if (l == 0 || l == 1 || l == 2 || l == 3 || l == 6 || l == 8 || l == 19 || l == 22 || l == 47) capture(*d, "ffn_out", d->f(20), H, l);
                hip(launch_hc_combine(residual, d->f(20), d->f(8), 1, alternate, d->flag(), d->stream));
                std::swap(residual, alternate);
                d->check("layer " + std::to_string(l));
                capture(*d, "l_last-" + std::to_string(l), residual, W, l);
            }
            head_hc->mix(*d, residual, epsilon, true); capture(*d, "result_norm", d->f(6), H);
            head.apply(*d, d->f(6), d->logits.get());
            hip(hipMemcpyAsync(host_logits.data(), d->logits.get(), V * sizeof(float), hipMemcpyDeviceToHost, d->stream));
            d->check("head");
            for (float v : host_logits) if (!std::isfinite(v)) throw std::runtime_error("logits nonfinite");
            ++statistics.consumed_tokens;
            statistics.last_completed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            return host_logits;
        } catch (...) { usable = false; throw; }
    }
};

Session::Session(const std::string& path, SessionConfig config) : impl_(std::make_unique<Impl>(path, std::move(config))) {}
Session::~Session() = default;
std::span<const float> Session::step(std::int32_t token) { return impl_->step(token); }
void Session::reset() { impl_->reset(); }
SessionStats Session::stats() const { return impl_->statistics; }
SessionMemory Session::memory() const { return impl_->memory(); }

}
