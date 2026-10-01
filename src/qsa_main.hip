#include "dense.hpp"
#include "kv.hpp"
#include "qsa.hpp"
#include "qsa_index.hpp"
#include "hip/attention.cuh"
#include "hip/qsa_index.cuh"
#include "hip/qsa_select.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "QSA fixtures require finite checks and unfused FP32 CPU arithmetic"
#endif

namespace {
constexpr int capacity = 131072;
constexpr int key_elements = capacity / 4 * 128;
constexpr int tail_elements = 384;
constexpr int repeats = 20;
constexpr int protocol_rows = 62;
constexpr std::int32_t id_canary = -31337;
constexpr float output_canary = -777.0f;
// Frozen BEFORE the first GPU execution. All three gates use the CPU reference
// magnitude, never the GPU magnitude. No dynamic/relative-to-observed threshold.
constexpr double absolute_gate = 2e-4;
constexpr double relative_gate = 2e-4;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void cleanup(hipError_t status) noexcept {
    if (status != hipSuccess) {
        std::cerr << "QSA cleanup: " << hipGetErrorString(status) << '\n';
        std::abort();
    }
}
struct Gate {
    double max_abs = 0, max_bound_ratio = 0;
    std::size_t elements = 0;
    void compare(std::span<const float> got, std::span<const float> ref, const char* what) {
        require(got.size() == ref.size(), "QSA comparison shape");
        for (std::size_t i = 0; i < got.size(); ++i) {
            require(std::isfinite(got[i]) && std::isfinite(ref[i]), what);
            const double error = std::abs(double(got[i]) - double(ref[i]));
            const double bound = absolute_gate + relative_gate * std::abs(double(ref[i]));
            max_abs = std::max(max_abs, error);
            max_bound_ratio = std::max(max_bound_ratio, error / bound);
            ++elements;
            if (error > bound)
                std::cerr << what << ": element=" << i << " got=" << std::setprecision(17) << got[i]
                          << " reference=" << ref[i] << " error=" << error << " bound=" << bound << '\n';
            require(error <= bound, what);
        }
    }
};
template<class T> void exact(std::span<const T> got, std::span<const T> ref, const char* what) {
    require(got.size() == ref.size(), what);
    require(got.empty() || std::memcmp(got.data(), ref.data(), got.size_bytes()) == 0, what);
}
template<class T> void exact(const std::vector<T>& got, const std::vector<T>& ref, const char* what) {
    exact<T>(std::span<const T>(got), std::span<const T>(ref), what);
}
std::array<float, tail_elements> canonical_tail(std::span<const float> active) {
    std::array<float, tail_elements> result{};
    require(active.size() <= result.size(), "QSA tail shape");
    std::copy(active.begin(), active.end(), result.begin());
    return result;
}
void json_string(std::string_view value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << char(c);
        else if (c < 32) {
            constexpr char hex[] = "0123456789abcdef";
            std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else std::cout << char(c);
    }
    std::cout << '"';
}
void gate_json(const Gate& gate) {
    std::cout << "{\"max_abs\":" << gate.max_abs
              << ",\"max_bound_ratio\":" << gate.max_bound_ratio
              << ",\"elements\":" << gate.elements << '}';
}

struct Data {
    qwen::QsaIndexConfig config;
    float rope_base = 0;
    std::vector<float> qnorm, knorm, frequencies, raw_queries, raw_keys;
    explicit Data(const qwen::Model& model) {
        const auto& architecture = model.metadata_value("general.architecture");
        require(architecture.type == qwen::MetadataType::STRING &&
                architecture.get<std::string>() == "qwen4exp", "QSA requires qwen4exp GGUF");
        const auto& rotary = model.metadata_value("qwen4exp.rope.dimension_count");
        const auto& base = model.metadata_value("qwen4exp.rope.freq_base");
        const auto& epsilon = model.metadata_value("qwen4exp.attention.layer_norm_rms_epsilon");
        const auto& sections = model.metadata_value("qwen4exp.rope.dimension_sections");
        require(rotary.type == qwen::MetadataType::UINT32 && rotary.get<std::uint32_t>() == 64,
                "QSA actual rotary metadata type/value");
        require(base.type == qwen::MetadataType::FLOAT32 && base.get<float>() == 10000000.0f,
                "QSA actual RoPE base metadata type/value");
        require(epsilon.type == qwen::MetadataType::FLOAT32 && epsilon.get<float>() == 1e-6f,
                "QSA actual epsilon metadata type/value");
        require(sections.type == qwen::MetadataType::ARRAY, "QSA actual RoPE sections type");
        const auto& array = sections.get<qwen::MetadataArray>();
        require(array.element_type == qwen::MetadataType::INT32 &&
                array.get<std::int32_t>() == std::vector<std::int32_t>({11, 11, 10, 0}),
                "QSA actual RoPE sections type/value");
        config.capacity = capacity;
        config.rotary_dim = rotary.get<std::uint32_t>();
        config.rms_epsilon = epsilon.get<float>();
        rope_base = base.get<float>();
        frequencies.resize(32);
        for (int i = 0; i < 32; ++i)
            frequencies[i] = 1.0f / std::pow(rope_base, float(2 * i) / 64.0f);
        auto load = [&](const char* name, std::initializer_list<std::uint64_t> dims,
                        qwen::TensorType type) {
            const auto& tensor = model.tensor(name);
            require(tensor.type == type && tensor.rank == dims.size() &&
                    std::equal(dims.begin(), dims.end(), tensor.dimensions.begin()),
                    "QSA actual tensor type/geometry");
            std::vector<std::byte> bytes(tensor.byte_size);
            model.read_tensor(name, bytes);
            return bytes;
        };
        auto norm = [&](const char* name) {
            const auto bytes = load(name, {128}, qwen::TensorType::F32);
            std::vector<float> result(128);
            for (int i = 0; i < 128; ++i) {
                result[i] = qwen::tensor_element(qwen::TensorType::F32, bytes, i);
                require(std::isfinite(result[i]), "QSA nonfinite loaded norm");
            }
            return result;
        };
        qnorm = norm("blk.3.indexer.q_norm.weight");
        knorm = norm("blk.3.indexer.k_norm.weight");
        std::vector<float> x(8 * 2560);
        for (std::size_t i = 0; i < x.size(); ++i)
            x[i] = .3f * std::sin(float(i) * .137f) + .1f * std::cos(float(i) * .071f);
        auto project = [&](const char* name, int width) {
            const auto bytes = load(name, {2560, std::uint64_t(width)}, qwen::TensorType::BF16);
            std::vector<float> result(8 * width);
            qwen::matmul_f32({qwen::TensorType::BF16, 2560, width, bytes}, x, 8, result);
            return result;
        };
        raw_queries = project("blk.3.indexer.q_proj.weight", 512);
        raw_keys = project("blk.3.indexer.k_proj.weight", 128);
    }
    qwen::QsaIndexParameters parameters() const { return {qnorm, knorm, frequencies}; }
    std::span<const float> query(int visible) const {
        return std::span<const float>(raw_queries).subspan(((visible - 1) % 8) * 512, 512);
    }
    std::vector<float> keys(int start, int n) const {
        std::vector<float> result(n * 128);
        for (int t = 0; t < n; ++t)
            std::copy_n(raw_keys.data() + ((start + t) % 8) * 128, 128, result.data() + t * 128);
        return result;
    }
};

std::vector<int> visibilities() {
    std::vector<int> result{1, 2, 3, 4, 63, 64, 65};
    for (int n = 2047; n <= 2056; ++n) result.push_back(n);
    for (int n : {4096, 32768, capacity}) result.push_back(n);
    return result;
}

// Test-only retained CPU history: the production device restore copies only 384
// tail floats. The eight projected columns, NOT new prompt projections, repeat.
struct Timeline {
    qwen::QsaIndexCpu cpu;
    std::map<int, qwen::QsaIndexCheckpoint> prefixes;
    std::vector<float> keys;
    Timeline(const Data& data) : cpu(data.config, data.parameters()) {
        int start = 0;
        for (int end : visibilities()) {
            while (start < end) {
                const int n = std::min(128, end - start);
                cpu.append(data.keys(start, n), n);
                start += n;
            }
            cpu.save(prefixes[end]);
        }
        keys.assign(cpu.completed_keys().begin(), cpu.completed_keys().end());
    }
    std::vector<float> scores(const Data& data, int visible) {
        cpu.restore(prefixes.at(visible));
        std::vector<float> result(visible / 4);
        cpu.score(data.query(visible), result);
        return result;
    }
};

struct Summary {
    Gate keys, common_scores, actual_scores, attention;
    std::size_t cache_cases = 0, prefix_checks = 0, restores = 0, reject_windows = 0;
    std::size_t q4_cases = 0, append_rejects = 0, selection_checks = 0;
    std::size_t selection_rejects = 0, score_checks = 0, score_rejects = 0;
    std::size_t attention_checks = 0, attention_rejects = 0;
    std::map<int, std::vector<std::int32_t>> gpu_ids, cpu_ids;
};

struct Fixture {
    int device, length = 0;
    hipStream_t stream{};
    hipEvent_t begin{}, end{};
    std::vector<void*> owned;
    qwen::QsaIndexDeviceParameters parameters{};
    qwen::QsaIndexAppendWorkspace append_workspace{};
    qwen::QsaSelectWorkspace select_workspace{};
    qwen::AttentionQ4DeviceWorkspace attention_workspace{};
    float *cache = nullptr, *tail = nullptr, *saved_tail = nullptr;
    float *raw_keys = nullptr, *raw_query = nullptr, *normalized_query = nullptr, *scores = nullptr;
    std::int32_t *token_storage = nullptr, *block_storage = nullptr;
    int *count_storage = nullptr, *error = nullptr;
    qwen::Q4_0 *attention_keys = nullptr, *attention_values = nullptr;
    float *attention_query = nullptr, *output_storage = nullptr;
    explicit Fixture(int d) : device(d) {}
    Fixture(const Fixture&) = delete;
    Fixture& operator=(const Fixture&) = delete;
    ~Fixture() {
        cleanup(hipSetDevice(device));
        if (stream) cleanup(hipStreamSynchronize(stream));
        for (void* allocation : owned) cleanup(hipFree(allocation));
        if (begin) cleanup(hipEventDestroy(begin));
        if (end) cleanup(hipEventDestroy(end));
        if (stream) cleanup(hipStreamDestroy(stream));
    }
    template<class T> T* allocate(std::size_t n) {
        void* pointer = nullptr;
        check(hipMalloc(&pointer, n * sizeof(T)));
        try { owned.push_back(pointer); }
        catch (...) { cleanup(hipFree(pointer)); throw; }
        return static_cast<T*>(pointer);
    }
    template<class T> void put(T* pointer, std::span<const T> values) {
        if (!values.empty())
            check(hipMemcpyAsync(pointer, values.data(), values.size_bytes(), hipMemcpyHostToDevice, stream));
        // Fixture uploads are outside timers. Completion protects temporary host
        // vectors even on runtimes where pageable async copies do not stage eagerly.
        check(hipStreamSynchronize(stream));
    }
    template<class T, class Container> void put(T* pointer, const Container& values) {
        put<T>(pointer, std::span<const T>(values));
    }
    template<class T> std::vector<T> get(const T* pointer, std::size_t n) {
        std::vector<T> result(n);
        if (n != 0)
            check(hipMemcpyAsync(result.data(), pointer, n * sizeof(T), hipMemcpyDeviceToHost, stream));
        check(hipStreamSynchronize(stream));
        return result;
    }
    void initialize(const Data& data) {
        check(hipSetDevice(device));
        hipDeviceProp_t properties{};
        check(hipGetDeviceProperties(&properties, device));
        const std::string arch(properties.gcnArchName);
        require((arch == "gfx906" || arch.starts_with("gfx906:")) && properties.warpSize == 64,
                "QSA requires exactly gfx906 wave64");
        check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        check(hipEventCreate(&begin)); check(hipEventCreate(&end));
        parameters = {data.config, allocate<float>(128), allocate<float>(128), allocate<float>(32)};
        put(const_cast<float*>(parameters.query_norm), data.qnorm);
        put(const_cast<float*>(parameters.key_norm), data.knorm);
        put(const_cast<float*>(parameters.inverse_frequencies), data.frequencies);
        exact<float>(get(parameters.inverse_frequencies, 32), data.frequencies, "QSA frequency upload bits");
        error = allocate<int>(1);
        cache = allocate<float>(key_elements); tail = allocate<float>(tail_elements);
        saved_tail = allocate<float>(tail_elements);
        raw_keys = allocate<float>(128 * 128); raw_query = allocate<float>(512);
        normalized_query = allocate<float>(512); scores = allocate<float>(capacity / 4 + 1);
        append_workspace = {allocate<qwen::Q4_0>(128 * 4), allocate<float>(128 * 128),
                            allocate<float>(33 * 128), allocate<float>(tail_elements),
                            allocate<float>(129 * tail_elements), error};
        select_workspace = {allocate<int>(32 * 256), allocate<std::uint64_t>(16),
                            allocate<std::uint64_t>(512), error};
        token_storage = allocate<std::int32_t>(2053);
        block_storage = allocate<std::int32_t>(514); count_storage = allocate<int>(4);
        attention_keys = allocate<qwen::Q4_0>(std::size_t(capacity) * 16);
        attention_values = allocate<qwen::Q4_0>(std::size_t(capacity) * 16);
        attention_query = allocate<float>(6144); output_storage = allocate<float>(6146);
        attention_workspace = {allocate<std::uint16_t>(qwen::attention_gather_elements),
                               allocate<std::uint16_t>(qwen::attention_gather_elements),
                               allocate<float>(qwen::attention_partial_elements),
                               allocate<float2>(qwen::attention_partial_ml_elements),
                               allocate<float>(6144), error};
        reset();
    }
    void clear_error() { check(hipMemsetAsync(error, 0, sizeof(int), stream)); }
    int flag() { return get(error, 1)[0]; }
    void valid() { require(flag() == 0, "QSA unexpected sticky error"); }
    void reset() {
        check(hipMemsetAsync(cache, 0xa5, std::size_t(key_elements) * sizeof(float), stream));
        check(hipMemsetAsync(tail, 0, tail_elements * sizeof(float), stream));
        clear_error(); length = 0;
    }
    void append(int start, int n, bool prefixes = false) {
        auto workspace = append_workspace;
        if (!prefixes) workspace.tail_prefixes = nullptr;
        check(qwen::launch_qsa_index_append(parameters, raw_keys, start, n, cache, tail, workspace, stream));
    }
    void consume(int n, bool prefixes = false) {
        const int old = length;
        append(old, n, prefixes);
        valid(); // Logical publication is caller-held and ONLY after flag == 0.
        length = old + n;
    }
    void prepare(const Data& data, int n) {
        reset();
        while (length < n) {
            const int count = std::min(128, n - length);
            put(raw_keys, data.keys(length, count));
            consume(count);
        }
    }
    void restore(int start, int slot) {
        check(hipMemcpyAsync(tail, append_workspace.tail_prefixes + slot * tail_elements,
                             tail_elements * sizeof(float), hipMemcpyDeviceToDevice, stream));
        valid(); length = start + slot;
    }
    void score(int visible) {
        check(qwen::launch_qsa_index_score(parameters, raw_query, cache, visible,
                                         normalized_query, scores, error, stream));
    }
    void select(int visible) {
        check(qwen::launch_qsa_select(scores, visible, token_storage + 1, block_storage + 1,
                                     count_storage + 1, select_workspace, stream));
    }
    void attention(int visible) {
        check(qwen::launch_attention_q4(attention_query, attention_keys, attention_values, capacity,
                                       token_storage + 1, count_storage + 1, visible,
                                       output_storage + 1, attention_workspace, stream));
    }
    void id_canaries() {
        put(token_storage, std::vector<std::int32_t>(2053, id_canary));
        put(block_storage, std::vector<std::int32_t>(514, id_canary));
        put(count_storage, std::array<int, 4>{id_canary, id_canary, id_canary, id_canary});
    }
    void output_canaries() { put(output_storage, std::vector<float>(6146, output_canary)); }
    template<class Prepare, class Run, class Validate>
    double timed(Prepare prepare, Run run, Validate validate) {
        double total = 0;
        for (int r = 0; r < repeats; ++r) {
            prepare(); // All reset/restore/H2D/flag clearing is OUTSIDE events.
            check(hipStreamSynchronize(stream));
            check(hipEventRecord(begin, stream)); run(); check(hipEventRecord(end, stream));
            check(hipEventSynchronize(end));
            float ms = 0; check(hipEventElapsedTime(&ms, begin, end));
            require(std::isfinite(ms) && ms > 0, "QSA invalid completed event time");
            total += ms;
            // Validate EVERY completed repetition, including its sticky flag.
            // Host checks and D2H are outside both events and the next interval.
            valid(); validate();
        }
        return total / repeats;
    }
};

void compare_cache(Fixture& f, const qwen::QsaIndexCpu& cpu, Gate& gate) {
    require(f.length == int(cpu.consumed_tokens()), "QSA logical length parity");
    gate.compare(f.get(f.cache, cpu.completed_keys().size()), cpu.completed_keys(), "QSA pooled key gate");
    const auto ref = canonical_tail(cpu.raw_tail());
    exact<float>(f.get(f.tail, tail_elements), ref, "QSA canonical raw tail byte parity");
}
void compare_prefixes(Fixture& f, std::span<const qwen::QsaIndexCheckpoint> prefixes, Summary& summary) {
    const auto got = f.get(f.append_workspace.tail_prefixes, prefixes.size() * tail_elements);
    for (std::size_t i = 0; i < prefixes.size(); ++i) {
        const auto ref = canonical_tail(prefixes[i].raw_tail());
        exact<float>(std::span<const float>(got).subspan(i * tail_elements, tail_elements), ref,
                     "QSA chronological N+1 tail prefix bytes");
        ++summary.prefix_checks;
    }
}

void q4_cases(Fixture& f, const Data& data, Summary& summary) {
    std::vector<float> input(2 * 128);
    // Independent qualified kv.hpp CPU pack/decode. Includes all-zero, signed
    // zero, both first-max signs, FP16-subnormal scale and an original-vs-rounded
    // reciprocal boundary. No standalone test GPU quant kernel is introduced.
    for (int i = 32; i < 64; ++i) input[i] = i % 2 ? -0.0f : 0.0f;
    input[64] = 8.003f; input[65] = -8.003f; input[66] = .5002f;
    input[96] = -17.3f; input[97] = 17.3f;
    for (int i = 128; i < 160; ++i) input[i] = std::ldexp(float(i - 144), -20);
    input[160] = 8.003f; input[161] = .5001f; input[162] = .5002f;
    input[192] = -8.0f; input[208] = 8.0f;
    std::vector<qwen::Q4_0> packed(8);
    std::vector<float> decoded(input.size());
    qwen::quantize_q4(input, packed); qwen::dequantize_q4(packed, decoded);
    require(packed[0].d == 0x8000 && packed[1].d == 0x8000, "QSA zero scale fixture");
    require((packed[2].d & 0x8000) && !(packed[3].d & 0x8000), "QSA first-max signs fixture");
    require((packed[4].d & 0x7c00) == 0 && (packed[4].d & 0x7fff) != 0,
            "QSA half-subnormal scale fixture");
    const int rounded_code = std::min(15, int(input[161] * (1.0f / qwen::half_to_float(packed[5].d)) + 8.5f));
    require(int(packed[5].qs[1] & 15) != rounded_code, "QSA unrounded-code boundary fixture qualification");
    f.reset(); f.put(f.raw_keys, input); f.consume(2);
    exact<qwen::Q4_0>(f.get(f.append_workspace.packed, 8), packed, "QSA fused Q4 bytes");
    exact<float>(f.get(f.append_workspace.roundtrip, input.size()), decoded, "QSA fused Q4 decode bits");
    qwen::QsaIndexCpu cpu(data.config, data.parameters()); cpu.append(input, 2);
    compare_cache(f, cpu, summary.keys);
    summary.q4_cases = 8;
}

void cache_cases(Fixture& f, const Data& data, Summary& summary) {
    for (int phase = 0; phase < 4; ++phase) for (int n : {1, 2, 3, 128}) {
        const int start = 4096 + phase;
        f.prepare(data, start);
        qwen::QsaIndexCpu cpu(data.config, data.parameters()); cpu.append(data.keys(0, start), start);
        const auto before = f.get(f.cache, key_elements);
        const auto input = data.keys(start, n);
        std::vector<qwen::QsaIndexCheckpoint> slots(n + 1);
        cpu.append(input, n, slots);
        f.put(f.raw_keys, input); f.clear_error(); f.consume(n, true);
        compare_cache(f, cpu, summary.keys); compare_prefixes(f, slots, summary);
        const auto after = f.get(f.cache, key_elements);
        const int first = start / 4 * 128, last = (start + n) / 4 * 128;
        exact<float>(std::span<const float>(after).first(first), std::span<const float>(before).first(first),
                     "QSA append modified completed prefix");
        exact<float>(std::span<const float>(after).subspan(last), std::span<const float>(before).subspan(last),
                     "QSA append modified unpublished suffix");
        ++summary.cache_cases;
        // Same input history via resident single-token appends, all four phases.
        f.prepare(data, start);
        for (int t = 0; t < n; ++t) {
            const auto old_keys = f.get(f.cache, std::size_t(f.length / 4) * 128);
            f.put(f.raw_keys, data.keys(start + t, 1)); f.clear_error(); f.consume(1);
            exact<float>(f.get(f.cache, old_keys.size()), old_keys, "QSA sequential immutable prefix");
        }
        compare_cache(f, cpu, summary.keys); ++summary.cache_cases;
    }
    for (int phase = 0; phase < 4; ++phase) for (int accept = 0; accept < 3; ++accept) {
        const int initial = 8 + phase;
        f.prepare(data, initial);
        qwen::QsaIndexCpu cpu(data.config, data.parameters()); cpu.append(data.keys(0, initial), initial);
        for (int window = 0; window < 2; ++window) {
            const int start = f.length;
            // Distinct repeated fixture columns for drafts/replacement prevent
            // stale-tail acceptance from being hidden by identical inputs.
            const auto verify = data.keys(2 + window * 3, 3);
            std::array<qwen::QsaIndexCheckpoint, 4> slots;
            cpu.append(verify, 3, slots);
            f.put(f.raw_keys, verify); f.clear_error(); f.consume(3, true);
            compare_cache(f, cpu, summary.keys); compare_prefixes(f, slots, summary);
            const int slot = 1 + accept;
            cpu.restore(slots[slot]); f.restore(start, slot); ++summary.restores;
            compare_cache(f, cpu, summary.keys);
            const auto retained = f.get(f.cache, std::size_t(f.length / 4) * 128);
            const auto replacement = data.keys(7 - window, 1);
            cpu.append(replacement, 1);
            f.put(f.raw_keys, replacement); f.clear_error(); f.consume(1);
            exact<float>(f.get(f.cache, retained.size()), retained, "QSA replacement modified retained prefix");
            compare_cache(f, cpu, summary.keys);
            if (accept == 0) ++summary.reject_windows;
        }
    }
}

void append_rejections(Fixture& f, const Data& data, Summary& summary) {
    f.prepare(data, 11);
    const auto before = f.get(f.cache, key_elements), tail = f.get(f.tail, tail_elements);
    const auto valid_input = data.keys(11, 3);
    auto unchanged = [&] {
        require(f.length == 11, "QSA failed append published logical length");
        exact<float>(f.get(f.cache, key_elements), before, "QSA failed chunk published keys");
        exact<float>(f.get(f.tail, tail_elements), tail, "QSA failed chunk published tail");
    };
    for (int kind = 0; kind < 6; ++kind) {
        auto input = valid_input;
        if (kind < 4) {
            const std::array<float, 4> bad{std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::infinity(), 1e-38f, std::numeric_limits<float>::max()};
            std::fill(input.end() - 128, input.end(), bad[kind]);
            std::vector<qwen::Q4_0> packed(12);
            bool rejected = false;
            try { qwen::quantize_q4(input, packed); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "QSA CPU Q4 bad fixture qualification");
        } else {
            auto norm = kind == 4 ? data.knorm : data.qnorm;
            norm[127] = std::numeric_limits<float>::quiet_NaN();
            f.put(const_cast<float*>(kind == 4 ? f.parameters.key_norm : f.parameters.query_norm), norm);
        }
        f.put(f.raw_keys, input); f.clear_error(); f.append(11, 3, true);
        const int rejected_flag = f.flag();
        require((rejected_flag & (kind == 2 || kind == 3 ? 2 : 1)) != 0, "QSA append rejection flag");
        unchanged(); ++summary.append_rejects;
        f.put(const_cast<float*>(f.parameters.key_norm), data.knorm);
        f.put(const_cast<float*>(f.parameters.query_norm), data.qnorm);
        f.put(f.raw_keys, valid_input); f.append(11, 3, true);
        require(f.flag() == rejected_flag, "QSA append changed sticky error"); unchanged();
    }
    f.clear_error(); f.consume(3);
    qwen::QsaIndexCpu cpu(data.config, data.parameters());
    cpu.append(data.keys(0, 11), 11); cpu.append(valid_input, 3);
    compare_cache(f, cpu, summary.keys);
}

struct Selection {
    std::vector<std::int32_t> tokens = std::vector<std::int32_t>(2053, id_canary);
    std::vector<std::int32_t> blocks = std::vector<std::int32_t>(514, id_canary);
    std::array<int, 4> counts{id_canary, id_canary, id_canary, id_canary};
    Selection(std::span<const float> scores, int visible) {
        const auto result = qwen::qsa_select(scores, visible, std::span(tokens).subspan(1, 2051),
                                             std::span(blocks).subspan(1, 512));
        counts[1] = int(result.token_count); counts[2] = int(result.block_count);
    }
    std::vector<std::int32_t> ids() const { return {tokens.begin() + 1, tokens.begin() + 1 + counts[1]}; }
    void compare(Fixture& f) const {
        exact<std::int32_t>(f.get(f.token_storage, 2053), tokens, "QSA exact selected IDs/unused token canaries");
        exact<std::int32_t>(f.get(f.block_storage, 514), blocks, "QSA exact block IDs/unused block canaries");
        exact<int>(f.get(f.count_storage, 4), counts, "QSA exact counts/count canaries");
    }
};
std::vector<float> score_pattern(int visible, int pattern) {
    std::vector<float> result(capacity / 4 + 1, std::numeric_limits<float>::quiet_NaN());
    std::uint32_t random = 0x13579bdfU;
    for (int i = 0; i < visible / 4; ++i) {
        random ^= random << 13; random ^= random >> 17; random ^= random << 5;
        switch (pattern) {
        case 0: result[i] = float(i); break;
        case 1: result[i] = float(int(random % 65521) - 32760) / 32.0f; break;
        case 2: result[i] = i < 511 ? 2.0f : i < 1024 ? 1.0f : -1.0f; break;
        case 3: result[i] = 0.0f; break;
        case 4: result[i] = i % 2 ? -0.0f : 0.0f; break;
        case 5: result[i] = std::bit_cast<float>(std::uint32_t((i % 2 ? 0x80000000U : 0) | (1 + i % 31))); break;
        case 6: {
            constexpr std::array<std::uint32_t, 8> bits{0x7f7fffffU, 0xff7fffffU, 0x00800000U,
                0x80800000U, 0x00000001U, 0x80000001U, 0x00000000U, 0x80000000U};
            result[i] = std::bit_cast<float>(bits[i % 8]); break;
        }
        case 7: result[i] = -float(1 + (i * 73) % 997); break;
        default: throw std::runtime_error("QSA unknown score fixture");
        }
    }
    return result;
}
void selection_cases(Fixture& f, Summary& summary) {
    for (int visible : visibilities()) for (int pattern = 0; pattern < 8; ++pattern) {
        const auto values = score_pattern(visible, pattern);
        const Selection reference(std::span<const float>(values).first(visible / 4), visible);
        f.put(f.scores, values); f.id_canaries(); f.clear_error(); f.select(visible); f.valid();
        reference.compare(f); ++summary.selection_checks;
    }
    for (int visible : {64, 2052, capacity}) {
        auto values = score_pattern(visible, 0);
        const Selection reference(std::span<const float>(values).first(visible / 4), visible);
        f.put(f.scores, values); f.id_canaries(); f.clear_error(); f.select(visible); f.valid();
        reference.compare(f);
        // ID0 would be discarded in the radix cases, but ALL visible scores must
        // still validate. Every score past the prefix already contains NaN.
        values[0] = std::numeric_limits<float>::quiet_NaN();
        f.put(f.scores, values); f.clear_error(); f.select(visible);
        require((f.flag() & 1) != 0, "QSA discarded-prefix NaN not rejected");
        reference.compare(f); ++summary.selection_rejects;
        values[0] = 0; f.put(f.scores, values); f.select(visible);
        require((f.flag() & 1) != 0, "QSA selector cleared sticky flag"); reference.compare(f);
        f.clear_error(); f.select(visible); f.valid(); reference.compare(f);
    }
}

void score_cases(Fixture& f, const Data& data, Timeline& timeline, Summary& summary) {
    f.prepare(data, capacity);
    const auto actual_keys = f.get(f.cache, key_elements);
    summary.keys.compare(actual_keys, timeline.keys, "QSA capacity pooled keys gate");
    for (int visible : visibilities()) {
        const auto reference = timeline.scores(data, visible);
        // Prepared future keys remain resident. FIRST isolate score arithmetic
        // with bit-identical CPU keys, then use the real GPU pooled-key path.
        for (int path = 0; path < 2; ++path) {
            f.put(f.cache, path == 0 ? timeline.keys : actual_keys);
            f.put(f.raw_query, data.query(visible));
            f.put(f.scores, std::vector<float>(capacity / 4 + 1, output_canary));
            f.clear_error(); f.score(visible); f.valid();
            const auto got = f.get(f.scores, capacity / 4 + 1);
            (path == 0 ? summary.common_scores : summary.actual_scores).compare(
                std::span<const float>(got).first(visible / 4), reference, "QSA causal score gate");
            require(std::all_of(got.begin() + visible / 4, got.end(),
                               [](float x) { return x == output_canary; }), "QSA score wrote future suffix");
            // EXACT selection is always compared on the SAME score floats. The
            // CPU and GPU arithmetic paths need NOT have identical cutoff IDs.
            const Selection same_score(std::span<const float>(got).first(visible / 4), visible);
            f.id_canaries(); f.select(visible); f.valid(); same_score.compare(f);
            ++summary.selection_checks; ++summary.score_checks;
            if (path == 1 && visible >= 2047 && visible <= 2056) summary.gpu_ids[visible] = same_score.ids();
        }
        if (visible >= 2047 && visible <= 2056) summary.cpu_ids[visible] = Selection(reference, visible).ids();
        // Poison EVERY future pooled key, not merely the first unseen row.
        auto poisoned = actual_keys;
        std::fill(poisoned.begin() + (visible / 4) * 128, poisoned.end(),
                  std::numeric_limits<float>::quiet_NaN());
        f.put(f.cache, poisoned); f.clear_error(); f.score(visible); f.valid();
        summary.actual_scores.compare(f.get(f.scores, visible / 4), reference, "QSA poisoned future key read");
        ++summary.score_checks;
    }
    f.put(f.cache, actual_keys); f.put(f.raw_query, data.query(4096));
    auto invalid_query = std::vector<float>(data.query(4096).begin(), data.query(4096).end());
    invalid_query[511] = std::numeric_limits<float>::quiet_NaN();
    f.put(f.raw_query, invalid_query); f.clear_error(); f.score(4096);
    require((f.flag() & 1) != 0, "QSA invalid index query not rejected"); ++summary.score_rejects;
    f.put(f.raw_query, data.query(4096)); f.score(4096);
    require((f.flag() & 1) != 0, "QSA scorer cleared sticky error");
    f.clear_error(); f.score(4096); f.valid();
    summary.actual_scores.compare(f.get(f.scores, 1024), timeline.scores(data, 4096), "QSA score valid reuse");
    auto invalid_keys = actual_keys; invalid_keys[0] = std::numeric_limits<float>::quiet_NaN();
    f.put(f.cache, invalid_keys); f.clear_error(); f.score(4096);
    require((f.flag() & 1) != 0, "QSA selected index key NaN not rejected"); ++summary.score_rejects;
    f.put(f.cache, actual_keys); f.score(4096);
    require((f.flag() & 1) != 0, "QSA key rejection sticky error");
    f.clear_error(); f.score(4096); f.valid();
    summary.actual_scores.compare(f.get(f.scores, 1024), timeline.scores(data, 4096), "QSA key rejection valid reuse");
}

struct AttentionData {
    std::vector<qwen::Q4_0> keys = std::vector<qwen::Q4_0>(std::size_t(capacity) * 16);
    std::vector<qwen::Q4_0> values = std::vector<qwen::Q4_0>(std::size_t(capacity) * 16);
    std::array<std::array<qwen::Q4_0, 16>, 8> key_rows{}, value_rows{};
    std::vector<float> query = std::vector<float>(6144);
    std::vector<std::int32_t> previous_ids;
    AttentionData() {
        for (auto& block : keys) block.d = 0x7c00;
        for (auto& block : values) block.d = 0x7c00;
        for (int t = 0; t < 8; ++t) {
            std::array<float, 512> k{}, v{};
            for (int h = 0; h < 2; ++h) for (int d = 0; d < 256; ++d) {
                k[h * 256 + d] = float((t * 71 + h * 97 + d * 29 + d * t * 3) % 257 - 128) / 128.0f;
                v[h * 256 + d] = float((t * 43 + h * 59 + d * 17) % 193 - 96) / 64.0f;
            }
            qwen::quantize_q4(k, key_rows[t]); qwen::quantize_q4(v, value_rows[t]);
        }
        for (int h = 0; h < 24; ++h) for (int d = 0; d < 256; ++d)
            query[h * 256 + d] = float((h * 37 + d * 47) % 127 - 63) / 128.0f;
    }
    void selected(std::span<const std::int32_t> ids) {
        for (int id : previous_ids) for (int b = 0; b < 16; ++b) {
            keys[id * 16 + b].d = 0x7c00; values[id * 16 + b].d = 0x7c00;
        }
        for (int id : ids) {
            require(id >= 0 && id < capacity, "QSA attention fixture ID range");
            std::copy(key_rows[id % 8].begin(), key_rows[id % 8].end(), keys.begin() + id * 16);
            std::copy(value_rows[id % 8].begin(), value_rows[id % 8].end(), values.begin() + id * 16);
        }
        previous_ids.assign(ids.begin(), ids.end());
    }
    void upload(Fixture& f, std::span<const std::int32_t> ids) {
        selected(ids); f.put(f.attention_keys, keys); f.put(f.attention_values, values);
        f.put(f.attention_query, query); f.id_canaries();
        f.put(f.token_storage + 1, ids);
        f.put(f.count_storage + 1, std::array<int, 2>{int(ids.size()), std::min(512, int(ids.size() / 4))});
        f.output_canaries(); f.clear_error();
    }
    std::vector<float> reference(std::span<const std::int32_t> ids, int visible) const {
        std::vector<float> scores(ids.size()), staged(6144), result(6144);
        qwen::attention_q4({keys, values, capacity}, query, ids, visible,
                           qwen::AttentionQ4Mode::gathered_fp16, {scores, staged}, result);
        return result;
    }
    void cache_bytes(Fixture& f) const {
        exact<qwen::Q4_0>(f.get(f.attention_keys, keys.size()), keys, "QSA attention modified read-only K cache");
        exact<qwen::Q4_0>(f.get(f.attention_values, values.size()), values, "QSA attention modified read-only V cache");
    }
};
std::vector<std::int32_t> attention_ids(int n) {
    std::vector<std::int32_t> ids(n);
    for (int i = 0; i < n; ++i) ids[i] = i != 0 && i % 7 == 0 ? ids[i - 1] : (i * 73 + 17) % 4096;
    return ids;
}
void compare_attention(Fixture& f, std::span<const float> reference, Gate& gate) {
    const auto got = f.get(f.output_storage, 6146);
    require(got.front() == output_canary && got.back() == output_canary, "QSA attention output canaries");
    gate.compare(std::span<const float>(got).subspan(1, 6144), reference, "QSA gathered-FP16 attention gate");
}
void attention_cases(Fixture& f, AttentionData& data, Summary& summary) {
    for (int visible = 2047; visible <= 2056; ++visible) {
        for (int path = 0; path < 2; ++path) {
            const auto& ids = (path == 0 ? summary.gpu_ids : summary.cpu_ids).at(visible);
            data.upload(f, ids);
            const auto ref = data.reference(ids, visible);
            f.attention(visible); f.valid(); compare_attention(f, ref, summary.attention);
            data.cache_bytes(f);
            ++summary.attention_checks;
        }
    }
    for (int n : {1, 63, 64, 65, 2048, 2051}) {
        const auto ids = attention_ids(n); data.upload(f, ids);
        const auto ref = data.reference(ids, 4096);
        f.attention(4096); f.valid(); compare_attention(f, ref, summary.attention);
        data.cache_bytes(f);
        ++summary.attention_checks;
    }
}
void attention_rejections(Fixture& f, AttentionData& data, Summary& summary) {
    const auto ids = attention_ids(65);
    data.upload(f, ids);
    const auto ref = data.reference(ids, 4096);
    f.attention(4096); f.valid(); compare_attention(f, ref, summary.attention);
    const auto before = f.get(f.output_storage, 6146);
    for (int kind = 0; kind < 11; ++kind) {
        // Keep successful output as the sentinel: errors must preserve ALL bits.
        auto counts = std::array<int, 2>{65, 16};
        auto invalid_ids = ids;
        if (kind == 0) counts[0] = 0;
        if (kind == 1) counts[0] = 2052;
        if (kind == 2) counts[1] = -1;
        if (kind == 3) counts[1] = 513;
        if (kind == 4) counts[1] = 17; // > actual tokens/4
        if (kind == 5) invalid_ids.back() = -1;
        if (kind == 6) invalid_ids.back() = 4096; // in allocation, outside visibility
        f.put(f.count_storage + 1, counts); f.put(f.token_storage + 1, invalid_ids);
        if (kind == 7) {
            auto query = data.query; query.back() = std::numeric_limits<float>::quiet_NaN();
            f.put(f.attention_query, query);
        }
        const std::size_t block_index = std::size_t(ids.back()) * 16 + 15;
        if (kind >= 8) {
            auto block = kind == 9 ? data.values[block_index] : data.keys[block_index];
            block.d = kind == 10 ? 0x7bff : 0x7c00;
            if (kind == 10) std::fill(std::begin(block.qs), std::end(block.qs), 0);
            f.put((kind == 9 ? f.attention_values : f.attention_keys) + block_index,
                  std::span<const qwen::Q4_0>(&block, 1));
        }
        f.clear_error(); f.attention(4096);
        const int rejected_flag = f.flag();
        require((rejected_flag & (kind == 10 ? 2 : 1)) != 0, "QSA attention invalid input not rejected");
        exact<float>(f.get(f.output_storage, 6146), before, "QSA attention failure published output");
        ++summary.attention_rejects;
        f.put(f.count_storage + 1, std::array<int, 2>{65, 16}); f.put(f.token_storage + 1, ids);
        f.put(f.attention_query, data.query);
        f.put(f.attention_keys + block_index, std::span<const qwen::Q4_0>(&data.keys[block_index], 1));
        f.put(f.attention_values + block_index, std::span<const qwen::Q4_0>(&data.values[block_index], 1));
        f.attention(4096);
        require(f.flag() == rejected_flag, "QSA attention changed sticky flag");
        exact<float>(f.get(f.output_storage, 6146), before, "QSA sticky attention published output");
        f.clear_error(); f.attention(4096); f.valid(); compare_attention(f, ref, summary.attention);
    }
    data.cache_bytes(f);
}

void summary_json(int device, const Summary& s) {
    // Exact finite coverage counters are part of protocol v1. A forgotten case
    // is an error rather than a smaller apparently successful protocol.
    require(s.cache_cases == 32 && s.prefix_checks == 648 && s.restores == 24 && s.reject_windows == 8 &&
            s.q4_cases == 8 && s.append_rejects == 6 && s.selection_checks == 200 &&
            s.selection_rejects == 3 && s.score_checks == 60 && s.score_rejects == 2 &&
            s.attention_checks == 26 && s.attention_rejects == 11, "QSA incomplete finite fixture coverage");
    std::cout << "{\"kind\":\"qsa_correctness\",\"device\":" << device
              << ",\"cache_cases\":" << s.cache_cases << ",\"prefix_checks\":" << s.prefix_checks
              << ",\"restore_cases\":" << s.restores << ",\"reject_windows\":" << s.reject_windows
              << ",\"q4_blocks\":" << s.q4_cases << ",\"append_rejects\":" << s.append_rejects
              << ",\"selection_checks\":" << s.selection_checks << ",\"count_checks\":" << s.selection_checks
              << ",\"selection_rejects\":" << s.selection_rejects
              << ",\"score_checks\":" << s.score_checks << ",\"score_rejects\":" << s.score_rejects
              << ",\"attention_checks\":" << s.attention_checks << ",\"attention_rejects\":" << s.attention_rejects
              << ",\"key_error\":"; gate_json(s.keys);
    std::cout << ",\"common_score_error\":"; gate_json(s.common_scores);
    std::cout << ",\"actual_score_error\":"; gate_json(s.actual_scores);
    std::cout << ",\"attention_error\":"; gate_json(s.attention);
    std::cout << ",\"q4_byte_parity\":true,\"q4_dequant_bit_parity\":true,\"tail_byte_parity\":true"
                 ",\"unused_id_canaries\":true,\"cache_readonly_byte_parity\":true,\"chunk_atomic\":true"
                 ",\"sticky_reuse\":true,\"future_poison_unread\":true,\"passed\":true}\n";
}

void append_measurements(Fixture& f, const Data& data) {
    for (int phase = 0; phase < 4; ++phase) for (int n : {1, 2, 3, 128}) {
        const int start = 4096 + phase;
        f.prepare(data, start);
        qwen::QsaIndexCpu cpu(data.config, data.parameters()); cpu.append(data.keys(0, start), start);
        const auto input = data.keys(start, n); cpu.append(input, n);
        f.put(f.raw_keys, input);
        check(hipMemcpyAsync(f.saved_tail, f.tail, tail_elements * sizeof(float), hipMemcpyDeviceToDevice, f.stream));
        f.valid();
        Gate errors;
        const auto immutable = f.get(f.cache, std::size_t(start / 4) * 128);
        const double ms = f.timed([&] {
            check(hipMemcpyAsync(f.tail, f.saved_tail, tail_elements * sizeof(float), hipMemcpyDeviceToDevice, f.stream));
            f.length = start; f.clear_error();
        }, [&] { f.append(start, n); }, [&] {
            f.length = start + n;
            compare_cache(f, cpu, errors);
            exact<float>(f.get(f.cache, immutable.size()), immutable, "QSA timed append immutable prefix");
        });
        std::cout << "{\"kind\":\"qsa_append\",\"device\":" << f.device << ",\"N\":" << n
                  << ",\"base\":" << start << ",\"phase\":" << phase << ",\"visible\":" << start + n
                  << ",\"repeats\":20,\"launches\":" << (phase + n >= 4 ? 4 : 3) << ",\"resident_ms\":" << ms
                  << ",\"scope\":\"resident_raw_K_q4_roundtrip_new_pool_norm_rope_publish; no_prefix_snapshots\""
                     ",\"timing_excludes\":\"projection,H2D,D2H,reset,tail_restore,error_clear,validation\""
                  << ",\"key_error\":"; gate_json(errors);
        std::cout << ",\"completed_blocks\":" << (start + n) / 4
                  << ",\"new_blocks\":" << (phase + n) / 4
                  << ",\"tail_byte_parity\":true,\"prefix_byte_parity\":true}\n";
    }
}
void score_measurements(Fixture& f, const Data& data, Timeline& timeline) {
    f.prepare(data, capacity);
    for (int visible : {4096, 32768, capacity}) {
        const auto ref = timeline.scores(data, visible); f.put(f.raw_query, data.query(visible));
        f.put(f.scores, std::vector<float>(capacity / 4 + 1, output_canary));
        Gate errors;
        const double ms = f.timed([&] { f.clear_error(); }, [&] { f.score(visible); }, [&] {
            const auto got = f.get(f.scores, capacity / 4 + 1);
            errors.compare(std::span<const float>(got).first(visible / 4), ref, "QSA timed score gate");
            require(std::all_of(got.begin() + visible / 4, got.end(),
                               [](float x) { return x == output_canary; }), "QSA timed score suffix canary");
        });
        std::cout << "{\"kind\":\"qsa_score\",\"device\":" << f.device << ",\"visible\":" << visible
                  << ",\"prepared\":131072,\"repeats\":20,\"launches\":2,\"resident_ms\":" << ms
                  << ",\"scope\":\"resident_raw_Q_norm_rope_and_causal_score; two_launches; actual_GPU_keys\""
                     ",\"timing_excludes\":\"projection,H2D,D2H,append,error_clear,validation\""
                  << ",\"completed_blocks\":" << visible / 4 << ",\"score_error\":"; gate_json(errors);
        std::cout << ",\"future_output_canary\":true}\n";
    }
}
void select_measurements(Fixture& f) {
    for (int visible : {2052, 4096, 32768, capacity}) {
        const auto values = score_pattern(visible, 1);
        const Selection ref(std::span<const float>(values).first(visible / 4), visible);
        f.put(f.scores, values); f.id_canaries();
        const double ms = f.timed([&] { f.clear_error(); }, [&] { f.select(visible); }, [&] { ref.compare(f); });
        std::cout << "{\"kind\":\"qsa_select\",\"device\":" << f.device << ",\"visible\":" << visible
                  << ",\"repeats\":20,\"launches\":19,\"resident_ms\":" << ms
                  << ",\"scope\":\"resident_selection_only; nineteen_launches; random_fixed_same_score_floats\""
                     ",\"timing_excludes\":\"projection,H2D,D2H,index_score,error_clear,validation\""
                  << ",\"token_count\":" << ref.counts[1] << ",\"block_count\":" << ref.counts[2]
                  << ",\"ids_exact\":true,\"counts_exact\":true,\"unused_capacity_byte_parity\":true}\n";
    }
}
void attention_measurements(Fixture& f, AttentionData& data) {
    for (int n : {1, 63, 64, 65, 2048, 2051}) {
        const auto ids = attention_ids(n); data.upload(f, ids);
        const auto ref = data.reference(ids, 4096); Gate errors;
        const double ms = f.timed([&] { f.clear_error(); }, [&] { f.attention(4096); }, [&] {
            compare_attention(f, ref, errors);
        });
        data.cache_bytes(f);
        std::cout << "{\"kind\":\"qsa_attention\",\"device\":" << f.device
                  << ",\"visible\":4096,\"capacity\":131072,\"selected\":" << n
                  << ",\"repeats\":20,\"launches\":6,\"resident_ms\":" << ms
                  << ",\"scope\":\"resident_selected_Q4_gather_FP16_and_attention_publish; Q24_KV2_D256\""
                     ",\"cache_source\":\"synthetic_canonical_Q4; all_unselected_scales_Inf\""
                     ",\"reference\":\"CPU_gathered_fp16_same_ID_order_and_repeats\""
                     ",\"timing_excludes\":\"projection,Hadamard,RoPE,gate,index_select,H2D,D2H,error_clear,validation\""
                  << ",\"token_count\":" << n << ",\"block_count\":" << std::min(512, n / 4)
                  << ",\"attention_error\":"; gate_json(errors);
        std::cout << ",\"cache_readonly_byte_parity\":true,\"output_canaries\":true}\n";
    }
}

void source_json(const Data& data, std::string_view model_path) {
    std::cout << "{\"kind\":\"qsa_source\",\"protocol\":1,\"revision\":"; json_string(CORE_REVISION);
    std::cout << ",\"dirty\":" << (CORE_DIRTY ? "true" : "false") << ",\"model_path\":"; json_string(model_path);
    std::cout << ",\"model_variant\":\"qwen38-keep1-Q4_0\",\"layer\":3"
                 ",\"scope\":\"fixture_only_loaded_indexer_weights_CPU_raw_FP32_projection_of_eight_synthetic_columns_repeated_in_cache_timeline; synthetic_attention_KV; not_prompt_or_full_QSA_block_or_inference\""
                 ",\"synthetic_x\":\"FP32_0.3*sin(i*0.137)+0.1*cos(i*0.071); eight_2560_element_columns\""
                 ",\"projection_types\":{\"q_proj\":\"BF16[2560,512]\",\"k_proj\":\"BF16[2560,128]\",\"q_norm\":\"F32[128]\",\"k_norm\":\"F32[128]\"}"
                 ",\"metadata_types\":{\"rope.dimension_count\":\"UINT32\",\"rope.freq_base\":\"FLOAT32\",\"attention.layer_norm_rms_epsilon\":\"FLOAT32\",\"rope.dimension_sections\":\"ARRAY_INT32\"}"
                 ",\"config\":{\"capacity\":131072,\"index_D\":128,\"index_Q\":4,\"index_KV\":1,\"compress\":4,\"budget\":2048"
                 ",\"rotary_dim\":64,\"rope_base\":" << data.rope_base << ",\"rms_epsilon\":" << data.config.rms_epsilon
              << ",\"rope_scale\":1,\"sections\":[11,11,10,0],\"text_position_axes\":\"identical_split_half\""
                 ",\"frequencies\":\"FP32_1/pow(base,2*i/64); bit_identical_CPU_and_uploaded_GPU\""
                 ",\"pooled_cache_bytes_per_device\":16777216,\"Q4_cache_bytes_each_per_device\":37748736}"
                 ",\"gates\":{\"key\":{\"abs\":0.0002,\"rel\":0.0002},\"score\":{\"abs\":0.0002,\"rel\":0.0002},\"attention\":{\"abs\":0.0002,\"rel\":0.0002},\"bound\":\"abs+rel*abs(CPU_reference)\",\"max_bound_ratio_limit\":1}"
                 ",\"correctness_before_any_metrics\":true,\"validate_after_each_timed_repeat\":true"
                 ",\"event_protocol\":\"20_individual_completed_GPU_event_intervals; arithmetic_mean_ms; reset_restore_and_flag_clear_outside_events\""
                 ",\"rows\":{\"qsa_source\":1,\"qsa_correctness\":2,\"qsa_append\":32,\"qsa_score\":6,\"qsa_select\":8,\"qsa_attention\":12,\"qsa_complete\":1,\"total\":62}}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("usage: core-qsa target.gguf");
        qwen::Model model(argv[1]); const Data data(model);
        std::cout << std::setprecision(17);
        source_json(data, argv[1]);
        int count = 0; check(hipGetDeviceCount(&count));
        require(count == 2, "QSA requires exactly two gfx906 wave64 devices");
        Timeline timeline(data); AttentionData attention_data;
        std::array<std::unique_ptr<Fixture>, 2> fixtures;
        std::array<Summary, 2> summaries;
        // Complete the ENTIRE correctness protocol on BOTH devices before any
        // metric. No GPU workloads overlap; each chain completes before switching.
        for (int d = 0; d < 2; ++d) {
            fixtures[d] = std::make_unique<Fixture>(d); auto& f = *fixtures[d]; f.initialize(data);
            auto& s = summaries[d];
            q4_cases(f, data, s); cache_cases(f, data, s); append_rejections(f, data, s);
            selection_cases(f, s); score_cases(f, data, timeline, s);
            attention_cases(f, attention_data, s); attention_rejections(f, attention_data, s);
        }
        for (int d = 0; d < 2; ++d) summary_json(d, summaries[d]);
        for (int d = 0; d < 2; ++d) {
            check(hipSetDevice(d)); auto& f = *fixtures[d];
            append_measurements(f, data); score_measurements(f, data, timeline);
            select_measurements(f); attention_measurements(f, attention_data);
        }
        // Checked destruction precedes passed=true: cleanup errors cannot follow
        // a successful completion row (even when a second initialization failed).
        for (auto& f : fixtures) f.reset();
        std::cout << "{\"kind\":\"qsa_complete\",\"protocol\":1,\"rows\":" << protocol_rows
                  << ",\"devices\":2,\"passed\":true}\n";
        require(bool(std::cout), "QSA JSONL output failure");
    } catch (const std::exception& error) {
        std::cerr << "core-qsa: " << error.what() << '\n';
        return 1;
    }
}
