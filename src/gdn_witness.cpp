#include "dense.hpp"
#include "gdn.hpp"
#include "model.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <vector>

// Standalone diagnostic, not a selectable runtime normalization mode. Build ALL
// linked CPU modules with -ffp-contract=off and without fast-math. No HIP headers,
// device work, JSON parser, projection recomputation, or out_proj is involved.
#if defined(__FAST_MATH__)
#error "GDN witness requires no fast-math and -ffp-contract=off"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace {

static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
constexpr float l2_epsilon = 1.0e-6f;
constexpr double absolute_gate = 2.0e-4;
constexpr double relative_gate = 2.0e-4;
constexpr std::string_view usage =
    "core-gdn-witness MODEL LAYER QKV Z ALPHA BETA OWN_FINAL ORACLE_FINAL "
    "ORACLE_CONV_SILU ORACLE_Q_L2 ORACLE_K_L2";

std::string json_string(std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 32 || c >= 127) {
            // Escape arbitrary path/error bytes too: output is always ASCII JSON.
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    result += '"';
    return result;
}

template<class T> T finite(T value) {
    if (!std::isfinite(value))
        throw std::runtime_error("nonfinite GDN witness arithmetic");
    return value;
}

void finite_values(std::span<const float> values, std::string_view name) {
    for (std::size_t i = 0; i < values.size(); ++i)
        if (!std::isfinite(values[i]))
            throw std::runtime_error(std::string(name) + ": nonfinite element " + std::to_string(i));
}

struct File {
    int fd;
    explicit File(const char* path) : fd(::open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK)) {
        if (fd < 0) throw std::system_error(errno, std::generic_category(), std::string("open ") + path);
    }
    ~File() { ::close(fd); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
};

std::vector<float> read_f32(const char* path, std::size_t count) {
    File file(path);
    const auto bytes_count = count * sizeof(float); // Fixed, small witness shapes.
    auto check_file = [&] {
        struct stat status{};
        if (::fstat(file.fd, &status) != 0)
            throw std::system_error(errno, std::generic_category(), std::string("stat ") + path);
        if (!S_ISREG(status.st_mode) || status.st_size < 0 ||
            static_cast<std::uint64_t>(status.st_size) != bytes_count)
            throw std::runtime_error(std::string(path) + ": expected regular little-endian F32 file of exactly " +
                                     std::to_string(bytes_count) + " bytes");
    };
    check_file();
    std::vector<std::byte> bytes(bytes_count);
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const auto received = ::pread(file.fd, bytes.data() + offset, bytes.size() - offset,
                                      static_cast<off_t>(offset));
        if (received < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), std::string("read ") + path);
        }
        if (received == 0) throw std::runtime_error(std::string(path) + ": truncated during read");
        offset += static_cast<std::size_t>(received);
    }
    check_file();
    std::vector<float> values(count);
    for (std::size_t i = 0; i < count; ++i)
        values[i] = qwen::tensor_element(qwen::TensorType::F32, bytes, i);
    finite_values(values, path);
    return values;
}

unsigned parse_layer(std::string_view text) {
    unsigned layer = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), layer);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || layer >= 48 ||
        (layer + 1) % 4 == 0)
        throw std::invalid_argument("LAYER must be a zero-based GDN layer in 0..47 (not 3,7,...,47)");
    return layer;
}

const qwen::TensorView& tensor_shape(const qwen::Model& model, const std::string& name,
                                    std::initializer_list<std::uint64_t> dimensions) {
    const auto& tensor = model.tensor(name);
    if (tensor.rank != dimensions.size() ||
        !std::equal(dimensions.begin(), dimensions.end(), tensor.dimensions.begin()))
        throw std::runtime_error("GDN actual tensor geometry: " + name);
    return tensor;
}

std::vector<float> parameter(const qwen::Model& model, const std::string& name,
                             std::initializer_list<std::uint64_t> dimensions) {
    const auto& tensor = tensor_shape(model, name, dimensions);
    if (tensor.type != qwen::TensorType::F32)
        throw std::runtime_error("GDN parameter must be F32: " + name);
    std::vector<std::byte> bytes(static_cast<std::size_t>(tensor.byte_size));
    model.read_tensor(name, bytes);
    std::vector<float> values(static_cast<std::size_t>(tensor.elements));
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = qwen::tensor_element(tensor.type, bytes, i);
    finite_values(values, name);
    return values;
}

struct Parameters {
    qwen::GdnConfig config;
    std::vector<float> conv, a, dt, norm;
    explicit Parameters(const qwen::Model& model, unsigned layer) {
        const auto& architecture = model.metadata_value("general.architecture");
        const auto& layers = model.metadata_value("qwen4exp.block_count");
        const auto& epsilon = model.metadata_value("qwen4exp.attention.layer_norm_rms_epsilon");
        if (architecture.type != qwen::MetadataType::STRING ||
            architecture.get<std::string>() != "qwen4exp" ||
            layers.type != qwen::MetadataType::UINT32 || layers.get<std::uint32_t>() != 48 ||
            epsilon.type != qwen::MetadataType::FLOAT32 || epsilon.get<float>() != l2_epsilon)
            throw std::runtime_error("witness requires qwen4exp target, 48 layers, FLOAT32 RMS epsilon 1e-6");
        config.rms_epsilon = epsilon.get<float>();
        const auto prefix = "blk." + std::to_string(layer) + ".";
        conv = parameter(model, prefix + "ssm_conv1d.weight", {4, 10240});
        a = parameter(model, prefix + "ssm_a", {48});
        dt = parameter(model, prefix + "ssm_dt.bias", {48});
        norm = parameter(model, prefix + "ssm_norm.weight", {128});
        // Derive the recurrence geometry from checked, actual loaded tensors.
        config.conv_width = static_cast<std::size_t>(model.tensor(prefix + "ssm_conv1d.weight").dimensions[0]);
        config.value_heads = a.size();
        config.value_head_dim = norm.size();
        config.key_head_dim = norm.size();
        const auto features = conv.size() / config.conv_width;
        config.key_heads = (features - config.value_elements()) / (2 * config.key_head_dim);
        if (config.key_heads != 16 || config.qkv_elements() != features ||
            config.key_elements() != 2048 || config.value_elements() != 6144 ||
            std::any_of(a.begin(), a.end(), [](float x) { return x > 0; }))
            throw std::runtime_error("unsupported GDN geometry or ssm_a is not -exp(A_log)");
        (void)tensor_shape(model, prefix + "attn_qkv.weight", {2560, features});
        (void)tensor_shape(model, prefix + "attn_gate.weight", {2560, config.value_elements()});
        for (const char* suffix : {"ssm_alpha.weight", "ssm_beta.weight"}) {
            const auto name = prefix + suffix;
            const auto& tensor = tensor_shape(model, name, {2560, config.value_heads});
            if (tensor.type != qwen::TensorType::F32)
                throw std::runtime_error("GDN alpha/beta projection must be F32: " + tensor.name);
        }
    }
    qwen::GdnParameters view() const { return {conv, dt, a, norm}; }
};

struct Metrics {
    std::size_t count = 0, exceed_count = 0;
    double max_abs = 0, rms = 0, max_bound_ratio = 0;
};

template<class Actual, class Reference>
Metrics compare(const Actual& actual, const Reference& reference) {
    if (actual.size() != reference.size() || actual.empty())
        throw std::runtime_error("witness comparison size mismatch");
    Metrics result;
    result.count = actual.size();
    double squared = 0;
    for (std::size_t i = 0; i < result.count; ++i) {
        const double expected = finite(double(reference[i]));
        const double error = finite(std::abs(finite(double(actual[i])) - expected));
        const double bound = absolute_gate + relative_gate * std::abs(expected);
        result.max_abs = std::max(result.max_abs, error);
        result.max_bound_ratio = std::max(result.max_bound_ratio, error / bound);
        result.exceed_count += error > bound;
        squared = finite(squared + error * error);
    }
    result.rms = std::sqrt(squared / double(result.count));
    return result;
}

void print_metrics(std::ostream& out, const Metrics& metrics) {
    out << "{\"count\":" << metrics.count << ",\"all_finite\":true,\"max_abs\":" << metrics.max_abs
        << ",\"rms\":" << metrics.rms << ",\"max_bound_ratio\":" << metrics.max_bound_ratio
        << ",\"exceed_count\":" << metrics.exceed_count
        << ",\"within_frozen_gate\":" << (metrics.exceed_count == 0) << '}';
}

template<class T> T sigmoid(T x) {
    const T e = std::exp(-std::abs(x));
    return x >= 0 ? T(1) / (T(1) + e) : e / (T(1) + e);
}

std::vector<float> first_conv(const Parameters& parameters, std::span<const float> qkv) {
    std::vector<float> result(qkv.size());
    const auto width = parameters.config.conv_width;
    for (std::size_t i = 0; i < qkv.size(); ++i) {
        const float x = finite(qkv[i] * parameters.conv[i * width + width - 1]);
        result[i] = finite(x * sigmoid(x));
    }
    return result;
}

struct Head {
    float sum_f32 = 0, raw_min = 0, raw_max = 0, additive_inv = 0, floor_inv = 0;
    double sum_f64 = 0, oracle_dot_raw = 0, oracle_squared = 0;
    double additive_inv_f64 = 0, floor_inv_f64 = 0;
};

struct Normalized {
    std::vector<float> additive, floor;
    std::vector<double> additive_f64, floor_f64;
    std::vector<Head> heads;
};

Normalized normalize(std::span<const float> raw, std::span<const float> oracle,
                     const qwen::GdnConfig& config) {
    if (raw.size() != config.key_elements() || oracle.size() != raw.size())
        throw std::runtime_error("Q/K normalization shape mismatch");
    Normalized result;
    result.additive.resize(raw.size()); result.floor.resize(raw.size());
    result.additive_f64.resize(raw.size()); result.floor_f64.resize(raw.size());
    result.heads.resize(config.key_heads);
    for (std::size_t h = 0; h < config.key_heads; ++h) {
        auto& head = result.heads[h];
        head.raw_min = head.raw_max = raw[h * config.key_head_dim];
        for (std::size_t k = 0; k < config.key_head_dim; ++k) {
            const auto i = h * config.key_head_dim + k;
            const float x = raw[i];
            head.sum_f32 = finite(head.sum_f32 + x * x);
            head.sum_f64 += double(x) * double(x);
            head.oracle_dot_raw += double(oracle[i]) * double(x);
            head.oracle_squared += double(oracle[i]) * double(oracle[i]);
            head.raw_min = std::min(head.raw_min, x);
            head.raw_max = std::max(head.raw_max, x);
        }
        // Ascending-k FP32 multiply/add, no reassociation or fused contractions.
        // Q remains UNSCALED here; 1/sqrt(Dk) is applied only by recurrence below.
        head.additive_inv = 1.0f / std::sqrt(finite(head.sum_f32 + l2_epsilon));
        head.floor_inv = 1.0f / std::sqrt(std::max(head.sum_f32, l2_epsilon * l2_epsilon));
        const double eps = double(l2_epsilon);
        head.additive_inv_f64 = 1.0 / std::sqrt(head.sum_f64 + eps);
        head.floor_inv_f64 = 1.0 / std::sqrt(std::max(head.sum_f64, eps * eps));
        for (std::size_t k = 0; k < config.key_head_dim; ++k) {
            const auto i = h * config.key_head_dim + k;
            result.additive[i] = finite(raw[i] * head.additive_inv);
            result.floor[i] = finite(raw[i] * head.floor_inv);
            result.additive_f64[i] = finite(double(raw[i]) * head.additive_inv_f64);
            result.floor_f64[i] = finite(double(raw[i]) * head.floor_inv_f64);
        }
    }
    return result;
}

void print_heads(std::ostream& out, const Normalized& normalized) {
    out << '[';
    for (std::size_t h = 0; h < normalized.heads.size(); ++h) {
        if (h) out << ',';
        const auto& head = normalized.heads[h];
        out << "{\"head\":" << h << ",\"sum_of_squares_f32\":" << head.sum_f32
            << ",\"sum_of_squares_f64\":" << head.sum_f64
            << ",\"raw_min\":" << head.raw_min << ",\"raw_max\":" << head.raw_max
            << ",\"additive_inverse_norm_f32\":" << head.additive_inv
            << ",\"floor_inverse_norm_f32\":" << head.floor_inv
            << ",\"additive_over_floor_scale_f32\":" << head.additive_inv / head.floor_inv
            << ",\"additive_over_floor_scale_f64\":" << head.additive_inv_f64 / head.floor_inv_f64
            << ",\"oracle_norm_f64\":" << std::sqrt(head.oracle_squared)
            << ",\"oracle_inverse_norm_fit_f64\":";
        if (head.sum_f64 == 0) {
            out << "null,\"oracle_over_floor_scale_fit_f64\":null,\"oracle_over_additive_scale_fit_f64\":null";
        } else {
            const double fit = head.oracle_dot_raw / head.sum_f64;
            out << finite(fit) << ",\"oracle_over_floor_scale_fit_f64\":" << finite(fit / head.floor_inv_f64)
                << ",\"oracle_over_additive_scale_fit_f64\":" << finite(fit / head.additive_inv_f64);
        }
        out << '}';
    }
    out << ']';
}

void print_sum_range(std::ostream& out, const Normalized& normalized) {
    double min_f32 = normalized.heads.front().sum_f32, max_f32 = min_f32;
    double min_f64 = normalized.heads.front().sum_f64, max_f64 = min_f64;
    for (const auto& head : normalized.heads) {
        min_f32 = std::min(min_f32, double(head.sum_f32));
        max_f32 = std::max(max_f32, double(head.sum_f32));
        min_f64 = std::min(min_f64, head.sum_f64);
        max_f64 = std::max(max_f64, head.sum_f64);
    }
    out << "{\"f32_min\":" << min_f32 << ",\"f32_max\":" << max_f32
        << ",\"f64_min\":" << min_f64 << ",\"f64_max\":" << max_f64 << '}';
}

// Independent ZERO-state rank-one formula, not a second stateful GDN runtime.
// S[k,v] = (sigmoid(beta) * k[k]) * V[v]; o[v] = sum_k(q[k]/sqrt(Dk) * S[k,v]).
// Explicit outer-product read (not an optimized q.k shortcut), then RMS(Dv),
// direct gamma, sigmoid(Z). Alpha/decay cannot affect an initially zero S.
// FP32 is an ascending-k/v CPU diagnostic, not emulation of HIP reduction/FMA.
// FP64 starts from the SAME captured FP32 conv/V/Z/beta, with double arithmetic.
template<class T, class Query, class Key>
std::vector<T> first_output(const Query& query, const Key& key, std::span<const float> conv,
                            std::span<const float> z, std::span<const float> beta,
                            const Parameters& parameters) {
    const auto& c = parameters.config;
    std::vector<T> result(c.value_elements());
    const T scale = T(1) / std::sqrt(T(c.key_head_dim));
    for (std::size_t h = 0; h < c.value_heads; ++h) {
        const auto kh = h % c.key_heads; // GGUF tiled V heads, not raw HF grouped order.
        const T b = sigmoid(T(beta[h]));
        T squared = 0;
        for (std::size_t v = 0; v < c.value_head_dim; ++v) {
            const auto i = h * c.value_head_dim + v;
            const T value = T(conv[2 * c.key_elements() + i]);
            T read = 0;
            for (std::size_t k = 0; k < c.key_head_dim; ++k) {
                const auto j = kh * c.key_head_dim + k;
                const T state = finite(finite(b * T(key[j])) * value);
                const T q = finite(T(query[j]) * scale);
                read = finite(read + state * q);
            }
            result[i] = read;
            squared = finite(squared + read * read);
        }
        const T inverse_rms = T(1) / std::sqrt(finite(squared / T(c.value_head_dim) + T(c.rms_epsilon)));
        for (std::size_t v = 0; v < c.value_head_dim; ++v) {
            const auto i = h * c.value_head_dim + v;
            result[i] = finite(finite(finite(result[i] * inverse_rms) * T(parameters.norm[v])) * sigmoid(T(z[i])));
        }
    }
    return result;
}

void print_delta_alignment(std::ostream& out, std::span<const float> own, std::span<const float> oracle,
                           std::span<const float> additive, std::span<const float> floor) {
    std::vector<double> observed(own.size()), predicted(own.size());
    double observed_squared = 0, predicted_squared = 0, dot = 0, residual_squared = 0;
    for (std::size_t i = 0; i < own.size(); ++i) {
        observed[i] = double(oracle[i]) - double(own[i]);
        predicted[i] = double(floor[i]) - double(additive[i]);
        observed_squared += observed[i] * observed[i];
        predicted_squared += predicted[i] * predicted[i];
        dot += observed[i] * predicted[i];
        const double residual = observed[i] - predicted[i];
        residual_squared += residual * residual;
    }
    out << "{\"observed\":\"ORACLE_FINAL - OWN_FINAL\",\"predicted\":\"CPU_floor - CPU_additive; same oracle conv, own Z/beta\""
        << ",\"observed_rms\":" << std::sqrt(observed_squared / double(own.size()))
        << ",\"predicted_rms\":" << std::sqrt(predicted_squared / double(own.size()))
        << ",\"residual_rms\":" << std::sqrt(residual_squared / double(own.size()))
        << ",\"cosine\":";
    if (observed_squared == 0 || predicted_squared == 0) out << "null";
    else out << finite(dot / (std::sqrt(observed_squared) * std::sqrt(predicted_squared)));
    out << ",\"predicted_over_observed_l2\":";
    if (observed_squared == 0) out << "null,\"relative_residual_l2\":null";
    else out << finite(std::sqrt(predicted_squared / observed_squared))
             << ",\"relative_residual_l2\":" << finite(std::sqrt(residual_squared / observed_squared));
    out << ",\"predicted_vs_observed\":";
    print_metrics(out, compare(predicted, observed));
    out << '}';
}

void print_gates(std::ostream& out) {
    out << "{\"abs\":" << absolute_gate << ",\"rel\":" << relative_gate
        << ",\"bound\":\"abs + rel * abs(reference)\",\"max_bound_ratio_limit\":1"
           ",\"primary_reference\":\"qualified GdnCpu.step; OWN raw projections, first zero-state token\""
           ",\"diagnostic_comparisons_affect_passed\":false,\"parent_session_gate_waived\":false}";
}

int witness(int argc, char** argv) {
    if (argc != 12) throw std::invalid_argument(std::string(usage));
    const unsigned layer = parse_layer(argv[2]);
    const qwen::Model model(argv[1]); // Validates all inventory dimensions, offsets, sizes.
    const Parameters parameters(model, layer); // Reads only the actual GDN parameters.
    const auto& c = parameters.config;
    constexpr std::array<std::string_view, 9> names{
        "qkv", "z", "alpha", "beta", "own_final", "oracle_final", "oracle_conv_silu", "oracle_q_l2", "oracle_k_l2"};
    const std::array<std::size_t, 9> counts{
        c.qkv_elements(), c.value_elements(), c.value_heads, c.value_heads, c.value_elements(),
        c.value_elements(), c.qkv_elements(), c.key_elements(), c.key_elements()};
    std::array<std::vector<float>, 9> data;
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = read_f32(argv[i + 3], counts[i]);
    // No numerical calculation until ALL nine binary inputs and parameters pass.
    const auto& [qkv, z, alpha, beta, own, oracle, conv, oracle_q, oracle_k] = data;
    qwen::GdnCpu cpu(c, parameters.view());
    if (cpu.state().consumed_tokens() != 0 ||
        std::any_of(cpu.state().recurrent().begin(), cpu.state().recurrent().end(), [](float x) { return x != 0; }) ||
        std::any_of(cpu.state().conv_history().begin(), cpu.state().conv_history().end(), [](float x) { return x != 0; }))
        throw std::runtime_error("GdnCpu did not start with zero state/history");
    std::vector<float> hf(c.value_elements());
    cpu.step({qkv, z, alpha, beta}, hf);
    if (cpu.state().consumed_tokens() != 1) throw std::runtime_error("wrong GdnCpu consumed token count");
    finite_values(cpu.state().recurrent(), "GdnCpu recurrent");
    finite_values(cpu.state().conv_history(), "GdnCpu conv history");
    const Metrics primary = compare(own, hf);
    const auto own_conv = first_conv(parameters, qkv);
    const auto q = normalize(std::span(conv).first(c.key_elements()), oracle_q, c);
    const auto k = normalize(std::span(conv).subspan(c.key_elements(), c.key_elements()), oracle_k, c);
    const auto captured_output = first_output<float>(oracle_q, oracle_k, conv, z, beta, parameters);
    const auto floor_output = first_output<float>(q.floor, k.floor, conv, z, beta, parameters);
    const auto additive_output = first_output<float>(q.additive, k.additive, conv, z, beta, parameters);
    const auto captured_f64 = first_output<double>(oracle_q, oracle_k, conv, z, beta, parameters);
    const auto floor_f64 = first_output<double>(q.floor_f64, k.floor_f64, conv, z, beta, parameters);
    const auto additive_f64 = first_output<double>(q.additive_f64, k.additive_f64, conv, z, beta, parameters);

    // Assemble atomically: an invalid/overflowing diagnostic cannot emit partial
    // JSON or a successful primary result without its completed evidence.
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::boolalpha << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "{\"kind\":\"gdn_first_token_cpu_witness\",\"protocol\":1,\"completed\":true,\"source\":{\"model\":" << json_string(argv[1])
        << ",\"layer\":" << layer << ",\"inputs\":{";
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i) out << ',';
        out << json_string(names[i]) << ":{\"path\":" << json_string(argv[i + 3]) << ",\"count\":" << counts[i]
            << ",\"bytes\":" << counts[i] * sizeof(float) << ",\"dtype\":\"little_endian_F32\",\"regular_exact_finite\":true}";
    }
    out << "},\"hf\":{\"revision\":\"a005fc82babfe8871d87746decad2dbee100a125\","
           "\"operation\":\"Transformers l2norm / torch_recurrent_gated_delta_rule\","
           "\"formula\":\"x * rsqrt(sum(x*x) + 1e-6)\",\"qualified_cpu\":\"src/gdn.cpp GdnCpu::step\"},"
           "\"production\":{\"revision\":\"dcd685463d597d31f5ca759d32c94592a2740fa4\","
           "\"operation\":\"ggml L2 norm\",\"formula\":\"x * rsqrt(max(sum(x*x), eps*eps))\","
           "\"provenance\":\"caller-confirmed pinned source; numeric agreement diagnosed, not presumed\"},"
           "\"required_cpu_build\":\"C++20, -ffp-contract=off, no fast-math\","
           "\"assumptions\":[\"All traces are the same first CONSUMED token with zero recurrent and conv history; no state trace was supplied\","
           "\"OWN QKV/Z are caller-confirmed exact to the pinned oracle; oracle raw projections are not supplied here\","
           "\"OWN raw alpha/beta are used unchanged; caller bounds oracle projection deltas by 4.77e-7, not measured here\","
           "\"Alpha/decay has no mathematical contribution at zero state; beta is sigmoid(raw beta)\","
           "\"Oracle Q/K L2 traces are unscaled, contiguous [16][128]; query scale applies inside recurrence\","
           "\"All V-side traces and weights use GGUF tiled order h%16; final traces are before out_proj\"]},\"gates\":";
    print_gates(out);
    out << ",\"actualgeometry\":{\"validated\":true,\"architecture\":\"qwen4exp\",\"layers\":48,\"hidden\":2560"
        << ",\"key_heads\":" << c.key_heads << ",\"value_heads\":" << c.value_heads
        << ",\"key_head_dim\":" << c.key_head_dim << ",\"value_head_dim\":" << c.value_head_dim
        << ",\"conv_width\":" << c.conv_width << ",\"rms_epsilon_metadata_f32\":" << c.rms_epsilon
        << ",\"fixed_qk_epsilon_f32\":" << l2_epsilon
        << ",\"conv_weights\":\"F32[4,10240], axis0 fastest; oldest tap first\",\"ssm_a\":\"F32[48], -exp(A_log)\""
           ",\"dt_bias\":\"F32[48]\",\"norm\":\"F32[128], direct gamma shared by heads\""
           ",\"raw_projections_checked\":true,\"initial_consumed_tokens\":0,\"final_consumed_tokens\":1"
           ",\"zero_initial_recurrent_and_conv_history\":true},\"own_vs_HF\":";
    print_metrics(out, primary);
    out << ",\"floor_l2\":{\"reference\":\"captured ORACLE_Q_L2 / ORACLE_K_L2; unscaled\",\"q\":";
    print_metrics(out, compare(q.floor, oracle_q));
    out << ",\"k\":"; print_metrics(out, compare(k.floor, oracle_k));
    out << "},\"additive_l2\":{\"reference\":\"captured ORACLE_Q_L2 / ORACLE_K_L2; unscaled\",\"q\":";
    print_metrics(out, compare(q.additive, oracle_q));
    out << ",\"k\":"; print_metrics(out, compare(k.additive, oracle_k));
    out << "},\"firsttokenflooroutput\":{\"scope\":\"independent zero-state rank-one CPU formula using oracle conv/V and OWN Z/beta; diagnostic only\""
           ",\"float_order\":\"ascending k/v; S=(sigmoid(beta)*K)*V; q_scaled=Q/sqrt(128); RMS128, direct gamma, sigmoidZ\""
           ",\"oracle_comparison_gate_reference\":\"independent CPU formula; ORACLE_FINAL is actual\""
           ",\"captured_oracle_l2_vs_oracle_final\":";
    print_metrics(out, compare(oracle, captured_output));
    out << ",\"floor_l2_vs_oracle_final\":"; print_metrics(out, compare(oracle, floor_output));
    out << ",\"additive_l2_vs_oracle_final\":"; print_metrics(out, compare(oracle, additive_output));
    out << ",\"floor_vs_additive\":"; print_metrics(out, compare(floor_output, additive_output));
    out << ",\"f64\":{\"scope\":\"independent double normalization and formula from captured FP32 conv/V/Z/beta; double(float epsilon); no projection/conv/GPU emulation\""
           ",\"captured_oracle_l2_vs_oracle_final\":";
    print_metrics(out, compare(oracle, captured_f64));
    out << ",\"floor_l2_vs_oracle_final\":"; print_metrics(out, compare(oracle, floor_f64));
    out << ",\"additive_l2_vs_oracle_final\":"; print_metrics(out, compare(oracle, additive_f64));
    out << ",\"f32_vs_f64_floor_output\":"; print_metrics(out, compare(floor_output, floor_f64));
    out << ",\"f32_vs_f64_additive_output\":"; print_metrics(out, compare(additive_output, additive_f64));
    out << "}},\"hypothesis_metrics\":{\"own_first_conv_vs_oracle_conv_silu\":";
    print_metrics(out, compare(own_conv, conv));
    out << ",\"own_final_vs_oracle_final\":"; print_metrics(out, compare(own, oracle));
    out << ",\"HF_cpu_vs_oracle_final\":"; print_metrics(out, compare(hf, oracle));
    out << ",\"oracle_conv_additive_formula_vs_HF_cpu\":"; print_metrics(out, compare(additive_output, hf));
    out << ",\"epsilon_only_delta_alignment\":";
    print_delta_alignment(out, own, oracle, additive_output, floor_output);
    out << ",\"sum_of_squares_ranges\":{\"q\":"; print_sum_range(out, q);
    out << ",\"k\":"; print_sum_range(out, k);
    out << '}';
    out << ",\"normalization_heads\":{\"q\":"; print_heads(out, q);
    out << ",\"k\":"; print_heads(out, k);
    out << "}},\"all_finite\":true,\"passed_scope\":\"validated geometry and OWN_FINAL versus qualified HF CPU first-token output only\""
           ",\"parent_session_gate_waived\":false,\"passed\":" << (primary.exceed_count == 0) << '}';
    std::cout << out.str() << '\n';
    return primary.exceed_count == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return witness(argc, argv);
    } catch (const std::exception& error) {
        // Invalid input/geometry/arithmetic is distinct from a completed, failed
        // OWN-vs-HF comparison. Exactly one strict JSON object even on errors.
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "{\"kind\":\"gdn_first_token_cpu_witness\",\"protocol\":1,\"passed\":false,"
               "\"completed\":false,\"parent_session_gate_waived\":false,\"gates\":";
        print_gates(out);
        out << ",\"error\":" << json_string(error.what()) << ",\"usage\":" << json_string(usage) << '}';
        std::cout << out.str() << '\n';
        return 2;
    }
}
