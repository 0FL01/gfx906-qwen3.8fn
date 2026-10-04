// Standalone diagnostic only. Do not add to CMake/core or the production runtime.
// API/graph donor: mx dcd685463d597d31f5ca759d32c94592a2740fa4.
// JSON uses that donor's unmodified vendor/nlohmann/json.hpp (MIT).
// Model-free regression: mtp-teacher-oracle --self-test (sole argument).
#include "llama.h"
#include "llama-ext.h"
#include "llama-model.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "json.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#if !defined(CORE_ORACLE_LIBRARY_REVISION) || !defined(CORE_ORACLE_LIBRARY_DIR)
#error "Build with tools/build-mtp-oracle.sh and attested production libraries"
#endif
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Numerical gates require compilation without fast-math"
#endif

extern char ** environ;

namespace {
namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr std::string_view pin = "dcd685463d597d31f5ca759d32c94592a2740fa4";
constexpr size_t rows = 32, hidden = 10240, embedding = 2560, vocab = 248320;
constexpr size_t json_limit = 64 * 1024, log_limit = 64 * 1024;
constexpr size_t intermediate_limit = 64 * 1024 * 1024;
constexpr uint64_t main_bytes = 75399121792ULL, sidecar_bytes = 2786568256ULL;
// EXACT common/common.h::LLM_FFN_EXPS_REGEX / llm_ffn_exps_cpu_override().
constexpr const char * cpu_expert_pattern = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(llama_token) == 4 && sizeof(llama_pos) == 4);
static_assert(std::endian::native == std::endian::little && GGML_MAX_DIMS == 4);

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

class Fd {
    int value_ = -1;
public:
    explicit Fd(int value = -1) : value_(value) {
        if (value < 0) throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
    }
    ~Fd() { if (value_ >= 0) ::close(value_); }
    Fd(const Fd &) = delete;
    Fd & operator=(const Fd &) = delete;
    Fd(Fd && other) noexcept : value_(std::exchange(other.value_, -1)) {}
    int get() const { return value_; }
    void close_checked() {
        const int value = std::exchange(value_, -1);
        require(value >= 0 && ::close(value) == 0, "output close failed");
    }
};

void write_all(int fd, const void * source, size_t count) {
    const auto * p = static_cast<const char *>(source);
    while (count) {
        const ssize_t n = ::write(fd, p, count);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "artifact write failed");
        p += n;
        count -= static_cast<size_t>(n);
    }
}

void safe_name(std::string_view name) {
    require(!name.empty() && name != "." && name != ".." && name.find('/') == name.npos &&
            name.find('\\') == name.npos && name.find('\0') == name.npos, "artifact name is not a basename");
}

uint64_t file_size_checked(int fd, const std::string & label) {
    struct stat st{};
    require(::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) && st.st_size >= 0, label + ": not a regular file");
    return static_cast<uint64_t>(st.st_size);
}

std::vector<char> read_file(int dir, const char * name, size_t minimum, size_t maximum) {
    safe_name(name);
    Fd fd(::openat(dir, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    const uint64_t size = file_size_checked(fd.get(), name);
    require(size >= minimum && size <= maximum, std::string(name) + ": wrong file size");
    std::vector<char> bytes(static_cast<size_t>(size));
    size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t n = ::read(fd.get(), bytes.data() + done, bytes.size() - done);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, std::string(name) + ": truncated input");
        done += static_cast<size_t>(n);
    }
    char extra{};
    ssize_t n;
    do { n = ::read(fd.get(), &extra, 1); } while (n < 0 && errno == EINTR);
    require(n == 0 && file_size_checked(fd.get(), name) == size, std::string(name) + ": input size changed");
    return bytes;
}

template<class T> std::vector<T> read_array(int dir, const char * name, size_t count) {
    const auto bytes = read_file(dir, name, count * sizeof(T), count * sizeof(T));
    std::vector<T> result(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}

Json parse_json(const std::vector<char> & bytes) {
    // Existing library parser, with duplicate-key/depth rejection, not a hand-written parser.
    std::vector<std::set<std::string>> keys;
    auto callback = [&keys](int depth, Json::parse_event_t event, Json & parsed) {
        require(depth <= 16, "capture.json nesting exceeds 16");
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        if (event == Json::parse_event_t::key) {
            require(!keys.empty() && keys.back().insert(parsed.get<std::string>()).second,
                    "capture.json contains a duplicate key");
        }
        if (event == Json::parse_event_t::object_end) keys.pop_back();
        return true;
    };
    return Json::parse(bytes.begin(), bytes.end(), callback, true, false);
}

bool integer_equal(const Json & value, uint64_t expected) {
    return (value.is_number_unsigned() && value.get<uint64_t>() == expected) ||
           (value.is_number_integer() && value.get<int64_t>() >= 0 &&
            static_cast<uint64_t>(value.get<int64_t>()) == expected);
}

void exact_keys(const Json & value, std::initializer_list<const char *> names, const char * label) {
    require(value.is_object() && value.size() == names.size(), std::string(label) + ": wrong object keys");
    for (const char * name : names) require(value.contains(name), std::string(label) + ": missing " + name);
}

void file_descriptor(const Json & files, const char * role, const char * name,
                     const char * dtype, std::initializer_list<size_t> shape) {
    const auto & d = files.at(role);
    exact_keys(d, {"name", "dtype", "shape", "bytes"}, role);
    require(d.at("name") == name && d.at("dtype") == dtype, std::string(role) + ": name/dtype mismatch");
    require(d.at("shape").is_array() && d.at("shape").size() == shape.size(), std::string(role) + ": bad shape");
    size_t count = 1, i = 0;
    for (size_t dimension : shape) {
        require(integer_equal(d.at("shape").at(i++), dimension), std::string(role) + ": bad dimension");
        count *= dimension;
    }
    require(integer_equal(d.at("bytes"), count * 4), std::string(role) + ": bad byte count");
}

struct Options {
    fs::path main, sidecar, capture, output;
    int width = 0, capacity = 64, slots = 112, inserts = 2;
    bool intermediates = false, dense_f32_control = false, dense_short_canonical = false;
};

int number(std::string_view text) {
    int value = -1;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    require(ec == std::errc{} && end == text.data() + text.size(), "option requires an integer");
    return value;
}

Options parse_options(int argc, char ** argv) {
    require(argc >= 6, "usage: mtp-teacher-oracle MAIN SIDECAR CAPTURE_DIR NEW_OUTPUT_DIR "
            "--teacher-width 1|2|3 [--capacity 64 --cache-slots 112 --cache-inserts 2 "
            "--capture-intermediates --dense-f32-control [--dense-short-canonical]]; or mtp-teacher-oracle --self-test");
    Options o{argv[1], argv[2], argv[3], argv[4]};
    std::set<std::string> seen;
    for (int i = 5; i < argc; ++i) {
        const std::string arg = argv[i];
        require(seen.insert(arg).second, "duplicate option: " + arg);
        if (arg == "--capture-intermediates") { o.intermediates = true; continue; }
        if (arg == "--dense-f32-control") { o.dense_f32_control = true; continue; }
        if (arg == "--dense-short-canonical") { o.dense_short_canonical = true; continue; }
        require(i + 1 < argc, "missing value: " + arg);
        const int value = number(argv[++i]);
        if (arg == "--teacher-width") o.width = value;
        else if (arg == "--capacity") o.capacity = value;
        else if (arg == "--cache-slots") o.slots = value;
        else if (arg == "--cache-inserts") o.inserts = value;
        else throw std::runtime_error("unknown option: " + arg);
    }
    require(o.width >= 1 && o.width <= 3, "--teacher-width must be explicitly 1, 2 or 3");
    require(o.capacity == 64 && o.slots == 112 && o.inserts >= 1 && o.inserts <= o.slots,
            "fixture requires capacity 64, cache-slots 112 and cache-inserts in 1..112 (default 2)");
    require(!o.dense_short_canonical || o.dense_f32_control, "short canonical arithmetic requires explicit dense control");
    require(pin == CORE_ORACLE_LIBRARY_REVISION, "compiled library attestation does not match donor pin");
    require(!o.dense_f32_control || std::fegetround() == FE_TONEAREST,
            "dense FP32 control requires round-to-nearest arithmetic");
    return o;
}

Json environment() {
    Json values = Json::object();
    size_t bytes = 0;
    for (char ** p = environ; *p; ++p) {
        const std::string_view entry(*p);
        if (!(entry.starts_with("LLAMA_") || entry.starts_with("GGML_") || entry.starts_with("HIP_") ||
              entry.starts_with("ROCR_") || entry.starts_with("HSA_") || entry.starts_with("CUDA_"))) continue;
        bytes += entry.size();
        require(bytes <= 16384 && values.size() < 128, "backend environment exceeds bounded metadata");
        const size_t equals = entry.find('=');
        require(equals != entry.npos, "malformed backend environment");
        values[std::string(entry.substr(0, equals))] = entry.substr(equals + 1);
    }
    for (const char * name : {"LLAMA_MTP_DRAFT_VOCAB_RANGES", "LLAMA_MTP_HEAD_PROFILE"}) {
        const char * value = std::getenv(name);
        values[name] = value ? Json(value) : Json(nullptr);
    }
    const char * ranges = std::getenv("LLAMA_MTP_DRAFT_VOCAB_RANGES");
    const char * profile = std::getenv("LLAMA_MTP_HEAD_PROFILE");
    require(!ranges || !*ranges, "restricted MTP draft vocabulary is forbidden");
    require(!profile || std::strcmp(profile, "0") == 0, "MTP head profile must be unset or exactly 0");
    return values;
}

struct Inputs {
    Json descriptor, env;
    std::vector<char> descriptor_bytes;
    std::vector<llama_token> tokens;
    std::vector<float> previous, own_d, own_logits;
};

Inputs preflight(Options & o) {
    Inputs in;
    in.env = environment();
    Fd root(::open(o.capture.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    in.descriptor_bytes = read_file(root.get(), "capture.json", 2, json_limit);
    in.descriptor = parse_json(in.descriptor_bytes);
    const auto & d = in.descriptor;
    exact_keys(d, {"schema", "source", "geometry", "positions", "zero_row0", "files"}, "capture.json");
    require(d.at("schema") == "gfx906-mtp-teacher-capture-v1", "unsupported capture schema");
    const auto & source = d.at("source");
    exact_keys(source, {"revision", "dirty"}, "source");
    require(source.at("revision").is_string() && source.at("dirty").is_boolean(), "source needs revision and actual dirty bool");
    const auto revision = source.at("revision").get<std::string>();
    require(revision.size() == 40 && std::all_of(revision.begin(), revision.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }), "input source revision must be 40 lowercase hex characters");
    const auto & g = d.at("geometry");
    exact_keys(g, {"rows", "hidden", "vocab"}, "geometry");
    require(integer_equal(g.at("rows"), rows) && integer_equal(g.at("hidden"), hidden) &&
            integer_equal(g.at("vocab"), vocab), "capture geometry must be 32/10240/248320");
    require(d.at("positions").is_array() && d.at("positions").size() == rows, "capture needs 32 positions");
    for (size_t p = 0; p < rows; ++p) require(integer_equal(d.at("positions").at(p), p), "positions must be exactly 0..31");
    require(d.at("zero_row0").is_boolean() && d.at("zero_row0").get<bool>(), "fresh previous-hidden row0 zero must be declared");
    const auto & f = d.at("files");
    exact_keys(f, {"tokens", "previous_hidden", "D", "logits"}, "files");
    file_descriptor(f, "tokens", "teacher_tokens.i32.bin", "i32le", {rows});
    file_descriptor(f, "previous_hidden", "teacher_previous_hidden.f32.bin", "f32le", {rows, hidden});
    file_descriptor(f, "D", "own_D.f32.bin", "f32le", {rows, hidden});
    file_descriptor(f, "logits", "own_logits.f32.bin", "f32le", {rows, vocab});
    in.tokens = read_array<llama_token>(root.get(), "teacher_tokens.i32.bin", rows);
    in.previous = read_array<float>(root.get(), "teacher_previous_hidden.f32.bin", rows * hidden);
    in.own_d = read_array<float>(root.get(), "own_D.f32.bin", rows * hidden);
    in.own_logits = read_array<float>(root.get(), "own_logits.f32.bin", rows * vocab);
    for (auto token : in.tokens) require(token >= 0 && static_cast<size_t>(token) < vocab, "teacher token outside [0,248320)");
    for (const auto * a : {&in.previous, &in.own_d, &in.own_logits}) {
        require(std::all_of(a->begin(), a->end(), [](float x) { return std::isfinite(x); }), "capture contains nonfinite F32");
    }
    require(std::all_of(in.previous.begin(), in.previous.begin() + hidden, [](float x) { return x == 0.0f; }),
            "previous-hidden row0 is not zero");
    // File-stat preflight only; no model payload is read before all capture checks pass.
    for (const auto & item : {std::pair{o.main, main_bytes}, std::pair{o.sidecar, sidecar_bytes}}) {
        Fd fd(::open(item.first.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
        require(file_size_checked(fd.get(), item.first.string()) == item.second, "model file size differs from original fixture");
    }
    o.main = fs::canonical(o.main); o.sidecar = fs::canonical(o.sidecar); o.capture = fs::canonical(o.capture);
    require(o.main != o.sidecar, "target and sidecar must be distinct files");
    const auto absolute = fs::absolute(o.output);
    safe_name(absolute.filename().string());
    o.output = fs::canonical(absolute.parent_path()) / absolute.filename();
    require(!fs::exists(fs::symlink_status(o.output)), "output directory must be fresh");
    return in;
}

struct Outputs {
    Fd root;
    explicit Outputs(const fs::path & path) : root(make_directory(path)) {}
    static int make_directory(const fs::path & path) {
        require(::mkdir(path.c_str(), 0700) == 0, "cannot create fresh output directory");
        return ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    }
    Fd open(const std::string & name) const {
        safe_name(name);
        return Fd(::openat(root.get(), name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600));
    }
    void bytes(const std::string & name, const void * data, size_t count) const {
        auto fd = open(name); write_all(fd.get(), data, count); fd.close_checked();
    }
    void json(const std::string & name, const Json & object) const {
        std::string text = object.dump(2);
        // Full width-1 witness metadata can exceed the pretty-print budget.
        // Retain every field and the same hard byte bound; only compact whitespace.
        if (text.size() >= 256 * 1024) text = object.dump();
        text += '\n';
        require(text.size() <= 256 * 1024, "JSON artifact exceeds 256KiB bound");
        bytes(name, text.data(), text.size());
    }
};

struct BoundedLog {
    Fd fd;
    std::mutex mutex;
    size_t written = 0;
    bool truncated = false;
    std::atomic_bool failed{false};
    explicit BoundedLog(const Outputs & out) : fd(out.open("donor.log")) {}
    static void callback(ggml_log_level, const char * text, void * user) noexcept {
        auto & self = *static_cast<BoundedLog *>(user);
        try {
            std::lock_guard lock(self.mutex);
            const size_t size = ::strnlen(text, log_limit + 1);
            const size_t count = std::min(size, log_limit - self.written);
            self.truncated |= count < size;
            write_all(self.fd.get(), text, count);
            self.written += count;
        } catch (...) { self.failed = true; }
    }
};

struct Metrics {
    uint64_t compared = 0, nonfinite = 0, violations = 0, bitdiff = 0;
    double maxabs = 0.0, maxratio = 0.0;
    Json first = nullptr, abs_coordinate = nullptr, ratio_coordinate = nullptr;
    void row(std::span<const float> own, std::span<const float> ref, size_t position, double atol) {
        require(own.size() == ref.size(), "comparison row size mismatch");
        for (size_t c = 0; c < ref.size(); ++c) {
            ++compared;
            bitdiff += std::bit_cast<uint32_t>(own[c]) != std::bit_cast<uint32_t>(ref[c]);
            const auto coord = [position, c] { return Json{{"position", position}, {"column", c}}; };
            if (!std::isfinite(own[c]) || !std::isfinite(ref[c])) {
                ++nonfinite; ++violations;
                if (first.is_null()) first = coord();
                continue;
            }
            const double error = std::abs(static_cast<double>(own[c]) - static_cast<double>(ref[c]));
            const double ratio = error / (atol + 0.002 * std::abs(static_cast<double>(ref[c])));
            if (error > maxabs) { maxabs = error; abs_coordinate = coord(); }
            if (ratio > maxratio) { maxratio = ratio; ratio_coordinate = coord(); }
            if (ratio > 1.0) { ++violations; if (first.is_null()) first = coord(); }
        }
    }
    Json json(double atol) const {
        return {{"compared", compared}, {"nonfinite_count", nonfinite}, {"violations", violations},
            {"maxabs", maxabs}, {"maxratio", maxratio}, {"first_coordinate", first},
            {"maxabs_coordinate", abs_coordinate}, {"maxratio_coordinate", ratio_coordinate},
            {"diagnostic_bitdiff", bitdiff}, {"atol", atol}, {"rtol", 0.002},
            {"reference", "donor"}, {"passed", compared > 0 && violations == 0}};
    }
};

// Exact pinned cb() names and full typed element bounds, not keyword/regex matching.
struct Site { const char * name; size_t columns; size_t expected_occurrences = 1; };
constexpr std::array<Site, 13> sites{{
    {"mtp_tok_embd-48", embedding}, {"mtp_hnorm-48", hidden}, {"mtp_enorm-48", hidden},
    {"mtp_concat-48", hidden * 2}, {"mtp_fused-48", hidden}, {"l_last-48", hidden},
    {"h_nextn", hidden}, {"result_output", vocab},
    {"attn_pregate-48", 6144}, {"attn_output-48", embedding}, {"ffn_out-48", embedding},
    {"hc_combine-48", hidden}, {"hc_mixed-48", embedding, 2}}};
// The final hc_combine is renamed l_last-48. hc_mixed-48 survives twice,
// in dependency order: occurrence 0 is attention, occurrence 1 is FFN.
// qwen4exp.cpp cb(cur,"result_norm",-1), then cb(SAME cur,"mtp_head_input",-1).
// ggml_set_name overwrites the former name. The scheduler sees mtp_head_input.
constexpr Site result_norm_site{"mtp_head_input", embedding};

// Independent mathematical dense reference, not an HF training oracle. The
// opt-in correction replaces only the original FLASH output before inverse V
// Hadamard/gating/projection. It does not quantize Q/probabilities or reproduce
// the fork's FP16 numerator / shifted-maximum implementation.
constexpr size_t dense_dim = 256, dense_heads = 24, dense_kv_heads = 2;
constexpr size_t dense_columns = dense_dim * dense_heads, dense_math_prefix_limit = 4096;
constexpr size_t dense_storage_limit = 4 * 1024 * 1024;
constexpr float dense_scale = 0.0625f;

size_t dense_add_size(size_t a, size_t b) {
    require(b <= std::numeric_limits<size_t>::max() - a, "dense storage addition overflow");
    return a + b;
}

size_t dense_multiply_size(size_t a, size_t b) {
    require(a == 0 || b <= std::numeric_limits<size_t>::max() / a, "dense storage multiplication overflow");
    return a * b;
}

float dense_finite(float value) {
    require(std::isfinite(value), "nonfinite dense FP32 control arithmetic/input");
    return value;
}

float dense_product(float a, float b) {
    const volatile float result = a * b;
    return dense_finite(result);
}

float dense_sum(float a, float b) {
    const volatile float result = a + b;
    return dense_finite(result);
}

// Input query layout is [query,head,dimension], K/V [position,kv_head,dimension].
// A separately bounded mathematical function permits CPU fixtures beyond 2051
// positions; the real teacher hook below is restricted to this 32-row timeline.
std::vector<float> dense_attention_f32(std::span<const float> query, std::span<const float> keys,
                                     std::span<const float> values, size_t first, size_t count, bool short_canonical = false) {
    require(!short_canonical || (first <= 64 && count <= 64 - first), "short canonical reference requires <=64 visible rows");
    require(count >= 1 && count <= 3 && first < dense_math_prefix_limit &&
            count <= dense_math_prefix_limit - first, "dense mathematical prefix/query bound exceeded");
    const size_t prefix = first + count;
    require(query.size() == count * dense_columns && keys.size() == prefix * dense_kv_heads * dense_dim &&
            values.size() == keys.size(), "dense mathematical input dimensions mismatch");
    for (auto input : {query, keys, values}) for (float value : input) dense_finite(value);
    std::vector<float> result(query.size()), scores(prefix);
    for (size_t row = 0; row < count; ++row) {
        const size_t visible = first + row + 1;
        for (size_t head = 0; head < dense_heads; ++head) {
            const size_t kv_head = head / (dense_heads / dense_kv_heads);
            const size_t query_base = (row * dense_heads + head) * dense_dim;
            float maximum = -std::numeric_limits<float>::infinity();
            for (size_t position = 0; position < visible; ++position) {
                const size_t base = (position * dense_kv_heads + kv_head) * dense_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < dense_dim; ++d) {
                    dot = dense_sum(dot, dense_product(query[query_base + d], keys[base + d]));
                }
                scores[position] = dense_product(dot, dense_scale);
                maximum = std::max(maximum, scores[position]);
            }
            float denominator = 0.0f;
            for (size_t position = 0; position < visible; ++position) {
                const volatile float shifted = scores[position] - maximum;
                scores[position] = short_canonical
                    ? dense_finite(static_cast<float>(std::exp(static_cast<double>(dense_finite(shifted)))))
                    : dense_finite(std::exp(dense_finite(shifted)));
                denominator = dense_sum(denominator, scores[position]);
            }
            require(denominator > 0.0f, "dense softmax denominator is empty");
            for (size_t d = 0; d < dense_dim; ++d) {
                float numerator = 0.0f;
                for (size_t position = 0; position < visible; ++position) {
                    const size_t base = (position * dense_kv_heads + kv_head) * dense_dim;
                    numerator = dense_sum(numerator, dense_product(scores[position], values[base + d]));
                }
                // Separate diagnostic of the existing <=64-key GPU arithmetic:
                // rounded reciprocal then rounded product, not direct division.
                const volatile float reciprocal = 1.0f / denominator;
                const volatile float normalized = short_canonical ? dense_product(numerator, reciprocal) : numerator / denominator;
                // One visible position is exactly its decoded V, including -0.
                result[query_base + d] = visible == 1 ? values[kv_head * dense_dim + d] : dense_finite(normalized);
            }
        }
    }
    return result;
}

struct DenseStorage { size_t bytes; uintptr_t address; };

DenseStorage dense_storage(const ggml_tensor * t, ggml_type type, const std::array<int64_t, 4> & shape) {
    require(t && t->type == type && t->data, "dense tensor type/data mismatch");
    const size_t unit = type == GGML_TYPE_Q4_0 ? 18 : type == GGML_TYPE_F16 ? 2 : 4;
    const size_t alignment = type == GGML_TYPE_F32 ? 4 : 2;
    require((type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_Q4_0) &&
            ggml_type_size(type) == unit && ggml_blck_size(type) == (type == GGML_TYPE_Q4_0 ? 32 : 1) &&
            t->nb[0] == unit && reinterpret_cast<uintptr_t>(t->data) % alignment == 0,
            "dense tensor canonical type/block layout/alignment mismatch");
    size_t span = unit;
    std::array<std::pair<size_t, size_t>, 4> axes{};
    for (size_t d = 0; d < shape.size(); ++d) {
        require(shape[d] > 0 && shape[d] <= 256 && t->ne[d] == shape[d] &&
                t->nb[d] >= unit && t->nb[d] % alignment == 0, "dense tensor dimensions/strides mismatch");
        const size_t block = type == GGML_TYPE_Q4_0 && d == 0 ? 32 : 1;
        require(static_cast<size_t>(shape[d]) % block == 0, "dense tensor quantized dimension mismatch");
        const size_t count = static_cast<size_t>(shape[d]) / block;
        axes[d] = {t->nb[d], count};
        span = dense_add_size(span, dense_multiply_size(count - 1, t->nb[d]));
    }
    // Accept actual permuted Q and either interleaved or capacity-strided KV
    // heads, but prove every typed element/block has a distinct bounded address.
    std::sort(axes.begin(), axes.end());
    size_t occupied = unit;
    for (const auto & [stride, count] : axes) if (count > 1) {
        require(stride >= occupied, "dense tensor has overlapping strided elements/blocks");
        occupied = dense_add_size(occupied, dense_multiply_size(count - 1, stride));
    }
    require(span <= dense_storage_limit && ggml_nbytes(t) == span, "dense tensor byte-span bound mismatch");
    auto * buffer = t->buffer ? t->buffer : (t->view_src ? t->view_src->buffer : nullptr);
    require(buffer != nullptr, "dense tensor has no backend allocation");
    const uintptr_t base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    const uintptr_t address = reinterpret_cast<uintptr_t>(t->data);
    const size_t allocation = ggml_backend_buffer_get_size(buffer);
    require(base != 0 && address >= base && address - base <= allocation &&
            span <= allocation - (address - base) && span <= std::numeric_limits<uintptr_t>::max() - address,
            "dense tensor escapes its backend allocation");
    return {span, address};
}

Json dense_tensor_metadata(const ggml_tensor * t, const DenseStorage & storage) {
    Json ne = Json::array(), nb = Json::array();
    for (size_t i = 0; i < 4; ++i) { ne.push_back(t->ne[i]); nb.push_back(t->nb[i]); }
    return {{"type_id", t->type}, {"ne", ne}, {"nb", nb}, {"bytes", storage.bytes}, {"view_offset", t->view_offs}};
}

float dense_q4_value(std::span<const char> row, size_t dimension) {
    require(row.size() == dense_dim / 32 * 18 && dimension < dense_dim, "dense Q4 row/dimension mismatch");
    const size_t block = dimension / 32 * 18, lane = dimension % 32;
    ggml_fp16_t bits{};
    std::memcpy(&bits, row.data() + block, sizeof(bits));
    const float scale = dense_finite(ggml_fp16_to_fp32(bits));
    const auto packed = static_cast<unsigned char>(row[block + 2 + lane % 16]);
    const int code = lane < 16 ? packed & 15 : packed >> 4;
    const ggml_fp16_t rounded = ggml_fp32_to_fp16(dense_product(scale, static_cast<float>(code - 8)));
    return dense_finite(ggml_fp16_to_fp32(rounded));
}

struct DenseF32Control {
    bool enabled;
    bool short_canonical = false;
    size_t first = 0, count = 0, observed = 0, calls = 0, corrected_rows = 0, corrected_values = 0;
    Json records = Json::array();
    explicit DenseF32Control(bool value, bool canonical = false) : enabled(value), short_canonical(canonical) {}
    void begin(size_t position, size_t n) {
        first = position; count = n; observed = 0;
        if (enabled) require(n >= 1 && n <= 3 && position == corrected_rows && position < rows && n <= rows - position,
                             "dense control requires the complete chronological 32-row teacher prefix");
    }
    std::array<DenseStorage, 5> validate(const ggml_tensor * t) const {
        require(t && t->op == GGML_OP_FLASH_ATTN_EXT && count >= 1 && count <= 3 &&
                first < rows && count <= rows - first,
                "dense control requires the original FLASH_ATTN_EXT operation");
        std::array<float, 3> parameters{};
        // Pinned public FLASH layout is also checked by attention_oracle.cpp:
        // op_params[0..2] = scale,max_bias,softcap; public getter checks precision.
        std::memcpy(parameters.data(), t->op_params, sizeof(parameters));
        require(parameters[0] == dense_scale && parameters[1] == 0.0f && parameters[2] == 0.0f &&
                ggml_flash_attn_ext_get_prec(t) == GGML_PREC_F32, "dense FLASH scale/bias/softcap/precision mismatch");
        for (size_t i = 4; i < GGML_MAX_SRC; ++i) require(!t->src[i], "dense FLASH has unsupported sink/extra source");
        for (size_t i = 0; i < 4; ++i) require(t->src[i] != nullptr, "dense FLASH requires original Q/K/V/causal mask");
        const int64_t n = static_cast<int64_t>(count), physical = 256;
        const int64_t mask_rows = t->src[3]->ne[1];
        require(mask_rows >= n && mask_rows <= physical, "dense causal mask padded query dimension mismatch");
        const std::array<DenseStorage, 5> storage{
            dense_storage(t->src[0], GGML_TYPE_F32, {256, n, 24, 1}),
            dense_storage(t->src[1], GGML_TYPE_Q4_0, {256, physical, 2, 1}),
            dense_storage(t->src[2], GGML_TYPE_Q4_0, {256, physical, 2, 1}),
            dense_storage(t->src[3], GGML_TYPE_F16, {physical, mask_rows, 1, 1}),
            dense_storage(t, GGML_TYPE_F32, {256, 24, n, 1})};
        require(ggml_is_contiguous(t) && storage[4].bytes == count * dense_columns * sizeof(float),
                "dense FLASH output is not contiguous [dimension,head,query,1] F32");
        for (size_t i = 0; i < 4; ++i) {
            require(!(storage[i].address < storage[4].address + storage[4].bytes &&
                      storage[4].address < storage[i].address + storage[i].bytes), "dense FLASH output aliases a raw input");
        }
        return storage;
    }
    void correct(ggml_tensor * t) {
        require(enabled && observed++ == 0, "dense control saw more than one FLASH operation per decode");
        require(std::fegetround() == FE_TONEAREST, "dense callback requires round-to-nearest arithmetic");
        const auto storage = validate(t); // All types/shapes/strides/allocations before any I/O.
        require(ggml_fp32_to_fp16(1.00048828125f) == 0x3c00 &&
                ggml_fp32_to_fp16(1.00146484375f) == 0x3c02 &&
                ggml_fp32_to_fp16(-1.00048828125f) == 0xbc00,
                "installed GGML half helper does not implement required FP16 RNE ties");
        std::vector<ggml_fp16_t> mask(256);
        for (size_t row = 0; row < count; ++row) {
            ggml_backend_tensor_get(t->src[3], mask.data(), row * t->src[3]->nb[1], mask.size() * sizeof(mask[0]));
            const size_t visible = first + row + 1;
            for (size_t position = 0; position < mask.size(); ++position) {
                const float value = ggml_fp16_to_fp32(mask[position]);
                require(position < visible ? std::isfinite(value) && value == 0.0f : std::isinf(value) && value < 0.0f,
                        "dense mask must select exactly the full absolute causal prefix; only zero/-inf allowed");
            }
        }
        std::vector<char> raw_query(storage[0].bytes);
        ggml_backend_tensor_get(t->src[0], raw_query.data(), 0, raw_query.size());
        std::vector<float> query(count * dense_columns), old(query.size());
        for (size_t row = 0; row < count; ++row) for (size_t head = 0; head < dense_heads; ++head) {
            for (size_t d = 0; d < dense_dim; ++d) {
                const size_t index = (row * dense_heads + head) * dense_dim + d;
                const size_t offset = row * t->src[0]->nb[1] + head * t->src[0]->nb[2] + d * sizeof(float);
                std::memcpy(&query[index], raw_query.data() + offset, sizeof(float));
                dense_finite(query[index]);
            }
        }
        ggml_backend_tensor_get(t, old.data(), 0, storage[4].bytes);
        for (float value : old) dense_finite(value); // Never conceal a nonfinite raw donor result.
        const size_t prefix = first + count;
        std::vector<float> keys(prefix * dense_kv_heads * dense_dim), values(keys.size());
        std::array<char, dense_dim / 32 * 18> raw_row{};
        for (size_t input = 1; input <= 2; ++input) {
            auto & decoded = input == 1 ? keys : values;
            const auto * source = t->src[input];
            // No payload read or scale decode of uninitialized masked future KV.
            for (size_t position = 0; position < prefix; ++position) for (size_t head = 0; head < dense_kv_heads; ++head) {
                ggml_backend_tensor_get(source, raw_row.data(), position * source->nb[1] + head * source->nb[2], raw_row.size());
                for (size_t d = 0; d < dense_dim; ++d) {
                    decoded[(position * dense_kv_heads + head) * dense_dim + d] = dense_q4_value(raw_row, d);
                }
            }
        }
        const auto corrected = dense_attention_f32(query, keys, values, first, count, short_canonical);
        double maxabs = 0.0;
        for (size_t i = 0; i < corrected.size(); ++i) {
            dense_finite(corrected[i]);
            maxabs = std::max(maxabs, std::abs(static_cast<double>(corrected[i]) - old[i]));
        }
        Json sources = Json::array();
        for (size_t i = 0; i < 4; ++i) sources.push_back(dense_tensor_metadata(t->src[i], storage[i]));
        records.push_back({{"first_position", first}, {"rows", count}, {"occurrence", 0},
            {"selection", "structured FLASH_ATTN_EXT, original typed src[0..3], MTP-context callback only"},
            {"target", dense_tensor_metadata(t, storage[4])}, {"sources", sources},
            {"visible_prefix_lengths", Json::array()}, {"scale", dense_scale}, {"max_bias", 0}, {"softcap", 0},
            {"old_new_maxabs_diagnostic", maxabs}, {"write_committed", false}, {"readback_verified", false}});
        for (size_t row = 0; row < count; ++row) records.back()["visible_prefix_lengths"].push_back(first + row + 1);
        // One synchronous publication after all checks; downstream graph range
        // uses the corrected FLASH output. Q/K/V/mask/model weights stay intact.
        ggml_backend_tensor_set(t, corrected.data(), 0, storage[4].bytes);
        ++calls; corrected_rows += count; corrected_values += corrected.size();
        records.back()["write_committed"] = true;
        std::vector<float> readback(corrected.size());
        ggml_backend_tensor_get(t, readback.data(), 0, storage[4].bytes);
        require(std::memcmp(readback.data(), corrected.data(), storage[4].bytes) == 0,
                "dense controlled FLASH output readback differs from committed values");
        for (float value : readback) dense_finite(value);
        records.back()["readback_verified"] = true;
    }
    void finish() const {
        if (enabled) require(observed == 1, "dense control requires exactly one FLASH operation EVERY decode");
    }
    Json metadata(int width) const {
        return {{"enabled", enabled}, {"mode", enabled ? (short_canonical ? "diagnostic_short_canonical_FP32_over_Q4_decoded_FP16_RNE" : "mathematical_dense_FP32_over_Q4_decoded_FP16_RNE") :
                "unmodified_dense_flashattn"}, {"short_canonical_diagnostic", short_canonical}, {"reference_scope", "pinned_fork_not_HF_training_reference"},
            {"semantic_reference_correction", enabled}, {"calls", calls}, {"rows", corrected_rows},
            {"corrected_values", corrected_values}, {"expected_calls", enabled ? (rows + width - 1) / width : 0},
            {"expected_rows", enabled ? rows : 0}, {"expected_corrected_values", enabled ? rows * dense_columns : 0},
            {"raw_query", "unquantized F32 after query Hadamard"},
            {"cache_decode", "original canonical Q4_0 nibble-8 times stored-half scale -> FP16 RNE -> F32, both K and V"},
            {"arithmetic", short_canonical ? "bounded<=64 ascending FP32; exp64 rounded FP32; numerator times rounded reciprocal" : "ascending full-prefix F32 dot, scale, stable maximum, exp, denominator and weighted V / denominator"},
            {"head_dim", dense_dim}, {"query_heads", dense_heads}, {"kv_heads", dense_kv_heads}, {"head_ratio", 12},
            {"stage", "FLASH output before inverse V Hadamard/gating/output projection"},
            {"query_cache_mask_weights_unchanged", true}, {"raw_nonfinite_is_failure", true},
            {"hf_training_proven", false}, {"performance_claim", false}, {"records", records}};
    }
};

struct Capture {
    const Outputs & out;
    bool enabled, active = false;
    size_t first = 0, count = 0, total_bytes = 0;
    std::array<size_t, sites.size() + 1> observed{};
    Json records = Json::array();
    std::string error;
    DenseF32Control dense;
    Capture(const Outputs & output, bool value, bool control = false, bool canonical = false) : out(output), enabled(value), dense(control, canonical) {}
    void begin(size_t position, size_t n) {
        first = position; count = n; observed.fill(0); active = enabled || dense.enabled;
        dense.begin(position, n);
    }
    static bool callback(ggml_tensor * tensor, bool ask, void * user) noexcept {
        auto & self = *static_cast<Capture *>(user);
        if (!self.active || !self.error.empty()) return false;
        try {
            if (self.dense.enabled && tensor->op == GGML_OP_FLASH_ATTN_EXT) {
                if (ask) { self.dense.validate(tensor); return true; }
                self.dense.correct(tensor);
                return true;
            }
            if (!self.enabled) return false;
            size_t index = sites.size();
            for (size_t i = 0; i < sites.size(); ++i) {
                if (std::strcmp(ggml_get_name(tensor), sites[i].name) == 0) { index = i; break; }
            }
            if (index == sites.size() && std::strcmp(ggml_get_name(tensor), result_norm_site.name) != 0) return false;
            if (ask) return true;
            const Site & site = index == sites.size() ? result_norm_site : sites[index];
            const size_t occurrence = self.observed[index]++;
            require(occurrence < site.expected_occurrences,
                    std::string("too many occurrences of exact intermediate graph site: ") + site.name);
            const size_t elements = site.columns * self.count;
            require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                    ggml_nelements(tensor) == static_cast<int64_t>(elements) &&
                    ggml_nbytes(tensor) == elements * sizeof(float), "intermediate type/shape/contiguity mismatch");
            const size_t bytes = elements * sizeof(float);
            require(bytes <= 3 * vocab * sizeof(float) && bytes <= intermediate_limit - self.total_bytes,
                    "intermediate capture byte bound exceeded");
            std::vector<float> values(elements);
            ggml_backend_tensor_get(tensor, values.data(), 0, bytes);
            const std::string suffix = site.expected_occurrences > 1 ? "-occ" + std::to_string(occurrence) : "";
            const std::string name = "tensor-p" + std::to_string(self.first) + "-" + site.name + suffix + ".f32.bin";
            self.out.bytes(name, values.data(), bytes);
            Json dimensions = Json::array(), strides = Json::array();
            for (int i = 0; i < GGML_MAX_DIMS; ++i) { dimensions.push_back(tensor->ne[i]); strides.push_back(tensor->nb[i]); }
            self.records.push_back({{"name", site.name}, {"occurrence", occurrence}, {"file", name}, {"first_position", self.first},
                {"rows", self.count}, {"columns_per_row", site.columns}, {"dtype", "f32le"},
                {"ne", dimensions}, {"nb", strides}, {"bytes", bytes},
                {"graph_site", index == sites.size() ? "result_norm (renamed mtp_head_input)" : site.name},
                {"nonfinite_count", std::count_if(values.begin(), values.end(), [](float x) { return !std::isfinite(x); })}});
            self.total_bytes += bytes;
            return true;
        } catch (const std::exception & e) { self.error = std::string(e.what()).substr(0, 1024); }
        catch (...) { self.error = "unknown capture callback failure"; }
        return false;
    }
    void finish() {
        active = false;
        require(error.empty(), "capture callback failed: " + error);
        dense.finish();
        if (enabled) for (size_t i = 0; i < observed.size(); ++i) {
            const Site & site = i == sites.size() ? result_norm_site : sites[i];
            require(observed[i] == site.expected_occurrences,
                    std::string("missing exact intermediate graph site occurrence: ") + site.name);
        }
    }
};

using Model = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
struct Backend {
    Backend() { ggml_backend_load_all_from_path(CORE_ORACLE_LIBRARY_DIR); llama_backend_init(); }
    ~Backend() { llama_backend_free(); }
};

Json tensor_descriptor(const ggml_tensor * t) {
    require(t != nullptr, "required loaded tensor is absent");
    Json shape = Json::array();
    for (int i = 0; i < GGML_MAX_DIMS; ++i) shape.push_back(t->ne[i]);
    return {{"name", ggml_get_name(t)}, {"type", ggml_type_name(t->type)}, {"ne", shape}, {"bytes", ggml_nbytes(t)}};
}

void tensor_shape(const ggml_tensor * t, ggml_type type, std::array<int64_t, 4> shape, const char * label) {
    require(t && t->type == type && std::equal(shape.begin(), shape.end(), t->ne), std::string(label) + ": loaded tensor mismatch");
}

void validate_models(const llama_model & main, const llama_model & draft, Json & meta) {
    require(main.arch == LLM_ARCH_QWEN4EXP && draft.arch == LLM_ARCH_QWEN4EXP, "both models must be qwen4exp");
    for (const auto * model : {&main, &draft}) {
        require(llama_model_n_layer(model) == 48 && llama_model_n_embd(model) == embedding &&
                llama_model_n_embd_out(model) == hidden && llama_vocab_n_tokens(llama_model_get_vocab(model)) == vocab &&
                model->hparams.n_expert == 512 && model->hparams.n_expert_used == 10 &&
                model->hparams.dsv4_hc_mult == 4, "loaded model geometry/top10 mismatch");
    }
    tensor_shape(main.tok_embd, GGML_TYPE_Q4_0, {embedding, vocab, 1, 1}, "target embedding");
    tensor_shape(main.output, GGML_TYPE_Q6_K, {embedding, vocab, 1, 1}, "target output");
    require(main.tok_embd != main.output && main.tok_embd->data != main.output->data &&
            draft.tok_embd == nullptr && draft.output == nullptr && draft.hc_head_norm == nullptr,
            "shared sidecar must borrow distinct target embedding/output, not target HC");
    require(draft.hparams.n_layer_nextn == 1 && draft.layers.size() == 49 &&
            draft.hparams.is_recr_impl[48] == 0 && draft.hparams.dsv4_compress_ratios[48] == 0 &&
            !draft.hparams.is_ple_impl[48], "sidecar block48 must be one dense, compression0, non-PLE MTP block");
    const auto & layer = draft.layers.at(48);
    require(layer.wq && layer.wk && layer.wv, "sidecar dense attention tensors absent");
    tensor_shape(layer.ffn_gate_exps, GGML_TYPE_Q8_0, {embedding, 640, 512, 1}, "sidecar gate experts");
    tensor_shape(layer.ffn_up_exps, GGML_TYPE_Q8_0, {embedding, 640, 512, 1}, "sidecar up experts");
    tensor_shape(layer.ffn_down_exps, GGML_TYPE_Q8_0, {640, embedding, 512, 1}, "sidecar down experts");
    const auto & nextn = layer.nextn;
    tensor_shape(nextn.hnorm, GGML_TYPE_F32, {hidden, 1, 1, 1}, "sidecar hnorm");
    tensor_shape(nextn.enorm, GGML_TYPE_F32, {embedding, 1, 1, 1}, "sidecar enorm");
    tensor_shape(nextn.eh_proj, GGML_TYPE_Q8_0, {embedding * 2, embedding, 1, 1}, "sidecar fused projection");
    tensor_shape(nextn.hc_norm, GGML_TYPE_F32, {hidden, 1, 1, 1}, "sidecar own HC norm");
    tensor_shape(nextn.hc_down, GGML_TYPE_Q8_0, {hidden, 320, 1, 1}, "sidecar own HC down");
    tensor_shape(nextn.hc_up, GGML_TYPE_Q8_0, {320, hidden, 1, 1}, "sidecar own HC up");
    require(nextn.fc_hidden == nullptr && nextn.hc_norm != main.hc_head_norm &&
            nextn.hc_down != main.hc_head_down && nextn.hc_up != main.hc_head_up, "sidecar must use its own fused/head weights");
    meta["loaded_tensors"] = {{"target_embedding", tensor_descriptor(main.tok_embd)},
        {"target_output", tensor_descriptor(main.output)}, {"hnorm", tensor_descriptor(nextn.hnorm)},
        {"enorm", tensor_descriptor(nextn.enorm)}, {"fusion", tensor_descriptor(nextn.eh_proj)},
        {"own_hc_norm", tensor_descriptor(nextn.hc_norm)}, {"own_hc_down", tensor_descriptor(nextn.hc_down)},
        {"own_hc_up", tensor_descriptor(nextn.hc_up)}, {"gate_experts", tensor_descriptor(layer.ffn_gate_exps)},
        {"up_experts", tensor_descriptor(layer.ffn_up_exps)}, {"down_experts", tensor_descriptor(layer.ffn_down_exps)}};
}

llama_context_params context_params(const Options & o) {
    auto cp = llama_context_default_params();
    cp.n_ctx = static_cast<uint32_t>(o.capacity);
    cp.n_batch = 3; cp.n_ubatch = 3; cp.n_seq_max = 1; cp.n_rs_seq = 0;
    cp.n_outputs_max = 3; cp.n_outputs_max_per_seq = 3;
    cp.n_threads = 16; cp.n_threads_batch = 16;
    cp.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;
    cp.type_k = GGML_TYPE_Q4_0; cp.type_v = GGML_TYPE_Q4_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL;
    cp.n_moe_cache_slots = o.slots; cp.n_moe_cache_inserts = o.inserts;
    cp.offload_kqv = true; cp.op_offload = true; cp.embeddings = false; cp.no_perf = true;
    cp.ctx_other = nullptr;
    return cp;
}

void run(const Options & o, const Inputs & in, const Outputs & out, Capture & capture, Json & summary) {
    Backend backend;
    summary["library_version_diagnostic"] = llama_version(); // NOT a runtime commit attestation
    std::vector<ggml_backend_dev_t> devices;
    Json device_records = Json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        const std::string reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        if (reg != "ROCm" && reg != "HIP") continue;
        ggml_backend_dev_props props{}; ggml_backend_dev_get_props(dev, &props);
        devices.push_back(dev);
        device_records.push_back({{"name", props.name}, {"description", props.description},
            {"backend", reg}, {"device_id", props.device_id ? props.device_id : ""}, {"memory_total", props.memory_total}});
    }
    require(devices.size() == 2 && ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU),
            "oracle requires exactly two HIP GPUs and canonical CPU backend");
    require(std::string(ggml_backend_dev_name(devices[0])) == "ROCm0" &&
            std::string(ggml_backend_dev_name(devices[1])) == "ROCm1", "expected ordered ROCm0/ROCm1 devices");
    summary["devices"] = device_records;
    std::array<ggml_backend_dev_t, 3> main_devices{devices[0], devices[1], nullptr};
    std::array<ggml_backend_dev_t, 2> draft_devices{devices[1], nullptr};
    std::vector<float> split(llama_max_devices(), 0.0f); split.at(0) = split.at(1) = 1.0f;
    const llama_model_tensor_buft_override overrides[] = {
        {cpu_expert_pattern, ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
    auto mp = llama_model_default_params();
    mp.devices = main_devices.data(); mp.tensor_buft_overrides = overrides;
    mp.n_gpu_layers = -1; mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
    mp.tensor_split = split.data(); mp.main_gpu = 0;
    mp.load_mode = LLAMA_LOAD_MODE_DIRECT_IO; mp.lazy_mode = LLAMA_LAZY_MODE_OFF;
    mp.use_extra_bufts = false; mp.no_host = false; mp.load_mtp = false;
    mp.vocab_only = false; mp.no_alloc = false; mp.check_tensors = false;
    mp.progress_callback = nullptr;
    Model main(llama_model_load_from_file(o.main.c_str(), mp), llama_model_free);
    require(main != nullptr, "target model load failed; inspect bounded donor.log");
    mp.devices = draft_devices.data(); mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.tensor_split = nullptr; mp.main_gpu = 0; mp.load_mtp = true;
    Model draft(llama_model_load_from_file(o.sidecar.c_str(), mp), llama_model_free);
    require(draft != nullptr, "sidecar model load failed; inspect bounded donor.log");
    validate_models(*main, *draft, summary);
    auto cp = context_params(o);
    Context target_ctx(llama_init_from_model(main.get(), cp), llama_free);
    require(target_ctx != nullptr, "target borrowing context initialization failed");
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP; cp.ctx_other = target_ctx.get();
    cp.cb_eval = Capture::callback; cp.cb_eval_user_data = &capture;
    Context draft_ctx(llama_init_from_model(draft.get(), cp), llama_free);
    require(draft_ctx != nullptr, "MTP context initialization failed");
    for (const auto * ctx : {target_ctx.get(), draft_ctx.get()}) {
        // Exact pinned llama-context.cpp applies GGML_PAD(n_ctx,256).
        require(llama_n_ctx(ctx) == 256 && llama_n_batch(ctx) == 3 && llama_n_ubatch(ctx) == 3 &&
                llama_n_seq_max(ctx) == 1, "actual context geometry differs from pinned padding 256/3/3/1");
    }
    summary["actual_context"] = {{"requested_capacity", 64}, {"donor_padded_capacity", 256},
        {"main_capacity", llama_n_ctx(target_ctx.get())}, {"sidecar_capacity", llama_n_ctx(draft_ctx.get())},
        {"n_batch", llama_n_batch(draft_ctx.get())}, {"n_ubatch", llama_n_ubatch(draft_ctx.get())},
        {"n_seq_max", llama_n_seq_max(draft_ctx.get())}};
    llama_set_embeddings_nextn(draft_ctx.get(), true, false);
    llama_set_nextn_layer_offset(draft_ctx.get(), 0);
    llama_set_mtp_chain(draft_ctx.get(), false);
    llama_set_mtp_prefill_kv_only(draft_ctx.get(), false);
    auto d_file = out.open("donor_D.f32.bin"), logits_file = out.open("donor_logits.f32.bin");
    Metrics d_metrics, logit_metrics;
    size_t argmax_matches = 0;
    std::vector<float> donor_d(hidden), donor_logits(vocab), previous_batch(3 * hidden);
    summary["row_metrics"] = Json::array();
    // One fresh MTP context per invocation, always replaying the WHOLE timeline.
    // Width changes batching only. No isolated window with a missing teacher prefix.
    for (size_t p = 0; p < rows; p += static_cast<size_t>(o.width)) {
        const size_t n = std::min(static_cast<size_t>(o.width), rows - p);
        std::array<llama_token, 3> tokens{};
        std::array<llama_pos, 3> positions{};
        std::array<int32_t, 3> sequence_counts{1, 1, 1};
        llama_seq_id sequence = 0;
        std::array<llama_seq_id *, 3> sequences{&sequence, &sequence, &sequence};
        std::array<int8_t, 3> wanted{1, 1, 1};
        for (size_t i = 0; i < n; ++i) { tokens[i] = in.tokens[p + i]; positions[i] = static_cast<llama_pos>(p + i); }
        std::copy_n(in.previous.data() + p * hidden, n * hidden, previous_batch.data());
        // Caller-owned token AND embd storage. llama_batch_init allocates only one.
        llama_batch batch{static_cast<int32_t>(n), tokens.data(), previous_batch.data(), positions.data(),
            sequence_counts.data(), sequences.data(), wanted.data()};
        capture.begin(p, n);
        const int result = llama_decode(draft_ctx.get(), batch); // standalone donor ONLY; never decode target
        llama_synchronize(draft_ctx.get());
        capture.finish();
        require(result == 0, "MTP decode failed at position " + std::to_string(p) + ": " + std::to_string(result));
        for (size_t i = 0; i < n; ++i) {
            // Both getters use BATCH-relative i. Copy before the next decode expires them.
            const float * d = llama_get_embeddings_nextn_ith(draft_ctx.get(), static_cast<int32_t>(i));
            const float * logits = llama_get_logits_ith(draft_ctx.get(), static_cast<int32_t>(i));
            require(d && logits, "MTP getter returned a null row");
            std::copy_n(d, hidden, donor_d.data()); std::copy_n(logits, vocab, donor_logits.data());
            write_all(d_file.get(), donor_d.data(), donor_d.size() * sizeof(float));
            write_all(logits_file.get(), donor_logits.data(), donor_logits.size() * sizeof(float));
            const size_t position = p + i;
            const std::span<const float> own_d(in.own_d.data() + position * hidden, hidden);
            const std::span<const float> own_logits(in.own_logits.data() + position * vocab, vocab);
            Metrics row_d, row_logits;
            row_d.row(own_d, donor_d, position, 0.002); row_logits.row(own_logits, donor_logits, position, 0.02);
            d_metrics.row(own_d, donor_d, position, 0.002); logit_metrics.row(own_logits, donor_logits, position, 0.02);
            const bool finite = row_logits.nonfinite == 0;
            const size_t own_argmax = static_cast<size_t>(std::max_element(own_logits.begin(), own_logits.end()) - own_logits.begin());
            const size_t donor_argmax = static_cast<size_t>(std::max_element(donor_logits.begin(), donor_logits.end()) - donor_logits.begin());
            argmax_matches += finite && own_argmax == donor_argmax;
            summary["row_metrics"].push_back({{"position", position}, {"token", in.tokens[position]},
                {"D", row_d.json(0.002)}, {"logits", row_logits.json(0.02)},
                {"argmax_diagnostic", {{"own", own_argmax}, {"donor", finite ? Json(donor_argmax) : Json(nullptr)},
                    {"match", finite && own_argmax == donor_argmax}}}});
            summary["completed_rows"] = position + 1;
            summary["D"] = d_metrics.json(0.002); summary["logits"] = logit_metrics.json(0.02);
        }
    }
    d_file.close_checked(); logits_file.close_checked();
    summary["argmax_diagnostic_matches"] = argmax_matches;
    require(d_metrics.compared == rows * hidden && logit_metrics.compared == rows * vocab, "incomplete full-value comparison");
    if (o.dense_f32_control) {
        require(capture.dense.calls == (rows + static_cast<size_t>(o.width) - 1) / static_cast<size_t>(o.width) &&
                capture.dense.corrected_rows == rows && capture.dense.corrected_values == rows * dense_columns,
                "dense control did not correct exactly one FLASH node per complete teacher decode");
    }
    summary["passed"] = d_metrics.violations == 0 && logit_metrics.violations == 0;
    summary["status"] = summary["passed"].get<bool>() ? "passed" : "numerical_failure";
}

Json initial_summary(const Options & o, const Inputs & in) {
    return {{"schema", "gfx906-mtp-teacher-oracle-v1"}, {"status", "running"}, {"passed", false},
        {"completed_rows", 0}, {"source_revision", pin}, {"source_tracked_dirty", false},
        {"library_revision_attested", CORE_ORACLE_LIBRARY_REVISION}, {"library_directory", CORE_ORACLE_LIBRARY_DIR},
        {"library_revision_runtime_verified", false},
        {"library_attestation_scope", "build-time verified image revision supplied by caller; not runtime content hash"},
        {"input_capture", in.descriptor}, {"input_directory", o.capture.string()},
        {"main_model", {{"path", o.main.string()}, {"bytes", main_bytes}, {"load_mtp", false},
            {"devices", {"ROCm0", "ROCm1"}}, {"split", "layer"}, {"tensor_split", {1, 1}}, {"main_gpu", 0}}},
        {"sidecar_model", {{"path", o.sidecar.string()}, {"bytes", sidecar_bytes}, {"load_mtp", true},
            {"devices", {"ROCm1"}}, {"split", "none"}, {"main_gpu_local_index", 0}}},
        {"teacher_width", o.width}, {"teacher_tokens", in.tokens}, {"positions", in.descriptor.at("positions")},
        {"teacher_history", "fresh context at position0; every preceding teacher row decoded chronologically"},
        {"previous_hidden", "supplied capture preserved; row0 zero production convention, later rows target H[p-1]"},
        {"zero_boundary_hf_proven", false}, {"tap", "post-block widened D before sidecar own HC; batch-relative getter"},
        {"tap_shape", {rows, hidden}}, {"logits_shape", {rows, vocab}},
        {"flags", {{"capacity", o.capacity}, {"n_batch", 3}, {"n_ubatch", 3}, {"n_outputs_max", 3},
            {"n_outputs_max_per_seq", 3}, {"n_seq_max", 1}, {"sequence_id", 0}, {"logits_each_row", true},
            {"n_rs_seq", 0}, {"threads", 16}, {"threads_batch", 16}, {"cache_slots", o.slots},
            {"cache_inserts", o.inserts}, {"cpu_expert_pattern", cpu_expert_pattern}, {"canonical_cpu_experts", true},
            {"K", "Q4_0"}, {"V", "Q4_0"}, {"causal", true}, {"flash_attention", true},
            {"offload_kqv", true}, {"op_offload", true}, {"no_perf", true}, {"use_extra_bufts", false},
            {"no_host", false}, {"direct_io", true}, {"lazy", false}, {"embeddings", false},
            {"embeddings_nextn", true}, {"embeddings_nextn_masked", false}, {"nextn_layer_offset", 0},
            {"mtp_chain", false}, {"mtp_prefill_kv_only", false}, {"full_vocab", true},
            {"top_experts", 10}, {"sidecar_experts", "Q8_0"}, {"sidecar_block", 48}, {"compression", 0},
            {"borrow_target_embedding_output", true}, {"borrow_target_hc", false}, {"target_decode_calls", 0},
            {"main_context_type", "DEFAULT"}, {"sidecar_context_type", "MTP"}, {"sampler", nullptr},
            {"capture_intermediates", o.intermediates}, {"dense_f32_control", o.dense_f32_control}, {"dense_short_canonical", o.dense_short_canonical}}},
        {"environment", in.env}, {"intermediate_byte_limit", intermediate_limit}, {"log_byte_limit", log_limit},
        {"outputs", {{"D", "donor_D.f32.bin"}, {"logits", "donor_logits.f32.bin"}, {"dtype", "f32le"},
            {"layout", "chronological-row-major"}}}, {"hf_reference_claim", false}, {"R6_complete_claim", false},
        {"performance_claim", false}, {"bitdiff_is_gate", false}, {"argmax_is_gate", false}};
}

// These are CPU helper regressions, not a mocked scheduler/backend oracle.
// Only self_test_q4() needs GGML: its real public half conversions run after
// ggml_init() initializes the CPU tables. No backend loading/device discovery.
struct SelfTestStats {
    size_t checks = 0, rejections = 0, math_calls = 0, compared = 0;
    double maxabs = 0.0, maxratio = 0.0;
    Json widths = Json::array();
    void check(bool ok, const char * label) {
        require(ok, std::string("self-test: ") + label);
        ++checks;
    }
    template<class F> void rejects(F && function, std::string_view message) {
        try { function(); }
        catch (const std::runtime_error & e) {
            check(std::string_view(e.what()).find(message) != std::string_view::npos,
                  "rejection must report the expected invariant");
            ++rejections;
            return;
        }
        throw std::runtime_error("self-test: expected rejection: " + std::string(message));
    }
    void compare(std::span<const float> actual, std::span<const double> expected) {
        check(actual.size() == expected.size(), "independent double comparison dimensions");
        for (size_t i = 0; i < actual.size(); ++i) {
            const double error = std::abs(static_cast<double>(actual[i]) - expected[i]);
            const double ratio = error / (2e-4 + 2e-4 * std::abs(expected[i]));
            check(std::isfinite(actual[i]) && std::isfinite(expected[i]) && ratio <= 1.0,
                  "independent double component gate 2e-4 + 2e-4*abs(reference)");
            maxabs = std::max(maxabs, error); maxratio = std::max(maxratio, ratio);
            ++compared;
        }
    }
};

// Separately expressed DOUBLE reference: no dense FP32 arithmetic helpers,
// Q4 decoder, production kernel or own-runtime code is called here.
std::vector<double> self_test_double(std::span<const float> q, std::span<const float> k,
                                     std::span<const float> v, size_t first, size_t count) {
    std::vector<double> result(q.size());
    for (size_t row = 0; row < count; ++row) for (size_t head = 0; head < 24; ++head) {
        std::vector<double> weights(first + row + 1);
        double largest = -std::numeric_limits<double>::infinity(), denominator = 0.0;
        for (size_t position = 0; position < weights.size(); ++position) {
            double dot = 0.0;
            for (size_t d = 0; d < 256; ++d) {
                dot += static_cast<double>(q[(row * 24 + head) * 256 + d]) *
                       static_cast<double>(k[(position * 2 + head / 12) * 256 + d]);
            }
            weights[position] = dot / 16.0;
            largest = std::max(largest, weights[position]);
        }
        for (double & weight : weights) { weight = std::exp(weight - largest); denominator += weight; }
        for (size_t d = 0; d < 256; ++d) {
            double sum = 0.0;
            for (size_t position = 0; position < weights.size(); ++position) {
                sum += weights[position] * static_cast<double>(v[(position * 2 + head / 12) * 256 + d]);
            }
            result[(row * 24 + head) * 256 + d] = sum / denominator;
        }
    }
    return result;
}

void self_test_math(SelfTestStats & s) {
    s.check(std::fegetround() == FE_TONEAREST, "round-to-nearest arithmetic required");
    const auto run = [&s](std::span<const float> q, std::span<const float> k, std::span<const float> v,
                          size_t first, size_t count) {
        auto result = dense_attention_f32(q, k, v, first, count);
        s.compare(result, self_test_double(q, k, v, first, count));
        if (first + count <= 64) {
            const auto canonical = dense_attention_f32(q, k, v, first, count, true);
            s.compare(canonical, self_test_double(q, k, v, first, count));
        }
        ++s.math_calls;
        return result;
    };
    std::vector<float> q(dense_columns), k(512), v(512);
    for (size_t i = 0; i < q.size(); ++i) q[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 64.0f;
    for (size_t i = 0; i < k.size(); ++i) {
        k[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 32.0f;
        v[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 32.0f;
    }
    v[0] = -0.0f; v[256] = -0.0f;
    const auto identity = run(q, k, v, 0, 1);
    for (size_t head = 0; head < 24; ++head) for (size_t d = 0; d < 256; ++d) {
        s.check(std::bit_cast<uint32_t>(identity[head * 256 + d]) ==
                std::bit_cast<uint32_t>(v[head / 12 * 256 + d]), "single KV identity including -0/head ratio");
    }

    // Guarded inputs contain poisoned physical future storage outside the exact
    // mathematical prefix span. Finite, nonconstant in-batch future values also
    // test causal exclusion from each earlier query's DOUBLE reference.
    constexpr float canary = 12345.5f;
    std::vector<float> guarded_q(rows * dense_columns + 2, canary);
    std::vector<float> guarded_k((rows + 1) * 512 + 2, canary), guarded_v(guarded_k);
    auto tq = std::span<float>(guarded_q).subspan(1, rows * dense_columns);
    auto tk = std::span<float>(guarded_k).subspan(1, rows * 512);
    auto tv = std::span<float>(guarded_v).subspan(1, rows * 512);
    for (size_t p = 0; p < rows; ++p) {
        for (size_t h = 0; h < 24; ++h) for (size_t d = 0; d < 256; ++d) {
            tq[(p * 24 + h) * 256 + d] = static_cast<float>(static_cast<int>((p + 3 * h + d) % 11) - 5) / 32.0f;
        }
        for (size_t h = 0; h < 2; ++h) for (size_t d = 0; d < 256; ++d) {
            tk[(p * 2 + h) * 256 + d] = static_cast<float>(static_cast<int>((3 * p + h + d) % 13) - 6) / 16.0f;
            tv[(p * 2 + h) * 256 + d] = static_cast<float>(static_cast<int>((p + 5 * h + d) % 23) - 11) / 4.0f;
        }
    }
    for (auto * input : {&guarded_k, &guarded_v}) {
        std::fill_n(input->begin() + 1 + rows * 512, 512, std::numeric_limits<float>::quiet_NaN());
    }
    const auto before_q = guarded_q, before_k = guarded_k, before_v = guarded_v;
    std::vector<float> n1;
    for (size_t width = 1; width <= 3; ++width) {
        std::vector<float> chronological(rows * dense_columns + 2, canary);
        size_t calls = 0, complete_rows = 0;
        for (size_t p = 0; p < rows; p += width) {
            const size_t n = std::min(width, rows - p);
            const auto result = run(tq.subspan(p * dense_columns, n * dense_columns),
                                    tk.first((p + n) * 512), tv.first((p + n) * 512), p, n);
            std::copy(result.begin(), result.end(), chronological.begin() + 1 + p * dense_columns);
            ++calls; complete_rows += n;
        }
        s.check(calls == (rows + width - 1) / width && complete_rows == rows,
                "all 32 chronological rows, width1/2/3 boundary and short final batch");
        s.check(chronological.front() == canary && chronological.back() == canary, "output canaries intact");
        if (width == 1) n1 = chronological;
        else s.check(std::memcmp(n1.data(), chronological.data(), chronological.size() * sizeof(float)) == 0,
                     "width changes batching only, not chronological mathematical values");
        s.widths.push_back({{"width", width}, {"calls", calls}, {"rows", complete_rows},
                           {"compared", rows * dense_columns}});
    }
    s.check(std::memcmp(before_q.data(), guarded_q.data(), guarded_q.size() * sizeof(float)) == 0 &&
            std::memcmp(before_k.data(), guarded_k.data(), guarded_k.size() * sizeof(float)) == 0 &&
            std::memcmp(before_v.data(), guarded_v.data(), guarded_v.size() * sizeof(float)) == 0,
            "all inputs, canaries and poisoned future storage unchanged");

    q.assign(dense_columns, 0.0f); k.assign(2 * 512, 0.0f); v.assign(2 * 512, 0.0f);
    for (size_t h = 0; h < 24; ++h) q[h * 256] = 1.0f;
    for (size_t h = 0; h < 2; ++h) { k[h * 256] = 16000.0f; k[(2 + h) * 256] = 16001.0f; }
    std::fill(v.begin() + 512, v.end(), 2.0f);
    run(q, k, v, 1, 1); // Logits near 1000 require stable maximum before exp.

    q.assign(3 * dense_columns, 0.0f);
    for (size_t i = 0; i < q.size(); ++i) q[i] = static_cast<float>(static_cast<int>(i % 7) - 3) / 64.0f;
    k.assign(2053 * 512 + 2, canary); v.assign(k.size(), canary);
    std::fill_n(k.begin() + 1, 2052 * 512, 0.0f); std::fill_n(v.begin() + 1, 2052 * 512, 0.0f);
    std::fill_n(v.begin() + 1, 512, 1.0f); std::fill_n(v.begin() + 1 + 2051 * 512, 512, 2052.0f);
    for (auto * input : {&k, &v}) {
        std::fill_n(input->begin() + 1 + 2052 * 512, 512, std::numeric_limits<float>::quiet_NaN());
    }
    const auto long_before_k = k, long_before_v = v;
    const auto long_result = run(q, std::span<const float>(k).subspan(1, 2052 * 512),
                                std::span<const float>(v).subspan(1, 2052 * 512), 2049, 3);
    for (size_t row = 0; row < 3; ++row) for (size_t column = 0; column < dense_columns; ++column) {
        const float expected = row == 2 ? 2053.0f / 2052.0f : 1.0f / static_cast<float>(2050 + row);
        s.check(long_result[row * dense_columns + column] == expected,
                "2052 full dense positions include earliest/tail values and exclude in-batch future");
    }
    s.check(std::memcmp(long_before_k.data(), k.data(), k.size() * sizeof(float)) == 0 &&
            std::memcmp(long_before_v.data(), v.data(), v.size() * sizeof(float)) == 0,
            "long-history input canaries/future poison unchanged");

    q.assign(dense_columns, 0.0f); k.assign(512, 0.0f); v.assign(512, 1.0f);
    std::array<float, 3> published{canary, -0.0f, -canary};
    const auto original_output = published;
    const auto invalid = [&s, &published, &original_output](auto && function, std::string_view error) {
        s.rejects([&] { const auto result = function(); published[0] = result.at(0); }, error);
        s.check(std::memcmp(published.data(), original_output.data(), sizeof(published)) == 0,
                "rejected arithmetic leaves caller output/canaries unchanged");
    };
    invalid([&] { return dense_attention_f32(q, k, v, 0, 0); }, "bound");
    invalid([&] { return dense_attention_f32(q, k, v, 0, 4); }, "bound");
    invalid([&] { return dense_attention_f32(q, k, v, 4096, 1); }, "bound");
    invalid([&] { return dense_attention_f32(q, k, v, 4095, 2); }, "bound");
    invalid([&] { return dense_attention_f32(q, k, v, std::numeric_limits<size_t>::max(), 1); }, "bound");
    invalid([&] { return dense_attention_f32(std::span<const float>(q).first(q.size() - 1), k, v, 0, 1); }, "dimensions");
    invalid([&] { return dense_attention_f32(q, std::span<const float>(k).first(511), v, 0, 1); }, "dimensions");
    invalid([&] { return dense_attention_f32(q, k, std::span<const float>(v).first(511), 0, 1); }, "dimensions");
    q[0] = std::numeric_limits<float>::quiet_NaN();
    invalid([&] { return dense_attention_f32(q, k, v, 0, 1); }, "nonfinite"); q[0] = 0.0f;
    k[0] = std::numeric_limits<float>::infinity();
    invalid([&] { return dense_attention_f32(q, k, v, 0, 1); }, "nonfinite"); k[0] = 0.0f;
    v[0] = -std::numeric_limits<float>::infinity();
    invalid([&] { return dense_attention_f32(q, k, v, 0, 1); }, "nonfinite"); v[0] = 1.0f;
    q[0] = std::numeric_limits<float>::max(); k[0] = 2.0f;
    invalid([&] { return dense_attention_f32(q, k, v, 0, 1); }, "nonfinite");
    q.assign(dense_columns, 0.0f); q[0] = q[1] = q[2] = std::numeric_limits<float>::max();
    k.assign(512, 0.0f); k[0] = k[1] = k[2] = 0.5f;
    invalid([&] { return dense_attention_f32(q, k, v, 0, 1); }, "nonfinite");
    q.assign(dense_columns, 0.0f); k.assign(2 * 512, 0.0f); v.assign(k.size(), std::numeric_limits<float>::max());
    invalid([&] { return dense_attention_f32(q, k, v, 1, 1); }, "nonfinite");
    s.rejects([] { dense_add_size(std::numeric_limits<size_t>::max(), 1); }, "addition overflow");
    s.rejects([] { dense_multiply_size(std::numeric_limits<size_t>::max(), 2); }, "multiplication overflow");
    s.check(dense_add_size(0, 7) == 7 && dense_multiply_size(0, std::numeric_limits<size_t>::max()) == 0,
            "valid zero size arithmetic");
}

void self_test_preflight(SelfTestStats & s) {
    for (const double atol : {0.002, 0.02}) {
        const float rounded = static_cast<float>(atol);
        const float below = static_cast<double>(rounded) <= atol ? rounded : std::nextafter(rounded, 0.0f);
        const float above = static_cast<double>(rounded) > atol ? rounded :
                            std::nextafter(rounded, std::numeric_limits<float>::infinity());
        Metrics boundary;
        const std::array<float, 2> own{below, above}, ref{};
        boundary.row(own, ref, 7, atol);
        s.check(boundary.violations == 1 && boundary.first.at("position") == 7 && boundary.first.at("column") == 1 &&
                boundary.json(atol).at("rtol") == 0.002, "frozen D/logit tolerance boundary and relative tolerance");
    }
    Metrics equality, scaled, invalid;
    const std::array<float, 2> own{0.0f, -0.0f}, ref{};
    equality.row(own, ref, 0, 0.002);
    s.check(equality.violations == 0 && equality.bitdiff == 1, "signed-zero bitdiff diagnostic is not a gate");
    const std::array<float, 1> scaled_own{101.0f}, scaled_ref{100.0f};
    scaled.row(scaled_own, scaled_ref, 4, 0.02);
    s.check(scaled.violations == 1 && std::abs(scaled.maxratio - 1.0 / 0.22) < 1e-12,
            "logit relative tolerance uses donor reference magnitude");
    const std::array<float, 2> bad{std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    invalid.row(ref, bad, 2, 0.002);
    s.check(invalid.nonfinite == 2 && invalid.violations == 2 && !invalid.json(0.002).at("passed").get<bool>(),
            "nonfinite donor values always fail numerical gates");
    const auto parse = [](const std::string & text) { return parse_json(std::vector<char>(text.begin(), text.end())); };
    s.check(parse("{\"source\":{\"dirty\":true}}").at("source").at("dirty").is_boolean(), "existing JSON parser");
    s.rejects([&] { parse("{\"source\":{\"dirty\":true,\"dirty\":false}}"); }, "duplicate key");
    s.rejects([&] { parse(std::string(18, '[') + "0" + std::string(18, ']')); }, "nesting");
    for (const std::string name : {"../capture.json", "/capture.json", ".", "..", "sub\\capture.json", ""}) {
        s.rejects([&] { safe_name(name); }, "basename");
    }
    s.rejects([] { safe_name(std::string("capture\0.json", 13)); }, "basename");
    Json descriptor{{"tokens", {{"name", "teacher_tokens.i32.bin"}, {"dtype", "i32le"},
                                {"shape", {rows}}, {"bytes", 128}}}};
    file_descriptor(descriptor, "tokens", "teacher_tokens.i32.bin", "i32le", {rows});
    ++s.checks;
    descriptor["tokens"]["bytes"] = 129;
    s.rejects([&] { file_descriptor(descriptor, "tokens", "teacher_tokens.i32.bin", "i32le", {rows}); }, "byte count");
    descriptor["tokens"]["bytes"] = 128; descriptor["tokens"]["shape"][0] = 31;
    s.rejects([&] { file_descriptor(descriptor, "tokens", "teacher_tokens.i32.bin", "i32le", {rows}); }, "dimension");
    descriptor["tokens"]["shape"][0] = rows; descriptor["tokens"]["name"] = "../teacher_tokens.i32.bin";
    s.rejects([&] { file_descriptor(descriptor, "tokens", "teacher_tokens.i32.bin", "i32le", {rows}); }, "name/dtype");
    s.check(!integer_equal(Json(true), 1) && !integer_equal(Json(-1), 1) && !integer_equal(Json(32.0), 32),
            "descriptor dimensions require nonnegative integers, not bool/float");

    std::array<std::string, 12> words{"mtp-teacher-oracle", "UNOPENED_MAIN", "UNOPENED_SIDECAR", "UNOPENED_CAPTURE",
        "UNCREATED_OUTPUT", "--teacher-width", "1", "--dense-f32-control", "--dense-f32-control",
        "--teacher-width", "2", "--self-test"};
    std::array<char *, 12> argv{};
    for (size_t i = 0; i < words.size(); ++i) argv[i] = words[i].data();
    for (int width = 1; width <= 3; ++width) {
        words[6] = std::to_string(width); argv[6] = words[6].data();
        const auto opts = parse_options(7, argv.data());
        s.check(opts.width == width && !opts.dense_f32_control && !opts.intermediates && opts.inserts == 2,
                "raw fork default/explicit width/cache defaults unchanged");
    }
    s.check(parse_options(8, argv.data()).dense_f32_control, "explicit dense control accepted");
    {
        std::array<std::string,9> text{"oracle","a","b","c","d","--teacher-width","1","--dense-f32-control","--dense-short-canonical"};
        std::array<char*,9> args{};
        for(size_t i=0;i<text.size();++i) args[i]=text[i].data();
        s.check(parse_options(9,args.data()).dense_short_canonical,"explicit bounded arithmetic diagnostic accepted");
        args[7]=text[8].data();
        s.rejects([&]{parse_options(8,args.data());},"requires explicit dense control");
        std::vector<float> q(dense_columns),kv(65*512);
        s.rejects([&]{dense_attention_f32(q,kv,kv,64,1,true);},"<=64");
    }

    s.rejects([&] { parse_options(9, argv.data()); }, "duplicate option");
    argv[7] = words[9].data(); argv[8] = words[10].data();
    s.rejects([&] { parse_options(9, argv.data()); }, "duplicate option");
    argv[7] = words[11].data();
    s.rejects([&] { parse_options(8, argv.data()); }, "missing value");
    s.rejects([&] { parse_options(9, argv.data()); }, "unknown option");
    s.rejects([&] { parse_options(2, argv.data()); }, "usage");
    DenseF32Control raw(false);
    for (size_t p = 0; p < rows; ++p) { raw.begin(p, 1); raw.finish(); }
    s.check(raw.calls == 0 && raw.corrected_rows == 0 && raw.corrected_values == 0 &&
            raw.metadata(1).at("expected_calls") == 0 && raw.metadata(1).at("mode") == "unmodified_dense_flashattn",
            "raw fork default has zero control overrides");
}

void self_test_q4(SelfTestStats & s) {
    s.check(ggml_type_size(GGML_TYPE_Q4_0) == 18 && ggml_blck_size(GGML_TYPE_Q4_0) == 32 &&
            ggml_row_size(GGML_TYPE_Q4_0, 256) == 144, "real GGML canonical Q4_0 geometry");
    for (const auto & [value, bits] : std::array<std::pair<float, ggml_fp16_t>, 8>{{
             {1.00048828125f, 0x3c00}, {1.00146484375f, 0x3c02},
             {-1.00048828125f, 0xbc00}, {-1.00146484375f, 0xbc02},
             {0.0f, 0x0000}, {-0.0f, 0x8000}, {65504.0f, 0x7bff}, {-65504.0f, 0xfbff}}}) {
        s.check(ggml_fp32_to_fp16(value) == bits, "real GGML positive/negative RNE ties/signed zero/scales");
    }
    s.check(std::bit_cast<uint32_t>(ggml_fp16_to_fp32(0x8000)) == 0x80000000U,
            "real GGML half signed zero preserved");
    std::array<char, 146> guarded{};
    guarded.front() = 0x5a; guarded.back() = 0x6b;
    auto row = std::span<char>(guarded).subspan(1, 144);
    const auto scale = [&row](size_t block, ggml_fp16_t bits) { std::memcpy(row.data() + block * 18, &bits, 2); };
    for (size_t block = 0; block < 8; ++block) {
        scale(block, block % 4 == 0 ? 0x3555 : block % 4 == 1 ? 0x3c01 : block % 4 == 2 ? 0xb555 : 0xbc01);
        std::fill_n(row.begin() + block * 18 + 2, 16, static_cast<char>(0x5b));
    }
    const auto before = guarded;
    for (size_t d = 0; d < 256; ++d) {
        const size_t block = d / 32;
        float expected = block % 2 == 0 ? 1.0f : 3.00390625f;
        if (block % 4 >= 2) expected = -expected;
        if (d % 32 >= 16) expected = -expected;
        s.check(dense_q4_value(row, d) == expected, "Q4 nibble order, signed scales and product FP16 RNE ties");
    }
    s.check(guarded == before, "Q4 input bytes/canaries unchanged");
    for (size_t block = 0; block < 8; ++block) {
        scale(block, 0x3c00);
        for (size_t lane = 0; lane < 16; ++lane) row[block * 18 + 2 + lane] = static_cast<char>(lane | ((15 - lane) << 4));
    }
    for (size_t d = 0; d < 256; ++d) {
        const int code = d % 32 < 16 ? static_cast<int>(d % 16) : 15 - static_cast<int>(d % 16);
        s.check(dense_q4_value(row, d) == static_cast<float>(code - 8), "all low/high canonical Q4 signed codes");
    }
    scale(0, 0x8000); row[2] = 0x5b;
    s.check(std::bit_cast<uint32_t>(dense_q4_value(row, 0)) == 0x80000000U &&
            std::bit_cast<uint32_t>(dense_q4_value(row, 16)) == 0U, "signed-zero Q4 scale/product");
    for (const ggml_fp16_t bits : std::array<ggml_fp16_t, 6>{0x7e00, 0xfe00, 0x7c00, 0xfc00, 0x7bff, 0xfbff}) {
        scale(0, bits); row[2] = 0x0f;
        const auto original = guarded;
        s.rejects([&] { dense_q4_value(row, 0); }, "nonfinite");
        s.check(guarded == original, "rejected Q4 header/product leaves source and canaries unchanged");
    }
    s.rejects([&] { dense_q4_value(row.first(143), 0); }, "dimension mismatch");
    s.rejects([&] { dense_q4_value(std::span<const char>(guarded).first(145), 0); }, "dimension mismatch");
    s.rejects([&] { dense_q4_value(row, 256); }, "dimension mismatch");
    s.rejects([&] { dense_q4_value(row, std::numeric_limits<size_t>::max()); }, "dimension mismatch");
}

int self_test() {
    SelfTestStats stats;
    try {

    require(pin == CORE_ORACLE_LIBRARY_REVISION, "compiled library attestation does not match donor pin");
        {
            // Real CPU context initializes GGML's conversion tables; no tensor
            // backend, graph, device registry, HIP runtime or model is requested.
            std::unique_ptr<ggml_context, decltype(&ggml_free)> cpu(
                ggml_init(ggml_init_params{64 * 1024, nullptr, true}), ggml_free);
            require(cpu != nullptr, "self-test: CPU GGML context initialization failed");
            self_test_q4(stats);
            self_test_math(stats);
            self_test_preflight(stats);
        } // CPU context and all temporary input storage are released before success.
#ifdef CORE_REVISION
        const std::string oracle_revision = CORE_REVISION;
#else
        const std::string oracle_revision = "d2f948ae44f5172dd2395fdbf1723e0b1ddd6237";
#endif
#ifdef CORE_DIRTY
        const bool oracle_dirty = CORE_DIRTY;
#else
        const bool oracle_dirty = true;
#endif
        const Json result{{"schema", "gfx906-mtp-teacher-oracle-self-test-v1"}, {"status", "passed"}, {"passed", true},
            {"CPU_ONLY_no_models_GPU", true}, {"cpu_context_released", true},
            {"backend_loading_calls", 0}, {"device_enumeration_calls", 0}, {"model_load_calls", 0},
            {"filesystem_artifacts", 0}, {"callback_transport_tested", false}, {"full_file_preflight_tested", false},
            {"scope", "real GGML public Q4/half helpers; independent CPU dense math; CLI/descriptor/comparison primitives"},
            {"library_revision_attested", CORE_ORACLE_LIBRARY_REVISION},
            {"oracle_code_source", {{"revision", oracle_revision}, {"dirty", oracle_dirty},
                {"attestation_scope", "declared current-slice base/dirty; compile-time CORE_REVISION/CORE_DIRTY when supplied"},
                {"runtime_git_verified", false}}},
            {"checks", stats.checks}, {"rejections", stats.rejections}, {"math_calls", stats.math_calls},
            {"double_reference_compared", stats.compared}, {"double_reference_maxabs", stats.maxabs},
            {"double_reference_maxratio", stats.maxratio}, {"component_atol", 2e-4}, {"component_rtol", 2e-4},
            {"chronological_widths", stats.widths}, {"long_dense_prefix", 2052}, {"raw_default_overrides", 0},
            {"frozen_D_atol", 0.002}, {"frozen_logits_atol", 0.02}, {"frozen_D_logits_rtol", 0.002},
            {"hf_reference_claim", false}, {"GPU_reference_claim", false}, {"performance_claim", false}};
        const std::string text = result.dump();
        require(text.size() + 1 <= json_limit, "self-test JSON exceeds 64KiB bound");
        require(std::puts(text.c_str()) >= 0, "self-test stdout write failed");
        return 0;
    } catch (const std::exception & e) {
        const Json failure{{"schema", "gfx906-mtp-teacher-oracle-self-test-v1"}, {"status", "self_test_failure"},
            {"passed", false}, {"CPU_ONLY_no_models_GPU", true}, {"checks", stats.checks},
            {"rejections", stats.rejections}, {"error", std::string(e.what()).substr(0, 2048)}};
        std::fprintf(stderr, "%s\n", failure.dump().c_str());
        return 1;
    }
}
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) return self_test();
        auto opts = parse_options(argc, argv);
        const auto input = preflight(opts); // strict preflight BEFORE mkdir/backend/model loading
        Outputs out(opts.output);
        Json summary = initial_summary(opts, input);
        Capture capture(out, opts.intermediates, opts.dense_f32_control, opts.dense_short_canonical);
        summary["dense_attention_reference"] = capture.dense.metadata(opts.width);
        // Current slice's declared code provenance is distinct from the clean
        // pinned donor source/library provenance above. Build macros may supply
        // the parent's actual checkout metadata; no runtime Git access occurs.
#ifdef CORE_REVISION
        const std::string oracle_revision = CORE_REVISION;
#else
        const std::string oracle_revision = "d2f948ae44f5172dd2395fdbf1723e0b1ddd6237";
#endif
#ifdef CORE_DIRTY
        const bool oracle_dirty = CORE_DIRTY;
#else
        const bool oracle_dirty = true;
#endif
        summary["oracle_code_source"] = {{"revision", oracle_revision}, {"dirty", oracle_dirty},
            {"attestation_scope", "declared current-slice base/dirty; compile-time CORE_REVISION/CORE_DIRTY when supplied"},
            {"runtime_git_verified", false}};
        Json frozen_run = summary;
        BoundedLog log(out);
        // llama_log_set also sets the ggml log callback in the exact pinned donor.
        llama_log_set(BoundedLog::callback, &log);
        try {
            out.bytes("input_capture.json", input.descriptor_bytes.data(), input.descriptor_bytes.size());
            out.bytes("teacher_tokens.i32.bin", input.tokens.data(), input.tokens.size() * sizeof(llama_token));
            out.bytes("teacher_previous_hidden.f32.bin", input.previous.data(), input.previous.size() * sizeof(float));
            run(opts, input, out, capture, summary);
            require(!log.failed, "bounded donor log write failed");
        } catch (const std::exception & e) {
            summary["passed"] = false; summary["status"] = "execution_failure";
            summary["error"] = std::string(e.what()).substr(0, 2048);
        }
        llama_log_set(nullptr, nullptr);
        summary["dense_attention_reference"] = capture.dense.metadata(opts.width);
        frozen_run["dense_attention_reference"] = summary["dense_attention_reference"];
        frozen_run["counter_phase"] = "after_execution_attempt; configuration/source frozen before model loading";
        out.json("run.json", frozen_run);
        summary["intermediates"] = capture.records;
        summary["intermediate_bytes"] = capture.total_bytes;
        summary["donor_log_bytes"] = log.written;
        summary["donor_log_truncated"] = log.truncated;
        out.json("summary.json", summary);
        const bool passed = summary.at("passed").get<bool>() && summary.at("completed_rows") == rows;
        const Json result{{"schema", "gfx906-mtp-teacher-oracle-result-v1"}, {"passed", passed},
            {"status", summary.at("status")}, {"completed_rows", summary.at("completed_rows")},
            {"dense_f32_control", opts.dense_f32_control},
            {"attention_reference_mode", summary.at("dense_attention_reference").at("mode")},
            {"output_directory", opts.output.string()}, {"summary", "summary.json"}};
        std::puts(result.dump().c_str());
        return passed ? 0 : 1;
    } catch (const std::exception & e) {
        const Json failure{{"schema", "gfx906-mtp-teacher-oracle-result-v1"}, {"passed", false},
            {"status", "preflight_or_artifact_failure"}, {"error", std::string(e.what()).substr(0, 2048)}};
        std::fprintf(stderr, "%s\n", failure.dump().c_str());
        return 1;
    }
}
