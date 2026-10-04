#include "mtp_model.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>

namespace {

using Bytes = std::vector<std::byte>;
std::size_t checks = 0;
std::size_t rejected_cases = 0;

void check(bool condition, const char* description) {
    ++checks;
    if (!condition) throw std::runtime_error(description);
}

template<class Function> void expect_failure(Function function, std::string_view message) {
    try {
        function();
    } catch (const std::runtime_error& error) {
        if (std::string_view(error.what()).find(message) == std::string_view::npos)
            throw std::runtime_error("unexpected rejection: " + std::string(error.what()));
        ++checks;
        ++rejected_cases;
        return;
    }
    throw std::runtime_error("malformed MTP inventory unexpectedly accepted");
}

// Independent literal GGUF writer, patterned on model_test.cpp. Real geometry
// is represented by sparse extents; no model payloads or model copies are used.
struct Writer {
    Bytes data;

    template<class T> void number(T value) {
        if constexpr (std::is_same_v<T, bool>) number<std::uint8_t>(value ? 1 : 0);
        else if constexpr (std::is_floating_point_v<T>) {
            using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
            number(std::bit_cast<Bits>(value));
        } else {
            using Unsigned = std::make_unsigned_t<T>;
            const auto bits = std::bit_cast<Unsigned>(value);
            for (std::size_t i = 0; i < sizeof(T); ++i)
                data.push_back(std::byte((std::uint64_t(bits) >> (8 * i)) & 255));
        }
    }

    void string(std::string_view value) {
        number<std::uint64_t>(value.size());
        const auto bytes = std::as_bytes(std::span(value.data(), value.size()));
        data.insert(data.end(), bytes.begin(), bytes.end());
    }

    void append(const Bytes& bytes) { data.insert(data.end(), bytes.begin(), bytes.end()); }
};

struct MetadataSpec {
    std::string key;
    std::uint32_t type;
    Bytes value;
};

struct TensorSpec {
    std::string name;
    std::uint32_t type;
    std::vector<std::uint64_t> dimensions;
};

struct TensorFields {
    std::size_t name;
    std::size_t rank;
    std::vector<std::size_t> dimensions;
    std::size_t type;
    std::size_t offset;
};

struct Image {
    Bytes header;
    std::uint64_t file_size;
    std::vector<TensorFields> tensors;
    std::map<std::string, std::size_t> metadata_values;
    std::map<std::string, std::size_t> metadata_types;
};

// Size oracle uses literal GGUF layouts, never production type_layout().
std::uint64_t byte_size(const TensorSpec& tensor) {
    std::uint64_t block = 1;
    std::uint64_t bytes = 4;
    switch (tensor.type) {
    case 0: break;
    case 1: case 30: bytes = 2; break;
    case 2: block = 32; bytes = 18; break;
    case 8: block = 32; bytes = 34; break;
    case 14: block = 256; bytes = 210; break;
    default: throw std::runtime_error("unknown fixture type");
    }
    auto size = tensor.dimensions.at(0) / block * bytes;
    for (std::size_t axis = 1; axis < tensor.dimensions.size(); ++axis)
        size *= tensor.dimensions[axis];
    return size;
}

struct Builder {
    std::vector<MetadataSpec> metadata;
    std::vector<TensorSpec> tensors;

    void value(std::string key, std::uint32_t type, Bytes payload) {
        const auto it = std::find_if(metadata.begin(), metadata.end(), [&](const auto& spec) {
            return spec.key == key;
        });
        if (it == metadata.end()) metadata.push_back({std::move(key), type, std::move(payload)});
        else *it = {std::move(key), type, std::move(payload)};
    }

    template<class T> void scalar(std::string key, std::uint32_t type, T scalar_value) {
        Writer writer;
        writer.number(scalar_value);
        value(std::move(key), type, std::move(writer.data));
    }

    void string(std::string key, std::string_view text) {
        Writer writer;
        writer.string(text);
        value(std::move(key), 8, std::move(writer.data));
    }

    template<class T> void array(std::string key, std::uint32_t type, const std::vector<T>& values) {
        Writer writer;
        writer.number(type);
        writer.number<std::uint64_t>(values.size());
        for (const auto& v : values) {
            if constexpr (std::is_same_v<T, std::string>) writer.string(v);
            else writer.number<T>(v);
        }
        value(std::move(key), 9, std::move(writer.data));
    }

    void erase(std::string_view key) {
        std::erase_if(metadata, [&](const auto& spec) { return spec.key == key; });
    }

    Image build() const {
        Writer writer;
        writer.number<std::uint32_t>(0x46554747);
        writer.number<std::uint32_t>(3);
        writer.number<std::uint64_t>(tensors.size());
        writer.number<std::uint64_t>(metadata.size());
        Image image{{}, 0, {}, {}, {}};
        for (const auto& spec : metadata) {
            writer.string(spec.key);
            image.metadata_types.emplace(spec.key, writer.data.size());
            writer.number(spec.type);
            image.metadata_values.emplace(spec.key, writer.data.size());
            writer.append(spec.value);
        }
        std::uint64_t extent = 0;
        for (const auto& spec : tensors) {
            TensorFields fields{writer.data.size(), 0, {}, 0, 0};
            writer.string(spec.name);
            fields.rank = writer.data.size();
            writer.number<std::uint32_t>(static_cast<std::uint32_t>(spec.dimensions.size()));
            for (const auto dimension : spec.dimensions) {
                fields.dimensions.push_back(writer.data.size());
                writer.number(dimension);
            }
            fields.type = writer.data.size();
            writer.number(spec.type);
            fields.offset = writer.data.size();
            writer.number(extent);
            image.tensors.push_back(std::move(fields));
            extent = (extent + byte_size(spec) + 31) & ~std::uint64_t{31};
        }
        while (writer.data.size() % 32 != 0) writer.data.push_back(std::byte{0});
        image.file_size = writer.data.size() + extent;
        image.header = std::move(writer.data);
        return image;
    }
};

template<class T> void patch(Image& image, std::size_t offset, T value) {
    Writer writer;
    writer.number(value);
    check(offset <= image.header.size() && writer.data.size() <= image.header.size() - offset,
          "fixture patch outside header");
    std::copy(writer.data.begin(), writer.data.end(),
              image.header.begin() + static_cast<std::ptrdiff_t>(offset));
}

class TempFile {
public:
    explicit TempFile(const Image& image) {
        auto name = (std::filesystem::temp_directory_path() / "qwen-mtp-model-test-XXXXXX").string();
        fd_ = ::mkstemp(name.data());
        if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "mkstemp MTP fixture");
        path_ = name;
        try {
            std::span remaining(image.header);
            while (!remaining.empty()) {
                const auto result = ::write(fd_, remaining.data(), remaining.size());
                if (result < 0 && errno == EINTR) continue;
                if (result <= 0) throw std::runtime_error("fixture write failed");
                remaining = remaining.subspan(static_cast<std::size_t>(result));
            }
            truncate(image.file_size);
        } catch (...) {
            ::close(fd_);
            ::unlink(path_.c_str());
            throw;
        }
    }

    ~TempFile() { ::close(fd_); ::unlink(path_.c_str()); }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;
    const std::string& path() const { return path_; }

    void truncate(std::uint64_t size) {
        if (size > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()))
            throw std::runtime_error("fixture file size not representable");
        if (::ftruncate(fd_, static_cast<off_t>(size)) != 0)
            throw std::system_error(errno, std::generic_category(), "truncate MTP fixture");
    }

private:
    int fd_;
    std::string path_;
};

Builder common_builder() {
    Builder b;
    b.string("general.architecture", "qwen4exp");
    for (const auto& [key, value] : std::initializer_list<std::pair<const char*, std::uint32_t>>{
             {"embedding_length", 2560}, {"context_length", 262144},
             {"hyper_connection.count", 4}, {"hyper_connection.low_rank", 320},
             {"attention.head_count", 24}, {"attention.head_count_kv", 2},
             {"attention.key_length", 256}, {"attention.value_length", 256},
             {"expert_count", 512}, {"expert_used_count", 10},
             {"expert_feed_forward_length", 640}, {"expert_shared_feed_forward_length", 640},
             {"rope.dimension_count", 64}})
        b.scalar<std::uint32_t>("qwen4exp." + std::string(key), 4, value);
    b.scalar<float>("qwen4exp.attention.layer_norm_rms_epsilon", 6, 1.0e-6F);
    b.scalar<float>("qwen4exp.rope.freq_base", 6, 1.0e7F);
    b.array<std::int32_t>("qwen4exp.rope.dimension_sections", 5, {11, 11, 10, 0});

    // Correct typed cardinalities, synthetic contents. Equality must still
    // inspect every byte/element, rather than trusting geometry or a sample.
    std::vector<std::string> tokens(248320, "t");
    tokens[tokens.size() / 2] = std::string("a\0z", 3);
    tokens.back() = "\xE2\x82\xAC";
    b.array<std::string>("tokenizer.ggml.tokens", 8, tokens);
    b.array<std::int32_t>("tokenizer.ggml.token_type", 5, std::vector<std::int32_t>(248320, 1));
    b.array<std::string>("tokenizer.ggml.merges", 8, std::vector<std::string>(247587, "a b"));
    b.string("tokenizer.ggml.model", "gpt2");
    b.string("tokenizer.ggml.pre", "qwen35");
    b.scalar<std::uint32_t>("tokenizer.ggml.bos_token_id", 4, 248044);
    b.scalar<std::uint32_t>("tokenizer.ggml.eos_token_id", 4, 248046);
    b.scalar<std::uint32_t>("tokenizer.ggml.padding_token_id", 4, 248044);
    b.scalar<bool>("tokenizer.ggml.add_bos_token", 7, false);
    std::string chat(8952, 'c');
    chat.replace(chat.size() - 3, 3, "\xE2\x82\xAC");
    b.string("tokenizer.chat_template", chat);
    return b;
}

Builder sidecar_builder() {
    static const auto common = common_builder();
    auto b = common;
    b.scalar<bool>("qwen4exp.nextn_shared_target_tensors", 7, true);
    b.scalar<std::uint32_t>("qwen4exp.block_count", 4, 49);
    b.scalar<std::uint32_t>("qwen4exp.nextn_predict_layers", 4, 1);
    std::vector<std::int32_t> ratios(49, 0);
    for (std::size_t i = 3; i < 48; i += 4) ratios[i] = 4;
    b.array<std::int32_t>("qwen4exp.attention.compress_ratios", 5, ratios);
    b.tensors = {
        {"nextn.enorm", 8, {2560}}, {"nextn.hnorm", 8, {10240}},
        {"nextn.eh_proj", 8, {5120, 2560}},
        {"nextn.hc_head_norm", 8, {10240}}, {"nextn.hc_head_down", 8, {10240, 320}},
        {"nextn.hc_head_up", 8, {320, 10240}},
        {"hc_attn_norm", 8, {10240}}, {"hc_ffn_norm", 8, {10240}},
        {"hc_attn_down", 8, {10240, 320}}, {"hc_ffn_down", 8, {10240, 320}},
        {"hc_attn_up", 8, {320, 10240}}, {"hc_ffn_up", 8, {320, 10240}},
        {"hc_attn_inject", 8, {10240, 4}}, {"hc_ffn_inject", 8, {10240, 4}},
        {"attn_q", 8, {2560, 12288}}, {"attn_k", 8, {2560, 512}},
        {"attn_v", 8, {2560, 512}}, {"attn_output", 8, {6144, 2560}},
        {"attn_q_norm", 8, {256}}, {"attn_k_norm", 8, {256}},
        {"ffn_gate_inp", 8, {2560, 512}}, {"ffn_gate_inp_shexp", 8, {2560}},
        {"ffn_gate_exps", 8, {2560, 640, 512}}, {"ffn_up_exps", 8, {2560, 640, 512}},
        {"ffn_down_exps", 8, {640, 2560, 512}},
        {"ffn_gate_shexp", 8, {2560, 640}}, {"ffn_up_shexp", 8, {2560, 640}},
        {"ffn_down_shexp", 8, {640, 2560}},
        {"indexer.q_proj", 8, {2560, 512}}, {"indexer.k_proj", 8, {2560, 128}},
        {"indexer.q_norm", 8, {128}}, {"indexer.k_norm", 8, {128}}
    };
    // Inventory matches the actual read-only donor GGUFReader observation.
    for (const std::size_t i : {0U, 1U, 3U, 6U, 7U, 18U, 19U, 20U, 21U, 30U, 31U})
        b.tensors[i].type = 0;
    b.tensors[28].type = 30;
    b.tensors[29].type = 30;
    for (auto& tensor : b.tensors) tensor.name = "blk.48." + tensor.name + ".weight";
    return b;
}

Builder target_builder() {
    static const auto common = common_builder();
    auto b = common;
    b.scalar<std::uint32_t>("qwen4exp.block_count", 4, 48);
    std::vector<std::int32_t> ratios(48, 0);
    for (std::size_t i = 3; i < ratios.size(); i += 4) ratios[i] = 4;
    b.array<std::int32_t>("qwen4exp.attention.compress_ratios", 5, ratios);
    b.tensors = {{"token_embd.weight", 2, {2560, 248320}}, {"output.weight", 14, {2560, 248320}},
                 {"hc_head_norm.weight", 0, {10240}}};
    return b;
}

void reject_sidecar(const Image& image, const qwen::Model& target, std::string_view reason) {
    TempFile file(image);
    expect_failure([&] { const qwen::MtpModel mtp(file.path(), target); }, reason);
}

void test_valid_descriptor() {
    static_assert(!std::is_copy_constructible_v<qwen::MtpModel>);
    static_assert(!std::is_move_constructible_v<qwen::MtpModel>);
    static_assert(!std::is_constructible_v<qwen::MtpModel, const std::filesystem::path&, qwen::Model&&>);
    static_assert(!std::is_constructible_v<qwen::MtpModel, const std::filesystem::path&, const qwen::Model&&>);
    static_assert(std::is_same_v<decltype(std::declval<qwen::MtpSharedTargetWeights>().embedding),
                                 const qwen::TensorView&>);
    static_assert(std::is_same_v<decltype(std::declval<qwen::MtpHeadWeights>().norm),
                                 const qwen::TensorView&>);
    const auto builder = sidecar_builder();
    TempFile file(builder.build());
    TempFile target_file(target_builder().build());
    qwen::Model target(target_file.path());
    {
        const qwen::MtpModel mtp(file.path(), target);
        const auto& g = mtp.geometry();
        check(g.layer == 48 && g.block_count == 49 && g.predict_layers == 1, "trained layer geometry");
        check(g.hidden_size == 2560 && g.hc_count == 4 && g.hc_rank == 320 && g.widened_size == 10240,
              "widened HC geometry");
        check(g.query_heads == 24 && g.kv_heads == 2 && g.head_dim == 256, "attention geometry");
        check(g.experts == 512 && g.required_experts_used == 10 && g.intermediate_size == 640,
              "metadata/tensor expert geometry and validated top-k");
        check(g.rotary_dim == 64 && g.rope_base == 1.0e7F && g.rms_epsilon == 1.0e-6F &&
              g.rope_sections == std::array<std::int32_t, 4>{11, 11, 10, 0}, "typed RoPE and epsilon");
        check(g.vocabulary_size == 248320 && g.kv_type == qwen::TensorType::Q4_0 &&
              g.required_compression_ratio == 0, "required dense Q4 KV contract");
        const auto counts = mtp.type_counts();
        check(counts.q8_0 == 19 && counts.f32 == 11 && counts.bf16 == 2, "actual supported type counts");
        check(&mtp.target() == &target, "borrowed target Model identity");
        const auto& shared = mtp.shared_target();
        check(&shared.embedding == &target.tensor("token_embd.weight"), "borrowed embedding view identity");
        check(&shared.output == &target.tensor("output.weight"), "borrowed output view identity");
        check(&shared.embedding != &shared.output && shared.embedding.file_offset != shared.output.file_offset,
              "untied target views and ranges");
        const auto& w = mtp.weights();
        const std::array<const qwen::TensorView*, 32> views{
            &w.embedding_norm, &w.hidden_norm, &w.embedding_hidden_projection,
            &w.head.norm, &w.head.down, &w.head.up,
            &w.attention_hc.mixer.norm, &w.ffn_hc.mixer.norm,
            &w.attention_hc.mixer.down, &w.ffn_hc.mixer.down,
            &w.attention_hc.mixer.up, &w.ffn_hc.mixer.up,
            &w.attention_hc.inject, &w.ffn_hc.inject,
            &w.attention.query, &w.attention.key, &w.attention.value, &w.attention.output,
            &w.attention.query_norm, &w.attention.key_norm,
            &w.experts.router, &w.experts.shared_router, &w.experts.gate, &w.experts.up, &w.experts.down,
            &w.experts.shared_gate, &w.experts.shared_up, &w.experts.shared_down,
            &w.indexer.query, &w.indexer.key, &w.indexer.query_norm, &w.indexer.key_norm};
        for (std::size_t i = 0; i < views.size(); ++i) {
            const auto& view = *views[i];
            check(view.name == builder.tensors[i].name, "role binding name");
            check(static_cast<std::uint32_t>(view.type) == builder.tensors[i].type,
                  "exact observed role dtype");
            check(&view == &mtp.sidecar().tensor(view.name), "no copied TensorView");
            check(view.byte_size == byte_size(builder.tensors[i]), "independent exact byte size");
            check(view.file_offset == mtp.sidecar().data_offset() + view.relative_offset,
                  "sidecar absolute file offset");
        }
        check(&w.head.norm != &target.tensor("hc_head_norm.weight"), "MTP head never borrows target HC");
        check(w.experts.gate.strides[2] == 1740800 && w.experts.up.strides[2] == 1740800 &&
              w.experts.down.strides[2] == 1740800,
              "routed Q8 expert byte strides");
        const auto requirements = mtp.validation_requirements();
        check(requirements.per_role_type_inventory && requirements.dense_compression_metadata &&
              requirements.attention_expert_metadata &&
               requirements.target_tokenizer_equivalence, "mandatory loader conditions enforced");
        check(w.indexer.query.type == qwen::TensorType::BF16 &&
              w.indexer.key.type == qwen::TensorType::BF16 &&
              w.embedding_hidden_projection.type == qwen::TensorType::Q8_0 &&
              w.head.down.type == qwen::TensorType::Q8_0, "actual BF16 placement, no old guess");
        check(!mtp.sidecar().find_metadata("tokenizer.ggml.scores") &&
              !target.find_metadata("tokenizer.ggml.scores") &&
              !mtp.sidecar().find_metadata("tokenizer.ggml.add_eos_token") &&
              !target.find_metadata("tokenizer.ggml.add_eos_token"), "optional assets absent on both");
        std::array<std::byte, 16> bytes{};
        mtp.target().read_slice(shared.embedding.name, 0, bytes);
        check(std::all_of(bytes.begin(), bytes.end(), [](std::byte b) { return b == std::byte{0}; }),
              "borrowed target descriptor remains readable");
    }
    check(target.tensor("output.weight").byte_size == 521472000, "target still owned after MTP destruction");
    std::array<std::byte, 1> byte{};
    target.read_slice("output.weight", 0, byte);

}

void test_shapes_types_and_inventory(const qwen::Model& target) {
    const auto base = sidecar_builder();
    // Independently reject a wrong axis 0 on EVERY role, including the last
    // tensor. Keep quantization block divisibility so shape checks are reached.
    for (std::size_t i = 0; i < base.tensors.size(); ++i) {
        auto b = base;
        b.tensors[i].dimensions[0] -= b.tensors[i].type == 8 ? 32U : 1U;
        reject_sidecar(b.build(), target, "shape/rank");
    }
    for (const auto i : {2U, 4U, 14U, 22U, 23U, 24U}) {
        auto b = base;
        std::swap(b.tensors[i].dimensions[0], b.tensors[i].dimensions[1]);
        reject_sidecar(b.build(), target, "shape/rank");
    }
    auto rank = base;
    rank.tensors.back().dimensions.push_back(1);
    reject_sidecar(rank.build(), target, "shape/rank");
    auto missing = base;
    missing.tensors.pop_back();
    reject_sidecar(missing.build(), target, "exactly 32");
    auto extra = base;
    extra.tensors.push_back({"blk.48.unexpected.weight", 0, {32}});
    reject_sidecar(extra.build(), target, "exactly 32");
    extra.tensors.back().name = "token_embd.weight";
    reject_sidecar(extra.build(), target, "exactly 32");
    auto wrong_layer = base;
    wrong_layer.tensors.back().name = "blk.47.indexer.k_norm.weight";
    reject_sidecar(wrong_layer.build(), target, "missing tensor");
    auto duplicate = base;
    duplicate.tensors.back().name = duplicate.tensors.front().name;
    reject_sidecar(duplicate.build(), target, "duplicate tensor");
    auto unsupported = base;
    unsupported.tensors.back().type = 1;
    reject_sidecar(unsupported.build(), target, "unsupported sidecar tensor type");
    auto counts = base;
    counts.tensors.back().type = 30;
    reject_sidecar(counts.build(), target, "tensor type for role");
    // Each role is challenged with both other supported types, while keeping
    // the complete aggregate dtype inventory unchanged (64 permutations).
    for (std::size_t i = 0; i < base.tensors.size(); ++i) {
        for (const auto type : {0U, 8U, 30U}) {
            if (type == base.tensors[i].type) continue;
            auto swapped = base;
            const auto it = std::find_if(swapped.tensors.begin(), swapped.tensors.end(),
                                        [&](const auto& spec) { return spec.type == type; });
            std::swap(swapped.tensors[i].type, it->type);
            reject_sidecar(swapped.build(), target, "tensor type for role");
        }
    }
}

void test_metadata(const qwen::Model& target) {
    const auto base = sidecar_builder();
    for (const auto& spec : base.metadata) {
        auto missing = base;
        missing.erase(spec.key);
        reject_sidecar(missing.build(), target, "missing metadata");
        auto wrong_type = base;
        // Replace the whole value, rather than reinterpreting a differently
        // sized payload. This remains valid GGUF and reaches typed validation.
        wrong_type.scalar<std::uint64_t>(spec.key, 10, 1);
        reject_sidecar(wrong_type.build(), target, "wrong metadata type");
        if (spec.type == 4 && !spec.key.starts_with("tokenizer.")) {
            auto wrong_value = base;
            wrong_value.scalar<std::uint32_t>(spec.key, 4, 0);
            reject_sidecar(wrong_value.build(), target, "unexpected metadata value");
        }
    }
    auto architecture = base;
    architecture.string("general.architecture", "qwen35moe");
    reject_sidecar(architecture.build(), target, "architecture");
    auto shared = base;
    shared.scalar<bool>("qwen4exp.nextn_shared_target_tensors", 7, false);
    reject_sidecar(shared.build(), target, "must be true");
    for (const auto key : {"qwen4exp.rope.freq_base", "qwen4exp.attention.layer_norm_rms_epsilon"}) {
        for (const float value : {0.0F, -0.0F, -1.0F, 0.01F, std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::quiet_NaN()}) {
            auto b = base;
            b.scalar<float>(key, 6, value);
            reject_sidecar(b.build(), target, "unexpected metadata value");
        }
    }
    auto section_type = base;
    section_type.array<std::uint32_t>("qwen4exp.rope.dimension_sections", 4, {11, 11, 10, 0});
    reject_sidecar(section_type.build(), target, "array element type");
    for (const auto& values : {std::vector<std::int32_t>{11, 11, 10},
                              std::vector<std::int32_t>{11, 11, 10, 0, 0},
                              std::vector<std::int32_t>{10, 11, 11, 0},
                              std::vector<std::int32_t>{11, 11, 10, -1}}) {
        auto b = base;
        b.array<std::int32_t>("qwen4exp.rope.dimension_sections", 5, values);
        reject_sidecar(b.build(), target, "unexpected metadata value");
    }
    auto duplicate = base;
    duplicate.metadata.push_back(duplicate.metadata.front());
    reject_sidecar(duplicate.build(), target, "duplicate metadata");
}

void test_compression(const qwen::Model& target) {
    constexpr auto key = "qwen4exp.attention.compress_ratios";
    const auto base = sidecar_builder();
    auto wrong_element_type = base;
    wrong_element_type.array<std::uint32_t>(key, 4, std::vector<std::uint32_t>(49, 0));
    reject_sidecar(wrong_element_type.build(), target, "array element type");
    for (const std::size_t size : {0U, 48U, 50U}) {
        auto b = base;
        b.array<std::int32_t>(key, 5, std::vector<std::int32_t>(size, 0));
        reject_sidecar(b.build(), target, "metadata length");
    }
    for (const std::size_t index : {0U, 24U, 47U, 48U}) {
        for (const std::int32_t value : {-1, 1, INT32_MAX}) {
            auto b = base;
            std::vector<std::int32_t> ratios(49, 0);
            ratios[index] = value;
            b.array<std::int32_t>(key, 5, ratios);
            reject_sidecar(b.build(), target, "unexpected metadata value");
        }
    }
    auto compressed_draft = base;
    std::vector<std::int32_t> ratios(49, 0);
    ratios.back() = 4;
    compressed_draft.array<std::int32_t>(key, 5, ratios);
    reject_sidecar(compressed_draft.build(), target, "compression ratio must be zero");
    auto reordered = base;
    std::reverse(ratios.begin(), ratios.end()); // nonzero is now in unused trunk, not blk.48.
    reordered.array<std::int32_t>(key, 5, ratios);
    TempFile file(reordered.build());
    const qwen::MtpModel mtp(file.path(), target);
    check(mtp.geometry().required_compression_ratio == 0,
          "unused trunk compression order is not an MTP requirement");
}

void accept_pair(const Builder& sidecar, const Builder& target) {
    TempFile sidecar_file(sidecar.build());
    TempFile target_file(target.build());
    const qwen::Model model(target_file.path());
    const qwen::MtpModel mtp(sidecar_file.path(), model);
    check(&mtp.target() == &model, "fully compatible typed metadata pair accepted");
}

void reject_pair(const Builder& sidecar, const Builder& target, std::string_view reason) {
    TempFile sidecar_file(sidecar.build());
    TempFile target_file(target.build());
    expect_failure([&] {
        const qwen::Model model(target_file.path());
        const qwen::MtpModel mtp(sidecar_file.path(), model);
    }, reason);
}

void test_tokenizer_assets(const qwen::Model& target) {
    const auto base = sidecar_builder();
    const auto target_base = target_builder();
    // Mutate beginning, middle and final element, including bytes after NUL and
    // UTF-8 continuation bytes. Array/string equality must cover the full assets.
    for (const std::size_t index : {0U, 124160U, 248319U}) {
        std::vector<std::string> tokens(248320, "t");
        tokens[124160] = std::string("a\0z", 3);
        tokens.back() = "\xE2\x82\xAC";
        tokens[index].back() = static_cast<char>(static_cast<unsigned char>(tokens[index].back()) ^ 1U);
        auto b = base;
        b.array<std::string>("tokenizer.ggml.tokens", 8, tokens);
        reject_sidecar(b.build(), target, "tokenizer metadata mismatch");
        std::vector<std::int32_t> types(248320, 1);
        types[index] = 2;
        b = base;
        b.array<std::int32_t>("tokenizer.ggml.token_type", 5, types);
        reject_sidecar(b.build(), target, "tokenizer metadata mismatch");
    }
    for (const std::size_t index : {0U, 123793U, 247586U}) {
        auto b = base;
        std::vector<std::string> merges(247587, "a b");
        merges[index].back() = 'c';
        b.array<std::string>("tokenizer.ggml.merges", 8, merges);
        reject_sidecar(b.build(), target, "tokenizer metadata mismatch");
    }
    for (const std::size_t size : {248319U, 248321U}) {
        auto b = base;
        b.array<std::string>("tokenizer.ggml.tokens", 8, std::vector<std::string>(size, "t"));
        reject_sidecar(b.build(), target, "vocabulary cardinality");
        // Even two equally malformed assets must not pass the identity check.
        auto t = target_base;
        t.array<std::string>("tokenizer.ggml.tokens", 8, std::vector<std::string>(size, "t"));
        reject_pair(b, t, "vocabulary cardinality");
        b = base;
        b.array<std::int32_t>("tokenizer.ggml.token_type", 5, std::vector<std::int32_t>(size, 1));
        reject_sidecar(b.build(), target, "vocabulary cardinality");
        t = target_base;
        t.array<std::int32_t>("tokenizer.ggml.token_type", 5, std::vector<std::int32_t>(size, 1));
        reject_pair(base, t, "vocabulary cardinality");
    }
    auto wrong_elements = base;
    wrong_elements.array<std::uint32_t>("tokenizer.ggml.token_type", 4, std::vector<std::uint32_t>(248320, 1));
    reject_sidecar(wrong_elements.build(), target, "array element type");
    for (const auto key : {"tokenizer.ggml.tokens", "tokenizer.ggml.merges"}) {
        wrong_elements = base;
        wrong_elements.array<std::int32_t>(key, 5, {1});
        reject_sidecar(wrong_elements.build(), target, "array element type");
    }
    for (const std::size_t size : {0U, 247586U, 247588U}) {
        auto b = base;
        b.array<std::string>("tokenizer.ggml.merges", 8, std::vector<std::string>(size, "a b"));
        reject_sidecar(b.build(), target, "tokenizer metadata mismatch");
    }
    for (const auto key : {"tokenizer.ggml.model", "tokenizer.ggml.pre"}) {
        auto b = base;
        b.string(key, "different");
        reject_sidecar(b.build(), target, "tokenizer metadata mismatch");
    }
    for (const std::size_t index : {0U, 4476U, 8951U}) {
        auto b = base;
        std::string chat(8952, 'c');
        chat.replace(chat.size() - 3, 3, "\xE2\x82\xAC");
        chat[index] = '\0';
        b.string("tokenizer.chat_template", chat);
        reject_sidecar(b.build(), target, "tokenizer metadata mismatch");
    }
    for (const auto key : {"tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                           "tokenizer.ggml.padding_token_id"}) {
        auto b = base;
        b.scalar<std::uint32_t>(key, 4, 0);
        reject_sidecar(b.build(), target, "tokenizer metadata mismatch");
        for (const std::uint32_t id : {248320U, UINT32_MAX}) {
            b.scalar<std::uint32_t>(key, 4, id);
            reject_sidecar(b.build(), target, "special token outside");
        }
    }
    auto changed_bool = base;
    changed_bool.scalar<bool>("tokenizer.ggml.add_bos_token", 7, true);
    reject_sidecar(changed_bool.build(), target, "tokenizer metadata mismatch");

    // Every tokenizer-prefixed key matters, not just the ten currently present.
    for (const auto key : {"tokenizer.ggml.scores", "tokenizer.ggml.add_eos_token",
                           "tokenizer.future_asset"}) {
        auto s = base;
        auto t = target_base;
        if (key == std::string_view("tokenizer.ggml.scores")) {
            s.array<float>(key, 6, std::vector<float>(248320, 0.0F));
            t.array<float>(key, 6, std::vector<float>(248320, 0.0F));
        } else {
            s.scalar<bool>(key, 7, false);
            t.scalar<bool>(key, 7, false);
        }
        reject_sidecar(s.build(), target, "tokenizer key-set mismatch");
        reject_pair(base, t, "tokenizer key-set mismatch");
        // Matching presence/type/value is sufficient; no absent-key defaults.
        accept_pair(s, t);
        t.scalar<std::uint8_t>(key, 0, 0);
        reject_pair(s, t, "tokenizer metadata mismatch");
    }
    auto missing_both_s = base;
    auto missing_both_t = target_base;
    missing_both_s.erase("tokenizer.ggml.tokens");
    missing_both_t.erase("tokenizer.ggml.tokens");
    reject_pair(missing_both_s, missing_both_t, "missing metadata");
    // Non-tokenizer metadata is not accidentally included in asset equality.
    auto unrelated = base;
    unrelated.string("tokenizer", "not under tokenizer dot prefix");
    unrelated.string("general.name", "synthetic draft");
    accept_pair(unrelated, target_base);
}

template<class T> void test_typed_tokenizer_equality(std::uint32_t type, T value, T changed) {
    auto s = sidecar_builder();
    auto t = target_builder();
    constexpr auto scalar_key = "tokenizer.test.scalar";
    constexpr auto array_key = "tokenizer.test.array";
    s.scalar<T>(scalar_key, type, value);
    t.scalar<T>(scalar_key, type, value);
    s.array<T>(array_key, type, {value, value});
    t.array<T>(array_key, type, {value, value});
    accept_pair(s, t);
    t.scalar<T>(scalar_key, type, changed);
    reject_pair(s, t, "tokenizer metadata mismatch");
    t.scalar<T>(scalar_key, type, value);
    t.array<T>(array_key, type, {value, changed});
    reject_pair(s, t, "tokenizer metadata mismatch");
    t.array<T>(array_key, type, {value});
    reject_pair(s, t, "tokenizer metadata mismatch");
    // Numeric values that happen to compare equally must still have same tags.
    t.array<std::uint32_t>(array_key, 4, {0, 0});
    if (type == 4) t.array<std::int32_t>(array_key, 5, {0, 0});
    reject_pair(s, t, "tokenizer metadata mismatch");
}

void test_tokenizer_bit_patterns() {
    test_typed_tokenizer_equality<std::uint8_t>(0, 0, UINT8_MAX);
    test_typed_tokenizer_equality<std::int8_t>(1, INT8_MIN, INT8_MAX);
    test_typed_tokenizer_equality<std::uint16_t>(2, 0, UINT16_MAX);
    test_typed_tokenizer_equality<std::int16_t>(3, INT16_MIN, INT16_MAX);
    test_typed_tokenizer_equality<std::uint32_t>(4, 0, UINT32_MAX);
    test_typed_tokenizer_equality<std::int32_t>(5, INT32_MIN, INT32_MAX);
    test_typed_tokenizer_equality<std::uint64_t>(10, UINT64_MAX, UINT64_MAX - 1);
    test_typed_tokenizer_equality<std::int64_t>(11, INT64_MIN, INT64_MAX);
    test_typed_tokenizer_equality<bool>(7, false, true);
    test_typed_tokenizer_equality<float>(6, 0.0F, -0.0F);
    test_typed_tokenizer_equality<double>(12, 0.0, -0.0);
    // Model's parser allows arbitrary floating metadata, so matching NaNs must
    // pass and differing NaN payloads must fail, for both scalars and arrays.
    test_typed_tokenizer_equality<float>(6, std::bit_cast<float>(0x7FC00001U),
                                       std::bit_cast<float>(0x7FC00002U));
    test_typed_tokenizer_equality<double>(12, std::bit_cast<double>(0x7FF8000000000001ULL),
                                        std::bit_cast<double>(0x7FF8000000000002ULL));
}

void test_parser_guards(const qwen::Model& target) {
    const auto base = sidecar_builder().build();
    const auto mutate = [&](auto function, std::string_view reason) {
        auto image = base;
        function(image);
        reject_sidecar(image, target, reason);
    };
    mutate([](Image& i) { patch<std::uint64_t>(i, 8, UINT64_MAX); }, "overflow");
    mutate([](Image& i) { patch<std::uint64_t>(i, 16, UINT64_MAX); }, "overflow");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors.back().name, UINT64_MAX); }, "length");
    for (const auto rank : {0U, 5U})
        mutate([&](Image& i) { patch<std::uint32_t>(i, i.tensors.back().rank, rank); }, "rank");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors.back().dimensions[0], 0); }, "zero tensor");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors.back().dimensions[0], UINT64_MAX); }, "overflow");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[22].dimensions[0], 2559); }, "axis 0");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors.back().offset, 1); }, "unaligned");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors.back().offset, 0); }, "overlapping");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors.back().offset, UINT64_MAX - 31); }, "overflow");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors.back().offset, 1ULL << 40); }, "outside");
    mutate([](Image& i) { --i.file_size; }, "outside");
    mutate([](Image& i) { patch<std::uint32_t>(i, i.tensors.back().type, 99); }, "unsupported tensor type");
    mutate([](Image& i) {
        patch<std::uint64_t>(i, i.metadata_values.at("qwen4exp.rope.dimension_sections") + 4, UINT64_MAX);
    }, "overflow");
    mutate([](Image& i) {
        patch<std::uint8_t>(i, i.metadata_values.at("qwen4exp.nextn_shared_target_tensors"), 2);
    }, "boolean");
    mutate([](Image& i) {
        patch<std::uint32_t>(i, i.metadata_types.at("qwen4exp.nextn_shared_target_tensors"), 99);
    }, "metadata type");
    mutate([](Image& i) {
        patch<std::uint64_t>(i, i.metadata_values.at("general.architecture"), UINT64_MAX);
    }, "length");
}

void test_borrowed_target_rejections() {
    TempFile sidecar_file(sidecar_builder().build());
    const auto reject = [&](const Builder& b, std::string_view reason) {
        TempFile file(b.build());
        expect_failure([&] {
            qwen::Model target(file.path());
            const qwen::MtpModel mtp(sidecar_file.path(), target);
        }, reason);
    };
    const auto base = target_builder();
    for (const auto i : {0U, 1U}) {
        auto missing = base;
        missing.tensors.erase(missing.tensors.begin() + static_cast<std::ptrdiff_t>(i));
        reject(missing, "missing tensor");
        auto wrong_shape = base;
        --wrong_shape.tensors[i].dimensions[1];
        reject(wrong_shape, "shape/rank");
        auto wrong_rank = base;
        wrong_rank.tensors[i].dimensions.push_back(1);
        reject(wrong_rank, "shape/rank");
        auto wrong_type = base;
        wrong_type.tensors[i].type = i == 0 ? 8U : 2U;
        reject(wrong_type, i == 0 ? "embedding must be Q4_0" : "output must be Q6_K");
    }
    for (const auto& spec : base.metadata) {
        auto missing = base;
        missing.erase(spec.key);
        reject(missing, "missing metadata");
        auto wrong_type = base;
        wrong_type.scalar<std::uint64_t>(spec.key, 10, 1);
        reject(wrong_type, "wrong metadata type");
        if (spec.type == 4 && !spec.key.starts_with("tokenizer.")) {
            auto wrong_value = base;
            wrong_value.scalar<std::uint32_t>(spec.key, 4, 0);
            reject(wrong_value, "unexpected metadata value");
        }
    }
    auto wrong_blocks = base;
    wrong_blocks.scalar<std::uint32_t>("qwen4exp.block_count", 4, 49);
    reject(wrong_blocks, "unexpected metadata value");
    auto wrong_hidden = base;
    wrong_hidden.scalar<std::uint32_t>("qwen4exp.embedding_length", 4, 10240);
    reject(wrong_hidden, "unexpected metadata value");
    auto wrong_arch = base;
    wrong_arch.string("general.architecture", "qwen35moe");
    reject(wrong_arch, "architecture");

    // Plausible alternate geometries also fail even though both borrowed
    // matrices retain valid, compatible shapes/types and parser-valid ranges.
    for (const auto& [key, value] : std::initializer_list<std::pair<const char*, std::uint32_t>>{
             {"hyper_connection.count", 2}, {"hyper_connection.low_rank", 640},
             {"attention.head_count", 32}, {"attention.head_count_kv", 4},
             {"attention.key_length", 128}, {"attention.value_length", 128},
             {"expert_count", 256}, {"expert_used_count", 8},
             {"expert_feed_forward_length", 768}, {"expert_shared_feed_forward_length", 768},
             {"rope.dimension_count", 32}, {"context_length", 131072}}) {
        auto alternate = base;
        alternate.scalar<std::uint32_t>("qwen4exp." + std::string(key), 4, value);
        reject(alternate, "unexpected metadata value");
    }

    for (const auto key : {"qwen4exp.rope.freq_base", "qwen4exp.attention.layer_norm_rms_epsilon"}) {
        auto wrong_value = base;
        const auto value = key == std::string_view("qwen4exp.rope.freq_base") ? 1.0e7F : 1.0e-6F;
        wrong_value.scalar<float>(key, 6, std::bit_cast<float>(std::bit_cast<std::uint32_t>(value) + 1));
        reject(wrong_value, "unexpected metadata value");
    }
    for (const auto& values : {std::vector<std::int32_t>{11, 11, 10},
                              std::vector<std::int32_t>{11, 11, 10, -1}}) {
        auto b = base;
        b.array<std::int32_t>("qwen4exp.rope.dimension_sections", 5, values);
        reject(b, "unexpected metadata value");
    }
    auto wrong_elements = base;
    wrong_elements.array<std::uint32_t>("qwen4exp.rope.dimension_sections", 4, {11, 11, 10, 0});
    reject(wrong_elements, "array element type");
    constexpr auto compression_key = "qwen4exp.attention.compress_ratios";
    wrong_elements = base;
    wrong_elements.array<std::uint32_t>(compression_key, 4, std::vector<std::uint32_t>(48, 0));
    reject(wrong_elements, "array element type");
    for (const std::size_t size : {0U, 47U, 49U}) {
        auto b = base;
        b.array<std::int32_t>(compression_key, 5, std::vector<std::int32_t>(size, 0));
        reject(b, "metadata length");
    }
    for (const std::size_t index : {0U, 24U, 47U}) {
        auto b = base;
        std::vector<std::int32_t> ratios(48, 0);
        ratios[index] = INT32_MAX;
        b.array<std::int32_t>(compression_key, 5, ratios);
        reject(b, "unexpected metadata value");
    }

    // The same parser guards also apply to borrowed target inventories.
    auto overlap = base.build();
    patch<std::uint64_t>(overlap, overlap.tensors[1].offset, 0);
    TempFile overlap_file(overlap);
    expect_failure([&] { qwen::Model target(overlap_file.path()); }, "overlapping");
    TempFile file(base.build());
    qwen::Model first(file.path());
    qwen::Model moved(std::move(first));
    expect_failure([&] { const qwen::MtpModel mtp(sidecar_file.path(), first); }, "missing metadata");
    const qwen::MtpModel mtp(sidecar_file.path(), moved);
    check(&mtp.shared_target().embedding == &moved.tensor("token_embd.weight"),
          "borrowing the live moved-to owner is valid");
}

} // namespace

int main() {
    try {
        test_valid_descriptor();
        TempFile target_file(target_builder().build());
        const qwen::Model target(target_file.path());
        test_shapes_types_and_inventory(target);
        test_metadata(target);
        test_compression(target);
        test_tokenizer_assets(target);
        test_tokenizer_bit_patterns();
        test_parser_guards(target);
        test_borrowed_target_rejections();
        std::cout << "mtp_model_test: passed (" << checks << " checks, " << rejected_cases
                  << " rejected cases; LOCAL synthetic loader proof only)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mtp_model_test: FAILED: " << error.what() << '\n';
        return 1;
    }
}
