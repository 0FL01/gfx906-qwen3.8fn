#include "model.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
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
        check(std::string_view(error.what()).find(message) != std::string_view::npos,
              "unexpected rejection reason");
        ++rejected_cases;
        return;
    }
    throw std::runtime_error("malformed file/read unexpectedly accepted");
}

// Independent fixture writer: literal GGUF IDs and block sizes below are the
// test oracle. No production loader helper is used to serialize or size files.
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

    void string(const std::string& value) {
        number<std::uint64_t>(value.size());
        const auto bytes = std::as_bytes(std::span(value.data(), value.size()));
        data.insert(data.end(), bytes.begin(), bytes.end());
    }

    void append(const Bytes& bytes) { data.insert(data.end(), bytes.begin(), bytes.end()); }
};

struct TensorSpec {
    std::string name;
    std::uint32_t type;
    std::vector<std::uint64_t> dimensions;
    std::uint64_t offset;
    Bytes payload;
};

struct TensorFields {
    std::size_t name_length;
    std::size_t rank;
    std::vector<std::size_t> dimensions;
    std::size_t type;
    std::size_t offset;
};

struct Image {
    Bytes bytes;
    std::map<std::string, std::size_t> metadata_values;
    std::map<std::string, std::size_t> metadata_keys;
    std::map<std::string, std::size_t> metadata_types;
    std::vector<TensorFields> tensors;
    std::size_t data_offset;
};

struct Builder {
    Writer metadata;
    std::uint64_t metadata_count = 0;
    std::map<std::string, std::size_t> metadata_values;
    std::map<std::string, std::size_t> metadata_keys;
    std::map<std::string, std::size_t> metadata_types;
    std::vector<TensorSpec> tensors;

    void key(const std::string& name, std::uint32_t type) {
        metadata_keys.emplace(name, 24 + metadata.data.size());
        metadata.string(name);
        metadata_types.emplace(name, 24 + metadata.data.size());
        metadata.number(type);
        metadata_values.emplace(name, 24 + metadata.data.size());
        ++metadata_count;
    }

    template<class T> void scalar(const std::string& name, std::uint32_t type, T value) {
        key(name, type);
        metadata.number(value);
    }

    void string(const std::string& name, const std::string& value) {
        key(name, 8);
        metadata.string(value);
    }

    template<class T> void array(const std::string& name, std::uint32_t type,
                                const std::vector<T>& values) {
        key(name, 9);
        metadata.number(type);
        metadata.number<std::uint64_t>(values.size());
        for (const auto& value : values) {
            if constexpr (std::is_same_v<T, std::string>) metadata.string(value);
            else metadata.number<T>(value);
        }
    }

    Image build(std::uint32_t alignment = 32) const {
        Writer writer;
        writer.number<std::uint32_t>(0x46554747);
        writer.number<std::uint32_t>(3);
        writer.number<std::uint64_t>(tensors.size());
        writer.number(metadata_count);
        writer.append(metadata.data);
        Image image{{}, metadata_values, metadata_keys, metadata_types, {}, 0};
        for (const auto& tensor : tensors) {
            TensorFields fields{writer.data.size(), 0, {}, 0, 0};
            writer.string(tensor.name);
            fields.rank = writer.data.size();
            writer.number<std::uint32_t>(tensor.dimensions.size());
            for (const auto dimension : tensor.dimensions) {
                fields.dimensions.push_back(writer.data.size());
                writer.number(dimension);
            }
            fields.type = writer.data.size();
            writer.number(tensor.type);
            fields.offset = writer.data.size();
            writer.number(tensor.offset);
            image.tensors.push_back(std::move(fields));
        }
        while (writer.data.size() % alignment != 0) writer.data.push_back(std::byte{0});
        image.data_offset = writer.data.size();
        for (const auto& tensor : tensors) {
            const auto start = image.data_offset + static_cast<std::size_t>(tensor.offset);
            writer.data.resize(std::max(writer.data.size(), start + tensor.payload.size()));
            std::copy(tensor.payload.begin(), tensor.payload.end(), writer.data.begin() + start);
        }
        image.bytes = std::move(writer.data);
        return image;
    }
};

Bytes pattern(std::size_t count, unsigned seed) {
    Bytes result(count);
    for (std::size_t i = 0; i < count; ++i) result[i] = std::byte((seed + i * 17) & 255);
    return result;
}

template<class T> void patch(Image& image, std::size_t offset, T value) {
    Writer writer;
    writer.number(value);
    check(offset + writer.data.size() <= image.bytes.size(), "fixture patch out of bounds");
    std::copy(writer.data.begin(), writer.data.end(), image.bytes.begin() + offset);
}

class TempFile {
public:
    explicit TempFile(const Bytes& bytes) {
        char name[] = "/tmp/qwen-gguf-test-XXXXXX";
        fd_ = ::mkstemp(name);
        if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "mkstemp");
        path_ = name;
        try {
            std::span remaining(bytes);
            while (!remaining.empty()) {
                const auto result = ::write(fd_, remaining.data(), remaining.size());
                if (result < 0 && errno == EINTR) continue;
                if (result <= 0) throw std::runtime_error("fixture write failed");
                remaining = remaining.subspan(static_cast<std::size_t>(result));
            }
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
        if (::ftruncate(fd_, static_cast<off_t>(size)) != 0)
            throw std::system_error(errno, std::generic_category(), "fixture truncate");
    }

private:
    int fd_;
    std::string path_;
};

Builder valid_builder() {
    Builder builder;
    builder.string("general.architecture", "qwen4exp");
    builder.scalar<std::uint32_t>("tokenizer.ggml.bos_token_id", 4, 248044);
    builder.scalar<std::uint32_t>("tokenizer.ggml.eos_token_id", 4, 248046);
    builder.scalar<std::uint32_t>("ple.eos_token_id", 4, 248044);
    builder.array<std::uint32_t>("ple.layers", 4, {1});
    builder.array<std::uint64_t>("ple.offsets", 10, {0, 20000003, 40000084});
    builder.array<std::string>("tokenizer.ggml.tokens", 8, {"a", "\xCE\xB2", "", std::string("a\0b", 3)});
    builder.scalar<bool>("nextn_shared_target_tensors", 7, true);
    builder.array<float>("tokenizer.ggml.scores", 6, {-1.5F, 0.0F, 2.25F});
    Writer floats;
    for (const float value : {1.0F, -2.0F, 0.5F, 0.0F, 8.0F, -4.0F}) floats.number(value);
    builder.tensors = {
        {"dense", 0, {3, 2}, 0, floats.data},
        {"blk.0.ffn_gate_exps.weight", 2, {32, 2, 3}, 32, pattern(108, 11)},
        {"blk.0.ffn_down_exps.weight", 3, {32, 3, 2}, 160, pattern(120, 73)},
    };
    return builder;
}

void reject(const Image& image, std::string_view message) {
    TempFile file(image.bytes);
    expect_failure([&] { qwen::Model model(file.path()); }, message);
}

void test_valid_inventory_and_reads() {
    const auto builder = valid_builder();
    const auto image = builder.build();
    TempFile file(image.bytes);
    qwen::Model model(file.path());
    check(model.alignment() == 32, "default alignment");
    check(model.file_size() == image.bytes.size(), "file size");
    check(model.data_offset() == image.data_offset, "aligned tensor data start");
    check(model.tensors().size() == 3, "tensor inventory count");
    check(model.metadata_value("general.architecture").get<std::string>() == "qwen4exp", "architecture");
    check(model.metadata_value("ple.eos_token_id").get<std::uint32_t>() == 248044, "PLE EOS");
    check(model.metadata_value("tokenizer.ggml.eos_token_id").get<std::uint32_t>() == 248046, "tokenizer EOS");
    check(model.metadata_value("ple.layers").get<qwen::MetadataArray>().get<std::uint32_t>() ==
          std::vector<std::uint32_t>{1}, "PLE layers");
    check(model.metadata_value("ple.offsets").get<qwen::MetadataArray>().get<std::uint64_t>() ==
          std::vector<std::uint64_t>({0, 20000003, 40000084}), "PLE offsets");
    check(model.metadata_value("tokenizer.ggml.tokens").get<qwen::MetadataArray>().get<std::string>() ==
          std::vector<std::string>({"a", "\xCE\xB2", "", std::string("a\0b", 3)}), "tokenizer byte strings");
    check(model.metadata_value("tokenizer.ggml.scores").get<qwen::MetadataArray>().get<float>() ==
          std::vector<float>({-1.5F, 0.0F, 2.25F}), "floating metadata array");
    check(model.metadata_value("nextn_shared_target_tensors").get<bool>(), "boolean metadata");
    check(model.find_tensor("missing") == nullptr, "missing tensor lookup");
    check(model.find_metadata("missing") == nullptr, "missing metadata lookup");

    const auto& gate = model.tensor("blk.0.ffn_gate_exps.weight");
    check(gate.type == qwen::TensorType::Q4_0 && gate.rank == 3, "gate type/rank");
    check(gate.dimensions == std::array<std::uint64_t, 4>{32, 2, 3, 0}, "fastest-first dimensions");
    check(gate.strides == std::array<std::uint64_t, 4>{18, 18, 36, 0}, "Q4 block and expert strides");
    check(gate.elements == 192 && gate.byte_size == 108, "quantized tensor size");
    check(gate.relative_offset == 32 && gate.file_offset == image.data_offset + 32, "relative/absolute offset");
    for (const auto& spec : builder.tensors) {
        Bytes destination(spec.payload.size());
        model.read_tensor(spec.name, destination);
        check(destination == spec.payload, "full tensor payload");
    }
    Bytes expert(36);
    model.read_expert(gate.name, 1, expert);
    check(expert == Bytes(builder.tensors[1].payload.begin() + 36,
                          builder.tensors[1].payload.begin() + 72), "explicit Q4_0 expert slice");
    Bytes down(60);
    model.read_expert("blk.0.ffn_down_exps.weight", 1, down);
    check(down == Bytes(builder.tensors[2].payload.begin() + 60,
                        builder.tensors[2].payload.end()), "explicit Q4_1 expert slice");
    Bytes slice(7);
    model.read_slice(gate.name, 19, slice);
    check(slice == Bytes(builder.tensors[1].payload.begin() + 19,
                         builder.tensors[1].payload.begin() + 26), "unaligned byte slice");
    model.read_slice(gate.name, gate.byte_size, {});
}

template<class T> void add_and_check_metadata(Builder& builder, std::uint32_t id,
                                            T scalar, std::vector<T> array) {
    const auto name = "type." + std::to_string(id);
    if constexpr (std::is_same_v<T, std::string>) builder.string(name, scalar);
    else builder.scalar(name, id, scalar);
    builder.array(name + ".array", id, array);
    TempFile file(builder.build().bytes);
    qwen::Model model(file.path());
    const auto& value = model.metadata_value(name);
    check(static_cast<std::uint32_t>(value.type) == id && value.get<T>() == scalar, "typed scalar metadata");
    const auto& result = model.metadata_value(name + ".array").get<qwen::MetadataArray>();
    check(static_cast<std::uint32_t>(result.element_type) == id && result.get<T>() == array, "typed array metadata");
}

void test_all_metadata_types() {
    Builder builder;
    add_and_check_metadata<std::uint8_t>(builder, 0, 255, {0, 255});
    add_and_check_metadata<std::int8_t>(builder, 1, -128, {-128, 127});
    add_and_check_metadata<std::uint16_t>(builder, 2, 65535, {0, 65535});
    add_and_check_metadata<std::int16_t>(builder, 3, -32768, {-32768, 32767});
    add_and_check_metadata<std::uint32_t>(builder, 4, 4000000085U, {0, 4000000085U});
    add_and_check_metadata<std::int32_t>(builder, 5, -2147483647 - 1, {-2147483647 - 1, 2147483647});
    add_and_check_metadata<float>(builder, 6, -1.25F, {0.5F, -2.5F});
    add_and_check_metadata<bool>(builder, 7, true, {false, true, false});
    add_and_check_metadata<std::string>(builder, 8, "qwen4exp", {"x", "", "y"});
    add_and_check_metadata<std::uint64_t>(builder, 10, UINT64_MAX, {0, UINT64_MAX});
    add_and_check_metadata<std::int64_t>(builder, 11, INT64_MIN, {INT64_MIN, INT64_MAX});
    add_and_check_metadata<double>(builder, 12, -1.234567890123, {0.5, -2.5});
    builder.array<std::int32_t>("empty", 5, {});
    TempFile file(builder.build().bytes);
    qwen::Model model(file.path());
    check(model.tensors().empty(), "metadata-only GGUF");
    check(model.metadata_value("empty").get<qwen::MetadataArray>().get<std::int32_t>().empty(), "typed empty array");
}

void test_all_tensor_types_and_alignments() {
    static_assert(static_cast<std::uint32_t>(qwen::TensorType::F32) == 0);
    static_assert(static_cast<std::uint32_t>(qwen::TensorType::F16) == 1);
    static_assert(static_cast<std::uint32_t>(qwen::TensorType::Q4_0) == 2);
    static_assert(static_cast<std::uint32_t>(qwen::TensorType::Q4_1) == 3);
    static_assert(static_cast<std::uint32_t>(qwen::TensorType::Q5_0) == 6);
    static_assert(static_cast<std::uint32_t>(qwen::TensorType::Q8_0) == 8);
    static_assert(static_cast<std::uint32_t>(qwen::TensorType::Q6_K) == 14);
    static_assert(static_cast<std::uint32_t>(qwen::TensorType::BF16) == 30);
    struct Layout { std::uint32_t id; std::uint64_t block; std::size_t bytes; };
    constexpr Layout layouts[] = {{0, 1, 4}, {1, 1, 2}, {2, 32, 18}, {3, 32, 20},
                                  {6, 32, 22}, {8, 32, 34}, {14, 256, 210}, {30, 1, 2}};
    Builder builder;
    std::uint64_t offset = 0;
    for (const auto layout : layouts) {
        builder.tensors.push_back({"tensor." + std::to_string(layout.id), layout.id,
                                   {layout.block * 2, 2, 1, 3}, offset,
                                   pattern(layout.bytes * 12, layout.id)});
        offset += ((layout.bytes * 12 + 31) / 32) * 32;
    }
    TempFile file(builder.build().bytes);
    qwen::Model model(file.path());
    for (std::size_t i = 0; i < std::size(layouts); ++i) {
        const auto& tensor = model.tensors()[i];
        check(tensor.rank == 4 && tensor.byte_size == layouts[i].bytes * 12, "rank-4 type bytes");
        check(tensor.strides[3] == layouts[i].bytes * 4, "rank-4 stride");
        check(!qwen::type_name(tensor.type).empty(), "type name");
        const auto layout = qwen::type_layout(tensor.type);
        check(layout.block_elements == layouts[i].block && layout.block_bytes == layouts[i].bytes, "block layout");
        Bytes result(tensor.byte_size);
        model.read_tensor(tensor.name, result);
        check(result == builder.tensors[i].payload, "all type payload reads");
    }
    for (const std::uint32_t alignment : {1U, 2U, 4U, 8U, 64U, 4096U}) {
        Builder aligned;
        aligned.scalar<std::uint32_t>("general.alignment", 4, alignment);
        // Unsorted descriptors, adjacent payloads at alignment=1/2/4.
        const std::uint64_t next = std::max(alignment, 4U);
        aligned.tensors = {{"second", 0, {1}, next, pattern(4, 8)},
                           {"first", 0, {1}, 0, pattern(4, 2)}};
        const auto image = aligned.build(alignment);
        TempFile aligned_file(image.bytes);
        qwen::Model aligned_model(aligned_file.path());
        check(aligned_model.alignment() == alignment, "explicit alignment");
        check(aligned_model.data_offset() == image.data_offset, "explicit aligned data start");
        check(aligned_model.tensors()[0].name == "second", "descriptor order preserved");
    }
}

void test_malformed_files() {
    const auto base = valid_builder().build();
    const auto mutate = [&](auto function, std::string_view message) {
        auto image = base;
        function(image);
        reject(image, message);
    };
    mutate([](Image& i) { patch<std::uint32_t>(i, 0, 0); }, "magic");
    mutate([](Image& i) { patch<std::uint32_t>(i, 4, 2); }, "v3");
    mutate([](Image& i) { patch<std::uint64_t>(i, 8, UINT64_MAX); }, "overflow");
    mutate([](Image& i) { patch<std::uint64_t>(i, 16, UINT64_MAX); }, "overflow");
    mutate([](Image& i) { patch<std::uint64_t>(i, 8, 1000); }, "truncated");
    for (const auto length : {0U, 3U, 7U, 23U})
        mutate([&](Image& i) { i.bytes.resize(length); }, "truncated");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.metadata_keys.at("general.architecture"), UINT64_MAX); }, "length");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.metadata_values.at("general.architecture"), UINT64_MAX); }, "length");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.metadata_values.at("ple.offsets") + 4, UINT64_MAX); }, "overflow");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.metadata_values.at("ple.offsets") + 4, 1000); }, "length");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.metadata_values.at("tokenizer.ggml.tokens") + 12, UINT64_MAX); }, "length");
    mutate([](Image& i) { patch<std::uint32_t>(i, i.metadata_values.at("ple.layers"), 9); }, "nested");
    mutate([](Image& i) { patch<std::uint32_t>(i, i.metadata_values.at("ple.layers"), 99); }, "metadata type");
    mutate([](Image& i) { patch<std::uint32_t>(i, i.metadata_types.at("general.architecture"), 99); }, "metadata type");
    mutate([](Image& i) { i.bytes[i.metadata_values.at("nextn_shared_target_tensors")] = std::byte{2}; }, "boolean");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[0].name_length, UINT64_MAX); }, "length");
    for (const auto rank : {0U, 5U})
        mutate([&](Image& i) { patch<std::uint32_t>(i, i.tensors[0].rank, rank); }, "rank");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[0].dimensions[0], 0); }, "zero tensor");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[0].dimensions[0], UINT64_MAX); }, "overflow");
    mutate([](Image& i) {
        patch<std::uint64_t>(i, i.tensors[0].dimensions[0], UINT64_MAX / 4 + 1);
        patch<std::uint64_t>(i, i.tensors[0].dimensions[1], 1);
    }, "overflow");
    // Total elements divisible by 32 is insufficient: each axis-0 row must be.
    mutate([](Image& i) {
        patch<std::uint64_t>(i, i.tensors[1].dimensions[0], 16);
        patch<std::uint64_t>(i, i.tensors[1].dimensions[1], 4);
    }, "axis 0");
    mutate([](Image& i) { patch<std::uint32_t>(i, i.tensors[0].type, 99); }, "tensor type");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[1].offset, 1); }, "unaligned");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[1].offset, 0); }, "overlapping");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[2].offset, 96); }, "overlapping");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[2].offset, UINT64_MAX - 31); }, "overflow");
    mutate([](Image& i) { patch<std::uint64_t>(i, i.tensors[2].offset, 1ULL << 40); }, "outside");
    mutate([](Image& i) { i.bytes.pop_back(); }, "outside");

    auto duplicate_metadata = valid_builder();
    duplicate_metadata.string("general.architecture", "duplicate");
    reject(duplicate_metadata.build(), "duplicate metadata");
    auto duplicate_tensor = valid_builder();
    duplicate_tensor.tensors[1].name = "dense";
    reject(duplicate_tensor.build(), "duplicate tensor");
    auto empty_metadata = valid_builder();
    empty_metadata.string("", "empty");
    reject(empty_metadata.build(), "empty metadata");
    auto empty_tensor = valid_builder();
    empty_tensor.tensors[0].name.clear();
    reject(empty_tensor.build(), "empty tensor");
    for (const auto alignment : {0U, 3U, 1U << 21}) {
        auto builder = valid_builder();
        builder.scalar<std::uint32_t>("general.alignment", 4, alignment);
        reject(builder.build(), "alignment must");
    }
    auto wrong_alignment_type = valid_builder();
    wrong_alignment_type.scalar<std::uint64_t>("general.alignment", 10, 32);
    reject(wrong_alignment_type.build(), "must be UINT32");
    Builder padding;
    padding.string("x", "y");
    auto missing_padding = padding.build();
    missing_padding.bytes.resize(46); // Complete metadata, but no aligned data start.
    reject(missing_padding, "outside");
}

void test_read_failures_and_move() {
    const auto image = valid_builder().build();
    TempFile file(image.bytes);
    qwen::Model first(file.path());
    qwen::Model moved(std::move(first));
    check(moved.tensor("dense").byte_size == 24, "move construction inventory");
    check(first.tensors().empty() && first.metadata().empty() && first.find_tensor("dense") == nullptr,
          "move construction clears source inventory");
    qwen::Model model(file.path());
    model = std::move(moved);
    check(model.metadata_value("general.architecture").get<std::string>() == "qwen4exp", "move assignment metadata");
    check(moved.tensors().empty() && moved.metadata().empty() && moved.find_tensor("dense") == nullptr,
          "move assignment clears source inventory");
    Bytes destination(24);
    model.read_tensor("dense", destination);
    check(destination == valid_builder().tensors[0].payload, "moved fd remains usable");
    const std::string gate = "blk.0.ffn_gate_exps.weight";
    expect_failure([&] { model.read_tensor("missing", destination); }, "missing tensor");
    expect_failure([&] { (void)model.metadata_value("missing"); }, "missing metadata");
    expect_failure([&] { model.read_tensor(gate, destination); }, "size mismatch");
    expect_failure([&] { model.read_slice("dense", 1, destination); }, "outside");
    expect_failure([&] { model.read_slice("dense", UINT64_MAX, {}); }, "outside");
    expect_failure([&] { model.read_slice("dense", 25, {}); }, "outside");
    expect_failure([&] { model.read_expert("dense", 0, destination); }, "rank 3");
    expect_failure([&] { model.read_expert(gate, 3, destination); }, "expert index");
    expect_failure([&] { model.read_expert(gate, UINT64_MAX, destination); }, "expert index");
    expect_failure([&] { model.read_expert(gate, 0, destination); }, "size mismatch");
    expect_failure([&] { (void)qwen::type_layout(static_cast<qwen::TensorType>(99)); }, "tensor type");
    expect_failure([&] { (void)qwen::type_name(static_cast<qwen::TensorType>(99)); }, "tensor type");
    file.truncate(model.tensor("dense").file_offset + 8);
    expect_failure([&] { model.read_tensor("dense", destination); }, "incomplete read");
}

} // namespace

int main() {
    try {
        test_valid_inventory_and_reads();
        test_all_metadata_types();
        test_all_tensor_types_and_alignments();
        test_malformed_files();
        test_read_failures_and_move();
        std::cout << "model_test: passed (" << checks << " checks, " << rejected_cases
                  << " rejected cases)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model_test: FAILED: " << error.what() << '\n';
        return 1;
    }
}
