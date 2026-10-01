#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace qwen {

// On-disk GGML type IDs, not a new quantization format.
enum class TensorType : std::uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6,
    Q8_0 = 8, Q6_K = 14, BF16 = 30,
};

struct TypeLayout {
    std::uint32_t block_elements;
    std::uint32_t block_bytes;
};

[[nodiscard]] TypeLayout type_layout(TensorType type);
[[nodiscard]] std::string_view type_name(TensorType type);

enum class MetadataType : std::uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10,
    INT64 = 11, FLOAT64 = 12,
};

struct MetadataArray {
    MetadataType element_type;
    using Values = std::variant<
        std::vector<std::uint8_t>, std::vector<std::int8_t>,
        std::vector<std::uint16_t>, std::vector<std::int16_t>,
        std::vector<std::uint32_t>, std::vector<std::int32_t>,
        std::vector<float>, std::vector<bool>, std::vector<std::string>,
        std::vector<std::uint64_t>, std::vector<std::int64_t>, std::vector<double>>;
    Values values;

    template<class T> [[nodiscard]] const std::vector<T>& get() const {
        return std::get<std::vector<T>>(values);
    }
};

struct MetadataValue {
    MetadataType type;
    using Value = std::variant<
        std::uint8_t, std::int8_t, std::uint16_t, std::int16_t,
        std::uint32_t, std::int32_t, float, bool, std::string,
        std::uint64_t, std::int64_t, double, MetadataArray>;
    Value value;

    template<class T> [[nodiscard]] const T& get() const {
        return std::get<T>(value);
    }
};

// Describes bytes in the file; it owns no tensor payload or mapped memory.
struct TensorView {
    std::string name;
    TensorType type;
    std::uint32_t rank;
    std::array<std::uint64_t, 4> dimensions{}; // GGUF: axis 0 is fastest.
    // stride[0] advances one quantization block (one element for F32/F16/BF16).
    // For present axes, stride[1] is row bytes; higher strides are byte strides.
    std::array<std::uint64_t, 4> strides{};
    std::uint64_t elements;
    std::uint64_t byte_size;
    std::uint64_t relative_offset; // Relative to Model::data_offset().
    std::uint64_t file_offset;
};

class Model {
public:
    using Metadata = std::map<std::string, MetadataValue, std::less<>>;
    // Explicit parser limits: ample room for tokenizer assets, not tensor data.
    static constexpr std::uint64_t max_inventory_bytes = 256ULL << 20;
    static constexpr std::uint32_t max_alignment = 1U << 20;

    explicit Model(const std::filesystem::path& path);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&& other) noexcept;
    Model& operator=(Model&& other) noexcept;

    [[nodiscard]] const Metadata& metadata() const noexcept { return metadata_; }
    [[nodiscard]] const MetadataValue* find_metadata(std::string_view key) const;
    [[nodiscard]] const MetadataValue& metadata_value(std::string_view key) const;
    [[nodiscard]] const std::vector<TensorView>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const TensorView* find_tensor(std::string_view name) const;
    [[nodiscard]] const TensorView& tensor(std::string_view name) const;
    [[nodiscard]] std::uint64_t file_size() const noexcept { return file_size_; }
    [[nodiscard]] std::uint64_t data_offset() const noexcept { return data_offset_; }
    [[nodiscard]] std::uint32_t alignment() const noexcept { return alignment_; }

    // All reads fill caller-owned memory using pread. No implicit dequantization.
    // A full tensor/expert requires an exactly sized destination; byte slices
    // may be unaligned but must lie entirely inside that tensor.
    void read_tensor(std::string_view name, std::span<std::byte> destination) const;
    void read_slice(std::string_view name, std::uint64_t byte_offset,
                    std::span<std::byte> destination) const;
    // Routed expert convention: rank 3, dimensions [input, output, expert].
    void read_expert(std::string_view name, std::uint64_t expert,
                     std::span<std::byte> destination) const;

private:
    int fd_ = -1;
    std::uint64_t file_size_ = 0;
    std::uint64_t data_offset_ = 0;
    std::uint32_t alignment_ = 32;
    Metadata metadata_;
    std::vector<TensorView> tensors_;
    std::map<std::string, std::size_t, std::less<>> tensor_indices_;
};

} // namespace qwen
