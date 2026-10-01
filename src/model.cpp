#include "model.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <type_traits>
#include <unistd.h>
#include <utility>

namespace qwen {
namespace {

[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("GGUF: " + message);
}

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) invalid("integer overflow");
    return a + b;
}

std::uint64_t multiply(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        invalid("integer overflow");
    return a * b;
}

void check_range(std::uint64_t offset, std::uint64_t size, std::uint64_t limit) {
    if (offset > limit || size > limit - offset) invalid("byte range outside file/tensor");
}

void pread_exact(int fd, std::uint64_t offset, std::span<std::byte> destination,
                 std::uint64_t file_size) {
    if (fd < 0) invalid("read from moved-from model");
    check_range(offset, destination.size(), file_size);
    while (!destination.empty()) {
        constexpr std::size_t max_chunk = 16U << 20;
        const auto count = std::min(destination.size(), max_chunk);
        const auto result = ::pread(fd, destination.data(), count, static_cast<off_t>(offset));
        if (result < 0) {
            if (errno == EINTR) continue;
            throw std::system_error(errno, std::generic_category(), "pread GGUF");
        }
        if (result == 0) invalid("incomplete read (file truncated)");
        const auto received = static_cast<std::size_t>(result);
        offset += received;
        destination = destination.subspan(received);
    }
}

// Only a bounded header buffer is read eagerly. Tensor data is never mapped.
class Reader {
public:
    Reader(int fd, std::uint64_t size) : fd_(fd), size_(size) {}
    [[nodiscard]] std::uint64_t position() const { return position_; }
    [[nodiscard]] std::uint64_t remaining() const { return size_ - position_; }

    void require(std::uint64_t bytes) const {
        if (bytes > remaining()) invalid("truncated header or invalid string/array length");
    }

    void account(std::uint64_t bytes) {
        if (bytes > Model::max_inventory_bytes - inventory_bytes_)
            invalid("metadata/inventory exceeds 256 MiB parser limit");
        inventory_bytes_ += bytes;
    }

    void bytes(std::span<std::byte> destination) {
        require(destination.size());
        while (!destination.empty()) {
            if (buffer_used_ == buffer_size_) {
                buffer_size_ = static_cast<std::size_t>(std::min<std::uint64_t>(
                    buffer_.size(), remaining()));
                pread_exact(fd_, position_, std::span(buffer_).first(buffer_size_), size_);
                buffer_used_ = 0;
            }
            const auto count = std::min(destination.size(), buffer_size_ - buffer_used_);
            std::memcpy(destination.data(), buffer_.data() + buffer_used_, count);
            destination = destination.subspan(count);
            buffer_used_ += count;
            position_ += count;
        }
    }

    template<class T> T scalar() {
        if constexpr (std::is_same_v<T, bool>) {
            const auto value = scalar<std::uint8_t>();
            if (value > 1) invalid("invalid boolean value");
            return value != 0;
        } else if constexpr (std::is_floating_point_v<T>) {
            using Bits = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
            return std::bit_cast<T>(scalar<Bits>());
        } else {
            std::array<std::byte, sizeof(T)> raw{};
            bytes(raw);
            std::uint64_t bits = 0;
            for (std::size_t i = 0; i < raw.size(); ++i)
                bits |= std::uint64_t(std::to_integer<unsigned char>(raw[i])) << (8 * i);
            using Unsigned = std::make_unsigned_t<T>;
            return std::bit_cast<T>(static_cast<Unsigned>(bits));
        }
    }

    std::string string() {
        const auto length = scalar<std::uint64_t>();
        require(length); // Check before allocating, including on sparse files.
        account(length);
        if (length > std::string{}.max_size()) invalid("string too large");
        std::string result(static_cast<std::size_t>(length), '\0');
        bytes(std::as_writable_bytes(std::span(result.data(), result.size())));
        return result;
    }

private:
    int fd_;
    std::uint64_t size_;
    std::uint64_t position_ = 0;
    std::uint64_t inventory_bytes_ = 0;
    std::array<std::byte, 64U << 10> buffer_{};
    std::size_t buffer_used_ = 0;
    std::size_t buffer_size_ = 0;
};

MetadataType metadata_type(Reader& reader) {
    const auto id = reader.scalar<std::uint32_t>();
    if (id > static_cast<std::uint32_t>(MetadataType::FLOAT64))
        invalid("unknown metadata type " + std::to_string(id));
    return static_cast<MetadataType>(id);
}

template<class T> std::vector<T> array(Reader& reader, std::uint64_t count) {
    constexpr auto disk_size = std::is_same_v<T, std::string> ? 8 : sizeof(T);
    reader.require(multiply(count, disk_size));
    reader.account(multiply(count, sizeof(T)));
    std::vector<T> result;
    if (count > result.max_size()) invalid("array too large");
    result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        if constexpr (std::is_same_v<T, std::string>) result.push_back(reader.string());
        else result.push_back(reader.scalar<T>());
    }
    return result;
}

MetadataValue parse_metadata(Reader& reader) {
    const auto type = metadata_type(reader);
    switch (type) {
    case MetadataType::UINT8: return {type, reader.scalar<std::uint8_t>()};
    case MetadataType::INT8: return {type, reader.scalar<std::int8_t>()};
    case MetadataType::UINT16: return {type, reader.scalar<std::uint16_t>()};
    case MetadataType::INT16: return {type, reader.scalar<std::int16_t>()};
    case MetadataType::UINT32: return {type, reader.scalar<std::uint32_t>()};
    case MetadataType::INT32: return {type, reader.scalar<std::int32_t>()};
    case MetadataType::FLOAT32: return {type, reader.scalar<float>()};
    case MetadataType::BOOL: return {type, reader.scalar<bool>()};
    case MetadataType::STRING: return {type, reader.string()};
    case MetadataType::UINT64: return {type, reader.scalar<std::uint64_t>()};
    case MetadataType::INT64: return {type, reader.scalar<std::int64_t>()};
    case MetadataType::FLOAT64: return {type, reader.scalar<double>()};
    case MetadataType::ARRAY: {
        const auto element_type = metadata_type(reader);
        if (element_type == MetadataType::ARRAY) invalid("nested metadata arrays are unsupported");
        const auto count = reader.scalar<std::uint64_t>();
        MetadataArray result{element_type, std::vector<std::uint8_t>{}};
        switch (element_type) {
        case MetadataType::UINT8: result.values = array<std::uint8_t>(reader, count); break;
        case MetadataType::INT8: result.values = array<std::int8_t>(reader, count); break;
        case MetadataType::UINT16: result.values = array<std::uint16_t>(reader, count); break;
        case MetadataType::INT16: result.values = array<std::int16_t>(reader, count); break;
        case MetadataType::UINT32: result.values = array<std::uint32_t>(reader, count); break;
        case MetadataType::INT32: result.values = array<std::int32_t>(reader, count); break;
        case MetadataType::FLOAT32: result.values = array<float>(reader, count); break;
        case MetadataType::BOOL: result.values = array<bool>(reader, count); break;
        case MetadataType::STRING: result.values = array<std::string>(reader, count); break;
        case MetadataType::UINT64: result.values = array<std::uint64_t>(reader, count); break;
        case MetadataType::INT64: result.values = array<std::int64_t>(reader, count); break;
        case MetadataType::FLOAT64: result.values = array<double>(reader, count); break;
        case MetadataType::ARRAY: invalid("nested metadata arrays are unsupported");
        }
        return {type, std::move(result)};
    }
    }
    invalid("unknown metadata type");
}

} // namespace

TypeLayout type_layout(TensorType type) {
    switch (type) {
    case TensorType::F32: return {1, 4};
    case TensorType::F16: case TensorType::BF16: return {1, 2};
    case TensorType::Q4_0: return {32, 18};
    case TensorType::Q4_1: return {32, 20};
    case TensorType::Q5_0: return {32, 22};
    case TensorType::Q8_0: return {32, 34};
    case TensorType::Q6_K: return {256, 210};
    }
    invalid("unsupported tensor type " + std::to_string(static_cast<std::uint32_t>(type)));
}

std::string_view type_name(TensorType type) {
    switch (type) {
    case TensorType::F32: return "F32";
    case TensorType::F16: return "F16";
    case TensorType::BF16: return "BF16";
    case TensorType::Q4_0: return "Q4_0";
    case TensorType::Q4_1: return "Q4_1";
    case TensorType::Q5_0: return "Q5_0";
    case TensorType::Q8_0: return "Q8_0";
    case TensorType::Q6_K: return "Q6_K";
    }
    invalid("unsupported tensor type " + std::to_string(static_cast<std::uint32_t>(type)));
}

Model::Model(const std::filesystem::path& path) {
    static_assert(sizeof(off_t) >= 8, "GGUF loader requires 64-bit file offsets");
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::system_error(errno, std::generic_category(), "open GGUF " + path.string());
    try {
        struct stat status{};
        if (::fstat(fd_, &status) != 0)
            throw std::system_error(errno, std::generic_category(), "stat GGUF");
        if (!S_ISREG(status.st_mode) || status.st_size < 0) invalid("expected a regular file");
        file_size_ = static_cast<std::uint64_t>(status.st_size);
        Reader reader(fd_, file_size_);
        if (reader.scalar<std::uint32_t>() != 0x46554747U) invalid("bad magic (expected GGUF)");
        if (reader.scalar<std::uint32_t>() != 3) invalid("only little-endian GGUF v3 is supported");
        const auto tensor_count = reader.scalar<std::uint64_t>();
        const auto metadata_count = reader.scalar<std::uint64_t>();
        // Smallest possible records: rank-1 tensor 32 bytes, scalar KV 13 bytes.
        reader.require(add(multiply(tensor_count, 32), multiply(metadata_count, 13)));
        reader.account(multiply(tensor_count, sizeof(TensorView) + sizeof(std::string) + sizeof(std::size_t)));
        reader.account(multiply(metadata_count, sizeof(Metadata::value_type)));
        for (std::uint64_t i = 0; i < metadata_count; ++i) {
            auto key = reader.string();
            if (key.empty()) invalid("empty metadata key");
            if (metadata_.contains(key)) invalid("duplicate metadata key: " + key);
            auto value = parse_metadata(reader);
            metadata_.emplace(std::move(key), std::move(value));
        }
        if (const auto* value = find_metadata("general.alignment")) {
            if (value->type != MetadataType::UINT32) invalid("general.alignment must be UINT32");
            alignment_ = value->get<std::uint32_t>();
        }
        if (alignment_ == 0 || alignment_ > max_alignment || !std::has_single_bit(alignment_))
            invalid("alignment must be a power of two in [1, 1048576]");

        for (std::uint64_t i = 0; i < tensor_count; ++i) {
            TensorView view{};
            view.name = reader.string();
            if (view.name.empty()) invalid("empty tensor name");
            if (tensor_indices_.contains(view.name)) invalid("duplicate tensor name: " + view.name);
            view.rank = reader.scalar<std::uint32_t>();
            if (view.rank < 1 || view.rank > 4) invalid("tensor rank must be 1..4: " + view.name);
            view.elements = 1;
            for (std::uint32_t axis = 0; axis < view.rank; ++axis) {
                view.dimensions[axis] = reader.scalar<std::uint64_t>();
                if (view.dimensions[axis] == 0) invalid("zero tensor dimension: " + view.name);
                view.elements = multiply(view.elements, view.dimensions[axis]);
            }
            view.type = static_cast<TensorType>(reader.scalar<std::uint32_t>());
            const auto layout = type_layout(view.type);
            if (view.dimensions[0] % layout.block_elements != 0)
                invalid("axis 0 is not divisible by quantization block: " + view.name);
            view.strides[0] = layout.block_bytes;
            auto size = multiply(view.dimensions[0] / layout.block_elements, layout.block_bytes);
            for (std::uint32_t axis = 1; axis < view.rank; ++axis) {
                view.strides[axis] = size;
                size = multiply(size, view.dimensions[axis]);
            }
            view.byte_size = size;
            view.relative_offset = reader.scalar<std::uint64_t>();
            if (view.relative_offset % alignment_ != 0) invalid("unaligned tensor offset: " + view.name);
            tensor_indices_.emplace(view.name, tensors_.size());
            tensors_.push_back(std::move(view));
        }
        data_offset_ = add(reader.position(), alignment_ - 1) & ~std::uint64_t(alignment_ - 1);
        check_range(data_offset_, 0, file_size_);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        ranges.reserve(tensors_.size());
        for (auto& view : tensors_) {
            view.file_offset = add(data_offset_, view.relative_offset);
            check_range(view.file_offset, view.byte_size, file_size_);
            ranges.emplace_back(view.file_offset, add(view.file_offset, view.byte_size));
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t i = 1; i < ranges.size(); ++i)
            if (ranges[i].first < ranges[i - 1].second) invalid("overlapping tensor payloads");
    } catch (...) {
        ::close(std::exchange(fd_, -1));
        throw;
    }
}

Model::~Model() {
    if (fd_ >= 0) ::close(fd_);
}

Model::Model(Model&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), file_size_(std::exchange(other.file_size_, 0)),
      data_offset_(std::exchange(other.data_offset_, 0)), alignment_(other.alignment_),
      metadata_(std::move(other.metadata_)), tensors_(std::move(other.tensors_)),
      tensor_indices_(std::move(other.tensor_indices_)) {
    other.metadata_.clear();
    other.tensors_.clear();
    other.tensor_indices_.clear();
}

Model& Model::operator=(Model&& other) noexcept {
    if (this != &other) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = std::exchange(other.fd_, -1);
        file_size_ = std::exchange(other.file_size_, 0);
        data_offset_ = std::exchange(other.data_offset_, 0);
        alignment_ = other.alignment_;
        metadata_ = std::move(other.metadata_);
        tensors_ = std::move(other.tensors_);
        tensor_indices_ = std::move(other.tensor_indices_);
        other.metadata_.clear();
        other.tensors_.clear();
        other.tensor_indices_.clear();
    }
    return *this;
}

const MetadataValue* Model::find_metadata(std::string_view key) const {
    const auto it = metadata_.find(key);
    return it == metadata_.end() ? nullptr : &it->second;
}

const MetadataValue& Model::metadata_value(std::string_view key) const {
    const auto* result = find_metadata(key);
    if (!result) invalid("missing metadata key: " + std::string(key));
    return *result;
}

const TensorView* Model::find_tensor(std::string_view name) const {
    const auto it = tensor_indices_.find(name);
    return it == tensor_indices_.end() ? nullptr : &tensors_[it->second];
}

const TensorView& Model::tensor(std::string_view name) const {
    const auto* result = find_tensor(name);
    if (!result) invalid("missing tensor: " + std::string(name));
    return *result;
}

void Model::read_tensor(std::string_view name, std::span<std::byte> destination) const {
    const auto& view = tensor(name);
    if (destination.size() != view.byte_size) invalid("tensor destination size mismatch: " + view.name);
    read_slice(name, 0, destination);
}

void Model::read_slice(std::string_view name, std::uint64_t byte_offset,
                       std::span<std::byte> destination) const {
    const auto& view = tensor(name);
    check_range(byte_offset, destination.size(), view.byte_size);
    pread_exact(fd_, add(view.file_offset, byte_offset), destination, file_size_);
}

void Model::read_expert(std::string_view name, std::uint64_t expert,
                        std::span<std::byte> destination) const {
    const auto& view = tensor(name);
    if (view.rank != 3) invalid("expert tensor must have rank 3: " + view.name);
    if (expert >= view.dimensions[2]) invalid("expert index outside tensor: " + view.name);
    if (destination.size() != view.strides[2]) invalid("expert destination size mismatch: " + view.name);
    read_slice(name, multiply(expert, view.strides[2]), destination);
}

} // namespace qwen
