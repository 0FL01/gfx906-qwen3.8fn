#include "mtp_model.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>

#ifndef CORE_REVISION
#error "core-mtp-model requires compiled 40-hex CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-mtp-model requires compiled CORE_DIRTY (0 or 1)"
#endif

namespace {

using qwen::MetadataType;
using qwen::TensorType;
using qwen::TensorView;

constexpr bool revision_valid(std::string_view revision) {
    if (revision.size() != 40) return false;
    for (const char c : revision)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
            return false;
    return true;
}
static_assert(revision_valid(CORE_REVISION));
static_assert(CORE_DIRTY == 0 || CORE_DIRTY == 1);

void require(bool condition, std::string_view message) {
    if (!condition) throw std::runtime_error(std::string(message));
}

std::uint64_t add(std::uint64_t a, std::uint64_t b) {
    require(b <= std::numeric_limits<std::uint64_t>::max() - a, "descriptor total overflow");
    return a + b;
}

// Accept only shortest-form UTF-8 Unicode scalar sequences. Invalid bytes in an
// error message are escaped individually; argv paths must pass this check first
// so their JSON strings preserve the original path without byte substitution.
std::size_t utf8_width(std::string_view text, std::size_t i) {
    const auto first = static_cast<unsigned char>(text[i]);
    if (first < 0x80) return 1;
    const std::size_t width = first >= 0xc2 && first <= 0xdf ? 2 :
        first >= 0xe0 && first <= 0xef ? 3 : first >= 0xf0 && first <= 0xf4 ? 4 : 0;
    if (width == 0 || width > text.size() - i) return 0;
    for (std::size_t j = 1; j < width; ++j) {
        const auto byte = static_cast<unsigned char>(text[i + j]);
        if (byte < 0x80 || byte > 0xbf) return 0;
    }
    const auto second = static_cast<unsigned char>(text[i + 1]);
    if ((first == 0xe0 && second < 0xa0) || (first == 0xed && second > 0x9f) ||
        (first == 0xf0 && second < 0x90) || (first == 0xf4 && second > 0x8f)) return 0;
    return width;
}

void quote(std::ostream& out, std::string_view text) {
    constexpr char hex[] = "0123456789abcdef";
    out << '"';
    for (std::size_t i = 0; i < text.size();) {
        const auto byte = static_cast<unsigned char>(text[i]);
        const auto width = utf8_width(text, i);
        if (byte < 32 || width == 0) {
            out << "\\u00" << hex[byte >> 4] << hex[byte & 15];
            ++i;
        } else if (byte == '"' || byte == '\\') {
            out << '\\' << text[i++];
        } else {
            out.write(text.data() + i, static_cast<std::streamsize>(width));
            i += width;
        }
    }
    out << '"';
}

struct Paths {
    std::string_view target;
    std::string_view sidecar;
};

Paths arguments(int argc, char** argv) {
    constexpr std::string_view usage = "usage: core-mtp-model TARGET.gguf SIDECAR.gguf";
    require(argc == 3, usage);
    for (int i = 1; i < argc; ++i) {
        require(argv[i] != nullptr, usage);
        const std::string_view path(argv[i]);
        require(!path.empty() && path.size() <= 4096 && path.front() != '-',
                "paths must be nonempty positional arguments of at most 4096 bytes (use ./ for a leading '-')");
        for (std::size_t j = 0; j < path.size();) {
            const auto width = utf8_width(path, j);
            require(width != 0, "paths must be valid UTF-8");
            j += width;
        }
    }
    return {argv[1], argv[2]};
}

struct Writer {
    std::size_t records = 0;
    std::uint64_t bytes = 0;

    template<class Function> void emit(std::string_view kind, Function fields) {
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << std::boolalpha << std::setprecision(std::numeric_limits<float>::max_digits10)
            << "{\"kind\":";
        quote(out, kind);
        out << ",\"protocol\":1";
        fields(out);
        out << "}\n";
        require(static_cast<bool>(out), "JSONL formatting failed");
        const auto line = out.str();
        require(line.size() <= (64U << 10), "JSONL row exceeds 64 KiB bound");
        const auto next_bytes = add(bytes, line.size());
        require(next_bytes <= (128U << 10), "JSONL output exceeds 128 KiB bound");
        std::cout << line;
        std::cout.flush();
        require(static_cast<bool>(std::cout), "JSONL output failed");
        bytes = next_bytes;
        ++records;
    }
};

std::string_view metadata_type_name(MetadataType type) {
    switch (type) {
    case MetadataType::UINT32: return "UINT32";
    case MetadataType::INT32: return "INT32";
    case MetadataType::BOOL: return "BOOL";
    case MetadataType::STRING: return "STRING";
    case MetadataType::ARRAY: return "ARRAY";
    default: throw std::runtime_error("unexpected reported metadata type");
    }
}

void type_fields(std::ostream& out, MetadataType type) {
    out << "\"type\":";
    quote(out, metadata_type_name(type));
    out << ",\"type_id\":" << static_cast<std::uint32_t>(type);
}

const qwen::MetadataValue& metadata(const qwen::Model& model, std::string_view key, MetadataType type) {
    const auto& value = model.metadata_value(key);
    require(value.type == type, "unexpected reported metadata type");
    return value;
}

template<class T> void scalar(std::ostream& out, const qwen::Model& model,
                             std::string_view key, MetadataType type) {
    out << '{';
    type_fields(out, type);
    out << ",\"value\":" << metadata(model, key, type).get<T>() << '}';
}

void file_fields(std::ostream& out, const qwen::Model& model, std::string_view path,
                 const qwen::MtpGeometry& geometry, bool sidecar) {
    out << "{\"path\":";
    quote(out, path);
    out << ",\"file_bytes\":" << model.file_size() << ",\"data_offset\":" << model.data_offset()
        << ",\"alignment\":" << model.alignment() << ",\"tensor_count\":" << model.tensors().size()
        << ",\"metadata_key_count\":" << model.metadata().size() << ",\"architecture\":{";
    type_fields(out, MetadataType::STRING);
    out << ",\"value\":";
    quote(out, metadata(model, "general.architecture", MetadataType::STRING).get<std::string>());
    out << "},\"context_length\":";
    scalar<std::uint32_t>(out, model, "qwen4exp.context_length", MetadataType::UINT32);
    out << ",\"block_count\":";
    scalar<std::uint32_t>(out, model, "qwen4exp.block_count", MetadataType::UINT32);
    out << ",\"expert_used_count\":";
    scalar<std::uint32_t>(out, model, "qwen4exp.expert_used_count", MetadataType::UINT32);
    const auto& ratios = metadata(model, "qwen4exp.attention.compress_ratios", MetadataType::ARRAY)
                             .get<qwen::MetadataArray>();
    require(ratios.element_type == MetadataType::INT32, "unexpected compression element type");
    const auto& values = ratios.get<std::int32_t>();
    out << ",\"compression\":{\"key\":\"qwen4exp.attention.compress_ratios\",";
    type_fields(out, MetadataType::ARRAY);
    out << ",\"element_type\":\"INT32\",\"element_type_id\":"
        << static_cast<std::uint32_t>(ratios.element_type) << ",\"count\":" << values.size();
    if (sidecar) {
        out << ",\"entry\":" << geometry.layer << ",\"entry_value\":" << values.at(geometry.layer);
    }
    out << '}';
    if (sidecar) {
        out << ",\"nextn_predict_layers\":";
        scalar<std::uint32_t>(out, model, "qwen4exp.nextn_predict_layers", MetadataType::UINT32);
        out << ",\"nextn_shared_target_tensors\":";
        scalar<bool>(out, model, "qwen4exp.nextn_shared_target_tensors", MetadataType::BOOL);
    }
    out << '}';
}

void tensor_fields(std::ostream& out, const TensorView& view) {
    out << "{\"name\":";
    quote(out, view.name);
    out << ",\"type\":";
    quote(out, qwen::type_name(view.type));
    out << ",\"type_id\":" << static_cast<std::uint32_t>(view.type) << ",\"rank\":" << view.rank
        << ",\"dimensions\":[";
    for (std::uint32_t i = 0; i < view.rank; ++i) out << (i ? "," : "") << view.dimensions[i];
    out << "],\"strides\":[";
    for (std::uint32_t i = 0; i < view.rank; ++i) out << (i ? "," : "") << view.strides[i];
    out << "],\"elements\":" << view.elements << ",\"bytes\":" << view.byte_size
        << ",\"relative_offset\":" << view.relative_offset << ",\"file_offset\":" << view.file_offset << '}';
}

struct Binding {
    std::string_view role;
    const TensorView* view;
    TensorType type;
};

std::array<Binding, 32> bindings(const qwen::MtpWeights& w) {
    return {{
        {"embedding_norm", &w.embedding_norm, TensorType::F32},
        {"hidden_norm", &w.hidden_norm, TensorType::F32},
        {"embedding_hidden_projection", &w.embedding_hidden_projection, TensorType::Q8_0},
        {"head.norm", &w.head.norm, TensorType::F32},
        {"head.down", &w.head.down, TensorType::Q8_0},
        {"head.up", &w.head.up, TensorType::Q8_0},
        {"attention_hc.mixer.norm", &w.attention_hc.mixer.norm, TensorType::F32},
        {"attention_hc.mixer.down", &w.attention_hc.mixer.down, TensorType::Q8_0},
        {"attention_hc.mixer.up", &w.attention_hc.mixer.up, TensorType::Q8_0},
        {"attention_hc.inject", &w.attention_hc.inject, TensorType::Q8_0},
        {"ffn_hc.mixer.norm", &w.ffn_hc.mixer.norm, TensorType::F32},
        {"ffn_hc.mixer.down", &w.ffn_hc.mixer.down, TensorType::Q8_0},
        {"ffn_hc.mixer.up", &w.ffn_hc.mixer.up, TensorType::Q8_0},
        {"ffn_hc.inject", &w.ffn_hc.inject, TensorType::Q8_0},
        {"attention.query", &w.attention.query, TensorType::Q8_0},
        {"attention.key", &w.attention.key, TensorType::Q8_0},
        {"attention.value", &w.attention.value, TensorType::Q8_0},
        {"attention.output", &w.attention.output, TensorType::Q8_0},
        {"attention.query_norm", &w.attention.query_norm, TensorType::F32},
        {"attention.key_norm", &w.attention.key_norm, TensorType::F32},
        {"experts.router", &w.experts.router, TensorType::F32},
        {"experts.shared_router", &w.experts.shared_router, TensorType::F32},
        {"experts.gate", &w.experts.gate, TensorType::Q8_0},
        {"experts.up", &w.experts.up, TensorType::Q8_0},
        {"experts.down", &w.experts.down, TensorType::Q8_0},
        {"experts.shared_gate", &w.experts.shared_gate, TensorType::Q8_0},
        {"experts.shared_up", &w.experts.shared_up, TensorType::Q8_0},
        {"experts.shared_down", &w.experts.shared_down, TensorType::Q8_0},
        {"indexer.query", &w.indexer.query, TensorType::BF16},
        {"indexer.key", &w.indexer.key, TensorType::BF16},
        {"indexer.query_norm", &w.indexer.query_norm, TensorType::F32},
        {"indexer.key_norm", &w.indexer.key_norm, TensorType::F32}
    }};
}

struct Summary {
    qwen::MtpTypeCounts counts{};
    std::size_t inventory_rows = 0;
    std::uint64_t payload_bytes = 0;
    std::array<std::uint64_t, 3> expert_strides{};
};

Summary validate_bindings(const qwen::MtpModel& mtp, const std::array<Binding, 32>& roles) {
    const auto& target = mtp.target();
    const auto& shared = mtp.shared_target();
    require(&shared.embedding == &target.tensor("token_embd.weight") &&
            &shared.output == &target.tensor("output.weight") && &shared.embedding != &shared.output,
            "borrowed target view identity mismatch");
    for (const auto* view : {&shared.embedding, &shared.output})
        require(view->rank == 2 && view->dimensions[0] == 2560 && view->dimensions[1] == 248320,
                "borrowed target shape mismatch");
    require(shared.embedding.type == TensorType::Q4_0 && shared.output.type == TensorType::Q6_K,
            "borrowed target dtype mismatch");
    require(add(shared.embedding.file_offset, shared.embedding.byte_size) <= shared.output.file_offset ||
            add(shared.output.file_offset, shared.output.byte_size) <= shared.embedding.file_offset,
            "borrowed target ranges overlap");
    require(mtp.sidecar().tensors().size() == roles.size(), "sidecar descriptor count mismatch");
    for (std::size_t i = 0; i < roles.size(); ++i) {
        const auto& role = roles[i];
        require(role.view == &mtp.sidecar().tensor(role.view->name) && role.view->type == role.type,
                "original sidecar role identity/dtype mismatch");
        for (std::size_t j = 0; j < i; ++j)
            require(role.view != roles[j].view, "duplicate sidecar role binding");
    }
    Summary result;
    result.counts = mtp.type_counts();
    require(result.counts.q8_0 == 19 && result.counts.f32 == 11 && result.counts.bf16 == 2,
            "sidecar aggregate dtype mismatch");
    for (const auto& view : mtp.sidecar().tensors()) {
        require(std::count_if(roles.begin(), roles.end(), [&](const Binding& role) {
            return role.view == &view;
        }) == 1, "unbound sidecar descriptor");
        result.payload_bytes = add(result.payload_bytes, view.byte_size);
    }
    const auto& experts = mtp.weights().experts;
    result.expert_strides = {experts.gate.strides[2], experts.up.strides[2], experts.down.strides[2]};
    const auto requirements = mtp.validation_requirements();
    require(requirements.per_role_type_inventory && requirements.dense_compression_metadata &&
            requirements.attention_expert_metadata && requirements.target_tokenizer_equivalence,
            "required constructor validation disabled");
    return result;
}

void source(Writer& writer, const Paths& paths, const qwen::MtpModel& mtp) {
    writer.emit("mtp_model_source", [&](std::ostream& out) {
        out << ",\"revision\":";
        quote(out, CORE_REVISION);
        out << ",\"dirty\":" << (CORE_DIRTY != 0)
            << ",\"scope\":\"parsed_metadata_and_descriptors\",\"target\":";
        file_fields(out, mtp.target(), paths.target, mtp.geometry(), false);
        out << ",\"sidecar\":";
        file_fields(out, mtp.sidecar(), paths.sidecar, mtp.geometry(), true);
        const auto& g = mtp.geometry();
        out << ",\"geometry\":{\"layer\":" << g.layer << ",\"block_count\":" << g.block_count
            << ",\"predict_layers\":" << g.predict_layers << ",\"hidden_size\":" << g.hidden_size
            << ",\"hc_count\":" << g.hc_count << ",\"hc_rank\":" << g.hc_rank
            << ",\"widened_size\":" << g.widened_size << ",\"query_heads\":" << g.query_heads
            << ",\"kv_heads\":" << g.kv_heads << ",\"head_dim\":" << g.head_dim
            << ",\"experts\":" << g.experts << ",\"intermediate_size\":" << g.intermediate_size
            << ",\"vocabulary_size\":" << g.vocabulary_size << ",\"rotary_dim\":" << g.rotary_dim
            << ",\"rope_base\":" << g.rope_base << ",\"rms_epsilon\":" << g.rms_epsilon
            << ",\"rope_sections\":[";
        for (std::size_t i = 0; i < g.rope_sections.size(); ++i)
            out << (i ? "," : "") << g.rope_sections[i];
        out << "]},\"required_validation_status\":{\"per_role_type_inventory\":\"passed\","
               "\"dense_compression_metadata\":\"passed\",\"attention_expert_metadata\":\"passed\","
               "\"target_tokenizer_equivalence\":\"passed\"}"
               ",\"expected_records\":{\"source\":1,\"tensor\":32,\"borrowed\":1,\"tokenizer\":1,\"complete\":1,\"total\":36}"
               ",\"axis_order\":\"GGUF_fastest_first\",\"stride_unit\":\"bytes; axis 0 advances one dtype block\"";
    });
}

void tokenizer(Writer& writer, const qwen::MtpModel& mtp) {
    // Report the ten mandatory assets, never their large array/template values.
    // Constructor equality also covers every additional tokenizer.* key, its
    // presence and typed full contents; key_count reports that complete set.
    constexpr std::array<std::string_view, 10> keys{
        "tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges",
        "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.add_bos_token", "tokenizer.chat_template"};
    const auto& model = mtp.sidecar();
    const auto key_count = std::count_if(model.metadata().begin(), model.metadata().end(), [](const auto& pair) {
        return pair.first.starts_with("tokenizer.");
    });
    writer.emit("mtp_model_tokenizer", [&](std::ostream& out) {
        out << ",\"status\":\"constructor_checked_full_typed_equality\",\"key_count\":" << key_count
            << ",\"reported_key_count\":" << keys.size() << ",\"assets\":[";
        for (std::size_t i = 0; i < keys.size(); ++i) {
            const auto& value = model.metadata_value(keys[i]);
            out << (i ? "," : "") << "{\"key\":";
            quote(out, keys[i]);
            out << ',';
            type_fields(out, value.type);
            switch (value.type) {
            case MetadataType::ARRAY: {
                const auto& array = value.get<qwen::MetadataArray>();
                out << ",\"element_type\":";
                quote(out, metadata_type_name(array.element_type));
                out << ",\"element_type_id\":" << static_cast<std::uint32_t>(array.element_type)
                    << ",\"count\":" << std::visit([](const auto& values) { return values.size(); }, array.values);
                break;
            }
            case MetadataType::STRING:
                out << ",\"count\":1,\"bytes\":" << value.get<std::string>().size();
                break;
            case MetadataType::UINT32:
                out << ",\"count\":1,\"value\":" << value.get<std::uint32_t>();
                break;
            case MetadataType::BOOL:
                out << ",\"count\":1,\"value\":" << value.get<bool>();
                break;
            default: throw std::runtime_error("unexpected required tokenizer asset type");
            }
            out << '}';
        }
        out << ']'; // The writer owns the outer row object.
    });
}

Summary inspect(Writer& writer, const Paths& paths) {
    // Target precedes the borrower and remains stationary. Both owners unwind
    // before the caller emits completion; only value-only totals escape here.
    const qwen::Model target{std::filesystem::path(paths.target)};
    const qwen::MtpModel mtp{std::filesystem::path(paths.sidecar), target};
    require(&mtp.target() == &target, "borrowed target Model identity mismatch");
    const auto roles = bindings(mtp.weights());
    auto summary = validate_bindings(mtp, roles);
    source(writer, paths, mtp);
    for (const auto& view : mtp.sidecar().tensors()) {
        const auto role = std::find_if(roles.begin(), roles.end(), [&](const Binding& item) {
            return item.view == &view;
        });
        writer.emit("mtp_model_tensor", [&](std::ostream& out) {
            out << ",\"ordinal\":" << summary.inventory_rows << ",\"role\":";
            quote(out, role->role);
            out << ",\"original_view_identity\":true,\"tensor\":";
            tensor_fields(out, view);
        });
        ++summary.inventory_rows;
    }
    writer.emit("mtp_model_borrowed", [&](std::ostream& out) {
        out << ",\"owner\":\"target\",\"target_model_identity\":true,\"original_view_identities\":true"
               ",\"distinct_views\":true,\"distinct_nonoverlapping_ranges\":true,\"embedding\":";
        tensor_fields(out, mtp.shared_target().embedding);
        out << ",\"output\":";
        tensor_fields(out, mtp.shared_target().output);
    });
    tokenizer(writer, mtp);
    return summary;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto paths = arguments(argc, argv); // Validate BOTH arguments before either Model.
        Writer writer;
        const auto summary = inspect(writer, paths); // MtpModel then target have now unwound.
        require(summary.inventory_rows == 32 && writer.records == 35, "incomplete metadata inventory output");
        writer.emit("mtp_model_complete", [&](std::ostream& out) {
            out << ",\"complete\":true,\"passed\":true,\"owners_unwound\":true"
                   ",\"scope\":\"parsed_metadata_and_descriptors\",\"inventory_rows\":" << summary.inventory_rows
                << ",\"type_counts\":{\"Q8_0\":" << summary.counts.q8_0 << ",\"F32\":" << summary.counts.f32
                << ",\"BF16\":" << summary.counts.bf16 << "},\"sidecar_payload_bytes\":" << summary.payload_bytes
                << ",\"expert_stride_bytes\":{\"gate\":" << summary.expert_strides[0]
                << ",\"up\":" << summary.expert_strides[1] << ",\"down\":" << summary.expert_strides[2]
                << "},\"records\":" << writer.records + 1
                << ",\"gpu_execution\":false,\"trained_mtp_execution\":false,\"performance_claim\":false,\"R6_complete_claim\":false";
        });
        return 0;
    } catch (const std::exception& error) {
        constexpr std::size_t error_bytes = 768;
        const std::string_view message(error.what());
        std::cerr << "core-mtp-model: ";
        quote(std::cerr, message.substr(0, error_bytes));
        if (message.size() > error_bytes) std::cerr << " (truncated)";
        std::cerr << '\n';
        return 1;
    }
}
