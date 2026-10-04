#include "model.hpp"
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

int main(int argc,char **argv) {
    try {
        if(argc!=2) throw std::runtime_error("usage: core-inspect model.gguf");
        qwen::Model model(argv[1]);
        if(model.metadata_value("general.architecture").get<std::string>()!="qwen4exp")
            throw std::runtime_error("expected qwen4exp architecture");
        std::map<qwen::TensorType,int> counts;
        for(const auto &tensor:model.tensors()) ++counts[tensor.type];
        std::cout<<"GGUF v3 qwen4exp: "<<model.tensors().size()<<" tensors; "
                 <<model.metadata().size()<<" metadata keys; "<<model.file_size()<<" file bytes; data offset "<<model.data_offset()<<'\n';
        for(auto [type,count]:counts) std::cout<<qwen::type_name(type)<<": "<<count<<'\n';
        for(const auto& [key,value]:model.metadata()) {
            if(key.starts_with("qwen4exp.rope.") || key=="qwen4exp.attention.layer_norm_rms_epsilon") {
                std::cout<<key<<" (metadata type "<<static_cast<unsigned>(value.type)<<"): ";
                std::visit([](const auto& v) {
                    using T=std::decay_t<decltype(v)>;
                    if constexpr(std::is_same_v<T,qwen::MetadataArray>) {
                        std::cout<<"element type "<<static_cast<unsigned>(v.element_type)<<"; ";
                        std::visit([](const auto& a) { for(const auto& x:a) {
                            if constexpr(std::is_same_v<std::decay_t<decltype(x)>,std::string>) std::cout<<x<<' ';
                            else std::cout<<+x<<' ';
                        } },v.values);
                    } else if constexpr(std::is_same_v<T,std::string>) std::cout<<v;
                    else std::cout<<+v;
                },value.value);
                std::cout<<'\n';
            }
        }
        bool sidecar=false;
        if(const auto *value=model.find_metadata("qwen4exp.nextn_shared_target_tensors")) sidecar=value->get<bool>();
        if(sidecar) std::cout<<"Sidecar requires shared target embedding/output\n";
        const std::string prefix=sidecar?"blk.48.":"blk.0.";
        for(const char *suffix:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"}) {
            const auto &tensor=model.tensor(prefix+suffix);
            std::cout<<tensor.name<<": "<<qwen::type_name(tensor.type)<<" ["<<tensor.dimensions[0]<<','
                     <<tensor.dimensions[1]<<','<<tensor.dimensions[2]<<"] expert bytes "<<tensor.strides[2]<<'\n';
            std::vector<std::byte> bytes(tensor.strides[2]);
            model.read_expert(tensor.name,0,bytes);
        }
        if(sidecar) {
            // Bounded complete inventory for the known 32-tensor shared sidecar.
            constexpr std::size_t max_tensors=64;
            constexpr std::size_t max_name=160;
            std::size_t printed=0;
            for(const auto &tensor:model.tensors()) {
                if(printed==max_tensors) break;
                ++printed;
                std::cout<<tensor.name.substr(0,max_name)<<": "<<qwen::type_name(tensor.type)<<" [";
                for(std::uint32_t axis=0;axis<tensor.rank;++axis) std::cout<<(axis?",":"")<<tensor.dimensions[axis];
                std::cout<<"] bytes "<<tensor.byte_size<<" file offset "<<tensor.file_offset<<'\n';
            }
            if(printed<model.tensors().size())
                std::cout<<"sidecar inventory omitted "<<model.tensors().size()-printed<<" tensors\n";
        } else {
            for(const auto &tensor:model.tensors()) {
                // Root embedding/PLE table/LM head and selected GDN, PLE and QSA
                // blocks: actual inventory, not a guessed universal schema.
                if(!tensor.name.starts_with("blk.") || tensor.name.starts_with("blk.0.")
                    || tensor.name.starts_with("blk.1.") || tensor.name.starts_with("blk.3.")) {
                    std::cout<<tensor.name<<": "<<qwen::type_name(tensor.type)<<" [";
                    for(std::uint32_t axis=0;axis<tensor.rank;++axis) std::cout<<(axis?",":"")<<tensor.dimensions[axis];
                    std::cout<<"] bytes "<<tensor.byte_size<<'\n';
                }
            }
        }
        std::cout<<"inventory and first expert reads passed\n";
    } catch(const std::exception &error) { std::cerr<<"core-inspect: "<<error.what()<<'\n'; return 1; }
}
