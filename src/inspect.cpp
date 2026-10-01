#include "model.hpp"
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

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
        if(!sidecar) {
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
