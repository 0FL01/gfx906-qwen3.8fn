#include "mtp_runner.hpp"
#include "session_cli.hpp"
#include <chrono>
#include <charconv>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>
#ifndef CORE_REVISION
#define CORE_REVISION "unknown"
#endif
#ifndef CORE_DIRTY
#define CORE_DIRTY 1
#endif
namespace {
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void json_string(std::string_view s) {
    constexpr char hex[]="0123456789abcdef";std::cout<<'"';
    for(unsigned char c:s) {
        if(c=='"' || c=='\\')std::cout<<'\\'<<char(c);
        else if(c<32)std::cout<<"\\u00"<<hex[c>>4]<<hex[c&15];
        else std::cout<<char(c);
    }
    std::cout<<'"';
}
void usage(){std::cout<<"core-mtp-run TARGET SIDECAR --sample --seed UINT64 --generate N "
    "[--capacity N --slots N --prefill-chunk N --layerwise-prefill N --prefill-pipeline N --prefill-residual-device --attention-query-tile N --ignore-eos "
    "--temperature T --top-p P --top-k K] token-id...\n"
    "Opt-in trained MTP2 token-ID runner; no tokenizer, HTTP or performance qualification.\n";}
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string_view(argv[1])=="--help"){usage();return 0;}
        if(argc<4){usage();return 2;}
        const std::string sidecar=argv[2];
        if(sidecar.empty())throw std::invalid_argument("sidecar required");
        std::vector<const char*> args{argv[0],argv[1]};
        int layerwise=0,pipeline=0;bool seen_layerwise=false,seen_pipeline=false,resident=false;
        for(int i=3;i<argc;++i) {
            if(std::string_view(argv[i])=="--layerwise-prefill") {
                if(seen_layerwise || ++i==argc)throw std::invalid_argument("one layerwise capacity required");
                seen_layerwise=true;const std::string_view value(argv[i]);
                const auto parsed=std::from_chars(value.data(),value.data()+value.size(),layerwise);
                if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || layerwise<0 || layerwise>16384)
                    throw std::invalid_argument("layerwise capacity must be 0..16384");
            } else if(std::string_view(argv[i])=="--prefill-pipeline") {
                if(seen_pipeline || ++i==argc)throw std::invalid_argument("one pipeline subwindow required");
                seen_pipeline=true;const std::string_view value(argv[i]);
                const auto parsed=std::from_chars(value.data(),value.data()+value.size(),pipeline);
                if(parsed.ec!=std::errc{} || parsed.ptr!=value.data()+value.size() || pipeline<0 || pipeline>4096)
                    throw std::invalid_argument("pipeline subwindow must be 0..4096");
            } else if(std::string_view(argv[i])=="--prefill-residual-device") {
                if(resident)throw std::invalid_argument("duplicate device residual option");
                resident=true;
            } else args.push_back(argv[i]);
        }
        auto o=qwen::session_cli::parse(int(args.size()),args.data());
        o.config.layerwise_prefill_capacity=layerwise;o.config.prefill_pipeline_tokens=pipeline;
        o.config.prefill_residual_device=resident;
        if(o.help){usage();return 0;}
        if(!o.sample || o.generate<1 || o.config.cpu_workers || o.hybrid_policy.mode!=qwen::SessionHybridMode::disabled ||
           !o.config.trace_directory.empty() || !o.logits_path.empty())
            throw std::invalid_argument("MTP runner requires explicit sampling; CPU hybrid/trace/logit export unsupported");
        std::cout<<std::setprecision(17);
        std::cout<<"{\"kind\":\"mtp_request_source\",\"revision\":\""<<CORE_REVISION
            <<"\",\"dirty\":"<<(CORE_DIRTY?"true":"false")<<",\"model\":";json_string(o.model);std::cout<<",\"sidecar\":";json_string(sidecar);
        std::cout<<",\"capacity\":"<<o.config.capacity<<",\"slots\":"<<o.config.expert_slots
            <<",\"prefill_chunk\":"<<o.config.max_batch_tokens<<",\"attention_query_tile\":"<<o.config.attention_query_tile
            <<",\"layerwise_prefill_capacity\":"<<o.config.layerwise_prefill_capacity
            <<",\"prefill_pipeline_tokens\":"<<o.config.prefill_pipeline_tokens
            <<",\"prefill_residual_device\":"<<(resident?"true":"false")
            <<",\"seed\":"<<o.sampling.seed<<",\"correction_seed\":"<<(o.sampling.seed^0x9e3779b97f4a7c15ULL)
            <<",\"temperature\":"<<o.sampling.temperature<<",\"top_p\":"<<o.sampling.top_p<<",\"top_k\":"<<o.sampling.top_k
            <<",\"requested_outputs\":"<<o.generate<<",\"ignore_eos\":"<<(o.ignore_eos?"true":"false")
            <<",\"prompt_tokens\":"<<o.tokens.size()<<",\"performance_qualified\":false}\n";
        if(!std::cout)throw std::runtime_error("source write failed");
        const auto load_start=Clock::now();
        qwen::MtpRunner runner(o.model,sidecar,o.config,o.sampling);
        const auto load_ms=elapsed(load_start);
        const auto request_start=Clock::now();
        auto emission=runner.begin(o.tokens,o.generate,o.ignore_eos);
        std::size_t outputs=0;std::int32_t pending=0;
        for(;;) {
            for(std::size_t i=0;i<emission.count;++i) {
                pending=emission.ids[i];
                std::cout<<"{\"kind\":\"mtp_output\",\"index\":"<<outputs++<<",\"token\":"<<pending<<"}\n";
            }
            if(!std::cout)throw std::runtime_error("output write failed");
            if(emission.finished)break;
            emission=runner.next();
        }
        const auto request_ms=elapsed(request_start);const auto s=runner.stats();
        if(s.outputs!=outputs || s.consumed!=o.tokens.size()+outputs-1)throw std::runtime_error("final pending/output contract");
        const auto copies=runner.layerwise_transfers();
        const char* stop=emission.reason==qwen::SpeculativeStopReason::eos?"eos":"output_budget";
        std::cout<<"{\"kind\":\"mtp_request_complete\",\"actual_outputs\":"<<outputs<<",\"consumed\":"<<s.consumed
            <<",\"pending\":"<<pending<<",\"stop\":\""<<stop<<"\",\"windows\":"<<s.windows
            <<",\"accepted\":["<<s.accepted[0]<<','<<s.accepted[1]<<','<<s.accepted[2]
            <<"],\"proposal_draws\":"<<s.proposal_draws<<",\"decision_draws\":"<<s.decision_draws
            <<",\"load_ms\":"<<load_ms<<",\"request_ms\":"<<request_ms<<",\"prefill_ms\":"<<s.prefill_ms
            <<",\"decode_ms\":"<<s.decode_ms<<",\"draft_ms\":"<<s.draft_ms<<",\"verify_ms\":"<<s.verify_ms
            <<",\"rebuild_ms\":"<<s.rebuild_ms<<",\"PP\":"<<(s.prefill_ms>0?1000*s.prompt_tokens/s.prefill_ms:0)
            <<",\"TG\":"<<(s.decode_ms>0?1000*(outputs-1)/s.decode_ms:0)
            <<",\"layerwise_residual_h2d_bytes\":"<<copies.residual_h2d_bytes
            <<",\"layerwise_residual_d2h_bytes\":"<<copies.residual_d2h_bytes
            <<",\"layerwise_residual_d2d_bytes\":"<<copies.residual_d2d_bytes
            <<",\"layerwise_ffn_h2d_bytes\":"<<copies.ffn_h2d_bytes
            <<",\"layerwise_ffn_d2h_bytes\":"<<copies.ffn_d2h_bytes
            <<",\"layerwise_router_d2h_bytes\":"<<copies.router_d2h_bytes
            <<",\"layerwise_route_h2d_bytes\":"<<copies.route_h2d_bytes
            <<",\"layerwise_loop_barriers\":"<<copies.loop_barriers
            <<",\"timing_scope\":\"completed_host_wall; PP includes target and teacher warmup; load separate; request includes output IO\""
            <<",\"performance_qualified\":false,\"passed\":true}\n";
        if(!std::cout)throw std::runtime_error("completion write failed");
        return 0;
    }catch(const std::exception& e){std::cerr<<"mtp_run: "<<e.what()<<'\n';return 1;}
}
