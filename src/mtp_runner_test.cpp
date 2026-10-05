#include "mtp_runner.hpp"
#include <bit>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <stdexcept>
#include <string_view>
namespace {
void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class F>void reject(F&& f){bool hit=false;try{f();}catch(const std::logic_error&){hit=true;}need(hit,"expected rejection");}
std::vector<std::int32_t> finish(qwen::MtpRunner& r,qwen::MtpEmission e){
    std::vector<std::int32_t> out;
    for(;;){out.insert(out.end(),e.ids.begin(),e.ids.begin()+e.count);if(e.finished)return out;e=r.next();}
}
} // namespace
int main(int argc,char** argv){
 try {
    static_assert(std::endian::native==std::endian::little);
    need(argc==4 || (argc==5 && std::string_view(argv[4])=="--layerwise"),"usage: core-mtp-runner-test TARGET SIDECAR CLI32_IDS [--layerwise]");
    std::vector<std::int32_t> expected(32);
    std::ifstream f(argv[3],std::ios::binary);
    need(bool(f.read(reinterpret_cast<char*>(expected.data()),128)) && f.peek()==std::char_traits<char>::eof(),"exact32 IDs required");
    for(auto t:expected)need(t>=0 && t<248320,"expected vocabulary");
    qwen::SessionConfig c;c.capacity=48;c.expert_slots=112;c.max_batch_tokens=8;
    if(argc==5)c.layerwise_prefill_capacity=16;
    qwen::MtpRunner r(argv[1],argv[2],c,qwen::SamplingConfig(12345));
    const std::array<std::int32_t,8> prompt{248044,100,101,102,103,104,105,106};
    need(r.requires_begin(),"new state");reject([&]{(void)r.next();});
    need(finish(r,r.begin(prompt,32,true))==expected,"PPchunk8/live32 vs CLIchunk1 IDs");
    auto stats=r.stats();
    need(stats.outputs==32 && stats.consumed==39 && stats.accepted==std::array<std::uint64_t,3>{10,6,3}
         && stats.proposal_draws==38 && stats.decision_draws==46,"actual sampler counters");
    need(r.requires_begin(),"finished state");reject([&]{(void)r.next();});
    auto e=r.begin(prompt,32,true);
    const auto before=r.stats();const std::array<std::int32_t,1> bad{-1};
    reject([&]{(void)r.begin(bad,1);});reject([&]{(void)r.begin(prompt,0);});
    reject([&]{(void)r.begin(prompt,42);});reject([&]{(void)r.begin(prompt,1,false,248320);});
    need(!r.requires_begin() && r.stats().outputs==before.outputs &&
         r.stats().proposal_draws==before.proposal_draws && r.stats().decision_draws==before.decision_draws,"invalid request preserves active state/RNG");
    need(finish(r,e)==expected,"invalid begin then continuation");
    // Custom stop-token IDs exercise EOS transitions; NOT evidence that the
    // model's actual EOS248046 naturally occurred in this generated fixture.
    e=r.begin(prompt,1,false,expected[0]);
    need(e.finished && e.reason==qwen::SpeculativeStopReason::eos && e.count==1 &&
         r.stats().consumed==8,"initial stop before forward, EOS wins budget tie");
    const auto two=finish(r,r.begin(prompt,2,true));
    e=r.begin(prompt,2,false,two[1]);
    need(!e.finished,"distinct initial token");e=r.next();
    need(e.finished && e.reason==qwen::SpeculativeStopReason::eos && e.count==1 &&
         e.ids[0]==two[1] && r.stats().consumed==9,"replacement stop/pending boundary");
    const auto full=finish(r,r.begin(prompt,41,true));
    need(full.size()==41 && r.stats().consumed==48 && r.requires_begin(),"actual capacity boundary");
    std::cout<<"{\"kind\":\"mtp_runner_api_test\",\"layerwise_prefill_capacity\":"<<c.layerwise_prefill_capacity
      <<",\"passed\":true,\"cli32_ids_equal\":true,\"prefill_chunks\":[1,8],"
      "\"repeat_reset\":true,\"invalid_request_preservation\":true,\"custom_stop_cases\":2,"
      "\"actual_model_EOS_observed\":false,\"capacity_boundary\":48,\"performance_claim\":false}\n";
 }catch(const std::exception& e){std::cerr<<"mtp_runner_test: "<<e.what()<<'\n';return 1;}
}
