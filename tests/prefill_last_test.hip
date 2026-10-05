#include "session.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>
namespace {
constexpr int V=248320,W=10240;
void need(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
void hip(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
struct Metrics{
    std::uint64_t values=0,bits=0;double maxabs=0,maxratio=0;
    void compare(std::span<const float> a,std::span<const float> b){
        need(a.size()==b.size(),"logit extent");
        for(std::size_t i=0;i<a.size();++i){
            need(std::isfinite(a[i])&&std::isfinite(b[i]),"finite logits");
            const auto error=std::abs(double(a[i])-b[i]),bound=.02+.002*std::abs(double(b[i]));
            maxabs=std::max(maxabs,error);maxratio=std::max(maxratio,error/bound);
            bits+=std::memcmp(&a[i],&b[i],4)!=0;++values;
            need(error<=bound,"frozen logit bound");
        }
    }
};
std::vector<float> tap_copy(qwen::SessionTargetTap tap){
    need(tap.device==1&&tap.rows>0&&tap.width==W&&tap.pointer,"tap shape");
    hip(hipSetDevice(tap.device));std::vector<float> out(std::size_t(tap.rows)*W);
    hip(hipMemcpy(out.data(),tap.pointer,out.size()*4,hipMemcpyDeviceToHost));return out;
}
struct Fixture{
    qwen::Session& s; qwen::SessionMemory memory;Metrics metrics;int rejected=0,cases=0;
    explicit Fixture(qwen::Session& session):s(session),memory(s.memory()){}
    void steady(){
        const auto now=s.memory();need(now.ownership_verified&&now.host_logit_capacity==memory.host_logit_capacity&&
            now.pinned_handoff==memory.pinned_handoff&&now.target_tap_bytes==memory.target_tap_bytes&&
            now.target_tap_staging_bytes==memory.target_tap_staging_bytes,"host/tap ownership changed");
        for(int d=0;d<2;++d)need(now.devices[d].owned_bytes==memory.devices[d].owned_bytes&&
            now.devices[d].owned_buffers==memory.devices[d].owned_buffers&&
            now.devices[d].owned_peak_bytes==memory.devices[d].owned_peak_bytes,"device ownership changed");
    }
    template<class F>void reject(F f,std::span<const float> published){
        const auto state=s.checkpoint_state();const auto tap=s.target_tap();
        const auto consumed=s.stats().consumed_tokens;const auto tensor=tap_copy(tap);
        const std::vector<float> old(published.begin(),published.end());
        bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}
        need(caught,"argument rejection");need(s.checkpoint_state()==state&&s.target_tap()==tap&&s.stats().consumed_tokens==consumed,"rejection state changed");
        const auto after=tap_copy(tap);
        need(std::memcmp(old.data(),published.data(),old.size()*4)==0&&
            std::memcmp(after.data(),tensor.data(),tensor.size()*4)==0,"rejection publication changed");
        steady();++rejected;
    }
};
}
int main(int argc,char** argv){
    try{
        need(argc==2,"usage: core-prefill-last-test MODEL");
        std::cout<<std::setprecision(17)<<"{\"kind\":\"prefill_last_source\",\"revision\":\""<<CORE_REVISION
            <<"\",\"dirty\":"<<(CORE_DIRTY?"true":"false")<<",\"reference\":\"same Session full-output batch32 and N1 continuations\",\"absolute_gate\":0.02,\"relative_gate\":0.002,\"performance_qualified\":false}\n";
        qwen::SessionConfig cfg;cfg.capacity=40;cfg.max_batch_tokens=32;cfg.expert_slots=1;
        cfg.attention_query_tile=8;cfg.speculative_checkpoints=true;
        std::array<std::int32_t,32> ids{};ids[0]=248044;for(int i=1;i<32;++i)ids[i]=99+i;
        std::vector<float> full,base_tap,continuations(8ULL*V);
        {
            qwen::Session session(argv[1],cfg);session.set_attention_batch(true);Fixture f(session);
            auto view=session.step_batch(ids);need(view.size()==32ULL*V,"full batch unchanged");
            full.assign(view.begin(),view.end());base_tap=tap_copy(session.target_tap());
            for(int i=0;i<8;++i){view=session.step(131+i);std::copy(view.begin(),view.end(),continuations.begin()+std::size_t(i)*V);}
            const auto cont=[&](int i){return std::span<const float>(continuations).subspan(std::size_t(i)*V,V);};
            for(int chunk:{1,2,3,8,17,32}){
                session.reset();std::vector<float> taps(32ULL*W);std::span<const float> last;
                for(int first=0;first<32;first+=chunk){
                    const int count=std::min(chunk,32-first);
                    last=session.prefill_last(std::span(ids).subspan(first,count));
                    need(last.size()==V,"last-row extent");
                    f.metrics.compare(last,std::span<const float>(full).subspan(std::size_t(first+count-1)*V,V));
                    const auto tap=session.target_tap();need(tap.rows==count&&tap.first_position==std::uint64_t(first),"all pre-head tap rows retained");
                    const auto copied=tap_copy(tap);std::copy(copied.begin(),copied.end(),taps.begin()+std::size_t(first)*W);
                    need(session.stats().consumed_tokens==std::uint64_t(first+count),"full window consumed");f.steady();
                }
                need(std::memcmp(taps.data(),base_tap.data(),taps.size()*4)==0,"all pre-head tap bits unchanged");
                const std::array<std::int32_t,2> bad{100,-1};const std::array<std::int32_t,33> long_window{};
                f.reject([&]{(void)session.prefill_last({});},last);
                f.reject([&]{(void)session.prefill_last(bad);},last);
                f.reject([&]{(void)session.prefill_last(long_window);},last);
                for(int i=0;i<8;++i){last=session.step(131+i);f.metrics.compare(last,cont(i));}
                const std::array<std::int32_t,1> beyond{139};
                f.reject([&]{(void)session.prefill_last(beyond);},last);
                ++f.cases;std::cout<<"{\"kind\":\"prefill_last_case\",\"chunk\":"<<chunk<<",\"rows\":32,\"continuations\":8,\"passed\":true}\n";
            }
            // Mix both ordinary output modes on an occupied prefix.
            session.reset();view=session.step_batch(std::span(ids).first(5));
            f.metrics.compare(view,std::span<const float>(full).first(5ULL*V));
            view=session.prefill_last(std::span(ids).subspan(5));f.metrics.compare(view,std::span<const float>(full).last(V));
            for(int i=0;i<8;++i){view=session.step(131+i);f.metrics.compare(view,cont(i));}++f.cases;f.steady();
            // Invalid ordinary call preserves pending verify; a successful one
            // invalidates it exactly like the existing step_batch contract.
            session.reset();view=session.prefill_last(ids);
            const std::array<std::int32_t,2> two{131,132};view=session.verify_window(two);
            need(view.size()==2ULL*V,"verify still returns full rows");f.metrics.compare(view, std::span<const float>(continuations).first(2ULL*V));
            const std::array<std::int32_t,1> bad{-1},next{133};
            f.reject([&]{(void)session.prefill_last(bad);},view);
            view=session.prefill_last(next);f.metrics.compare(view,cont(2));need(!session.checkpoint_state().pending,"ordinary call invalidates pending verify");
            f.reject([&]{session.restore_prefix(0);},view);
            for(int i=3;i<8;++i){view=session.step(131+i);f.metrics.compare(view,cont(i));}++f.cases;f.steady();
            // Restore, then last-output forward must follow retained history.
            session.reset();(void)session.prefill_last(ids);
            const std::array<std::int32_t,3> three{131,132,133};(void)session.verify_window(three);session.restore_prefix(1);
            const std::array<std::int32_t,2> after{132,133};view=session.prefill_last(after);f.metrics.compare(view,cont(2));
            for(int i=3;i<8;++i){view=session.step(131+i);f.metrics.compare(view,cont(i));}++f.cases;f.steady();
            std::cout<<"{\"kind\":\"prefill_last_metrics\",\"cases\":"<<f.cases<<",\"values\":"<<f.metrics.values<<",\"bit_mismatches\":"<<f.metrics.bits
                <<",\"maxabs\":"<<f.metrics.maxabs<<",\"maxboundratio\":"<<f.metrics.maxratio<<",\"host_rejections\":"<<f.rejected
                <<",\"tap_bits_exact\":true,\"steady_owned_allocations\":true,\"passed\":true}\n";
        }
        // The ordinary non-checkpoint destination writes directly to host_logits
        // instead of swapping staged publication. Exercise that path separately.
        {
            cfg.speculative_checkpoints=false;
            qwen::Session session(argv[1],cfg);session.set_attention_batch(true);Metrics metrics;
            const auto before=session.memory();auto view=session.prefill_last(ids);
            metrics.compare(view,std::span<const float>(full).last(V));
            for(int i=0;i<8;++i){view=session.step(131+i);metrics.compare(view,std::span<const float>(continuations).subspan(std::size_t(i)*V,V));}
            const std::vector<float> saved(view.begin(),view.end());const std::array<std::int32_t,1> beyond{139};
            bool caught=false;try{(void)session.prefill_last(beyond);}catch(const std::invalid_argument&){caught=true;}
            need(caught&&session.stats().consumed_tokens==40&&std::memcmp(saved.data(),view.data(),saved.size()*4)==0,"ordinary capacity rejection preserves logits");
            session.reset();view=session.prefill_last(std::span(ids).first(1));metrics.compare(view,std::span<const float>(full).first(V));
            const auto after=session.memory();need(after.ownership_verified&&after.host_logit_capacity==before.host_logit_capacity,"ordinary host owners");
            for(int d=0;d<2;++d)need(after.devices[d].owned_bytes==before.devices[d].owned_bytes&&after.devices[d].owned_buffers==before.devices[d].owned_buffers,"ordinary device owners");
            std::cout<<"{\"kind\":\"prefill_last_ordinary\",\"values\":"<<metrics.values<<",\"bit_mismatches\":"<<metrics.bits<<",\"maxabs\":"<<metrics.maxabs
                <<",\"maxboundratio\":"<<metrics.maxratio<<",\"capacity_rejection\":true,\"reset_reuse\":true,\"passed\":true}\n";
        }
        std::cout<<"{\"kind\":\"prefill_last_complete\",\"checkpoint_modes\":[true,false],\"cleanup_complete\":true,\"passed\":true,\"independent_HF\":false,\"speed_claim\":false}\n";
        need(bool(std::cout),"stdout");
    }catch(const std::exception& e){std::cerr<<"prefill-last: "<<e.what()<<'\n';return 1;}
}

