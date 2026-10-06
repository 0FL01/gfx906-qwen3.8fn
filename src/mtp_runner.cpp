#include "mtp_runner.hpp"
#include "mtp_session.hpp"
#include "profile_range.hpp"
#include "prefill_pipeline.hpp"
#include <algorithm>
#include <chrono>
#include <vector>
#include <stdexcept>
namespace qwen {
namespace {
constexpr int V=248320,W=10240;
using Clock=std::chrono::steady_clock;
double ms(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
SessionConfig checked(SessionConfig c) {
    if(c.capacity<4 || c.capacity>131072 || c.capacity%4 || c.max_batch_tokens<1 ||
       c.max_batch_tokens>1024 || c.expert_slots<1 || c.expert_slots>512 ||
       c.cpu_workers || c.hybrid_probe || !c.trace_directory.empty())
        throw std::invalid_argument("MTP runner requires valid GPU-only untraced SessionConfig");
    if(c.layerwise_prefill_capacity<0 || c.layerwise_prefill_capacity>16384 ||
       (c.layerwise_prefill_capacity && c.max_batch_tokens<4))
        throw std::invalid_argument("MTP layerwise capacity requires 0..16384 and frame>=4");
    if(c.prefill_pipeline_tokens<0 || c.prefill_pipeline_tokens>4096 ||
       c.prefill_pipeline_tokens>c.layerwise_prefill_capacity ||
       (c.layerwise_prefill_capacity>4096 && !c.prefill_pipeline_tokens) ||
       (c.prefill_pipeline_tokens && (c.layerwise_prefill_capacity+c.prefill_pipeline_tokens-1)/
          c.prefill_pipeline_tokens>PrefillPipeline::max_windows) ||
       (c.prefill_residual_device && !c.prefill_pipeline_tokens) ||
       c.prefill_pipeline_fail_stage!=-1 || c.prefill_pipeline_fail_window!=-1)
        throw std::invalid_argument("MTP pipeline requires layerwise resources and no diagnostic fault");
    c.speculative_checkpoints=true;c.max_batch_tokens=std::max(3,c.max_batch_tokens);
    return c;
}
SamplingConfig correction(SamplingConfig c){c.seed^=0x9e3779b97f4a7c15ULL;return c;}
}
struct MtpRunner::Impl {
    int capacity,chunk;
    bool layerwise;
    SamplingConfig sampling;
    // CPU samplers validate configuration before loading large model owners.
    Sampler proposal;
    SpeculativeSampler decision;
    Session target;
    MtpSession draft;
    std::array<std::vector<float>,3> p;
    std::array<std::vector<float>,2> q;
    MtpRunStats counters;
    std::size_t budget=0;
    std::int32_t pending=0,eos=248046;
    bool ignore=false,ready=false,done=false;
    Impl(const std::string& path,const std::string& sidecar,SessionConfig c,SamplingConfig s)
      :capacity(checked(c).capacity),chunk(c.layerwise_prefill_capacity ? c.layerwise_prefill_capacity : c.max_batch_tokens),
       layerwise(c.layerwise_prefill_capacity!=0),sampling(s),
       proposal(V,s),decision(V,correction(s)),target(path,checked(c)),
       draft(target,sidecar,{c.capacity,false}) {
        target.set_attention_batch(c.attention_query_tile>1);
        for(auto& x:p)x.resize(V);
        for(auto& x:q)x.resize(V);
    }
    MtpEmission begin(std::span<const std::int32_t> tokens,std::size_t outputs,bool ignore_eos,std::int32_t end) {
        if(tokens.empty() || tokens.size()>std::size_t(capacity) || outputs<1 ||
           outputs-1>std::size_t(capacity)-tokens.size() || end<0 || end>=V)
            throw std::invalid_argument("MTP prompt/output/capacity/EOS bounds");
        for(auto id:tokens)if(id<0 || id>=V)throw std::invalid_argument("MTP prompt vocabulary");
        ready=false;
        target.reset();draft.reset();proposal.reset_seed(sampling.seed);
        decision.reset_seed(sampling.seed^0x9e3779b97f4a7c15ULL);
        counters={};counters.prompt_tokens=tokens.size();budget=outputs;ignore=ignore_eos;eos=end;done=false;
        const auto start=Clock::now();
        std::span<const float> last;
        ProfileRange prefill_range("MTP_PP_COMPLETED");
        for(std::size_t first=0;first<tokens.size();) {
            const auto count=std::min<std::size_t>(chunk,tokens.size()-first);
            const auto ids=tokens.subspan(first,count);
            {
                ProfileRange target_range("MTP_PP_TARGET_CHUNK");
                last=layerwise ? target.prefill_layerwise(ids,Session::OutputMode::last_row) : target.prefill_last(ids);
            }
            const auto tap=target.target_tap();
            if(tap.rows!=int(count) || tap.width!=W || tap.first_position!=first || !tap.pointer)
                throw std::runtime_error("MTP prompt tap contract");
            for(std::size_t offset=0;offset<count;) {
                const auto n=std::min<std::size_t>(3,count-offset);
                std::array<MtpHiddenRow,3> hidden{};
                for(std::size_t i=0;i<n;++i) {
                    const auto local=offset+i;
                    hidden[i]=local ? MtpHiddenRow{1,tap.pointer+(local-1)*W,W} :
                        first ? draft.saved_target_carry() : MtpHiddenRow::zero();
                }
                (void)draft.teacher_append(std::span(hidden).first(n),ids.subspan(offset,n),MtpTeacherHead::skip);
                offset+=n;
            }
            draft.save_target_carry(int(count)-1);
            first+=count;
        }
        proposal.distribution(last.last(V),p[0]);
        pending=static_cast<std::int32_t>(proposal.draw(p[0]));
        counters.prefill_ms=ms(start);counters.outputs=1;counters.consumed=target.stats().consumed_tokens;
        if(counters.consumed!=tokens.size() || draft.prefix()!=tokens.size())
            throw std::runtime_error("MTP prompt history mismatch");
        const auto reason=!ignore && pending==eos ? SpeculativeStopReason::eos :
            budget==1 ? SpeculativeStopReason::output_budget : SpeculativeStopReason::window_complete;
        done=reason!=SpeculativeStopReason::window_complete;ready=true;
        return {{pending,0,0},1,done,reason};
    }
    MtpEmission next() {
        if(!ready || done)throw std::logic_error("MTP next requires active begin");
        const auto consumed=target.stats().consumed_tokens;
        if(consumed!=draft.prefix() || consumed!=counters.prompt_tokens+counters.outputs-1 ||
           consumed>=std::uint64_t(capacity))throw std::runtime_error("MTP entry prefix invariant");
        ready=false;
        const auto start=Clock::now();
        const auto remaining=budget-counters.outputs;
        const int horizon=int(std::min<std::size_t>({2,remaining-1,std::size_t(capacity)-consumed-1}));
        std::array<std::int32_t,3> inputs{pending,0,0};
        const auto dstart=Clock::now();
        ProfileRange draft_range("MTP_DRAFT_COMPLETED");
        for(int i=0;i<horizon;++i) {
            const std::array hidden{i ? draft.own_tap().row(0) : draft.saved_target_carry()};
            const auto logits=draft.step(hidden,std::span(&inputs[i],1));
            proposal.distribution(logits,q[i]);inputs[i+1]=static_cast<std::int32_t>(proposal.draw(q[i]));
        }
        draft.restore_prefix(consumed);
        draft_range.end();
        counters.draft_ms+=ms(dstart);
        const auto vstart=Clock::now();
        ProfileRange verify_range("MTP_TARGET_VERIFY_COMPLETED");
        const auto verified=target.verify_window(std::span(inputs).first(horizon+1));
        SpeculativeWindow window;window.horizon=horizon;
        for(int i=0;i<=horizon;++i) {
            proposal.distribution(verified.subspan(i*V,V),p[i]);window.p[i]=p[i];
            if(i<horizon){window.q[i]=q[i];window.draft_ids[i]=inputs[i+1];}
        }
        const auto selected=decision.decide(window,{remaining,std::size_t(eos),ignore});
        verify_range.end();
        counters.verify_ms+=ms(vstart);
        const auto rstart=Clock::now();
        ProfileRange rebuild_range("MTP_TEACHER_RESTORE_COMPLETED");
        const auto keep=selected.restore_prefix;
        const auto tap=target.target_tap();
        if(tap.rows!=horizon+1 || tap.first_position!=consumed || tap.width!=W || !tap.pointer ||
           keep<1 || keep>std::size_t(horizon+1) || selected.emitted_count!=keep)
            throw std::runtime_error("MTP verify/terminal contract");
        const std::array hidden{draft.saved_target_carry(),MtpHiddenRow{1,tap.pointer,W},
                               MtpHiddenRow{1,tap.pointer+W,W}};
        (void)draft.teacher_append(std::span(hidden).first(keep),std::span(inputs).first(keep),MtpTeacherHead::skip);
        draft.save_target_carry(int(keep)-1);target.restore_prefix(int(keep));
        rebuild_range.end();
        counters.rebuild_ms+=ms(rstart);
        MtpEmission out;out.count=selected.emitted_count;out.reason=selected.stop_reason;
        for(std::size_t i=0;i<out.count;++i)out.ids[i]=static_cast<std::int32_t>(selected.emitted_ids[i]);
        pending=static_cast<std::int32_t>(selected.pending_carry);
        counters.outputs+=out.count;counters.consumed=target.stats().consumed_tokens;
        ++counters.windows;++counters.accepted.at(selected.accepted_drafts);
        if(counters.consumed!=draft.prefix() || counters.consumed!=counters.prompt_tokens+counters.outputs-1)
            throw std::runtime_error("MTP restored prefix mismatch");
        counters.decode_ms+=ms(start);done=out.reason!=SpeculativeStopReason::window_complete;
        out.finished=done;ready=true;return out;
    }
};
MtpRunner::MtpRunner(const std::string& p,const std::string& s,SessionConfig c,SamplingConfig sample)
    :impl_(std::make_unique<Impl>(p,s,c,sample)){}
MtpRunner::~MtpRunner()=default;
MtpEmission MtpRunner::begin(std::span<const std::int32_t> t,std::size_t n,bool ignore,std::int32_t eos){
    return impl_->begin(t,n,ignore,eos);
}
MtpEmission MtpRunner::next(){return impl_->next();}
MtpRunStats MtpRunner::stats()const {
    auto out=impl_->counters;out.proposal_draws=impl_->proposal.random_draws();
    out.decision_draws=impl_->decision.random_draws();return out;
}
SessionLayerwiseTransfers MtpRunner::layerwise_transfers()const{return impl_->target.layerwise_transfers();}
bool MtpRunner::requires_begin()const{return !impl_->ready || impl_->done;}
}
