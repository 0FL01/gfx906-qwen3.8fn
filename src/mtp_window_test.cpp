#include "mtp_session.hpp"
#include "speculative.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef CORE_REVISION
#define CORE_REVISION "unknown"
#endif

namespace {
constexpr int V=248320, W=10240, CAP=48;
constexpr std::int32_t EOS=248046;
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
void compare(std::span<const float> a,std::span<const float> b,
             std::uint64_t& compared,double& maximum) {
    require(a.size()==b.size(),"comparison extent");
    for(std::size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite comparison");
        const double error=std::abs(double(a[i])-double(b[i]));
        maximum=std::max(maximum,error);
        require(error<=.02+.002*std::abs(double(b[i])),"full-vocabulary state replay mismatch");
        ++compared;
    }
}
struct Result {
    std::array<std::uint64_t,3> accepted{};
    std::uint64_t windows=0,target_compared=0,draft_compared=0,continuation_compared=0;
    std::uint64_t proposal_draws=0,decision_draws=0;
    double max_error=0;
    std::vector<std::int32_t> generated_ids;
};
// Diagnostic coordinator only. Every completed window is independently replayed
// through the same target's ordinary N1 path and rebuilt teacher history. This
// deliberately destroys performance meaning and avoids loading a second 70GB
// target. It is state/orchestration evidence, not an independent model oracle.
Result run(qwen::Session& target,qwen::MtpSession& draft,int budget,int max_horizon,
           bool ignore_eos,std::uint64_t seed,bool replay=true) {
    const std::array<std::int32_t,8> prompt{248044,100,101,102,103,104,105,106};
    require(budget>=1 && budget<=CAP-int(prompt.size())+1,"output budget");
    require(max_horizon>=0 && max_horizon<=2,"horizon");
    target.reset();draft.reset();
    qwen::SamplingConfig config(seed);
    qwen::Sampler sampler(V,config);
    // Separate explicit RNG streams: no claim of baseline random-stream parity.
    qwen::SamplingConfig correction(seed^0x9e3779b97f4a7c15ULL);
    qwen::SpeculativeSampler decide(V,correction);
    std::array<std::vector<float>,3> p;
    std::array<std::vector<float>,2> q;
    for(auto& v:p)v.resize(V);
    for(auto& v:q)v.resize(V);
    std::vector<float> verify_copy(3*V),draft_probe(V),target_probe(V);
    std::vector<std::int32_t> emitted;emitted.reserve(budget);
    std::vector<std::int32_t> history(prompt.begin(),prompt.end());
    history.reserve(CAP);
    Result result;
    std::span<const float> last;
    // Explicit chronological warmup: x0 has ZERO previous hidden; xp uses T[p-1].
    for(std::size_t pos=0;pos<prompt.size();++pos) {
        last=target.step(prompt[pos]);
        const std::array hidden{pos ? draft.saved_target_carry() : qwen::MtpHiddenRow::zero()};
        const std::span token(&prompt[pos],1);
        (void)draft.teacher_append(hidden,token,qwen::MtpTeacherHead::skip);
        draft.save_target_carry(0);
    }
    sampler.distribution(last,p[0]);
    std::int32_t pending=static_cast<std::int32_t>(sampler.draw(p[0]));
    emitted.push_back(pending);
    bool stopped=!ignore_eos && pending==EOS;
    while(!stopped && int(emitted.size())<budget) {
        const auto start=target.stats().consumed_tokens;
        require(start==draft.prefix() && start==history.size(),"entry prefix/pending invariant");
        require(start<CAP,"target capacity exhausted");
        const int remaining=budget-int(emitted.size());
        const int horizon=std::min({max_horizon,remaining-1,CAP-int(start)-1});
        std::array<std::int32_t,3> inputs{pending,0,0};
        for(int i=0;i<horizon;++i) {
            const std::array hidden{i==0 ? draft.saved_target_carry() : draft.own_tap().row(0)};
            const auto logits=draft.step(hidden,std::span(&inputs[i],1));
            sampler.distribution(logits,q[i]);
            inputs[i+1]=static_cast<std::int32_t>(sampler.draw(q[i]));
        }
        draft.restore_prefix(start); // discard proposal KV BEFORE target verify
        const auto verified=target.verify_window(std::span(inputs).first(horizon+1));
        require(verified.size()==std::size_t(horizon+1)*V,"verify extent");
        std::copy(verified.begin(),verified.end(),verify_copy.begin());
        qwen::SpeculativeWindow window;window.horizon=horizon;
        for(int i=0;i<=horizon;++i) {
            sampler.distribution(verified.subspan(i*V,V),p[i]);window.p[i]=p[i];
            if(i<horizon){window.q[i]=q[i];window.draft_ids[i]=inputs[i+1];}
        }
        const auto terminal=decide.decide(window,{std::size_t(remaining),std::size_t(EOS),ignore_eos});
        const int keep=static_cast<int>(terminal.restore_prefix);
        require(keep>=1 && keep<=horizon+1 && terminal.emitted_count==std::size_t(keep),"terminal consumed contract");
        const auto taps=target.target_tap();
        require(taps.rows==horizon+1 && taps.first_position==start && taps.width==W,"target tap window");
        const std::array teacher{draft.saved_target_carry(),
            qwen::MtpHiddenRow{1,taps.pointer,W},qwen::MtpHiddenRow{1,taps.pointer+W,W}};
        // Rebuild only the consumed prefix; rejected inputs need not be replayed.
        (void)draft.teacher_append(std::span(teacher).first(keep),std::span(inputs).first(keep),qwen::MtpTeacherHead::skip);
        draft.save_target_carry(keep-1);
        target.restore_prefix(keep); // target uses relative retained-input count
        history.insert(history.end(),inputs.begin(),inputs.begin()+keep);
        for(std::size_t i=0;i<terminal.emitted_count;++i)emitted.push_back(static_cast<std::int32_t>(terminal.emitted_ids[i]));
        pending=static_cast<std::int32_t>(terminal.pending_carry);
        require(pending==emitted.back() && history.size()==prompt.size()+emitted.size()-1,"pending output exclusion");
        require(target.stats().consumed_tokens==history.size() && draft.prefix()==history.size(),"restored target/draft prefix");
        ++result.accepted.at(terminal.accepted_drafts);++result.windows;
        stopped=terminal.stop_reason!=qwen::SpeculativeStopReason::window_complete;
        // Probe restored draft state before resetting/rebuilding either owner.
        // At the capacity boundary there is no legal next-row probe.
        if(replay) {
        const bool probe=history.size()<CAP;
        if(probe) {
            // Exercise the ACTUAL restored recurrent/QSA/PLE target state before
            // reset, rather than validating only cached pre-restore verify rows.
            const auto continuation=target.verify_window(std::span(&pending,1));
            std::copy(continuation.begin(),continuation.end(),target_probe.begin());
            target.restore_prefix(0);
            const std::array hidden{draft.saved_target_carry()};
            const auto logits=draft.step(hidden,std::span(&pending,1));
            std::copy(logits.begin(),logits.end(),draft_probe.begin());
            draft.restore_prefix(history.size());
        }
        target.reset();draft.reset();
        for(std::size_t pos=0;pos<history.size();++pos) {
            const auto logits=target.step(history[pos]);
            if(pos>=start)compare(std::span(verify_copy).subspan((pos-start)*V,V),logits,
                                   result.target_compared,result.max_error);
            const std::array hidden{pos ? draft.saved_target_carry() : qwen::MtpHiddenRow::zero()};
            (void)draft.teacher_append(hidden,std::span(&history[pos],1),qwen::MtpTeacherHead::skip);
            draft.save_target_carry(0);
        }
        if(probe) {
            const auto continuation=target.verify_window(std::span(&pending,1));
            compare(target_probe,continuation,result.continuation_compared,result.max_error);
            target.restore_prefix(0);
            const std::array hidden{draft.saved_target_carry()};
            const auto logits=draft.step(hidden,std::span(&pending,1));
            compare(draft_probe,logits,result.draft_compared,result.max_error);
            draft.restore_prefix(history.size());
        }
        } // optional diagnostic replay; live run retains actual state across windows
    }
    require(int(emitted.size())==budget || (!ignore_eos && emitted.back()==EOS),"actual output termination");
    require(target.stats().consumed_tokens==prompt.size()+emitted.size()-1,"final consumed count");
    result.generated_ids=emitted;
    result.proposal_draws=sampler.random_draws();result.decision_draws=decide.random_draws();
    std::cout<<"{\"kind\":\"trained_mtp_window_case\",\"budget\":"<<budget
        <<",\"horizon\":"<<max_horizon<<",\"ignore_eos\":"<<(ignore_eos?"true":"false")
        <<",\"replay_checks\":"<<(replay?"true":"false")<<",\"seed\":"<<seed<<",\"actual_outputs\":"<<emitted.size()
        <<",\"consumed\":"<<target.stats().consumed_tokens<<",\"pending\":"<<pending
        <<",\"windows\":"<<result.windows<<",\"accepted\":["<<result.accepted[0]<<','<<result.accepted[1]<<','<<result.accepted[2]
        <<"],\"target_compared\":"<<result.target_compared<<",\"draft_compared\":"<<result.draft_compared
        <<",\"continuation_compared\":"<<result.continuation_compared<<",\"maxabs\":"<<result.max_error<<",\"proposal_draws\":"<<result.proposal_draws
        <<",\"decision_draws\":"<<result.decision_draws<<",\"performance_claim\":false,\"passed\":true}\n";
    return result;
}
}
int main(int argc,char** argv) {
    try {
        require(argc==3,"usage: core-mtp-window-test TARGET SIDECAR");
        std::cout<<std::setprecision(17);
        qwen::SessionConfig c;c.capacity=CAP;c.expert_slots=112;c.max_batch_tokens=3;c.speculative_checkpoints=true;
        qwen::Session target(argv[1],c);qwen::MtpSession draft(target,argv[2],{CAP,false});
        std::cout<<"{\"kind\":\"trained_mtp_window_source\",\"revision\":\""<<CORE_REVISION
                 <<"\",\"capacity\":48,\"sampling\":{\"temperature\":1,\"top_p\":0.95,\"top_k\":20},\"independent_reference\":false}\n";
        (void)run(target,draft,1,2,true,12345);
        (void)run(target,draft,2,2,true,12345);
        (void)run(target,draft,3,1,true,12346);
        const auto checked=run(target,draft,32,2,true,12345);
        (void)run(target,draft,41,2,true,12347);
        (void)run(target,draft,16,2,false,12348);
        const auto live=run(target,draft,32,2,true,12345,false);
        require(live.generated_ids==checked.generated_ids && live.accepted==checked.accepted &&
                live.proposal_draws==checked.proposal_draws && live.decision_draws==checked.decision_draws,
                "uninterrupted target/draft state must match replay-checked execution");
        std::cout<<"{\"kind\":\"trained_mtp_window_complete\",\"cases\":7,\"uninterrupted_matches_replayed\":true,\"performance_claim\":false,\"passed\":true}\n";
    } catch(const std::exception& e){std::cerr<<"mtp_window_test: "<<e.what()<<'\n';return 1;}
}
