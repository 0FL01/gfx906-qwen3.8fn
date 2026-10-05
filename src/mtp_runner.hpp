#pragma once
#include "session.hpp"
#include "sampling.hpp"
#include "speculative.hpp"
#include <array>
#include <memory>
#include <span>
#include <string>

namespace qwen {
struct MtpEmission {
    std::array<std::int32_t,3> ids{};
    std::size_t count=0;
    bool finished=false;
    SpeculativeStopReason reason=SpeculativeStopReason::window_complete;
};
struct MtpRunStats {
    std::uint64_t prompt_tokens=0,outputs=0,consumed=0,windows=0;
    std::array<std::uint64_t,3> accepted{};
    std::uint64_t proposal_draws=0,decision_draws=0;
    double prefill_ms=0,decode_ms=0,draft_ms=0,verify_ms=0,rebuild_ms=0;
};
// Single-owner opt-in GPU-only runner. Owns target before its borrowing sidecar;
// the sidecar is destroyed first. Original core-session behavior is unchanged.
// begin() validates the full request before reset/model execution; it resets RNG.
// next() never resets/replays history. Any execution failure requires a new begin.
// Output is token IDs, not a tokenizer/chat template or a streaming HTTP server.
// capacity/slots/attention tile come from SessionConfig. max_batch_tokens sets
// ordinary PP chunk size; target verification internally reserves at least3.
// Optional prefill_pipeline_tokens subdivides each target layerwise window;
// warmup still consumes all actual tap rows after both GPU stages finish.
// Optional layerwise_prefill_capacity (1..16384, frame>=4) sets a larger prompt
// window, retaining every teacher tap; above 4096 requires the bounded pipeline.
// Zero preserves the default chunk path.
class MtpRunner {
public:
    MtpRunner(const std::string& target,const std::string& sidecar,
              SessionConfig config,SamplingConfig sampling);
    ~MtpRunner();
    MtpRunner(const MtpRunner&)=delete;
    MtpRunner& operator=(const MtpRunner&)=delete;
    MtpEmission begin(std::span<const std::int32_t> prompt,std::size_t outputs,
                      bool ignore_eos=false,std::int32_t eos=248046);
    MtpEmission next();
    MtpRunStats stats() const;
    bool requires_begin() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
