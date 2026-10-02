#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>

namespace qwen {

struct SessionConfig {
    int capacity = 4096;
    int expert_slots = 112;
    // Empty disables all synchronous intermediate capture. Diagnostics only.
    std::string trace_directory;
};

struct SessionStats {
    std::uint64_t consumed_tokens = 0;
    std::uint64_t expert_hits = 0, expert_misses = 0, expert_upload_bytes = 0;
    double last_completed_ms = 0;
};

// One exclusive interactive session, static 24/24 layer split. Canonical expert
// weights stay in RAM; each layer owns an immutable-weight LRU on its device.
// A failed step invalidates the session until reset(), rather than exposing a
// partially advanced cross-layer state. Returned logits live until the next call.
// No tokenizer/sampler and no linkage to llama/ggml execution.
class Session {
public:
    explicit Session(const std::string& model_path, SessionConfig config = {});
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    std::span<const float> step(std::int32_t token);
    void reset();
    SessionStats stats() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
