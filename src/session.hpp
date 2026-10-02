#pragma once

#include <array>
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

struct SessionDeviceMemory {
    int device = 0, first_layer = 0, last_layer = 0;
    int gdn_layers = 0, qsa_layers = 0;
    std::uint64_t weights = 0, expert_slots = 0, qsa_kv = 0, qsa_index = 0;
    std::uint64_t gdn_state = 0, ple_state = 0, workspace = 0;
    std::uint64_t owned_bytes = 0, owned_peak_bytes = 0, owned_buffers = 0;
    std::uint64_t total_vram = 0, free_vram = 0;
};

// Diagnostic snapshot, not a hot-path allocator or occupied-context test.
// GPU categories cover every live Session Buffer; workspace includes staged
// recurrent/conv state. HIP/context/rocBLAS allocations are visible only in
// total/free VRAM. Host capacities below exclude metadata/allocator overhead.
struct SessionMemory {
    int capacity = 0, expert_slots = 0;
    std::array<SessionDeviceMemory, 2> devices{};
    std::uint64_t ram_expert_capacity = 0, ram_expert_payload = 0;
    std::uint64_t host_embedding_capacity = 0, host_logit_capacity = 0;
    std::uint64_t pinned_handoff = 0;
    std::uint64_t expert_payload_reads = 0, expert_payload_bytes_read = 0;
    bool ownership_verified = false;
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
    // Synchronizes both owned streams and restores the caller's current device.
    // Does not modify logits, state, cache maps, counters or logical visibility.
    SessionMemory memory() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
