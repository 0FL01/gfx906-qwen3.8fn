#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace qwen {
// One persistent producer for GPU0 and the calling GPU1 consumer. Exactly two
// borrowed handoff slots; this is not a general task executor or GPU scheduler.
// Callback/context and the payload behind both slots outlive this owner.
// One caller serializes begin/acquire/release/finish/cancel/destruction. The
// producer callback may only query cancelled(); it must not reenter this API.
// Payload is writable by the producer only while its slot is filling, and by
// the consumer only after acquire until release. DMA must be complete before
// releasing a borrowed slot. A callback failure invalidates the whole batch.
// The callback must drain its device work before returning OR throwing; this
// host coordinator cannot drain HIP streams or consumer-owned asynchronous work.
class PrefillPipeline {
public:
    using Produce = void (*)(void* context, int window, int slot);
    static constexpr int max_windows = 4096;
    struct Stats {
        int windows = 0, started = 0, produced = 0, acquired = 0, released = 0;
        int max_occupied_slots = 0;
        bool active = false, cancelled = false, failed = false;
        bool operator==(const Stats&) const = default;
    };
    PrefillPipeline(void* context, Produce);
    ~PrefillPipeline();
    PrefillPipeline(const PrefillPipeline&) = delete;
    PrefillPipeline& operator=(const PrefillPipeline&) = delete;
    PrefillPipeline(PrefillPipeline&&) = delete;
    PrefillPipeline& operator=(PrefillPipeline&&) = delete;
    // No allocation/thread creation on normal batches. Arguments and logical
    // ownership are checked before changes; windows are consumed in order.
    void begin(int windows);
    int acquire(int window);
    void release(int window);
    void finish();
    // On consumer/callback failure, call this before discarding borrowed data
    // or starting another batch. It wakes blocked slot waits, drains the active
    // callback, and releases metadata, not payload. Destructor also drains.
    void cancel_and_drain() noexcept;
    bool cancelled() const noexcept;
    Stats stats() const;
    // Fixed host metadata only; excludes thread runtime bookkeeping/stack and
    // caller-owned payload. No device allocation is owned by this coordinator.
    std::size_t metadata_bytes() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
