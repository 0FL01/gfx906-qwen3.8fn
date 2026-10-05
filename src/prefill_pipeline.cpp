#include "prefill_pipeline.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace qwen {
struct PrefillPipeline::Impl {
    enum class State { empty, filling, ready, reading };
    struct Slot { State state = State::empty; int window = -1; };
    void* context;
    Produce produce;
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::array<Slot, 2> slots{};
    std::atomic<bool> cancel{false};
    bool stop = false, task_ready = false, active = false, producer_done = true;
    int next = 0, reading = -1;
    Stats counters{};
    std::exception_ptr failure;
    std::thread producer;

    Impl(void* c, Produce f) : context(c), produce(f) {
        if (!c || !f) throw std::invalid_argument("prefill pipeline callback/context");
        producer = std::thread([this] { loop(); });
    }
    ~Impl() {
        drain();
        { std::lock_guard lock(mutex); stop = true; }
        changed.notify_all();
        if (producer.joinable()) producer.join();
    }
    void loop() noexcept {
        std::unique_lock lock(mutex);
        for (;;) {
            changed.wait(lock, [&] { return stop || task_ready; });
            if (stop) return;
            task_ready = false;
            const int windows = counters.windows;
            for (int i = 0; i < windows; ++i) {
                auto& slot = slots[static_cast<std::size_t>(i % 2)];
                changed.wait(lock, [&] { return cancel.load(std::memory_order_acquire) || slot.state == State::empty; });
                if (cancel.load(std::memory_order_acquire)) break;
                slot = {State::filling, i};
                ++counters.started;
                const int occupied = int(slots[0].state != State::empty) + int(slots[1].state != State::empty);
                counters.max_occupied_slots = std::max(counters.max_occupied_slots, occupied);
                lock.unlock();
                try { produce(context, i, i % 2); }
                catch (...) {
                    lock.lock();
                    failure = std::current_exception();
                    cancel.store(true, std::memory_order_release);
                    slot = {};
                    changed.notify_all();
                    break;
                }
                lock.lock();
                if (cancel.load(std::memory_order_acquire)) { slot = {}; break; }
                slot.state = State::ready;
                ++counters.produced;
                changed.notify_all();
            }
            producer_done = true;
            changed.notify_all();
        }
    }
    void drain() noexcept {
        std::unique_lock lock(mutex);
        if (!active) return;
        cancel.store(true, std::memory_order_release);
        changed.notify_all();
        changed.wait(lock, [&] { return producer_done; });
        slots = {};
        active = false;
        reading = -1;
    }
};

PrefillPipeline::PrefillPipeline(void* context, Produce f) : impl_(std::make_unique<Impl>(context, f)) {}
PrefillPipeline::~PrefillPipeline() = default;
void PrefillPipeline::begin(int windows) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    if (windows < 1 || windows > max_windows || p.active)
        throw std::invalid_argument("prefill pipeline begin state/window bound");
    p.slots = {}; p.next = 0; p.reading = -1; p.failure = {};
    p.counters = {}; p.counters.windows = windows;
    p.cancel.store(false, std::memory_order_release);
    p.active = true; p.producer_done = false; p.task_ready = true;
    p.changed.notify_all();
}
int PrefillPipeline::acquire(int window) {
    auto& p = *impl_;
    std::unique_lock lock(p.mutex);
    if (!p.active || p.reading != -1 || window != p.next || window < 0 || window >= p.counters.windows)
        throw std::invalid_argument("prefill pipeline acquire order/ownership");
    auto& slot = p.slots[static_cast<std::size_t>(window % 2)];
    p.changed.wait(lock, [&] {
        return p.failure || p.cancel.load(std::memory_order_acquire) || p.producer_done ||
               (slot.state == Impl::State::ready && slot.window == window);
    });
    if (p.failure) std::rethrow_exception(p.failure);
    if (p.cancel.load(std::memory_order_acquire)) throw std::runtime_error("prefill pipeline cancelled");
    if (slot.state != Impl::State::ready || slot.window != window)
        throw std::runtime_error("prefill pipeline missing completed window");
    slot.state = Impl::State::reading; p.reading = window; ++p.counters.acquired;
    return window % 2;
}
void PrefillPipeline::release(int window) {
    auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    if (!p.active || window < 0 || window != p.next || window != p.reading)
        throw std::invalid_argument("prefill pipeline release order/ownership");
    auto& slot = p.slots[static_cast<std::size_t>(window % 2)];
    if (slot.state != Impl::State::reading || slot.window != window)
        throw std::invalid_argument("prefill pipeline release slot");
    slot = {}; p.reading = -1; ++p.next; ++p.counters.released;
    p.changed.notify_all();
}
void PrefillPipeline::finish() {
    auto& p = *impl_;
    std::unique_lock lock(p.mutex);
    if (!p.active) throw std::invalid_argument("prefill pipeline finish state");
    if (p.failure) std::rethrow_exception(p.failure);
    if (p.reading != -1 || p.next != p.counters.windows)
        throw std::invalid_argument("prefill pipeline finish before all releases");
    p.changed.wait(lock, [&] { return p.producer_done; });
    if (p.failure) std::rethrow_exception(p.failure);
    p.active = false;
}
void PrefillPipeline::cancel_and_drain() noexcept { impl_->drain(); }
bool PrefillPipeline::cancelled() const noexcept { return impl_->cancel.load(std::memory_order_acquire); }
PrefillPipeline::Stats PrefillPipeline::stats() const {
    const auto& p = *impl_;
    std::lock_guard lock(p.mutex);
    auto result = p.counters;
    result.active = p.active; result.cancelled = p.cancel.load(std::memory_order_acquire);
    result.failed = bool(p.failure);
    return result;
}
std::size_t PrefillPipeline::metadata_bytes() const noexcept { return sizeof(Impl); }
}
