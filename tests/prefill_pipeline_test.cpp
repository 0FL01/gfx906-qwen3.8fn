#include "prefill_pipeline.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <thread>

// All-thread C++ new counter, not a malloc/syscall or thread-stack measurement.
namespace heap_monitor {
std::atomic<bool> enabled{false};
std::atomic<std::size_t> calls{0};
void count() noexcept { if (enabled.load()) ++calls; }
void* allocate(std::size_t n) {
    count(); if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void* aligned(std::size_t n, std::size_t a) {
    count(); void* p = nullptr;
    if (posix_memalign(&p, a, n ? n : 1) == 0) return p;
    throw std::bad_alloc();
}
}
#ifndef PREFILL_TEST_DISABLE_HEAP_COUNTER
void* operator new(std::size_t n) { return heap_monitor::allocate(n); }
void* operator new[](std::size_t n) { return heap_monitor::allocate(n); }
void* operator new(std::size_t n, std::align_val_t a) { return heap_monitor::aligned(n, std::size_t(a)); }
void* operator new[](std::size_t n, std::align_val_t a) { return heap_monitor::aligned(n, std::size_t(a)); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
#endif

namespace {
using qwen::PrefillPipeline;
std::size_t checks = 0, cases = 0, rejects = 0;
void require(bool x, const char* what) { ++checks; if (!x) throw std::runtime_error(what); }
template<class F> void rejects_call(F f) {
    try { f(); } catch (const std::invalid_argument&) { ++rejects; return; }
    throw std::runtime_error("missing invalid-argument rejection");
}
template<class F> void until(F predicate) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > end) throw std::runtime_error("coordinator test timeout");
        std::this_thread::yield();
    }
}
struct Context {
    std::array<std::array<std::uint64_t, 64>, 2> payload{};
    PrefillPipeline* pipe = nullptr;
    std::thread::id owner = std::this_thread::get_id();
    std::atomic<int> calls{0};
    std::atomic<bool> entered{false}, returned{false}, wrong_thread{false};
    int epoch = 0, fail_at = -1;
    bool block = false;
    static std::uint64_t value(int epoch, int window, int element) {
        return std::uint64_t(epoch) * 1000000000 + std::uint64_t(window) * 1000 + unsigned(element);
    }
    static void produce(void* ptr, int window, int slot) {
        auto& c = *static_cast<Context*>(ptr);
        if (std::this_thread::get_id() == c.owner) c.wrong_thread = true;
        ++c.calls;
        if (c.block) {
            c.entered = true;
            while (!c.pipe->cancelled()) std::this_thread::yield();
            c.returned = true;
            return;
        }
        if (window == c.fail_at) throw std::runtime_error("injected producer failure");
        for (int j = 0; j < 64; ++j) c.payload[slot][j] = value(c.epoch, window, j);
    }
};
void run(Context& c, PrefillPipeline& p, int n) {
    ++c.epoch; c.calls = 0;
    p.begin(n);
    for (int i = 0; i < n; ++i) {
        const int slot = p.acquire(i);
        require(slot == i % 2, "slot identity");
        const auto before = c.payload[slot];
        for (int j = 0; j < 64; ++j) require(before[j] == Context::value(c.epoch, i, j), "payload/order/epoch");
        std::this_thread::yield();
        require(c.payload[slot] == before, "borrowed slot overwritten");
        p.release(i);
    }
    p.finish();
    const auto s = p.stats();
    require(!s.active && !s.failed && !s.cancelled, "clean completion");
    require(s.windows == n && s.started == n && s.produced == n && s.acquired == n && s.released == n, "completion counts");
    require(s.max_occupied_slots >= 1 && s.max_occupied_slots <= 2, "bounded occupancy");
    require(c.calls == n && !c.wrong_thread, "one persistent worker callback");
    ++cases;
}
}
int main() {
    try {
        Context c;
        rejects_call([&] { PrefillPipeline bad(nullptr, Context::produce); });
        rejects_call([&] { PrefillPipeline bad(&c, nullptr); });
        PrefillPipeline p(&c, Context::produce); c.pipe = &p;
        require(p.metadata_bytes() > 0 && p.metadata_bytes() < 4096, "bounded fixed metadata");
        for (int n : {1, 2, 3, 17, 256, 4096, 3, 1}) run(c, p, n);
        const auto before = p.stats();
        for (int n : {-1, 0, 4097}) rejects_call([&] { p.begin(n); });
        rejects_call([&] { p.acquire(0); });
        rejects_call([&] { p.release(0); });
        rejects_call([&] { p.finish(); });
        require(p.stats() == before, "idle rejects mutated state");

        p.begin(17); until([&] { return p.stats().produced == 2; });
        auto saturated = p.stats();
        require(saturated.started == 2 && saturated.max_occupied_slots == 2, "producer exceeded two slots");
        rejects_call([&] { p.begin(1); });
        rejects_call([&] { p.acquire(1); });
        rejects_call([&] { p.release(0); });
        rejects_call([&] { p.finish(); });
        require(p.stats() == saturated, "blocked invalid calls changed metadata");
        int slot = p.acquire(0); auto lease = c.payload[slot];
        rejects_call([&] { p.acquire(0); });
        rejects_call([&] { p.release(1); });
        rejects_call([&] { p.release(-1); });
        rejects_call([&] { p.finish(); });
        for (int i = 0; i < 100; ++i) std::this_thread::yield();
        require(c.payload[slot] == lease && p.stats().started == 2, "lease/backpressure");
        p.cancel_and_drain();
        require(!p.stats().active && p.stats().cancelled && !p.stats().failed, "cancel leased slot");
        p.cancel_and_drain(); run(c, p, 17); ++cases;

        for (int fail : {0, 5, 16}) {
            c.fail_at = fail; p.begin(17); bool caught = false;
            try {
                for (int i = 0; i < 17; ++i) { (void)p.acquire(i); p.release(i); }
                p.finish();
            } catch (const std::runtime_error&) { caught = true; }
            require(caught && p.stats().failed && p.cancelled(), "producer exception did not surface");
            p.cancel_and_drain();
            require(!p.stats().active, "failed producer not drained");
            c.fail_at = -1; run(c, p, 3); ++cases;
        }

        c.block = true; p.begin(17); until([&] { return c.entered.load(); });
        p.cancel_and_drain();
        require(c.returned && p.stats().produced == 0 && !p.stats().active, "active callback not drained");
        c.block = false; run(c, p, 17); ++cases;

        { Context d; auto owner = std::make_unique<PrefillPipeline>(&d, Context::produce);
          d.pipe = owner.get(); d.block = true; owner->begin(17);
          until([&] { return d.entered.load(); }); owner.reset();
          require(d.returned, "destructor failed to join callback"); ++cases; }
        { Context d; PrefillPipeline full(&d, Context::produce); d.pipe = &full;
          full.begin(17); until([&] { return full.stats().produced == 2; });
          full.cancel_and_drain(); require(full.stats().started == 2, "full queue cancel starts more work"); ++cases; }

        heap_monitor::calls = 0; heap_monitor::enabled = true;
        for (int i = 0; i < 64; ++i) run(c, p, 17);
        heap_monitor::enabled = false;
        require(heap_monitor::calls == 0, "normal batches allocate C++ heap");
#ifdef PREFILL_TEST_DISABLE_HEAP_COUNTER
        constexpr bool measured_allocations = false;
#else
        constexpr bool measured_allocations = true;
#endif
        std::cout << "{\"kind\":\"prefill_pipeline_cpu\",\"passed\":true,\"cases\":" << cases
                  << ",\"checks\":" << checks << ",\"rejects\":" << rejects
                  << ",\"hot_new_calls\":" << heap_monitor::calls
                  << ",\"allocations_measured\":" << (measured_allocations ? "true" : "false")
                  << ",\"metadata_bytes\":" << p.metadata_bytes()
                  << ",\"payload_slots\":2,\"max_windows\":4096,\"device_work_tested\":false}\n";
        return 0;
    } catch (const std::exception& e) {
        heap_monitor::enabled = false; std::cerr << e.what() << '\n'; return 1;
    }
}
