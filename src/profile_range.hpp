#pragma once

// Manual diagnostic builds only. No ROCTX dependency or calls in default builds.
#ifdef CORE_PROFILE_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>
#endif
namespace qwen {
class ProfileRange {
#ifdef CORE_PROFILE_ROCTX
    bool active_ = true;
#endif
public:
    explicit ProfileRange(const char* name) noexcept {
#ifdef CORE_PROFILE_ROCTX
        (void)roctxRangePushA(name);
#else
        (void)name;
#endif
    }
    ProfileRange(const ProfileRange&) = delete;
    ProfileRange& operator=(const ProfileRange&) = delete;
    void end() noexcept {
#ifdef CORE_PROFILE_ROCTX
        if (active_) { (void)roctxRangePop(); active_ = false; }
#endif
    }
    ~ProfileRange() { end(); }
};
} // namespace qwen
