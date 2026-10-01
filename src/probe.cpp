#include "probe.hpp"
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>
#include <filesystem>
#include <memory>
#include <immintrin.h>
#include <sched.h>

namespace qwen {
__attribute__((target("avx2,fma,f16c")))
double cpu_read_fma(const float *data,std::size_t begin,std::size_t end) {
    __m256 accum=_mm256_setzero_ps();
    const auto factor=_mm256_set1_ps(0.5f);
    for(unsigned repeat=0;repeat<4;++repeat)
        for(auto i=begin;i<end;i+=8) accum=_mm256_fmadd_ps(_mm256_loadu_ps(data+i),factor,accum);
    alignas(32) float lanes[8]; _mm256_store_ps(lanes,accum);
    double sum=0; for(auto x:lanes) sum+=x; return sum;
}
}
namespace {
using Clock = std::chrono::steady_clock;
void cpu_probe() {
    __builtin_cpu_init();
    const bool avx2 = __builtin_cpu_supports("avx2");
    const bool fma = __builtin_cpu_supports("fma");
    const bool f16c = __builtin_cpu_supports("f16c");
    cpu_set_t allowed; CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed)) throw std::runtime_error("sched_getaffinity");
    std::set<std::pair<int,int>> cores;
    std::vector<int> representatives;
    for (int cpu=0; cpu<CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu,&allowed)) {
        const auto root = "/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
        int package=-1, core=-1;
        if (!(std::ifstream(root+"physical_package_id")>>package) || !(std::ifstream(root+"core_id")>>core))
            throw std::runtime_error("CPU topology read");
        if (cores.emplace(package,core).second) representatives.push_back(cpu);
    }
    unsigned nodes=0;
    for (auto &entry:std::filesystem::directory_iterator("/sys/devices/system/node")) {
        const auto name=entry.path().filename().string();
        if (name.starts_with("node") && name.size()>4 && name.find_first_not_of("0123456789",4)==std::string::npos) ++nodes;
    }
    std::cout<<"{\"kind\":\"cpu\",\"avx2\":"<<avx2<<",\"fma\":"<<fma<<",\"f16c\":"<<f16c
             <<",\"physical_cores\":"<<cores.size()<<",\"logical_cpus\":"<<CPU_COUNT(&allowed)<<",\"numa_nodes\":"<<nodes<<"}\n";
    if (!avx2 || !fma || !f16c) throw std::runtime_error("target requires AVX2/FMA/F16C");
    // > LLC. First-touch and affinity on actual physical cores; this is read+FMA,
    // not a claim about quantized expert GEMV performance (measured in R1).
    constexpr std::size_t n=32*1024*1024;
    for (unsigned threads : {1u, 8u, 16u}) {
        if (threads>representatives.size()) continue;
        std::unique_ptr<float[]> data(new float[n]); std::vector<double> sums(threads);
        std::vector<std::thread> workers;
        for (unsigned t=0;t<threads;++t) workers.emplace_back([&,t] {
            cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(representatives[t], &mask);
            if (pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)) std::terminate();
            const auto begin=n*t/threads,end=n*(t+1)/threads;
            for (auto i=begin;i<end;++i) data[i]=static_cast<float>(i%127)*0.001f;
        });
        for (auto &w:workers) w.join();
        workers.clear();
        const auto start=Clock::now();
        for (unsigned t=0;t<threads;++t) workers.emplace_back([&,t] {
            cpu_set_t mask; CPU_ZERO(&mask); CPU_SET(representatives[t], &mask);
            if (pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask)) std::terminate();
            sums[t]=qwen::cpu_read_fma(data.get(),n*t/threads,n*(t+1)/threads);
        });
        for (auto &w:workers) w.join();
        const double seconds=std::chrono::duration<double>(Clock::now()-start).count();
        double checksum=0; for (auto x:sums) checksum+=x;
        if (!std::isfinite(checksum) || checksum<=0) throw std::runtime_error("RAM checksum");
        std::cout<<"{\"kind\":\"ram_read_fma\",\"threads\":"<<threads<<",\"bytes\":"<<n*4*4
                 <<",\"seconds\":"<<seconds<<",\"GB_s\":"<<n*4*4/seconds/1e9<<",\"checksum\":"<<checksum<<"}\n";
    }
}
}
int main() {
    try {
        std::cout<<"{\"kind\":\"source\",\"revision\":\""<<CORE_REVISION<<"\",\"dirty\":"<<CORE_DIRTY<<"}\n";
        cpu_probe(); qwen::gpu_probe();
        std::cout<<"{\"kind\":\"probe_complete\",\"passed\":true}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr<<"core-probe: "<<e.what()<<'\n'; return 1; }
}
