#include "probe.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace qwen {
namespace {
void check(hipError_t s) { if(s!=hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void check(rocblas_status s) { if(s!=rocblas_status_success) throw std::runtime_error("rocBLAS status "+std::to_string(s)); }
void cleanup(hipError_t s) noexcept {
    if(s!=hipSuccess) { std::cerr<<"HIP cleanup: "<<hipGetErrorString(s)<<'\n'; std::abort(); }
}
struct Buffer {
    void *p=nullptr; int device; bool pinned;
    Buffer(std::size_t n,int d,bool host=false):device(d),pinned(host) {
        check(hipSetDevice(d)); if(host) check(hipHostMalloc(&p,n)); else check(hipMalloc(&p,n));
    }
    ~Buffer() { cleanup(hipSetDevice(device)); if(pinned) cleanup(hipHostFree(p)); else cleanup(hipFree(p)); }
    Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
};
struct Stream {
    hipStream_t s{}; int device;
    explicit Stream(int d):device(d) { check(hipSetDevice(d)); check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking)); }
    ~Stream() { cleanup(hipSetDevice(device)); cleanup(hipStreamDestroy(s)); }
    Stream(const Stream&)=delete; Stream& operator=(const Stream&)=delete;
};
struct Event {
    hipEvent_t e{}; int device;
    explicit Event(int d):device(d) { check(hipSetDevice(d)); check(hipEventCreate(&e)); }
    ~Event() { cleanup(hipSetDevice(device)); cleanup(hipEventDestroy(e)); }
    Event(const Event&)=delete; Event& operator=(const Event&)=delete;
};
float interval(Event &begin,Event &end) {
    check(hipEventSynchronize(end.e)); float ms=0; check(hipEventElapsedTime(&ms,begin.e,end.e));
    if(!std::isfinite(ms)||ms<=0) throw std::runtime_error("invalid HIP event duration");
    return ms;
}
__global__ void empty() {}
__global__ void affine(const float *in,float *out,std::size_t n) {
    auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) out[i]=2*in[i]+1;
}
__global__ void pattern(std::uint32_t *out,std::size_t n,std::uint32_t epoch) {
    auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<n) out[i]=static_cast<std::uint32_t>(i)^(epoch*2654435761u);
}
__global__ void consume(std::uint32_t *data,std::size_t n) {
    auto i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x; if(i<n) data[i]^=0xa5a5a5a5u;
}
void launch_check() { check(hipGetLastError()); }
void gemm(int device,int m,int n,int k) {
    Stream stream(device); Buffer a(std::size_t(m)*k*4,device), b(std::size_t(k)*n*4,device), c(std::size_t(m)*n*4,device);
    std::vector<float> ha(std::size_t(m)*k),hb(std::size_t(k)*n),hc(std::size_t(m)*n);
    for(std::size_t i=0;i<ha.size();++i) ha[i]=float(int(i%17)-8)/32;
    for(std::size_t i=0;i<hb.size();++i) hb[i]=float(int(i%13)-6)/32;
    check(hipMemcpy(a.p,ha.data(),ha.size()*4,hipMemcpyHostToDevice));
    check(hipMemcpy(b.p,hb.data(),hb.size()*4,hipMemcpyHostToDevice));
    rocblas_handle handle{}; check(rocblas_create_handle(&handle));
    try {
        check(rocblas_set_stream(handle,stream.s)); const float alpha=1,beta=0;
        auto run=[&]{ check(rocblas_sgemm(handle,rocblas_operation_none,rocblas_operation_none,m,n,k,&alpha,
                                        static_cast<float*>(a.p),m,static_cast<float*>(b.p),k,&beta,static_cast<float*>(c.p),m)); };
        run(); check(hipStreamSynchronize(stream.s)); Event begin(device),end(device);
        check(hipEventRecord(begin.e,stream.s)); for(int i=0;i<5;++i) run(); check(hipEventRecord(end.e,stream.s));
        float ms=interval(begin,end)/5;
        check(hipMemcpy(hc.data(),c.p,hc.size()*4,hipMemcpyDeviceToHost));
        double max_error=0;
        for(std::size_t index=0;index<hc.size();++index) {
            if(!std::isfinite(hc[index])) throw std::runtime_error("GEMM NaN");
            if(m>32 && index%997!=0 && index+1!=hc.size()) continue;
            int row=index%m,col=index/m; double ref=0;
            for(int t=0;t<k;++t) ref+=double(ha[row+t*m])*hb[t+col*k];
            max_error=std::max(max_error,std::abs(ref-hc[index]));
            if(std::abs(ref-hc[index])>1e-4*(1+std::abs(ref))) throw std::runtime_error("GEMM parity");
        }
        std::cout<<"{\"kind\":\"rocblas_sgemm\",\"device\":"<<device<<",\"m\":"<<m<<",\"n\":"<<n<<",\"k\":"<<k
                 <<",\"ms\":"<<ms<<",\"max_abs_error\":"<<max_error<<",\"TFLOPS\":"<<2.0*m*n*k/ms/1e9<<"}\n";
    } catch(...) { rocblas_destroy_handle(handle); throw; }
    check(rocblas_destroy_handle(handle));
}
void device_probe(int device) {
    check(hipSetDevice(device)); hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p,device));
    std::size_t free=0,total=0; check(hipMemGetInfo(&free,&total)); char bdf[32]{}; check(hipDeviceGetPCIBusId(bdf,sizeof(bdf),device));
    std::cout<<"{\"kind\":\"gpu\",\"device\":"<<device<<",\"name\":"<<std::quoted(p.name)<<",\"arch\":"<<std::quoted(p.gcnArchName)
             <<",\"wave_size\":"<<p.warpSize<<",\"CU\":"<<p.multiProcessorCount<<",\"async_engines\":"<<p.asyncEngineCount
             <<",\"total_bytes\":"<<p.totalGlobalMem<<",\"free_bytes\":"<<free<<",\"bdf\":\""<<bdf<<"\"}\n";
    if(!std::string(p.gcnArchName).starts_with("gfx906")||p.warpSize!=64) throw std::runtime_error("not gfx906 wave64");
    constexpr std::size_t maxbytes=64*1024*1024; Stream transfer(device),compute(device);
    Buffer in(maxbytes,device),out(maxbytes,device),pin(maxbytes,device,true);
    std::vector<float> pageable(maxbytes/4),result(maxbytes/4);
    auto *pinned=static_cast<float*>(pin.p);
    for(std::size_t i=0;i<pageable.size();++i) pageable[i]=pinned[i]=float(i%1024);
    Event ready(device),consumed(device),begin(device),end(device);
    for(bool host_pinned:{false,true}) for(std::size_t bytes:{std::size_t(40960),std::size_t(2764800),std::size_t(33177600),maxbytes}) {
        void *host=host_pinned?pin.p:pageable.data();
        check(hipMemcpyAsync(in.p,host,bytes,hipMemcpyHostToDevice,transfer.s)); check(hipStreamSynchronize(transfer.s));
        const auto wall=std::chrono::steady_clock::now();
        check(hipEventRecord(begin.e,transfer.s));
        for(int repeat=0;repeat<10;++repeat) check(hipMemcpyAsync(in.p,host,bytes,hipMemcpyHostToDevice,transfer.s));
        check(hipEventRecord(end.e,transfer.s)); float ms=interval(begin,end)/10;
        double wallms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-wall).count()/10;
        check(hipEventRecord(ready.e,transfer.s)); check(hipStreamWaitEvent(compute.s,ready.e,0));
        affine<<<(bytes/4+255)/256,256,0,compute.s>>>(static_cast<float*>(in.p),static_cast<float*>(out.p),bytes/4); launch_check();
        check(hipEventRecord(consumed.e,compute.s)); check(hipStreamWaitEvent(transfer.s,consumed.e,0));
        check(hipMemcpyAsync(result.data(),out.p,bytes,hipMemcpyDeviceToHost,transfer.s)); check(hipStreamSynchronize(transfer.s));
        for(std::size_t i=0;i<bytes/4;++i) if(result[i]!=2*float(i%1024)+1) throw std::runtime_error("H2D/kernel parity");
        std::cout<<"{\"kind\":\"h2d\",\"device\":"<<device<<",\"pinned\":"<<host_pinned<<",\"bytes\":"<<bytes
                 <<",\"event_ms\":"<<ms<<",\"wall_ms\":"<<wallms<<",\"GB_s\":"<<bytes/ms/1e6<<"}\n";
    }
    check(hipEventRecord(begin.e,compute.s));
    for(int i=0;i<1000;++i) { empty<<<1,64,0,compute.s>>>(); launch_check(); }
    check(hipEventRecord(end.e,compute.s)); float ms=interval(begin,end);
    std::cout<<"{\"kind\":\"empty_launch\",\"device\":"<<device<<",\"completed_us\":"<<ms<<"}\n";
    check(hipEventRecord(begin.e,transfer.s));
    for(int i=0;i<10;++i) check(hipMemcpyAsync(out.p,in.p,maxbytes,hipMemcpyDeviceToDevice,transfer.s));
    check(hipEventRecord(end.e,transfer.s)); ms=interval(begin,end)/10;
    check(hipMemcpy(result.data(),out.p,maxbytes,hipMemcpyDeviceToHost));
    for(std::size_t i=0;i<maxbytes/4;++i) if(result[i]!=float(i%1024)) throw std::runtime_error("D2D copy parity");
    std::cout<<"{\"kind\":\"device_copy\",\"device\":"<<device<<",\"bytes\":"<<maxbytes<<",\"ms\":"<<ms<<",\"GB_s\":"<<maxbytes/ms/1e6<<"}\n";
    gemm(device,32,3,64); gemm(device,2560,128,2560);
}
void peer(int src,int dst) {
    int available=0; check(hipDeviceCanAccessPeer(&available,dst,src));
    if(!available) { std::cout<<"{\"kind\":\"peer\",\"src\":"<<src<<",\"dst\":"<<dst<<",\"available\":false}\n"; return; }
    check(hipSetDevice(dst)); auto status=hipDeviceEnablePeerAccess(src,0);
    if(status!=hipSuccess && status!=hipErrorPeerAccessAlreadyEnabled) check(status);
    constexpr std::size_t bytes=4*1024*1024,n=bytes/4;
    Buffer a(bytes,src),b(bytes,dst); Stream producer(src),consumer(dst); Event ready(src),done(dst),begin(dst),end(dst);
    std::vector<std::uint32_t> host(n);
    for(std::uint32_t epoch=1;epoch<=20;++epoch) {
        check(hipSetDevice(src)); if(epoch>1) check(hipStreamWaitEvent(producer.s,done.e,0));
        pattern<<<(n+255)/256,256,0,producer.s>>>(static_cast<std::uint32_t*>(a.p),n,epoch); launch_check(); check(hipEventRecord(ready.e,producer.s));
        check(hipSetDevice(dst)); check(hipStreamWaitEvent(consumer.s,ready.e,0)); check(hipEventRecord(begin.e,consumer.s));
        check(hipMemcpyPeerAsync(b.p,dst,a.p,src,bytes,consumer.s)); check(hipEventRecord(end.e,consumer.s));
        consume<<<(n+255)/256,256,0,consumer.s>>>(static_cast<std::uint32_t*>(b.p),n); launch_check(); check(hipEventRecord(done.e,consumer.s));
        check(hipMemcpyAsync(host.data(),b.p,bytes,hipMemcpyDeviceToHost,consumer.s)); check(hipStreamSynchronize(consumer.s));
        for(std::size_t i=0;i<n;++i) if(host[i]!=(std::uint32_t(i)^(epoch*2654435761u)^0xa5a5a5a5u)) throw std::runtime_error("P2P producer-copy-consumer parity");
    }
    float ms=interval(begin,end);
    std::cout<<"{\"kind\":\"peer\",\"src\":"<<src<<",\"dst\":"<<dst<<",\"available\":true,\"epochs\":20,\"bytes\":"<<bytes
             <<",\"ms_last\":"<<ms<<",\"GB_s_last\":"<<bytes/ms/1e6<<"}\n";
}
void simultaneous() {
    constexpr std::size_t bytes=32*1024*1024;
    Buffer host(bytes,0,true),a(bytes,0),b(bytes,1); auto *data=static_cast<float*>(host.p);
    for(std::size_t i=0;i<bytes/4;++i) data[i]=float(i%127)*0.001f;
    Stream s0(0),s1(1); Event begin0(0),end0(0),begin1(1),end1(1);
    for(bool cpu:{false,true}) {
        double checksum=0; std::thread worker;
        const auto start=std::chrono::steady_clock::now();
        if(cpu) worker=std::thread([&]{ for(int r=0;r<5;++r) checksum+=cpu_read_fma(data,0,bytes/4); });
        check(hipSetDevice(0)); check(hipEventRecord(begin0.e,s0.s));
        check(hipSetDevice(1)); check(hipEventRecord(begin1.e,s1.s));
        for(int r=0;r<20;++r) {
            check(hipSetDevice(0)); check(hipMemcpyAsync(a.p,host.p,bytes,hipMemcpyHostToDevice,s0.s));
            check(hipSetDevice(1)); check(hipMemcpyAsync(b.p,host.p,bytes,hipMemcpyHostToDevice,s1.s));
        }
        check(hipSetDevice(0)); check(hipEventRecord(end0.e,s0.s));
        check(hipSetDevice(1)); check(hipEventRecord(end1.e,s1.s));
        check(hipSetDevice(0)); auto ms0=interval(begin0,end0);
        check(hipSetDevice(1)); auto ms1=interval(begin1,end1);
        if(worker.joinable()) worker.join();
        double wallms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::vector<float> check_data(bytes/4);
        check(hipSetDevice(0)); check(hipMemcpy(check_data.data(),a.p,bytes,hipMemcpyDeviceToHost));
        if(!std::equal(check_data.begin(),check_data.end(),data)) throw std::runtime_error("dual H2D parity 0");
        check(hipSetDevice(1)); check(hipMemcpy(check_data.data(),b.p,bytes,hipMemcpyDeviceToHost));
        if(!std::equal(check_data.begin(),check_data.end(),data)) throw std::runtime_error("dual H2D parity 1");
        if(cpu && (!std::isfinite(checksum)||checksum<=0)) throw std::runtime_error("CPU overlap checksum");
        std::cout<<"{\"kind\":\"dual_h2d\",\"cpu_read_fma\":"<<cpu<<",\"bytes_per_gpu\":"<<bytes*20
                 <<",\"gpu0_ms\":"<<ms0<<",\"gpu1_ms\":"<<ms1<<",\"completed_wall_ms\":"<<wallms<<",\"checksum\":"<<checksum<<"}\n";
    }
}
}
void gpu_probe() {
    int count=0,runtime=0,driver=0; check(hipGetDeviceCount(&count)); check(hipRuntimeGetVersion(&runtime)); check(hipDriverGetVersion(&driver));
    std::cout<<"{\"kind\":\"hip\",\"runtime\":"<<runtime<<",\"driver\":"<<driver<<",\"device_count\":"<<count<<"}\n";
    if(count!=2) throw std::runtime_error("expected exactly two assigned GPU devices");
    for(int device=0;device<count;++device) device_probe(device);
    peer(0,1); peer(1,0); simultaneous();
}
}
