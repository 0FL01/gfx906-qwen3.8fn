#include "kv.hpp"
#include "hip/kv.cuh"
#include <hip/hip_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void check(hipError_t s) { if(s!=hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void cleanup(hipError_t s) noexcept {
    if(s!=hipSuccess) { std::cerr<<"KV cleanup: "<<hipGetErrorString(s)<<'\n'; std::abort(); }
}
struct Fixture {
    int device;
    hipStream_t stream{}; hipEvent_t begin{},end{};
    float *x=nullptr,*y=nullptr,*z=nullptr;
    qwen::Q4_0 *q=nullptr; int *error=nullptr;
    std::vector<void*> owned;
    explicit Fixture(int d):device(d) {}
    ~Fixture() {
        cleanup(hipSetDevice(device));
        if(stream) cleanup(hipStreamSynchronize(stream));
        for(auto p:owned) cleanup(hipFree(p));
        if(begin) cleanup(hipEventDestroy(begin));
        if(end) cleanup(hipEventDestroy(end));
        if(stream) cleanup(hipStreamDestroy(stream));
    }
    template<class T> void allocate(T *&p,std::size_t n) {
        void *value=nullptr; check(hipMalloc(&value,n*sizeof(T)));
        try { owned.push_back(value); } catch(...) { cleanup(hipFree(value)); throw; }
        p=static_cast<T*>(value);
    }
    void initialize() {
        check(hipSetDevice(device)); hipDeviceProp_t props{}; check(hipGetDeviceProperties(&props,device));
        if(!std::string(props.gcnArchName).starts_with("gfx906") || props.warpSize!=64) throw std::runtime_error("KV requires gfx906 wave64");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        check(hipEventCreate(&begin)); check(hipEventCreate(&end));
        constexpr int size=24*256*128;
        allocate(x,size); allocate(y,size); allocate(z,size); allocate(q,size/32); allocate(error,1);
    }
    void upload(std::span<const float> data) {
        check(hipMemcpyAsync(x,data.data(),data.size_bytes(),hipMemcpyHostToDevice,stream));
        check(hipMemsetAsync(error,0,sizeof(int),stream));
    }
    void valid() {
        int flag=0; check(hipMemcpyAsync(&flag,error,sizeof(flag),hipMemcpyDeviceToHost,stream));
        check(hipStreamSynchronize(stream)); if(flag) throw std::runtime_error("unexpected GPU Q4 error");
    }
    template<class F> double time(F launch,int repeats) {
        launch(); valid();
        check(hipEventRecord(begin,stream));
        for(int r=0;r<repeats;++r) launch();
        check(hipEventRecord(end,stream)); check(hipEventSynchronize(end));
        float ms=0; check(hipEventElapsedTime(&ms,begin,end));
        if(!std::isfinite(ms) || ms<=0) throw std::runtime_error("KV event duration");
        valid(); return ms/repeats;
    }
    void packing(std::span<const float> data) {
        std::vector<qwen::Q4_0> expected(data.size()/32),got(expected.size());
        std::vector<float> ref(data.size()),decoded(data.size());
        qwen::quantize_q4(data,expected); qwen::dequantize_q4(expected,ref);
        upload(data); qwen::launch_quantize_q4(x,q,int(data.size()),error,stream);
        qwen::launch_dequantize_q4(q,z,int(data.size()),stream);
        check(hipMemcpyAsync(got.data(),q,got.size()*sizeof(qwen::Q4_0),hipMemcpyDeviceToHost,stream));
        check(hipMemcpyAsync(decoded.data(),z,decoded.size()*sizeof(float),hipMemcpyDeviceToHost,stream));
        valid();
        if(std::memcmp(got.data(),expected.data(),got.size()*sizeof(qwen::Q4_0))) throw std::runtime_error("CPU/GPU Q4 byte mismatch");
        if(std::memcmp(decoded.data(),ref.data(),ref.size()*sizeof(float))) throw std::runtime_error("CPU/GPU Q4 dequantization mismatch");
    }
};
// Diagnostic A/B only: exact serial scan and nibble order of mx
// cpy-utils.cuh::quantize_f32_q4_0_block, dcd685... (MIT, third_party/mx-LICENSE).
// Not a second runtime backend; cooperative packing remains the candidate.
__global__ void serial_pack(const float *input,qwen::Q4_0 *output,int blocks) {
    const int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=blocks) return;
    const float *x=input+b*32; float amax=0,maximum=0;
    for(int i=0;i<32;++i) if(fabsf(x[i])>amax) { amax=fabsf(x[i]); maximum=x[i]; }
    const float d=maximum/-8.0f,inv=d?1.0f/d:0.0f;
    output[b].d=__half_as_ushort(__float2half_rn(d));
    for(int i=0;i<16;++i) output[b].qs[i]=min(15,int(x[i]*inv+8.5f))|(min(15,int(x[i+16]*inv+8.5f))<<4);
}
void edges(Fixture &f) {
    std::vector<float> x(32*13,0);
    f.packing(x); std::fill(x.begin(),x.end(),-0.0f); f.packing(x);
    for(float scale:{1.0003f,8.0f,std::ldexp(8.0f,-24),std::ldexp(8.0f,-14),500000.0f}) {
        for(std::size_t i=0;i<x.size();++i) x[i]=scale*float(int(i%32)-16)/16;
        f.packing(x);
        for(std::size_t b=0;b<x.size()/32;++b) { x[b*32]=scale; x[b*32+31]=-scale; }
        f.packing(x);
    }
    // Dense random FP32 blocks, not just powers of two, test reciprocal/code ABI.
    x.resize(32*4096); std::uint32_t random=1;
    for(auto &v:x) { random=random*1664525+1013904223; v=float(std::int32_t(random))/2147483648.0f; }
    f.packing(x);
    x.assign(32,0); x[0]=1.0003f; x[1]=(x[0]/-8)*(2.5f-.0001f); f.packing(x);
    for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),
                   std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::max()}) {
        x.assign(64,bad); f.upload(x); qwen::launch_quantize_q4(f.x,f.q,64,f.error,f.stream);
        int flag=0; check(hipMemcpyAsync(&flag,f.error,sizeof(flag),hipMemcpyDeviceToHost,f.stream)); check(hipStreamSynchronize(f.stream));
        if(flag!=(std::isfinite(bad)?2:1)) throw std::runtime_error("GPU invalid Q4 flag");
        x.assign(64,.125f); f.packing(x); // A failed write cannot poison later use.
    }
}
void measure(Fixture &f,int columns,const char *role,int heads,int group,bool pack) {
    const int elements=heads*256*columns;
    std::vector<float> x(elements),rotated(elements),got(elements),ref(elements);
    for(int i=0;i<elements;++i) x[i]=.7f*std::sin(float(i)*.137f)+.2f*std::cos(float(i)*.071f);
    qwen::hadamard(x,rotated,group); std::vector<qwen::Q4_0> expected(elements/32),codes(expected.size());
    if(pack) { qwen::quantize_q4(rotated,expected); qwen::dequantize_q4(expected,ref); }
    f.upload(x);
    const auto launch=[&] {
        qwen::launch_hadamard(f.x,f.y,elements,group,f.stream);
        if(pack) { qwen::launch_quantize_q4(f.y,f.q,elements,f.error,f.stream); qwen::launch_dequantize_q4(f.q,f.z,elements,f.stream); }
    };
    launch();
    check(hipMemcpyAsync(got.data(),f.y,got.size()*sizeof(float),hipMemcpyDeviceToHost,f.stream)); f.valid();
    if(std::memcmp(got.data(),rotated.data(),got.size()*sizeof(float))) throw std::runtime_error("GPU Hadamard byte parity");
    // Same transform in place is needed for inverse V output before gate.
    qwen::launch_hadamard(f.y,f.y,elements,group,f.stream);
    qwen::hadamard(rotated,ref,group);
    check(hipMemcpyAsync(got.data(),f.y,got.size()*sizeof(float),hipMemcpyDeviceToHost,f.stream)); f.valid();
    if(std::memcmp(got.data(),ref.data(),got.size()*sizeof(float))) throw std::runtime_error("GPU in-place inverse Hadamard parity");
    launch();
    if(pack) {
        qwen::dequantize_q4(expected,ref);
        check(hipMemcpyAsync(codes.data(),f.q,codes.size()*sizeof(qwen::Q4_0),hipMemcpyDeviceToHost,f.stream));
        check(hipMemcpyAsync(got.data(),f.z,got.size()*sizeof(float),hipMemcpyDeviceToHost,f.stream)); f.valid();
        if(std::memcmp(codes.data(),expected.data(),codes.size()*sizeof(qwen::Q4_0)) || std::memcmp(got.data(),ref.data(),got.size()*sizeof(float)))
            throw std::runtime_error("rotated Q4 KV roundtrip parity");
    }
    constexpr int repeats=100;
    const double ms=f.time(launch,repeats);
    std::cout<<"{\"kind\":\"kv_gpu\",\"device\":"<<f.device<<",\"columns\":"<<columns<<",\"role\":\""<<role
             <<"\",\"group\":"<<group<<",\"elements\":"<<elements<<",\"repeats\":"<<repeats<<",\"resident_ms\":"<<ms
             <<",\"max_abs_error\":0,\"byte_parity\":true}\n";
    if(std::strcmp(role,"k")!=0) return;
    // Both variants consume identical already-rotated inputs, and each is
    // checked against the same packed bytes after timing (not just final B).
    check(hipMemcpyAsync(f.x,rotated.data(),rotated.size()*sizeof(float),hipMemcpyHostToDevice,f.stream));
    for(bool cooperative:{false,true,false}) {
        auto pack_launch=[&] {
            if(cooperative) qwen::launch_quantize_q4(f.x,f.q,elements,f.error,f.stream);
            else { serial_pack<<<(elements/32+255)/256,256,0,f.stream>>>(f.x,f.q,elements/32); check(hipGetLastError()); }
        };
        const double time=f.time(pack_launch,repeats);
        check(hipMemcpyAsync(codes.data(),f.q,codes.size()*sizeof(qwen::Q4_0),hipMemcpyDeviceToHost,f.stream)); f.valid();
        if(std::memcmp(codes.data(),expected.data(),codes.size()*sizeof(qwen::Q4_0))) throw std::runtime_error("A/B/A packed bytes");
        std::cout<<"{\"kind\":\"kv_pack\",\"device\":"<<f.device<<",\"columns\":"<<columns<<",\"layout\":\""
                 <<(cooperative?"cooperative":"mx_serial")<<"\",\"repeats\":"<<repeats<<",\"resident_ms\":"<<time<<",\"byte_parity\":true}\n";
    }
}
}
int main() {
    try {
        std::cout<<std::setprecision(9)<<"{\"kind\":\"kv_source\",\"revision\":\""<<CORE_REVISION<<"\",\"dirty\":"<<CORE_DIRTY
                 <<",\"kv\":\"Q4_0\",\"scope\":\"synthetic resident KV transforms; no attention or full inference\"}\n";
        int count=0; check(hipGetDeviceCount(&count)); if(count!=2) throw std::runtime_error("expected two gfx906 GPUs");
        for(int d=0;d<count;++d) {
            Fixture fixture(d); fixture.initialize(); edges(fixture);
            for(int columns:{1,2,3,128}) {
                measure(fixture,columns,"q",24,256,false);
                measure(fixture,columns,"k",2,256,true);
                measure(fixture,columns,"v",2,64,true);
            }
        }
        std::cout<<"{\"kind\":\"kv_complete\",\"passed\":true}\n";
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
