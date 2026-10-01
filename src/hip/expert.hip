#include "gpu_expert.hpp"
#include "dpp.cuh"
#include <hip/hip_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qwen {
namespace {
void check(hipError_t status) {
    if(status!=hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void cleanup(hipError_t status) noexcept {
    if(status!=hipSuccess) { std::cerr<<"HIP expert cleanup: "<<hipGetErrorString(status)<<'\n'; std::abort(); }
}
float elapsed(hipEvent_t begin,hipEvent_t end) {
    check(hipEventSynchronize(end)); float ms=0; check(hipEventElapsedTime(&ms,begin,end));
    if(!std::isfinite(ms)||ms<=0) throw std::runtime_error("expert event duration");
    return ms;
}
__device__ float from_half(std::uint16_t bits) { return __half2float(__ushort_as_half(bits)); }
__device__ std::uint16_t to_half(float value) { return __half_as_ushort(__float2half_rn(value)); }

// ADAPT mx quantize.cu::quantize_q8_1, same raw-sum/half/roundf ABI.
__global__ void quantize(const float *input,Q8_1 *output,int size,int *error) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=size) return; // size divisible by 32; no partial reduction group.
    float x=input[index];
    if(!isfinite(x)) { atomicOr(error,1); x=0; }
    float amax=fabsf(x);
    for(int mask=1;mask<32;mask*=2) amax=fmaxf(amax,__shfl_xor(amax,mask,32));
    const float sum=sum32(x),d=amax/127.0f;
    const auto hd=to_half(d),hs=to_half(sum);
    const bool valid=isfinite(sum)&&(hs&0x7c00)!=0x7c00&&(hd&0x7c00)!=0x7c00&&(amax==0 || hd!=0);
    if(!valid) atomicOr(error,2);
    auto &block=output[index/32];
    block.qs[index%32]=valid&&amax!=0?static_cast<std::int8_t>(roundf(x/d)):0;
    if(index%32==0) { block.d=valid?hd:0; block.s=valid&&amax!=0?hs:0; }
}
__global__ void silu_product(const float *gate,const float *up,float *output,int size) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index<size) output[index]=(gate[index]/(1.0f+expf(-gate[index])))*up[index];
}
__device__ __forceinline__ std::uint32_t word2(const std::uint8_t *bytes,int index) {
    const auto *halfwords=reinterpret_cast<const std::uint16_t*>(bytes);
    return std::uint32_t(halfwords[index*2])|(std::uint32_t(halfwords[index*2+1])<<16);
}

// ADAPT mx canonical SDOT4 arithmetic + donor multi-column weight reuse.
// Reinstinct's 40-byte Q8, FP32 d and QUANTIZED sum correction are NOT imported:
// all variants here retain our 36-byte mx Q8_1 and original raw sum correction.
// Planar Q4_0 byte layout matches mx q8_repack (codes then scales), no padding
// for our 80/20 input blocks. Furnace repack-gcn does not support these types.
// 4 wave64s x 2 output rows, each loaded weight reused across 1..3 columns.
template<bool Offset,bool Planar> __global__ void matrix(
        const std::uint8_t *weights,const Q8_1 *input,float *output,int blocks,int rows,int columns) {
    const int wave=threadIdx.x/64,lane=threadIdx.x%64;
    const int row0=blockIdx.x*8+wave*2,column0=blockIdx.y*3;
    float acc[2][3]{};
    for(int k=lane;k<blocks;k+=64) {
        std::uint32_t codes[2][4]{}; float d[2]{},m[2]{};
        for(int r=0;r<2;++r) {
            const int row=row0+r;
            if(row>=rows) continue;
            const auto block=std::size_t(row)*blocks+k;
            if constexpr(Planar && !Offset) {
                const uint4 values=reinterpret_cast<const uint4*>(weights)[block];
                codes[r][0]=values.x; codes[r][1]=values.y; codes[r][2]=values.z; codes[r][3]=values.w;
                d[r]=from_half(reinterpret_cast<const std::uint16_t*>(weights+std::size_t(rows)*blocks*16)[block]);
            } else {
                const auto *base=weights+block*(Offset?20:18);
                d[r]=from_half(*reinterpret_cast<const std::uint16_t*>(base));
                if constexpr(Offset) m[r]=from_half(*reinterpret_cast<const std::uint16_t*>(base+2));
                for(int j=0;j<4;++j) codes[r][j]=word2(base+(Offset?4:2),j);
            }
        }
        for(int c=0;c<3 && column0+c<columns;++c) {
            const auto &x=input[std::size_t(column0+c)*blocks+k];
            const auto *q8=reinterpret_cast<const int*>(x.qs);
            const float dx=from_half(x.d),sx=from_half(x.s);
            for(int r=0;r<2;++r) {
                if(row0+r>=rows) continue;
                int dot=0;
                for(int j=0;j<4;++j) {
                    dot=__builtin_amdgcn_sdot4(codes[r][j]&0x0f0f0f0f,q8[j],dot,false);
                    dot=__builtin_amdgcn_sdot4((codes[r][j]>>4)&0x0f0f0f0f,q8[j+4],dot,false);
                }
                if constexpr(Offset) {
                    const float dd=from_half(to_half(d[r]*dx)),ms=from_half(to_half(m[r]*sx));
                    acc[r][c]+=float(dot)*dd+ms;
                } else acc[r][c]+=d[r]*(float(dot)*dx-8.0f*sx);
            }
        }
    }
    for(int r=0;r<2;++r) for(int c=0;c<3;++c) {
        const float value=sum64(acc[r][c]);
        if(lane==0 && row0+r<rows && column0+c<columns) output[std::size_t(column0+c)*rows+row0+r]=value;
    }
}
}

struct GpuExpert::Impl {
    int device,capacity,k,h;
    bool planar;
    TensorType down_type;
    hipStream_t stream{};
    hipEvent_t begin{},end{};
    std::vector<void*> buffers;
    std::uint8_t *gate=nullptr,*up=nullptr,*down=nullptr;
    float *x=nullptr,*g=nullptr,*u=nullptr,*mid=nullptr,*y=nullptr;
    Q8_1 *qx=nullptr,*qm=nullptr;
    int *error=nullptr;
    double upload_ms=0;
    Impl(int d,int n,int input,int hidden,bool p,TensorType dt):device(d),capacity(n),k(input),h(hidden),planar(p),down_type(dt) {}
    ~Impl() {
        cleanup(hipSetDevice(device));
        if(stream) cleanup(hipStreamSynchronize(stream));
        for(auto p:buffers) cleanup(hipFree(p));
        if(begin) cleanup(hipEventDestroy(begin));
        if(end) cleanup(hipEventDestroy(end));
        if(stream) cleanup(hipStreamDestroy(stream));
    }
    template<class T> void allocate(T *&pointer,std::size_t count) {
        void *value=nullptr; check(hipMalloc(&value,count*sizeof(T)));
        try { buffers.push_back(value); } catch(...) { cleanup(hipFree(value)); throw; }
        pointer=static_cast<T*>(value);
    }
    void weights(std::uint8_t *&pointer,QMatrix matrix) {
        allocate(pointer,matrix.weights.size());
        if(planar && matrix.type==TensorType::Q4_0) {
            const std::size_t blocks=matrix.weights.size()/18;
            std::vector<std::byte> repack(matrix.weights.size());
            for(std::size_t b=0;b<blocks;++b) {
                std::memcpy(repack.data()+b*16,matrix.weights.data()+b*18+2,16);
                std::memcpy(repack.data()+blocks*16+b*2,matrix.weights.data()+b*18,2);
            }
            check(hipMemcpyAsync(pointer,repack.data(),repack.size(),hipMemcpyHostToDevice,stream));
            check(hipStreamSynchronize(stream)); // Temporary slot pack stays alive until DMA completes.
        } else check(hipMemcpyAsync(pointer,matrix.weights.data(),matrix.weights.size(),hipMemcpyHostToDevice,stream));
    }
    void launch(const std::uint8_t *weights,TensorType type,const Q8_1 *input,float *output,int input_dim,int output_dim,int columns) {
        const dim3 grid((output_dim+7)/8,(columns+2)/3);
        if(type==TensorType::Q4_1) matrix<true,false><<<grid,256,0,stream>>>(weights,input,output,input_dim/32,output_dim,columns);
        else if(planar) matrix<false,true><<<grid,256,0,stream>>>(weights,input,output,input_dim/32,output_dim,columns);
        else matrix<false,false><<<grid,256,0,stream>>>(weights,input,output,input_dim/32,output_dim,columns);
        check(hipGetLastError());
    }
    void quant(const float *input,Q8_1 *output,int count) {
        quantize<<<(count+255)/256,256,0,stream>>>(input,output,count,error); check(hipGetLastError());
    }
    void check_error() {
        int value=0; check(hipMemcpyAsync(&value,error,sizeof(value),hipMemcpyDeviceToHost,stream));
        check(hipStreamSynchronize(stream));
        if(value) throw std::invalid_argument("GPU activation not representable in Q8_1");
    }
    void forward(int columns) {
        quant(x,qx,k*columns);
        launch(gate,TensorType::Q4_0,qx,g,k,h,columns); launch(up,TensorType::Q4_0,qx,u,k,h,columns);
        silu_product<<<(h*columns+255)/256,256,0,stream>>>(g,u,mid,h*columns); check(hipGetLastError());
        quant(mid,qm,h*columns); launch(down,down_type,qm,y,h,k,columns);
    }
};

GpuExpert::GpuExpert(QMatrix gate,QMatrix up,QMatrix down,int device,int max_columns,bool planar) {
    if(max_columns<=0 || max_columns>1024 || gate.input<=0 || gate.input>16384 || gate.input%32
            || gate.output<=0 || gate.output>16384 || gate.output%32 || up.input!=gate.input || up.output!=gate.output
            || down.input!=gate.output || down.output!=gate.input || gate.type!=TensorType::Q4_0 || up.type!=TensorType::Q4_0
            || (down.type!=TensorType::Q4_0 && down.type!=TensorType::Q4_1)) throw std::invalid_argument("GPU expert shape/type");
    for(auto view:{gate,up,down})
        if(view.weights.size()!=std::size_t(view.input/32)*view.output*type_layout(view.type).block_bytes)
            throw std::invalid_argument("GPU expert byte size");
    check(hipSetDevice(device)); hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties,device));
    if(!std::string(properties.gcnArchName).starts_with("gfx906") || properties.warpSize!=64) throw std::runtime_error("expert requires gfx906 wave64");
    auto value=std::make_unique<Impl>(device,max_columns,gate.input,gate.output,planar,down.type);
    check(hipStreamCreateWithFlags(&value->stream,hipStreamNonBlocking));
    check(hipEventCreate(&value->begin)); check(hipEventCreate(&value->end));
    value->allocate(value->x,std::size_t(value->k)*max_columns); value->allocate(value->y,std::size_t(std::max(value->k,value->h))*max_columns);
    value->allocate(value->g,std::size_t(value->h)*max_columns); value->allocate(value->u,std::size_t(value->h)*max_columns);
    value->allocate(value->mid,std::size_t(value->h)*max_columns);
    value->allocate(value->qx,std::size_t(std::max(value->k,value->h)/32)*max_columns); value->allocate(value->qm,std::size_t(value->h/32)*max_columns);
    value->allocate(value->error,1);
    const auto start=std::chrono::steady_clock::now();
    value->weights(value->gate,gate); value->weights(value->up,up); value->weights(value->down,down);
    check(hipStreamSynchronize(value->stream));
    value->upload_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    impl_=std::move(value);
}
GpuExpert::~GpuExpert()=default;
void GpuExpert::check_quantization(std::span<const float> input,std::span<const Q8_1> expected) {
    auto &v=*impl_;
    if(input.empty() || input.size()>std::size_t(v.k)*v.capacity || input.size()%32 || expected.size()!=input.size()/32)
        throw std::invalid_argument("GPU quantization span");
    check(hipSetDevice(v.device)); check(hipMemsetAsync(v.error,0,sizeof(int),v.stream));
    check(hipMemcpyAsync(v.x,input.data(),input.size_bytes(),hipMemcpyHostToDevice,v.stream));
    v.quant(v.x,v.qx,static_cast<int>(input.size()));
    std::vector<Q8_1> result(expected.size()); check(hipMemcpyAsync(result.data(),v.qx,result.size()*sizeof(Q8_1),hipMemcpyDeviceToHost,v.stream));
    v.check_error();
    if(std::memcmp(result.data(),expected.data(),expected.size_bytes())) throw std::runtime_error("CPU/GPU Q8_1 ABI mismatch");
}
void GpuExpert::linear(bool down,bool up,std::span<const Q8_1> input,int columns,std::span<float> output) {
    auto &v=*impl_; const int in=down?v.h:v.k,out=down?v.k:v.h;
    if(columns<=0 || columns>v.capacity || input.size()!=std::size_t(in/32)*columns || output.size()!=std::size_t(out)*columns)
        throw std::invalid_argument("GPU linear span");
    check(hipSetDevice(v.device));
    check(hipMemcpyAsync(v.qx,input.data(),input.size_bytes(),hipMemcpyHostToDevice,v.stream));
    v.launch(down?v.down:up?v.up:v.gate,down?v.down_type:TensorType::Q4_0,v.qx,v.y,in,out,columns);
    check(hipMemcpyAsync(output.data(),v.y,output.size_bytes(),hipMemcpyDeviceToHost,v.stream)); check(hipStreamSynchronize(v.stream));
}
ExpertTiming GpuExpert::run(std::span<const float> input,int columns,std::span<float> output,int repeats) {
    auto &v=*impl_;
    if(columns<=0 || columns>v.capacity || repeats<=0 || repeats>10000 || input.size()!=std::size_t(v.k)*columns || output.size()!=input.size())
        throw std::invalid_argument("GPU expert input/output");
    check(hipSetDevice(v.device)); check(hipMemsetAsync(v.error,0,sizeof(int),v.stream));
    check(hipMemcpyAsync(v.x,input.data(),input.size_bytes(),hipMemcpyHostToDevice,v.stream));
    v.forward(columns); v.check_error(); // warmup outside timing.
    const auto start=std::chrono::steady_clock::now(); check(hipEventRecord(v.begin,v.stream));
    for(int repeat=0;repeat<repeats;++repeat) v.forward(columns);
    check(hipEventRecord(v.end,v.stream)); const auto ms=elapsed(v.begin,v.end)/repeats;
    v.check_error(); check(hipMemcpyAsync(output.data(),v.y,output.size_bytes(),hipMemcpyDeviceToHost,v.stream)); check(hipStreamSynchronize(v.stream));
    const auto wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/repeats;
    return {v.upload_ms,ms,wall};
}
}
