#include "kv.cuh"
#include <hip/hip_fp16.h>
#include <stdexcept>

namespace qwen {
namespace {
void check(hipError_t value) { if(value!=hipSuccess) throw std::runtime_error(hipGetErrorString(value)); }
__device__ std::uint16_t half_bits(float x) { return __half_as_ushort(__float2half_rn(x)); }
// ADAPT mx cpy-utils.cuh quantization to one width32 subgroup per block.
// Reduce (absolute value, original index) lexicographically: first maximum
// wins sign ties EXACTLY as the donor's serial scan, not shuffle tie order.
__global__ void quant(const float *input,Q4_0 *output,int elements,int *error) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=elements) return; // launch elements is divisible by 32.
    const int lane=index%32;
    float x=input[index];
    int invalid=!isfinite(x); if(invalid) { atomicOr(error,1); x=0; }
    float a=fabsf(x),maximum=x; int first=lane;
    for(int mask=1;mask<32;mask*=2) {
        const float other=__shfl_xor(a,mask,32),v=__shfl_xor(maximum,mask,32);
        const int where=__shfl_xor(first,mask,32),bad=__shfl_xor(invalid,mask,32);
        if(other>a || (other==a && where<first)) { a=other; maximum=v; first=where; }
        invalid|=bad;
    }
    const float d=(a==0?0.0f:maximum)/-8.0f;
    const auto hd=half_bits(d);
    const bool representable=(hd&0x7c00)!=0x7c00 && (a==0 || (hd&0x7fff)!=0);
    if(!representable) atomicOr(error,2);
    const bool valid=!invalid && representable;
    const float inv=valid && d!=0?1.0f/d:0.0f;
    const int code=valid?min(15,static_cast<int>(x*inv+8.5f)):8;
    const int high=__shfl_xor(code,16,32);
    auto &b=output[index/32];
    if(lane<16) b.qs[lane]=static_cast<std::uint8_t>(code|(high<<4));
    if(lane==0) b.d=valid?hd:0x8000;
}
__global__ void dequant(const Q4_0 *input,float *output,int elements) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<elements) {
        const auto &b=input[i/32]; const int j=i%32;
        const int code=j<16?b.qs[j]&15:b.qs[j-16]>>4;
        output[i]=__half2float(__ushort_as_half(b.d))*(code-8);
    }
}
// PORT mx fwht.cu::fwht_cuda, pinned dcd685...; wave64 rows/register layout,
// ascending shuffle then register butterfly, normalization before summation.
template<int N> __global__ __launch_bounds__(256) void fwht(const float *src,float *dst,int rows) {
    const int r=blockIdx.x*4+threadIdx.x/64,lane=threadIdx.x%64;
    if(r>=rows) return;
    float reg[N/64];
    constexpr float scale=N==64?0.125f:N==256?0.0625f:0.08838834764831844055f;
    for(int j=0;j<N/64;++j) reg[j]=src[r*N+j*64+lane]*scale;
    for(int h=1;h<64;h*=2) for(int j=0;j<N/64;++j) {
        const float a=reg[j],b=__shfl_xor(a,h,64);
        reg[j]=(lane&h)==0?a+b:b-a;
    }
    for(int h=64;h<N;h*=2) for(int j=0;j<N/64;j+=2*(h/64)) for(int k=0;k<h/64;++k) {
        const float a=reg[j+k],b=reg[j+k+h/64];
        reg[j+k]=a+b; reg[j+k+h/64]=a-b;
    }
    for(int j=0;j<N/64;++j) dst[r*N+j*64+lane]=reg[j];
}
}
void launch_quantize_q4(const float *input,Q4_0 *output,int elements,int *error,hipStream_t stream) {
    if(elements<0 || elements%32 || (elements && (!input || !output || !error))) throw std::invalid_argument("GPU Q4 quantization shape");
    if(elements) { quant<<<elements/256+(elements%256!=0),256,0,stream>>>(input,output,elements,error); check(hipGetLastError()); }
}
void launch_dequantize_q4(const Q4_0 *input,float *output,int elements,hipStream_t stream) {
    if(elements<0 || elements%32 || (elements && (!input || !output))) throw std::invalid_argument("GPU Q4 dequantization shape");
    if(elements) { dequant<<<elements/256+(elements%256!=0),256,0,stream>>>(input,output,elements); check(hipGetLastError()); }
}
void launch_hadamard(const float *input,float *output,int elements,int group,hipStream_t stream) {
    if((group!=64 && group!=128 && group!=256) || elements<0 || elements%group || (elements && (!input || !output)))
        throw std::invalid_argument("GPU Hadamard shape");
    if(!elements) return;
    const int rows=elements/group,blocks=(rows+3)/4;
    if(group==64) fwht<64><<<blocks,256,0,stream>>>(input,output,rows);
    else if(group==128) fwht<128><<<blocks,256,0,stream>>>(input,output,rows);
    else fwht<256><<<blocks,256,0,stream>>>(input,output,rows);
    check(hipGetLastError());
}
}
