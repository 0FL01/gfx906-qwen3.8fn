#include "kv.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace qwen {
namespace {
bool overlap(const void *a,std::size_t as,const void *b,std::size_t bs) {
    const auto *x=static_cast<const std::byte*>(a),*y=static_cast<const std::byte*>(b);
    const std::less<const std::byte*> before;
    return as && bs && before(x,y+bs) && before(y,x+as);
}
float block_scale(const float *x) {
    float absolute=0,maximum=0;
    for(int j=0;j<32;++j) {
        if(!std::isfinite(x[j])) throw std::invalid_argument("Q4 input is nonfinite");
        if(absolute<std::fabs(x[j])) { absolute=std::fabs(x[j]); maximum=x[j]; }
    }
    const float d=maximum/-8.0f;
    const auto h=float_to_half(d);
    if((h&0x7c00)==0x7c00 || (absolute!=0 && (h&0x7fff)==0))
        throw std::invalid_argument("Q4 scale is not representable");
    return d;
}
}
void quantize_q4(std::span<const float> input,std::span<Q4_0> output) {
    if(input.size()%32 || output.size()!=input.size()/32 || overlap(input.data(),input.size_bytes(),output.data(),output.size_bytes()))
        throw std::invalid_argument("Q4 quantization shape/overlap");
    for(std::size_t b=0;b<output.size();++b) (void)block_scale(input.data()+b*32);
    for(std::size_t b=0;b<output.size();++b) {
        const auto *x=input.data()+b*32;
        const float d=block_scale(x),inv=d?1.0f/d:0.0f;
        auto &y=output[b]; y.d=float_to_half(d);
        for(int j=0;j<16;++j) {
            const int low=std::min(15,static_cast<int>(x[j]*inv+8.5f));
            const int high=std::min(15,static_cast<int>(x[j+16]*inv+8.5f));
            y.qs[j]=static_cast<std::uint8_t>(low|(high<<4));
        }
    }
}
void dequantize_q4(std::span<const Q4_0> input,std::span<float> output) {
    if(output.size()%32 || input.size()!=output.size()/32 || overlap(input.data(),input.size_bytes(),output.data(),output.size_bytes()))
        throw std::invalid_argument("Q4 dequantization shape/overlap");
    for(const auto &b:input) if((b.d&0x7c00)==0x7c00) throw std::invalid_argument("Q4 scale is nonfinite");
    for(std::size_t b=0;b<input.size();++b) {
        const float d=half_to_float(input[b].d);
        for(int j=0;j<16;++j) {
            output[b*32+j]=d*(int(input[b].qs[j]&15)-8);
            output[b*32+j+16]=d*(int(input[b].qs[j]>>4)-8);
        }
    }
}
void hadamard(std::span<const float> input,std::span<float> output,int group) {
    if((group!=64 && group!=128 && group!=256) || input.size()!=output.size() || input.size()%std::size_t(group)
            || (input.data()!=output.data() && overlap(input.data(),input.size_bytes(),output.data(),output.size_bytes())))
        throw std::invalid_argument("Hadamard shape/overlap");
    for(float x:input) if(!std::isfinite(x)) throw std::invalid_argument("Hadamard input is nonfinite");
    const float scale=1.0f/std::sqrt(float(group));
    std::array<float,256> values{};
    for(std::size_t start=0;start<input.size();start+=group) {
        for(int i=0;i<group;++i) values[i]=input[start+i]*scale;
        for(int h=1;h<group;h*=2) for(int base=0;base<group;base+=2*h) for(int i=0;i<h;++i) {
            const float a=values[base+i],b=values[base+i+h];
            values[base+i]=a+b; values[base+i+h]=a-b;
        }
        std::copy_n(values.data(),group,output.data()+start);
    }
}
}
