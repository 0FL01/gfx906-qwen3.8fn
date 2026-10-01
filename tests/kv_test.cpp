#include "kv.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
std::size_t checks=0,rejections=0;
void require(bool value,const char *message) { ++checks; if(!value) throw std::runtime_error(message); }
template<class F> void rejected(F f) {
    try { f(); } catch(const std::invalid_argument&) { ++rejections; return; }
    throw std::runtime_error("invalid KV input accepted");
}
void packing() {
    std::array<float,32> x{};
    std::array<qwen::Q4_0,1> y{};
    for(float zero:{0.0f,-0.0f}) {
        x.fill(zero); qwen::quantize_q4(x,y);
        require(y[0].d==0x8000,"baseline negative-zero scale");
        for(auto q:y[0].qs) require(q==0x88,"zero block nibble encoding");
    }
    // Signed scale and first-occurrence tie must not follow last max/sign.
    for(int first:{0,3,15,16,31}) for(float sign:{-1.0f,1.0f}) {
        x.fill(0); x[first]=8*sign;
        if(first!=31) x[31]=-8*sign;
        qwen::quantize_q4(x,y);
        require(qwen::half_to_float(y[0].d)==-sign,"first absolute maximum tie/sign");
        for(int i=0;i<32;++i) {
            const int expected=std::min(15,int(x[i]/-sign+8.5f));
            const int got=i<16?y[0].qs[i]&15:y[0].qs[i-16]>>4;
            require(got==expected,"canonical nibble order/ties");
        }
    }
    // Codes calculated using rounded half(d) would cross a discrete boundary.
    x.fill(0); x[0]=1.0003f; const float d=x[0]/-8;
    x[1]=d*(2.5f-0.0001f); qwen::quantize_q4(x,y);
    require((y[0].qs[1]&15)==10,"original FP32 scale code");
    require(int(x[1]/qwen::half_to_float(y[0].d)+8.5f)==11,"fixture distinguishes rounded scale");
    for(float maximum:{std::ldexp(8.0f,-24),std::ldexp(8.0f,-14),17.3f,500000.0f}) {
        for(int i=0;i<32;++i) x[i]=maximum*float(i-16)/16;
        qwen::quantize_q4(x,y); std::array<float,32> decoded{}; qwen::dequantize_q4(y,decoded);
        for(int i=0;i<32;++i) {
            const int code=i<16?y[0].qs[i]&15:y[0].qs[i-16]>>4;
            require(std::bit_cast<std::uint32_t>(decoded[i])==std::bit_cast<std::uint32_t>(qwen::half_to_float(y[0].d)*(code-8)),"dequant signed/subnormal scale");
        }
    }
    // Independent dense dot on dequantized Q4 blocks checks exact nibble signs.
    std::vector<float> many(32*257),decoded(many.size()); std::vector<qwen::Q4_0> packed(many.size()/32);
    for(std::size_t i=0;i<many.size();++i) many[i]=.75f*std::sin(float(i)*.131f);
    qwen::quantize_q4(many,packed); qwen::dequantize_q4(packed,decoded);
    for(std::size_t i=0;i<many.size();++i) require(std::fabs(decoded[i]-many[i])<=.101f,"quantization bounded error");
}
void rotations() {
    for(int n:{64,128,256}) {
        std::vector<float> x(n*5),y(x.size()),z(x.size());
        for(std::size_t i=0;i<x.size();++i) x[i]=std::sin(float(i)*.317f)*.7f;
        qwen::hadamard(x,y,n); qwen::hadamard(y,z,n);
        double energy=0,rot_energy=0;
        for(std::size_t i=0;i<x.size();++i) {
            require(std::fabs(x[i]-z[i])<4e-7,"Hadamard inverse parity");
            energy+=double(x[i])*x[i]; rot_energy+=double(y[i])*y[i];
        }
        require(std::fabs(energy-rot_energy)<=energy*3e-7,"Hadamard energy");
        auto inplace=x; qwen::hadamard(inplace,inplace,n);
        require(inplace==y,"Hadamard exact in-place parity");
        // Dense normalized +/-1 Sylvester matrix is an independent oracle.
        const double scale=1/std::sqrt(double(n));
        for(int r=0;r<n;++r) {
            double sum=0; for(int c=0;c<n;++c) sum+=((std::popcount(unsigned(r&c))&1)?-1:1)*double(x[c])*scale;
            require(std::fabs(sum-y[r])<7e-7,"Hadamard dense matrix oracle");
        }
    }
}
void invalid_inputs() {
    std::vector<float> x(64,1),out(64); std::array<qwen::Q4_0,2> packed{};
    std::memset(packed.data(),0x5a,sizeof(packed)); const auto original=packed;
    for(float bad:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),
                   std::numeric_limits<float>::denorm_min(),std::numeric_limits<float>::max()}) {
        x[63]=bad;
        if(std::isfinite(bad)) std::fill(x.begin()+32,x.end(),bad);
        rejected([&] { qwen::quantize_q4(x,packed); });
        require(std::memcmp(packed.data(),original.data(),sizeof(packed))==0,"invalid quantization atomicity");
        std::fill(x.begin(),x.end(),1);
    }
    rejected([&] { qwen::quantize_q4(std::span(x).first(63),packed); });
    rejected([&] { qwen::quantize_q4(x,std::span(packed).first(1)); });
    packed[1].d=0x7c00; std::fill(out.begin(),out.end(),12);
    rejected([&] { qwen::dequantize_q4(packed,out); });
    require(std::all_of(out.begin(),out.end(),[](float v) { return v==12; }),"invalid dequantization atomicity");
    rejected([&] { qwen::dequantize_q4(std::span(packed).first(1),out); });
    rejected([&] { qwen::hadamard(x,out,32); });
    rejected([&] { qwen::hadamard(std::span(x).first(63),out,64); });
    std::vector<float> overlapping(128,1);
    rejected([&] { qwen::hadamard(std::span(overlapping).first(64),std::span(overlapping).subspan(1,64),64); });
    rejected([&] { qwen::quantize_q4(std::span(overlapping).first(64),
        std::span(reinterpret_cast<qwen::Q4_0*>(overlapping.data()),2)); });
    rejected([&] { qwen::dequantize_q4(
        std::span(reinterpret_cast<const qwen::Q4_0*>(overlapping.data()),2),std::span(overlapping).first(64)); });
    x[4]=std::numeric_limits<float>::quiet_NaN(); rejected([&] { qwen::hadamard(x,out,64); });
    qwen::quantize_q4({},{}); qwen::dequantize_q4({},{}); qwen::hadamard({},{},64);
}
}
int main() {
    try { packing(); rotations(); invalid_inputs(); std::cout<<"{\"test\":\"kv\",\"checks\":"<<checks<<",\"rejections\":"<<rejections<<",\"passed\":true}\n"; }
    catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
