#include "gpu_expert.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <string>
#include <vector>
#include <sched.h>

namespace {
using Clock=std::chrono::steady_clock;
struct Matrix {
    qwen::TensorType type;
    int input,output;
    std::vector<std::byte> bytes;
    Matrix(const qwen::Model &model,const std::string &name,int input_dim,int output_dim)
        :type(model.tensor(name).type),input(input_dim),output(output_dim) {
        const auto &tensor=model.tensor(name);
        if(tensor.rank!=3 || tensor.dimensions[0]!=std::uint64_t(input) || tensor.dimensions[1]!=std::uint64_t(output)
                || tensor.dimensions[2]!=512) throw std::runtime_error("expert geometry mismatch: "+name);
        bytes.resize(tensor.strides[2]); model.read_expert(name,0,bytes);
    }
    qwen::QMatrix view() const { return {type,input,output,bytes}; }
};
struct Error { double max_abs=0,rms=0; };
Error parity(std::span<const float> reference,std::span<const float> result,double absolute,double relative,const char *stage) {
    if(reference.size()!=result.size()) throw std::runtime_error("parity size");
    Error error;
    for(std::size_t i=0;i<reference.size();++i) {
        const double difference=std::abs(double(reference[i])-result[i]);
        if(!std::isfinite(result[i]) || !std::isfinite(reference[i]) || difference>absolute+relative*std::abs(reference[i])) {
            std::cerr<<stage<<" parity index "<<i<<": ref="<<reference[i]<<" got="<<result[i]<<" diff="<<difference<<'\n';
            throw std::runtime_error(std::string(stage)+" parity");
        }
        error.max_abs=std::max(error.max_abs,difference); error.rms+=difference*difference;
    }
    error.rms=std::sqrt(error.rms/reference.size()); return error;
}
// Bounded numerical gates fixed before the first GPU run; not tuned to pass it.
// Common-quant linear: 2e-4 + 2e-5 |ref| (FP32 reduction order).
// Float pipeline: 2e-3 + 2e-4 |ref| (includes nonlinear/Q8 decision boundaries).
struct CpuExpert {
    qwen::QMatrix gate,up,down;
    int columns;
    std::vector<qwen::Q8_1> qx,qm;
    std::vector<float> g,u,mid,out;
    CpuExpert(qwen::QMatrix a,qwen::QMatrix b,qwen::QMatrix c,int n):gate(a),up(b),down(c),columns(n),
        qx(std::size_t(a.input/32)*n),qm(std::size_t(a.output/32)*n),g(std::size_t(a.output)*n),u(g.size()),mid(g.size()),out(std::size_t(a.input)*n) {}
    void run(std::span<const float> input,bool avx,bool baseline=false) {
        qwen::quantize_q8(input,qx);
        auto multiply=avx?(baseline?qwen::matmul_avx2_baseline:qwen::matmul_avx2):qwen::matmul_scalar;
        multiply(gate,qx,columns,g); multiply(up,qx,columns,u);
        for(std::size_t i=0;i<g.size();++i) mid[i]=(g[i]/(1.0f+std::exp(-g[i])))*u[i];
        qwen::quantize_q8(mid,qm); multiply(down,qm,columns,out);
    }
    double bench(std::span<const float> input,int repeats,bool baseline) {
        run(input,true,baseline);
        const auto start=Clock::now(); for(int repeat=0;repeat<repeats;++repeat) run(input,true,baseline);
        return std::chrono::duration<double,std::milli>(Clock::now()-start).count()/repeats;
    }
};
}

int main(int argc,char **argv) {
    try {
        if(argc!=2) throw std::runtime_error("usage: core-expert target.gguf");
        cpu_set_t allowed;
        if(sched_getaffinity(0,sizeof(allowed),&allowed)) throw std::system_error(errno,std::generic_category(),"expert affinity");
        int cpu_id=-1;
        for(int id=0;id<CPU_SETSIZE;++id) if(CPU_ISSET(id,&allowed)) { cpu_id=id; break; }
        if(cpu_id<0) throw std::runtime_error("no allowed CPU");
        cpu_set_t selected; CPU_ZERO(&selected); CPU_SET(cpu_id,&selected);
        if(sched_setaffinity(0,sizeof(selected),&selected)) throw std::system_error(errno,std::generic_category(),"pin expert CPU");
        qwen::Model model(argv[1]);
        if(model.metadata_value("general.architecture").get<std::string>()!="qwen4exp") throw std::runtime_error("expert architecture");
        constexpr int k=2560,h=640;
        Matrix gate(model,"blk.0.ffn_gate_exps.weight",k,h),up(model,"blk.0.ffn_up_exps.weight",k,h),down(model,"blk.0.ffn_down_exps.weight",h,k);
        if(gate.type!=qwen::TensorType::Q4_0 || up.type!=qwen::TensorType::Q4_0 || down.type!=qwen::TensorType::Q4_1)
            throw std::runtime_error("layer 0 must retain Q4_0 gate/up and Q4_1 down");
        std::cout<<std::setprecision(9);
        std::cout<<"{\"kind\":\"expert_source\",\"revision\":\""<<CORE_REVISION<<"\",\"dirty\":"<<CORE_DIRTY
                 <<",\"model\":\"qwen38-keep1-Q4_0.gguf\",\"layer\":0,\"expert\":0,\"weight_bytes\":"
                 <<gate.bytes.size()+up.bytes.size()+down.bytes.size()<<",\"avx2\":"<<qwen::cpu_has_avx2()<<",\"cpu_affinity\":"<<cpu_id<<"}\n";
        for(int columns:{1,2,3,128}) {
            std::vector<float> input(std::size_t(k)*columns);
            for(std::size_t i=0;i<input.size();++i) input[i]=0.7f*std::sin(float(i)*0.137f)+0.2f*std::cos(float(i)*0.071f);
            CpuExpert cpu(gate.view(),up.view(),down.view(),columns); cpu.run(input,false);
            const auto reference=cpu.out; const auto quantized=cpu.qx;
            const auto gate_reference=cpu.g,up_reference=cpu.u;
            const auto down_input=cpu.qm;
            cpu.run(input,true); parity(reference,cpu.out,2e-6,2e-6,"CPU AVX2");
            const int repeats=columns==128?5:50;
            for(bool baseline:{true,false,true}) {
                const double cpu_ms=cpu.bench(input,repeats,baseline);
                const auto cpu_error=parity(reference,cpu.out,2e-6,2e-6,"CPU benchmark");
                std::cout<<"{\"kind\":\"expert_cpu\",\"columns\":"<<columns<<",\"workers\":1,\"kernel\":\""
                         <<(baseline?"block_calls":"inlined_f16c")<<"\",\"ms\":"<<cpu_ms
                         <<",\"max_abs_error\":"<<cpu_error.max_abs<<",\"rms_error\":"<<cpu_error.rms<<",\"repeats\":"<<repeats<<"}\n";
            }
            for(int device:{0,1}) for(int variant:{0,1,0}) { // Paired A/B/A; never two competing residents.
                qwen::GpuExpert gpu(gate.view(),up.view(),down.view(),device,columns,variant!=0);
                gpu.check_quantization(input,quantized);
                if(columns==1) {
                    std::vector<float> edge(k,-0.0f); std::vector<qwen::Q8_1> packed(k/32);
                    qwen::quantize_q8(edge,packed); gpu.check_quantization(edge,packed);
                    for(float scale:{0.125f,std::ldexp(1.0f,-24)}) {
                        for(int b=0;b<k/32;++b) for(int i=0;i<32;++i)
                            edge[b*32+i]=i==0?127.0f*scale:(float(i)-15.5f)*scale;
                        qwen::quantize_q8(edge,packed); gpu.check_quantization(edge,packed);
                    }
                    for(float value:{std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),
                                     std::numeric_limits<float>::denorm_min(),1e8f}) {
                        std::fill(edge.begin(),edge.end(),value);
                        bool rejected=false;
                        try { gpu.check_quantization(edge,packed); } catch(const std::invalid_argument&) { rejected=true; }
                        if(!rejected) throw std::runtime_error("GPU quantization invalid input accepted");
                    }
                    gpu.check_quantization(input,quantized); // Rejected input cannot poison the next call.
                }
                std::vector<float> linear(std::size_t(h)*columns),output(reference.size());
                gpu.linear(false,false,quantized,columns,linear); const auto gate_error=parity(gate_reference,linear,2e-4,2e-5,"gate");
                gpu.linear(false,true,quantized,columns,linear); const auto up_error=parity(up_reference,linear,2e-4,2e-5,"up");
                gpu.linear(true,false,down_input,columns,output); const auto down_error=parity(reference,output,2e-4,2e-5,"Q4_1 down");
                // Also exercise full signed int8 range, intentionally independent of float quantization.
                auto corners=quantized;
                for(std::size_t b=0;b<corners.size();++b) {
                    corners[b].d=qwen::float_to_half(1.0f/512); corners[b].s=qwen::float_to_half(-0.1875f);
                    for(int i=0;i<32;++i) corners[b].qs[i]=static_cast<std::int8_t>(int((b*32+i)%256)-128);
                }
                std::vector<float> corner_reference(linear.size());
                qwen::matmul_scalar(gate.view(),corners,columns,corner_reference);
                gpu.linear(false,false,corners,columns,linear); parity(corner_reference,linear,2e-4,2e-5,"int8 extrema");
                const auto timing=gpu.run(input,columns,output,repeats);
                const auto final_error=parity(reference,output,2e-3,2e-4,"float expert");
                std::cout<<"{\"kind\":\"expert_gpu\",\"device\":"<<device<<",\"columns\":"<<columns
                         <<",\"layout\":\""<<(variant?"planar":"canonical")<<"\",\"repeats\":"<<repeats
                         <<",\"upload_pack_complete_ms\":"<<timing.upload_ms<<",\"resident_ms\":"<<timing.resident_ms
                         <<",\"completed_wall_ms\":"<<timing.completed_wall_ms<<",\"gate_error\":"<<gate_error.max_abs
                         <<",\"up_error\":"<<up_error.max_abs<<",\"down_error\":"<<down_error.max_abs
                         <<",\"max_abs_error\":"<<final_error.max_abs<<",\"rms_error\":"<<final_error.rms<<"}\n";
            }
        }
        std::cout<<"{\"kind\":\"expert_complete\",\"passed\":true}\n";
    } catch(const std::exception &error) { std::cerr<<"core-expert: "<<error.what()<<'\n'; return 1; }
}
