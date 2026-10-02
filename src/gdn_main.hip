#include "dense.hpp"
#include "gdn.hpp"
#include "hip/gdn.cuh"
#include <algorithm>
#include <array>
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
    if(s!=hipSuccess) { std::cerr<<"GDN cleanup: "<<hipGetErrorString(s)<<'\n'; std::abort(); }
}
struct Errors { double output=0,state=0,output_ratio=0,state_ratio=0; };
void near(std::span<const float> got,std::span<const float> expected,bool state,Errors &errors) {
    if(got.size()!=expected.size()) throw std::runtime_error("GDN comparison size");
    for(std::size_t i=0;i<got.size();++i) {
        const double error=std::abs(double(got[i])-expected[i]);
        (state?errors.state:errors.output)=std::max(state?errors.state:errors.output,error);
        const double bound=(state?2e-5:2e-4)+2e-4*std::abs(double(expected[i]));
        (state?errors.state_ratio:errors.output_ratio)=std::max(state?errors.state_ratio:errors.output_ratio,error/bound);
        // Frozen before first GPU run; differences in FP32 reduction order allowed.
        if(!std::isfinite(got[i]) || !std::isfinite(expected[i]) ||
           error>bound)
            throw std::runtime_error(state?"GDN recurrent parity":"GDN output parity");
    }
}
void exact(std::span<const float> got,std::span<const float> expected,const char *message) {
    if(got.size()!=expected.size() || std::memcmp(got.data(),expected.data(),got.size_bytes()))
        throw std::runtime_error(message);
    for(float x:got) if(!std::isfinite(x)) throw std::runtime_error("GDN nonfinite exact comparison");
}
struct Data {
    qwen::GdnConfig c;
    std::vector<float> conv,dt,a,norm,qkv,z,alpha,beta;
    static constexpr int seed_tokens=8;
    explicit Data(const qwen::Model &model) {
        if(model.metadata_value("general.architecture").get<std::string>()!="qwen4exp")
            throw std::runtime_error("GDN fixture requires qwen4exp target");
        c.rms_epsilon=model.metadata_value("qwen4exp.attention.layer_norm_rms_epsilon").get<float>();
        auto load=[&](const char *name,std::span<const std::uint64_t> dims,qwen::TensorType type) {
            const auto &t=model.tensor(name);
            if(t.type!=type || t.rank!=dims.size() || !std::equal(dims.begin(),dims.end(),t.dimensions.begin()))
                throw std::runtime_error(std::string("GDN actual tensor geometry/type: ")+name);
            std::vector<std::byte> bytes(t.byte_size); model.read_tensor(name,bytes); return bytes;
        };
        auto parameter=[&](const char *name,std::initializer_list<std::uint64_t> dims) {
            auto bytes=load(name,std::span(dims.begin(),dims.size()),qwen::TensorType::F32);
            std::vector<float> result(bytes.size()/4);
            for(std::size_t i=0;i<result.size();++i) result[i]=qwen::tensor_element(qwen::TensorType::F32,bytes,i);
            return result;
        };
        conv=parameter("blk.0.ssm_conv1d.weight",{4,10240});
        dt=parameter("blk.0.ssm_dt.bias",{48}); a=parameter("blk.0.ssm_a",{48});
        norm=parameter("blk.0.ssm_norm.weight",{128});
        std::vector<float> x(2560*seed_tokens);
        for(std::size_t i=0;i<x.size();++i) x[i]=.3f*std::sin(float(i)*.137f)+.1f*std::cos(float(i)*.071f);
        auto project=[&](const char *name,int rows,qwen::TensorType type) {
            const std::array<std::uint64_t,2> dims{2560,std::uint64_t(rows)};
            auto bytes=load(name,dims,type); std::vector<float> result(rows*seed_tokens);
            qwen::matmul_f32({type,2560,rows,bytes},x,seed_tokens,result); return result;
        };
        qkv=project("blk.0.attn_qkv.weight",10240,qwen::TensorType::Q4_0);
        z=project("blk.0.attn_gate.weight",6144,qwen::TensorType::Q4_0);
        alpha=project("blk.0.ssm_alpha.weight",48,qwen::TensorType::F32);
        beta=project("blk.0.ssm_beta.weight",48,qwen::TensorType::F32);
    }
    qwen::GdnParameters parameters() const { return {conv,dt,a,norm}; }
    qwen::GdnInput input(int start,int tokens) const {
        return {std::span(qkv).subspan(start*10240,tokens*10240),std::span(z).subspan(start*6144,tokens*6144),
                std::span(alpha).subspan(start*48,tokens*48),std::span(beta).subspan(start*48,tokens*48)};
    }
    Data expanded(int tokens) const {
        Data result=*this;
        auto repeat=[&](const std::vector<float> &original,int width) {
            std::vector<float> output(tokens*width);
            for(int t=0;t<tokens;++t) std::copy_n(original.data()+(t%seed_tokens)*width,width,output.data()+t*width);
            return output;
        };
        result.qkv=repeat(qkv,10240); result.z=repeat(z,6144);
        result.alpha=repeat(alpha,48); result.beta=repeat(beta,48); return result;
    }
};
struct Fixture {
    int device;
    hipStream_t stream{}; hipEvent_t begin{},end{};
    qwen::GdnDeviceParameters p{}; qwen::GdnDeviceInput in{}; qwen::GdnDeviceWorkspace w{};
    float *state=nullptr,*history=nullptr,*output=nullptr,*rp=nullptr,*hp=nullptr;
    std::vector<void*> owned;
    explicit Fixture(int d):device(d) {}
    ~Fixture() {
        cleanup(hipSetDevice(device)); if(stream) cleanup(hipStreamSynchronize(stream));
        for(auto b:owned) cleanup(hipFree(b));
        if(begin) cleanup(hipEventDestroy(begin));
        if(end) cleanup(hipEventDestroy(end));
        if(stream) cleanup(hipStreamDestroy(stream));
    }
    template<class T> T *allocate(std::size_t n) {
        void *b=nullptr; check(hipMalloc(&b,n*sizeof(T)));
        try { owned.push_back(b); } catch(...) { cleanup(hipFree(b)); throw; }
        return static_cast<T*>(b);
    }
    void put(float *dest,std::span<const float> source) {
        check(hipMemcpyAsync(dest,source.data(),source.size_bytes(),hipMemcpyHostToDevice,stream));
    }
    std::vector<float> get(const float *source,std::size_t n) {
        std::vector<float> result(n);
        check(hipMemcpyAsync(result.data(),source,n*4,hipMemcpyDeviceToHost,stream));
        check(hipStreamSynchronize(stream)); return result;
    }
    void initialize(const Data &data,int capacity=128) {
        check(hipSetDevice(device)); hipDeviceProp_t props{}; check(hipGetDeviceProperties(&props,device));
        if(!std::string(props.gcnArchName).starts_with("gfx906") || props.warpSize!=64)
            throw std::runtime_error("GDN requires gfx906 wave64");
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking)); check(hipEventCreate(&begin)); check(hipEventCreate(&end));
        p={allocate<float>(40960),allocate<float>(48),allocate<float>(48),allocate<float>(128),data.c.rms_epsilon};
        put(p.conv_weights,data.conv); put(p.dt_bias,data.dt); put(p.ssm_a,data.a); put(p.norm_weight,data.norm);
        in={allocate<float>(capacity*10240),allocate<float>(capacity*6144),allocate<float>(capacity*48),allocate<float>(capacity*48)};
        w={allocate<float>(capacity*10240),allocate<float>(786432),allocate<float>(30720),allocate<float>(capacity*6144),
            nullptr,nullptr,allocate<int>(1)};
        state=allocate<float>(786432); history=allocate<float>(30720); output=allocate<float>(capacity*6144);
        const int slots=std::min(capacity,3)+1;
        rp=allocate<float>(slots*786432); hp=allocate<float>(slots*30720);
        reset(); check(hipStreamSynchronize(stream));
    }
    void reset() {
        check(hipMemsetAsync(state,0,786432*4,stream)); check(hipMemsetAsync(history,0,30720*4,stream));
        check(hipMemsetAsync(w.error,0,4,stream));
    }
    void upload(qwen::GdnInput input) {
        put(const_cast<float*>(in.qkv),input.qkv); put(const_cast<float*>(in.z),input.z);
        put(const_cast<float*>(in.alpha),input.alpha); put(const_cast<float*>(in.beta),input.beta);
        check(hipMemsetAsync(w.error,0,4,stream));
    }
    void launch(int n,bool prefixes=false) {
        auto work=w; if(prefixes) { work.recurrent_prefixes=rp; work.history_prefixes=hp; }
        qwen::launch_gdn(p,in,n,state,history,output,work,stream);
    }
    void launch_steps(int n,bool prefixes=false) {
        for(int t=0;t<n;++t) {
            const qwen::GdnDeviceInput token{in.qkv+t*10240,in.z+t*6144,in.alpha+t*48,in.beta+t*48};
            auto work=w;
            if(prefixes) { work.recurrent_prefixes=rp+t*786432; work.history_prefixes=hp+t*30720; }
            qwen::launch_gdn(p,token,1,state,history,output+t*6144,work,stream);
        }
    }
    int flag() {
        int result=0; check(hipMemcpyAsync(&result,w.error,4,hipMemcpyDeviceToHost,stream));
        check(hipStreamSynchronize(stream)); return result;
    }
    void valid() { if(flag()) throw std::runtime_error("GDN unexpected numeric flag"); }
    void compare(const qwen::GdnCheckpoint &expected,Errors &errors) {
        near(get(state,786432),expected.recurrent(),true,errors);
        auto h=get(history,30720);
        if(std::memcmp(h.data(),expected.conv_history().data(),h.size()*4)) throw std::runtime_error("GDN raw history byte parity");
    }
    void restore(int consumed) {
        check(hipMemcpyAsync(state,rp+consumed*786432,786432*4,hipMemcpyDeviceToDevice,stream));
        check(hipMemcpyAsync(history,hp+consumed*30720,30720*4,hipMemcpyDeviceToDevice,stream));
    }
};
void exact_short_cases(Fixture &f,const Data &data) {
    // Part of the existing correctness fixture, not another JSON record/counter.
    // Exact-size inputs/scratch/output/prefixes exercise the last legal token and
    // slot; no hidden N128 input tail can mask a short-window indexing error.
    for(int n:{2,3}) {
        Fixture short_f(f.device); short_f.initialize(data,n);
        for(bool occupied:{false,true}) {
            short_f.reset();
            if(occupied) { short_f.upload(data.input(0,2)); short_f.launch_steps(2); short_f.valid(); }
            const auto before=short_f.get(short_f.state,786432),history=short_f.get(short_f.history,30720);
            short_f.upload(data.input(2,n)); short_f.launch_steps(n,true); short_f.valid();
            const auto output=short_f.get(short_f.output,n*6144),state=short_f.get(short_f.state,786432);
            const auto after_history=short_f.get(short_f.history,30720);
            const auto rp=short_f.get(short_f.rp,(n+1)*786432),hp=short_f.get(short_f.hp,(n+1)*30720);
            exact(std::span(rp).first(786432),before,"GDN N1 prefix zero state");
            exact(std::span(hp).first(30720),history,"GDN N1 prefix zero history");
            exact(std::span(rp).subspan(n*786432,786432),state,"GDN N1 final prefix state");
            exact(std::span(hp).subspan(n*30720,30720),after_history,"GDN N1 final prefix history");
            for(bool prefixes:{true,false}) {
                short_f.put(short_f.state,before); short_f.put(short_f.history,history);
                // Poison previous results so missing final stores/prefixes
                // cannot pass by reusing the preceding sequential-N1 run.
                check(hipMemsetAsync(short_f.w.convolved,0xff,n*10240*4,short_f.stream));
                check(hipMemsetAsync(short_f.w.recurrent,0xff,786432*4,short_f.stream));
                check(hipMemsetAsync(short_f.w.history,0xff,30720*4,short_f.stream));
                check(hipMemsetAsync(short_f.w.output,0xff,n*6144*4,short_f.stream));
                check(hipMemsetAsync(short_f.output,0xff,n*6144*4,short_f.stream));
                check(hipMemsetAsync(short_f.rp,0xff,(n+1)*786432*4,short_f.stream));
                check(hipMemsetAsync(short_f.hp,0xff,(n+1)*30720*4,short_f.stream));
                short_f.upload(data.input(2,n)); short_f.launch(n,prefixes); short_f.valid();
                exact(short_f.get(short_f.output,n*6144),output,"GDN short-window exact N1 output");
                exact(short_f.get(short_f.state,786432),state,"GDN short-window exact N1 state");
                exact(short_f.get(short_f.history,30720),after_history,"GDN short-window exact N1 history");
                if(prefixes) {
                    exact(short_f.get(short_f.rp,(n+1)*786432),rp,"GDN short-window exact N1 state prefixes");
                    exact(short_f.get(short_f.hp,(n+1)*30720),hp,"GDN short-window exact N1 history prefixes");
                    // Every accepted/rejected prefix must also replay exactly.
                    for(int consumed=0;consumed<=n;++consumed) {
                        short_f.put(short_f.state,std::span(rp).subspan(consumed*786432,786432));
                        short_f.put(short_f.history,std::span(hp).subspan(consumed*30720,30720));
                        short_f.upload(data.input(6,1)); short_f.launch(1); short_f.valid();
                        const auto replay_output=short_f.get(short_f.output,6144),replay_state=short_f.get(short_f.state,786432);
                        const auto replay_history=short_f.get(short_f.history,30720);
                        short_f.restore(consumed); short_f.launch(1); short_f.valid();
                        exact(short_f.get(short_f.output,6144),replay_output,"GDN short-window exact restored output");
                        exact(short_f.get(short_f.state,786432),replay_state,"GDN short-window exact restored state");
                        exact(short_f.get(short_f.history,30720),replay_history,"GDN short-window exact restored history");
                    }
                }
            }
        }
    }
}
void measure(Fixture &f,const Data &base,int n) {
    const Data data=base.expanded(n);
    qwen::GdnCpu cpu(data.c,data.parameters()); std::vector<float> expected(n*6144);
    const auto start=std::chrono::steady_clock::now(); cpu.run(data.input(0,n),n,expected);
    const double cpu_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    f.reset(); f.upload(data.input(0,n)); f.launch(n); f.valid();
    Errors errors; near(f.get(f.output,n*6144),expected,false,errors); f.compare(cpu.state(),errors);
    // Reset + launch is timed as a completed resident transaction; separate from
    // projection/host transfer. Repeat identical zero-state workloads, not drift.
    constexpr int repeats=20;
    check(hipEventRecord(f.begin,f.stream));
    for(int r=0;r<repeats;++r) { f.reset(); f.launch(n); }
    check(hipEventRecord(f.end,f.stream)); check(hipEventSynchronize(f.end));
    float ms=0; check(hipEventElapsedTime(&ms,f.begin,f.end)); f.valid();
    near(f.get(f.output,n*6144),expected,false,errors); f.compare(cpu.state(),errors);
    std::cout<<"{\"kind\":\"gdn_gpu\",\"device\":"<<f.device<<",\"tokens\":"<<n<<",\"repeats\":"<<repeats
             <<",\"reset_and_resident_ms\":"<<ms/repeats<<",\"cpu_oracle_ms\":"<<cpu_ms
             <<",\"max_output_error\":"<<errors.output<<",\"max_state_error\":"<<errors.state
             <<",\"output_gate_ratio\":"<<errors.output_ratio<<",\"state_gate_ratio\":"<<errors.state_ratio<<",\"history_byte_parity\":true}\n";
    if(n==1) return;
    // A/B/A dispatch qualification: identical raw projections and zero state.
    // A is validated CPW2 decode repeated N times, not an invented scalar GPU.
    for(bool chunk:{false,true,false}) {
        check(hipEventRecord(f.begin,f.stream));
        for(int r=0;r<repeats;++r) {
            f.reset(); if(chunk) f.launch(n); else f.launch_steps(n);
        }
        check(hipEventRecord(f.end,f.stream)); check(hipEventSynchronize(f.end));
        check(hipEventElapsedTime(&ms,f.begin,f.end)); f.valid();
        Errors paired; near(f.get(f.output,n*6144),expected,false,paired); f.compare(cpu.state(),paired);
        std::cout<<"{\"kind\":\"gdn_dispatch\",\"device\":"<<f.device<<",\"tokens\":"<<n
                 <<",\"path\":\""<<(chunk?"resident_chunk":"decode_steps")<<"\",\"repeats\":"<<repeats
                 <<",\"reset_and_resident_ms\":"<<ms/repeats<<",\"max_output_error\":"<<paired.output
                 <<",\"max_state_error\":"<<paired.state<<",\"output_gate_ratio\":"<<paired.output_ratio
                 <<",\"state_gate_ratio\":"<<paired.state_ratio<<",\"history_byte_parity\":true}\n";
    }
}
void restore_cases(Fixture &f,const Data &data) {
    exact_short_cases(f,data);
    qwen::GdnCpu cpu(data.c,data.parameters()); std::vector<float> tmp(3*6144),expected(3*6144);
    std::array<qwen::GdnCheckpoint,4> prefixes{qwen::GdnCheckpoint(data.c),qwen::GdnCheckpoint(data.c),
        qwen::GdnCheckpoint(data.c),qwen::GdnCheckpoint(data.c)};
    Errors errors;
    f.reset(); cpu.run(data.input(0,2),2,std::span(tmp).first(2*6144)); f.upload(data.input(0,2)); f.launch(2); f.valid();
    int cases=0;
    for(int window=0;window<2;++window) {
        // Verify one pending input + two drafts. Chronological slot1+accepted.
        cpu.run(data.input(2,3),3,expected,prefixes); f.upload(data.input(2,3)); f.launch(3,true); f.valid();
        near(f.get(f.output,3*6144),expected,false,errors);
        for(int consumed=0;consumed<=3;++consumed) {
            near(f.get(f.rp+consumed*786432,786432),prefixes[consumed].recurrent(),true,errors);
            auto history=f.get(f.hp+consumed*30720,30720);
            if(std::memcmp(history.data(),prefixes[consumed].conv_history().data(),history.size()*4))
                throw std::runtime_error("GDN chronological conv prefix");
            cpu.restore(prefixes[consumed]); f.restore(consumed);
            cpu.step(data.input(6,1),std::span(tmp).first(6144)); f.upload(data.input(6,1)); f.launch(1); f.valid();
            near(f.get(f.output,6144),std::span(tmp).first(6144),false,errors); f.compare(cpu.state(),errors); ++cases;
        }
        // Keep rejecting both drafts; history must not carry either into next verify.
        cpu.restore(prefixes[1]); f.restore(1);
        cpu.step(data.input(6,1),std::span(tmp).first(6144)); f.upload(data.input(6,1)); f.launch(1); f.valid();
    }
    // A single 3-token resident chunk and three CPW2 steps share external state.
    cpu.reset(); f.reset(); cpu.run(data.input(0,3),3,expected);
    for(int t=0;t<3;++t) { f.upload(data.input(t,1)); f.launch(1); f.valid();
        near(f.get(f.output,6144),std::span(expected).subspan(t*6144,6144),false,errors); }
    f.compare(cpu.state(),errors);
    std::cout<<"{\"kind\":\"gdn_restore\",\"device\":"<<f.device<<",\"cases\":"<<cases
             <<",\"windows\":2,\"max_output_error\":"<<errors.output<<",\"max_state_error\":"<<errors.state
             <<",\"output_gate_ratio\":"<<errors.output_ratio<<",\"state_gate_ratio\":"<<errors.state_ratio
             <<",\"history_byte_parity\":true,\"split_chunk_parity\":true}\n";
}
void invalid_cases(Fixture &f,const Data &data) {
    // Seed all three caller output rows so any partial chunk publication is
    // observable, including a late invalid control/output gate. Counters still
    // count the original three QKV + three control rejection families.
    const std::vector<float> sentinel(3*6144,-123.0f); f.put(f.output,sentinel);
    f.reset(); f.upload(data.input(0,1)); f.launch(1); f.valid();
    const auto before=f.get(f.state,786432),history=f.get(f.history,30720),out=f.get(f.output,3*6144);
    auto unchanged=[&] {
        exact(f.get(f.state,786432),before,"GDN failure was not chunk-atomic: state");
        exact(f.get(f.history,30720),history,"GDN failure was not chunk-atomic: history");
        exact(f.get(f.output,3*6144),out,"GDN failure was not chunk-atomic: output");
    };
    int cases=0;
    for(float bad:{std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity(),std::numeric_limits<float>::max()}) {
        for(int n:{2,3}) for(bool prefixes:{false,true}) {
            auto input=data.expanded(n); std::fill(input.qkv.begin()+(n-1)*10240,input.qkv.end(),bad);
            f.upload(input.input(0,n)); f.launch(n,prefixes);
            if(!(f.flag()&(std::isfinite(bad)?2:1))) throw std::runtime_error("GDN invalid-input flag");
            unchanged();
        }
        ++cases;
    }
    for(int field=0;field<3;++field) {
        for(int n:{1,2,3}) for(bool prefixes:{false,true}) {
            auto input=data.expanded(n); auto &control=field==0?input.z:field==1?input.alpha:input.beta;
            for(std::size_t index:{std::size_t(0),control.size()-1}) {
                const float saved=control[index]; control[index]=std::numeric_limits<float>::quiet_NaN();
                f.upload(input.input(0,n)); f.launch(n,prefixes);
                if(!(f.flag()&1)) throw std::runtime_error("GDN invalid control flag");
                unchanged(); control[index]=saved;
            }
        }
        ++cases;
    }
    auto rejects=[&](auto call) { try { call(); } catch(const std::invalid_argument&) { ++cases; return; }
        throw std::runtime_error("GDN invalid host argument accepted"); };
    rejects([&] { f.launch(0); }); rejects([&] { f.launch(129); });
    rejects([&] { auto w=f.w; w.recurrent_prefixes=f.rp; qwen::launch_gdn(f.p,f.in,1,f.state,f.history,f.output,w,f.stream); });
    rejects([&] { qwen::launch_gdn(f.p,f.in,1,f.state,f.history,f.state,f.w,f.stream); });
    rejects([&] { auto p=f.p; p.rms_epsilon=0; qwen::launch_gdn(p,f.in,1,f.state,f.history,f.output,f.w,f.stream); });
    // Explicit restore + valid reuse following rejected chunks.
    f.reset(); f.upload(data.input(0,1)); f.launch(1); f.valid();
    unchanged();
    std::cout<<"{\"kind\":\"gdn_invalid\",\"device\":"<<f.device<<",\"rejections\":"<<cases
             <<",\"chunk_atomic\":true,\"reuse\":true}\n";
}
}
int main(int argc,char **argv) {
    try {
        if(argc!=2) throw std::invalid_argument("usage: core-gdn target.gguf");
        qwen::Model model(argv[1]); Data data(model);
        std::cout<<std::setprecision(9)<<"{\"kind\":\"gdn_source\",\"revision\":\""<<CORE_REVISION<<"\",\"dirty\":"<<CORE_DIRTY
                 <<",\"model\":\"qwen38-keep1-Q4_0.gguf\",\"layer\":0,\"projection\":\"CPU raw FP32 oracle\",\"rms_epsilon\":"<<data.c.rms_epsilon
                 <<",\"scope\":\"loaded layer0 weights; eight synthetic projected tokens repeated for N128; not full inference\"}\n";
        int count=0; check(hipGetDeviceCount(&count)); if(count!=2) throw std::runtime_error("expected two gfx906 GPUs");
        for(int d=0;d<count;++d) {
            Fixture f(d); f.initialize(data);
            for(int n:{1,2,3,128}) measure(f,data,n);
            restore_cases(f,data); invalid_cases(f,data);
        }
        std::cout<<"{\"kind\":\"gdn_complete\",\"passed\":true}\n";
    } catch(const std::exception &e) { std::cerr<<"core-gdn: "<<e.what()<<'\n'; return 1; }
}
