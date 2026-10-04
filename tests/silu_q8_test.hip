#include "hip/linear.cuh"
#include "hip/session_ops.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace {
void check(hipError_t s) { if(s!=hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void require(bool b,const char* m) { if(!b) throw std::runtime_error(m); }
struct Buffer {
    void* p=nullptr; std::size_t n;
    explicit Buffer(std::size_t bytes):n(bytes){check(hipMalloc(&p,n+128));}
    ~Buffer(){if(p && hipFree(p)!=hipSuccess)std::abort();}
    template<class T>T* get()const{return reinterpret_cast<T*>(static_cast<char*>(p)+64);}
    void poison(hipStream_t s){check(hipMemsetAsync(p,0xa5,n+128,s));}
    std::vector<unsigned char> read(hipStream_t s)const{
        std::vector<unsigned char> a(n+128);check(hipMemcpyAsync(a.data(),p,n+128,hipMemcpyDeviceToHost,s));check(hipStreamSynchronize(s));return a;
    }
    void guards(hipStream_t s,std::size_t used)const{
        const auto a=read(s);for(std::size_t i=0;i<64;++i)require(a[i]==0xa5,"prefix guard");
        for(std::size_t i=64+used;i<a.size();++i)require(a[i]==0xa5,"suffix guard");
    }
};
struct Gpu {
    hipStream_t s{};hipEvent_t begin{},end{};
    std::size_t cases=0,values=0,bytes=0,rejects=0,graphs=0,recoveries=0,aliases=0;
    explicit Gpu(int d){check(hipSetDevice(d));check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));check(hipEventCreate(&begin));check(hipEventCreate(&end));}
    ~Gpu(){if(hipStreamSynchronize(s)!=hipSuccess||hipEventDestroy(begin)!=hipSuccess||hipEventDestroy(end)!=hipSuccess||hipStreamDestroy(s)!=hipSuccess)std::abort();}
    template<class F>double time(F f){
        for(int i=0;i<5;++i)f();check(hipStreamSynchronize(s));
        check(hipEventRecord(begin,s));for(int i=0;i<100;++i)f();check(hipEventRecord(end,s));check(hipEventSynchronize(end));
        float ms=0;check(hipEventElapsedTime(&ms,begin,end));return ms/100;
    }
    void run(int device,int w,int n,int variant){
        const std::size_t count=std::size_t(w)*n, fbytes=count*4,qbytes=count/32*sizeof(qwen::Q8_1);
        Buffer g(fbytes),u(fbytes),a(fbytes),b(fbytes),qa(qbytes),qb(qbytes),ea(4),eb(4);
        std::vector<float> gate(count),up(count);
        for(std::size_t i=0;i<count;++i){
            gate[i]=float(int(i*37%4093)-2046)/256;
            up[i]=float(int(i*53%2039)-1019)/1024;
            if(variant==1)gate[i]=std::ldexp(float(int(i%31)-15),-22);
            if(variant==2){gate[i]=i%2?-1000:1000;up[i]=0.00001f*float(int(i%13)-6);}
            if(variant==3){gate[i]=i%2?-0.0f:0.0f;up[i]=i%3?-1.0f:1.0f;}
            if(variant==6){gate[i]=std::numeric_limits<float>::max();up[i]=2;}
            if(variant==7){gate[i]=100000;up[i]=1;}
        }
        if(variant==4){gate[0]=std::numeric_limits<float>::quiet_NaN();up.back()=std::numeric_limits<float>::infinity();}
        if(variant==5){gate[31]=-std::numeric_limits<float>::infinity();up[count/2]=std::numeric_limits<float>::quiet_NaN();}
        for(auto* x:{&g,&u,&a,&b,&qa,&qb,&ea,&eb})x->poison(s);
        check(hipMemcpyAsync(g.get<float>(),gate.data(),fbytes,hipMemcpyHostToDevice,s));
        check(hipMemcpyAsync(u.get<float>(),up.data(),fbytes,hipMemcpyHostToDevice,s));
        const int sticky=variant==3?4:0;
        check(hipMemcpyAsync(ea.get<int>(),&sticky,4,hipMemcpyHostToDevice,s));
        check(hipMemcpyAsync(eb.get<int>(),&sticky,4,hipMemcpyHostToDevice,s));
        auto old=[&]{check(qwen::launch_silu_up(g.get<float>(),u.get<float>(),int(count),a.get<float>(),ea.get<int>(),s));
                    check(qwen::launch_quantize_q8_1(a.get<float>(),qa.get<qwen::Q8_1>(),w,n,ea.get<int>(),s));};
        auto fused=[&]{check(qwen::launch_silu_up_q8_1(g.get<float>(),u.get<float>(),w,n,b.get<float>(),qb.get<qwen::Q8_1>(),eb.get<int>(),s));};
        old();fused();check(hipStreamSynchronize(s));
        auto parity=[&]{
            require(a.read(s)==b.read(s),"FP32 middle mismatch/canary");
            require(qa.read(s)==qb.read(s),"Q8 bytes mismatch/canary");
            require(ea.read(s)==eb.read(s),"sticky flags mismatch/canary");
            a.guards(s,fbytes);b.guards(s,fbytes);qa.guards(s,qbytes);qb.guards(s,qbytes);
            ea.guards(s,4);eb.guards(s,4);
        };
        parity();
        const auto flag_bytes=ea.read(s);int observed=0;std::memcpy(&observed,flag_bytes.data()+64,4);
        require(variant<3?observed==0:variant==3?observed==4:observed!=0,"expected numeric/sticky flags");
        if(variant==0){
            check(hipStreamBeginCapture(s,hipStreamCaptureModeThreadLocal));fused();
            hipGraph_t graph{};check(hipStreamEndCapture(s,&graph));std::size_t nodes=0;
            check(hipGraphGetNodes(graph,nullptr,&nodes));require(nodes==1,"fused launch count");
            check(hipGraphDestroy(graph));++graphs;
            if(w==640){
                const auto x=time(old),y=time(fused),z=time(old);parity();
                std::cout<<"{\"kind\":\"silu_q8_timing\",\"device\":"<<device<<",\"width\":"<<w<<",\"columns\":"<<n
                         <<",\"old_a_ms\":"<<x<<",\"fused_b_ms\":"<<y<<",\"old_a2_ms\":"<<z<<"}\n";
            }
        }
        const auto gg=g.read(s),uu=u.read(s);
        require(std::memcmp(gg.data()+64,gate.data(),fbytes)==0 && std::memcmp(uu.data()+64,up.data(),fbytes)==0,"input modified");
        g.guards(s,fbytes);u.guards(s,fbytes);++cases;values+=count;bytes+=qbytes;
        if(variant>=4){
            // Actual numeric failure followed by valid work in the SAME owners,
            // without clearing the sticky error or poisoning output anew.
            std::fill(gate.begin(),gate.end(),0.0f);std::fill(up.begin(),up.end(),1.0f);
            check(hipMemcpyAsync(g.get<float>(),gate.data(),fbytes,hipMemcpyHostToDevice,s));
            check(hipMemcpyAsync(u.get<float>(),up.data(),fbytes,hipMemcpyHostToDevice,s));
            old();fused();parity();require(ea.read(s)==flag_bytes,"recovery cleared sticky bits");++recoveries;
        }
        if(variant==0){
            check(qwen::launch_silu_up(g.get<float>(),g.get<float>(),int(count),a.get<float>(),ea.get<int>(),s));
            check(qwen::launch_quantize_q8_1(a.get<float>(),qa.get<qwen::Q8_1>(),w,n,ea.get<int>(),s));
            check(qwen::launch_silu_up_q8_1(g.get<float>(),g.get<float>(),w,n,b.get<float>(),qb.get<qwen::Q8_1>(),eb.get<int>(),s));
            parity();++aliases;
        }
    }
    void invalid(){
        Buffer g(640*128*4),u(640*128*4),m(640*128*4),q(640*128/32*sizeof(qwen::Q8_1)),e(4);
        for(auto* p:{&g,&u,&m,&q,&e})p->poison(s);
        auto reject=[&](const float* a,const float* b,int w,int n,float* c,qwen::Q8_1* d,int* f){
            require(qwen::launch_silu_up_q8_1(a,b,w,n,c,d,f,s)==hipErrorInvalidValue,"invalid accepted");
            for(auto* p:{&g,&u,&m,&q,&e})p->guards(s,0);++rejects;
        };
        auto G=g.get<float>(),U=u.get<float>(),M=m.get<float>();auto Q=q.get<qwen::Q8_1>();auto E=e.get<int>();
        for(int w:{0,-1,31,16385})reject(G,U,w,1,M,Q,E);
        for(int n:{0,-1,129,std::numeric_limits<int>::max()})reject(G,U,640,n,M,Q,E);
        reject(nullptr,U,640,1,M,Q,E);reject(G,nullptr,640,1,M,Q,E);reject(G,U,640,1,nullptr,Q,E);
        reject(G,U,640,1,M,nullptr,E);reject(G,U,640,1,M,Q,nullptr);
        reject(G,U,640,1,G,Q,E);reject(G,U,640,1,M,reinterpret_cast<qwen::Q8_1*>(U),E);
        reject(G,U,640,1,M,reinterpret_cast<qwen::Q8_1*>(M),E);
        reject(G,U,640,1,M,Q,reinterpret_cast<int*>(Q));
        reject(G,U,640,1,M,Q,reinterpret_cast<int*>(G));
        reject(G,U,640,1,M,Q,reinterpret_cast<int*>(U));
        reject(G,U,640,1,M,Q,reinterpret_cast<int*>(M));
        reject(G,U,640,1,U,Q,E);
        reject(reinterpret_cast<float*>(reinterpret_cast<char*>(G)+1),U,640,1,M,Q,E);
        reject(G,reinterpret_cast<float*>(reinterpret_cast<char*>(U)+1),640,1,M,Q,E);
        reject(G,U,640,1,reinterpret_cast<float*>(reinterpret_cast<char*>(M)+1),Q,E);
        reject(G,U,640,1,M,reinterpret_cast<qwen::Q8_1*>(reinterpret_cast<char*>(Q)+1),E);
        reject(G,U,640,1,M,Q,reinterpret_cast<int*>(reinterpret_cast<char*>(E)+1));
        reject(reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max()-3),U,640,1,M,Q,E);
        reject(G,reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max()-3),640,1,M,Q,E);
        reject(G,U,640,1,reinterpret_cast<float*>(std::numeric_limits<std::uintptr_t>::max()-3),Q,E);
        reject(G,U,640,1,M,reinterpret_cast<qwen::Q8_1*>(std::numeric_limits<std::uintptr_t>::max()-3),E);
        reject(G,U,640,1,M,Q,reinterpret_cast<int*>(std::numeric_limits<std::uintptr_t>::max()-3));
    }
};
}
int main(){
    try{
        std::cout<<std::setprecision(17)<<"{\"kind\":\"silu_q8_source\",\"revision\":\""<<CORE_REVISION<<"\",\"dirty\":"<<(CORE_DIRTY?"true":"false")<<",\"reference\":\"canonical two-launch GPU SiLU plus Q8\",\"performance_qualified\":false}\n";
        int devices=0;check(hipGetDeviceCount(&devices));require(devices==2,"two GPUs");
        for(int d=0;d<2;++d){
            Gpu g(d);g.invalid();
            for(int w:{32,640,16384})for(int n:{1,2,3,4,7,8,9,16,17,32,127,128})for(int v=0;v<8;++v)g.run(d,w,n,v);
            std::cout<<"{\"kind\":\"silu_q8_device\",\"device\":"<<d<<",\"cases\":"<<g.cases<<",\"middle_values\":"<<g.values<<",\"q8_bytes\":"<<g.bytes<<",\"host_rejects\":"<<g.rejects<<",\"graph_checks\":"<<g.graphs<<",\"same_owner_recovery_cases\":"<<g.recoveries<<",\"readonly_alias_cases\":"<<g.aliases<<",\"passed\":true}\n";
        }
        std::cout<<"{\"kind\":\"silu_q8_complete\",\"devices\":2,\"cleanup_complete\":true,\"passed\":true}\n";
    }catch(const std::exception& e){std::cerr<<"silu-q8: "<<e.what()<<'\n';return 1;}
}

