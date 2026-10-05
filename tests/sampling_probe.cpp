#include "sampling.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
namespace {
constexpr std::size_t vocabulary=248320, rows=8;
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
struct Snapshot {
    bool capture; std::fstream file;std::uint64_t bytes=0;
    Snapshot(const std::string& mode,const std::filesystem::path& path):capture(mode=="--capture"){
        require(capture||mode=="--compare","expected --capture/--compare");
        if(capture)require(!std::filesystem::exists(path),"snapshot already exists");
        file.open(path,std::ios::binary|(capture?std::ios::out:std::ios::in));require(bool(file),"snapshot open");
        const std::array<std::uint64_t,4> header{0x513853414d504c31ULL,1,vocabulary,rows};
        verify(std::span(header));
    }
    template<class T,std::size_t Extent>void verify(std::span<T,Extent> a){
        static_assert(std::is_trivially_copyable_v<T>);
        const std::uint64_t n=a.size_bytes();
        if(capture){
            file.write(reinterpret_cast<const char*>(&n),8);
            file.write(reinterpret_cast<const char*>(a.data()),static_cast<std::streamsize>(n));
        }else{
            std::uint64_t expected=0;file.read(reinterpret_cast<char*>(&expected),8);require(expected==n,"snapshot extent");
            std::vector<unsigned char> data(n);file.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(n));
            require(bool(file)&&std::memcmp(data.data(),a.data(),n)==0,"cross-build snapshot differs");
        }
        require(bool(file),"snapshot IO");bytes+=n;
    }
    void finish(){
        const std::array<std::uint64_t,1> footer{0x454e4453414d504cULL};verify(std::span(footer));
        if(capture){file.flush();require(bool(file),"snapshot flush");}
        else require(file.peek()==std::char_traits<char>::eof(),"snapshot trailing data");
    }
};
double checksum=0;
template<class F>double measure(F f){
    for(int i=0;i<5;++i)checksum+=f();
    const auto begin=std::chrono::steady_clock::now();
    for(int i=0;i<100;++i)checksum+=f();
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()/100;
}
}
int main(int argc,char** argv){
    try{
        require(argc==4,"usage: core-sampling-probe LOGITS --capture|--compare SNAPSHOT");
        const auto size=std::filesystem::file_size(argv[1]);
        require(size>=rows*vocabulary*4 && size%(vocabulary*4)==0 && size<=128*vocabulary*4,"bounded full-vocabulary logit rows");
        std::ifstream input(argv[1],std::ios::binary);std::vector<float> logits(rows*vocabulary);
        input.read(reinterpret_cast<char*>(logits.data()),std::streamsize(logits.size()*4));require(bool(input),"logit read");
        require(std::all_of(logits.begin(),logits.end(),[](float x){return std::isfinite(x);}),"finite source logits");
        Snapshot snapshot(argv[2],argv[3]);
        std::cout<<std::setprecision(17)<<"{\"kind\":\"sampling_probe_source\",\"revision\":\""<<CORE_REVISION
                 <<"\",\"dirty\":"<<(CORE_DIRTY?"true":"false")<<",\"vocabulary\":"<<vocabulary<<",\"rows\":"<<rows
                 <<",\"input\":"<<std::quoted(argv[1])<<",\"scope\":\"saved actual logits; CPU component only\",\"performance_qualified\":false}\n";
        std::vector<float> p(vocabulary),q(vocabulary),residual(vocabulary);
        std::size_t cases=0;
        for(int variant=0;variant<4;++variant){
            qwen::SamplingConfig config(12345);if(variant==1){config.top_k=0;config.top_p=1;}
            if(variant==2){config.temperature=0;}
            if(variant==3){config.top_k=1;config.top_p=0.5;}
            qwen::Sampler sampler(vocabulary,config);
            for(std::size_t row=0;row<rows;++row){
                const auto a=std::span(logits).subspan(row*vocabulary,vocabulary);
                const auto b=std::span(logits).subspan(((row+1)%rows)*vocabulary,vocabulary);
                sampler.distribution(a,p);sampler.distribution(b,q);
                const auto token=std::size_t(std::max_element(q.begin(),q.end())-q.begin());
                const double acceptance=qwen::acceptance_probability(p,q,token);
                std::fill(residual.begin(),residual.end(),-12345);
                const auto status=qwen::residual_distribution(p,q,residual);
                const std::array<std::uint64_t,4> meta{std::uint64_t(variant),row,token,std::uint64_t(status)};
                snapshot.verify(std::span(meta));snapshot.verify(std::span(p));snapshot.verify(std::span(q));snapshot.verify(std::span(residual));
                snapshot.verify(std::span(&acceptance,1));
                sampler.reset_seed(12345+row);
                std::array<std::uint64_t,128> draws{};
                for(std::size_t i=0;i<draws.size();i+=2){draws[i]=sampler.draw(q);draws[i+1]=sampler.accept(p,q,token)?1:0;}
                snapshot.verify(std::span(draws));const auto rng=sampler.random_draws();snapshot.verify(std::span(&rng,1));
                require(rng==128,"one RNG draw per call");++cases;
            }
        }
        snapshot.finish();
        qwen::Sampler sampler(vocabulary,qwen::SamplingConfig(12345));
        const auto a=std::span(logits).first(vocabulary),b=std::span(logits).subspan(vocabulary,vocabulary);
        sampler.distribution(a,p);sampler.distribution(b,q);
        const auto token=std::size_t(std::max_element(q.begin(),q.end())-q.begin());
        const auto distribution=measure([&]{sampler.distribution(a,p);return double(p[token]);});
        const auto draw=measure([&]{return double(sampler.draw(q));});
        const auto accept=measure([&]{return qwen::acceptance_probability(p,q,token);});
        const auto residual_ms=measure([&]{return double(qwen::residual_distribution(p,q,residual)==qwen::ResidualStatus::ready);});
        require(std::isfinite(checksum),"timing checksum");
        std::cout<<"{\"kind\":\"sampling_probe_timing\",\"iterations\":100,\"distribution_ms\":"<<distribution
                 <<",\"draw_ms\":"<<draw<<",\"acceptance_ms\":"<<accept<<",\"residual_ms\":"<<residual_ms
                 <<",\"checksum\":"<<checksum<<",\"scope\":\"single-thread completed caller wall; not MTP runtime\"}\n";
        std::cout<<"{\"kind\":\"sampling_probe_complete\",\"cases\":"<<cases<<",\"snapshot_bytes\":"<<snapshot.bytes
                 <<",\"cross_build_compare\":"<<(snapshot.capture?"false":"true")<<",\"passed\":true}\n";
        require(bool(std::cout),"stdout");
    }catch(const std::exception& e){std::cerr<<"sampling-probe: "<<e.what()<<'\n';return 1;}
}

