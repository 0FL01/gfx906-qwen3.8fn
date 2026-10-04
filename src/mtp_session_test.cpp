#include "mtp_session.hpp"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef CORE_REVISION
#error "core-mtp-session-test requires CORE_REVISION"
#endif
#ifndef CORE_DIRTY
#error "core-mtp-session-test requires CORE_DIRTY"
#endif

// Actual-model fixture, not a donor/HF oracle. Legacy inputs are externally
// captured little-endian F32 [3][10240]: ZERO,T0,T1 for the supplied tokens.
// Dataset/capture-origin are required caller metadata, NEVER invented here.
// --target GGUF --sidecar GGUF --hidden F32 --tokens x0,x1,x2
// --verify-tokens c,a,b --dataset-id ID --capture-origin ORIGIN --capacity 8
// Optional --logits-out F32 --tap-out F32 export N3 for an independent oracle.
// Fixed strict self-tolerances abs2e-3+rel2e-4, NEVER command-line loosened.
// Full vocabulary, raw gates, routes, D and head input are compared, not prose.
// Chronological capture mode (no external hidden file):
// --target GGUF --sidecar GGUF --teacher-tokens I32_32 --capture NEWDIR
// --dataset-id ID --capture-origin ORIGIN --capacity 64 --expert-slots 112
// --teacher-width 1|2|3. All four raw arrays and capture.json are written only
// after BOTH runtimes are destroyed. The JSON file is the completion marker.
// capture.json is the independent tool's EXACT six-key v1 descriptor; the
// adjacent capture_metadata.json keeps detailed protocol/provenance/gates.
// Optional --capture-first-row 1 adds fifteen bounded graph-site F32 files and
// first_row.json in that same NEWDIR; capture.json's six-key schema stays exact.
// Optional --capture-probe-position UINT selects absolute0..31 (default0), only
// with --capture-first-row 1. A window containing it may start BEFORE that row.
// Independent oracle handoff: MODEL SIDECAR CAPTURE_DIR NEW_OUTDIR
// --teacher-width 1|2|3; capacity64/cache112 and protocol1 are explicit in JSON.

namespace {
constexpr int H=2560,W=10240,V=248320,Q=6144,N=3;
constexpr int teacher_rows=32,capture_protocol=1,capture_capacity=64,capture_slots=112;
static_assert(sizeof(float)==4 && std::endian::native==std::endian::little);
static_assert(sizeof(std::int32_t)==4 && CORE_DIRTY>=0 && CORE_DIRTY<=1);
static_assert(std::string_view(CORE_REVISION).size()==40);
constexpr double absolute_tolerance=2e-3,relative_tolerance=2e-4;
void require(bool ok,const std::string& what) {if (!ok) throw std::runtime_error(what);}
// Same checked JSON quote as the accepted target restore fixture.
void json_string(std::ostream& out,std::string_view value) {
    constexpr char hex[]="0123456789abcdef";out<<'"';
    for(unsigned char c:value) {
        if(c=='"' || c=='\\')out<<'\\'<<static_cast<char>(c);
        else if(c<0x20)out<<"\\u00"<<hex[c>>4]<<hex[c&15];
        else out<<static_cast<char>(c);
    }
    out<<'"';
}
void write_binary(const std::filesystem::path& path,std::span<const std::byte> values) {
    std::ofstream file(path,std::ios::binary|std::ios::trunc);
    require(bool(file.write(reinterpret_cast<const char*>(values.data()),values.size_bytes())),"write fixture array");
    file.flush();require(bool(file),"flush fixture array");
    file.close();require(!file.fail(),"close fixture array");
}
void hip_ok(hipError_t e) {if (e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e));}
void cleanup(hipError_t e) noexcept {if(e!=hipSuccess)std::abort();}
struct DeviceGuard {
    int previous=0;
    DeviceGuard(){hip_ok(hipGetDevice(&previous));hip_ok(hipSetDevice(1));}
    ~DeviceGuard(){cleanup(hipSetDevice(previous));}
};
struct DeviceFloats {
    float* pointer=nullptr;
    explicit DeviceFloats(std::span<const float> input) {
        DeviceGuard d;hip_ok(hipMalloc(reinterpret_cast<void**>(&pointer),input.size_bytes()));
        try{hip_ok(hipMemcpy(pointer,input.data(),input.size_bytes(),hipMemcpyHostToDevice));}
        catch(...){cleanup(hipFree(pointer));pointer=nullptr;throw;}
    }
    ~DeviceFloats(){DeviceGuard d;if(pointer)cleanup(hipFree(pointer));}
    DeviceFloats(const DeviceFloats&)=delete;
    DeviceFloats& operator=(const DeviceFloats&)=delete;
};
std::vector<float> copy_device(const float* pointer,std::size_t count) {
    DeviceGuard d;std::vector<float> result(count);
    if(count)hip_ok(hipMemcpy(result.data(),pointer,count*sizeof(float),hipMemcpyDeviceToHost));
    return result;
}
std::vector<float> read_hidden(const std::string& path) {
    require(std::filesystem::file_size(path)==N*W*sizeof(float),"capture must be EXACT [3][10240] F32");
    std::ifstream file(path,std::ios::binary);std::vector<float> values(N*W);
    require(bool(file.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(float))),"read capture");
    for(float x:values)require(std::isfinite(x),"nonfinite captured hidden");
    for(int j=0;j<W;++j)require(values[j]==0.0f,"captured history row0 MUST be ZERO (production convention)");
    return values;
}
void write_f32(const std::string& path,std::span<const float> values) {
    if(path.empty())return;
    write_binary(path,std::as_bytes(values));
}
int integer(std::string_view text,int low,int high) {
    int value=0;const auto [end,error]=std::from_chars(text.data(),text.data()+text.size(),value);
    require(error==std::errc{} && end==text.data()+text.size() && value>=low && value<=high,"bounded integer argument");
    return value;
}
std::array<std::int32_t,3> token_triplet(std::string_view text) {
    std::array<std::int32_t,3> out{};
    for(int i=0;i<3;++i) {
        const auto comma=text.find(',');
        require((i==2)==(comma==std::string_view::npos),"exact three token IDs required");
        out[i]=integer(text.substr(0,comma),0,V-1);
        if(i<2)text.remove_prefix(comma+1);
    }
    return out;
}
struct Args {
    std::string target,sidecar,hidden,dataset,origin,logits_out,tap_out;
    std::string capture_dir,teacher_tokens;
    std::array<std::int32_t,3> tokens{},verify{};
    int capacity=8,teacher_width=1;
    bool capture_first_row=false;
    int capture_probe_position=0;
};
void new_capture_directory(const std::filesystem::path& path) {
    require(!path.empty() && !path.filename().empty() && path.filename()!="." && path.filename()!="..","new capture directory name");
    require(std::filesystem::symlink_status(path).type()==std::filesystem::file_type::not_found,"capture directory must not exist (including symlink)");
    const auto parent=path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    require(std::filesystem::is_directory(parent),"capture parent directory must exist");
}
Args args(int argc,char** argv) {
    require(argc>=13 && argc<=33 && argc%2==1,"fixture expects bounded named key/value arguments");
    Args out;std::array<bool,16> seen{};
    for(int i=1;i<argc;i+=2) {
        const std::string_view name=argv[i],value=argv[i+1];
        require(!value.empty() && value.size()<=4096,"bounded nonempty argument");
        for(char c:value)require(static_cast<unsigned char>(c)>=32 && c!=127,"control byte in argument");
        int key=-1;
        if(name=="--target"){key=0;out.target=value;}
        else if(name=="--sidecar"){key=1;out.sidecar=value;}
        else if(name=="--hidden"){key=2;out.hidden=value;}
        else if(name=="--tokens"){key=3;out.tokens=token_triplet(value);}
        else if(name=="--verify-tokens"){key=4;out.verify=token_triplet(value);}
        else if(name=="--dataset-id"){key=5;require(value.size()<=512,"dataset ID bound");out.dataset=value;}
        else if(name=="--capture-origin"){key=6;require(value.size()<=512,"origin bound");out.origin=value;}
        else if(name=="--capacity"){key=7;out.capacity=integer(value,8,131072);require(out.capacity%4==0,"target capacity multiple4");}
        else if(name=="--logits-out"){key=8;out.logits_out=value;}
        else if(name=="--tap-out"){key=9;out.tap_out=value;}
        else if(name=="--capture"){key=10;out.capture_dir=value;}
        else if(name=="--teacher-tokens"){key=11;out.teacher_tokens=value;}
        else if(name=="--teacher-width"){key=12;out.teacher_width=integer(value,1,3);}
        else if(name=="--expert-slots"){key=13;require(integer(value,112,112)==capture_slots,"fixed target cache112");}
        else if(name=="--capture-first-row"){key=14;out.capture_first_row=integer(value,1,1)==1;}
        else if(name=="--capture-probe-position") {
            key=15;require(value.front()>='0' && value.front()<='9',"capture probe position is a decimal UINT");
            out.capture_probe_position=integer(value,0,teacher_rows-1);
        }
        else throw std::invalid_argument("unknown fixture argument");
        require(!seen[key],"duplicate fixture argument");seen[key]=true;
    }
    for(int i:{0,1,5,6})require(seen[i],"missing mandatory dataset/provenance/model argument");
    require(!seen[15] || seen[14],"capture probe position requires --capture-first-row 1");
    if(seen[10]) {
        require(seen[11] && seen[7] && seen[13] && out.capacity==capture_capacity,"capture requires teacher tokens, explicit capacity64/cache112");
        require(!seen[2] && !seen[3] && !seen[4],"capture obtains hidden/tokens/verify rows from its live chronological target snapshot");
        new_capture_directory(out.capture_dir);
    } else {
        for(int i:{2,3,4})require(seen[i],"missing legacy hidden/tokens/verify argument");
        require(!seen[11] && !seen[12] && !seen[14] && !seen[15],"teacher tokens/width/first-row diagnostics require --capture");
    }
    for(const auto& path:{out.logits_out,out.tap_out}) if(!path.empty()) {
        const auto output=std::filesystem::weakly_canonical(path);
        for(const auto& input:{out.target,out.sidecar,out.hidden,out.teacher_tokens}) if(!input.empty())
            require(output!=std::filesystem::weakly_canonical(input) &&
                (!std::filesystem::exists(output) || !std::filesystem::equivalent(output,input)),
                "fixture export may not overwrite model/capture");
    }
    require(out.logits_out.empty() || out.tap_out.empty() ||
        (std::filesystem::weakly_canonical(out.logits_out)!=std::filesystem::weakly_canonical(out.tap_out) &&
         (!std::filesystem::exists(out.logits_out) || !std::filesystem::exists(out.tap_out) ||
          !std::filesystem::equivalent(out.logits_out,out.tap_out))),"exports must be distinct");
    if(!out.capture_dir.empty())for(const auto& output:{out.logits_out,out.tap_out})if(!output.empty()) {
        const auto directory=std::filesystem::weakly_canonical(out.capture_dir);
        auto path=std::filesystem::weakly_canonical(output);
        for(;!path.empty();path=path.parent_path()) {
            require(path!=directory,"legacy export may not be inside capture directory");
            if(path==path.parent_path())break;
        }
    }
    return out;
}
double max_error=0;
void compare(std::span<const float> a,std::span<const float> b,const std::string& what) {
    require(a.size()==b.size(),what+" extent");
    for(std::size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]),what+" nonfinite");
        const double error=std::abs(double(a[i])-b[i]);max_error=std::max(max_error,error);
        require(error<=absolute_tolerance+relative_tolerance*std::abs(double(a[i])),what+" full-vector mismatch at "+std::to_string(i));
    }
}
template<class T> void append(std::vector<T>& destination,std::span<const T> source) {
    destination.insert(destination.end(),source.begin(),source.end());
}
struct Capture {
    std::vector<float> logits,tap,gates,head,weights;
    std::vector<std::int32_t> ids;
};
// Bounded CPU probe views: retain identity/extents and copy object bytes, not
// approximate float values. Check ONLY across operations that preserve the last
// publication; never read these views after a successful full forward or reset.
struct ProbeSnapshot {
    qwen::MtpSessionProbe views;
    std::vector<std::byte> gates,head,ids,weights;
    template<class T> static std::vector<std::byte> save(std::span<const T> view,
            std::size_t expected,const std::string& what) {
        require(view.size()==expected && (view.empty() || view.data()),what+" bounded extent");
        std::vector<std::byte> bytes(view.size_bytes());
        if(!bytes.empty())std::memcpy(bytes.data(),view.data(),bytes.size());
        return bytes;
    }
    ProbeSnapshot(qwen::MtpSessionProbe probe,int rows):views(probe) {
        require(rows>=0 && rows<=N,"bounded probe rows");
        gates=save(probe.raw_attention_gates,std::size_t(rows)*Q,"probe raw gates");
        head=save(probe.head_input,std::size_t(rows)*H,"probe head input");
        ids=save(probe.expert_ids,std::size_t(rows)*10,"probe expert IDs");
        weights=save(probe.route_weights,std::size_t(rows)*10,"probe route weights");
    }
    template<class T> static void check_view(std::span<const T> current,
            std::span<const T> retained,const std::vector<std::byte>& bytes,const std::string& what) {
        require(current.data()==retained.data() && current.size()==retained.size(),what+" pointer/extent");
        require(retained.size_bytes()==bytes.size() &&
            (bytes.empty() || !std::memcmp(retained.data(),bytes.data(),bytes.size())),what+" retained physical bits");
    }
    void check(qwen::MtpSessionProbe current,const std::string& what) const {
        check_view(current.raw_attention_gates,views.raw_attention_gates,gates,what+" raw gates");
        check_view(current.head_input,views.head_input,head,what+" head input");
        check_view(current.expert_ids,views.expert_ids,ids,what+" expert IDs");
        check_view(current.route_weights,views.route_weights,weights,what+" route weights");
    }
};
struct FirstRowSite {const char* name;std::size_t columns;};
constexpr std::array<FirstRowSite,13> first_row_sites{{
    {"mtp_tok_embd-48",H},{"mtp_hnorm-48",W},{"mtp_enorm-48",W},
    {"mtp_concat-48",2*W},{"mtp_fused-48",W},{"hc_mixed-attention-48",H},
    {"attn_pregate-48",Q},{"attn_output-48",H},{"hc_combine-attention-48",W},
    {"hc_mixed-ffn-48",H},{"ffn_out-48",H},{"l_last-48",W},{"mtp_head_input",H}}};
std::array<std::span<const float>,first_row_sites.size()> first_row_views(qwen::MtpFirstRowProbe p) {
    return {p.token_embedding,p.hidden_norm,p.embedding_norm_repeated,p.fusion_concat,p.fusion_output,
        p.attention_mixed,p.attention_pregate,p.attention_output,p.attention_combined,p.ffn_mixed,
        p.ffn_output,p.D,p.head_input};
}
struct FirstRowSnapshot {
    bool published;
    std::uint64_t position;
    std::array<std::span<const float>,first_row_sites.size()> views;
    std::array<std::vector<std::byte>,first_row_sites.size()> bytes;
    explicit FirstRowSnapshot(qwen::MtpFirstRowProbe p):published(p.published),position(p.first_position),views(first_row_views(p)) {
        for(std::size_t i=0;i<views.size();++i)
            bytes[i]=ProbeSnapshot::save(views[i],published ? first_row_sites[i].columns : 0,first_row_sites[i].name);
    }
    void check(qwen::MtpFirstRowProbe p,const std::string& what) const {
        require(p.published==published && p.first_position==position,what+" validity/absolute position");
        const auto current=first_row_views(p);
        for(std::size_t i=0;i<views.size();++i)
            ProbeSnapshot::check_view(current[i],views[i],bytes[i],what+" "+first_row_sites[i].name);
    }
};
struct FirstRowArtifact {
    bool captured=false;
    std::uint64_t first_position=0,source_first_position=0;
    int source_window_rows=1;
    std::array<std::vector<float>,first_row_sites.size()> values;
    void capture(qwen::MtpFirstRowProbe p,std::uint64_t source_first=0,int source_rows=1) {
        require(p.published && source_rows>=1 && source_rows<=N && source_first<teacher_rows &&
            p.first_position>=source_first && p.first_position-source_first<std::uint64_t(source_rows) &&
            std::uint64_t(source_rows)<=std::uint64_t(teacher_rows)-source_first,"selected diagnostic row is inside actual source window");
        const auto views=first_row_views(p);
        for(std::size_t i=0;i<views.size();++i) {
            require(views[i].size()==first_row_sites[i].columns && views[i].data(),"bounded first-row site extent");
            for(float x:views[i])require(std::isfinite(x),"finite first-row site");
            values[i].assign(views[i].begin(),views[i].end());
        }
        first_position=p.first_position;source_first_position=source_first;source_window_rows=source_rows;
        captured=true; // owning full vectors copied before any borrowed expiry
    }
};
void restore_preserves(qwen::MtpSession& s,std::uint64_t prefix) {
    const ProbeSnapshot probes(s.probe(),s.own_tap().rows);
    const FirstRowSnapshot first(s.first_row_probe());
    s.restore_prefix(prefix);probes.check(s.probe(),"logical restore probe publication");
    first.check(s.first_row_probe(),"logical restore first-row publication");
}
void teacher_skip_preserves(qwen::MtpSession& s,std::span<const qwen::MtpHiddenRow> hidden,
        std::span<const std::int32_t> tokens) {
    const ProbeSnapshot probes(s.probe(),s.own_tap().rows);
    const FirstRowSnapshot first(s.first_row_probe());
    require(s.teacher_append(hidden,tokens,qwen::MtpTeacherHead::skip).empty(),"explicit KV-only has no logits");
    probes.check(s.probe(),"KV-only probe publication");
    first.check(s.first_row_probe(),"KV-only first-row publication");
}
void capture(qwen::MtpSession& s,std::span<const float> logits,Capture& out) {
    append(out.logits,logits);const auto tap=s.own_tap();
    const auto data=copy_device(tap.pointer,tap.rows*W);append(out.tap,std::span<const float>(data));
    const auto p=s.probe();append(out.gates,p.raw_attention_gates);append(out.head,p.head_input);
    append(out.ids,p.expert_ids);append(out.weights,p.route_weights);
}
void compare(const Capture& a,const Capture& b,const std::string& what) {
    compare(a.logits,b.logits,what+" full248320 logits");compare(a.tap,b.tap,what+" pre-own-head D");
    compare(a.gates,b.gates,what+" raw Q gates");compare(a.head,b.head,what+" own HC head input");
    compare(a.weights,b.weights,what+" selected raw route weights");require(a.ids==b.ids,what+" stable top10 IDs");
}
std::array<qwen::MtpHiddenRow,3> rows(const DeviceFloats& data) {
    return {{{1,data.pointer,W},{1,data.pointer+W,W},{1,data.pointer+2*W,W}}};
}
Capture teacher_run(qwen::MtpSession& s,const std::array<qwen::MtpHiddenRow,3>& h,
                    const std::array<std::int32_t,3>& t,int chunk,bool poison=false) {
    s.reset();if(poison)s.diagnostic_poison_future();Capture out;
    for(int first=0;first<N;first+=chunk) {
        const int n=std::min(chunk,N-first);
        capture(s,s.teacher_append(std::span(h).subspan(first,n),std::span(t).subspan(first,n),
                                  qwen::MtpTeacherHead::publish),out);
    }
    return out;
}
template<class F> void rejects(qwen::MtpSession& s,F&& f,const std::string& what) {
    const auto prefix=s.prefix();const auto stats=s.stats();const auto tap=s.own_tap();
    const ProbeSnapshot probes(s.probe(),tap.rows);
    const FirstRowSnapshot first(s.first_row_probe());
    const auto logits=s.logits();std::vector<float> old(logits.begin(),logits.end());
    const auto physical=copy_device(tap.pointer,tap.rows*W);
    bool rejected=false;try{f();}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,what+" rejected");require(!s.requires_reset(),what+" must not invalidate");
    require(s.prefix()==prefix && s.stats()==stats && s.own_tap()==tap,what+" atomic metadata");
    require(s.logits().data()==logits.data() && s.logits().size()==old.size() &&
        (old.empty() || !std::memcmp(logits.data(),old.data(),logits.size_bytes())),what+" physical logits");
    const auto after=copy_device(tap.pointer,tap.rows*W);
    require(physical.size()==after.size() && (physical.empty() ||
        !std::memcmp(physical.data(),after.data(),physical.size()*sizeof(float))),what+" physical tap");
    probes.check(s.probe(),what+" probe publication");
    first.check(s.first_row_probe(),what+" first-row publication");
}
void failure_preserves(qwen::MtpSession& s,const std::array<qwen::MtpHiddenRow,3>& h,
    const std::array<std::int32_t,3>& t) {
    (void)teacher_run(s,h,t,3);restore_preserves(s,1);
    {
        const auto before=s.stats();const auto tap=s.own_tap();const auto prefix=s.prefix();
        const ProbeSnapshot probes(s.probe(),tap.rows);
        const FirstRowSnapshot first(s.first_row_probe());
        const auto view=s.logits();std::vector<float> old(view.begin(),view.end());
        const auto physical=copy_device(tap.pointer,tap.rows*W);
        s.diagnostic_fail_after_kv(true);
        bool failed=false;try{(void)s.step(std::span(h).subspan(1,2),std::span(t).subspan(1,2));}
        catch(const std::runtime_error&){failed=true;}
        require(failed && s.requires_reset(),"synthetic execution failure requires reset");
        require(s.stats()==before && s.prefix()==prefix && s.own_tap()==tap,"failure metadata preservation");
        require(s.logits().data()==view.data() && s.logits().size()==view.size() &&
            !std::memcmp(view.data(),old.data(),view.size_bytes()),"failed physical logit publication");
        const auto after=copy_device(tap.pointer,tap.rows*W);
        require(!std::memcmp(after.data(),physical.data(),physical.size()*sizeof(float)),"failed physical tap publication");
        probes.check(s.probe(),"synthetic failure probe publication");
        first.check(s.first_row_probe(),"synthetic failure first-row publication");
        bool unusable=false;try{s.restore_prefix(0);}catch(const std::runtime_error&){unusable=true;}
        require(unusable,"failed restore blocked until reset");probes.check(s.probe(),"blocked restore probe publication");
        first.check(s.first_row_probe(),"blocked restore first-row publication");
    } // Retained probe views end before reset invalidates the publication.
    s.reset();(void)ProbeSnapshot(s.probe(),0);
    require(!s.requires_reset() && s.prefix()==0 && s.own_tap().rows==0 && s.logits().empty(),"reset empty publications");
}
struct ComponentResult {Capture n3;qwen::MtpSessionMemory memory;};
ComponentResult component_fixture(qwen::Session& target,qwen::MtpSession& s,
        const Args& a,std::span<const float> input_hidden) {
        const std::vector<float> hidden(input_hidden.begin(),input_hidden.end());
        DeviceFloats inputs(hidden);const auto h=rows(inputs);
        const auto memory=s.memory();
        require(memory.ownership_verified && memory.immutable_ready_slots==512,"independent buffer ledger/READY512");
        require(memory.gpu_experts==2673868800ULL && memory.ram_expert_payload==memory.gpu_experts,"original RAM/GPU Q8 payload");
        require(memory.model_payload_reads==28 && memory.expert_payload_reads==3 && memory.shared_target_payload_reads==0,"payload read-once/borrow-only");
        require(memory.gpu_kv==std::uint64_t(a.capacity)*16*18*2 && memory.gpu_dense_scratch<=152248320ULL,"fullhistory capacity resource bounds");
        const auto n1=teacher_run(s,h,a.tokens,1);compare(n1,teacher_run(s,h,a.tokens,2),"N1 vs N2+N1");
        auto n3=teacher_run(s,h,a.tokens,3);compare(n1,n3,"N1 vs N3");
        compare(n1,teacher_run(s,h,a.tokens,3,true),"poisoned invisible future");
        auto explicit_zero=h;explicit_zero[0]=qwen::MtpHiddenRow::zero();
        compare(n1,teacher_run(s,explicit_zero,a.tokens,3),"explicit ZERO first-row convention");

        for(int k=0;k<=3;++k) {
            (void)teacher_run(s,h,a.tokens,3);const auto completed=s.stats();restore_preserves(s,k);
            require(s.prefix()==std::uint64_t(k) && s.stats().full_rows==completed.full_rows,"logical restore no completed-work rewind");
            if(k<3) {
                s.diagnostic_poison_future();Capture replay;
                capture(s,s.step(std::span(h).subspan(k),std::span(a.tokens).subspan(k)),replay);
                compare(std::span(n1.logits).subspan(k*V),replay.logits,"logical suffix overwrite/replay full logits");
                compare(std::span(n1.tap).subspan(k*W),replay.tap,"logical suffix overwrite/replay D");
            }
        }
        s.reset();teacher_skip_preserves(s,std::span(h).first(2),std::span(a.tokens).first(2));
        require(s.own_tap().rows==0 && s.logits().empty() && s.stats().kv_only_rows==2,"explicit KV-only head skip");
        restore_preserves(s,2); // zero extents remain zero, including all four probes
        compare(std::span(n1.logits).subspan(2*V),s.step(std::span(h).subspan(2),std::span(a.tokens).subspan(2)),"KV-only teacher history");
        s.reset();(void)s.step(std::span(h).first(1),std::span(a.tokens).first(1));
        const auto published=s.own_tap();const auto published_logits=s.logits();
        teacher_skip_preserves(s,std::span(h).subspan(1),std::span(a.tokens).subspan(1));
        require(s.own_tap()==published && s.logits().data()==published_logits.data(),"KV-only preserves successful publication");

        // Real target carry ownership across a SUCCESSFUL verify publication.
        // Captured teacher rows above are external fixture inputs, not asserted
        // equal to this live target run and not promoted to an independent oracle.
        (void)target.step_batch(a.tokens);s.save_target_carry(2);
        const auto old_carry=copy_device(s.saved_target_carry().pointer,W);
        restore_preserves(s,3);
        const std::array proposal0{s.saved_target_carry()};
        Capture draft0;capture(s,s.step(proposal0,std::span(a.verify).first(1)),draft0);
        const auto d0=s.own_tap();DeviceFloats recursive_row(std::span(draft0.tap).first(W));
        // Own HC head input is width2560; D is widened10240. Passing a head tap
        // instead of recursive D MUST fail the typed shape guard.
        require(s.probe().head_input.size()==H && d0.width==W,"D versus own-head extent");
        require(!std::equal(draft0.tap.begin(),draft0.tap.begin()+H,draft0.head.begin()),"D differs from own normalized HC head tap");
        const std::array bad_head{qwen::MtpHiddenRow{1,d0.pointer,H}};
        rejects(s,[&]{(void)s.step(bad_head,std::span(a.verify).subspan(1,1));},"recursive head-tap misuse");
        const std::array recursive{d0.row(0)};Capture draft1;
        capture(s,s.step(recursive,std::span(a.verify).subspan(1,1)),draft1);
        restore_preserves(s,4);const std::array replay_recursive{qwen::MtpHiddenRow{1,recursive_row.pointer,W}};
        Capture draft1_replay;capture(s,s.step(replay_recursive,std::span(a.verify).subspan(1,1)),draft1_replay);
        compare(draft1,draft1_replay,"same ONE trained block recursive proposal2");

        restore_preserves(s,3); // proposal KV is discarded BEFORE target verification
        (void)target.verify_window(a.verify);const auto v=target.target_tap();
        require(copy_device(s.saved_target_carry().pointer,W)==old_carry,"owned old target carry survives verify");
        const std::array rebuild{s.saved_target_carry(),qwen::MtpHiddenRow{1,v.pointer,W},qwen::MtpHiddenRow{1,v.pointer+W,W}};
        Capture teacher_rebuilt;capture(s,s.teacher_append(rebuild,a.verify,qwen::MtpTeacherHead::publish),teacher_rebuilt);
        restore_preserves(s,3);teacher_skip_preserves(s,rebuild,a.verify);
        restore_preserves(s,5); // retains c,a; pending b is excluded, carry is V1
        compare(std::span(teacher_rebuilt.logits).subspan(2*V),
            s.step(std::span(rebuild).subspan(2),std::span(a.verify).subspan(2)),"teacher rebuild/prefix/pending exclusion");
        if(a.capacity==8) rejects(s,[&]{(void)s.step(rebuild,a.verify);},"whole window capacity preflight");
        target.restore_prefix(3); // consume target pending ownership; no sampling

        (void)teacher_run(s,h,a.tokens,3);
        const std::array<std::int32_t,1> bad_id{-1},large_id{V};
        const std::array one{h[1]};
        rejects(s,[&]{(void)s.step(one,bad_id);},"negative token");
        rejects(s,[&]{(void)s.step(one,large_id);},"vocabulary token bound");
        rejects(s,[&]{(void)s.step({},{});},"empty window");
        const std::array<std::int32_t,4> four{0,0,0,0};
        rejects(s,[&]{(void)s.step(h,four);},"N4/hidden mismatch");
        rejects(s,[&]{(void)s.step({},std::span(a.tokens).first(1));},"missing hidden");
        const std::array wrong_device{qwen::MtpHiddenRow{0,inputs.pointer,W}};
        rejects(s,[&]{(void)s.step(wrong_device,std::span(a.tokens).first(1));},"wrong hidden device");
        const std::array zero{qwen::MtpHiddenRow::zero()};
        rejects(s,[&]{(void)s.step(zero,std::span(a.tokens).first(1));},"ZERO only at absolute0");
        const std::array wrapping{qwen::MtpHiddenRow{1,reinterpret_cast<const float*>(std::numeric_limits<std::uintptr_t>::max()-3),W}};
        rejects(s,[&]{(void)s.step(wrapping,std::span(a.tokens).first(1));},"wrapping hidden row");
        rejects(s,[&]{s.restore_prefix(s.prefix()+1);},"restore cannot expose unloaded rows");
        rejects(s,[&]{(void)s.teacher_append(one,std::span(a.tokens).first(1),static_cast<qwen::MtpTeacherHead>(99));},"teacher mode bound");
        // Whole N3 ARGUMENT preflight: both first rows are valid descriptors and
        // IDs for this occupied prefix. No GPU-residency/finiteness preflight is
        // claimed. Row 2 is absolute position3, so explicit null ZERO is invalid.
        restore_preserves(s,1);
        const std::array late_hidden{h[1],h[2],h[2]};
        const std::array late_tokens{a.tokens[1],a.tokens[2],a.tokens[2]};
        for(const auto id:{std::int32_t(-1),std::int32_t(V)}) {
            auto invalid=late_tokens;invalid[2]=id;
            rejects(s,[&]{(void)s.step(late_hidden,invalid);},"N3 last-row invalid ID");
        }
        auto late_bad=late_hidden;late_bad[2].device=0;
        rejects(s,[&]{(void)s.step(late_bad,late_tokens);},"N3 last-row wrong device");
        late_bad=late_hidden;late_bad[2].elements=W-1;
        rejects(s,[&]{(void)s.step(late_bad,late_tokens);},"N3 last-row short hidden width");
        late_bad=late_hidden;late_bad[2]=qwen::MtpHiddenRow::zero();
        rejects(s,[&]{(void)s.step(late_bad,late_tokens);},"N3 last-row null at nonzero position");
        Capture continued;capture(s,s.step(std::span(h).subspan(1),std::span(a.tokens).subspan(1)),continued);
        compare(std::span(n1.logits).subspan(V),continued.logits,"late rejection same-history continuation logits");
        compare(std::span(n1.tap).subspan(W),continued.tap,"late rejection same-history continuation D");
        failure_preserves(s,h,a.tokens);

        // Numeric zero-first failure also preserves the previous physical result.
        (void)teacher_run(s,h,a.tokens,3);restore_preserves(s,0);
        {
            const auto tap=s.own_tap();const auto old_stats=s.stats();const auto old_logits=s.logits();
            const ProbeSnapshot zero_probes(s.probe(),tap.rows);
            const FirstRowSnapshot zero_first(s.first_row_probe());
            const std::vector<float> saved_logits(old_logits.begin(),old_logits.end());
            const auto saved_tap=copy_device(tap.pointer,tap.rows*W);
            auto invalid_hidden=hidden;invalid_hidden[0]=1.0f;DeviceFloats bad_zero(invalid_hidden);
            const auto bad_rows=rows(bad_zero);bool numeric_failed=false;
            try{(void)s.step(std::span(bad_rows).first(1),std::span(a.tokens).first(1));}
            catch(const std::runtime_error&){numeric_failed=true;}
            require(numeric_failed && s.requires_reset() && s.stats()==old_stats && s.own_tap()==tap,"nonzero-first invalidation/publication");
            const auto after_D=copy_device(tap.pointer,tap.rows*W);
            require(s.logits().data()==old_logits.data() && s.logits().size()==old_logits.size() &&
                !std::memcmp(after_D.data(),saved_tap.data(),saved_tap.size()*sizeof(float)) &&
                !std::memcmp(old_logits.data(),saved_logits.data(),old_logits.size_bytes()),"numeric failure physical publication");
            zero_probes.check(s.probe(),"zero-first genuine failure probe publication");
            zero_first.check(s.first_row_probe(),"zero-first genuine failure first-row publication");
        } // No retained probe read after the following successful reset/forward.
        s.reset();compare(n1,teacher_run(s,h,a.tokens,3),"reset deterministic replay");
        // Genuine nonfinite input at a NONZERO absolute position; no fault hook.
        restore_preserves(s,1);const auto finite_tap=s.own_tap();const auto finite_stats=s.stats();
        {
            const ProbeSnapshot finite_probes(s.probe(),finite_tap.rows);
            const FirstRowSnapshot finite_first(s.first_row_probe());
            const auto finite_logits=s.logits();const std::vector<float> finite_copy(finite_logits.begin(),finite_logits.end());
            const auto finite_D=copy_device(finite_tap.pointer,finite_tap.rows*W);
            auto nonfinite_hidden=hidden;nonfinite_hidden[W+17]=std::numeric_limits<float>::quiet_NaN();
            DeviceFloats nonfinite_input(nonfinite_hidden);const auto nonfinite_rows=rows(nonfinite_input);
            bool nonfinite_failed=false;
            try{(void)s.step(std::span(nonfinite_rows).subspan(1,2),std::span(a.tokens).subspan(1,2));}
            catch(const std::runtime_error&){nonfinite_failed=true;}
            require(nonfinite_failed && s.requires_reset() && s.prefix()==1 && s.stats()==finite_stats &&
                s.own_tap()==finite_tap,"genuine numeric failure preserves metadata");
            const auto after_D=copy_device(finite_tap.pointer,finite_tap.rows*W);
            require(s.logits().data()==finite_logits.data() && s.logits().size()==finite_logits.size() &&
                !std::memcmp(finite_logits.data(),finite_copy.data(),finite_logits.size_bytes()) &&
                !std::memcmp(after_D.data(),finite_D.data(),finite_D.size()*sizeof(float)),"genuine numeric failure preserves physical publication");
            finite_probes.check(s.probe(),"nonfinite genuine failure probe publication");
            finite_first.check(s.first_row_probe(),"nonfinite genuine failure first-row publication");
        }
        s.reset();compare(n1,teacher_run(s,h,a.tokens,3),"genuine error/reset/reuse");
        const auto after=s.memory();
        require(after.ownership_verified && after.owned_bytes==memory.owned_bytes &&
            after.model_payload_reads==memory.model_payload_reads && after.model_payload_bytes==memory.model_payload_bytes &&
            after.expert_payload_reads==3 && after.shared_target_payload_reads==0,"no hot allocation/model reread");
        return {std::move(n3),after};
}
void report_component(const Args& a,const qwen::MtpSessionMemory& after) {
        std::cout<<"{\"protocol\":1,\"test\":\"trained_mtp_forward_self_fixture\",\"passed\":true,\"independent_teacher_oracle\":false,"
            <<"\"revision\":\""<<CORE_REVISION<<"\",\"dirty\":"<<(CORE_DIRTY ? "true" : "false")<<','
            <<"\"hnorm_contract\":\"mx_per_branch_2560_gamma_4x2560\",\"hnorm_width\":10240,"
            <<"\"hnorm_group\":2560,\"hnorm_groups_per_token\":4,\"hnorm_gamma_shared\":false,\"dataset_id\":"<<std::quoted(a.dataset)
            <<",\"capture_origin\":"<<std::quoted(a.origin)<<",\"hidden_file\":"<<std::quoted(a.hidden)
            <<",\"target_file\":"<<std::quoted(a.target)<<",\"sidecar_file\":"<<std::quoted(a.sidecar)
            <<",\"tokens\":["<<a.tokens[0]<<','<<a.tokens[1]<<','<<a.tokens[2]<<"],\"verify_tokens\":["
            <<a.verify[0]<<','<<a.verify[1]<<','<<a.verify[2]<<"],\"capacity\":"<<a.capacity
            <<",\"full_vocab\":248320,\"max_abs_self_error\":"<<max_error
            <<",\"atol\":"<<absolute_tolerance<<",\"rtol\":"<<relative_tolerance<<",\"owned_bytes\":"<<after.owned_bytes
            <<",\"gpu_nonexperts\":"<<after.gpu_nonexperts<<",\"gpu_experts\":"<<after.gpu_experts
            <<",\"gpu_kv\":"<<after.gpu_kv<<",\"gpu_dense_scratch\":"<<after.gpu_dense_scratch
            <<",\"model_payload_reads\":"<<after.model_payload_reads<<",\"model_payload_bytes\":"<<after.model_payload_bytes
            <<",\"target_payload_reads\":0}\n";
}

// Fixed protocol1 payloads. These are owned CPU copies, never expired borrowed
// spans or normalized head inputs. This struct survives BOTH runtime owners.
struct TeacherArtifact {
    std::array<std::int32_t,teacher_rows> tokens{};
    std::vector<float> previous_hidden,own_D,own_logits;
    qwen::MtpSessionMemory memory;
    FirstRowArtifact first_row;
};
std::array<std::int32_t,teacher_rows> read_teacher_tokens(const std::string& path) {
    std::array<std::int32_t,teacher_rows> tokens{};
    require(std::filesystem::file_size(path)==sizeof(tokens),"teacher tokens must be exactly 32 little-endian int32 IDs");
    std::ifstream file(path,std::ios::binary);
    require(bool(file.read(reinterpret_cast<char*>(tokens.data()),sizeof(tokens))),"read 32 teacher tokens");
    for(auto id:tokens)require(id>=0 && id<V,"teacher token vocabulary bound");
    return tokens;
}
std::vector<float> target_previous_hidden(qwen::Session& target,
        const std::array<std::int32_t,teacher_rows>& tokens) {
    target.reset();std::vector<float> previous(teacher_rows*W,0.0f);
    for(int first=0;first<teacher_rows;first+=N) {
        const int n=std::min(N,teacher_rows-first);
        const auto logits=target.step_batch(std::span(tokens).subspan(first,n));
        require(logits.size()==std::size_t(n)*V,"target full-logit extent during capture");
        for(float x:logits)require(std::isfinite(x),"nonfinite target capture logit");
        const auto tap=target.target_tap();
        require(tap.device==1 && tap.pointer && tap.rows==n && tap.width==W &&
            tap.first_position==std::uint64_t(first),"typed chronological target pre-root-HC tap");
        // Copy EACH full row before the next successful target call expires it.
        const auto raw=copy_device(tap.pointer,std::size_t(n)*W);
        for(float x:raw)require(std::isfinite(x),"nonfinite target pre-root-HC tap");
        for(int t=0;t<n && first+t+1<teacher_rows;++t)
            std::copy_n(raw.data()+t*W,W,previous.data()+(first+t+1)*W);
    }
    return previous; // row0 ZERO; rowp actual T[p-1], absolute positions 0..31
}
Capture chronological_run(qwen::MtpSession& s,
        const std::array<qwen::MtpHiddenRow,teacher_rows>& h,
        const std::array<std::int32_t,teacher_rows>& tokens,int width,bool ordinary=false,
        FirstRowArtifact* first_row=nullptr,int probe_position=0) {
    require(width>=1 && width<=3,"bounded chronological teacher width");
    require(probe_position>=0 && probe_position<teacher_rows,"bounded chronological diagnostic position");
    s.reset();Capture out;
    for(int first=0;first<teacher_rows;first+=width) {
        const int n=std::min(width,teacher_rows-first);
        const auto hidden=std::span(h).subspan(first,n);
        const auto ids=std::span(tokens).subspan(first,n);
        const auto logits=ordinary ? s.step(hidden,ids) :
            s.teacher_append(hidden,ids,qwen::MtpTeacherHead::publish);
        require(s.prefix()==std::uint64_t(first+n) && s.own_tap().first_position==std::uint64_t(first) &&
            s.own_tap().rows==n,"MTP absolute chronological publication");
        capture(s,logits,out);
        if(first_row && first<=probe_position && probe_position<first+n) {
            require(s.first_row_probe().first_position==std::uint64_t(probe_position),"exact selected diagnostic row");
            first_row->capture(s.first_row_probe(),first,n);
        }
    }
    require(out.logits.size()==teacher_rows*V && out.tap.size()==teacher_rows*W,"32 full frozen output rows");
    return out;
}
void chronological_fixture(qwen::MtpSession& s,const Args& a,TeacherArtifact& artifact) {
    require(artifact.previous_hidden.size()==teacher_rows*W,"teacher previous-hidden extent");
    DeviceFloats inputs(artifact.previous_hidden);
    std::array<qwen::MtpHiddenRow,teacher_rows> h{};
    for(int p=0;p<teacher_rows;++p)h[p]={1,inputs.pointer+p*W,W};
    const auto initial=s.memory();auto baseline=chronological_run(s,h,artifact.tokens,1,false,
        a.capture_first_row && a.teacher_width==1 ? &artifact.first_row : nullptr,a.capture_probe_position);
    Capture selected;if(a.teacher_width==1)selected=baseline;
    for(int width=2;width<=3;++width) {
        auto candidate=chronological_run(s,h,artifact.tokens,width,false,
            a.capture_first_row && a.teacher_width==width ? &artifact.first_row : nullptr,a.capture_probe_position);
        compare(baseline,candidate,"32 chronological teacher N1 versus N"+std::to_string(width));
        if(a.teacher_width==width)selected=std::move(candidate);
    }
    compare(baseline,chronological_run(s,h,artifact.tokens,3,true),"32 ordinary/reset/full-history reuse");
    restore_preserves(s,29);s.diagnostic_poison_future();Capture replay;
    capture(s,s.step(std::span(h).subspan(29),std::span(artifact.tokens).subspan(29)),replay);
    compare(std::span(baseline.logits).subspan(29*V),replay.logits,"occupied teacher prefix29 suffix overwrite logits");
    compare(std::span(baseline.tap).subspan(29*W),replay.tap,"occupied teacher prefix29 suffix overwrite D");
    s.reset();
    for(int first=0;first<29;first+=N) {
        const int n=std::min(N,29-first);
        teacher_skip_preserves(s,std::span(h).subspan(first,n),std::span(artifact.tokens).subspan(first,n));
    }
    require(s.prefix()==29 && s.logits().empty() && s.own_tap().rows==0,"KV-only occupied teacher history");
    Capture after_skip;capture(s,s.step(std::span(h).subspan(29),std::span(artifact.tokens).subspan(29)),after_skip);
    compare(std::span(baseline.logits).subspan(29*V),after_skip.logits,"KV-only occupied history full logits");
    compare(std::span(baseline.tap).subspan(29*W),after_skip.tap,"KV-only occupied history D");
    if(a.capture_first_row)require(artifact.first_row.captured &&
        artifact.first_row.first_position==std::uint64_t(a.capture_probe_position) &&
        !std::memcmp(artifact.first_row.values[11].data(),selected.tap.data()+a.capture_probe_position*W,W*sizeof(float)) &&
        !std::memcmp(artifact.first_row.values[12].data(),selected.head.data()+a.capture_probe_position*H,H*sizeof(float)),
        "selected-row D/head diagnostic bits match the selected frozen publication");
    artifact.own_D=std::move(selected.tap);artifact.own_logits=std::move(selected.logits);artifact.memory=s.memory();
    require(artifact.memory.ownership_verified && artifact.memory.owned_bytes==initial.owned_bytes &&
        artifact.memory.model_payload_reads==28 && artifact.memory.expert_payload_reads==3 &&
        artifact.memory.shared_target_payload_reads==0 && artifact.memory.model_payload_bytes==initial.model_payload_bytes,
        "32 teacher no hot allocation/model payload reread");
}
void validate_teacher_artifact(const Args& a,const TeacherArtifact& artifact) {
    require(a.capacity==capture_capacity && a.teacher_width>=1 && a.teacher_width<=3,"fixed capture capacity/width");
    require(artifact.previous_hidden.size()==teacher_rows*W && artifact.own_D.size()==teacher_rows*W &&
        artifact.own_logits.size()==teacher_rows*V,"exact protocol1 array dimensions");
    for(auto id:artifact.tokens)require(id>=0 && id<V,"captured ID bound");
    for(const auto* array:{&artifact.previous_hidden,&artifact.own_D,&artifact.own_logits})
        for(float x:*array)require(std::isfinite(x),"nonfinite protocol1 array");
    for(int j=0;j<W;++j)require(artifact.previous_hidden[j]==0.0f,"fresh global row0 ZERO production convention");
    for(int p=1;p<teacher_rows;++p)
        require(std::any_of(artifact.previous_hidden.begin()+p*W,artifact.previous_hidden.begin()+(p+1)*W,
            [](float x){return x!=0.0f;}),"subsequent teacher row is a real nonzero target tap");
    require(artifact.memory.ownership_verified && artifact.memory.immutable_ready_slots==512 &&
        artifact.memory.model_payload_reads==28 && artifact.memory.expert_payload_reads==3 &&
        artifact.memory.shared_target_payload_reads==0 && artifact.memory.model_payload_bytes==2772343808ULL &&
        artifact.memory.expert_payload_bytes==2673868800ULL,"capture read-once typed payload contract");
    require(artifact.first_row.captured==a.capture_first_row,"first-row diagnostic opt-in contract");
    if(a.capture_first_row)require(artifact.first_row.first_position==std::uint64_t(a.capture_probe_position) &&
        artifact.first_row.source_window_rows>=1 && artifact.first_row.source_window_rows<=a.teacher_width &&
        artifact.first_row.source_first_position<=artifact.first_row.first_position &&
        artifact.first_row.first_position-artifact.first_row.source_first_position<std::uint64_t(artifact.first_row.source_window_rows),
        "selected absolute row and actual source-window metadata");
    if(a.capture_first_row)for(std::size_t i=0;i<first_row_sites.size();++i) {
        const auto& values=artifact.first_row.values[i];
        require(values.size()==first_row_sites[i].columns,"exact first-row diagnostic extent before export");
        for(float x:values)require(std::isfinite(x),"finite first-row diagnostic before export");
    }
}
void write_capture_descriptor(std::ostream& out) {
    // READ-ONLY handoff contract with tools/mtp_teacher_oracle.cpp::preflight.
    // Do not add keys: its schema deliberately rejects unknown metadata.
    out<<"{\"schema\":\"gfx906-mtp-teacher-capture-v1\",\"source\":{\"revision\":\""
        <<CORE_REVISION<<"\",\"dirty\":"<<(CORE_DIRTY ? "true" : "false")
        <<"},\"geometry\":{\"rows\":32,\"hidden\":10240,\"vocab\":248320},\"positions\":[";
    for(int p=0;p<teacher_rows;++p){if(p)out<<',';out<<p;}
    out<<"],\"zero_row0\":true,\"files\":{"
        <<"\"tokens\":{\"name\":\"teacher_tokens.i32.bin\",\"dtype\":\"i32le\",\"shape\":[32],\"bytes\":128},"
        <<"\"previous_hidden\":{\"name\":\"teacher_previous_hidden.f32.bin\",\"dtype\":\"f32le\",\"shape\":[32,10240],\"bytes\":1310720},"
        <<"\"D\":{\"name\":\"own_D.f32.bin\",\"dtype\":\"f32le\",\"shape\":[32,10240],\"bytes\":1310720},"
        <<"\"logits\":{\"name\":\"own_logits.f32.bin\",\"dtype\":\"f32le\",\"shape\":[32,248320],\"bytes\":31784960}}}\n";
}
void write_first_row_artifact(const std::filesystem::path& directory,const Args& a,
        const TeacherArtifact& artifact) {
    if(!a.capture_first_row)return;
    // Fixed comparable graph sites, ONE absolute row, no generic callback/export
    // registry. The donor's hc_mixed/hc_combine labels repeat within blk48, so
    // the attention/FFN occurrence is explicit in our corresponding filenames.
    std::ofstream out(directory/"first_row.json",std::ios::binary|std::ios::trunc);
    out<<"{\"schema\":\"gfx906-mtp-first-row-v1\",\"revision\":\""<<CORE_REVISION
        <<"\",\"dirty\":"<<(CORE_DIRTY ? "true" : "false")
        <<",\"first_position\":"<<artifact.first_row.first_position<<",\"rows\":1,\"source_teacher_width\":"<<a.teacher_width
        <<",\"source_first_position\":"<<artifact.first_row.source_first_position
        <<",\"source_window_rows\":"<<artifact.first_row.source_window_rows
        <<",\"source_row_index\":"<<artifact.first_row.first_position-artifact.first_row.source_first_position
        <<",\"runtime_destroyed_before_export\":true,\"hf_training_proof\":false,\"sites\":[";
    std::size_t total_bytes=0;
    const auto site=[&](const char* name,std::span<const float> values,bool comma) {
        require(!values.empty() && values.size()<=std::size_t(V),"bounded first-row export site");
        for(float x:values)require(std::isfinite(x),"finite first-row export site");
        const std::string file="tensor-p"+std::to_string(artifact.first_row.first_position)+"-"+name+".f32.bin";
        write_binary(directory/file,std::as_bytes(values));total_bytes+=values.size_bytes();
        require(total_bytes<=2*1024*1024,"first-row artifact byte bound");
        if(comma)out<<',';
        out<<"{\"name\":";json_string(out,name);out<<",\"file\":";json_string(out,file);
        out<<",\"dtype\":\"f32le\",\"rows\":1,\"columns_per_row\":"<<values.size()
            <<",\"bytes\":"<<values.size_bytes()<<",\"layout\":\"C\",\"nonfinite_count\":0}";
    };
    for(std::size_t i=0;i<first_row_sites.size();++i)
        site(first_row_sites[i].name,artifact.first_row.values[i],i!=0);
    site("h_nextn",artifact.first_row.values[11],true); // same full pre-own-HC D
    site("result_output",std::span(artifact.own_logits).subspan(a.capture_probe_position*V,V),true);
    out<<"],\"total_bytes\":"<<total_bytes<<"}\n";
    require(bool(out) && out.tellp()>0 && out.tellp()<=16384,"bounded first-row diagnostic metadata");
    out.flush();require(bool(out),"flush first-row diagnostic metadata");
    out.close();require(!out.fail(),"close first-row diagnostic metadata");
}
void write_teacher_artifact(const Args& a,const TeacherArtifact& artifact) {
    // Called ONLY after the lexical scope holding MtpSession and Session ends.
    // Validate all extents/finiteness BEFORE creating the fresh output directory.
    validate_teacher_artifact(a,artifact);new_capture_directory(a.capture_dir);
    const std::filesystem::path directory(a.capture_dir);
    require(std::filesystem::create_directory(directory),"create fresh capture directory");
    write_binary(directory/"teacher_tokens.i32.bin",std::as_bytes(std::span(artifact.tokens)));
    write_binary(directory/"teacher_previous_hidden.f32.bin",std::as_bytes(std::span(artifact.previous_hidden)));
    write_binary(directory/"own_D.f32.bin",std::as_bytes(std::span(artifact.own_D)));
    write_binary(directory/"own_logits.f32.bin",std::as_bytes(std::span(artifact.own_logits)));
    write_first_row_artifact(directory,a,artifact);
    std::ofstream out(directory/"capture_metadata.json",std::ios::binary|std::ios::trunc);
    out<<std::setprecision(17)<<"{\"protocol\":"<<capture_protocol<<",\"kind\":\"mtp_teacher_capture\",\"passed\":true,"
        <<"\"revision\":\""<<CORE_REVISION<<"\",\"dirty\":"<<(CORE_DIRTY ? "true" : "false")
        <<",\"rows\":32,\"capacity\":64,\"expert_slots\":112,\"teacher_width\":"<<a.teacher_width
        <<",\"capture_first_row\":"<<(a.capture_first_row ? "true" : "false")
        <<",\"capture_probe_position\":"<<a.capture_probe_position<<",\"dataset_id\":";
    json_string(out,a.dataset);out<<",\"capture_origin\":";json_string(out,a.origin);
    out<<",\"target_file\":";json_string(out,a.target);out<<",\"sidecar_file\":";json_string(out,a.sidecar);
    out<<",\"teacher_tokens_source\":";json_string(out,a.teacher_tokens);
    out<<",\"geometry\":{\"layer\":48,\"predict_layers\":1,\"hidden_size\":2560,\"widened_size\":10240,"
        <<"\"hc_count\":4,\"hc_rank\":320,\"query_heads\":24,\"kv_heads\":2,\"head_dim\":256,"
        <<"\"experts\":512,\"experts_used\":10,\"intermediate_size\":640,\"vocabulary_size\":248320,"
        <<"\"rotary_dim\":64,\"rope_base\":10000000,\"rope_sections\":[11,11,10,0],\"rms_epsilon\":1e-6,"
        <<"\"hnorm_group\":2560,\"hnorm_gamma_shared\":false,\"compression\":0,\"kv_type\":\"Q4_0\","
        <<"\"projection_type\":\"Q8_0\",\"norm_type\":\"F32\",\"embedding_type\":\"Q4_0\",\"output_type\":\"Q6_K\"}"
        <<",\"hnorm_source\":{\"revision\":\"dcd685463d597d31f5ca759d32c94592a2740fa4\","
        <<"\"file\":\"src/models/qwen4exp.cpp\",\"lines\":[438,450]}"
        <<",\"positions\":[";
    for(int p=0;p<teacher_rows;++p){if(p)out<<',';out<<p;}
    out<<"],\"previous_hidden_positions\":[null";
    for(int p=1;p<teacher_rows;++p)out<<','<<p-1;
    out<<"],\"teacher_boundary\":{\"first_position\":0,\"row0\":\"ZERO\",\"subsequent\":\"target_H[p-1]\","
        <<"\"target_hidden\":\"layer47_pre_root_hc\",\"own_D\":\"blk48_pre_own_hc_head\","
        <<"\"convention\":\"pinned_production_zero_first_row\"},\"arrays\":{"
        <<"\"teacher_tokens\":{\"file\":\"teacher_tokens.i32.bin\",\"dtype\":\"<i4\",\"shape\":[32],\"layout\":\"C\",\"byte_size\":128},"
        <<"\"teacher_previous_hidden\":{\"file\":\"teacher_previous_hidden.f32.bin\",\"dtype\":\"<f4\",\"shape\":[32,10240],\"layout\":\"C\",\"byte_size\":1310720},"
        <<"\"own_D\":{\"file\":\"own_D.f32.bin\",\"dtype\":\"<f4\",\"shape\":[32,10240],\"layout\":\"C\",\"byte_size\":1310720},"
        <<"\"own_logits\":{\"file\":\"own_logits.f32.bin\",\"dtype\":\"<f4\",\"shape\":[32,248320],\"layout\":\"C\",\"byte_size\":31784960}}"
        <<",\"all_finite\":true,\"runtime_destroyed_before_export\":true,\"independent_teacher_oracle\":false,"
        <<"\"hf_training_proof\":false,\"gate_waiver\":false,\"reference_gates\":{"
        <<"\"full_logits\":{\"atol\":0.02,\"rtol\":0.002},\"D\":{\"atol\":0.002,\"rtol\":0.002}}"
        <<",\"self_checks\":{\"widths\":[1,2,3],\"ordinary\":true,\"reset\":true,\"genuine_error_reuse\":true,"
        <<"\"restore_prefix\":29,\"poison_suffix_overwrite\":true,\"kv_only_history\":true,\"atol\":"<<absolute_tolerance
        <<",\"rtol\":"<<relative_tolerance<<",\"max_abs_error\":"<<max_error<<'}'
        <<",\"model_payload_reads\":"<<artifact.memory.model_payload_reads
        <<",\"model_payload_bytes\":"<<artifact.memory.model_payload_bytes
        <<",\"expert_payload_reads\":"<<artifact.memory.expert_payload_reads
        <<",\"expert_payload_bytes\":"<<artifact.memory.expert_payload_bytes
        <<",\"target_payload_reads\":0,\"immutable_ready_slots\":512,\"owned_bytes\":"<<artifact.memory.owned_bytes<<"}\n";
    require(bool(out) && out.tellp()>0 && out.tellp()<=32768,"bounded capture metadata write");
    out.flush();require(bool(out),"flush capture metadata");out.close();require(!out.fail(),"close capture metadata");
    std::ofstream descriptor(directory/"capture.json.tmp",std::ios::binary|std::ios::trunc);
    write_capture_descriptor(descriptor);
    require(bool(descriptor) && descriptor.tellp()>0 && descriptor.tellp()<=4096,"bounded capture descriptor write");
    descriptor.flush();require(bool(descriptor),"flush capture descriptor");
    descriptor.close();require(!descriptor.fail(),"close capture descriptor");
    // Publish the oracle's completion marker atomically, after every payload and
    // the richer metadata file has been completely written and closed.
    std::filesystem::rename(directory/"capture.json.tmp",directory/"capture.json");
}
} // namespace

int main(int argc,char** argv) {
    try {
        for(char c:std::string_view(CORE_REVISION))require((c>='0' && c<='9') || (c>='a' && c<='f'),"actual lowercase source revision required");
        auto a=args(argc,argv);TeacherArtifact artifact;ComponentResult component;
        std::vector<float> hidden;
        if(a.capture_dir.empty())hidden=read_hidden(a.hidden);
        else {
            artifact.tokens=read_teacher_tokens(a.teacher_tokens); // ALL32 IDs preflight before model mutation
            std::copy_n(artifact.tokens.begin(),N,a.tokens.begin());
            std::copy_n(artifact.tokens.begin()+N,N,a.verify.begin());
        }
        {
            qwen::SessionConfig tc;tc.capacity=a.capacity;tc.expert_slots=capture_slots;
            tc.max_batch_tokens=N;tc.speculative_checkpoints=true;
            qwen::Session target(a.target,tc);const auto target_before=target.memory();
            const auto borrowed=target.borrow_mtp_weights();
            qwen::MtpSession s(target,a.sidecar,{a.capacity,true,a.capture_first_row,a.capture_probe_position});
            const auto target_after=target.memory();const auto borrowed_after=target.borrow_mtp_weights();
            require(target_before.devices[1].owned_bytes==target_after.devices[1].owned_bytes &&
                target_before.expert_payload_reads==target_after.expert_payload_reads &&
                borrowed.output.pointer==borrowed_after.output.pointer &&
                borrowed.embedding.cpu_bytes.data()==borrowed_after.embedding.cpu_bytes.data(),"no target payload copies/allocations");
            if(!a.capture_dir.empty()) {
                artifact.previous_hidden=target_previous_hidden(target,artifact.tokens);
                hidden.assign(artifact.previous_hidden.begin(),artifact.previous_hidden.begin()+N*W);
                target.reset(); // component's live target verify starts at its true absolute0
            }
            component=component_fixture(target,s,a,hidden);
            if(!a.capture_dir.empty())chronological_fixture(s,a,artifact);
        } // MtpSession drains/destroys FIRST, then the borrowed target Session.
        if(!a.capture_dir.empty())write_teacher_artifact(a,artifact);
        write_f32(a.logits_out,component.n3.logits);write_f32(a.tap_out,component.n3.tap);
        report_component(a,component.memory);
        return 0;
    } catch(const std::exception& e) {std::cerr<<"mtp_session_test: "<<e.what()<<'\n';return 1;}
}
