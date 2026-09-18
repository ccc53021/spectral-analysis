// Exact template FWHT / histogram kernel. No floating point probability arithmetic.
// Build: g++ -O3 -march=native -fopenmp -std=c++17 outer_kernel.cpp -o outer_kernel
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace fs = std::filesystem;
using Pair = std::pair<uint32_t, uint32_t>;
#ifndef FWHT_TILE_LOG2
#define FWHT_TILE_LOG2 10
#endif

template<class T> T read(std::istream &in) {
    T value{}; in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated binary input");
    return value;
}
template<class T> void write(std::ostream &out, T value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
void magic(std::istream &in, const char *expected) {
    char actual[8]; in.read(actual, 8);
    if (!in || std::memcmp(actual, expected, 8)) throw std::runtime_error("wrong binary file magic");
}

struct Data {
    uint32_t terms{}, length{};
    std::array<uint32_t, 4> dimensions{};
    std::vector<uint32_t> indices;
    std::vector<int8_t> weights;
    std::array<std::vector<int8_t>, 4> columns;
    std::string signature;
    uint64_t safety_bound{};
    explicit Data(const fs::path &path) {
        std::ifstream in(path, std::ios::binary); magic(in, "BLKOUTR1");
        signature.resize(64); in.read(signature.data(), 64);
        terms=read<uint32_t>(in); length=read<uint32_t>(in);
        for (auto &d:dimensions) d=read<uint32_t>(in);
        if (length!=(1u<<21) || dimensions!=std::array<uint32_t,4>{66,66,66,4})
            throw std::runtime_error("unexpected shape");
        indices.resize(terms); weights.resize(terms);
        in.read(reinterpret_cast<char *>(indices.data()), terms*sizeof(uint32_t));
        in.read(reinterpret_cast<char *>(weights.data()), terms);
        for (int c=0;c<4;++c) {
            columns[c].resize(size_t(dimensions[c])*terms);
            in.read(reinterpret_cast<char *>(columns[c].data()), columns[c].size());
        }
        if (!in) throw std::runtime_error("truncated coefficient data");
        for(uint32_t i=0;i<terms;++i) {
            uint64_t bound=std::abs(int(weights[i]));
            for(int c=0;c<4;++c) {
                int maximum=0;
                for(uint32_t k=0;k<dimensions[c];++k)
                    maximum=std::max(maximum,std::abs(int(columns[c][size_t(k)*terms+i])));
                bound*=maximum;
            }
            safety_bound+=bound;
        }
        if (safety_bound>=uint64_t(std::numeric_limits<int32_t>::max()))
            throw std::runtime_error("int32 safety bound failed");
    }
};

struct Result {
    uint32_t id{},minimum{},maximum{},shift{};
    int64_t l1_lower{};
    uint64_t elapsed_ns{};
    std::vector<Pair> histogram;
};

void fwht(int32_t *__restrict a, uint32_t length) {
    // The first ten stages stay inside a 4 KiB tile instead of repeatedly
    // streaming the entire 8 MiB table. All intermediate values are bounded
    // by the independently checked, universal coefficient L1 bound.
    constexpr uint32_t tile=1u<<FWHT_TILE_LOG2;
    for(uint32_t base=0;base<length;base+=tile) {
        for(uint32_t width=1;width<tile;width*=2) {
            for(uint32_t start=base;start<base+tile;start+=2*width) {
                int32_t *__restrict left=a+start;
                int32_t *__restrict right=left+width;
                #pragma omp simd
                for(uint32_t j=0;j<width;++j) {
                    int32_t x=left[j],y=right[j]; left[j]=x+y;right[j]=x-y;
                }
            }
        }
    }
    for(uint32_t width=tile;width<length;width*=2) {
        for(uint32_t start=0;start<length;start+=2*width) {
            int32_t *__restrict left=a+start;
            int32_t *__restrict right=left+width;
            #pragma omp simd
            for(uint32_t j=0;j<width;++j) {
                int32_t x=left[j],y=right[j];left[j]=x+y;right[j]=x-y;
            }
        }
    }
}

Result calculate(const Data &data,uint32_t id,bool bound_only,std::vector<int32_t> &dense,std::vector<uint32_t> &counts) {
    auto began=std::chrono::steady_clock::now();
    Result result;result.id=id;
    uint32_t temp=id;
    std::array<uint32_t,4> classes{};
    for(int c=3;c>=0;--c){classes[c]=temp%data.dimensions[c];temp/=data.dimensions[c];}
    if(temp)throw std::runtime_error("template id out of range");
    std::array<const int8_t *,4> v{};
    for(int c=0;c<4;++c)v[c]=data.columns[c].data()+size_t(classes[c])*data.terms;
    if(!bound_only){dense.resize(data.length);std::fill(dense.begin(),dense.end(),0);}
    int64_t l1=0,dc=0;
    for(uint32_t i=0;i<data.terms;++i){
        int32_t n=int32_t(data.weights[i])*v[0][i]*v[1][i]*v[2][i]*v[3][i];
        if(!bound_only)dense[data.indices[i]]=n;
        l1+=std::abs(n);if(data.indices[i]==0)dc=n;
    }
    result.l1_lower=2*dc-l1;
    if(bound_only){result.elapsed_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-began).count();return result;}
    fwht(dense.data(),data.length);
    int32_t minimum=std::numeric_limits<int32_t>::max(),maximum=0;
    uint32_t bits=0;
    for(int32_t value:dense){minimum=std::min(minimum,value);maximum=std::max(maximum,value);bits|=uint32_t(value);}
    if(minimum<0)throw std::runtime_error("negative exact endpoint probability");
    result.minimum=minimum;result.maximum=maximum;
    result.shift=bits?__builtin_ctz(bits):0;
    counts.assign((uint32_t(maximum)>>result.shift)+1,0);
    for(int32_t value:dense)++counts[uint32_t(value)>>result.shift];
    for(uint32_t i=0;i<counts.size();++i)if(counts[i])result.histogram.emplace_back(i<<result.shift,counts[i]);
    result.elapsed_ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-began).count();
    return result;
}

std::map<uint32_t,std::vector<Pair>> references(const fs::path &path) {
    std::map<uint32_t,std::vector<Pair>> out;
    if(path.empty())return out;
    std::ifstream in(path,std::ios::binary);magic(in,"OUTREF01");
    uint32_t size=read<uint32_t>(in);
    for(uint32_t i=0;i<size;++i){
        uint32_t id=read<uint32_t>(in),count=read<uint32_t>(in);
        auto &rows=out[id];rows.reserve(count);
        for(uint32_t j=0;j<count;++j){uint32_t n=read<uint32_t>(in),c=read<uint32_t>(in);rows.emplace_back(n,c);}
    }
    return out;
}

void status(const fs::path &out,const std::string &state,size_t done,size_t total,double elapsed,int threads,int64_t min_lower,size_t references_checked,const Data &data,bool bound_only) {
    fs::path temporary=out/"status.tmp";
    std::ofstream f(temporary);
    f<<"{\n  \"state\": \""<<state<<"\",\n  \"mode\": \""<<(bound_only?"bound":"hist")<<"\",\n  \"completed\": "<<done<<",\n  \"total\": "<<total<<",\n  \"threads\": "<<threads<<",\n  \"elapsed_seconds\": "<<elapsed<<",\n  \"eta_seconds\": "<<(done?elapsed*(total-done)/done:0)<<",\n  \"minimum_L1_lower_bound\": "<<min_lower<<",\n  \"python_histograms_checked\": "<<references_checked<<",\n  \"universal_int32_safety_bound\": "<<data.safety_bound<<",\n  \"data_sha256\": \""<<data.signature<<"\"\n}\n";
    f.close();fs::rename(temporary,out/"status.json");
}

int main(int argc,char **argv) {
 try {
    fs::path data_path="outer_data.bin",cases_path,reference_path,out="kernel_output";
    uint32_t start=0,count=1024,chunk=64;int threads=4;bool bound_only=false,resume=false;
    for(int i=1;i<argc;++i){std::string option=argv[i];
        auto value=[&](){if(i+1>=argc)throw std::runtime_error("missing option value");return std::string(argv[++i]);};
        if(option=="--data")data_path=value();else if(option=="--cases")cases_path=value();
        else if(option=="--reference")reference_path=value();else if(option=="--output")out=value();
        else if(option=="--start")start=std::stoul(value());else if(option=="--count")count=std::stoul(value());
        else if(option=="--chunk")chunk=std::stoul(value());else if(option=="--threads")threads=std::stoi(value());
        else if(option=="--bound-only")bound_only=true;else if(option=="--resume")resume=true;
        else throw std::runtime_error("unknown option "+option);
    }
    if(threads<1||chunk<1)throw std::runtime_error("invalid threads/chunk");
    Data data(data_path);auto expected=references(reference_path);
    std::vector<uint32_t> cases;
    if(!cases_path.empty()){
        std::ifstream in(cases_path,std::ios::binary);magic(in,"OUTCASE1");uint32_t size=read<uint32_t>(in);
        cases.resize(size);in.read(reinterpret_cast<char *>(cases.data()),size*sizeof(uint32_t));
        if(!in||uint64_t(start)+count>cases.size())throw std::runtime_error("case range exceeds input");
        cases=std::vector<uint32_t>(cases.begin()+start,cases.begin()+start+count);
    } else {if(uint64_t(start)+count>66u*66u*66u*4u)throw std::runtime_error("template range exceeds complete space");
        cases.resize(count);std::iota(cases.begin(),cases.end(),start);}
    fs::create_directories(out);
    std::ostringstream config;config<<data.signature<<"\n"<<bound_only<<" "<<start<<" "<<count<<" "<<chunk<<"\n";
    for(auto id:cases)config<<id<<",";
    if(fs::exists(out/"config.txt")){
        std::ifstream old(out/"config.txt");std::string previous((std::istreambuf_iterator<char>(old)),{});
        if(previous!=config.str()||!resume)throw std::runtime_error("output exists; exact matching --resume required");
    } else {std::ofstream config_file(out/"config.txt");config_file<<config.str();}
    omp_set_num_threads(threads);
    std::vector<std::vector<int32_t>> buffers(threads);
    std::vector<std::vector<uint32_t>> hist_buffers(threads);
    int64_t global_lower=std::numeric_limits<int64_t>::max();size_t checked=0,processed=0;
    auto began=std::chrono::steady_clock::now();
    for(size_t base=0;base<cases.size();base+=chunk){
        size_t size=std::min<size_t>(chunk,cases.size()-base);
        fs::path target=out/("chunk_"+std::to_string(base)+".bin");
        if(resume&&fs::exists(target)){
            std::ifstream prior(target,std::ios::binary);magic(prior,"OUTCHK02");uint32_t existing=read<uint32_t>(prior);
            if(existing!=size)throw std::runtime_error("checkpoint count mismatch");
            for(size_t j=0;j<size;++j){
                uint32_t id=read<uint32_t>(prior);int64_t lower=read<int64_t>(prior);
                auto minimum=read<uint32_t>(prior);auto maximum=read<uint32_t>(prior);auto shift=read<uint32_t>(prior);uint32_t pairs=read<uint32_t>(prior);
                auto elapsed_ns=read<uint64_t>(prior);
                if(id!=cases[base+j])throw std::runtime_error("checkpoint case mismatch");
                std::vector<Pair> histogram;histogram.reserve(pairs);
                for(uint32_t k=0;k<pairs;++k){uint32_t n=read<uint32_t>(prior),c=read<uint32_t>(prior);histogram.emplace_back(n,c);}
                global_lower=std::min(global_lower,lower);
                auto reference=expected.find(id);if(!bound_only&&reference!=expected.end()){
                    if(histogram!=reference->second)throw std::runtime_error("checkpoint reference mismatch");++checked;}
            }
            processed+=size;continue;
        }
        std::vector<Result> results(size);
        #pragma omp parallel for schedule(dynamic,1)
        for(size_t j=0;j<size;++j){int thread=omp_get_thread_num();results[j]=calculate(data,cases[base+j],bound_only,buffers[thread],hist_buffers[thread]);}
        std::ofstream checkpoint(target.string()+".tmp",std::ios::binary);checkpoint.write("OUTCHK02",8);write<uint32_t>(checkpoint,size);
        for(auto &result:results){
            global_lower=std::min(global_lower,result.l1_lower);
            auto reference=expected.find(result.id);
            if(!bound_only&&reference!=expected.end()){
                if(result.histogram!=reference->second)throw std::runtime_error("complete Python histogram mismatch at template "+std::to_string(result.id));++checked;}
            write(checkpoint,result.id);write(checkpoint,result.l1_lower);write(checkpoint,result.minimum);write(checkpoint,result.maximum);write(checkpoint,result.shift);write<uint32_t>(checkpoint,result.histogram.size());
            write(checkpoint,result.elapsed_ns);
            for(auto row:result.histogram){write(checkpoint,row.first);write(checkpoint,row.second);}
        }
        checkpoint.close();fs::rename(target.string()+".tmp",target);processed+=size;
        double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
        status(out,"running",processed,cases.size(),elapsed,threads,global_lower,checked,data,bound_only);
        std::cout<<processed<<"/"<<cases.size()<<" elapsed="<<elapsed<<" lower="<<global_lower<<" checked="<<checked<<std::endl;
    }
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-began).count();
    status(out,"complete",processed,cases.size(),elapsed,threads,global_lower,checked,data,bound_only);
    std::cout<<"COMPLETE elapsed="<<elapsed<<" checked="<<checked<<std::endl;
    return 0;
 } catch(const std::exception &error){std::cerr<<"ERROR: "<<error.what()<<std::endl;return 1;}
}
