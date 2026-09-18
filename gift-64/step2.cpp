/**
 * step2.cpp: v3 span enumeration, coefficient aggregation, and serialization.
 * Build: x86_64-w64-mingw32-g++ -O3 -std=c++17 step2.cpp -o step2.exe
 * Usage: step2.exe [dim=10] [input_x=False] [characteristic_limit=all]
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <map>
#include <filesystem>
using namespace std;

#include "step2_generated_config.h"

// ============ Constants matching cipher_config.py ============
const int SBOX[16]={0x1,0xa,0x4,0xc,0x6,0xf,0x3,0x9,0x2,0xd,0xb,0x7,0x5,0x0,0x8,0xe};
const int SBOX_BITS=4;
const int STATE_BITS=64;
const int STATE_WORDS=16;
const int KEY_BITS=128;
const int MASK_BITS=KEY_BITS+STATE_BITS;
const int ROUND_CONSTANTS[33]={
    0x00,0x01,0x03,0x07,0x0F,0x1F,0x3E,0x3D,0x3B,0x37,0x2F,0x1E,0x3C,0x39,0x33,0x27,
    0x0E,0x1D,0x3A,0x35,0x2B,0x16,0x2C,0x18,0x30,0x21,0x02,0x05,0x0B,0x17,0x2E,0x1C,0x38
};
const int PERM_K[128]={
    32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,
    64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,
    80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,
    96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,
    112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,
    12,13,14,15,0,1,2,3,4,5,6,7,8,9,10,11,
    18,19,20,21,22,23,24,25,26,27,28,29,30,31,16,17
};

// Precompute master_round_key_table.
vector<vector<int>> master_round_key_table;
void precompute_key_table(){
    int total_r=TOTAL_ROUNDS+BEGIN_ROUND;
    master_round_key_table.resize(total_r+1,vector<int>(128));
    for(int i=0;i<128;i++) master_round_key_table[0][i]=i;
    for(int r=0;r<total_r;r++){
        auto& prev=master_round_key_table[r];
        auto& curr=master_round_key_table[r+1];
        for(int i=0;i<128;i++) curr[i]=prev[PERM_K[i]];
    }
}

// ============ QDTM correlation functions matching utils.py ============
int vector_inner_product(int u,int x,int n){
    int left=0;
    for(int i=0;i<n;i++) left+=((u>>i)&1)*((x>>i)&1);
    return left%2;
}
double get_correlation_by_abuv_and_c(int a,int b,int u,int v,int c){
    double cor=0;
    for(int x=0;x<(1<<SBOX_BITS);x++){
        int x1=x^a, y=SBOX[x^c], y1=SBOX[x1^c];
        if((y^y1)==b){
            int dot=vector_inner_product(u,x,SBOX_BITS)^vector_inner_product(v,y,SBOX_BITS);
            cor+=(dot==0)?1:-1;
        }
    }
    return cor/(1<<SBOX_BITS);
}
uint64_t get_constant(const vector<int>& l_rc){
    int l_ii[7]={3,7,11,15,19,23,63};
    uint64_t result=0;
    for(int i=0;i<7;i++) if(l_rc[i]) result|=(1ull<<l_ii[i]);
    return result;
}
double get_correlation_of_quasidifferential(const vector<uint64_t>& characteristic,
                                             const vector<uint64_t>& trail){
    double cor=1.0;
    for(int i=0;i<TOTAL_ROUNDS;i++){
        int rc=ROUND_CONSTANTS[i];
        vector<int> l_rc={(rc>>0)&1,(rc>>1)&1,(rc>>2)&1,(rc>>3)&1,(rc>>4)&1,(rc>>5)&1,1};
        uint64_t round_c=(i!=0)?get_constant(l_rc):0;
        for(int j=0;j<STATE_BITS;j+=SBOX_BITS){
            int da=(characteristic[ROUTE_NUMBER*i]>>j)&0xf;
            int db=(characteristic[ROUTE_NUMBER*i+1]>>j)&0xf;
            int ma=(trail[TRAIL_NUMBER*i]>>j)&0xf;
            int mb=(trail[TRAIL_NUMBER*i+1]>>j)&0xf;
            int cc=(round_c>>j)&0xf;
            cor*=get_correlation_by_abuv_and_c(da,db,ma,mb,cc);
            if(cor==0.0) return cor;
        }
    }
    return cor;
}
vector<uint64_t> get_a_key_trail(const vector<uint64_t>& q_trail){
    vector<uint64_t> kt(TOTAL_ROUNDS);
    for(int i=0;i<TOTAL_ROUNDS;i++) kt[i]=q_trail[TRAIL_NUMBER*(i+1)];
    return kt;
}
vector<int> get_master_key_mask(const vector<uint64_t>& k_trail){
    vector<int> k_expr;
    for(int r=0;r<(int)k_trail.size();r++){
        uint64_t cur=k_trail[r];
        for(int i=0;i<STATE_BITS/SBOX_BITS;i++){
            if((cur>>(SBOX_BITS*i))&1){
                int mk=master_round_key_table[r+BEGIN_ROUND][i];
                auto it=find(k_expr.begin(),k_expr.end(),mk);
                if(it==k_expr.end()) k_expr.push_back(mk); else k_expr.erase(it);
            }
            if((cur>>(SBOX_BITS*i+1))&1){
                int mk=master_round_key_table[r+BEGIN_ROUND][i+16];
                auto it=find(k_expr.begin(),k_expr.end(),mk);
                if(it==k_expr.end()) k_expr.push_back(mk); else k_expr.erase(it);
            }
        }
    }
    sort(k_expr.begin(),k_expr.end());
    vector<int> km(KEY_BITS,0);
    for(int kk:k_expr) km[kk]=1;
    return km;
}
pair<double,vector<int>> get_cor_and_key_by_one_quasidifferential(
        const vector<uint64_t>& characteristic,const vector<uint64_t>& trail){
    double cor=get_correlation_of_quasidifferential(characteristic,trail);
    auto joint_mask=get_master_key_mask(get_a_key_trail(trail));
    uint64_t input_x=trail[0];
    for(int i=0;i<STATE_BITS;i++)
        joint_mask.push_back((input_x>>i)&1ull);
    return {cor,joint_mask};
}

void print_correlation_distribution(const map<double,uint64_t>& counts,
                                    const string& title){
    cout<<endl<<title<<endl;
    cout<<"  distinct correlations: "<<counts.size()<<endl;

    vector<pair<double,uint64_t>> nonzero_counts;
    uint64_t zero_count=0;
    for(const auto& [cor,count]:counts){
        if(cor==0.0) zero_count+=count;
        else nonzero_counts.push_back({cor,count});
    }

    sort(nonzero_counts.begin(),nonzero_counts.end(),
         [](const auto& lhs,const auto& rhs){
             long long lhs_weight=llround(-log2(abs(lhs.first)));
             long long rhs_weight=llround(-log2(abs(rhs.first)));
             if(lhs_weight!=rhs_weight) return lhs_weight<rhs_weight;
             return lhs.first<rhs.first;
         });

    for(const auto& [cor,count]:nonzero_counts){
        long long weight=llround(-log2(abs(cor)));
        cout<<"  cor="<<showpos<<scientific<<setprecision(17)<<cor<<noshowpos
            <<"  count="<<count
            <<"  -log2(abs(cor))="<<weight<<endl;
    }
    if(zero_count!=0)
        cout<<"  cor="<<showpos<<scientific<<setprecision(17)<<0.0<<noshowpos
            <<"  count="<<zero_count
            <<"  -log2(abs(cor))=infinity"<<endl;
    cout<<defaultfloat;
}

bool is_analysis_characteristic(int idx){
    return find(ANALYSIS_CHARACTERISTIC_INDICES.begin(),
                ANALYSIS_CHARACTERISTIC_INDICES.end(),idx)
           !=ANALYSIS_CHARACTERISTIC_INDICES.end();
}

string packed_mask_to_hex(const string& packed_mask){
    ostringstream out;
    out<<"0x"<<hex<<setfill('0');
    for(int i=(int)packed_mask.size()-1;i>=0;i--)
        out<<setw(2)<<static_cast<unsigned int>(
            static_cast<unsigned char>(packed_mask[i]));
    return out.str();
}

void write_characteristic_spectrum(
        ofstream& out,
        int characteristic_index,
        double average_weight,
        int span_dimension,
        const unordered_map<string,double>& coefficients){
    vector<pair<string,double>> nonzero;
    nonzero.reserve(coefficients.size());
    for(const auto& item:coefficients)
        if(item.second!=0.0) nonzero.push_back(item);
    sort(nonzero.begin(),nonzero.end(),
         [](const auto& lhs,const auto& rhs){ return lhs.first<rhs.first; });

    out<<"{\"type\":\"characteristic_spectrum\""
       <<",\"characteristic_index\":"<<characteristic_index
       <<",\"average_weight\":"<<average_weight
       <<",\"span_dimension\":"<<span_dimension
       <<",\"nonzero_mask_count\":"<<nonzero.size()
       <<",\"masks_hex\":[";
    for(int i=0;i<(int)nonzero.size();i++){
        if(i>0) out<<",";
        out<<"\""<<packed_mask_to_hex(nonzero[i].first)<<"\"";
    }
    out<<"],\"coefficients\":["<<scientific<<setprecision(17);
    for(int i=0;i<(int)nonzero.size();i++){
        if(i>0) out<<",";
        out<<nonzero[i].second;
    }
    out<<"]}\n"<<defaultfloat;
    out.flush();
}

// ============ Gray code span ============
void span_trails(const vector<vector<uint64_t>>& basis_trails,
                 vector<vector<uint64_t>>& out_trails){
    int d=(int)basis_trails.size(), N=1<<d;
    int trail_len=(int)basis_trails[0].size();
    out_trails.resize(N);
    out_trails[0]=vector<uint64_t>(trail_len,0);
    for(int j=0;j<d;j++){
        int step=1<<j; auto& bj=basis_trails[j];
        for(int i=0;i<step;i++){
            int idx=step+i;
            auto& prev=out_trails[i];
            auto& cur=out_trails[idx];
            cur.resize(trail_len);
            for(int k=0;k<trail_len;k++) cur[k]=prev[k]^bj[k];
        }
    }
}

// ============ DDT and average weight ============
double get_average_w(const vector<uint64_t>& characteristic){
    double ddt[16][16]={};
    for(int a=0;a<16;a++)
        for(int x=0;x<16;x++)
            ddt[SBOX[x]^SBOX[x^a]][a]+=1.0/16.0;
    double avg=0;
    for(int i=0;i<TOTAL_ROUNDS;i++)
        for(int j=0;j<STATE_BITS;j+=SBOX_BITS){
            int da=(characteristic[ROUTE_NUMBER*i]>>j)&0xf;
            int db=(characteristic[ROUTE_NUMBER*i+1]>>j)&0xf;
            double p=ddt[db][da];
            if(p==0){ cerr<<"Invalid diff a="<<da<<" b="<<db<<endl; exit(1); }
            avg+=int(-log2(p));
        }
    return avg;
}

// ============ JSONL parsing ============
struct TrailRecord{ int trail_id,weight,sign; vector<uint64_t> trail; };
struct Step1Data{ vector<TrailRecord> trails; };
Step1Data load_step1(const string& fname,int dim){
    ifstream f(fname);
    if(!f){ cerr<<"Cannot open "<<fname<<endl; exit(1); }
    Step1Data res;
    string line;
    while(getline(f,line)){
        if(line.empty()||line.find("\"trail\"")==string::npos) continue;
        int tid=-1,w=0,s=0;
        auto tid_pos=line.find("\"trail_id\":");
        if(tid_pos!=string::npos) tid=stoi(line.substr(tid_pos+12));
        if(tid>=dim) break;
        auto w_pos=line.find("\"weight\":");
        if(w_pos!=string::npos) w=stoi(line.substr(w_pos+9));
        auto s_pos=line.find("\"sign\":");
        if(s_pos!=string::npos) s=stoi(line.substr(s_pos+7));
        auto arr_start=line.find("\"trail\": [");
        if(arr_start==string::npos) continue;
        size_t p=arr_start+10;
        vector<uint64_t> trail;
        while(p<line.size()){
            if(line.substr(p,2)=="0x"||line.substr(p,2)=="0X"){
                trail.push_back(strtoull(line.c_str()+p+2,nullptr,16));
                p+=18;
            }else if(line[p]==']') break;
            else p++;
        }
        res.trails.push_back({tid,w,s,trail});
    }
    return res;
}

// ============ Main program ============
int main(int argc,char** argv){
    precompute_key_table();

    const int dim_truncated=(argc>1)?atoi(argv[1]):10;
    const string input_x=(argc>2)?argv[2]:"False";
    const int requested_characteristic_limit=(argc>3)?atoi(argv[3]):-1;
    const int basis_number=BASIS_NUMBER;
    const int weight_range=WEIGHT_RANGE;
    const int characteristic_count=(requested_characteristic_limit>0)
        ?min(requested_characteristic_limit,(int)diffs.size())
        :(int)diffs.size();
    const string limit_suffix=(characteristic_count<(int)diffs.size())
        ?"_limit_"+to_string(characteristic_count)
        :"";

    filesystem::create_directories("output/record_trails_and_coefficients");
    string spectrum_path=
        "output/record_trails_and_coefficients/step2_characteristic_spectra_r_"
        +to_string(TOTAL_ROUNDS)+"_"+D_STR+"_x_"+input_x
        +"_basis_number_"+to_string(basis_number)
        +"_weight_"+to_string(weight_range)
        +"_dim_"+to_string(dim_truncated)+limit_suffix+".jsonl";
    ofstream spectrum_out(spectrum_path);
    if(!spectrum_out){
        cerr<<"Cannot open "<<spectrum_path<<endl;
        return 1;
    }
    spectrum_out<<"{\"type\":\"meta\",\"input_x\":\""<<input_x
        <<"\",\"basis_number\":"<<basis_number
        <<",\"weight\":"<<weight_range
        <<",\"dim\":"<<dim_truncated
        <<",\"mask_bits\":"<<MASK_BITS
        <<",\"characteristic_count\":"<<characteristic_count
        <<",\"analysis_characteristic_indices\":[";
    bool first_analysis_index=true;
    for(int idx:ANALYSIS_CHARACTERISTIC_INDICES){
        if(idx<0||idx>=characteristic_count) continue;
        if(!first_analysis_index) spectrum_out<<",";
        spectrum_out<<idx;
        first_analysis_index=false;
    }
    spectrum_out<<"]}\n";

    // Aggregate across all characteristics, matching the Python implementation.
    vector<vector<int>> k_trails;
    vector<double> k_coefficients;
    map<double,uint64_t> all_correlation_counts;
    unordered_map<string,int> km_to_idx;
    auto km_to_str=[](const vector<int>& mask){
        string s(MASK_BITS/8,'\0');
        for(int i=0;i<MASK_BITS;i++)
            if(mask[i]) s[i/8]|=static_cast<char>(1u<<(i%8));
        return s;
    };

    auto t0=chrono::steady_clock::now();

    for(int idx=0; idx<characteristic_count; idx++){
        auto& characteristic=diffs[idx];
        int count_trail=0;
        int count_valid=0;
        map<double,uint64_t> dc_correlation_counts;
        if((int)characteristic.size()!=ROUTE_NUMBER*TOTAL_ROUNDS+1){
            cerr<<"ERROR: characteristic["<<idx<<"] length mismatch"<<endl; return 1;
        }

        double avg_w=get_average_w(characteristic);
        int min_w=(int)avg_w, max_w=min_w+weight_range;

        string fname="output/record_quasidifferentials/step1_quasidc_search_r_"
            +to_string(TOTAL_ROUNDS)+"_"+D_STR+"_c_"+to_string(idx)
            +"_x_"+input_x
            +"_basis_number_"+to_string(basis_number)
            +"_w_"+to_string(min_w)+"_to_"+to_string(max_w)+".jsonl";

        cout<<"DC["<<idx<<"]: avg_w="<<avg_w<<"  file="<<fname<<endl;

        auto data=load_step1(fname,dim_truncated);
        int d=(int)data.trails.size();
        if(d==0){ cout<<"  DC["<<idx<<"]: no trails, skip"<<endl; continue; }
        cout<<"  Loaded "<<d<<" trails  span=2^"<<d<<"="<<(1ull<<d)<<endl;

        sort(data.trails.begin(),data.trails.end(),
             [](auto& a,auto& b){return a.trail_id<b.trail_id;});
        vector<vector<uint64_t>> basis_trails;
        const int expected_trail_len=TRAIL_NUMBER*TOTAL_ROUNDS+1;
        for(auto& t:data.trails){
            if((int)t.trail.size()!=expected_trail_len){
                cerr<<"ERROR: DC["<<idx<<"] trail_id="<<t.trail_id
                    <<" length is "<<t.trail.size()
                    <<", expected "<<expected_trail_len<<endl;
                return 1;
            }
            basis_trails.push_back(t.trail);
        }

        const bool track_characteristic=is_analysis_characteristic(idx);
        unordered_map<string,double> characteristic_coefficients;
        if(track_characteristic)
            characteristic_coefficients.reserve(1ull<<min(d,20));

        // Generate the low-memory span incrementally without caching.
        int N=1<<d;
        vector<uint64_t> cur_trail(basis_trails[0].size(), 0);
        for(int combo=0; combo<N; combo++){
            // XOR basis vectors selected by the combination index.
            fill(cur_trail.begin(), cur_trail.end(), 0);
            for(int j=0;j<d;j++) if((combo>>j)&1)
                for(int k=0;k<(int)cur_trail.size();k++) cur_trail[k]^=basis_trails[j][k];
            auto [cor,km]=get_cor_and_key_by_one_quasidifferential(characteristic,cur_trail);
            if((int)km.size()!=MASK_BITS){
                cerr<<"ERROR: joint mask length is "<<km.size()
                    <<", expected "<<MASK_BITS<<endl;
                return 1;
            }
            dc_correlation_counts[cor]++;
            all_correlation_counts[cor]++;
            if(cor==0){ count_trail++; continue; }
            string ks=km_to_str(km);
            if(track_characteristic)
                characteristic_coefficients[ks]+=cor;
            auto it=km_to_idx.find(ks);
            if(it==km_to_idx.end()){
                km_to_idx[ks]=(int)k_trails.size();
                k_trails.push_back(km);
                k_coefficients.push_back(cor);
            }else{
                k_coefficients[it->second]+=cor;
            }
            count_valid++;
            count_trail++;
        }
        cout<<"  DC["<<idx<<"]: spanned "<<count_trail<<" trails, "
            <<count_valid<<" valid trails, "
            <<k_trails.size()<<" key masks so far, "
            <<chrono::duration<double>(chrono::steady_clock::now()-t0).count()<<"s"<<endl;
        print_correlation_distribution(
            dc_correlation_counts,
            "  DC["+to_string(idx)+"] quasidifferential correlation distribution:"
        );
        if(track_characteristic){
            write_characteristic_spectrum(
                spectrum_out,idx,avg_w,d,characteristic_coefficients);
            cout<<"  DC["<<idx<<"] spectrum: "
                <<characteristic_coefficients.size()<<" masks saved"<<endl;
        }
    }

    spectrum_out.close();
    cout<<"Saved characteristic spectra: "<<spectrum_path<<endl;

    cout<<endl<<"After all DCs: "<<k_trails.size()<<" aggregated key masks"<<endl;
    print_correlation_distribution(
        all_correlation_counts,
        "All quasidifferential correlation distribution:"
    );

    // ADP
    vector<int> zero_mask(MASK_BITS,0);
    auto zit=km_to_idx.find(km_to_str(zero_mask));
    if(zit!=km_to_idx.end())
        cout<<"C_[uk=0, ux=0] = 2^"<<log2(k_coefficients[zit->second])<<endl;
    else cout<<"WARNING: C_[uk=0, ux=0] not found!"<<endl;

    // Save.
    string save_path="output/record_trails_and_coefficients/step2_save_trails_and_coefficients_r_"
        +to_string(TOTAL_ROUNDS)+"_"+D_STR+"_x_"+input_x
        +"_basis_number_"+to_string(basis_number)+"_weight_"+to_string(weight_range)
        +"_dim_"+to_string(dim_truncated)+limit_suffix+".jsonl";
    ofstream fout(save_path);
    fout<<"{\"input_x\":\""<<input_x<<"\",\"basis_number\":"<<basis_number
        <<",\"weight\":"<<weight_range<<",\"dim\":"<<dim_truncated<<",\"k_trails\":[";
    for(int i=0;i<(int)k_trails.size();i++){
        if(i>0) fout<<",";
        fout<<"[";
        for(int j=0;j<MASK_BITS;j++){
            if(j>0) fout<<",";
            fout<<k_trails[i][j];
        }
        fout<<"]";
    }
    fout<<"],\"k_coefficients\":[";
    fout<<scientific<<setprecision(17);
    for(int i=0;i<(int)k_coefficients.size();i++){
        if(i>0) fout<<",";
        fout<<k_coefficients[i];
    }
    // fout<<"],\"k_indexes\":[";
    // for(int i=0;i<(int)k_indexes.size();i++){
    //     if(i>0) fout<<","; fout<<"[";
    //     for(int j=0;j<(int)k_indexes[i].size();j++){
    //         if(j>0) fout<<","; fout<<k_indexes[i][j];
    //     }
    //     fout<<"]";
    // }
    fout<<"]}\n"; fout.close();

    auto dt=chrono::steady_clock::now()-t0;
    cout<<"Saved: "<<save_path<<endl;
    cout<<"Time: "<<chrono::duration<double>(dt).count()<<"s"<<endl;
    return 0;
}
