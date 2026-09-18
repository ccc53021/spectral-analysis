/**
 * RECTANGLE span aggregation: dense Fourier coordinates, with no Step-2 spool.
 * Usage: step2.exe [dim=10] [True|False] [characteristic_limit]
 *                 [--stream (default) | --legacy-json (dimension <= 20)]
 * Stream stdout is exactly one JSON metadata line followed by little-endian
 * float64 coefficients. All progress and diagnostics go to stderr.
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
#include <map>
#include <filesystem>
#include <stdexcept>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <csignal>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
using namespace std;

#include "step2_generated_config.h"

const int SBOX_BITS=4;
const int STATE_BITS=64;
const int STATE_WORDS=16;
const int MASK_BITS=KEY_BITS+STATE_BITS;

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
double get_correlation_of_quasidifferential(const vector<uint64_t>& characteristic,
                                             const vector<uint64_t>& trail){
    double cor=1.0;
    for(int i=0;i<TOTAL_ROUNDS;i++){
        uint64_t round_c=ROUND_CONSTANTS[i];
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
void print_correlation_distribution(const map<double,uint64_t>& counts,
                                    const string& title){
    cerr<<endl<<title<<endl;
    cerr<<"  distinct correlations: "<<counts.size()<<endl;

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
        cerr<<"  cor="<<showpos<<scientific<<setprecision(17)<<cor<<noshowpos
            <<"  count="<<count
            <<"  -log2(abs(cor))="<<weight<<endl;
    }
    if(zero_count!=0)
        cerr<<"  cor="<<showpos<<scientific<<setprecision(17)<<0.0<<noshowpos
            <<"  count="<<zero_count
            <<"  -log2(abs(cor))=infinity"<<endl;
    cerr<<defaultfloat;
}

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
            if(p==0) throw runtime_error("Characteristic has an impossible S-box difference");
            avg+=int(-log2(p));
        }
    return avg;
}


// A mask's word 0 contains bits 0..63; joint bits are expanded keys then x.
using Mask=vector<uint64_t>;

void xor_mask(Mask& target,const Mask& source){
    if(target.size()!=source.size()) throw runtime_error("Internal mask width mismatch");
    for(size_t i=0;i<target.size();i++) target[i]^=source[i];
}
bool mask_bit(const Mask& row,int bit){ return (row.at(bit/64)>>(bit%64))&1ULL; }
bool nonzero_mask(const Mask& row){
    return any_of(row.begin(),row.end(),[](uint64_t value){ return value!=0; });
}
Mask joint_mask(const vector<uint64_t>& trail){
    Mask mask(TOTAL_ROUNDS+1,0);
    for(int round=0;round<TOTAL_ROUNDS;round++) mask[round]=trail.at(TRAIL_NUMBER*round);
    mask.back()=trail.at(0);
    return mask;
}
uint64_t permute_word(uint64_t value){
    uint64_t result=0;
    for(int target=0;target<STATE_BITS;target++)
        result|=((value>>PERMUTATION_TABLE[target])&1ULL)<<target;
    return result;
}

struct RowSpace{ vector<Mask> rows; vector<int> pivots; };
RowSpace rref(vector<Mask> rows,int width){
    size_t next=0;
    vector<int> pivots;
    for(int column=0;column<width && next<rows.size();column++){
        size_t pivot=next;
        while(pivot<rows.size() && !mask_bit(rows[pivot],column)) pivot++;
        if(pivot==rows.size()) continue;
        swap(rows[next],rows[pivot]);
        for(size_t index=0;index<rows.size();index++)
            if(index!=next && mask_bit(rows[index],column)) xor_mask(rows[index],rows[next]);
        pivots.push_back(column);
        next++;
    }
    rows.resize(next);
    return {move(rows),move(pivots)};
}
uint64_t coordinate(Mask value,const RowSpace& space){
    uint64_t result=0;
    for(size_t i=0;i<space.rows.size();i++)
        if(mask_bit(value,space.pivots[i])){
            result|=uint64_t(1)<<i;
            xor_mask(value,space.rows[i]);
        }
    if(nonzero_mask(value)) throw runtime_error("Mask is outside the Fourier row space");
    return result;
}
Mask expand_coordinate(uint64_t value,const RowSpace& space){
    Mask result((MASK_BITS+63)/64,0);
    for(size_t i=0;i<space.rows.size();i++)
        if((value>>i)&1ULL) xor_mask(result,space.rows[i]);
    return result;
}
string mask_hex(const Mask& mask){
    ostringstream out;
    out<<"0x"<<hex<<setfill('0');
    bool first=true;
    for(size_t i=mask.size();i-->0;){
        if(first && mask[i]==0) continue;
        if(first){ out<<mask[i]; first=false; }
        else out<<setw(16)<<mask[i];
    }
    if(first) out<<"0";
    return out.str();
}
string json_quote(const string& value){
    ostringstream out;
    out<<'"';
    for(unsigned char c:value){
        if(c=='"' || c=='\\') out<<'\\'<<c;
        else if(c<32) out<<"\\u"<<hex<<setw(4)<<setfill('0')<<unsigned(c);
        else out<<c;
    }
    out<<'"';
    return out.str();
}

// Parse the small flat Step-1 JSON schema, not substrings tied to whitespace.
void skip_space(const string& text,size_t& pos){
    while(pos<text.size() && isspace(static_cast<unsigned char>(text[pos]))) pos++;
}
string json_string(const string& text,size_t& pos){
    skip_space(text,pos);
    if(pos==text.size() || text[pos++]!='"') throw runtime_error("Expected JSON string");
    string result;
    while(pos<text.size()){
        char c=text[pos++];
        if(c=='"') return result;
        if(static_cast<unsigned char>(c)<32) throw runtime_error("Control character in JSON string");
        if(c=='\\'){
            if(pos==text.size()) break;
            char escaped=text[pos++];
            switch(escaped){
                case '"': case '\\': case '/': c=escaped; break;
                case 'b': c='\b'; break; case 'f': c='\f'; break;
                case 'n': c='\n'; break; case 'r': c='\r'; break; case 't': c='\t'; break;
                default: throw runtime_error("Unsupported JSON string escape in Step-1 schema");
            }
        }
        result.push_back(c);
    }
    throw runtime_error("Unterminated JSON string");
}
map<string,string> json_fields(const string& line){
    map<string,string> fields;
    size_t pos=0;
    skip_space(line,pos);
    if(pos==line.size() || line[pos++]!='{') throw runtime_error("Expected JSON object");
    for(;;){
        skip_space(line,pos);
        if(pos<line.size() && line[pos]=='}'){ pos++; break; }
        string key=json_string(line,pos);
        skip_space(line,pos);
        if(pos==line.size() || line[pos++]!=':') throw runtime_error("Expected JSON colon");
        skip_space(line,pos);
        size_t start=pos;
        int depth=0;
        while(pos<line.size()){
            if(line[pos]=='"'){ json_string(line,pos); continue; }
            if(line[pos]=='[' || line[pos]=='{') depth++;
            else if(line[pos]==']'){
                if(--depth<0) throw runtime_error("Malformed JSON array");
            }else if(line[pos]=='}'){
                if(depth==0) break;
                depth--;
            }else if(line[pos]==',' && depth==0) break;
            pos++;
        }
        size_t end=pos;
        while(end>start && isspace(static_cast<unsigned char>(line[end-1]))) end--;
        if(depth!=0 || end==start || !fields.emplace(key,line.substr(start,end-start)).second)
            throw runtime_error("Malformed or duplicate JSON field: "+key);
        skip_space(line,pos);
        if(pos==line.size()) throw runtime_error("Unterminated JSON object");
        if(line[pos]=='}'){ pos++; break; }
        if(line[pos++]!=',') throw runtime_error("Expected JSON comma");
        skip_space(line,pos);
        if(pos==line.size() || line[pos]=='}') throw runtime_error("Trailing JSON comma");
    }
    skip_space(line,pos);
    if(pos!=line.size()) throw runtime_error("Trailing data after JSON object");
    return fields;
}
const string& field(const map<string,string>& fields,const string& name){
    auto found=fields.find(name);
    if(found==fields.end()) throw runtime_error("Missing Step-1 field: "+name);
    return found->second;
}
string string_value(const string& text){
    size_t pos=0;
    string value=json_string(text,pos);
    skip_space(text,pos);
    if(pos!=text.size()) throw runtime_error("Trailing data after JSON string");
    return value;
}
int parse_int(const string& text,const string& name,int low,int high){
    if(text.empty()) throw runtime_error("Missing integer: "+name);
    size_t start=text[0]=='-' ? 1 : 0;
    if(start==text.size() || !all_of(text.begin()+start,text.end(),
            [](unsigned char c){ return c>='0' && c<='9'; }))
        throw runtime_error("Invalid integer for "+name+": "+text);
    size_t consumed=0;
    long long value;
    try{ value=stoll(text,&consumed,10); }
    catch(const exception&){ throw runtime_error("Integer out of range for "+name); }
    if(consumed!=text.size() || value<low || value>high)
        throw runtime_error(name+" must be in ["+to_string(low)+","+to_string(high)+"]");
    return static_cast<int>(value);
}
int integer_field(const map<string,string>& fields,const string& name,int low,int high){
    return parse_int(field(fields,name),name,low,high);
}
vector<uint64_t> hex_array(const string& text){
    size_t pos=0;
    skip_space(text,pos);
    if(pos==text.size() || text[pos++]!='[') throw runtime_error("Expected hexadecimal word array");
    vector<uint64_t> result;
    skip_space(text,pos);
    if(pos<text.size() && text[pos]==']') pos++;
    else for(;;){
        string word=json_string(text,pos);
        if(word.size()<3 || word.size()>18 || word[0]!='0' || (word[1]!='x' && word[1]!='X')
           || !all_of(word.begin()+2,word.end(),[](unsigned char c){ return isxdigit(c); }))
            throw runtime_error("Invalid 64-bit hexadecimal trail word: "+word);
        result.push_back(stoull(word.substr(2),nullptr,16));
        skip_space(text,pos);
        if(pos==text.size()) throw runtime_error("Unterminated word array");
        char separator=text[pos++];
        if(separator==']') break;
        if(separator!=',') throw runtime_error("Expected comma in word array");
    }
    skip_space(text,pos);
    if(pos!=text.size()) throw runtime_error("Trailing data after word array");
    return result;
}
void require_integer(const map<string,string>& fields,const string& name,int expected){
    if(integer_field(fields,name,0,numeric_limits<int>::max())!=expected)
        throw runtime_error("Step-1 metadata mismatch: "+name);
}

struct RouteData{
    vector<vector<uint64_t>> trails;
    vector<uint64_t> coordinates;
};

RouteData load_step1(const string& filename,int dimension,int route_index,
                     const string& input_x,int min_weight,int max_weight){
    ifstream input(filename);
    if(!input) throw runtime_error("Cannot open "+filename);
    RouteData result;
    bool have_meta=false;
    string line;
    size_t line_number=0;
    int next_id=0;
    while(getline(input,line)){
        line_number++;
        if(line.find_first_not_of(" \t\r\n")==string::npos) continue;
        try{
            auto fields=json_fields(line);
            string type=string_value(field(fields,"type"));
            if(type=="meta"){
                if(have_meta || next_id) throw runtime_error("Repeated or misplaced metadata");
                have_meta=true;
                if(string_value(field(fields,"cipher"))!=CIPHER_NAME
                    || string_value(field(fields,"key_model"))!="expanded")
                    throw runtime_error("Step-1 cipher/key model mismatch");
                require_integer(fields,"version",1);
                require_integer(fields,"round_key_bits",ROUND_KEY_BITS);
                require_integer(fields,"key_bits",KEY_BITS);
                require_integer(fields,"total_rounds",TOTAL_ROUNDS);
                require_integer(fields,"begin_round",BEGIN_ROUND);
                require_integer(fields,"state_bits",STATE_BITS);
                require_integer(fields,"basis_number",BASIS_NUMBER);
                require_integer(fields,"average_weight",min_weight);
                require_integer(fields,"min_weight",min_weight);
                require_integer(fields,"max_weight",max_weight);
                if(field(fields,"input_x")!=(input_x=="True" ? "true" : "false"))
                    throw runtime_error("Step-1 input_x mismatch");
                if(hex_array(field(fields,"differential_route"))!=diffs.at(route_index))
                    throw runtime_error("Step-1 characteristic mismatch");
            }else if(type=="trail"){
                if(!have_meta) throw runtime_error("Trail precedes metadata");
                int id=integer_field(fields,"trail_id",0,BASIS_NUMBER-1);
                if(id!=next_id++) throw runtime_error("Trail IDs must be unique and consecutive from zero");
                if(id>=dimension) continue;
                int weight=integer_field(fields,"weight",min_weight,max_weight-1);
                int sign=integer_field(fields,"sign",0,1);
                auto trail=hex_array(field(fields,"trail"));
                const int expected=TRAIL_NUMBER*TOTAL_ROUNDS+1;
                require_integer(fields,"trail_len",expected);
                if(trail.size()!=static_cast<size_t>(expected))
                    throw runtime_error("Incomplete basis trail");
                if(input_x=="False" && trail[0]!=0)
                    throw runtime_error("input_x=False requires a zero input mask");
                if(trail.back()!=0) throw runtime_error("Final output mask must be zero");
                for(int round=0;round<TOTAL_ROUNDS;round++)
                    if(permute_word(trail[TRAIL_NUMBER*round+1])!=trail[TRAIL_NUMBER*(round+1)])
                        throw runtime_error("Basis trail violates a ShiftRow mask equation");
                double cor=get_correlation_of_quasidifferential(diffs.at(route_index),trail);
                if(cor==0.0 || !isfinite(cor)) throw runtime_error("Selected basis has zero/nonfinite correlation");
                double recorded=ldexp(sign ? -1.0 : 1.0,-weight);
                if(recorded==0.0 || cor!=recorded)
                    throw runtime_error("Selected basis correlation does not match its weight/sign");
                result.trails.push_back(move(trail));
            }else throw runtime_error("Unknown Step-1 record type: "+type);
        }catch(const exception& error){
            throw runtime_error(filename+":"+to_string(line_number)+": "+error.what());
        }
    }
    if(input.bad()) throw runtime_error("Read failed: "+filename);
    if(next_id!=BASIS_NUMBER)
        throw runtime_error(filename+": metadata requests "+to_string(BASIS_NUMBER)
                            +" basis records, but file contains "+to_string(next_id));
    if(!have_meta || result.trails.size()!=static_cast<size_t>(dimension))
        throw runtime_error(filename+": requested "+to_string(dimension)+" basis trails, got "
                            +to_string(result.trails.size())+"; refusing a smaller span");
    if(rref(result.trails,64*(TRAIL_NUMBER*TOTAL_ROUNDS+1)).rows.size()!=result.trails.size())
        throw runtime_error(filename+": selected Step-1 trails are not linearly independent");
    return result;
}

string output_stem(const string& prefix,const string& input_x,int dimension,const string& suffix){
    return "output/record_trails_and_coefficients/"+prefix+"_r_"+to_string(TOTAL_ROUNDS)
        +"_"+D_STR+"_x_"+input_x+"_basis_number_"+to_string(BASIS_NUMBER)
        +"_weight_"+to_string(WEIGHT_RANGE)+"_dim_"+to_string(dimension)+suffix+".jsonl";
}
void base_metadata(ostream& out,const string& input_x,int dimension){
    out<<"\"cipher\":"<<json_quote(CIPHER_NAME)<<",\"key_model\":\"expanded\""
       <<",\"round_key_bits\":"<<ROUND_KEY_BITS<<",\"key_bits\":"<<KEY_BITS
       <<",\"mask_bits\":"<<MASK_BITS<<",\"total_rounds\":"<<TOTAL_ROUNDS
       <<",\"begin_round\":"<<BEGIN_ROUND<<",\"input_x\":"<<json_quote(input_x)
       <<",\"basis_number\":"<<BASIS_NUMBER<<",\"weight\":"<<WEIGHT_RANGE
       <<",\"dim\":"<<dimension;
}
void write_legacy(const string& path,const string& input_x,int dimension,
                  const vector<double>& coefficients,const vector<bool>& visited,const RowSpace& space){
    ofstream out(path);
    if(!out) throw runtime_error("Cannot open legacy Step-2 output: "+path);
    out<<'{';
    base_metadata(out,input_x,dimension);
    out<<",\"k_trails\":[";
    bool first=true;
    for(size_t index=0;index<coefficients.size();index++) if(visited[index]){
        if(!first) out<<',';
        first=false;
        auto mask=expand_coordinate(index,space);
        out<<'[';
        for(int bit=0;bit<MASK_BITS;bit++){
            if(bit) out<<',';
            out<<static_cast<int>(mask_bit(mask,bit));
        }
        out<<']';
    }
    out<<"],\"k_coefficients\":["<<scientific<<setprecision(17);
    first=true;
    for(size_t index=0;index<coefficients.size();index++) if(visited[index]){
        if(!first) out<<',';
        first=false;
        out<<coefficients[index];
    }
    out<<"]}\n";
    out.close();
    if(!out) throw runtime_error("Writing legacy Step-2 output failed: "+path);
}
void write_stream(const string& input_x,int dimension,int characteristic_count,
                  const RowSpace& space,const vector<double>& coefficients,
                  uint64_t aggregated_count,uint64_t nonzero_count){
    static_assert(sizeof(double)==8 && numeric_limits<double>::is_iec559,
                  "The stream requires IEEE-754 float64");
#ifdef _WIN32
    if(_setmode(_fileno(stdout),_O_BINARY)==-1) throw runtime_error("Cannot make stdout binary");
#endif
    cout<<"{\"type\":\"rectangle_fourier\",\"version\":1,";
    base_metadata(cout,input_x,dimension);
    cout<<",\"d_str\":"<<json_quote(D_STR)<<",\"characteristic_count\":"<<characteristic_count
        <<",\"dim_k\":"<<space.rows.size()<<",\"entry_count\":"<<coefficients.size()
        <<",\"coefficient_dtype\":\"<f8\",\"basis_masks_hex\":[";
    for(size_t index=0;index<space.rows.size();index++){
        if(index) cout<<',';
        cout<<json_quote(mask_hex(space.rows[index]));
    }
    cout<<"],\"pivots\":[";
    for(size_t index=0;index<space.pivots.size();index++){
        if(index) cout<<',';
        cout<<space.pivots[index];
    }
    cout<<"],\"aggregated_mask_count\":"<<aggregated_count
        <<",\"nonzero_coefficient_count\":"<<nonzero_count<<"}\n";
    if(!cout) throw runtime_error("Writing stream metadata to stdout failed");
    const uint16_t endian_probe=1;
    const bool little_endian=*reinterpret_cast<const unsigned char*>(&endian_probe)==1;
    const size_t block_values=131072;
    vector<char> swapped(little_endian ? 0 : block_values*sizeof(double));
    for(size_t start=0;start<coefficients.size();start+=block_values){
        size_t count=min(block_values,coefficients.size()-start);
        const char* data=reinterpret_cast<const char*>(coefficients.data()+start);
        if(!little_endian){
            for(size_t i=0;i<count;i++)
                reverse_copy(data+i*8,data+(i+1)*8,swapped.data()+i*8);
            data=swapped.data();
        }
        cout.write(data,static_cast<streamsize>(count*sizeof(double)));
        if(!cout) throw runtime_error("Writing Fourier coefficient stream failed (consumer closed?)");
    }
    cout.flush();
    if(!cout) throw runtime_error("Flushing Fourier coefficient stream failed");
}

int run(int argc,char** argv){
    bool legacy=false,mode_seen=false;
    vector<string> positional;
    for(int i=1;i<argc;i++){
        string argument=argv[i];
        if(argument=="--stream" || argument=="--legacy-json"){
            if(mode_seen) throw runtime_error("Specify exactly one output mode");
            mode_seen=true;
            legacy=argument=="--legacy-json";
        }else if(argument.rfind("--",0)==0) throw runtime_error("Unknown option: "+argument);
        else positional.push_back(argument);
    }
    if(positional.size()>3) throw runtime_error("Usage: step2.exe dim True|False [characteristic_limit] [--stream|--legacy-json]");
    int dimension=positional.empty() ? 10 : parse_int(positional[0],"dim",1,30);
    string input_x=positional.size()>1 ? positional[1] : "False";
    if(input_x!="True" && input_x!="False") throw runtime_error("input_x must be exactly True or False");
    if(BASIS_NUMBER<1 || dimension>BASIS_NUMBER || WEIGHT_RANGE<=0
       || TOTAL_ROUNDS<1 || TOTAL_ROUNDS>1000 || BEGIN_ROUND<0
       || TRAIL_NUMBER!=2 || (ROUTE_NUMBER!=2 && ROUTE_NUMBER!=3)
       || ROUND_KEY_BITS!=64 || KEY_BITS!=TOTAL_ROUNDS*64
       || sizeof(ROUND_CONSTANTS)/sizeof(ROUND_CONSTANTS[0])!=static_cast<size_t>(TOTAL_ROUNDS))
        throw runtime_error("Invalid generated configuration or dimension exceeds basis_number");
    bool permutation_seen[STATE_BITS]={};
    for(int index:PERMUTATION_TABLE){
        if(index<0 || index>=STATE_BITS || permutation_seen[index])
            throw runtime_error("Invalid generated ShiftRow permutation");
        permutation_seen[index]=true;
    }
    bool sbox_seen[16]={};
    for(int value:SBOX){
        if(value<0 || value>=16 || sbox_seen[value]) throw runtime_error("Invalid generated S-box");
        sbox_seen[value]=true;
    }
    if(diffs.empty() || diffs.size()>static_cast<size_t>(numeric_limits<int>::max()))
        throw runtime_error("Invalid characteristic count");
    int characteristic_count=positional.size()>2
        ? parse_int(positional[2],"characteristic_limit",1,static_cast<int>(diffs.size()))
        : static_cast<int>(diffs.size());
    int max_dimension=26;
    if(const char* configured=getenv("MAX_FWHT_DIM"))
        max_dimension=parse_int(configured,"MAX_FWHT_DIM",0,30);
    if(legacy && dimension>20) throw runtime_error("--legacy-json is restricted to dimension <= 20");
    string suffix=characteristic_count<static_cast<int>(diffs.size())
        ? "_limit_"+to_string(characteristic_count) : "";
    vector<RouteData> routes;
    vector<Mask> all_basis_masks;
    auto started=chrono::steady_clock::now();
    // Read/validate every selected route before any exponential allocation/work.
    for(int index=0;index<characteristic_count;index++){
        if(diffs[index].size()!=static_cast<size_t>(ROUTE_NUMBER*TOTAL_ROUNDS+1))
            throw runtime_error("Characteristic "+to_string(index)+" has the wrong length");
        for(int round=0;round<TOTAL_ROUNDS;round++){
            uint64_t next=permute_word(diffs[index][ROUTE_NUMBER*round+1]);
            if(next!=diffs[index][ROUTE_NUMBER*(round+1)]
               || (ROUTE_NUMBER==3 && next!=diffs[index][ROUTE_NUMBER*round+2]))
                throw runtime_error("Characteristic "+to_string(index)+" violates ShiftRow");
        }
        double average_weight=get_average_w(diffs[index]);
        int minimum=static_cast<int>(average_weight);
        if(WEIGHT_RANGE>numeric_limits<int>::max()-minimum)
            throw runtime_error("Step-1 weight range overflows integer filenames");
        int maximum=minimum+WEIGHT_RANGE;
        string filename="output/record_quasidifferentials/step1_quasidc_search_r_"
            +to_string(TOTAL_ROUNDS)+"_"+D_STR+"_c_"+to_string(index)+"_x_"+input_x
            +"_basis_number_"+to_string(BASIS_NUMBER)+"_w_"+to_string(minimum)
            +"_to_"+to_string(maximum)+".jsonl";
        cerr<<"DC["<<index<<"]: avg_w="<<average_weight<<"  file="<<filename<<endl;
        auto route=load_step1(filename,dimension,index,input_x,minimum,maximum);
        for(const auto& trail:route.trails) all_basis_masks.push_back(joint_mask(trail));
        routes.push_back(move(route));
    }
    auto space=rref(move(all_basis_masks),MASK_BITS);
    size_t rank=space.rows.size();
    if(rank>static_cast<size_t>(max_dimension) || rank>30)
        throw runtime_error("Fourier rank "+to_string(rank)+" exceeds MAX_FWHT_DIM="
                            +to_string(max_dimension)+" (hard maximum 30)");
    if(legacy && rank>20) throw runtime_error("--legacy-json is restricted to Fourier rank <= 20");
    if(rank>=numeric_limits<size_t>::digits) throw runtime_error("Fourier size overflows this platform");
    uint64_t entries=uint64_t(1)<<rank;
    for(auto& route:routes)
        for(const auto& trail:route.trails) route.coordinates.push_back(coordinate(joint_mask(trail),space));
    cerr<<"Fourier coordinate rank: "<<rank<<"; "<<entries<<" float64 coefficients ("
        <<entries*sizeof(double)<<" bytes), plus visited bits"<<endl;
    vector<double> coefficients(static_cast<size_t>(entries),0.0);
    vector<bool> visited(static_cast<size_t>(entries),false);
    uint64_t aggregated_count=0;
    map<double,uint64_t> all_correlation_counts;
    for(int index=0;index<characteristic_count;index++){
        auto& route=routes[index];
        uint64_t valid=0;
        map<double,uint64_t> correlation_counts;
        uint64_t combinations=uint64_t(1)<<dimension;
        vector<uint64_t> trail(route.trails[0].size(),0);
        uint64_t previous=0,mask_coordinate=0;
        // The original binary combo order is unchanged. Toggle only changed bits
        // instead of rebuilding the entire trail and its 960-bit mask each time.
        for(uint64_t combo=0;combo<combinations;combo++){
            uint64_t changed=combo^previous;
            for(int bit=0;changed;bit++,changed>>=1) if(changed&1ULL){
                xor_mask(trail,route.trails[bit]);
                mask_coordinate^=route.coordinates[bit];
            }
            previous=combo;
            double cor=get_correlation_of_quasidifferential(diffs[index],trail);
            if(!isfinite(cor)) throw runtime_error("Nonfinite span correlation");
            correlation_counts[cor]++;
            all_correlation_counts[cor]++;
            if(cor==0.0) continue;
            valid++;
            if(!visited[mask_coordinate]){
                visited[mask_coordinate]=true;
                aggregated_count++;
            }
            coefficients[mask_coordinate]+=cor;
        }
        cerr<<"  DC["<<index<<"]: spanned "<<combinations<<" trails, "<<valid
            <<" valid trails, "<<aggregated_count<<" key masks so far, "
            <<chrono::duration<double>(chrono::steady_clock::now()-started).count()<<"s"<<endl;
        print_correlation_distribution(correlation_counts,
            "  DC["+to_string(index)+"] quasidifferential correlation distribution:");
    }
    uint64_t nonzero_count=0;
    for(double value:coefficients){
        if(!isfinite(value)) throw runtime_error("Nonfinite aggregated coefficient");
        if(value!=0.0) nonzero_count++;
    }
    cerr<<"\nAfter all DCs: "<<aggregated_count<<" aggregated key masks"<<endl;
    print_correlation_distribution(all_correlation_counts,"All quasidifferential correlation distribution:");
    if(visited[0]){
        if(coefficients[0]>0.0) cerr<<"C_[uk=0, ux=0] = 2^"<<log2(coefficients[0])<<endl;
        else cerr<<"C_[uk=0, ux=0] = "<<setprecision(17)<<coefficients[0]<<endl;
    }else cerr<<"WARNING: C_[uk=0, ux=0] not found!"<<endl;
    if(legacy){
        filesystem::create_directories("output/record_trails_and_coefficients");
        string path=output_stem("step2_save_trails_and_coefficients",input_x,dimension,suffix);
        write_legacy(path,input_x,dimension,coefficients,visited,space);
        cerr<<"Saved: "<<path<<endl;
    }else write_stream(input_x,dimension,characteristic_count,space,coefficients,
                       aggregated_count,nonzero_count);
    cerr<<"Time: "<<chrono::duration<double>(chrono::steady_clock::now()-started).count()<<"s"<<endl;
    return 0;
}
int main(int argc,char** argv){
#ifdef SIGPIPE
    signal(SIGPIPE,SIG_IGN); // Report a broken consumer pipe as a clear error.
#endif
    try{ return run(argc,argv); }
    catch(const bad_alloc&){ cerr<<"ERROR: insufficient memory for the requested Fourier span"<<endl; }
    catch(const exception& error){ cerr<<"ERROR: "<<error.what()<<endl; }
    return 1;
}
