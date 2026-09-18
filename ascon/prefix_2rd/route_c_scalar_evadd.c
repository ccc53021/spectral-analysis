/* Power-of-two normalized edge-valued ADD for the full four-round RB function.
 * All label assignments are represented, no pruning/epsilon. Scalar factors
 * are float64; normalization by signed powers of two is exact binary scaling.
 */
#define __USE_MINGW_ANSI_STDIO 1
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <setjmp.h>
#include <math.h>
#if defined(_WIN32) && defined(ROUTE_C_REPORT_MEMORY)
#include <windows.h>
#include <psapi.h>
#endif
#ifndef ROUTE_C_LABEL_BITS
#define ROUTE_C_LABEL_BITS 32
#endif
#if ROUTE_C_LABEL_BITS < 1 || ROUTE_C_LABEL_BITS > 63
#error ROUTE_C_LABEL_BITS must leave one bit for the affine constant
#endif
#define N 32768u
#define C 64u
#ifndef NODE_CAP
#define NODE_CAP 2000000u
#endif
#ifndef UNIQUE_SIZE
#define UNIQUE_SIZE (1u<<22)
#endif
#define CACHE_SIZE (1u<<20)
#define UNSET UINT32_MAX
typedef struct{uint32_t node;double scale;} Ref;
typedef struct{Ref lo,hi;uint8_t level;} Node;
typedef struct{uint64_t hash;Ref a,b,result;int multiply;} Cache;
#define ZERO ((Ref){0,0.})
#define ONE ((Ref){1,1.})
static const int S[32]={4,11,31,20,26,21,9,2,27,5,8,18,29,3,6,28,30,19,7,14,0,13,17,24,16,12,1,25,22,10,15,23};
static const int ROT[5][2]={{19,28},{61,39},{1,6},{10,17},{7,41}};
static Node *nodes;
static uint32_t *unique_nodes;
static uint64_t *solutions;
static double *solution_values;
static Cache *cache;
static Ref *memo,monomials[ROUTE_C_LABEL_BITS][ROUTE_C_LABEL_BITS];
static uint8_t *bounds;
static uint64_t *count_cache,linear_forms[C][5][15],phase_forms[C][15];
static uint16_t *phase_quadratic;
static int lat[32][32],lat_count[32];
static double lat_values[32][32];
static uint32_t lat_bits[32],projections[64],dimensions,node_count=2,solution_count=0,enum_cap=65536;
static uint64_t operations=0,visited=0,cache_hits=0;
static double time_limit=60;
static clock_t started;
static jmp_buf failure;
static const char *reason="unknown";
static double elapsed(void){return (double)(clock()-started)/CLOCKS_PER_SEC;}
static void memory_fields(void){
#if defined(_WIN32) && defined(ROUTE_C_REPORT_MEMORY)
    PROCESS_MEMORY_COUNTERS pm;memset(&pm,0,sizeof(pm));pm.cb=sizeof(pm);
    if(GetProcessMemoryInfo(GetCurrentProcess(),&pm,sizeof(pm)))
        printf(",\"peak_working_set_bytes\":%"PRIu64",\"peak_commit_bytes\":%"PRIu64,
               (uint64_t)pm.PeakWorkingSetSize,(uint64_t)pm.PeakPagefileUsage);
#endif
    printf(",\"label_capacity\":%u,\"label_storage_bits\":64,\"pointer_bits\":%u",ROUTE_C_LABEL_BITS,(unsigned)(8*sizeof(void*)));
}
static void die(const char *msg){reason=msg;longjmp(failure,1);}
static void budget(void){if((++operations&16383)==0 && elapsed()>time_limit)die("EVADD time budget");}
static uint64_t hash(uint64_t z){z^=z>>30;z*=UINT64_C(0xbf58476d1ce4e5b9);z^=z>>27;z*=UINT64_C(0x94d049bb133111eb);return z^(z>>31);}
static uint64_t double_bits(double x){uint64_t b;memcpy(&b,&x,8);return b;}
static uint64_t ref_hash(Ref a){return hash(a.node)^hash(double_bits(a.scale));}
static void *allocate(size_t n,size_t s){void *p=calloc(n,s);if(!p)die("allocation failure");return p;}
static int same(Ref a,Ref b){return a.node==b.node && a.scale==b.scale;}
static Ref scaled(Ref a,double s){
    if(!a.node||s==0.)return ZERO;double v=a.scale*s;
    if(v==0. || !isfinite(v))die("nonzero scale underflow or nonfinite arithmetic");return (Ref){a.node,v};
}
static double factor(double a,double b){
    double m=fmax(fabs(a),fabs(b));if(m==0.)return 1.;int e;frexp(m,&e);
    double p=ldexp(1.,e-1);if(p==0.||!isfinite(p))die("normalization range");
    return (a!=0.?a:b)<0.?-p:p;
}
static Ref make_node(uint32_t level,Ref lo,Ref hi){
    if(same(lo,hi))return lo;
    double f=factor(lo.scale,hi.scale);lo.scale/=f;hi.scale/=f;
    if(!lo.node)lo=ZERO;if(!hi.node)hi=ZERO;
    uint64_t key=hash(level)^ref_hash(lo)^hash(ref_hash(hi));uint32_t slot=(uint32_t)key&(UNIQUE_SIZE-1);
    while(unique_nodes[slot]){
        uint32_t id=unique_nodes[slot];Node n=nodes[id];
        if(n.level==level && same(n.lo,lo)&&same(n.hi,hi))return (Ref){id,f};
        slot=(slot+1)&(UNIQUE_SIZE-1);
    }
    if(node_count>=NODE_CAP)die("EVADD node budget");
    uint32_t id=node_count++;nodes[id]=(Node){lo,hi,(uint8_t)level};unique_nodes[slot]=id;return (Ref){id,f};
}
static Ref apply(int multiply,Ref a,Ref b){
    budget();
    if(multiply){if(!a.node||!b.node)return ZERO;if(a.node==1)return scaled(b,a.scale);if(b.node==1)return scaled(a,b.scale);}
    else{if(!a.node)return b;if(!b.node)return a;if(a.node==b.node){double v=a.scale+b.scale;return v==0.?ZERO:(Ref){a.node,v};}}
    double f;
    if(multiply){f=a.scale*b.scale;if(f==0.||!isfinite(f))die("product scale range");a.scale=b.scale=1.;}
    else{f=factor(a.scale,b.scale);a.scale/=f;b.scale/=f;}
    if(a.node>b.node || (a.node==b.node && a.scale>b.scale)){Ref t=a;a=b;b=t;}
    uint64_t key=(hash(multiply)^ref_hash(a)^hash(ref_hash(b)))|1;
    uint32_t slot=(uint32_t)key&(CACHE_SIZE-1);Cache entry=cache[slot];
    if(entry.hash==key && entry.multiply==multiply && same(entry.a,a)&&same(entry.b,b)){++cache_hits;return scaled(entry.result,f);}
    uint32_t level=nodes[a.node].level<nodes[b.node].level?nodes[a.node].level:nodes[b.node].level;
    Ref al=nodes[a.node].level==level?scaled(nodes[a.node].lo,a.scale):a;
    Ref ah=nodes[a.node].level==level?scaled(nodes[a.node].hi,a.scale):a;
    Ref bl=nodes[b.node].level==level?scaled(nodes[b.node].lo,b.scale):b;
    Ref bh=nodes[b.node].level==level?scaled(nodes[b.node].hi,b.scale):b;
    Ref lo=apply(multiply,al,bl),hi=apply(multiply,ah,bh),result=make_node(level,lo,hi);
    cache[slot]=(Cache){key,a,b,result,multiply};return scaled(result,f);
}
static Ref equation(uint64_t form,int sign){
    Ref a=((form>>ROUTE_C_LABEL_BITS)&1)?(sign?((Ref){1,-1.}):ZERO):ONE;
    Ref b=((form>>ROUTE_C_LABEL_BITS)&1)?ONE:(sign?((Ref){1,-1.}):ZERO);
    for(int j=(int)dimensions-1;j>=0;j--)if((form>>j)&1){Ref x=a,y=b;a=make_node(j,x,y);b=make_node(j,y,x);}
    return a;
}
typedef struct{uint32_t node;double coefficient;} Edge;
static int compare_edges(const void *pa,const void *pb){
    uint32_t a=((const Edge*)pa)->node,b=((const Edge*)pb)->node;return (a>b)-(a<b);
}
static Ref get(int stage,int col,uint32_t k){
    budget();size_t key=((size_t)stage*C+col)*N+k;
    if(!bounds[key])return ZERO;if(k==0)return ONE;if(memo[key].node!=UNSET)return memo[key];
    ++visited;Ref value;
    if(stage==0){
        value=ONE;
        for(int j=0;j<5 && value.node;j++){
            uint64_t form=0;for(int bit=0;bit<15;bit++)if((k>>bit)&1)form^=linear_forms[col][j][bit];
            value=apply(1,value,equation(form,0));
        }
        if(value.node){
            uint64_t form=0;for(int bit=0;bit<15;bit++)if((k>>bit)&1)form^=phase_forms[col][bit];
            Ref phase=equation(form,1);uint32_t q=0,qn=dimensions*(dimensions-1)/2;
            for(uint32_t i=0;i<dimensions;i++)for(uint32_t j=i+1;j<dimensions;j++,q++){
                if(__builtin_parity(k&phase_quadratic[col*qn+q])){
                    if(monomials[i][j].node==UNSET)monomials[i][j]=make_node(i,ONE,make_node(j,ONE,((Ref){1,-1.})));
                    phase=apply(1,phase,monomials[i][j]);
                }
            }
            value=apply(1,value,phase);
        }
    }else if(stage&1){
        value=ONE;
        for(int d=0;d<64 && value.node;d++)if(projections[d]){
            uint32_t q=k&projections[d];if(q)value=apply(1,value,get(stage-1,(col-d+64)%64,q));
        }
    }else{
        int u1=k>>10,u2=(k>>5)&31,u3=k&31,u0=u1^u2^u3;value=ZERO;
        Edge edges[4096];int n=0;
        for(int a=0;a<lat_count[u1];a++)for(int b=0;b<lat_count[u2];b++)for(int c=0;c<lat_count[u3];c++){
            int v1=lat[u1][a],v2=lat[u2][b],v3=lat[u3][c],v0=v1^v2^v3;
            if(!(lat_bits[u0]&(UINT32_C(1)<<v0)))continue;
            Ref child=get(stage-1,col,(v1<<10)|(v2<<5)|v3);
            if(child.node){double w=lat_values[u0][v0]*lat_values[u1][v1]*lat_values[u2][v2]*lat_values[u3][v3];
                if(n>=4096)die("edge capacity");edges[n++]=(Edge){child.node,w*child.scale};}
        }
        /* Combine exactly identical normalized child functions before ADD
         * construction. No edge threshold; zero sums alone disappear. */
        qsort(edges,n,sizeof(Edge),compare_edges);
        for(int i=0;i<n;){uint32_t id=edges[i].node;long double total=0.;
            do{total+=edges[i++].coefficient;}while(i<n && edges[i].node==id);
            double coefficient=(double)total;if(coefficient!=0.)value=apply(0,value,((Ref){id,coefficient}));}
    }
    return memo[key]=value;
}
static uint64_t count_from_node(uint32_t n){
    if(n<2)return n;if(count_cache[n]!=UINT64_MAX)return count_cache[n];Node nd=nodes[n];
    return count_cache[n]=(count_from_node(nd.lo.node)<<(nodes[nd.lo.node].level-nd.level-1))+
                         (count_from_node(nd.hi.node)<<(nodes[nd.hi.node].level-nd.level-1));
}
static void enumerate(Ref ref,uint32_t level,uint64_t t){
    if(!ref.node)return;if(level==dimensions){if(solution_count>=enum_cap)die("enumeration overflow");solutions[solution_count]=t;solution_values[solution_count++]=ref.scale;return;}
    if(ref.node==1 || nodes[ref.node].level>level){enumerate(ref,level+1,t);enumerate(ref,level+1,t|(UINT64_C(1)<<level));}
    else{enumerate(scaled(nodes[ref.node].lo,ref.scale),level+1,t);enumerate(scaled(nodes[ref.node].hi,ref.scale),level+1,t|(UINT64_C(1)<<level));}
}
static void read_items(FILE *f,void *dest,size_t size,size_t n){if(fread(dest,size,n,f)!=n)die("truncated input");}
int main(int argc,char **argv){
    started=clock();
    if(setjmp(failure)){printf("{\"complete\":false,\"reason\":\"%s\",\"EVADD_nodes\":%u,\"function_nodes\":%"PRIu64",\"operations\":%"PRIu64",\"seconds\":%.6f",reason,node_count,visited,operations,elapsed());memory_fields();printf("}\n");return 4;}
    if(argc<2)die("input path required");if(argc>2)time_limit=atof(argv[2]);if(argc>3)enum_cap=(uint32_t)strtoul(argv[3],0,10);
    FILE *f=fopen(argv[1],"rb");if(!f)die("input open failed");uint32_t h[6];read_items(f,h,4,6);
    if(h[0]!=0x243add15 || h[1]>ROUTE_C_LABEL_BITS || h[2]!=64 || h[3]!=N || h[4]!=6 || h[5]!=64)die("invalid input");dimensions=h[1];
    bounds=allocate(6*C*N,1);read_items(f,bounds,1,6*C*N);
    uint16_t *base=allocate(C*5,2),*dirs=allocate(C*dimensions*5,2);read_items(f,base,2,C*5);read_items(f,dirs,2,C*dimensions*5);
    uint16_t *pbase=allocate(C,2),*pdirs=allocate(C*dimensions,2);phase_quadratic=allocate(C*dimensions*(dimensions-1)/2,2);
    read_items(f,pbase,2,C);read_items(f,pdirs,2,C*dimensions);read_items(f,phase_quadratic,2,C*dimensions*(dimensions-1)/2);
    uint32_t final[64];double final_weights[64];read_items(f,final,4,64);read_items(f,final_weights,8,64);fclose(f);
    for(int col=0;col<64;col++)for(int j=0;j<5;j++)for(int bit=0;bit<15;bit++){
        uint64_t form=(uint64_t)((base[col*5+j]>>bit)&1)<<ROUTE_C_LABEL_BITS;
        for(uint32_t t=0;t<dimensions;t++)form|=(uint64_t)((dirs[(col*dimensions+t)*5+j]>>bit)&1)<<t;
        linear_forms[col][j][bit]=form;
    }
    for(int col=0;col<64;col++)for(int bit=0;bit<15;bit++){
        uint64_t form=(uint64_t)((pbase[col]>>bit)&1)<<ROUTE_C_LABEL_BITS;
        for(uint32_t t=0;t<dimensions;t++)form|=(uint64_t)((pdirs[col*dimensions+t]>>bit)&1)<<t;
        phase_forms[col][bit]=form;
    }
    free(base);free(dirs);free(pbase);free(pdirs);memset(monomials,255,sizeof(monomials));
    for(int u=0;u<32;u++)for(int v=0;v<32;v++){
        int sum=0;for(int x=0;x<32;x++)sum+=__builtin_parity((u&S[x])^(v&x))?-1:1;
        lat_values[u][v]=(double)sum/32;if(sum){lat[u][lat_count[u]++]=v;lat_bits[u]|=UINT32_C(1)<<v;}
    }
    for(int row=0;row<5;row++)for(int j=0;j<3;j++){int d=j?ROT[row][j-1]:0,b=1<<(4-row);projections[d]^=(b<<10)|(b<<5)|b;}
    nodes=allocate(NODE_CAP,sizeof(Node));nodes[0]=(Node){ZERO,ZERO,(uint8_t)dimensions};nodes[1]=(Node){ONE,ONE,(uint8_t)dimensions};
    unique_nodes=allocate(UNIQUE_SIZE,4);cache=allocate(CACHE_SIZE,sizeof(Cache));
    memo=allocate(6*C*N,sizeof(Ref));memset(memo,255,6*C*N*sizeof(Ref));Ref root=ZERO;
    for(int i=0;i<64;i++)root=apply(0,root,scaled(get(5,28,final[i]),final_weights[i]));
    /* Construction caches do not own node storage. Release them before the
     * independent counting/output phase, keeping peak address-space bounded. */
    free(memo);memo=NULL;free(cache);cache=NULL;free(unique_nodes);unique_nodes=NULL;
    free(bounds);bounds=NULL;free(phase_quadratic);phase_quadratic=NULL;
    if(argc>4){FILE *g=fopen(argv[4],"wb");if(!g)die("graph output open failed");uint32_t gh[4]={0x243eed15,dimensions,node_count,root.node};fwrite(gh,4,4,g);fwrite(&root.scale,8,1,g);
        for(uint32_t i=0;i<node_count;i++){uint32_t row[3]={nodes[i].level,nodes[i].lo.node,nodes[i].hi.node};double v[2]={nodes[i].lo.scale,nodes[i].hi.scale};fwrite(row,4,3,g);fwrite(v,8,2,g);}if(fclose(g))die("graph output close failed");}
    count_cache=allocate(node_count,8);memset(count_cache,255,node_count*8);uint64_t count=count_from_node(root.node)<<nodes[root.node].level;
    if(count<=enum_cap){solutions=allocate(enum_cap,sizeof(*solutions));solution_values=allocate(enum_cap,8);enumerate(root,0,0);if(solution_count!=count)die("count mismatch");}
    printf("{\"complete\":true,\"dimension\":%u,\"nonzero_endpoints\":%"PRIu64",\"all_model_values_numeric_zero\":%s,\"candidates_enumerated\":%s,\"EVADD_nodes\":%u,\"function_nodes\":%"PRIu64",\"operations\":%"PRIu64",\"cache_hits\":%"PRIu64",\"seconds\":%.6f,\"candidate_labels\":[",dimensions,count,count==0?"true":"false",count<=enum_cap?"true":"false",node_count,visited,operations,cache_hits,elapsed());
    for(uint32_t i=0;i<solution_count;i++)printf("%s%"PRIu64,i?",":"",solutions[i]);printf("],\"correlations\":[");
    for(uint32_t i=0;i<solution_count;i++)printf("%s%.17g",i?",":"",solution_values[i]);printf("]");memory_fields();printf("}\n");return 0;
}
