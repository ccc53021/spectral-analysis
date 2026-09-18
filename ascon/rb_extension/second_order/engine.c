/* Four-message Ascon DAG with JOINT original-input Fourier labels.
 *
 * A local coordinate is (a00,a10,a01,a11), four physical message masks.
 * S-box edges have the product of four LAT entries. Linear edges multiply
 * polynomials by XOR convolution of their original-input labels. The root
 * is (lambda,lambda,lambda,lambda), equivalent to relative coordinates
 * (0,lambda,lambda,lambda). No zero-value-mask approximation is used.
 *
 * Sparse reachable-node cache; bounded label maps at intermediate nodes.
 * Root output is captured BEFORE any root label-cap truncation. Resource
 * exhaustion produces an explicit incomplete status and NO root spectrum.
 * C99 is used because the available local toolchain has a C compiler only.
 */
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <setjmp.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define WORDS 5
#define MAX_ROUNDS 12
typedef struct { uint64_t w[WORDS]; } Mask;
typedef struct { Mask mask; double coefficient; } Term;
typedef struct { size_t size; Term *terms; unsigned int phase_offset; } Polynomial;
typedef struct { Mask mask; long double value; unsigned char occupied; } Slot;
typedef struct { Slot *slots; size_t size, capacity; } Accumulator;
typedef struct { unsigned int index; double weight; unsigned char occupied; } EdgeSlot;
typedef struct { EdgeSlot *slots; size_t size, capacity; } EdgeAccumulator;
typedef struct Node { uint64_t key; Polynomial polynomial; Polynomial *views[3]; } Node;
typedef struct { unsigned char input; double coefficient; } LatEntry;

static const unsigned char SBOX[32] = {
    4,11,31,20,26,21,9,2,27,5,8,18,29,3,6,28,
    30,19,7,14,0,13,17,24,16,12,1,25,22,10,15,23
};
static const unsigned char RC[MAX_ROUNDS] = {
    0xf0,0xe1,0xd2,0xc3,0xb4,0xa5,0x96,0x87,0x78,0x69,0x5a,0x4b
};
static const int ROT[5][2] = {{19,28},{61,39},{1,6},{10,17},{7,41}};
static LatEntry LAT[32][32];
static unsigned char LAT_SIZE[32], PREPARED[64][32][64];
static uint64_t DELTA1[5], DELTA2[5], DELTA3[5], OUTPUT[5], IV, EQUAL_BITS;
/* Legacy EOF protocol keeps its fixed-IV domain and implicit d3=d1^d2.
 * SUFFIX_V1 adds a full permutation domain and an explicit third offset. */
static int FULL_INPUT_DOMAIN, EXPLICIT_DELTA3, BOUNDARY_RECTANGLE, KLEIN_CACHE_REQUESTED;
static int ROUNDS, BEGIN_ROUND, ROOT_COLUMN, FIRST_RECTANGLE;
static int KLEIN_CACHE;
static int MERGE_LINEAR_ALIAS_EDGES;
static uint64_t GENERIC_EDGES_BEFORE, GENERIC_EDGES_AFTER, GENERIC_ZERO_EDGE_GROUPS;
static size_t PEAK_EDGE_GROUPS;
static uint64_t KLEIN_ALIAS_HITS, KLEIN_VIEWS, STABILIZER_REMOVED_TERMS;
static unsigned int ROOT_INDEX;
static size_t TERM_CAP, MAX_NODES, MAX_ACCUMULATOR_TERMS;
static double TIME_LIMIT, START_TIME;
/* Observability only: no cache, pruning, or traversal decisions use these. */
static double PROGRESS_INTERVAL, NEXT_PROGRESS_SECONDS;
static Node **CACHE;
static size_t CACHE_CAPACITY, NODE_COUNT, CACHED_TERMS, PEAK_ACCUMULATOR_TERMS;
static size_t SBOX_NODES[MAX_ROUNDS], LINEAR_NODES[MAX_ROUNDS];
static uint64_t PRUNED_MAPS, DISCARDED_TERMS, XOR_PRODUCTS, HASH_ADDITIONS;
static uint64_t LEAF_ALIAS_HITS, FIRST_LINEAR_ALIAS_HITS;
static unsigned int FIRST_LINEAR_BASIS[64][20];
static unsigned int FIRST_LINEAR_LOOKUP[64][4][32];
static unsigned int KNOWN_CANON[2][MAX_ROUNDS][64][4][32];
static unsigned char POSSIBLE_DIFF[2][MAX_ROUNDS][6][64];
static const int PAIRS[6][2]={{0,1},{0,2},{0,3},{1,2},{1,3},{2,3}};
static uint64_t SUPPORT_ALIAS_HITS;
static double *FIRST_WEIGHTS;
static unsigned char *FIRST_WEIGHT_SEEN;
static unsigned int FIRST_WEIGHT_KEYS[65536];
static uint64_t FIRST_EDGES_BEFORE, FIRST_EDGES_AFTER;
static long double DISCARDED_ABSOLUTE_SUM;
static jmp_buf FAILURE;
static const char *FAILURE_STATUS;
static Term ONE_TERM = {{{0,0,0,0,0}}, 1.0};
static Polynomial ONE = {1, &ONE_TERM};

static double now_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&frequency);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    return (double)clock() / CLOCKS_PER_SEC;
#endif
}
static void fail(const char *status) { FAILURE_STATUS = status; longjmp(FAILURE, 1); }
static void print_progress(double elapsed) {
    int round;
    fprintf(stderr,"{\"schema\":\"ascon_dag_progress_v1\",\"event\":\"progress\",\"elapsed_seconds\":%.9g,\"node_count\":%zu,\"sbox_nodes\":[",elapsed,NODE_COUNT);
    for(round=0;round<ROUNDS;++round) {
        if(round)fputc(',',stderr);fprintf(stderr,"%zu",SBOX_NODES[round]);
    }
    fprintf(stderr,"],\"linear_nodes\":[");
    for(round=0;round<ROUNDS-1;++round) {
        if(round)fputc(',',stderr);fprintf(stderr,"%zu",LINEAR_NODES[round]);
    }
    fprintf(stderr,"],\"cached_terms\":%zu,\"peak_accumulator_terms\":%zu,\"peak_edge_groups\":%zu,",CACHED_TERMS,PEAK_ACCUMULATOR_TERMS,PEAK_EDGE_GROUPS);
    fprintf(stderr,"\"generic_edges_before_merge\":%" PRIu64 ",\"generic_edges_after_merge\":%" PRIu64 ",\"strict_zero_edge_groups\":%" PRIu64 ",",GENERIC_EDGES_BEFORE,GENERIC_EDGES_AFTER,GENERIC_ZERO_EDGE_GROUPS);
    fprintf(stderr,"\"klein_shared_views\":%" PRIu64 ",\"klein_alias_hits\":%" PRIu64 ",\"hash_additions\":%" PRIu64 ",\"xor_products\":%" PRIu64 ",\"pruned_maps\":%" PRIu64 "}\n",KLEIN_VIEWS,KLEIN_ALIAS_HITS,HASH_ADDITIONS,XOR_PRODUCTS,PRUNED_MAPS);
    fflush(stderr);
}
static void check_time(void) {
    double elapsed=now_seconds()-START_TIME;
    if(PROGRESS_INTERVAL>0 && elapsed>=NEXT_PROGRESS_SECONDS) {
        print_progress(elapsed);NEXT_PROGRESS_SECONDS=elapsed+PROGRESS_INTERVAL;
    }
    if (elapsed > TIME_LIMIT) fail("time_budget");
}
static void *checked_calloc(size_t count, size_t size) {
    void *result;
    if (size && count > SIZE_MAX / size) fail("memory_budget");
    result = calloc(count, size);
    if (result == NULL) fail("memory_budget");
    return result;
}
static unsigned int edge_hash(unsigned int x) {
    x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;return x^(x>>16);
}
static void add_edge(EdgeAccumulator *a,unsigned int index,double weight) {
    size_t slot;
    if(!a->capacity || (a->size+1)*10>a->capacity*7) {
        size_t capacity=a->capacity?2*a->capacity:32,i;
        EdgeSlot *slots=(EdgeSlot *)checked_calloc(capacity,sizeof(EdgeSlot));
        for(i=0;i<a->capacity;++i)if(a->slots[i].occupied) {
            slot=edge_hash(a->slots[i].index)&(capacity-1);
            while(slots[slot].occupied)slot=(slot+1)&(capacity-1);
            slots[slot]=a->slots[i];
        }
        free(a->slots);a->slots=slots;a->capacity=capacity;
    }
    slot=edge_hash(index)&(a->capacity-1);
    while(a->slots[slot].occupied && a->slots[slot].index!=index)slot=(slot+1)&(a->capacity-1);
    if(!a->slots[slot].occupied) {
        a->slots[slot].occupied=1;a->slots[slot].index=index;++a->size;
        if(a->size>PEAK_EDGE_GROUPS)PEAK_EDGE_GROUPS=a->size;
    }
    /* Products of four Ascon LAT entries, and their at-most-65536 sums,
     * are exactly representable dyadics in double. No epsilon is used. */
    a->slots[slot].weight+=weight;
}
static uint64_t mix64(uint64_t x) {
    x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static uint64_t mask_hash(const Mask *mask) {
    uint64_t h = UINT64_C(0x9e3779b97f4a7c15);
    int row;
    for (row = 0; row < 5; ++row) h = mix64(h ^ mask->w[row]);
    return h;
}
static int mask_equal(const Mask *a, const Mask *b) {
    return memcmp(a->w, b->w, sizeof(a->w)) == 0;
}
static int mask_zero(const Mask *a) {
    return !(a->w[0] | a->w[1] | a->w[2] | a->w[3] | a->w[4]);
}
static int parity(unsigned int x) { return __builtin_parity(x); }
static int label_offset_parity(const Mask *mask,unsigned int offset) {
    uint64_t combined=0;int row;
    for(row=0;row<5;++row)combined^=mask->w[row]
        &(((offset&1)?DELTA1[row]:0)^((offset&2)?DELTA2[row]:0));
    return __builtin_parityll(combined);
}
static double coefficient(const Polynomial *poly,size_t index) {
    double value=poly->terms[index].coefficient;
    if(poly->phase_offset && label_offset_parity(&poly->terms[index].mask,poly->phase_offset))value=-value;
    return value;
}
static unsigned int pack(const unsigned int a[4]) {
    return (a[0] << 15) | (a[1] << 10) | (a[2] << 5) | a[3];
}
static void unpack(unsigned int packed, unsigned int a[4]) {
    a[0] = packed >> 15; a[1] = (packed >> 10) & 31;
    a[2] = (packed >> 5) & 31; a[3] = packed & 31;
}
static unsigned int column(const uint64_t words[5], int col) {
    unsigned int result = 0; int row;
    for (row = 0; row < 5; ++row)
        result |= (unsigned int)((words[row] >> (63-col)) & 1) << (4-row);
    return result;
}
static void grow_accumulator(Accumulator *a) {
    size_t new_capacity = a->capacity ? 2*a->capacity : 32, i;
    Slot *slots = (Slot *)checked_calloc(new_capacity, sizeof(Slot));
    for (i = 0; i < a->capacity; ++i) if (a->slots[i].occupied) {
        size_t index = (size_t)mask_hash(&a->slots[i].mask) & (new_capacity-1);
        while (slots[index].occupied) index = (index+1) & (new_capacity-1);
        slots[index] = a->slots[i];
    }
    free(a->slots); a->slots = slots; a->capacity = new_capacity;
}
static void add(Accumulator *a, const Mask *mask, long double value) {
    size_t index;
    if (value == 0) return;
    if ((++HASH_ADDITIONS & 4095) == 0) check_time();
    if (!a->capacity || (a->size+1)*10 > a->capacity*7) grow_accumulator(a);
    index = (size_t)mask_hash(mask) & (a->capacity-1);
    while (a->slots[index].occupied && !mask_equal(&a->slots[index].mask, mask))
        index = (index+1) & (a->capacity-1);
    if (!a->slots[index].occupied) {
        if (a->size >= MAX_ACCUMULATOR_TERMS) fail("term_budget");
        a->slots[index].occupied = 1; a->slots[index].mask = *mask;
        ++a->size;
        if (a->size > PEAK_ACCUMULATOR_TERMS) PEAK_ACCUMULATOR_TERMS = a->size;
    }
    a->slots[index].value += value;
}
static int compare_terms(const void *ap, const void *bp) {
    const Term *a = (const Term *)ap, *b = (const Term *)bp; int row;
    double aa = fabs(a->coefficient), bb = fabs(b->coefficient);
    if (aa > bb) return -1;
    if (aa < bb) return 1;
    for (row = 0; row < 5; ++row) {
        if (a->mask.w[row] < b->mask.w[row]) return -1;
        if (a->mask.w[row] > b->mask.w[row]) return 1;
    }
    return 0;
}
static Polynomial finish(Accumulator *a, int keep_full) {
    Polynomial result = {0, NULL};
    size_t i, nonzero = 0, retained;
    for (i = 0; i < a->capacity; ++i)
        if (a->slots[i].occupied && a->slots[i].value != 0) ++nonzero;
    if (nonzero) {
        result.terms = (Term *)checked_calloc(nonzero, sizeof(Term));
        for (i = 0; i < a->capacity; ++i) {
            if (!a->slots[i].occupied || a->slots[i].value == 0) continue;
            result.terms[result.size].mask = a->slots[i].mask;
            result.terms[result.size].coefficient = (double)a->slots[i].value;
            ++result.size;
        }
        qsort(result.terms, result.size, sizeof(Term), compare_terms);
    }
    free(a->slots); a->slots = NULL; a->size = a->capacity = 0;
    retained = result.size;
    if (!keep_full && TERM_CAP && retained > TERM_CAP) {
        size_t zero_index = SIZE_MAX;
        retained = TERM_CAP;
        for (i = retained; i < result.size; ++i)
            if (mask_zero(&result.terms[i].mask)) { zero_index = i; break; }
        if (zero_index != SIZE_MAX) {
            Term temporary = result.terms[retained-1];
            result.terms[retained-1] = result.terms[zero_index];
            result.terms[zero_index] = temporary;
        }
        ++PRUNED_MAPS; DISCARDED_TERMS += result.size-retained;
        for (i = retained; i < result.size; ++i)
            DISCARDED_ABSOLUTE_SUM += fabs(result.terms[i].coefficient);
        result.size = retained;
        qsort(result.terms, retained, sizeof(Term), compare_terms);
        /* Shrink allocations: discarded labels must not stay resident. */
        {
            Term *compact = (Term *)checked_calloc(retained, sizeof(Term));
            memcpy(compact, result.terms, retained*sizeof(Term));
            free(result.terms); result.terms = compact;
        }
    }
    return result;
}
static Polynomial multiply(const Polynomial *left, const Polynomial *right) {
    Accumulator a = {NULL,0,0}; size_t i,j; ++XOR_PRODUCTS;
    for (i = 0; i < left->size; ++i) for (j = 0; j < right->size; ++j) {
        Mask mask; int row;
        for (row = 0; row < 5; ++row) mask.w[row] = left->terms[i].mask.w[row] ^ right->terms[j].mask.w[row];
        add(&a, &mask, (long double)coefficient(left,i) * coefficient(right,j));
    }
    return finish(&a, 0);
}
static Polynomial copy_polynomial(const Polynomial *source) {
    Polynomial result = {source->size, NULL, 0};
    if (result.size) {
        result.terms = (Term *)checked_calloc(result.size, sizeof(Term));
        memcpy(result.terms, source->terms, result.size*sizeof(Term));
        if(source->phase_offset) {
            size_t i;for(i=0;i<result.size;++i)result.terms[i].coefficient=coefficient(source,i);
        }
    }
    return result;
}
static const Polynomial *node_view(Node *node,unsigned int offset) {
    if(!offset)return &node->polynomial;
    if(!node->views[offset-1]) {
        Polynomial *view=(Polynomial *)checked_calloc(1,sizeof(Polynomial));
        *view=node->polynomial;view->phase_offset=offset;
        node->views[offset-1]=view;++KLEIN_VIEWS;
    }
    return node->views[offset-1];
}
static void grow_cache(void) {
    size_t capacity = CACHE_CAPACITY ? 2*CACHE_CAPACITY : 1024, i;
    Node **slots = (Node **)checked_calloc(capacity, sizeof(Node *));
    for (i = 0; i < CACHE_CAPACITY; ++i) if (CACHE[i]) {
        size_t index = (size_t)mix64(CACHE[i]->key) & (capacity-1);
        while (slots[index]) index = (index+1) & (capacity-1);
        slots[index] = CACHE[i];
    }
    free(CACHE); CACHE = slots; CACHE_CAPACITY = capacity;
}
static Node *get_node(uint64_t key, int *fresh) {
    size_t index;
    if (!CACHE_CAPACITY) grow_cache();
    index = (size_t)mix64(key) & (CACHE_CAPACITY-1);
    while (CACHE[index] && CACHE[index]->key != key) index = (index+1) & (CACHE_CAPACITY-1);
    if (CACHE[index]) { *fresh = 0; return CACHE[index]; }
    check_time();
    if (NODE_COUNT >= MAX_NODES) fail("node_budget");
    if ((NODE_COUNT+1)*10 > CACHE_CAPACITY*7) {
        grow_cache(); index = (size_t)mix64(key) & (CACHE_CAPACITY-1);
        while (CACHE[index]) index = (index+1) & (CACHE_CAPACITY-1);
    }
    CACHE[index] = (Node *)checked_calloc(1, sizeof(Node));
    CACHE[index]->key = key; ++NODE_COUNT; *fresh = 1;
    return CACHE[index];
}
static uint64_t node_key(int stage, int round, int col, unsigned int index) {
    return (((uint64_t)(round*2+stage)*64 + (unsigned int)col) << 20) | index;
}
static unsigned int canonical_known(int stage,int round,int col,unsigned int index) {
    return KNOWN_CANON[stage][round][col][0][index&31]
        ^ KNOWN_CANON[stage][round][col][1][(index>>5)&31]
        ^ KNOWN_CANON[stage][round][col][2][(index>>10)&31]
        ^ KNOWN_CANON[stage][round][col][3][index>>15];
}
static unsigned int permuted(unsigned int index,unsigned int offset) {
    unsigned int input[4],output[4];int j;unpack(index,input);
    for(j=0;j<4;++j)output[j]=input[j^offset];
    return pack(output);
}
static unsigned int canonical_klein(int stage,int round,int col,unsigned int index,unsigned int *offset) {
    unsigned int best=index,shift;*offset=0;
    if(!KLEIN_CACHE||round==0)return index;
    for(shift=1;shift<4;++shift) {
        unsigned int candidate=canonical_known(stage,round,col,permuted(index,shift));
        if(candidate<best) {best=candidate;*offset=shift;}
    }
    if(*offset)++KLEIN_ALIAS_HITS;
    return best;
}
static void enforce_stabilizer(Polynomial *poly,int stage,int round,int col,unsigned int index) {
    unsigned int shifts[3],count=0,shift;size_t i,kept=0;
    if(!KLEIN_CACHE||round==0)return;
    for(shift=1;shift<4;++shift)
        if(canonical_known(stage,round,col,permuted(index,shift))==index)shifts[count++]=shift;
    if(!count)return;
    for(i=0;i<poly->size;++i) {
        unsigned int j;int remove=0;
        for(j=0;j<count;++j)if(label_offset_parity(&poly->terms[i].mask,shifts[j])) {remove=1;break;}
        if(remove)++STABILIZER_REMOVED_TERMS;
        else poly->terms[kept++]=poly->terms[i];
    }
    poly->size=kept;
}
static unsigned int canonical_leaf(int col, unsigned int index) {
    unsigned int masks[4], offsets[4]; int i,j;
    index=canonical_known(0,0,col,index);unpack(index,masks);
    offsets[0]=0; offsets[1]=column(DELTA1,col); offsets[2]=column(DELTA2,col); offsets[3]=column(DELTA3,col);
    for (i=1;i<4;++i) for (j=0;j<i;++j) if (offsets[i]==offsets[j]) {
        masks[j]^=masks[i]; masks[i]=0; break;
    }
    return pack(masks);
}
static unsigned int canonical_first_linear(int col, unsigned int index) {
    return FIRST_LINEAR_LOOKUP[col][0][index&31]
        ^ FIRST_LINEAR_LOOKUP[col][1][(index>>5)&31]
        ^ FIRST_LINEAR_LOOKUP[col][2][(index>>10)&31]
        ^ FIRST_LINEAR_LOOKUP[col][3][index>>15];
}
static void prepare_support_aliases(void) {
    unsigned char sbox_bound[32]={0};int support,d,x,col,pair,round,row,stage;
    for(support=0;support<32;++support) for(d=0;d<32;++d) if(!(d&~support))
        for(x=0;x<32;++x)sbox_bound[support]|=SBOX[x]^SBOX[x^d];
    for(col=0;col<64;++col) {
        unsigned int offsets[4]={0,0,0,0};
        unsigned int constant=(col>=56 && ((RC[BEGIN_ROUND]>>(63-col))&1))?4:0;
        offsets[1]=column(DELTA1,col);offsets[2]=column(DELTA2,col);offsets[3]=column(DELTA3,col);
        for(x=0;x<32;++x) {
            if(!FULL_INPUT_DOMAIN && (unsigned int)(x>>4)!=((IV>>(63-col))&1))continue;
            if(!FULL_INPUT_DOMAIN && ((EQUAL_BITS>>(63-col))&1) && ((x&1)!=((x>>1)&1)))continue;
            for(pair=0;pair<6;++pair)
                POSSIBLE_DIFF[0][0][pair][col]|=SBOX[x^offsets[PAIRS[pair][0]]^constant]
                    ^SBOX[x^offsets[PAIRS[pair][1]]^constant];
        }
    }
    for(round=0;round<ROUNDS;++round) {
        if(round)for(pair=0;pair<6;++pair)for(col=0;col<64;++col)
            POSSIBLE_DIFF[0][round][pair][col]=sbox_bound[POSSIBLE_DIFF[1][round-1][pair][col]];
        for(pair=0;pair<6;++pair)for(col=0;col<64;++col)for(row=0;row<5;++row) {
            unsigned char bit=(unsigned char)(1U<<(4-row));
            /* OR gives a conservative support bound. XOR here would be
             * unsound: possible activity at two sources need not cancel. */
            POSSIBLE_DIFF[1][round][pair][col]|=bit & (
                POSSIBLE_DIFF[0][round][pair][col]
                |POSSIBLE_DIFF[0][round][pair][(col+64-ROT[row][0])%64]
                |POSSIBLE_DIFF[0][round][pair][(col+64-ROT[row][1])%64]);
        }
    }
    for(stage=0;stage<2;++stage)for(round=0;round<ROUNDS;++round)for(col=0;col<64;++col) {
        unsigned int basis[20]={0};int bucket,value,bit;
        for(row=0;row<5;++row) {
            int owners[4]={0,1,2,3},message;
            for(pair=0;pair<6;++pair) if(!(POSSIBLE_DIFF[stage][round][pair][col]&(1U<<(4-row)))) {
                int a=owners[PAIRS[pair][0]],b=owners[PAIRS[pair][1]],smaller=a<b?a:b;
                for(message=0;message<4;++message)if(owners[message]==a||owners[message]==b)owners[message]=smaller;
            }
            for(message=0;message<4;++message)
                basis[(3-message)*5+4-row]=1U<<((3-owners[message])*5+4-row);
        }
        for(bucket=0;bucket<4;++bucket)for(value=0;value<32;++value)
            for(bit=0;bit<5;++bit)if(value&(1<<bit))
                KNOWN_CANON[stage][round][col][bucket][value]^=basis[bucket*5+bit];
    }
}
static void prepare_first_linear_aliases(void) {
    int col,bit,source;
    /* The signature of an L0 mask is the vector of its canonical first-S
     * leaf coordinates. This map is GF(2)-linear even when initial local
     * directions overlap. A 20-bit quotient representative avoids storing
     * a large signature or assuming that later messages remain paired. */
    for(col=0;col<64;++col) {
        unsigned int pivots[20][64]={{0}}, representations[20]={0};
        int positions[20],rank=0;
        for(bit=0;bit<20;++bit) {
            unsigned int signature[64],masks[4],representation=1U<<bit;
            int p,highest=-1;
            unpack(1U<<bit,masks);
            for(source=0;source<64;++source) {
                unsigned int input[4];int j;
                for(j=0;j<4;++j)input[j]=PREPARED[col][masks[j]][source];
                signature[source]=canonical_leaf(source,pack(input));
            }
            for(p=0;p<rank;++p) {
                int location=positions[p];
                if(signature[location/20]&(1U<<(location%20))) {
                    for(source=0;source<64;++source) signature[source]^=pivots[p][source];
                    representation^=representations[p];
                }
            }
            for(source=63;source>=0;--source) if(signature[source]) {
                highest=source*20+31-__builtin_clz(signature[source]);break;
            }
            if(highest<0)FIRST_LINEAR_BASIS[col][bit]=representation^(1U<<bit);
            else {
                int insertion=rank;
                while(insertion>0 && positions[insertion-1]<highest) {
                    positions[insertion]=positions[insertion-1];
                    representations[insertion]=representations[insertion-1];
                    memcpy(pivots[insertion],pivots[insertion-1],sizeof(signature));
                    --insertion;
                }
                positions[insertion]=highest;representations[insertion]=representation;
                memcpy(pivots[insertion],signature,sizeof(signature));++rank;
                FIRST_LINEAR_BASIS[col][bit]=1U<<bit;
            }
        }
        {
            int bucket,value;
            for(bucket=0;bucket<4;++bucket)for(value=0;value<32;++value)
                for(bit=0;bit<5;++bit)if(value&(1<<bit))
                    FIRST_LINEAR_LOOKUP[col][bucket][value]^=FIRST_LINEAR_BASIS[col][bucket*5+bit];
        }
    }
}
static Polynomial first_leaf(int col, unsigned int index, int keep_full) {
    unsigned int masks[4], offsets[4];
    int free_rows[5]={1,2,3,4,0}, free_count = ((EQUAL_BITS >> (63-col)) & 1) ? 3 : 4;
    int length, z,u,j,k,width;
    double values[32]; Accumulator a={NULL,0,0};
    unsigned int constant = (col>=56 && ((RC[BEGIN_ROUND] >> (63-col)) & 1)) ? 4 : 0;
    unsigned int iv_bit = (unsigned int)((IV >> (63-col)) & 1);
    if(FULL_INPUT_DOMAIN) {
        free_count=5;for(j=0;j<5;++j)free_rows[j]=j;
    }
    length=1<<free_count;
    unpack(index,masks);
    offsets[0]=0; offsets[1]=column(DELTA1,col); offsets[2]=column(DELTA2,col); offsets[3]=column(DELTA3,col);
    for (z=0;z<length;++z) {
        unsigned int x=FULL_INPUT_DOMAIN?0:iv_bit<<4, phase=0;
        for (j=0;j<free_count;++j) x |= ((z>>j)&1) << (4-free_rows[j]);
        if (!FULL_INPUT_DOMAIN && free_count==3) x |= (x>>1)&1;
        for (j=0;j<4;++j) phase ^= parity(masks[j] & SBOX[x ^ offsets[j] ^ constant]);
        values[z]=phase ? -1.0 : 1.0;
    }
    for (width=1;width<length;width*=2) for (z=0;z<length;z+=2*width)
        for (k=0;k<width;++k) {
            double left=values[z+k], right=values[z+k+width];
            values[z+k]=left+right; values[z+k+width]=left-right;
        }
    for (u=0;u<length;++u) if (values[u]!=0) {
        Mask mask={{0,0,0,0,0}};
        for (j=0;j<free_count;++j) if ((u>>j)&1) mask.w[free_rows[j]] |= UINT64_C(1) << (63-col);
        add(&a,&mask,values[u]/length);
    }
    return finish(&a,keep_full);
}

static const Polynomial *eval_sbox(int round,int col,unsigned int index);
static const Polynomial *eval_linear(int round,int col,unsigned int index) {
    Node *node; int fresh,source,j; unsigned int masks[4]; Polynomial accumulator;
    unsigned int phase_offset=0;
    {
        unsigned int canonical=canonical_known(1,round,col,index);
        if(canonical!=index)++SUPPORT_ALIAS_HITS;
        index=canonical;
    }
    if (round==0) {
        unsigned int canonical=canonical_first_linear(col,index);
        if (canonical!=index) ++FIRST_LINEAR_ALIAS_HITS;
        index=canonical;
    }
    index=canonical_klein(1,round,col,index,&phase_offset);
    if (!index) return &ONE;
    node=get_node(node_key(1,round,col,index),&fresh);
    if (!fresh) return node_view(node,phase_offset);
    ++LINEAR_NODES[round]; unpack(index,masks); accumulator=copy_polynomial(&ONE);
    for (source=0;source<64;++source) {
        unsigned int input[4], input_index; const Polynomial *factor;
        for (j=0;j<4;++j) input[j]=PREPARED[col][masks[j]][source];
        input_index=pack(input); if (!input_index) continue;
        factor=eval_sbox(round,source,input_index);
        /* Visit every reachable factor even when prior capped coefficients
         * cancelled. Label-cap truncation must not silently trim the V DAG. */
        if (accumulator.size) {
            Polynomial next=multiply(&accumulator,factor);
            free(accumulator.terms); accumulator=next;
        }
    }
    enforce_stabilizer(&accumulator,1,round,col,index);
    node->polynomial=accumulator; CACHED_TERMS+=accumulator.size;
    return node_view(node,phase_offset);
}
static const Polynomial *eval_sbox(int round,int col,unsigned int index) {
    Node *node; int fresh,j0,j1,j2,j3,keep_full;
    unsigned int output[4]; Accumulator accumulator={NULL,0,0};
    size_t first_key_count=0;
    EdgeAccumulator edges={NULL,0,0};
    unsigned int phase_offset=0;
    {
        unsigned int canonical=canonical_known(0,round,col,index);
        if(canonical!=index)++SUPPORT_ALIAS_HITS;
        index=canonical;
    }
    if (round==0) {
        unsigned int canonical=canonical_leaf(col,index);
        if (canonical!=index) ++LEAF_ALIAS_HITS;
        index=canonical;
    }
    index=canonical_klein(0,round,col,index,&phase_offset);
    if (!index) return &ONE;
    node=get_node(node_key(0,round,col,index),&fresh);
    if (!fresh) return node_view(node,phase_offset);
    ++SBOX_NODES[round];
    keep_full=(round==ROUNDS-1 && col==ROOT_COLUMN);
    if (round==0) {
        node->polynomial=first_leaf(col,index,keep_full);
        CACHED_TERMS+=node->polynomial.size;
        return &node->polynomial;
    }
    unpack(index,output);
    for (j0=0;j0<LAT_SIZE[output[0]];++j0)
    for (j1=0;j1<LAT_SIZE[output[1]];++j1)
    for (j2=0;j2<LAT_SIZE[output[2]];++j2)
    for (j3=0;j3<LAT_SIZE[output[3]];++j3) {
        unsigned int input[4]; size_t term;
        const Polynomial *child;
        double weight;
        input[0]=LAT[output[0]][j0].input; input[1]=LAT[output[1]][j1].input;
        input[2]=LAT[output[2]][j2].input; input[3]=LAT[output[3]][j3].input;
        weight=LAT[output[0]][j0].coefficient*LAT[output[1]][j1].coefficient
              *LAT[output[2]][j2].coefficient*LAT[output[3]][j3].coefficient;
        if (col>=56 && ((RC[BEGIN_ROUND+round]>>(63-col))&1)
            && ((input[0]^input[1]^input[2]^input[3])&4)) weight=-weight;
        if(round==1) {
            unsigned int canonical=canonical_first_linear(col,pack(input));
            ++FIRST_EDGES_BEFORE;
            if(!FIRST_WEIGHT_SEEN[canonical]) {
                FIRST_WEIGHT_SEEN[canonical]=1;
                FIRST_WEIGHT_KEYS[first_key_count++]=canonical;
            }
            FIRST_WEIGHTS[canonical]+=weight;
        } else if(MERGE_LINEAR_ALIAS_EDGES) {
            unsigned int canonical=canonical_known(1,round-1,col,pack(input));
            add_edge(&edges,canonical,weight);++GENERIC_EDGES_BEFORE;
        } else {
            child=eval_linear(round-1,col,pack(input));
            for (term=0;term<child->size;++term)
                add(&accumulator,&child->terms[term].mask,(long double)weight*coefficient(child,term));
        }
        if ((j3&15)==0) check_time();
    }
    if(round==1) {
        size_t key;
        for(key=0;key<first_key_count;++key) {
            unsigned int canonical=FIRST_WEIGHT_KEYS[key];
            double weight=FIRST_WEIGHTS[canonical];
            FIRST_WEIGHTS[canonical]=0;FIRST_WEIGHT_SEEN[canonical]=0;
            if(weight) {
                const Polynomial *child=eval_linear(0,col,canonical);size_t term;
                ++FIRST_EDGES_AFTER;
                for(term=0;term<child->size;++term)
                    add(&accumulator,&child->terms[term].mask,(long double)weight*coefficient(child,term));
            }
            if((key&255)==0)check_time();
        }
    }
    if(round>=2 && MERGE_LINEAR_ALIAS_EDGES) {
        size_t edge;
        for(edge=0;edge<edges.capacity;++edge)if(edges.slots[edge].occupied) {
            double weight=edges.slots[edge].weight;
            if(weight) {
                const Polynomial *child=eval_linear(round-1,col,edges.slots[edge].index);size_t term;
                ++GENERIC_EDGES_AFTER;
                for(term=0;term<child->size;++term)
                    add(&accumulator,&child->terms[term].mask,(long double)weight*coefficient(child,term));
            } else ++GENERIC_ZERO_EDGE_GROUPS;
            if((edge&255)==0)check_time();
        }
        free(edges.slots);
    }
    node->polynomial=finish(&accumulator,keep_full);
    enforce_stabilizer(&node->polynomial,0,round,col,index);
    CACHED_TERMS+=node->polynomial.size;
    return node_view(node,phase_offset);
}
static void prepare_boundary(void) {
    int row;
    BOUNDARY_RECTANGLE=1;KLEIN_CACHE_REQUESTED=KLEIN_CACHE;
    for(row=0;row<5;++row) {
        if(!EXPLICIT_DELTA3)DELTA3[row]=DELTA1[row]^DELTA2[row];
        if(DELTA3[row]!=(DELTA1[row]^DELTA2[row]))BOUNDARY_RECTANGLE=0;
    }
    /* Translation sharing requires the actual four offsets to be a
     * parallelogram AND its two translations to preserve the input domain. */
    if(!BOUNDARY_RECTANGLE || (!FULL_INPUT_DOMAIN &&
       (DELTA1[0] || DELTA2[0] || ((DELTA1[3]^DELTA1[4])&EQUAL_BITS)
        || ((DELTA2[3]^DELTA2[4])&EQUAL_BITS))))KLEIN_CACHE=0;
}
static void prepare(void) {
    int v,u,x,col,row;
    prepare_boundary();FIRST_RECTANGLE=BOUNDARY_RECTANGLE;
    for (v=0;v<32;++v) for (u=0;u<32;++u) {
        int sum=0;
        for (x=0;x<32;++x) sum+=parity((unsigned int)((v&SBOX[x])^(u&x))) ? -1 : 1;
        if (sum) {
            int i=LAT_SIZE[v]++;
            LAT[v][i].input=(unsigned char)u; LAT[v][i].coefficient=sum/32.0;
        }
    }
    for (col=0;col<64;++col) {
        if (column(DELTA1,col) && column(DELTA2,col)) FIRST_RECTANGLE=0;
        for (v=0;v<32;++v) for (row=0;row<5;++row) if (v&(1<<(4-row))) {
            unsigned char bit=(unsigned char)(1<<(4-row));
            PREPARED[col][v][col]^=bit;
            PREPARED[col][v][(col+64-ROT[row][0])%64]^=bit;
            PREPARED[col][v][(col+64-ROT[row][1])%64]^=bit;
        }
    }
    prepare_support_aliases();prepare_first_linear_aliases();
    if(ROUNDS>=2) {
        FIRST_WEIGHTS=(double *)checked_calloc(1U<<20,sizeof(double));
        FIRST_WEIGHT_SEEN=(unsigned char *)checked_calloc(1U<<20,sizeof(unsigned char));
    }
}
static void print_result(const Polynomial *root, const char *status) {
    int round; size_t i; double elapsed=now_seconds()-START_TIME;
    printf("{\"status\":\"%s\",\"complete\":%s,\"terms\":[",status,root?"true":"false");
    if (root) for (i=0;i<root->size;++i) {
        int row; if (i) putchar(','); printf("{\"mask_words\":[");
        for (row=0;row<5;++row) {
            if (row) putchar(','); printf("\"0x%016" PRIx64 "\"",root->terms[i].mask.w[row]);
        }
        printf("],\"coefficient\":%.17g}",coefficient(root,i));
    }
    printf("],\"stats\":{\"node_count\":%zu,\"sbox_nodes\":[",NODE_COUNT);
    for (round=0;round<ROUNDS;++round) { if(round)putchar(',');printf("%zu",SBOX_NODES[round]); }
    printf("],\"linear_nodes\":[");
    for (round=0;round<ROUNDS-1;++round) { if(round)putchar(',');printf("%zu",LINEAR_NODES[round]); }
    printf("],\"cached_terms\":%zu,\"peak_accumulator_terms\":%zu,",CACHED_TERMS,PEAK_ACCUMULATOR_TERMS);
    printf("\"pruned_maps\":%" PRIu64 ",\"discarded_terms\":%" PRIu64 ",\"discarded_local_absolute_sum\":%.17g,",PRUNED_MAPS,DISCARDED_TERMS,(double)DISCARDED_ABSOLUTE_SUM);
    printf("\"xor_products\":%" PRIu64 ",\"hash_additions\":%" PRIu64 ",\"leaf_alias_hits\":%" PRIu64 ",\"first_linear_alias_hits\":%" PRIu64 ",",XOR_PRODUCTS,HASH_ADDITIONS,LEAF_ALIAS_HITS,FIRST_LINEAR_ALIAS_HITS);
    printf("\"first_edges_before_alias_merge\":%" PRIu64 ",\"first_edges_after_alias_merge\":%" PRIu64 ",\"initial_disjoint_columns\":%s,",FIRST_EDGES_BEFORE,FIRST_EDGES_AFTER,FIRST_RECTANGLE?"true":"false");
    printf("\"proven_equal_bit_alias_hits\":%" PRIu64 ",",SUPPORT_ALIAS_HITS);
    printf("\"klein_cache\":%s,\"klein_alias_hits\":%" PRIu64 ",\"klein_shared_views\":%" PRIu64 ",\"stabilizer_removed_terms\":%" PRIu64 ",",KLEIN_CACHE?"true":"false",KLEIN_ALIAS_HITS,KLEIN_VIEWS,STABILIZER_REMOVED_TERMS);
    printf("\"merge_linear_alias_edges\":%s,\"generic_edges_before_merge\":%" PRIu64 ",\"generic_edges_after_merge\":%" PRIu64 ",\"strict_zero_edge_groups\":%" PRIu64 ",\"peak_edge_groups\":%zu,",MERGE_LINEAR_ALIAS_EDGES?"true":"false",GENERIC_EDGES_BEFORE,GENERIC_EDGES_AFTER,GENERIC_ZERO_EDGE_GROUPS,PEAK_EDGE_GROUPS);
    if(FULL_INPUT_DOMAIN)printf("\"full_input_domain\":true,\"boundary_rectangle\":%s,\"klein_cache_requested\":%s,",BOUNDARY_RECTANGLE?"true":"false",KLEIN_CACHE_REQUESTED?"true":"false");
    printf("\"elapsed_seconds\":%.9g,\"pointer_bits\":%zu,\"root_pre_cap\":true,\"root_term_count\":%zu,\"no_label_pruning\":%s}}\n",elapsed,8*sizeof(void *),root?root->size:0,PRUNED_MAPS?"false":"true");
}
int main(void) {
    unsigned long long cap,nodes,terms; int row,col,active=0;
    const Polynomial *root; unsigned int masks[4];
    const char *progress_environment;
    char extension[32];int extension_read;
    START_TIME=now_seconds();
    progress_environment=getenv("ASCON_DAG_PROGRESS_SECONDS");
    if(progress_environment) {
        char *end;
        double interval=strtod(progress_environment,&end);
        if(end!=progress_environment && *end=='\0' && isfinite(interval) && interval>0) {
            PROGRESS_INTERVAL=interval;NEXT_PROGRESS_SECONDS=interval;
        }
    }
    if (scanf("%d %d %llu %llu %lf %llu %d %d",&ROUNDS,&BEGIN_ROUND,&cap,&nodes,&TIME_LIMIT,&terms,&KLEIN_CACHE,&MERGE_LINEAR_ALIAS_EDGES)!=8) return 3;
    TERM_CAP=(size_t)cap;MAX_NODES=(size_t)nodes;MAX_ACCUMULATOR_TERMS=(size_t)terms;
    if (ROUNDS<1||ROUNDS>MAX_ROUNDS||BEGIN_ROUND<0||BEGIN_ROUND>MAX_ROUNDS-ROUNDS||!nodes||!terms||!isfinite(TIME_LIMIT)||TIME_LIMIT<=0
        ||cap>SIZE_MAX||nodes>SIZE_MAX||terms>SIZE_MAX||(KLEIN_CACHE!=0&&KLEIN_CACHE!=1)
        ||(MERGE_LINEAR_ALIAS_EDGES!=0&&MERGE_LINEAR_ALIAS_EDGES!=1)) return 3;
    if (scanf("%" SCNx64 " %" SCNx64,&IV,&EQUAL_BITS)!=2) return 3;
    for(row=0;row<5;++row)if(scanf("%" SCNx64,&DELTA1[row])!=1)return 3;
    for(row=0;row<5;++row)if(scanf("%" SCNx64,&DELTA2[row])!=1)return 3;
    for(row=0;row<5;++row)if(scanf("%" SCNx64,&OUTPUT[row])!=1)return 3;
    extension_read=scanf("%31s",extension);
    if(extension_read==1) {
        int domain;
        if(strcmp(extension,"SUFFIX_V1") || scanf("%d",&domain)!=1 || domain!=1)return 3;
        FULL_INPUT_DOMAIN=1;EXPLICIT_DELTA3=1;EQUAL_BITS=0;
        for(row=0;row<5;++row)if(scanf("%" SCNx64,&DELTA3[row])!=1)return 3;
        if(scanf("%31s",extension)!=EOF)return 3;
    } else if(extension_read!=EOF)return 3;
    if(!FULL_INPUT_DOMAIN && (ROUNDS>6 || BEGIN_ROUND!=0))return 3;
    for(col=0;col<64;++col) if(column(OUTPUT,col)) { ROOT_COLUMN=col;ROOT_INDEX=column(OUTPUT,col);++active; }
    if(active>1)return 3;
    if(setjmp(FAILURE)) { print_result(NULL,FAILURE_STATUS);return 2; }
    if(!active) { prepare_boundary();check_time();print_result(&ONE,"complete");return 0; }
    prepare(); masks[0]=masks[1]=masks[2]=masks[3]=ROOT_INDEX;
    root=eval_sbox(ROUNDS-1,ROOT_COLUMN,pack(masks));
    check_time();print_result(root,"complete");
    return 0;
}
