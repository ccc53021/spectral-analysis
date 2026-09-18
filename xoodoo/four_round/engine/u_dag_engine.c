#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Xoodoo chi-input -> chi-output first-order non-zero round-based engine.
 *
 * Every reachable local (value-mask,difference-mask) coordinate is retained
 * in the V-DAG.  Its scalar coefficient is lifted to a sparse polynomial in
 * the original 384-bit input-value Fourier label U.  Sum nodes merge equal U;
 * linear-layer products XOR-convolve U.  TERM_CAP bounds only intermediate U
 * maps.  With no pruning the result is the exact joint-U version of the
 * authors' non-zero recurrence, not an exact Xoodoo hull.
 */

#define WORDS 6
#define COLUMNS 128
#define N 8
#define TENSOR 64
#define MAX_ROUNDS 12
#define MAX_TERMS 4096
#define MAX_TARGETS 4096
#define MAX_COSET_DIM 12
#define HASH_SIZE 1048576

typedef uint64_t Mask[WORDS];

typedef struct {
    Mask mask;
    double coefficient;
} Term;

typedef struct {
    uint16_t size;
    Term *terms;
} Polynomial;

typedef struct {
    Mask mask;
    double coefficient;
    uint32_t stamp;
} HashSlot;

typedef struct {
    Term term;
    Mask quotient;
} CosetTerm;

typedef struct {
    Mask quotient;
    size_t start;
    size_t count;
    double squared_norm;
    int is_zero_coset;
} CosetGroup;

static const uint8_t SBOX[N] = {0, 3, 6, 1, 5, 4, 2, 7};

static const uint32_t ROUND_CONSTANTS[MAX_ROUNDS] = {
    0x00000058, 0x00000038, 0x000003c0, 0x000000d0,
    0x00000120, 0x00000014, 0x00000060, 0x0000002c,
    0x00000380, 0x000000f0, 0x000001a0, 0x00000012
};

static double LAT[N][N];
static uint8_t PREPARED[COLUMNS][N][COLUMNS];
static HashSlot HASH_TABLE[HASH_SIZE];
static uint32_t *HASH_USED;
static uint32_t HASH_STAMP = 1;
static size_t HASH_USED_COUNT = 0;

static Polynomial **SBOX_CACHE[MAX_ROUNDS];
static Polynomial **LINEAR_CACHE[MAX_ROUNDS - 1];
static size_t SBOX_NODE_COUNT[MAX_ROUNDS];
static size_t LINEAR_NODE_COUNT[MAX_ROUNDS - 1];
static size_t XOR_PRODUCT_COUNT = 0;
static size_t PRUNED_POLYNOMIAL_COUNT = 0;
static size_t MAX_UNPRUNED_TERM_COUNT = 0;

static int ROUNDS = 4;
static int TERM_CAP = 16;
static uint64_t DIFFERENCE[WORDS];
static uint64_t OUTPUT_MASK[WORDS];
static int ROOT_COLUMN = -1;
static int ROOT_INDEX = 0;
static int TARGET_COUNT = 0;
static Mask TARGET_MASKS[MAX_TARGETS];
static int TARGET_PRESENT[MAX_TARGETS];
static double TARGET_COEFFICIENTS[MAX_TARGETS];
static size_t ROOT_HASH_TERM_COUNT = 0;
static const char *ROOT_DUMP_PATH = NULL;
static int COSET_DIM = 0;
static Mask COSET_BASIS[MAX_COSET_DIM];
static Mask COSET_ROWS[WORDS * 64];
static unsigned char COSET_ROW_PRESENT[WORDS * 64];

static int parity32(uint32_t value) {
    return __builtin_parity(value);
}

static void mask_zero(Mask mask) {
    memset(mask, 0, sizeof(Mask));
}

static int mask_equal(const Mask left, const Mask right) {
    int word;
    for (word = 0; word < WORDS; ++word)
        if (left[word] != right[word]) return 0;
    return 1;
}

static int mask_is_zero(const Mask mask) {
    int word;
    for (word = 0; word < WORDS; ++word)
        if (mask[word] != 0) return 0;
    return 1;
}

static void mask_xor_inplace(Mask left, const Mask right) {
    int word;
    for (word = 0; word < WORDS; ++word) left[word] ^= right[word];
}

static int mask_highest_bit(const Mask mask) {
    int word;
    for (word = WORDS - 1; word >= 0; --word) {
        uint64_t value = mask[word];
        if (value != 0)
            return word * 64 + 63 - __builtin_clzll(value);
    }
    return -1;
}

static int mask_lex_compare(const Mask left, const Mask right) {
    int word;
    for (word = 0; word < WORDS; ++word) {
        if (left[word] < right[word]) return -1;
        if (left[word] > right[word]) return 1;
    }
    return 0;
}

static uint64_t mask_hash(const Mask mask) {
    uint64_t value = UINT64_C(0x9e3779b97f4a7c15);
    int word;
    for (word = 0; word < WORDS; ++word) {
        uint64_t x = mask[word] + UINT64_C(0x9e3779b97f4a7c15)
                   + (value << 6) + (value >> 2);
        value ^= x;
        value ^= value >> 30;
        value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        value *= UINT64_C(0x94d049bb133111eb);
        value ^= value >> 31;
    }
    return value;
}

static void hash_clear(void) {
    ++HASH_STAMP;
    HASH_USED_COUNT = 0;
    if (HASH_STAMP == 0) {
        memset(HASH_TABLE, 0, sizeof(HASH_TABLE));
        HASH_STAMP = 1;
    }
}

static void hash_add(const Mask mask, double coefficient) {
    size_t position;
    if (coefficient == 0.0) return;
    position = (size_t)(mask_hash(mask) & (HASH_SIZE - 1));
    for (;;) {
        HashSlot *slot = &HASH_TABLE[position];
        if (slot->stamp != HASH_STAMP) {
            memcpy(slot->mask, mask, sizeof(Mask));
            slot->coefficient = coefficient;
            slot->stamp = HASH_STAMP;
            HASH_USED[HASH_USED_COUNT++] = (uint32_t)position;
            return;
        }
        if (mask_equal(slot->mask, mask)) {
            slot->coefficient += coefficient;
            return;
        }
        position = (position + 1) & (HASH_SIZE - 1);
    }
}

static int term_compare(const void *left_ptr, const void *right_ptr) {
    const Term *left = (const Term *)left_ptr;
    const Term *right = (const Term *)right_ptr;
    double left_abs = fabs(left->coefficient);
    double right_abs = fabs(right->coefficient);
    if (left_abs > right_abs) return -1;
    if (left_abs < right_abs) return 1;
    return mask_lex_compare(left->mask, right->mask);
}

static int coset_term_compare(const void *left_ptr, const void *right_ptr) {
    const CosetTerm *left = (const CosetTerm *)left_ptr;
    const CosetTerm *right = (const CosetTerm *)right_ptr;
    int order = mask_lex_compare(left->quotient, right->quotient);
    if (order != 0) return order;
    return term_compare(&left->term, &right->term);
}

static int coset_group_compare(const void *left_ptr, const void *right_ptr) {
    const CosetGroup *left = (const CosetGroup *)left_ptr;
    const CosetGroup *right = (const CosetGroup *)right_ptr;
    if (left->is_zero_coset != right->is_zero_coset)
        return right->is_zero_coset - left->is_zero_coset;
    if (left->squared_norm > right->squared_norm) return -1;
    if (left->squared_norm < right->squared_norm) return 1;
    return mask_lex_compare(left->quotient, right->quotient);
}

static void prepare_coset_rows(void) {
    int index;
    memset(COSET_ROWS, 0, sizeof(COSET_ROWS));
    memset(COSET_ROW_PRESENT, 0, sizeof(COSET_ROW_PRESENT));
    for (index = 0; index < COSET_DIM; ++index) {
        Mask value;
        int pivot;
        memcpy(value, COSET_BASIS[index], sizeof(Mask));
        while ((pivot = mask_highest_bit(value)) >= 0) {
            if (!COSET_ROW_PRESENT[pivot]) {
                memcpy(COSET_ROWS[pivot], value, sizeof(Mask));
                COSET_ROW_PRESENT[pivot] = 1;
                break;
            }
            mask_xor_inplace(value, COSET_ROWS[pivot]);
        }
        if (pivot < 0) {
            fprintf(stderr, "dependent coset basis\n");
            exit(2);
        }
    }
}

static void coset_quotient(const Mask mask, Mask quotient) {
    int pivot;
    memcpy(quotient, mask, sizeof(Mask));
    for (pivot = WORDS * 64 - 1; pivot >= 0; --pivot) {
        if (COSET_ROW_PRESENT[pivot] &&
            ((quotient[pivot / 64] >> (pivot % 64)) & UINT64_C(1)))
            mask_xor_inplace(quotient, COSET_ROWS[pivot]);
    }
}

static void polynomial_from_terms_coset(
    Polynomial *output, Term *all_terms, size_t count
) {
    CosetTerm *items = NULL;
    CosetGroup *groups = NULL;
    size_t used = 0, group_count = 0, retained = 0;
    int zero_found = 0;
    Term zero_term;

    if (count != 0) {
        items = (CosetTerm *)malloc(count * sizeof(CosetTerm));
        groups = (CosetGroup *)malloc(count * sizeof(CosetGroup));
        if (items == NULL || groups == NULL) {
            fprintf(stderr, "coset allocation failed\n");
            exit(2);
        }
    }
    for (used = 0; used < count; ++used) {
        items[used].term = all_terms[used];
        coset_quotient(all_terms[used].mask, items[used].quotient);
        if (mask_is_zero(all_terms[used].mask)) {
            zero_found = 1;
            zero_term = all_terms[used];
        }
    }
    qsort(items, count, sizeof(CosetTerm), coset_term_compare);
    used = 0;
    while (used < count) {
        size_t end = used + 1;
        double norm = items[used].term.coefficient * items[used].term.coefficient;
        while (end < count && mask_equal(items[used].quotient, items[end].quotient)) {
            double value = items[end].term.coefficient;
            norm += value * value;
            ++end;
        }
        memcpy(groups[group_count].quotient, items[used].quotient, sizeof(Mask));
        groups[group_count].start = used;
        groups[group_count].count = end - used;
        groups[group_count].squared_norm = norm;
        groups[group_count].is_zero_coset = mask_is_zero(items[used].quotient);
        ++group_count;
        used = end;
    }
    qsort(groups, group_count, sizeof(CosetGroup), coset_group_compare);

    free(output->terms);
    output->size = (uint16_t)(count < (size_t)TERM_CAP ? count : TERM_CAP);
    output->terms = output->size ? (Term *)malloc(output->size * sizeof(Term)) : NULL;
    if (output->size && output->terms == NULL) {
        fprintf(stderr, "coset retain failed\n");
        exit(2);
    }
    for (used = 0; used < group_count && retained < output->size; ++used) {
        size_t item;
        size_t take = groups[used].count;
        if (take > output->size - retained) take = output->size - retained;
        for (item = 0; item < take; ++item)
            output->terms[retained++] = items[groups[used].start + item].term;
    }
    if (zero_found && output->size) {
        int present = 0;
        for (used = 0; used < output->size; ++used)
            if (mask_is_zero(output->terms[used].mask)) present = 1;
        if (!present) output->terms[output->size - 1] = zero_term;
    }
    qsort(output->terms, output->size, sizeof(Term), term_compare);
    free(groups);
    free(items);
}

static void polynomial_from_hash(Polynomial *output) {
    Term *all_terms = NULL;
    size_t used, count = 0;
    int zero_found = 0;
    Term zero_term;

    if (HASH_USED_COUNT != 0) {
        all_terms = (Term *)malloc(HASH_USED_COUNT * sizeof(Term));
        if (all_terms == NULL) {
            fprintf(stderr, "term allocation failed\n");
            exit(2);
        }
    }
    for (used = 0; used < HASH_USED_COUNT; ++used) {
        HashSlot *slot = &HASH_TABLE[HASH_USED[used]];
        if (slot->coefficient == 0.0) continue;
        memcpy(all_terms[count].mask, slot->mask, sizeof(Mask));
        all_terms[count].coefficient = slot->coefficient;
        if (mask_is_zero(slot->mask)) {
            zero_found = 1;
            zero_term = all_terms[count];
        }
        ++count;
    }
    if (count > MAX_UNPRUNED_TERM_COUNT) MAX_UNPRUNED_TERM_COUNT = count;
    if (count > (size_t)TERM_CAP) ++PRUNED_POLYNOMIAL_COUNT;
    if (COSET_DIM > 0) {
        polynomial_from_terms_coset(output, all_terms, count);
        free(all_terms);
        return;
    }
    qsort(all_terms, count, sizeof(Term), term_compare);
    free(output->terms);
    output->size = (uint16_t)(count < (size_t)TERM_CAP ? count : TERM_CAP);
    output->terms = output->size ? (Term *)malloc(output->size * sizeof(Term)) : NULL;
    if (output->size && output->terms == NULL) {
        fprintf(stderr, "retain allocation failed\n");
        exit(2);
    }
    for (used = 0; used < output->size; ++used) output->terms[used] = all_terms[used];
    if (zero_found && output->size) {
        int present = 0;
        for (used = 0; used < output->size; ++used)
            if (mask_is_zero(output->terms[used].mask)) present = 1;
        if (!present) output->terms[output->size - 1] = zero_term;
        qsort(output->terms, output->size, sizeof(Term), term_compare);
    }
    free(all_terms);
}

static void polynomial_one(Polynomial *output) {
    free(output->terms);
    output->terms = (Term *)malloc(sizeof(Term));
    if (output->terms == NULL) exit(2);
    output->size = 1;
    mask_zero(output->terms[0].mask);
    output->terms[0].coefficient = 1.0;
}

static void capture_root_targets(void);
static void dump_root_hash(void);

static void polynomial_multiply(
    const Polynomial *left, const Polynomial *right, Polynomial *output,
    int capture_root
) {
    uint16_t i, j;
    Mask mask;
    ++XOR_PRODUCT_COUNT;
    if (left->size == 0 || right->size == 0) {
        if (capture_root) {
            hash_clear();
            capture_root_targets();
            dump_root_hash();
        }
        free(output->terms);
        output->terms = NULL;
        output->size = 0;
        return;
    }
    hash_clear();
    for (i = 0; i < left->size; ++i) {
        for (j = 0; j < right->size; ++j) {
            int word;
            for (word = 0; word < WORDS; ++word)
                mask[word] = left->terms[i].mask[word] ^ right->terms[j].mask[word];
            hash_add(mask, left->terms[i].coefficient * right->terms[j].coefficient);
        }
    }
    if (capture_root) {
        capture_root_targets();
        dump_root_hash();
    }
    polynomial_from_hash(output);
}

static uint32_t rotl32(uint32_t value, unsigned amount) {
    amount &= 31u;
    if (amount == 0) return value;
    return (value << amount) | (value >> (32u - amount));
}

static void rho_east(const uint32_t input[12], uint32_t output[12]) {
    int x;
    memcpy(output, input, 12 * sizeof(uint32_t));
    for (x = 0; x < 4; ++x) {
        output[4 + x] = rotl32(input[4 + x], 1);
        output[8 + ((x + 2) & 3)] = rotl32(input[8 + x], 8);
    }
}

static void theta(const uint32_t input[12], uint32_t output[12]) {
    uint32_t parity[4], effect[4] = {0, 0, 0, 0};
    int x, y;
    for (x = 0; x < 4; ++x)
        parity[x] = input[x] ^ input[4 + x] ^ input[8 + x];
    for (x = 0; x < 4; ++x) {
        int target = (x + 1) & 3;
        effect[target] ^= rotl32(parity[x], 5) ^ rotl32(parity[x], 14);
    }
    for (y = 0; y < 3; ++y)
        for (x = 0; x < 4; ++x)
            output[4 * y + x] = input[4 * y + x] ^ effect[x];
}

static void rho_west(const uint32_t input[12], uint32_t output[12]) {
    int x;
    memcpy(output, input, 12 * sizeof(uint32_t));
    for (x = 0; x < 4; ++x) {
        output[4 + ((x + 1) & 3)] = input[4 + x];
        output[8 + x] = rotl32(input[8 + x], 11);
    }
}

static void inter_chi_linear(const uint32_t input[12], uint32_t output[12]) {
    uint32_t first[12], second[12];
    rho_east(input, first);
    theta(first, second);
    rho_west(second, output);
}

static void make_prepared_masks(void) {
    int sx, sy, sz, lane;
    memset(PREPARED, 0, sizeof(PREPARED));
    for (sy = 0; sy < 3; ++sy) {
        for (sx = 0; sx < 4; ++sx) {
            for (sz = 0; sz < 32; ++sz) {
                uint32_t input[12] = {0};
                uint32_t output[12];
                int source_column = 32 * sx + sz;
                input[4 * sy + sx] = UINT32_C(1) << sz;
                inter_chi_linear(input, output);
                for (lane = 0; lane < 12; ++lane) {
                    uint32_t value = output[lane];
                    int oy = lane / 4;
                    int ox = lane % 4;
                    while (value) {
                        int oz = __builtin_ctz(value);
                        int output_column = 32 * ox + oz;
                        int mask;
                        value &= value - 1;
                        for (mask = 1; mask < N; ++mask)
                            if ((mask >> oy) & 1)
                                PREPARED[output_column][mask][source_column] ^=
                                    (uint8_t)(1u << sy);
                    }
                }
            }
        }
    }
}

static void make_lat(void) {
    int mask_out, mask_in, value;
    for (mask_out = 0; mask_out < N; ++mask_out) {
        for (mask_in = 0; mask_in < N; ++mask_in) {
            int total = 0;
            for (value = 0; value < N; ++value) {
                int phase = parity32((uint32_t)(mask_in & value)) ^
                            parity32((uint32_t)(mask_out & SBOX[value]));
                total += phase ? -1 : 1;
            }
            LAT[mask_out][mask_in] = (double)total / (double)N;
        }
    }
}

static int state_bit(int x, int y, int z) {
    return z + 32 * (x + 4 * y);
}

static int local_from_words(const uint64_t words[WORDS], int column) {
    int x = column / 32;
    int z = column % 32;
    int y, result = 0;
    for (y = 0; y < 3; ++y) {
        int bit = state_bit(x, y, z);
        result |= (int)((words[bit / 64] >> (bit % 64)) & UINT64_C(1)) << y;
    }
    return result;
}

static void local_character(int column, int local_mask, Mask result) {
    int x = column / 32;
    int z = column % 32;
    int y;
    mask_zero(result);
    for (y = 0; y < 3; ++y) {
        int bit;
        if (((local_mask >> y) & 1) == 0) continue;
        bit = state_bit(x, y, z);
        result[bit / 64] ^= UINT64_C(1) << (bit % 64);
    }
}

static int connector_constant_bit(int round, int column) {
    int start = MAX_ROUNDS - ROUNDS;
    int x, z;
    uint32_t constant;
    if (round <= 0) return 0;
    x = column / 32;
    z = column % 32;
    if (x != 0) return 0;
    constant = ROUND_CONSTANTS[start + round];
    return (int)((constant >> z) & 1u);
}

static Polynomial *allocate_polynomial(void) {
    Polynomial *result = (Polynomial *)malloc(sizeof(Polynomial));
    if (result == NULL) exit(2);
    result->size = 0;
    result->terms = NULL;
    return result;
}

static Polynomial *eval_sbox(int round, int column, int output_index);
static Polynomial *eval_linear(int round, int column, int output_index) {
    size_t flat = (size_t)column * TENSOR + (size_t)output_index;
    Polynomial *cached = LINEAR_CACHE[round][flat];
    Polynomial accumulator = {0, NULL};
    Polynomial temporary = {0, NULL};
    int value_mask, difference_mask, source_column;
    if (cached != NULL) return cached;

    cached = allocate_polynomial();
    LINEAR_CACHE[round][flat] = cached;
    ++LINEAR_NODE_COUNT[round];
    value_mask = output_index / N;
    difference_mask = output_index % N;
    polynomial_one(&accumulator);
    for (source_column = 0; source_column < COLUMNS; ++source_column) {
        int local_value = PREPARED[column][value_mask][source_column];
        int local_difference = PREPARED[column][difference_mask][source_column];
        int input_index = local_value * N + local_difference;
        Polynomial *factor;
        if (input_index == 0) continue;
        factor = eval_sbox(round, source_column, input_index);
        if (accumulator.size != 0) {
            polynomial_multiply(&accumulator, factor, &temporary, 0);
            free(accumulator.terms);
            accumulator = temporary;
            temporary.size = 0;
            temporary.terms = NULL;
        }
    }
    *cached = accumulator;
    return cached;
}

static void eval_first_sbox_leaf(int column, int output_index, Polynomial *output) {
    int value_out = output_index / N;
    int difference_out = output_index % N;
    int mixed_out = value_out ^ difference_out;
    int local_difference = local_from_words(DIFFERENCE, column);
    int difference_in, value_in;
    if (output_index == 0) {
        polynomial_one(output);
        return;
    }
    hash_clear();
    for (difference_in = 0; difference_in < N; ++difference_in) {
        double right = LAT[difference_out][difference_in];
        if (right == 0.0) continue;
        for (value_in = 0; value_in < N; ++value_in) {
            double transition = LAT[mixed_out][difference_in ^ value_in] * right;
            double phase;
            Mask character;
            if (transition == 0.0) continue;
            phase = parity32((uint32_t)(difference_in & local_difference)) ? -1.0 : 1.0;
            local_character(column, value_in, character);
            hash_add(character, transition * phase);
        }
    }
    if (ROUNDS == 1 && column == ROOT_COLUMN && output_index == ROOT_INDEX) {
        capture_root_targets();
        dump_root_hash();
    }
    polynomial_from_hash(output);
}

static void capture_root_targets(void) {
    int target;
    size_t used;
    ROOT_HASH_TERM_COUNT = 0;
    memset(TARGET_PRESENT, 0, sizeof(TARGET_PRESENT));
    memset(TARGET_COEFFICIENTS, 0, sizeof(TARGET_COEFFICIENTS));
    for (used = 0; used < HASH_USED_COUNT; ++used) {
        HashSlot *slot = &HASH_TABLE[HASH_USED[used]];
        if (slot->coefficient == 0.0) continue;
        ++ROOT_HASH_TERM_COUNT;
        for (target = 0; target < TARGET_COUNT; ++target) {
            if (mask_equal(slot->mask, TARGET_MASKS[target])) {
                TARGET_PRESENT[target] = 1;
                TARGET_COEFFICIENTS[target] = slot->coefficient;
            }
        }
    }
}

static void dump_root_hash(void) {
    FILE *stream;
    size_t used;
    uint64_t count = (uint64_t)ROOT_HASH_TERM_COUNT;
    static const unsigned char magic[8] = {'X','U','D','R','T','0','0','1'};
    if (ROOT_DUMP_PATH == NULL || *ROOT_DUMP_PATH == '\0') return;
    stream = fopen(ROOT_DUMP_PATH, "wb");
    if (stream == NULL) {
        fprintf(stderr, "cannot open root dump\n");
        exit(2);
    }
    if (fwrite(magic, 1, 8, stream) != 8 ||
        fwrite(&count, sizeof(count), 1, stream) != 1) exit(2);
    for (used = 0; used < HASH_USED_COUNT; ++used) {
        HashSlot *slot = &HASH_TABLE[HASH_USED[used]];
        if (slot->coefficient == 0.0) continue;
        if (fwrite(&slot->coefficient, sizeof(double), 1, stream) != 1 ||
            fwrite(slot->mask, sizeof(uint64_t), WORDS, stream) != WORDS) exit(2);
    }
    fclose(stream);
}

static Polynomial *eval_sbox(int round, int column, int output_index) {
    size_t flat = (size_t)column * TENSOR + (size_t)output_index;
    Polynomial *cached = SBOX_CACHE[round][flat];
    int value_out, difference_out, mixed_out;
    int difference_in, value_in;
    int rc_bit;
    if (cached != NULL) return cached;
    cached = allocate_polynomial();
    SBOX_CACHE[round][flat] = cached;
    ++SBOX_NODE_COUNT[round];
    if (output_index == 0) {
        polynomial_one(cached);
        return cached;
    }
    if (round == 0) {
        eval_first_sbox_leaf(column, output_index, cached);
        return cached;
    }

    value_out = output_index / N;
    difference_out = output_index % N;
    mixed_out = value_out ^ difference_out;
    rc_bit = connector_constant_bit(round, column);

    /* Complete child discovery before reusing the shared accumulation hash. */
    for (difference_in = 0; difference_in < N; ++difference_in) {
        double right = LAT[difference_out][difference_in];
        if (right == 0.0) continue;
        for (value_in = 0; value_in < N; ++value_in) {
            double transition = LAT[mixed_out][difference_in ^ value_in] * right;
            if (transition != 0.0)
                (void)eval_linear(round - 1, column, value_in * N + difference_in);
        }
    }

    hash_clear();
    for (difference_in = 0; difference_in < N; ++difference_in) {
        double right = LAT[difference_out][difference_in];
        if (right == 0.0) continue;
        for (value_in = 0; value_in < N; ++value_in) {
            double transition = LAT[mixed_out][difference_in ^ value_in] * right;
            Polynomial *source;
            uint16_t term;
            if (transition == 0.0) continue;
            if (rc_bit && (value_in & 1)) transition = -transition;
            source = eval_linear(round - 1, column, value_in * N + difference_in);
            for (term = 0; term < source->size; ++term)
                hash_add(source->terms[term].mask,
                         transition * source->terms[term].coefficient);
        }
    }
    if (round == ROUNDS - 1 && column == ROOT_COLUMN && output_index == ROOT_INDEX) {
        capture_root_targets();
        dump_root_hash();
    }
    polynomial_from_hash(cached);
    return cached;
}

static int read_words(uint64_t words[WORDS]) {
    int word;
    for (word = 0; word < WORDS; ++word)
        if (scanf("%" SCNx64, &words[word]) != 1) return 0;
    return 1;
}

static void free_caches(void) {
    int round;
    size_t index;
    size_t cache_size = (size_t)COLUMNS * TENSOR;
    for (round = 0; round < ROUNDS; ++round) {
        if (SBOX_CACHE[round] == NULL) continue;
        for (index = 0; index < cache_size; ++index) {
            if (SBOX_CACHE[round][index] != NULL) {
                free(SBOX_CACHE[round][index]->terms);
                free(SBOX_CACHE[round][index]);
            }
        }
        free(SBOX_CACHE[round]);
    }
    for (round = 0; round + 1 < ROUNDS; ++round) {
        if (LINEAR_CACHE[round] == NULL) continue;
        for (index = 0; index < cache_size; ++index) {
            if (LINEAR_CACHE[round][index] != NULL) {
                free(LINEAR_CACHE[round][index]->terms);
                free(LINEAR_CACHE[round][index]);
            }
        }
        free(LINEAR_CACHE[round]);
    }
}

int main(void) {
    Polynomial *root;
    Polynomial root_accumulator = {0, NULL};
    Polynomial root_temporary = {0, NULL};
    int round, index, output_column = -1, output_local = 0, output_columns = 0;
    size_t cache_size = (size_t)COLUMNS * TENSOR;
    ROOT_DUMP_PATH = getenv("XOODOO_U_ROOT_DUMP");

    if (scanf("%d%d", &ROUNDS, &TERM_CAP) != 2 ||
        ROUNDS < 1 || ROUNDS > MAX_ROUNDS ||
        TERM_CAP < 1 || TERM_CAP > MAX_TERMS) {
        fprintf(stderr, "invalid header\n");
        return 1;
    }
    if (!read_words(DIFFERENCE) || !read_words(OUTPUT_MASK)) {
        fprintf(stderr, "invalid masks\n");
        return 1;
    }
    if (scanf("%d", &TARGET_COUNT) != 1 || TARGET_COUNT < 0 || TARGET_COUNT > MAX_TARGETS)
        return 1;
    for (index = 0; index < TARGET_COUNT; ++index)
        if (!read_words(TARGET_MASKS[index])) return 1;
    if (scanf("%d", &COSET_DIM) != 1 || COSET_DIM < 0 || COSET_DIM > MAX_COSET_DIM)
        return 1;
    for (index = 0; index < COSET_DIM; ++index)
        if (!read_words(COSET_BASIS[index])) return 1;
    if (COSET_DIM > 0) prepare_coset_rows();

    for (index = 0; index < COLUMNS; ++index) {
        int local = local_from_words(OUTPUT_MASK, index);
        if (local == 0) continue;
        if (output_column < 0) {
            output_column = index;
            output_local = local;
        }
        ++output_columns;
    }
    if (output_column < 0) return 1;
    if (output_columns == 1) {
        ROOT_COLUMN = output_column;
        ROOT_INDEX = output_local; /* root value-mask is zero, difference-mask is lambda */
    }

    HASH_USED = (uint32_t *)malloc(HASH_SIZE * sizeof(uint32_t));
    if (HASH_USED == NULL) return 2;
    make_lat();
    make_prepared_masks();
    for (round = 0; round < ROUNDS; ++round) {
        SBOX_CACHE[round] = (Polynomial **)calloc(cache_size, sizeof(Polynomial *));
        if (SBOX_CACHE[round] == NULL) return 2;
    }
    for (round = 0; round + 1 < ROUNDS; ++round) {
        LINEAR_CACHE[round] = (Polynomial **)calloc(cache_size, sizeof(Polynomial *));
        if (LINEAR_CACHE[round] == NULL) return 2;
    }

    if (output_columns == 1) {
        root = eval_sbox(ROUNDS - 1, output_column, output_local);
    } else {
        int seen = 0;
        polynomial_one(&root_accumulator);
        for (index = 0; index < COLUMNS; ++index) {
            int local = local_from_words(OUTPUT_MASK, index);
            Polynomial *factor;
            if (local == 0) continue;
            factor = eval_sbox(ROUNDS - 1, index, local);
            ++seen;
            polynomial_multiply(
                &root_accumulator, factor, &root_temporary,
                seen == output_columns
            );
            free(root_accumulator.terms);
            root_accumulator = root_temporary;
            root_temporary.size = 0;
            root_temporary.terms = NULL;
        }
        root = &root_accumulator;
    }
    printf("terms %u\n", (unsigned)root->size);
    for (index = 0; index < root->size; ++index) {
        int word;
        printf("%.17g", root->terms[index].coefficient);
        for (word = 0; word < WORDS; ++word)
            printf(" %016" PRIx64, root->terms[index].mask[word]);
        putchar('\n');
    }
    printf("nodes");
    for (round = 0; round < ROUNDS; ++round) printf(" %zu", SBOX_NODE_COUNT[round]);
    for (round = 0; round + 1 < ROUNDS; ++round) printf(" %zu", LINEAR_NODE_COUNT[round]);
    printf(" %zu %zu %zu\n", XOR_PRODUCT_COUNT,
           PRUNED_POLYNOMIAL_COUNT, MAX_UNPRUNED_TERM_COUNT);
    printf("targets %d\n", TARGET_COUNT);
    for (index = 0; index < TARGET_COUNT; ++index) {
        int word;
        printf("%d %.17g", TARGET_PRESENT[index], TARGET_COEFFICIENTS[index]);
        for (word = 0; word < WORDS; ++word)
            printf(" %016" PRIx64, TARGET_MASKS[index][word]);
        putchar('\n');
    }
    printf("root_hash_terms %zu\n", ROOT_HASH_TERM_COUNT);

    if (output_columns > 1) free(root_accumulator.terms);
    free_caches();
    free(HASH_USED);
    return 0;
}
