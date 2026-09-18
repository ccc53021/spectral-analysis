#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Complete reachable V-DAG with bounded U-label maps.
 *
 * No S-box/linear coordinate (V node) reachable from the requested root is
 * removed.  First-round S-box leaves are exact local input-label maps.  Sum
 * nodes add equal U labels; product nodes XOR-convolve them.  TERM_CAP limits
 * only the U map stored at an intermediate DAG node. */

#define WORDS 5
#define COLUMNS 64
#define N 32
#define TENSOR 1024
#define MAX_ROUNDS 15
#define MAX_TERMS 16384
#define MAX_TARGETS 4096
#define MAX_COSET_DIM 20
#define MAX_PROJECTION_DIM 10
#define HASH_SIZE 4194304

typedef uint64_t Mask[WORDS];

typedef struct {
    Mask mask;
    double coefficient;
} Term;

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

typedef struct {
    uint16_t size;
    Term *terms;
} Polynomial;

typedef struct {
    Mask mask;
    double coefficient;
    uint32_t stamp;
} HashSlot;

static const uint8_t SBOX[N] = {
    0x04, 0x0b, 0x1f, 0x14, 0x1a, 0x15, 0x09, 0x02,
    0x1b, 0x05, 0x08, 0x12, 0x1d, 0x03, 0x06, 0x1c,
    0x1e, 0x13, 0x07, 0x0e, 0x00, 0x0d, 0x11, 0x18,
    0x10, 0x0c, 0x01, 0x19, 0x16, 0x0a, 0x0f, 0x17,
};

static const uint8_t ROUND_CONSTANTS[15] = {
    0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87,
    0x78, 0x69, 0x5a, 0x4b, 0x3c, 0x2b, 0x1a,
};

static const int ROT[WORDS][2] = {
    {19, 28}, {61, 39}, {1, 6}, {10, 17}, {7, 41}
};

static double LAT[N][N];
static uint8_t PREPARED[COLUMNS][N][COLUMNS];
static HashSlot HASH_TABLE[HASH_SIZE];
static uint32_t *HASH_USED;
static uint32_t HASH_STAMP = 1;
static size_t HASH_USED_COUNT = 0;

static Polynomial **SBOX_CACHE[MAX_ROUNDS];
static Polynomial **LINEAR_CACHE[MAX_ROUNDS - 1];
static size_t SBOX_NODE_COUNT[MAX_ROUNDS] = {0};
static size_t LINEAR_NODE_COUNT[MAX_ROUNDS - 1] = {0};
static size_t XOR_PRODUCT_COUNT = 0;
static size_t PRUNED_POLYNOMIAL_COUNT = 0;
static size_t MAX_UNPRUNED_TERM_COUNT = 0;

static int TERM_CAP = 16;
static int ROUNDS = 4;
static int BEGIN_ROUND = 0;
static int DOMAIN = 1;
static uint64_t IV = UINT64_C(0x80400c0600000000);
static uint64_t DIFFERENCE[WORDS];
static uint64_t OUTPUT_MASK[WORDS];
static int TARGET_COUNT = 0;
static Mask TARGET_MASKS[MAX_TARGETS];
static double TARGET_COEFFICIENTS[MAX_TARGETS];
static int TARGET_PRESENT[MAX_TARGETS];
static int ROOT_COLUMN = -1;
static int ROOT_INDEX = 0;
static size_t ROOT_HASH_TERM_COUNT = 0;
static const char *ROOT_DUMP_PATH = NULL;
static int COSET_DIM = 0;
static Mask COSET_BASIS[MAX_COSET_DIM];
static Mask COSET_ROWS[WORDS * 64];
static unsigned char COSET_ROW_PRESENT[WORDS * 64];
static int PROJECTION_DIM = 0;
static Mask PROJECTION_ROWS[MAX_PROJECTION_DIM];
static Mask PROJECTION_SIGN;

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

static int mask_dot(const Mask left, const Mask right) {
    int word;
    int parity = 0;
    for (word = 0; word < WORDS; ++word)
        parity ^= __builtin_parityll(left[word] & right[word]);
    return parity;
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

static void mask_xor_inplace(Mask left, const Mask right) {
    int word;
    for (word = 0; word < WORDS; ++word) left[word] ^= right[word];
}

static void prepare_coset_rows(void) {
    int index;
    memset(COSET_ROW_PRESENT, 0, sizeof(COSET_ROW_PRESENT));
    memset(COSET_ROWS, 0, sizeof(COSET_ROWS));
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
            fprintf(stderr, "suffix_label_dag_engine: dependent coset basis\n");
            exit(2);
        }
    }
}

static void coset_quotient(const Mask mask, Mask quotient) {
    int pivot;
    memcpy(quotient, mask, sizeof(Mask));
    for (pivot = WORDS * 64 - 1; pivot >= 0; --pivot)
        if (COSET_ROW_PRESENT[pivot] &&
            ((quotient[pivot / 64] >> (pivot % 64)) & UINT64_C(1)))
            mask_xor_inplace(quotient, COSET_ROWS[pivot]);
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
            int word;
            if (HASH_USED_COUNT >= HASH_SIZE) {
                fprintf(stderr, "suffix_label_dag_engine: hash overflow\n");
                exit(2);
            }
            slot->stamp = HASH_STAMP;
            for (word = 0; word < WORDS; ++word)
                slot->mask[word] = mask[word];
            slot->coefficient = coefficient;
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

static void capture_root_targets(void) {
    int target;
    for (target = 0; target < TARGET_COUNT; ++target) {
        size_t position = (size_t)(mask_hash(TARGET_MASKS[target])
                                   & (HASH_SIZE - 1));
        TARGET_PRESENT[target] = 0;
        TARGET_COEFFICIENTS[target] = 0.0;
        for (;;) {
            HashSlot *slot = &HASH_TABLE[position];
            if (slot->stamp != HASH_STAMP) break;
            if (mask_equal(slot->mask, TARGET_MASKS[target])) {
                TARGET_PRESENT[target] = 1;
                TARGET_COEFFICIENTS[target] = slot->coefficient;
                break;
            }
            position = (position + 1) & (HASH_SIZE - 1);
        }
    }
}

static void dump_root_hash(void) {
    static const unsigned char magic[8] = {
        'S', 'L', 'D', 'R', 'T', '0', '0', '1'
    };
    FILE *stream = NULL;
    uint64_t count64;
    size_t used;

    ROOT_HASH_TERM_COUNT = 0;
    for (used = 0; used < HASH_USED_COUNT; ++used) {
        HashSlot *slot = &HASH_TABLE[HASH_USED[used]];
        if (slot->coefficient != 0.0) ++ROOT_HASH_TERM_COUNT;
    }
    if (ROOT_DUMP_PATH == NULL || ROOT_DUMP_PATH[0] == '\0') return;

    stream = fopen(ROOT_DUMP_PATH, "wb");
    if (stream == NULL) {
        fprintf(stderr, "suffix_label_dag_engine: cannot open root dump\n");
        exit(2);
    }
    count64 = (uint64_t)ROOT_HASH_TERM_COUNT;
    if (fwrite(magic, 1, sizeof(magic), stream) != sizeof(magic) ||
        fwrite(&count64, sizeof(count64), 1, stream) != 1) {
        fprintf(stderr, "suffix_label_dag_engine: root dump header failed\n");
        fclose(stream);
        exit(2);
    }
    for (used = 0; used < HASH_USED_COUNT; ++used) {
        HashSlot *slot = &HASH_TABLE[HASH_USED[used]];
        if (slot->coefficient == 0.0) continue;
        if (fwrite(&slot->coefficient, sizeof(slot->coefficient), 1, stream) != 1 ||
            fwrite(slot->mask, sizeof(uint64_t), WORDS, stream) != WORDS) {
            fprintf(stderr, "suffix_label_dag_engine: root dump write failed\n");
            fclose(stream);
            exit(2);
        }
    }
    if (fclose(stream) != 0) {
        fprintf(stderr, "suffix_label_dag_engine: root dump close failed\n");
        exit(2);
    }
}

static int term_compare(const void *left_ptr, const void *right_ptr) {
    const Term *left = (const Term *)left_ptr;
    const Term *right = (const Term *)right_ptr;
    double left_abs = fabs(left->coefficient);
    double right_abs = fabs(right->coefficient);
    int word;
    if (left_abs > right_abs) return -1;
    if (left_abs < right_abs) return 1;
    for (word = 0; word < WORDS; ++word) {
        if (left->mask[word] < right->mask[word]) return -1;
        if (left->mask[word] > right->mask[word]) return 1;
    }
    return 0;
}

static int mask_lex_compare(const Mask left, const Mask right) {
    int word;
    for (word = 0; word < WORDS; ++word) {
        if (left[word] < right[word]) return -1;
        if (left[word] > right[word]) return 1;
    }
    return 0;
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

static void polynomial_from_terms_coset(
    Polynomial *output, Term *all_terms, size_t count
) {
    CosetTerm *coset_terms;
    CosetGroup *groups;
    size_t used, group_count = 0, retained = 0;
    int zero_position = -1;

    coset_terms = (CosetTerm *)malloc(count * sizeof(CosetTerm));
    groups = (CosetGroup *)malloc(count * sizeof(CosetGroup));
    if ((coset_terms == NULL || groups == NULL) && count != 0) {
        fprintf(stderr, "suffix_label_dag_engine: coset allocation failed\n");
        exit(2);
    }
    for (used = 0; used < count; ++used) {
        coset_terms[used].term = all_terms[used];
        coset_quotient(all_terms[used].mask, coset_terms[used].quotient);
        if (mask_is_zero(all_terms[used].mask)) zero_position = (int)used;
    }
    qsort(coset_terms, count, sizeof(CosetTerm), coset_term_compare);
    used = 0;
    while (used < count) {
        size_t end = used + 1;
        double squared_norm =
            coset_terms[used].term.coefficient *
            coset_terms[used].term.coefficient;
        while (end < count &&
               mask_equal(coset_terms[used].quotient,
                          coset_terms[end].quotient)) {
            double coefficient = coset_terms[end].term.coefficient;
            squared_norm += coefficient * coefficient;
            ++end;
        }
        memcpy(groups[group_count].quotient,
               coset_terms[used].quotient, sizeof(Mask));
        groups[group_count].start = used;
        groups[group_count].count = end - used;
        groups[group_count].squared_norm = squared_norm;
        groups[group_count].is_zero_coset =
            mask_is_zero(coset_terms[used].quotient);
        ++group_count;
        used = end;
    }
    qsort(groups, group_count, sizeof(CosetGroup), coset_group_compare);

    free(output->terms);
    output->terms = NULL;
    output->size = (uint16_t)(count < (size_t)TERM_CAP ? count : TERM_CAP);
    if (output->size != 0) {
        output->terms = (Term *)malloc((size_t)output->size * sizeof(Term));
        if (output->terms == NULL) {
            fprintf(stderr, "suffix_label_dag_engine: coset retain failed\n");
            exit(2);
        }
    }
    for (used = 0; used < group_count && retained < output->size; ++used) {
        size_t item;
        size_t take = groups[used].count;
        if (take > (size_t)output->size - retained)
            take = (size_t)output->size - retained;
        for (item = 0; item < take; ++item)
            output->terms[retained++] =
                coset_terms[groups[used].start + item].term;
    }

    /* Retain the literal zero label even if its coefficient is locally weak. */
    if (zero_position >= 0 && output->size > 0) {
        int present = 0;
        Term zero_term = all_terms[zero_position];
        for (used = 0; used < output->size; ++used)
            if (mask_is_zero(output->terms[used].mask)) present = 1;
        if (!present) output->terms[output->size - 1] = zero_term;
    }
    qsort(output->terms, output->size, sizeof(Term), term_compare);
    free(groups);
    free(coset_terms);
}

static void polynomial_from_hash(Polynomial *output) {
    size_t used;
    int zero_position = -1;
    Term *all_terms;
    size_t count = 0;

    all_terms = (Term *)malloc(HASH_USED_COUNT * sizeof(Term));
    if (all_terms == NULL && HASH_USED_COUNT != 0) {
        fprintf(stderr, "suffix_label_dag_engine: term allocation failed\n");
        exit(2);
    }
    for (used = 0; used < HASH_USED_COUNT; ++used) {
        HashSlot *slot = &HASH_TABLE[HASH_USED[used]];
        if (slot->coefficient == 0.0) continue;
        memcpy(all_terms[count].mask, slot->mask, sizeof(Mask));
        all_terms[count].coefficient = slot->coefficient;
        if (mask_is_zero(slot->mask)) zero_position = (int)count;
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
    output->terms = NULL;
    output->size = (uint16_t)(count < (size_t)TERM_CAP ? count : TERM_CAP);
    if (output->size != 0) {
        output->terms = (Term *)malloc((size_t)output->size * sizeof(Term));
        if (output->terms == NULL) {
            fprintf(stderr, "suffix_label_dag_engine: retained term allocation failed\n");
            exit(2);
        }
    }
    for (used = 0; used < output->size; ++used)
        output->terms[used] = all_terms[used];

    /* Retain the zero-label channel even when it is locally small. */
    if (zero_position >= 0) {
        int present = 0;
        size_t index;
        Term zero_term;
        for (index = 0; index < count; ++index) {
            if (mask_is_zero(all_terms[index].mask)) {
                zero_term = all_terms[index];
                break;
            }
        }
        for (index = 0; index < output->size; ++index)
            if (mask_is_zero(output->terms[index].mask)) present = 1;
        if (!present && output->size > 0)
            output->terms[output->size - 1] = zero_term;
        qsort(output->terms, output->size, sizeof(Term), term_compare);
    }
    free(all_terms);
}

static void polynomial_one(Polynomial *output) {
    free(output->terms);
    output->terms = (Term *)malloc(sizeof(Term));
    if (output->terms == NULL) {
        fprintf(stderr, "suffix_label_dag_engine: unit allocation failed\n");
        exit(2);
    }
    output->size = 1;
    mask_zero(output->terms[0].mask);
    output->terms[0].coefficient = 1.0;
}

static void polynomial_multiply(
    const Polynomial *left,
    const Polynomial *right,
    Polynomial *output
) {
    uint16_t i, j;
    Mask mask;
    ++XOR_PRODUCT_COUNT;
    if (left->size == 0 || right->size == 0) {
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
                mask[word] = left->terms[i].mask[word]
                           ^ right->terms[j].mask[word];
            hash_add(
                mask,
                left->terms[i].coefficient * right->terms[j].coefficient
            );
        }
    }
    polynomial_from_hash(output);
}

static void make_lat(void) {
    int mask_out, mask_in, value;
    for (mask_out = 0; mask_out < N; ++mask_out) {
        for (mask_in = 0; mask_in < N; ++mask_in) {
            int total = 0;
            for (value = 0; value < N; ++value) {
                int phase = parity32((uint32_t)(mask_in & value))
                          ^ parity32((uint32_t)(mask_out & SBOX[value]));
                total += phase ? -1 : 1;
            }
            LAT[mask_out][mask_in] = (double)total / (double)N;
        }
    }
}

static void make_prepared_masks(void) {
    int column, local, word;
    for (column = 0; column < COLUMNS; ++column) {
        for (local = 0; local < N; ++local) {
            for (word = 0; word < WORDS; ++word) {
                int bit = 4 - word;
                uint8_t packed;
                int c0, c1, c2;
                if (((local >> bit) & 1) == 0) continue;
                packed = (uint8_t)(1u << bit);
                c0 = column;
                c1 = (column + COLUMNS - ROT[word][0]) % COLUMNS;
                c2 = (column + COLUMNS - ROT[word][1]) % COLUMNS;
                PREPARED[column][local][c0] ^= packed;
                PREPARED[column][local][c1] ^= packed;
                PREPARED[column][local][c2] ^= packed;
            }
        }
    }
}

static int local_from_words(const uint64_t words[WORDS], int column) {
    int word;
    int value = 0;
    for (word = 0; word < WORDS; ++word)
        value |= (int)(((words[word] >> (63 - column)) & UINT64_C(1))
                       << (4 - word));
    return value;
}

static void canonical_input_character(
    int column,
    int local_value_mask,
    Mask global_mask,
    double *phase
) {
    uint64_t physical_bit = UINT64_C(1) << (63 - column);
    int row;
    mask_zero(global_mask);
    *phase = 1.0;
    if (DOMAIN != 0 && ((local_value_mask >> 4) & 1)) {
        if ((IV >> (63 - column)) & UINT64_C(1)) *phase = -*phase;
    }
    for (row = 0; row < WORDS; ++row) {
        int local_bit = 4 - row;
        if (((local_value_mask >> local_bit) & 1) == 0) continue;
        if (DOMAIN != 0 && row == 0) continue;
        if (DOMAIN == 2 && row == 4)
            global_mask[3] ^= physical_bit;
        else
            global_mask[row] ^= physical_bit;
    }
    if (PROJECTION_DIM > 0) {
        Mask physical_mask;
        int bit;
        memcpy(physical_mask, global_mask, sizeof(Mask));
        if (mask_dot(physical_mask, PROJECTION_SIGN)) *phase = -*phase;
        mask_zero(global_mask);
        for (bit = 0; bit < PROJECTION_DIM; ++bit)
            if (mask_dot(physical_mask, PROJECTION_ROWS[bit]))
                global_mask[0] ^= UINT64_C(1) << bit;
    }
}

static Polynomial *allocate_polynomial(void) {
    Polynomial *result = (Polynomial *)malloc(sizeof(Polynomial));
    if (result == NULL) {
        fprintf(stderr, "suffix_label_dag_engine: polynomial allocation failed\n");
        exit(2);
    }
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
        if (input_index == 0) continue; /* S-box coordinate (0,0) is one. */
        factor = eval_sbox(round, source_column, input_index);
        /* Always visit the factor so V-DAG coverage is independent of the
         * bounded U map.  Once the accumulator is empty, later products stay
         * empty and need no additional XOR convolution. */
        if (accumulator.size != 0) {
            polynomial_multiply(&accumulator, factor, &temporary);
            free(accumulator.terms);
            accumulator = temporary;
            temporary.size = 0;
            temporary.terms = NULL;
        }
    }
    free(cached->terms);
    *cached = accumulator;
    return cached;
}

static void eval_first_sbox_leaf(
    int column, int output_index, Polynomial *output
) {
    int value_out = output_index / N;
    int difference_out = output_index % N;
    int mixed_out = value_out ^ difference_out;
    int local_difference = local_from_words(DIFFERENCE, column);
    int rc_bit = 0;
    int difference_in, value_in;
    uint8_t rc = ROUND_CONSTANTS[BEGIN_ROUND];
    if (column >= 56) rc_bit = (rc >> (63 - column)) & 1;
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
            if (rc_bit && (value_in & 0x04)) transition = -transition;
            canonical_input_character(column, value_in, character, &phase);
            if (parity32((uint32_t)(difference_in & local_difference)))
                phase = -phase;
            hash_add(character, transition * phase);
        }
    }
    polynomial_from_hash(output);
}

static Polynomial *eval_sbox(int round, int column, int output_index) {
    size_t flat = (size_t)column * TENSOR + (size_t)output_index;
    Polynomial *cached = SBOX_CACHE[round][flat];
    int value_out, difference_out, mixed_out;
    int difference_in, value_in;
    int rc_bit = 0;
    uint8_t rc;
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
    rc = ROUND_CONSTANTS[BEGIN_ROUND + round];
    if (column >= 56) rc_bit = (rc >> (63 - column)) & 1;

    /* Resolve every child before opening the shared accumulation hash. */
    for (difference_in = 0; difference_in < N; ++difference_in) {
        double right = LAT[difference_out][difference_in];
        if (right == 0.0) continue;
        for (value_in = 0; value_in < N; ++value_in) {
            double transition = LAT[mixed_out][difference_in ^ value_in] * right;
            if (transition != 0.0)
                (void)eval_linear(
                    round - 1, column, value_in * N + difference_in
                );
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
            if (rc_bit && (value_in & 0x04)) transition = -transition;
            source = eval_linear(
                round - 1, column, value_in * N + difference_in
            );
            for (term = 0; term < source->size; ++term)
                hash_add(
                    source->terms[term].mask,
                    transition * source->terms[term].coefficient
                );
        }
    }
    if (round == ROUNDS - 1 && column == ROOT_COLUMN &&
        output_index == ROOT_INDEX) {
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
    for (round = 0; round < ROUNDS; ++round) {
        if (SBOX_CACHE[round] == NULL) continue;
        for (index = 0; index < (size_t)COLUMNS * TENSOR; ++index) {
            if (SBOX_CACHE[round][index] != NULL)
                free(SBOX_CACHE[round][index]->terms);
            free(SBOX_CACHE[round][index]);
        }
        free(SBOX_CACHE[round]);
    }
    for (round = 0; round + 1 < ROUNDS; ++round) {
        if (LINEAR_CACHE[round] == NULL) continue;
        for (index = 0; index < (size_t)COLUMNS * TENSOR; ++index) {
            if (LINEAR_CACHE[round][index] != NULL)
                free(LINEAR_CACHE[round][index]->terms);
            free(LINEAR_CACHE[round][index]);
        }
        free(LINEAR_CACHE[round]);
    }
}

int main(void) {
    Polynomial *root;
    int round, column, output_column = -1, output_lambda = 0;
    size_t cache_size = (size_t)COLUMNS * TENSOR;

    ROOT_DUMP_PATH = getenv("SUFFIX_LABEL_ROOT_DUMP");
    if (scanf("%d%d%d%" SCNx64 "%d",
              &ROUNDS, &BEGIN_ROUND, &DOMAIN, &IV, &TERM_CAP) != 5) {
        fprintf(stderr, "suffix_label_dag_engine: invalid header\n");
        return 1;
    }
    if (ROUNDS < 1 || ROUNDS > MAX_ROUNDS || BEGIN_ROUND < 0 ||
        BEGIN_ROUND + ROUNDS > 15 || DOMAIN < 0 || DOMAIN > 2 ||
        TERM_CAP < 1 || TERM_CAP > MAX_TERMS) {
        fprintf(stderr, "suffix_label_dag_engine: invalid configuration\n");
        return 1;
    }
    if (!read_words(DIFFERENCE) || !read_words(OUTPUT_MASK)) {
        fprintf(stderr, "suffix_label_dag_engine: invalid masks\n");
        return 1;
    }
    if (scanf("%d", &TARGET_COUNT) != 1 ||
        TARGET_COUNT < 0 || TARGET_COUNT > MAX_TARGETS) {
        fprintf(stderr, "suffix_label_dag_engine: invalid target count\n");
        return 1;
    }
    for (column = 0; column < TARGET_COUNT; ++column) {
        if (!read_words(TARGET_MASKS[column])) {
            fprintf(stderr, "suffix_label_dag_engine: invalid target mask\n");
            return 1;
        }
    }
    if (scanf("%d", &COSET_DIM) != 1 ||
        COSET_DIM < 0 || COSET_DIM > MAX_COSET_DIM) {
        fprintf(stderr, "suffix_label_dag_engine: invalid coset dimension\n");
        return 1;
    }
    for (column = 0; column < COSET_DIM; ++column) {
        if (!read_words(COSET_BASIS[column])) {
            fprintf(stderr, "suffix_label_dag_engine: invalid coset basis\n");
            return 1;
        }
    }
    if (COSET_DIM > 0) prepare_coset_rows();
    if (scanf("%d", &PROJECTION_DIM) == 1) {
        if (PROJECTION_DIM < 0 || PROJECTION_DIM > MAX_PROJECTION_DIM ||
            (PROJECTION_DIM > 0 && COSET_DIM > 0) ||
            (PROJECTION_DIM > 0 &&
             TERM_CAP < (1 << PROJECTION_DIM))) {
            fprintf(stderr, "suffix_label_dag_engine: invalid projection configuration\n");
            return 1;
        }
        for (column = 0; column < PROJECTION_DIM; ++column) {
            if (!read_words(PROJECTION_ROWS[column])) {
                fprintf(stderr, "suffix_label_dag_engine: invalid projection row\n");
                return 1;
            }
        }
        if (PROJECTION_DIM > 0 && !read_words(PROJECTION_SIGN)) {
            fprintf(stderr, "suffix_label_dag_engine: invalid projection sign\n");
            return 1;
        }
    } else {
        PROJECTION_DIM = 0;
    }
    for (column = 0; column < COLUMNS; ++column) {
        int local = local_from_words(OUTPUT_MASK, column);
        if (!local) continue;
        if (output_column >= 0) {
            fprintf(stderr, "suffix_label_dag_engine: output must have one active column\n");
            return 1;
        }
        output_column = column;
        output_lambda = local;
    }
    if (output_column < 0) {
        fprintf(stderr, "suffix_label_dag_engine: empty output mask\n");
        return 1;
    }
    ROOT_COLUMN = output_column;
    ROOT_INDEX = output_lambda;

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

    root = eval_sbox(ROUNDS - 1, output_column, output_lambda);
    printf("terms %u\n", (unsigned)root->size);
    for (column = 0; column < root->size; ++column) {
        int word;
        printf("%.17g", root->terms[column].coefficient);
        for (word = 0; word < WORDS; ++word)
            printf(" %016" PRIx64, root->terms[column].mask[word]);
        putchar('\n');
    }
    printf("nodes");
    for (round = 0; round < ROUNDS; ++round)
        printf(" %zu", SBOX_NODE_COUNT[round]);
    for (round = 0; round + 1 < ROUNDS; ++round)
        printf(" %zu", LINEAR_NODE_COUNT[round]);
    printf(" %zu %zu %zu\n", XOR_PRODUCT_COUNT,
           PRUNED_POLYNOMIAL_COUNT, MAX_UNPRUNED_TERM_COUNT);
    printf("targets %d\n", TARGET_COUNT);
    for (column = 0; column < TARGET_COUNT; ++column) {
        int word;
        printf("%d %.17g", TARGET_PRESENT[column],
               TARGET_COEFFICIENTS[column]);
        for (word = 0; word < WORDS; ++word)
            printf(" %016" PRIx64, TARGET_MASKS[column][word]);
        putchar('\n');
    }
    printf("root_hash_terms %zu\n", ROOT_HASH_TERM_COUNT);

    free_caches();
    free(HASH_USED);
    return 0;
}
