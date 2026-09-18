#include <math.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define WORDS 5
#define COLUMNS 64
#define N 32
#define TENSOR (N * N)

typedef uint64_t state_words[WORDS];

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

static long double LAT[N][N];
/* PREPARED[s][v][j] is the local input mask at column j induced by
 * output mask v at column s through the transpose of Ascon's linear layer. */
static uint8_t PREPARED[COLUMNS][N][COLUMNS];

static int parity32(uint32_t value) {
    return __builtin_parity(value);
}

static void make_lat(void) {
    int mask_in, mask_out, x;
    for (mask_in = 0; mask_in < N; ++mask_in) {
        for (mask_out = 0; mask_out < N; ++mask_out) {
            int sum = 0;
            for (x = 0; x < N; ++x) {
                int phase = parity32((uint32_t)(mask_in & x)) ^
                            parity32((uint32_t)(mask_out & SBOX[x]));
                sum += phase ? -1 : 1;
            }
            LAT[mask_out][mask_in] = (long double)sum / (long double)N;
        }
    }
}

static void make_prepared_masks(void) {
    int s, v, word;
    for (s = 0; s < COLUMNS; ++s) {
        for (v = 0; v < N; ++v) {
            for (word = 0; word < WORDS; ++word) {
                int local_bit = 4 - word;
                uint8_t packed_bit;
                int c0, c1, c2;
                if (((v >> local_bit) & 1) == 0) continue;
                packed_bit = (uint8_t)(1u << local_bit);
                c0 = s;
                c1 = (s + COLUMNS - ROT[word][0]) % COLUMNS;
                c2 = (s + COLUMNS - ROT[word][1]) % COLUMNS;
                PREPARED[s][v][c0] ^= packed_bit;
                PREPARED[s][v][c1] ^= packed_bit;
                PREPARED[s][v][c2] ^= packed_bit;
            }
        }
    }
}

static uint8_t local_from_words(const uint64_t words[WORDS], int column) {
    int word;
    uint8_t value = 0;
    for (word = 0; word < WORDS; ++word) {
        value |= (uint8_t)(((words[word] >> (63 - column)) & UINT64_C(1))
                           << (4 - word));
    }
    return value;
}

static int input_value_is_allowed(int domain, uint64_t iv, int column, int x) {
    int iv_bit;
    if (domain == 0) return 1; /* permutation */
    iv_bit = (int)((iv >> (63 - column)) & UINT64_C(1));
    if (((x >> 4) & 1) != iv_bit) return 0;
    if (domain == 2 && (((x >> 1) & 1) != (x & 1))) return 0;
    return 1;
}

static void fwht(long double *data) {
    int len;
    for (len = 1; len < TENSOR; len <<= 1) {
        int i;
        for (i = 0; i < TENSOR; i += 2 * len) {
            int j;
            for (j = i; j < i + len; ++j) {
                long double a = data[j];
                long double b = data[j + len];
                data[j] = a + b;
                data[j + len] = a - b;
            }
        }
    }
}

static void update_full_sbox(const long double *input, long double *output) {
    int s;
    for (s = 0; s < COLUMNS; ++s) {
        long double d[N][N] = {{0.0L}};
        long double dd[N][N] = {{0.0L}};
        const long double *x = input + s * TENSOR;
        long double *y = output + s * TENSOR;
        int t, u1, u0, v0, v1;

        for (t = 0; t < N; ++t) {
            for (u1 = 0; u1 < N; ++u1) {
                long double sum = 0.0L;
                for (u0 = 0; u0 < N; ++u0) {
                    sum += LAT[t][u1 ^ u0] * x[u0 * N + u1];
                }
                d[u1][t] = sum;
            }
        }
        for (t = 0; t < N; ++t) {
            for (v1 = 0; v1 < N; ++v1) {
                long double sum = 0.0L;
                for (u1 = 0; u1 < N; ++u1) {
                    sum += LAT[v1][u1] * d[u1][t];
                }
                dd[t][v1] = sum;
            }
        }
        for (v0 = 0; v0 < N; ++v0) {
            for (v1 = 0; v1 < N; ++v1) {
                y[v0 * N + v1] = dd[v0 ^ v1][v1];
            }
        }
    }
}

static void update_linear(const long double *input, long double *output) {
    int s, v0, v1;
    for (s = 0; s < COLUMNS; ++s) {
        for (v0 = 0; v0 < N; ++v0) {
            for (v1 = 0; v1 < N; ++v1) {
                long double product = 1.0L;
                int j;
                for (j = 0; j < COLUMNS; ++j) {
                    int input_v0 = PREPARED[s][v0][j];
                    int input_v1 = PREPARED[s][v1][j];
                    int index = input_v0 * N + input_v1;
                    product *= input[j * TENSOR + index];
                    if (product == 0.0) break;
                }
                output[s * TENSOR + v0 * N + v1] = product;
            }
        }
    }
}

static void trace_state(const char *label, const long double *state) {
    const char *enabled = getenv("RB_TRACE");
    int column, index;
    int nonzero_entries = 0;
    int nonzero_columns = 0;
    long double maximum = 0.0L;
    long double minimum = 0.0L;
    if (enabled == NULL || *enabled == '\0') return;
    for (column = 0; column < COLUMNS; ++column) {
        int column_nonzero = 0;
        for (index = 0; index < TENSOR; ++index) {
            long double magnitude = fabsl(state[column * TENSOR + index]);
            if (magnitude == 0.0L) continue;
            ++nonzero_entries;
            column_nonzero = 1;
            if (maximum == 0.0L || magnitude > maximum) maximum = magnitude;
            if (minimum == 0.0L || magnitude < minimum) minimum = magnitude;
        }
        nonzero_columns += column_nonzero;
    }
    fprintf(
        stderr,
        "%s: nonzero_columns=%d nonzero_entries=%d min=%0.6Le max=%0.6Le\n",
        label, nonzero_columns, nonzero_entries, minimum, maximum
    );
}

static int state_has_nonzero(const long double *state) {
    size_t index;
    const size_t size = (size_t)COLUMNS * TENSOR;
    for (index = 0; index < size; ++index) {
        if (state[index] != 0.0L) return 1;
    }
    return 0;
}

static long double evaluate_one(int rounds, int begin_round, int domain, uint64_t iv,
                           const uint64_t difference[WORDS],
                           const uint64_t output_mask[WORDS],
                           const uint64_t input_value_mask[WORDS],
                           const uint64_t constraint_mask[WORDS],
                           const uint64_t constraint_value[WORDS]) {
    long double *x;
    long double *y;
    int column, r;
    long double coefficient;

    x = (long double *)calloc((size_t)COLUMNS * TENSOR, sizeof(long double));
    y = (long double *)malloc((size_t)COLUMNS * TENSOR * sizeof(long double));
    if (x == NULL || y == NULL) {
        free(x);
        free(y);
        return NAN;
    }
    for (column = 0; column < COLUMNS; ++column) {
        int local_difference = local_from_words(difference, column);
        int local_input_mask = local_from_words(input_value_mask, column);
        int local_constraint_mask = local_from_words(constraint_mask, column);
        int local_constraint_value = local_from_words(constraint_value, column);
        int value, index;
        int allowed_count = 0;
        long double *component = x + column * TENSOR;
        for (value = 0; value < N; ++value) {
            if (!input_value_is_allowed(domain, iv, column, value)) continue;
            if (((value ^ local_constraint_value) & local_constraint_mask) != 0)
                continue;
            ++allowed_count;
            component[value * N + local_difference] =
                parity32((uint32_t)(local_input_mask & value)) ? -1.0 : 1.0;
        }
        if (allowed_count == 0) {
            free(x);
            free(y);
            return NAN;
        }
        fwht(component);
        for (index = 0; index < TENSOR; ++index)
            component[index] /= (long double)allowed_count;
    }
    trace_state("initial", x);
    for (r = 0; r < rounds; ++r) {
        int rc_index = begin_round + r;
        uint8_t rc;
        if (rc_index < 0 || rc_index >= 15) {
            free(x);
            free(y);
            return NAN;
        }
        rc = ROUND_CONSTANTS[rc_index];
        for (column = 56; column < 64; ++column) {
            int v0, v1;
            if (((rc >> (63 - column)) & 1) == 0) continue;
            for (v0 = 0; v0 < N; ++v0) {
                if (((v0 >> 2) & 1) == 0) continue;
                for (v1 = 0; v1 < N; ++v1) {
                    x[column * TENSOR + v0 * N + v1] *= -1.0;
                }
            }
        }

        update_full_sbox(x, y);
        {
            long double *tmp = x;
            x = y;
            y = tmp;
        }
        {
            char label[32];
            snprintf(label, sizeof(label), "round_%d_sbox", r);
            trace_state(label, x);
        }
        /* The authors' getBias_Permutation omits the final linear layer. */
        if (r + 1 < rounds) {
            update_linear(x, y);
            {
                long double *tmp = x;
                x = y;
                y = tmp;
            }
            {
                char label[32];
                snprintf(label, sizeof(label), "round_%d_linear", r);
                trace_state(label, x);
            }
            if (!state_has_nonzero(x)) {
                free(x);
                free(y);
                return 0.0L;
            }
        }
    }

    coefficient = 1.0L;
    for (column = 0; column < COLUMNS; ++column) {
        int local_output_mask = local_from_words(output_mask, column);
        coefficient *= x[column * TENSOR + local_output_mask];
        if (coefficient == 0.0) break;
    }
    free(x);
    free(y);
    return coefficient;
}

static int read_words(uint64_t words[WORDS]) {
    int i;
    for (i = 0; i < WORDS; ++i) {
        uint64_t value;
        if (scanf("%" SCNx64, &value) != 1) return 0;
        words[i] = value;
    }
    return 1;
}

int main(void) {
    int rounds, begin_round, domain;
    uint64_t iv_value;
    unsigned int count;
    state_words difference;
    state_words output_mask;
    state_words *input_masks;
    long double *results;
    unsigned int i;

    if (scanf("%d%d%d%" SCNx64 "%u", &rounds, &begin_round, &domain,
              &iv_value, &count) != 5) {
        fprintf(stderr, "round_based_engine: invalid request header\n");
        return 1;
    }
    if (domain < 0 || domain > 2) {
        fprintf(stderr, "round_based_engine: invalid input domain\n");
        return 1;
    }
    if (!read_words(difference) || !read_words(output_mask)) {
        fprintf(stderr, "round_based_engine: could not read fixed states\n");
        return 1;
    }
    /* Each request stores (Fourier weight mask, constraint mask, value). */
    input_masks = (state_words *)malloc((size_t)count * 3 * sizeof(state_words));
    results = (long double *)malloc((size_t)count * sizeof(long double));
    if (input_masks == NULL || results == NULL) {
        fprintf(stderr, "round_based_engine: allocation failed\n");
        free(input_masks);
        free(results);
        return 1;
    }
    for (i = 0; i < count; ++i) {
        if (!read_words(input_masks[3 * i]) ||
            !read_words(input_masks[3 * i + 1]) ||
            !read_words(input_masks[3 * i + 2])) {
            fprintf(stderr, "round_based_engine: could not read query %u\n", i);
            free(input_masks);
            free(results);
            return 1;
        }
    }

    make_lat();
    make_prepared_masks();
    for (i = 0; i < count; ++i) {
        results[i] = evaluate_one(
            rounds, begin_round, domain, (uint64_t)iv_value,
            difference, output_mask, input_masks[3 * i],
            input_masks[3 * i + 1], input_masks[3 * i + 2]);
    }
    for (i = 0; i < count; ++i) printf("%.21Lg\n", results[i]);

    free(input_masks);
    free(results);
    return 0;
}
