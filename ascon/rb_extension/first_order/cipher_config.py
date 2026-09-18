"""Configuration for the round-based input-value analysis.

The default difference/mask pair is Distinguisher I from Table 2 of
ePrint 2026/358.  v4 evaluates that pair under the requested experimental
variant: four rounds of Ascon-128 initialization, no DDT prefix, and the
first-order non-zero (value-mask preserving) round-based model.  This is not
the first-two-round optimized evaluation used for the paper's Table 2 value.

The five 64-bit words are ordered as (x0, x1, x2, x3, x4).  Bit 63 of a
word belongs to S-box column 0, exactly as in the authors' Ascon code.
"""

ROUNDS = 4
BEGIN_ROUND = 0

# Supported values:
#   "permutation"     - all 320 input-value bits are uniform;
#   "ascon128"        - x0 is the fixed IV, x1..x4 are uniform;
#   "ascon128_equal"  - ascon128 plus x3 == x4 in every S-box column.
INPUT_DOMAIN = "ascon128"
IV = 0x80400C0600000000

# Table 2, DL Distinguisher I.
INPUT_DIFFERENCE_WORDS = (
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x8000000000000000,
    0x8000000000000000,
)

OUTPUT_MASK_WORDS = (
    0x0000000000000200,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
    0x0000000000000000,
)

# First-version candidate domain: every non-zero value mask supported on one
# input S-box column.  For Ascon-128, x0 is the fixed IV and the four free bits
# (K0, K1, N0, N1) give 15 candidates per column, hence 64 * 15 = 960.
CANDIDATE_MODE = "single_column"

# Candidates are sorted by W(U) = -log2(abs(C_hat(U))).  Independent masks are
# accepted greedily until this limit is reached.  If fewer candidates have a
# non-zero coefficient, the actual dimension is smaller.
MAX_BASIS_DIM = 8

# Separate engine processes are used because this workspace's MinGW runtime
# has no OpenMP pthread library.  Set to 1 for deterministic single-process
# profiling.
ENGINE_WORKERS = 8

# Optional empirical verification (python monte_carlo.py).  Increase to 24 or
# above for publication-quality confidence intervals.
MONTE_CARLO_SAMPLE_LOG2 = 20
MONTE_CARLO_SEED = 358

# Numerical values below this threshold are printed as zero.  Raw values are
# always retained in the JSON records.
DISPLAY_EPSILON = 1.0e-15

ROUND_CONSTANTS = (
    0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
    0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2B, 0x1A,
)

ROT = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))
