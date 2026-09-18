import os


RECTANGLE_SBOX = [0x6, 0x5, 0xc, 0xa, 0x1, 0xe, 0x7, 0x9, 0xb, 0x0, 0x3, 0xd, 0x8, 0xf, 0x4, 0x2]

cipher_name = "RECTANGLE"
n = 4
m = 4
sbox = RECTANGLE_SBOX
state_bits = 64
state_words = 16
sbox_bits = 4
pro_bit = 15

# Tables 5 and 6 of 2026-1162. Analyze ONE characteristic per run.
# To select Table 6, change characteristic_index to 1 (also changes filenames).
from differentials import r_14_characteristic_0, r_14_characteristic_1
characteristic_index = int(os.environ.get("RECTANGLE_CHARACTERISTIC", "0"))
if characteristic_index not in (0, 1):
    raise ValueError("RECTANGLE_CHARACTERISTIC must be 0 or 1")
diffs = [r_14_characteristic_0, r_14_characteristic_1][characteristic_index]
d_str = f"dc_{characteristic_index}"

characteristic_model_number = 3
trail_model_number = 2
total_rounds = 14
begin_round = 0

# Independent expanded keys K_0, ..., K_13; no master-key schedule.
# A final whitening key has zero output mask and does not affect the probability.
key_model = "expanded"
round_key_bits = state_bits
key_bits = total_rounds * round_key_bits

basis_number = int(os.environ.get("RECTANGLE_BASIS_NUMBER", "30"))
weight_range = int(os.environ.get("RECTANGLE_WEIGHT_RANGE", "100"))
# Keep the existing span truncation parameter. Set to 30 for the full 2^30 span.
dim_truncated = int(os.environ.get("RECTANGLE_DIM_TRUNCATED", "30"))
input_x = os.environ.get("RECTANGLE_INPUT_X", "False").lower() == "true"

# RECTANGLE has NO data-path round constants: its constants enter key expansion.
# This list describes S-box-input data-path constants, NOT key-schedule RCs.
# In particular, do not reuse GIFT's shifted constants or its fixed high bit.
round_constants = [0] * total_rounds

# Keep the framework's nibble-packed, LSB-first numbering:
# bit 4*j+b = row b, column j. ShiftRow rotates rows left by 0, 1, 12, 13.
# The existing permutation routine uses output[i] = input[table[i]] (gather).
shift_rows = (0, 1, 12, 13)
permutation_bits_table_64 = [
    4 * ((column - shift_rows[row]) % 16) + row
    for column in range(16) for row in range(4)
]


def key_bit_name(index):
    """Name a flattened expanded-key bit without mapping to a master key."""
    if not 0 <= index < key_bits:
        raise ValueError(f"Expanded-key bit index out of range: {index}")
    local_round, bit = divmod(index, round_key_bits)
    return f"k{begin_round + local_round}_{bit}"
