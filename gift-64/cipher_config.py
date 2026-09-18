import os

from differentials import (
    r_9_characteristic_0,
    r_10_characteristic_0,
    r_12_characteristic_0,
    r_12_characteristic_1,
    r_12_characteristic_2,
    r_12_characteristic_3,
    r_13_characteristic_0,
    r_13_characteristic_1,
    r_13_characteristic_2,
    r_13_characteristic_3,
    r_18_characteristic_0,
)
from r_18_differential_0_117 import r_18_differential_0_117


GIFT_SBOX = [0x1, 0xa, 0x4, 0xc, 0x6, 0xf, 0x3, 0x9, 0x2, 0xd, 0xb, 0x7, 0x5, 0x0, 0x8, 0xe]

cipher_name = "GIFT-64"
n = 4
m = 4
sbox = GIFT_SBOX
state_bits = 64
state_words = 16
sbox_bits = 4
key_bits = 128
pro_bit = 15

TARGETS = {
    "d6": (r_9_characteristic_0, 9, "dc_0", 30),
    "d7": (r_10_characteristic_0, 10, "dc_0", 30),
    "d8": (r_12_characteristic_0, 12, "dc_0", 30),
    "d9": (r_12_characteristic_1, 12, "dc_1", 30),
    "d10": (r_12_characteristic_2, 12, "dc_2", 30),
    "d11": (r_12_characteristic_3, 12, "dc_3", 30),
    "d12": (r_13_characteristic_0, 13, "dc_0", 30),
    "d13": (r_13_characteristic_1, 13, "dc_1", 30),
    "d14": (r_13_characteristic_2, 13, "dc_2", 30),
    "d15": (r_13_characteristic_3, 13, "dc_3", 30),
    "d16": (r_18_characteristic_0, 18, "dc_0", 30),
    "d17": (r_18_differential_0_117, 18, "d_0_117", 20),
}

target_name = os.environ.get("GIFT_TARGET", "d6").lower()
if target_name not in TARGETS:
    raise ValueError(f"Unknown GIFT_TARGET: {target_name}")
diffs, total_rounds, d_str, default_basis = TARGETS[target_name]

analysis_characteristic_indices = [0, 1, 2, 3] if target_name == "d17" else [0]

characteristic_model_number = (len(diffs[0]) - 1) // total_rounds

begin_round = 0

basis_number = int(os.environ.get("GIFT_BASIS_NUMBER", str(default_basis)))
weight_range = int(os.environ.get("GIFT_WEIGHT_RANGE", "100"))

dim_truncated = int(os.environ.get("GIFT_DIM_TRUNCATED", str(default_basis)))
# Sun26a compares right-key spaces after averaging over plaintext values, so
# the direct paper-comparison run must keep the initial x Fourier mask at zero.
# Use input_x=True only as a supplementary joint (key, x) analysis.
input_x = os.environ.get("GIFT_INPUT_X", "False").lower() == "true"


trail_model_number = 2


round_constants = [0x00, 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38]

permutation_bits_table_64 = [
    0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63,
    12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59,
    8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55,
    4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51
]

permutation_bits_table_k = [32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17]

def precompute_round_bit_map(table, rounds):
    n = len(table)
    round_bit_map = [list(range(n))]   # round 0: identity

    for r in range(rounds):
        prev_map = round_bit_map[-1]
        curr_map = [prev_map[table[i]] for i in range(n)]
        round_bit_map.append(curr_map)

    return round_bit_map

master_round_key_table = precompute_round_bit_map(permutation_bits_table_k, total_rounds + begin_round)

def precompute_master_key_bit_position():
    pos_map = []
    for r in range(total_rounds):
        mp = {}
        row = master_round_key_table[r + begin_round]
        for pos, bit in enumerate(row):
            mp[bit] = pos
        pos_map.append(mp)
    return pos_map

master_key_bit_pos_map = precompute_master_key_bit_position()
