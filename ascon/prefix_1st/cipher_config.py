import importlib
import os

import numpy


ASCON_SBOX = [
    0x4, 0xB, 0x1F, 0x14, 0x1A, 0x15, 0x9, 0x2,
    0x1B, 0x5, 0x8, 0x12, 0x1D, 0x3, 0x6, 0x1C,
    0x1E, 0x13, 0x7, 0xE, 0x0, 0xD, 0x11, 0x18,
    0x10, 0xC, 0x1, 0x19, 0x16, 0xA, 0xF, 0x17,
]

TARGETS = {
    "d3": ("dl_3_d_2_trails_540_316", "dl_3", "permutation", "ascon", 5),
    "d4": ("dl_2_unrestricted_d_2_trails_500_44", "dl_2_unrestricted", "unrestricted", "ascon128", 5),
    "d5": ("dl_4_d_2_trails_800_389", "dl_4", "permutation", "ascon", 5),
    "d6": ("dl_5_unrestricted_d_2_trails_500_14", "dl_5_unrestricted", "unrestricted", "ascon128a", 6),
    "d7": ("dl_6_d_2_trails_500_14", "dl_6", "permutation", "ascon", 6),
}

target = os.environ.get("ASCON_TARGET", "d3").lower()
if target not in TARGETS:
    raise ValueError(f"unknown ASCON_TARGET: {target}")

route_module, d_str, mode, aead_variant, full_rounds = TARGETS[target]
routes = importlib.import_module(route_module)
diffs = routes.trails

n = 5
m = 5
sbox = ASCON_SBOX
sbox_bits = 5
state_words = 64
state_bits = sbox_bits * state_words
pro_bit = 10
characteristic_model_number = 2
trail_model_number = 2
total_rounds = 2
begin_round = 0

if aead_variant == "ascon":
    cipher_name = "ASCON-p"
    IV = 0
elif aead_variant == "ascon128":
    cipher_name = "ASCON-128"
    IV = 0x80400C0600000000
else:
    cipher_name = "ASCON-128a"
    IV = 0x80800C0800000000

basis_number = int(os.environ.get("ASCON_BASIS_NUMBER", "30"))
weight_range = int(os.environ.get("ASCON_WEIGHT_RANGE", "100"))
dim_truncated = int(os.environ.get("ASCON_DIM_TRUNCATED", "20"))
top_basis_number = int(os.environ.get("ASCON_TOP_BASIS_NUMBER", "20"))

round_constants = [
    0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
    0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2B, 0x1A,
]
ROT = [(19, 28), (61, 39), (1, 6), (10, 17), (7, 41)]


def gen_ascon_linear_matrix_transpose():
    mat = numpy.zeros((320, 320), dtype=numpy.int32)
    for row_idx in range(5):
        for j in range(64):
            row = 64 * row_idx + j
            mat[row, row] = 1
            mat[row, 64 * row_idx + (j + 64 - ROT[row_idx][0]) % 64] = 1
            mat[row, 64 * row_idx + (j + 64 - ROT[row_idx][1]) % 64] = 1
    return mat


linear_mat_transpose = gen_ascon_linear_matrix_transpose()
