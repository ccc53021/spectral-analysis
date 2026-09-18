"""Blink-128 constants and linear operations used by the Figure 7 model.

The state is stored nibble-wise in a Python integer.  Cell ``column + 8*row``
is row ``row`` and column ``column`` of the 4-by-8 state drawn in CGZ+26.
This agrees with the little-endian cell convention of the official software
implementation.
"""

SBOX = (
    0x1, 0x0, 0x9, 0x3, 0x8, 0x5, 0xE, 0x7,
    0x4, 0x2, 0xC, 0xB, 0xA, 0xF, 0x6, 0xD,
)

# Official Blink-128 cell permutation:
# [s0,...,s31] <- [sP[0],...,sP[31]].
PBOX = (
    5, 12, 4, 1, 17, 9, 10, 16,
    28, 14, 21, 22, 11, 27, 8, 13,
    2, 25, 18, 3, 30, 6, 19, 20,
    0, 23, 24, 31, 7, 15, 29, 26,
)

# Six forward constants of the 16-round, 1024-bit-key Blink-128 variant.
# Bytes in the official C++ implementation are little endian.
ROUND_CONSTANTS = (
    0x243F6A8885A308D313198A2E03707344,
    0xA4093822299F31D0082EFA98EC4E6C89,
    0x452821E638D01377BE5466CF34E90C6C,
    0xC0AC29B7C97C50DD3F84D5B5B5470917,
    0x9216D5D98979FB1BD1310BA698DFB5AC,
    0x2FFD72DBD01ADFB7B8E1AFED6A267E96,
)

INVERSE_ROUND_CONSTANTS = (
    0xA458FEA3F4933D7E0D95748F728EB658,
    0x718BCD5882154AEE7B54A41DC25A59B5,
    0x9C30D5392AF26013C5D1B023286085F0,
    0xCA417918B8DB38EF8E79DCB0603A180E,
    0x6C9E0E8BB01E8A3ED71577C1BD314B27,
    0x78AF2FDA55605C60E65525F3AA55AB94,
)

STATE_ROWS = 4
STATE_COLUMNS = 8
STATE_NIBBLES = STATE_ROWS * STATE_COLUMNS
STATE_BITS = 4 * STATE_NIBBLES


def int_to_nibbles(value):
    if value < 0 or value >> STATE_BITS:
        raise ValueError("Blink-128 state must be a 128-bit nonnegative integer")
    return [(value >> (4 * index)) & 0xF for index in range(STATE_NIBBLES)]


def nibbles_to_int(cells):
    if len(cells) != STATE_NIBBLES:
        raise ValueError("Blink-128 state must contain 32 nibbles")
    value = 0
    for index, cell in enumerate(cells):
        value |= (int(cell) & 0xF) << (4 * index)
    return value


def mix_columns(state):
    cells = int_to_nibbles(state)
    output = [0] * STATE_NIBBLES
    for column in range(STATE_COLUMNS):
        column_cells = [
            cells[column + STATE_COLUMNS * row] for row in range(STATE_ROWS)
        ]
        for row in range(STATE_ROWS):
            value = 0
            for other_row in range(STATE_ROWS):
                if other_row != row:
                    value ^= column_cells[other_row]
            output[column + STATE_COLUMNS * row] = value
    return nibbles_to_int(output)


def shuffle_cells(state):
    cells = int_to_nibbles(state)
    return nibbles_to_int([cells[PBOX[index]] for index in range(STATE_NIBBLES)])


def inverse_shuffle_cells(state):
    cells = int_to_nibbles(state)
    output = [0] * STATE_NIBBLES
    for output_index, input_index in enumerate(PBOX):
        output[input_index] = cells[output_index]
    return nibbles_to_int(output)


def difference_distribution_table():
    table = [[0] * 16 for _ in range(16)]
    for input_difference in range(16):
        for value in range(16):
            difference = SBOX[value] ^ SBOX[value ^ input_difference]
            table[input_difference][difference] += 1
    return table


DDT = difference_distribution_table()

