"""Blink-64 constants and linear operations required by the 27D model."""

SBOX = [
    0x1, 0x0, 0x9, 0x3, 0x8, 0x5, 0xE, 0x7,
    0x4, 0x2, 0xC, 0xB, 0xA, 0xF, 0x6, 0xD,
]

# Paper Section 5.2: [s0,...,s15] <- [sP[0],...,sP[15]].
PBOX = [0, 5, 11, 10, 1, 6, 4, 13, 2, 12, 9, 15, 3, 7, 14, 8]

ROUND_CONSTANTS = [
    0x13198A2E03707344,
    0x082EFA98EC4E6C89,
    0xBE5466CF34E90C6C,
    0x3F84D5B5B5470917,
    0xD1310BA698DFB5AC,
]

INVERSE_ROUND_CONSTANTS = (
    0x0D95748F728EB658,
    0x7B54A41DC25A59B5,
    0xC5D1B023286085F0,
    0x8E79DCB0603A180E,
    0xD71577C1BD314B27,
)

MASTER_KEY_BITS = 448


def int_to_nibbles(value):
    return [(value >> (4 * index)) & 0xF for index in range(16)]


def nibbles_to_int(cells):
    if len(cells) != 16:
        raise ValueError("Blink-64 state must contain 16 nibbles")
    value = 0
    for index, cell in enumerate(cells):
        value |= (int(cell) & 0xF) << (4 * index)
    return value


def mix_columns(state):
    cells = int_to_nibbles(state)
    output = [0] * 16
    for column in range(4):
        column_cells = [cells[column + 4 * row] for row in range(4)]
        for row in range(4):
            value = 0
            for other_row in range(4):
                if other_row != row:
                    value ^= column_cells[other_row]
            output[column + 4 * row] = value
    return nibbles_to_int(output)


def shuffle_cells(state):
    cells = int_to_nibbles(state)
    return nibbles_to_int([cells[PBOX[index]] for index in range(16)])


def inverse_shuffle_cells(state):
    cells = int_to_nibbles(state)
    output = [0] * 16
    for output_index, input_index in enumerate(PBOX):
        output[input_index] = cells[output_index]
    return nibbles_to_int(output)


def difference_distribution_table():
    table = [[0] * 16 for _ in range(16)]
    for input_difference in range(16):
        for value in range(16):
            output_difference = SBOX[value] ^ SBOX[value ^ input_difference]
            table[input_difference][output_difference] += 1
    return table


DDT = difference_distribution_table()
