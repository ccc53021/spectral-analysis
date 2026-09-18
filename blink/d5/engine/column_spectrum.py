"""Exact 16-bit column spectra shared by every Blink-128 Superbox.

Blink-128 has eight columns, but each column still contains four 4-bit S-boxes.
Consequently all expensive local transforms remain of length ``2^16``.  The
global code composes these column objects symbolically; this module never
constructs a Cartesian product of state-wide input masks.
"""

from functools import lru_cache
from itertools import product

import numpy

from .figure7_trail import G_VALUES
from .gf2 import LinearCoordinates, ordered_basis
from .linear_layer import SBOX, STATE_COLUMNS, STATE_ROWS


NIBBLE_BITS = 4
COLUMN_BITS = NIBBLE_BITS * STATE_ROWS
COLUMN_SIZE = 1 << COLUMN_BITS
_COLUMN_VALUES = numpy.arange(COLUMN_SIZE, dtype=numpy.uint16)
_PARITY16 = numpy.fromiter(
    ((value.bit_count() & 1) for value in range(COLUMN_SIZE)),
    dtype=numpy.int8,
    count=COLUMN_SIZE,
)


def nibble(value, index):
    return (int(value) >> (NIBBLE_BITS * index)) & 0xF


def word_to_nibbles(word):
    return [nibble(word, row) for row in range(STATE_ROWS)]


def nibbles_to_word(cells):
    if len(cells) != STATE_ROWS:
        raise ValueError("a Blink column contains four nibbles")
    return sum(
        (int(cell) & 0xF) << (NIBBLE_BITS * row)
        for row, cell in enumerate(cells)
    )


def extract_column(state_mask, column):
    if not 0 <= column < STATE_COLUMNS:
        raise ValueError(f"invalid Blink-128 column {column}")
    return sum(
        nibble(state_mask, column + STATE_COLUMNS * row)
        << (NIBBLE_BITS * row)
        for row in range(STATE_ROWS)
    )


def insert_column(column_mask, column):
    if int(column_mask) >> COLUMN_BITS:
        raise ValueError("column mask must be 16-bit")
    if not 0 <= column < STATE_COLUMNS:
        raise ValueError(f"invalid Blink-128 column {column}")
    return sum(
        nibble(column_mask, row)
        << (NIBBLE_BITS * (column + STATE_COLUMNS * row))
        for row in range(STATE_ROWS)
    )


def mix_column_word(word):
    cells = word_to_nibbles(word)
    return nibbles_to_word(
        [
            cells[(row + 1) % STATE_ROWS]
            ^ cells[(row + 2) % STATE_ROWS]
            ^ cells[(row + 3) % STATE_ROWS]
            for row in range(STATE_ROWS)
        ]
    )


def sub_column_word(word):
    return nibbles_to_word([SBOX[cell] for cell in word_to_nibbles(word)])


def fixed_middle_superbox_word(word, middle_value=0):
    """Evaluate S-M-(xor middle_value)-S on one 16-bit column."""
    if int(middle_value) >> COLUMN_BITS:
        raise ValueError("middle value must be 16-bit")
    return sub_column_word(
        mix_column_word(sub_column_word(word)) ^ int(middle_value)
    )


def fwht_int(values):
    values = numpy.asarray(values, dtype=numpy.int64).copy()
    width = 1
    while width < len(values):
        blocks = values.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return values


@lru_cache(maxsize=None)
def sbox_output_values(input_difference, output_difference):
    return tuple(
        SBOX[value]
        for value in range(16)
        if SBOX[value] ^ SBOX[value ^ input_difference] == output_difference
    )


@lru_cache(maxsize=None)
def sbox_input_values(input_difference, output_difference):
    return tuple(
        value
        for value in range(16)
        if SBOX[value] ^ SBOX[value ^ input_difference] == output_difference
    )


def internal_difference_vectors(source_activity, target_activity):
    active_rows = [row for row in range(STATE_ROWS) if source_activity >> row & 1]
    for values in product(G_VALUES, repeat=len(active_rows)):
        alpha = [0] * STATE_ROWS
        for row, value in zip(active_rows, values):
            alpha[row] = value
        beta = word_to_nibbles(mix_column_word(nibbles_to_word(alpha)))
        if all(
            (value in G_VALUES) if target_activity >> row & 1 else value == 0
            for row, value in enumerate(beta)
        ):
            yield tuple(alpha), tuple(beta)


def indicator_from_products(value_lists, transform=None):
    indicator = numpy.zeros(COLUMN_SIZE, dtype=numpy.int64)
    for cells in product(*value_lists):
        word = nibbles_to_word(cells)
        if transform is not None:
            word = transform(word)
        indicator[word] = 1
    return indicator


@lru_cache(maxsize=None)
def key_probability_numerators(source_activity, target_activity):
    """Right-input counts for all ``2^16`` middle-key column values."""
    result = numpy.zeros(COLUMN_SIZE, dtype=numpy.int64)
    source_difference = [
        0x8 if source_activity >> row & 1 else 0 for row in range(STATE_ROWS)
    ]
    target_difference = [
        0x8 if target_activity >> row & 1 else 0 for row in range(STATE_ROWS)
    ]
    for alpha, beta in internal_difference_vectors(source_activity, target_activity):
        left_lists = [
            sbox_output_values(source_difference[row], alpha[row])
            for row in range(STATE_ROWS)
        ]
        right_lists = [
            sbox_input_values(beta[row], target_difference[row])
            for row in range(STATE_ROWS)
        ]
        left = indicator_from_products(left_lists, transform=mix_column_word)
        right = indicator_from_products(right_lists)
        result += fwht_int(fwht_int(left) * fwht_int(right)) // COLUMN_SIZE
    return result


@lru_cache(maxsize=None)
def key_fourier_coefficients(source_activity, target_activity):
    """Return F[q] where p(k)=sum_q F[q]/2^32 (-1)^(q.k)."""
    return fwht_int(key_probability_numerators(source_activity, target_activity))


@lru_cache(maxsize=16)
def fixed_middle_output_table(middle_value=0):
    return numpy.fromiter(
        (
            fixed_middle_superbox_word(value, middle_value)
            for value in range(COLUMN_SIZE)
        ),
        dtype=numpy.uint16,
        count=COLUMN_SIZE,
    )


@lru_cache(maxsize=256)
def fixed_middle_input_spectrum(
    source_activity, target_activity, output_mask=0, middle_value=0
):
    """Input-mask Fourier numerators for a fixed middle value/output mask.

    The returned coefficient for input mask ``a`` is ``result[a]/2^16``.
    """
    input_difference = sum(
        0x8 << (NIBBLE_BITS * row)
        for row in range(STATE_ROWS)
        if source_activity >> row & 1
    )
    output_difference = sum(
        0x8 << (NIBBLE_BITS * row)
        for row in range(STATE_ROWS)
        if target_activity >> row & 1
    )
    output_table = fixed_middle_output_table(middle_value)
    paired = output_table[_COLUMN_VALUES ^ input_difference]
    valid = (output_table ^ paired) == output_difference
    phase = _PARITY16[numpy.bitwise_and(output_table, output_mask)]
    values = numpy.where(valid, 1 - 2 * phase, 0).astype(numpy.int64)
    return fwht_int(values)


def quotient_table(fourier_values, denominator_bits):
    """Compress a 16-bit Fourier function to its exact support quotient.

    The result is keyed by the ordered support basis.  ``values[s]`` is the
    numerator over ``2^denominator_bits`` for keys/inputs whose basis-parity
    syndrome is ``s``.
    """
    support = [
        (mask, int(value))
        for mask, value in enumerate(fourier_values)
        if int(value)
    ]
    basis = ordered_basis(support)
    coordinates = LinearCoordinates(basis)
    coefficients = numpy.zeros(1 << len(basis), dtype=numpy.int64)
    for mask, value in support:
        coordinate = coordinates.coordinate(mask)
        if coordinate is None:
            raise AssertionError("Fourier support escaped its computed span")
        coefficients[coordinate] = value
    values = fwht_int(coefficients)
    return {
        "basis": basis,
        "rank": len(basis),
        "support_size": len(support),
        "values": values,
        "denominator_bits": int(denominator_bits),
    }


@lru_cache(maxsize=None)
def key_quotient(source_activity, target_activity):
    return quotient_table(
        key_fourier_coefficients(source_activity, target_activity), 32
    )


@lru_cache(maxsize=None)
def input_quotient(source_activity, target_activity, middle_value=0):
    return quotient_table(
        fixed_middle_input_spectrum(
            source_activity, target_activity, 0, middle_value
        ),
        16,
    )

