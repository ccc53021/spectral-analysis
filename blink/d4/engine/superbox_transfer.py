"""Exact cluster-aware quasidifferential transfer for Figure-3 Superboxes.

For one two-S-layer Superbox, ``T_j(a, c, q)`` keeps all three value-mask
interfaces open:

* ``a`` is the value mask before the first S layer;
* ``q`` is the value mask after M and at the middle key addition;
* ``c`` is the value mask after the second S layer.

The first S-layer output mask is ``M^T q``.  Blink's M is symmetric, so this
is just ``M q``.  Every compatible concrete differential inside the selected
Figure-3 cluster is summed exactly.  The implementation factorizes over the
four 16-bit columns and never expands the 25,165,824 global characteristics.
"""

from functools import lru_cache

import numpy

from .linear_layer import SBOX
from .superbox_spectrum import (
    internal_difference_vectors,
    mix_column_word,
)


NIBBLE_BITS = 4
COLUMN_ROWS = 4
COLUMN_BITS = NIBBLE_BITS * COLUMN_ROWS
STATE_COLUMNS = 4
SBOX_SCALE = 1 << NIBBLE_BITS
COLUMN_DENOMINATOR = SBOX_SCALE ** (2 * COLUMN_ROWS)
COLUMN_SIZE = 1 << COLUMN_BITS
_COLUMN_VALUES = numpy.arange(COLUMN_SIZE, dtype=numpy.uint16)
_PARITY16 = numpy.fromiter(
    ((value.bit_count() & 1) for value in range(COLUMN_SIZE)),
    dtype=numpy.int8,
    count=COLUMN_SIZE,
)


def nibble(value, index):
    return (value >> (NIBBLE_BITS * index)) & 0xF


def extract_column(state_mask, column):
    """Pack one 4-nibble state column into a contiguous 16-bit word."""
    if not 0 <= column < STATE_COLUMNS:
        raise ValueError(f"invalid column {column}")
    result = 0
    for row in range(COLUMN_ROWS):
        cell = column + STATE_COLUMNS * row
        result |= nibble(state_mask, cell) << (NIBBLE_BITS * row)
    return result


def insert_column(column_mask, column):
    """Embed a contiguous 16-bit mask in one 64-bit state column."""
    if not 0 <= column < STATE_COLUMNS:
        raise ValueError(f"invalid column {column}")
    result = 0
    for row in range(COLUMN_ROWS):
        cell = column + STATE_COLUMNS * row
        result |= nibble(column_mask, row) << (NIBBLE_BITS * cell)
    return result


def word_to_nibbles(word):
    return [nibble(word, row) for row in range(COLUMN_ROWS)]


def nibbles_to_word(cells):
    return sum((cell & 0xF) << (NIBBLE_BITS * row) for row, cell in enumerate(cells))


def sub_column_word(word):
    return nibbles_to_word([SBOX[cell] for cell in word_to_nibbles(word)])


def fixed_middle_superbox_word(word, middle_value=0):
    """Evaluate S-M-(xor middle_value)-S on one 16-bit column."""
    return sub_column_word(mix_column_word(sub_column_word(word)) ^ middle_value)


def fwht_int(values):
    values = values.copy()
    width = 1
    while width < len(values):
        blocks = values.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return values


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
def fixed_middle_column_spectrum(
    source_activity, target_activity, c, middle_value=0
):
    """Return every input-mask numerator for one fixed output mask c."""
    input_difference = sum(
        (0x8 << (NIBBLE_BITS * row))
        for row in range(COLUMN_ROWS)
        if (source_activity >> row) & 1
    )
    output_difference = sum(
        (0x8 << (NIBBLE_BITS * row))
        for row in range(COLUMN_ROWS)
        if (target_activity >> row) & 1
    )
    output_table = fixed_middle_output_table(middle_value)
    paired = output_table[_COLUMN_VALUES ^ input_difference]
    valid = (output_table ^ paired) == output_difference
    phase = _PARITY16[numpy.bitwise_and(output_table, c)]
    values = numpy.where(valid, 1 - 2 * phase, 0).astype(numpy.int64)
    return fwht_int(values)


def activity_column(state_activity, column):
    """Return the four active/inactive bits of one state column."""
    return sum(
        ((state_activity >> (column + STATE_COLUMNS * row)) & 1) << row
        for row in range(COLUMN_ROWS)
    )


@lru_cache(maxsize=None)
def sbox_qd_numerator(input_difference, output_difference, input_mask, output_mask):
    """Return 16 times one exact 4-bit S-box QD correlation."""
    total = 0
    for value in range(SBOX_SCALE):
        paired = value ^ input_difference
        output = SBOX[value]
        if output ^ SBOX[paired] != output_difference:
            continue
        phase = (input_mask & value).bit_count()
        phase ^= (output_mask & output).bit_count()
        total += -1 if phase & 1 else 1
    return total


@lru_cache(maxsize=1 << 20)
def column_transfer_numerator(source_activity, target_activity, a, c, q):
    """Return the numerator of one exact 16-bit column transfer.

    The denominator is ``COLUMN_DENOMINATOR = 16^8 = 2^32``.  Summation over
    all compatible internal differences happens before the value is returned,
    preserving cancellation between local characteristics.
    """
    if any(value >> COLUMN_BITS for value in (a, c, q)):
        raise ValueError("column masks must be 16-bit")

    source_difference = [
        0x8 if (source_activity >> row) & 1 else 0
        for row in range(COLUMN_ROWS)
    ]
    target_difference = [
        0x8 if (target_activity >> row) & 1 else 0
        for row in range(COLUMN_ROWS)
    ]
    first_output_mask = mix_column_word(q)  # M^T q = M q for Blink.

    total = 0
    for alpha, beta in internal_difference_vectors(
        source_activity, target_activity
    ):
        term = 1
        for row in range(COLUMN_ROWS):
            term *= sbox_qd_numerator(
                source_difference[row],
                alpha[row],
                nibble(a, row),
                nibble(first_output_mask, row),
            )
            if term == 0:
                break
            term *= sbox_qd_numerator(
                beta[row],
                target_difference[row],
                nibble(q, row),
                nibble(c, row),
            )
            if term == 0:
                break
        total += term
    return total


@lru_cache(maxsize=1 << 20)
def fixed_middle_column_transfer_numerator(
    source_activity, target_activity, a, c, middle_value=0
):
    """Return ``sum_q T(a,c,q)(-1)^(q.middle_value)`` exactly.

    A known middle XOR value removes the key variable, but it does *not* force
    the internal Fourier mask q to zero.  Evaluating the composed 16-bit
    Superbox directly performs the complete q contraction in O(2^16) time.
    The denominator of the returned QD coefficient is 2^16.
    """
    if any(value >> COLUMN_BITS for value in (a, c, middle_value)):
        raise ValueError("column masks and middle value must be 16-bit")

    return int(
        fixed_middle_column_spectrum(
            source_activity, target_activity, c, middle_value
        )[a]
    )
