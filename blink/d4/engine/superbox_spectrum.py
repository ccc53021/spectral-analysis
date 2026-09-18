"""Exact local Fourier spectra for the published Blink Superbox clusters.

For one 4-nibble column and a fixed 16-bit middle key, this module sums every
compatible concrete characteristic in the selected truncated pattern.  XOR
convolution/FWHT evaluates all 2^16 keys exactly, avoiding a 2^32 key/input
loop.  These local spectra are the cluster-aware input to the existing
Step1--Step3 pipeline.
"""

from functools import lru_cache
from itertools import product

import numpy

from .linear_layer import SBOX
from .trail import G_VALUES


WORD_BITS = 16
WORD_SIZE = 1 << WORD_BITS


def fwht(values):
    values = values.copy()
    width = 1
    while width < len(values):
        for start in range(0, len(values), 2 * width):
            left = values[start:start + width].copy()
            right = values[start + width:start + 2 * width].copy()
            values[start:start + width] = left + right
            values[start + width:start + 2 * width] = left - right
        width *= 2
    return values


def inverse_fwht(values):
    return fwht(values) // len(values)


def nibbles_to_word(cells):
    return sum((cell & 0xF) << (4 * row) for row, cell in enumerate(cells))


def word_to_nibbles(word):
    return [(word >> (4 * row)) & 0xF for row in range(4)]


def mix_column_word(word):
    cells = word_to_nibbles(word)
    return nibbles_to_word(
        [cells[(row + 1) % 4] ^ cells[(row + 2) % 4] ^ cells[(row + 3) % 4]
         for row in range(4)]
    )


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


def active_rows(mask):
    return [row for row in range(4) if (mask >> row) & 1]


def internal_difference_vectors(source_mask, target_mask):
    positions = active_rows(source_mask)
    for values in product(G_VALUES, repeat=len(positions)):
        alpha = [0] * 4
        for position, value in zip(positions, values):
            alpha[position] = value
        beta = word_to_nibbles(mix_column_word(nibbles_to_word(alpha)))
        if all(
            (value in G_VALUES) if ((target_mask >> row) & 1) else value == 0
            for row, value in enumerate(beta)
        ):
            yield tuple(alpha), tuple(beta)


def indicator_from_products(value_lists, transform=None):
    indicator = numpy.zeros(WORD_SIZE, dtype=numpy.int64)
    for cells in product(*value_lists):
        word = nibbles_to_word(cells)
        if transform is not None:
            word = transform(word)
        indicator[word] = 1
    return indicator


@lru_cache(maxsize=None)
def key_probability_numerators(source_mask, target_mask):
    """Return exact right-pair counts for every 16-bit middle key."""
    numerators = numpy.zeros(WORD_SIZE, dtype=numpy.int64)
    source_difference = [0x8 if (source_mask >> row) & 1 else 0 for row in range(4)]
    target_difference = [0x8 if (target_mask >> row) & 1 else 0 for row in range(4)]

    for alpha, beta in internal_difference_vectors(source_mask, target_mask):
        x_lists = [
            sbox_output_values(source_difference[row], alpha[row])
            for row in range(4)
        ]
        y_lists = [
            sbox_input_values(beta[row], target_difference[row])
            for row in range(4)
        ]
        a = indicator_from_products(x_lists, transform=mix_column_word)
        y = indicator_from_products(y_lists)
        numerators += inverse_fwht(fwht(a) * fwht(y))
    return numerators


def fourier_coefficients(source_mask, target_mask):
    """Return integer Fourier numerators for p(k)=N(k)/2^16.

    With the convention p(k)=sum_m c[m](-1)^(m.k), the returned integer is
    c[m]*2^32.  Keeping integers preserves exact cancellation in Step 2.
    """
    return fwht(key_probability_numerators(source_mask, target_mask))
