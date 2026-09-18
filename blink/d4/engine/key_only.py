"""Exact master-key distribution for the Blink-64 D.4 cluster."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from .linear_layer import INVERSE_ROUND_CONSTANTS, MASTER_KEY_BITS, ROUND_CONSTANTS
from .superbox_spectrum import fourier_coefficients
from .trail import C, D


RK2_MASTER_OFFSET = 192
RK4_MASTER_OFFSET = 320
ACTIVE_COLUMN = 3


def column_mask(state_mask, column):
    return sum(
        ((state_mask >> (column + 4 * row)) & 1) << row
        for row in range(4)
    )


SB2_MASKS = (column_mask(C, ACTIVE_COLUMN), column_mask(D, ACTIVE_COLUMN))
SB4_MASKS = (column_mask(D, ACTIVE_COLUMN), column_mask(C, ACTIVE_COLUMN))


def embed_column_mask(local_mask, master_offset):
    result = 0
    for row in range(4):
        for bit in range(4):
            if (local_mask >> (4 * row + bit)) & 1:
                round_key_bit = 4 * (ACTIVE_COLUMN + 4 * row) + bit
                result |= 1 << (master_offset + round_key_bit)
    return result


def local_terms(masks, master_offset, round_constant):
    coefficients = fourier_coefficients(*masks)
    terms = []
    for local_mask, numerator in enumerate(coefficients):
        if not numerator:
            continue
        master_mask = embed_column_mask(local_mask, master_offset)
        column_constant = 0
        for row in range(4):
            nibble = (round_constant >> (4 * (ACTIVE_COLUMN + 4 * row))) & 0xF
            column_constant |= nibble << (4 * row)
        phase = -1 if (local_mask & column_constant).bit_count() & 1 else 1
        terms.append((master_mask, Fraction(phase * int(numerator), 1 << 32)))
    return terms


def combined_fourier_terms():
    sb2 = local_terms(SB2_MASKS, RK4_MASTER_OFFSET, ROUND_CONSTANTS[3])
    sb4 = local_terms(SB4_MASKS, RK2_MASTER_OFFSET, INVERSE_ROUND_CONSTANTS[1])
    constant_superbox_factor = Fraction(3, 1 << 44)
    combined = {}
    for mask2, coefficient2 in sb2:
        for mask4, coefficient4 in sb4:
            mask = mask2 ^ mask4
            combined[mask] = (
                combined.get(mask, Fraction(0))
                + constant_superbox_factor * coefficient2 * coefficient4
            )
    return {mask: value for mask, value in combined.items() if value}


def rref_basis(values, width):
    rows = [value for value in values if value]
    row_index = 0
    pivots = []
    for column in range(width):
        selected = next(
            (
                index
                for index in range(row_index, len(rows))
                if (rows[index] >> column) & 1
            ),
            None,
        )
        if selected is None:
            continue
        rows[row_index], rows[selected] = rows[selected], rows[row_index]
        pivot_row = rows[row_index]
        for index in range(len(rows)):
            if index != row_index and ((rows[index] >> column) & 1):
                rows[index] ^= pivot_row
        pivots.append(column)
        row_index += 1
        if row_index == len(rows):
            break
    return rows[:row_index], pivots


def coordinate(value, rows, pivots):
    result = 0
    for index, (row, pivot) in enumerate(zip(rows, pivots)):
        if (value >> pivot) & 1:
            value ^= row
            result |= 1 << index
    if value:
        raise ValueError("Fourier mask is outside its generated span")
    return result


def exact_fwht(values):
    width = 1
    while width < len(values):
        for start in range(0, len(values), 2 * width):
            for offset in range(width):
                left = values[start + offset]
                right = values[start + width + offset]
                values[start + offset] = left + right
                values[start + width + offset] = left - right
        width *= 2
    return values


def exact_record(value):
    return {"numerator": value.numerator, "denominator": value.denominator}


def write_key_only_distribution(path):
    terms = combined_fourier_terms()
    rows, pivots = rref_basis(terms, MASTER_KEY_BITS)
    values = [Fraction(0) for _ in range(1 << len(rows))]
    for mask, coefficient in terms.items():
        values[coordinate(mask, rows, pivots)] += coefficient
    probabilities = exact_fwht(values)
    positive = [value for value in probabilities if value > 0]
    histogram = Counter(probabilities)
    record = {
        "cipher": "Blink-64",
        "source": "D.4 ten-round cluster",
        "fourier_support_size": len(terms),
        "basis_dimension": len(rows),
        "p_max_exact": exact_record(max(positive)),
        "p_min_nonzero_exact": exact_record(min(positive)),
        "probability_histogram": [
            {
                "probability_exact": exact_record(probability),
                "class_count": count,
            }
            for probability, count in sorted(histogram.items())
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
