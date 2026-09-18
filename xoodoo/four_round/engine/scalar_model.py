"""Zero-value-mask scalar reference for DDT-prefix compatibility checks.

This is a small self-contained copy of the v3 round-based recurrence.  It is
kept next to the joint-U implementation so every p=0/1/2 run can report the
old scalar answer for exactly the same characteristic batch.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import prod

from . import common, qd_kernel
from . import linear_layer as xoodoo_linear
from .prefix_characteristics import CharacteristicBatch


def _make_lat() -> tuple[tuple[Fraction, ...], ...]:
    # Indexed as LAT[output_mask][input_mask], matching the v3 recurrence.
    rows = []
    for output_mask in range(8):
        row = []
        for input_mask in range(8):
            numerator = sum(
                -1
                if ((input_mask & value).bit_count()
                    ^ (output_mask & qd_kernel.SBOX[value]).bit_count())
                & 1
                else 1
                for value in range(8)
            )
            row.append(Fraction(numerator, 8))
        rows.append(tuple(row))
    return tuple(rows)


LAT = _make_lat()
SQUARED_LAT = tuple(
    tuple(coefficient * coefficient for coefficient in row) for row in LAT
)
LocalVector = tuple[Fraction, ...]
RoundState = tuple[LocalVector, ...]


@lru_cache(maxsize=None)
def _linear_local_entries(
    output_column: int, output_local_mask: int
) -> tuple[tuple[int, int], ...]:
    pulled = xoodoo_linear.inter_chi_adjoint(
        common.column_mask(output_column, output_local_mask)
    )
    return xoodoo_linear.column_entries(pulled)


def _initial_vectors(difference: int) -> RoundState:
    return tuple(
        tuple(
            Fraction(-1 if (mask & local_difference).bit_count() & 1 else 1, 1)
            for mask in range(8)
        )
        for local_difference in xoodoo_linear.columns(difference)
    )


def _update_chi(state: RoundState) -> RoundState:
    return tuple(
        tuple(
            sum(
                (
                    SQUARED_LAT[output_mask][input_mask]
                    * vector[input_mask]
                    for input_mask in range(8)
                ),
                Fraction(0, 1),
            )
            for output_mask in range(8)
        )
        for vector in state
    )


def _update_linear(state: RoundState) -> RoundState:
    return tuple(
        tuple(
            prod(
                state[input_column][input_mask]
                for input_column, input_mask in _linear_local_entries(
                    output_column, output_mask
                )
            )
            for output_mask in range(8)
        )
        for output_column in range(128)
    )


def _read_mask(state: RoundState, output_mask: int) -> Fraction:
    return prod(
        state[column][local_mask]
        for column, local_mask in xoodoo_linear.column_entries(output_mask)
    )


def approximate_suffix(
    input_difference: int,
    output_mask: int,
    rounds: int,
) -> Fraction:
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if rounds == 0:
        sign = -1 if (input_difference & output_mask).bit_count() & 1 else 1
        return Fraction(sign, 1)
    state = _initial_vectors(input_difference)
    for round_index in range(rounds):
        state = _update_chi(state)
        if round_index + 1 < rounds:
            state = _update_linear(state)
    return _read_mask(state, output_mask)


def evaluate_batch(
    batch: CharacteristicBatch,
    output_mask: int,
    total_rounds: int,
) -> Fraction:
    """Evaluate the author-style zero-mode scalar for a prefix batch."""
    suffix_rounds = total_rounds - batch.prefix_rounds
    result = Fraction(0, 1)
    for characteristic in batch.characteristics:
        suffix_difference = characteristic.boundary_difference
        value = approximate_suffix(suffix_difference, output_mask, suffix_rounds)
        # Unlike the joint Q operator, the scalar suffix is conditioned only
        # by an ordinary DDT probability, so it must be multiplied here.
        result += characteristic.weighted_probability * value
    return result


def fraction_record(value: Fraction) -> dict:
    floating = float(value)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "coefficient": floating,
        "weight": common.weight(floating),
    }


__all__ = [
    "LAT",
    "SQUARED_LAT",
    "approximate_suffix",
    "evaluate_batch",
    "fraction_record",
]
