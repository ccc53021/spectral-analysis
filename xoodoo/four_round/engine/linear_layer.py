"""Integer Xoodoo linear helpers used by the joint-DDT prefix."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from . import common


LANE_MASK = (1 << 32) - 1
STATE_MASK = (1 << common.STATE_BITS) - 1


def lanes(state: int) -> list[int]:
    state = int(state) & STATE_MASK
    return [(state >> (32 * index)) & LANE_MASK for index in range(12)]


def from_lanes(values: Sequence[int]) -> int:
    if len(values) != 12:
        raise ValueError("a Xoodoo state needs twelve lanes")
    return sum((int(value) & LANE_MASK) << (32 * index) for index, value in enumerate(values))


def rotl32(value: int, amount: int) -> int:
    amount %= 32
    value &= LANE_MASK
    if amount == 0:
        return value
    return ((value << amount) | (value >> (32 - amount))) & LANE_MASK


def _shift_plane(plane: Sequence[int], x_shift: int, z_shift: int) -> list[int]:
    shifted = [0] * 4
    for x in range(4):
        shifted[(x + x_shift) % 4] = rotl32(int(plane[x]), z_shift)
    return shifted


def theta(state: int) -> int:
    values = lanes(state)
    planes = [values[4 * y : 4 * y + 4] for y in range(3)]
    parity_plane = [planes[0][x] ^ planes[1][x] ^ planes[2][x] for x in range(4)]
    p5 = _shift_plane(parity_plane, 1, 5)
    p14 = _shift_plane(parity_plane, 1, 14)
    effect = [p5[x] ^ p14[x] for x in range(4)]
    return from_lanes([planes[y][x] ^ effect[x] for y in range(3) for x in range(4)])


def rho_west(state: int) -> int:
    values = lanes(state)
    result = values[:]
    result[4:8] = _shift_plane(values[4:8], 1, 0)
    result[8:12] = _shift_plane(values[8:12], 0, 11)
    return from_lanes(result)


def rho_east(state: int) -> int:
    values = lanes(state)
    result = values[:]
    result[4:8] = _shift_plane(values[4:8], 0, 1)
    result[8:12] = _shift_plane(values[8:12], 2, 8)
    return from_lanes(result)


def inter_chi_linear(state: int) -> int:
    return rho_west(theta(rho_east(state)))


def pre_chi_linear(state: int) -> int:
    """Linear part from a full-round input to its first chi input."""
    return rho_west(theta(state))


def get_column(state: int, column: int) -> int:
    x, z = common.column_coordinates(column)
    return sum(
        ((int(state) >> common.bit_index(x, y, z)) & 1) << y for y in range(3)
    )


def columns(state: int) -> tuple[int, ...]:
    return tuple(get_column(state, column) for column in range(128))


def column_entries(state: int) -> tuple[tuple[int, int], ...]:
    return tuple((column, value) for column, value in enumerate(columns(state)) if value)


def active_columns(state: int) -> int:
    return sum(value != 0 for value in columns(state))


def constant_state(constant: int) -> int:
    return int(constant) & LANE_MASK


def _make_adjoint_rows(linear) -> tuple[int, ...]:
    rows = [0] * common.STATE_BITS
    for input_bit in range(common.STATE_BITS):
        image = linear(1 << input_bit)
        while image:
            lowest = image & -image
            output_bit = lowest.bit_length() - 1
            rows[output_bit] |= 1 << input_bit
            image ^= lowest
    return tuple(rows)


_INTER_CHI_ADJOINT_ROWS = _make_adjoint_rows(inter_chi_linear)
_PRE_CHI_ADJOINT_ROWS = _make_adjoint_rows(pre_chi_linear)
_RHO_EAST_ADJOINT_ROWS = _make_adjoint_rows(rho_east)


def _apply_adjoint(mask: int, rows: tuple[int, ...]) -> int:
    value = int(mask) & STATE_MASK
    result = 0
    while value:
        lowest = value & -value
        output_bit = lowest.bit_length() - 1
        result ^= rows[output_bit]
        value ^= lowest
    return result


@lru_cache(maxsize=1 << 16)
def inter_chi_adjoint(mask: int) -> int:
    """Return ``L^T(mask)`` for the inter-chi linear map ``L``."""
    return _apply_adjoint(mask, _INTER_CHI_ADJOINT_ROWS)


@lru_cache(maxsize=1 << 16)
def pre_chi_adjoint(mask: int) -> int:
    return _apply_adjoint(mask, _PRE_CHI_ADJOINT_ROWS)


@lru_cache(maxsize=1 << 16)
def rho_east_adjoint(mask: int) -> int:
    return _apply_adjoint(mask, _RHO_EAST_ADJOINT_ROWS)
