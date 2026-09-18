"""Reference Xoodoo permutation and linear-boundary helpers.

The state is one Python integer with bit index

    i = z + 32 * (x + 4 * y).

A local chi column is encoded as a three-bit integer whose bit ``y`` is
``A[x, y, z]``.  The round-based code uses chi-input/chi-output boundaries;
the affine layer between two consecutive chi layers is therefore

    rho_west(theta(rho_east(state))) + round_constant.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Iterable, Sequence

LANE_BITS = 32
LANES_PER_PLANE = 4
PLANES = 3
COLUMNS = LANE_BITS * LANES_PER_PLANE
STATE_BITS = COLUMNS * PLANES
LANE_MASK = (1 << LANE_BITS) - 1
STATE_MASK = (1 << STATE_BITS) - 1

# c[-11], ..., c[0], in the order used by Xoodoo[12].
ROUND_CONSTANTS = (
    0x00000058,
    0x00000038,
    0x000003C0,
    0x000000D0,
    0x00000120,
    0x00000014,
    0x00000060,
    0x0000002C,
    0x00000380,
    0x000000F0,
    0x000001A0,
    0x00000012,
)


def bit_index(x: int, y: int, z: int) -> int:
    if not 0 <= x < 4 or not 0 <= y < 3 or not 0 <= z < 32:
        raise ValueError("Xoodoo coordinates are x in [0,4), y in [0,3), z in [0,32)")
    return z + 32 * (x + 4 * y)


def column_index(x: int, z: int) -> int:
    if not 0 <= x < 4 or not 0 <= z < 32:
        raise ValueError("Xoodoo column coordinates are x in [0,4), z in [0,32)")
    return 32 * x + z


def column_coordinates(column: int) -> tuple[int, int]:
    if not 0 <= column < COLUMNS:
        raise ValueError("column must be in [0,128)")
    return divmod(column, 32)


def column_mask(column: int, value: int) -> int:
    """Place a three-bit local column value into a 384-bit state."""
    if not 0 <= value < 8:
        raise ValueError("column value must be in [0,8)")
    x, z = column_coordinates(column)
    result = 0
    for y in range(3):
        result |= ((value >> y) & 1) << bit_index(x, y, z)
    return result


def get_column(state: int, column: int) -> int:
    x, z = column_coordinates(column)
    return sum(((state >> bit_index(x, y, z)) & 1) << y for y in range(3))


def columns(state: int) -> tuple[int, ...]:
    return tuple(get_column(state, column) for column in range(COLUMNS))


def from_columns(values: Sequence[int]) -> int:
    if len(values) != COLUMNS:
        raise ValueError("a Xoodoo state has exactly 128 columns")
    result = 0
    for column, value in enumerate(values):
        result |= column_mask(column, int(value))
    return result


def lanes(state: int) -> list[int]:
    return [(state >> (32 * i)) & LANE_MASK for i in range(12)]


def from_lanes(values: Sequence[int]) -> int:
    if len(values) != 12:
        raise ValueError("a Xoodoo state has exactly 12 lanes")
    return sum((int(value) & LANE_MASK) << (32 * i) for i, value in enumerate(values))


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
    a = lanes(state)
    planes = [a[4 * y : 4 * y + 4] for y in range(3)]
    parity_plane = [planes[0][x] ^ planes[1][x] ^ planes[2][x] for x in range(4)]
    p5 = _shift_plane(parity_plane, 1, 5)
    p14 = _shift_plane(parity_plane, 1, 14)
    effect = [p5[x] ^ p14[x] for x in range(4)]
    return from_lanes(
        [planes[y][x] ^ effect[x] for y in range(3) for x in range(4)]
    )


def rho_west(state: int) -> int:
    a = lanes(state)
    result = a[:]
    result[4:8] = _shift_plane(a[4:8], 1, 0)
    result[8:12] = _shift_plane(a[8:12], 0, 11)
    return from_lanes(result)


def rho_east(state: int) -> int:
    a = lanes(state)
    result = a[:]
    result[4:8] = _shift_plane(a[4:8], 0, 1)
    result[8:12] = _shift_plane(a[8:12], 2, 8)
    return from_lanes(result)


def add_round_constant(state: int, constant: int) -> int:
    # The non-zero lane of C_i is at (x=0, y=0).
    return state ^ (int(constant) & LANE_MASK)


def chi_column(value: int) -> int:
    if not 0 <= value < 8:
        raise ValueError("chi input must be a three-bit column")
    bits = [(value >> y) & 1 for y in range(3)]
    output = [
        bits[y] ^ ((bits[(y + 1) % 3] ^ 1) & bits[(y + 2) % 3])
        for y in range(3)
    ]
    return sum(bit << y for y, bit in enumerate(output))


CHI = tuple(chi_column(value) for value in range(8))


def chi(state: int) -> int:
    return from_columns([CHI[value] for value in columns(state)])


def pre_chi_linear(state: int) -> int:
    """The external linear prefix theta then rho_west."""
    return rho_west(theta(state))


def inter_chi_linear(state: int) -> int:
    """The linear map from one chi output to the next chi input."""
    return rho_west(theta(rho_east(state)))


def round_from_chi_input(state: int, next_constant: int) -> int:
    """Map a chi input to the following round's chi input."""
    return add_round_constant(inter_chi_linear(chi(state)), next_constant)


def chi_to_chi(state: int, rounds: int, constants: Sequence[int] | None = None) -> int:
    """Apply ``rounds`` chi layers, starting at chi input and ending at chi output.

    There are ``rounds - 1`` affine connectors.  Their constants are the
    constants of the following full Xoodoo rounds.  Constants are optional for
    differential-only tests, but real DL experiments should supply them.
    """
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    if rounds == 0:
        return int(state) & STATE_MASK
    if constants is None:
        constants = (0,) * max(0, rounds - 1)
    if len(constants) != max(0, rounds - 1):
        raise ValueError("chi-to-chi needs one constant for every affine connector")

    result = int(state) & STATE_MASK
    for round_index in range(rounds):
        result = chi(result)
        if round_index + 1 < rounds:
            result = inter_chi_linear(result)
            result = add_round_constant(result, constants[round_index])
    return result


def xoodoo_round(state: int, constant: int) -> int:
    state = theta(state)
    state = rho_west(state)
    state = add_round_constant(state, constant)
    state = chi(state)
    return rho_east(state)


def reduced_round_constants(rounds: int) -> tuple[int, ...]:
    if not 0 <= rounds <= 12:
        raise ValueError("the supplied Xoodoo instance supports 0 through 12 rounds")
    return ROUND_CONSTANTS[12 - rounds :]


def permute(state: int, rounds: int = 12) -> int:
    result = int(state) & STATE_MASK
    for constant in reduced_round_constants(rounds):
        result = xoodoo_round(result, constant)
    return result


def parity(value: int) -> int:
    return int(value).bit_count() & 1


def adjoint(linear: Callable[[int], int], mask: int) -> int:
    """Return L^T(mask) for a 384-bit GF(2)-linear function L."""
    result = 0
    for bit in range(STATE_BITS):
        if parity(mask & linear(1 << bit)):
            result |= 1 << bit
    return result


@lru_cache(maxsize=None)
def local_adjoint_table(name: str) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Sparse local masks of L^T(tau_column(local_mask)).

    Each table entry is a tuple of ``(input_column, three_bit_mask)`` pairs.
    ``name`` is deliberately restricted so cached results cannot accidentally
    refer to an arbitrary mutable callable.
    """
    functions: dict[str, Callable[[int], int]] = {
        "pre_chi": pre_chi_linear,
        "inter_chi": inter_chi_linear,
        "rho_east": rho_east,
    }
    try:
        linear = functions[name]
    except KeyError as exc:
        raise ValueError(f"unknown Xoodoo linear map: {name}") from exc

    result = []
    for output_column in range(COLUMNS):
        local_entries = []
        for local_mask in range(8):
            pulled_back = adjoint(linear, column_mask(output_column, local_mask))
            local_entries.append(
                tuple(
                    (input_column, value)
                    for input_column, value in enumerate(columns(pulled_back))
                    if value
                )
            )
        result.append(tuple(local_entries))
    return tuple(result)


def state_to_hex_lanes(state: int) -> tuple[str, ...]:
    return tuple(f"{lane:08x}" for lane in lanes(state))


def state_from_active_columns(entries: Iterable[tuple[int, int]]) -> int:
    result = 0
    for column, value in entries:
        result ^= column_mask(column, value)
    return result
