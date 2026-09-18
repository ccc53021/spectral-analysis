"""Exact five-Superbox transfer with all inter-Superbox masks open.

The known-zero middle values h1(0), h(0), and h2(0) are contracted inside
SB1, SB3, and SB5.  Only the variable middle masks of SB2 (rk4) and SB4
(rk2) remain explicit.  Nonzero boundary masks expose the connector keys
rk3, rk5, rk1, and rk3 through the transposed affine maps.
"""

from functools import lru_cache

from .linear_layer import (
    INVERSE_ROUND_CONSTANTS,
    MASTER_KEY_BITS,
    ROUND_CONSTANTS,
    inverse_shuffle_cells,
    mix_columns,
    shuffle_cells,
)
from .superbox_transfer import (
    activity_column,
    column_transfer_numerator,
    extract_column,
    fixed_middle_column_transfer_numerator,
)
from .trail import SUPERBOXES


SUPERBOX_COUNT = 5
STATE_MASK_BITS = 64
GLOBAL_TRANSFER_DENOMINATOR_BITS = 448


def pmp(state):
    return shuffle_cells(mix_columns(shuffle_cells(state)))


def inverse_pmp(state):
    return inverse_shuffle_cells(mix_columns(inverse_shuffle_cells(state)))


KEY_MAPS = (
    None,
    shuffle_cells,
    lambda value: value,
    shuffle_cells,
    None,
    inverse_shuffle_cells,
    lambda value: value,
    inverse_shuffle_cells,
    None,
)

KEY_OFFSETS = (None, 256, 320, 384, None, 128, 192, 256, None)
CONSTANTS = (
    0,
    ROUND_CONSTANTS[2],
    ROUND_CONSTANTS[3],
    ROUND_CONSTANTS[4],
    0,
    INVERSE_ROUND_CONSTANTS[0],
    INVERSE_ROUND_CONSTANTS[1],
    INVERSE_ROUND_CONSTANTS[2],
    0,
)


def parity(value):
    return value.bit_count() & 1


@lru_cache(maxsize=None)
def transpose_map_name(name, mask):
    function = {
        "M": mix_columns,
        "P": shuffle_cells,
        "PI": inverse_shuffle_cells,
        "PMP": pmp,
        "PIMPI": inverse_pmp,
    }[name]
    result = 0
    for bit in range(STATE_MASK_BITS):
        if parity(mask & function(1 << bit)):
            result |= 1 << bit
    return result


def transpose_linear(index, mask):
    names = ("M", "PMP", "M", "PMP", "M", "PIMPI", "M", "PIMPI", "M")
    return transpose_map_name(names[index], mask)


def transpose_key_map(index, mask):
    if KEY_MAPS[index] is None:
        return 0
    names = (None, "P", "I", "P", None, "PI", "I", "PI", None)
    if names[index] == "I":
        return mask
    return transpose_map_name(names[index], mask)


def joint_mask(a_masks):
    mask = a_masks[0] << MASTER_KEY_BITS
    for index, offset in enumerate(KEY_OFFSETS):
        if offset is None:
            continue
        local = transpose_key_map(index, a_masks[index + 1])
        mask ^= local << offset
    return mask


def boundary_output_masks(boundary_inputs):
    """Derive each Superbox output mask from the next input mask."""
    if len(boundary_inputs) != SUPERBOX_COUNT:
        raise ValueError("five Superbox input masks are required")
    outputs = []
    for superbox_index in range(SUPERBOX_COUNT - 1):
        connection_index = 2 * superbox_index + 1
        outputs.append(
            transpose_linear(connection_index, boundary_inputs[superbox_index + 1])
        )
    outputs.append(0)
    return outputs


def expanded_a_masks(boundary_inputs, q2, q4):
    """Embed the contracted variables in the old ten-S-layer a-mask layout."""
    if len(boundary_inputs) != SUPERBOX_COUNT:
        raise ValueError("five Superbox input masks are required")
    return [
        boundary_inputs[0], 0,
        boundary_inputs[1], q2,
        boundary_inputs[2], 0,
        boundary_inputs[3], q4,
        boundary_inputs[4], 0,
    ]


def contracted_joint_mask(boundary_inputs, q2, q4):
    """Return the joint (master-key,input-value) Fourier mask."""
    return joint_mask(expanded_a_masks(boundary_inputs, q2, q4))


def round_constant_phase(boundary_inputs, q2, q4):
    a_masks = expanded_a_masks(boundary_inputs, q2, q4)
    phase = 0
    for index, constant in enumerate(CONSTANTS):
        if constant:
            key_map = KEY_MAPS[index]
            if key_map is None:
                raise AssertionError("nonzero constant has no affine key map")
            phase ^= parity(a_masks[index + 1] & key_map(constant))
    return phase


def contracted_trail_numerator(boundary_inputs, q2, q4):
    """Return the integer numerator over the common denominator ``2^448``.

    Keeping the common denominator is substantially faster during a 2^20
    span expansion than repeatedly constructing and reducing Fractions.
    """
    outputs = boundary_output_masks(boundary_inputs)
    numerator = 1
    for superbox_index in range(SUPERBOX_COUNT):
        row = SUPERBOXES[superbox_index]
        fixed_middle = superbox_index in (0, 2, 4)
        middle_mask = q2 if superbox_index == 1 else q4
        for column in range(4):
            source = activity_column(row["source"], column)
            target = activity_column(row["target"], column)
            a = extract_column(boundary_inputs[superbox_index], column)
            c = extract_column(outputs[superbox_index], column)
            if fixed_middle:
                local = fixed_middle_column_transfer_numerator(
                    source, target, a, c, 0
                )
            else:
                q = extract_column(middle_mask, column)
                local = column_transfer_numerator(source, target, a, c, q)
            numerator *= local
            if numerator == 0:
                return 0
    if round_constant_phase(boundary_inputs, q2, q4):
        numerator = -numerator
    return numerator


def internal_trail_vector(boundary_inputs, q2, q4):
    """Pack the seven independent 64-bit masks used for basis independence."""
    values = list(boundary_inputs) + [q2, q4]
    result = 0
    for index, value in enumerate(values):
        if value >> STATE_MASK_BITS:
            raise ValueError("trail masks must be 64-bit")
        result |= value << (STATE_MASK_BITS * index)
    return result
