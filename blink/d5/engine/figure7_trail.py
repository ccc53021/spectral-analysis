"""The five two-round Superboxes in CGZ+26 Figure 7 for Blink-128.

The Figure 7 trail covers rounds 2--11.  It is split into five S-M-key-S
Superboxes.  ``8`` and ``G`` are both represented as active cells here; the
outer differences are fixed to 0x8 and compatible internal differences are
summed over G={2,5,8,a} by :mod:`column_spectrum`.

The variable middle values are ``rk5 xor rc5`` in Superbox 2 and
``rk2 xor rc2'`` in Superbox 4.  With tweak t=0, the middle values of
Superboxes 1, 3, and 5 are fixed to zero.
"""

from .linear_layer import STATE_COLUMNS, STATE_ROWS


G_VALUES = (0x2, 0x5, 0x8, 0xA)


def pattern(*rows):
    """Convert four Figure-7 rows to an activity mask."""
    if len(rows) != STATE_ROWS:
        raise ValueError("a Blink-128 pattern must have four rows")
    if any(len(row) != STATE_COLUMNS for row in rows):
        raise ValueError("a Blink-128 pattern must be a 4-by-8 array")
    mask = 0
    for row, text in enumerate(rows):
        for column, symbol in enumerate(text):
            if symbol != ".":
                mask |= 1 << (column + STATE_COLUMNS * row)
    return mask


# Read directly from the colored 4-by-8 cells of Figure 7 (PDF page 16).
A = pattern(
    ".8.8....",
    "8....88.",
    "8..8..8.",
    ".8...8..",
)
C = pattern(
    "....88..",
    ".....8..",
    ".....8..",
    "....88..",
)
D = pattern(
    "....88..",
    "........",
    ".....8..",
    "....8...",
)
E = pattern(
    "......8.",
    "......8.",
    "8.......",
    "8.......",
)


SUPERBOXES = (
    {
        "name": "rounds_2_3",
        "source": A,
        "target": A,
        "direction": "forward",
        "middle": "h1(t)=0",
    },
    {
        "name": "rounds_4_5",
        "source": C,
        "target": D,
        "direction": "forward",
        "middle": "rk5 xor rc5",
    },
    {
        "name": "rounds_6_7",
        "source": E,
        "target": E,
        "direction": "center",
        "middle": "h1(t) xor h2(t)=0",
    },
    {
        "name": "rounds_8_9",
        "source": D,
        "target": C,
        "direction": "inverse",
        "middle": "rk2 xor rc2'",
    },
    {
        "name": "rounds_10_11",
        "source": A,
        "target": A,
        "direction": "inverse",
        "middle": "h2(t)=0",
    },
)


def active_indices(mask):
    return [index for index in range(STATE_ROWS * STATE_COLUMNS) if mask >> index & 1]


def activity_column(state_activity, column):
    if not 0 <= column < STATE_COLUMNS:
        raise ValueError(f"invalid Blink-128 column {column}")
    return sum(
        ((state_activity >> (column + STATE_COLUMNS * row)) & 1) << row
        for row in range(STATE_ROWS)
    )


def superbox_column_pairs(superbox_index):
    row = SUPERBOXES[superbox_index]
    return tuple(
        (
            activity_column(row["source"], column),
            activity_column(row["target"], column),
        )
        for column in range(STATE_COLUMNS)
    )
