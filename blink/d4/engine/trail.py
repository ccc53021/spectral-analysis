"""Figure-3 weak-tweak-key trail from Chen--Guo--Zhang (Blink-64).

The ten S layers are the middle layers 3..12 of the 14-S-layer Blink-64
construction.  A cell is either inactive (0), fixed to difference 0x8, or is
in G={0x2,0x5,0x8,0xa}.  Figure 3 is represented as five independent local
S-M-S Superboxes; their Fourier probability functions can later be convolved
without expanding the Cartesian product of all concrete characteristics.
"""

G_VALUES = (0x2, 0x5, 0x8, 0xA)


def pattern(*rows):
    """Convert four visual rows ('.' inactive, other active) to cell mask."""
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("a Blink-64 pattern must be a 4x4 array")
    mask = 0
    for row, text in enumerate(rows):
        for column, symbol in enumerate(text):
            if symbol != ".":
                mask |= 1 << (column + 4 * row)
    return mask


# Extracted from the colored 4x4 vector cells in Figure 3 (PDF page 13).
A = pattern(".888", ".888", ".8.8", "..8.")
B = pattern("....", "....", "..8.", ".8.8")
C = pattern("....", "....", "...8", "....")
D = pattern("...8", "...8", "....", "...8")
E = pattern("88..", "8.8.", "..88", "88.8")


SUPERBOXES = (
    {"name": "rounds_1_2", "source": A, "target": B, "direction": "forward"},
    {"name": "rounds_3_4", "source": C, "target": D, "direction": "forward"},
    {"name": "rounds_5_6", "source": E, "target": E, "direction": "center"},
    {"name": "rounds_7_8", "source": D, "target": C, "direction": "inverse"},
    {"name": "rounds_9_10", "source": B, "target": A, "direction": "inverse"},
)
