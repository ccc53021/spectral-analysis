"""Exact local Fourier-resolved DDT kernel for Xoodoo's three-bit chi."""

from __future__ import annotations

from fractions import Fraction


SBOX = (0, 3, 6, 1, 5, 4, 2, 7)
N = 8


def parity(value: int) -> int:
    return int(value).bit_count() & 1


def _make_numerators() -> tuple:
    table = []
    for difference_in in range(N):
        gamma_rows = []
        for difference_out in range(N):
            u_rows = []
            for value_mask_in in range(N):
                v_row = []
                for value_mask_out in range(N):
                    total = 0
                    for value in range(N):
                        if SBOX[value] ^ SBOX[value ^ difference_in] != difference_out:
                            continue
                        phase = parity(value_mask_in & value) ^ parity(
                            value_mask_out & SBOX[value]
                        )
                        total += -1 if phase else 1
                    v_row.append(total)
                u_rows.append(tuple(v_row))
            gamma_rows.append(tuple(u_rows))
        table.append(tuple(gamma_rows))
    return tuple(table)


NUMERATOR = _make_numerators()
KERNEL = tuple(
    tuple(
        tuple(
            tuple(Fraction(NUMERATOR[d][g][u][v], N) for v in range(N))
            for u in range(N)
        )
        for g in range(N)
    )
    for d in range(N)
)

DDT_COUNTS = tuple(
    tuple(NUMERATOR[d][g][0][0] for g in range(N)) for d in range(N)
)
DDT_SUPPORT = tuple(
    tuple(gamma for gamma, count in enumerate(row) if count)
    for row in DDT_COUNTS
)


def row_numerators(
    difference_in: int, difference_out: int, value_mask_out: int
) -> tuple[tuple[int, int], ...]:
    """Return non-zero ``(input value-mask, numerator)`` entries."""
    return tuple(
        (value_mask_in, NUMERATOR[difference_in][difference_out][value_mask_in][value_mask_out])
        for value_mask_in in range(N)
        if NUMERATOR[difference_in][difference_out][value_mask_in][value_mask_out]
    )


def validate_identities() -> None:
    for difference_in in range(N):
        for difference_out in range(N):
            assert NUMERATOR[difference_in][difference_out][0][0] == DDT_COUNTS[
                difference_in
            ][difference_out]
            if difference_out not in DDT_SUPPORT[difference_in]:
                assert all(
                    NUMERATOR[difference_in][difference_out][u][v] == 0
                    for u in range(N)
                    for v in range(N)
                )
        for u in range(N):
            for v in range(N):
                expected = sum(
                    -1
                    if parity(u & value) ^ parity(v & SBOX[value])
                    else 1
                    for value in range(N)
                )
                assert sum(
                    NUMERATOR[difference_in][difference_out][u][v]
                    for difference_out in range(N)
                ) == expected

