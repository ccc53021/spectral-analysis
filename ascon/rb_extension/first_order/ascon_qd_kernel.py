"""Exact 5-bit Fourier-resolved DDT kernel for the Ascon S-box."""

from __future__ import annotations

from fractions import Fraction


SBOX = (
    0x04, 0x0B, 0x1F, 0x14, 0x1A, 0x15, 0x09, 0x02,
    0x1B, 0x05, 0x08, 0x12, 0x1D, 0x03, 0x06, 0x1C,
    0x1E, 0x13, 0x07, 0x0E, 0x00, 0x0D, 0x11, 0x18,
    0x10, 0x0C, 0x01, 0x19, 0x16, 0x0A, 0x0F, 0x17,
)
N = 32


def parity(value: int) -> int:
    return int(value).bit_count() & 1


def _make_numerators() -> tuple:
    table = []
    for difference_in in range(N):
        gamma_rows = []
        for difference_out in range(N):
            u_rows = []
            for value_mask_in in range(N):
                row = []
                for value_mask_out in range(N):
                    total = 0
                    for value in range(N):
                        if SBOX[value] ^ SBOX[value ^ difference_in] != difference_out:
                            continue
                        phase = parity(value_mask_in & value) ^ parity(
                            value_mask_out & SBOX[value]
                        )
                        total += -1 if phase else 1
                    row.append(total)
                u_rows.append(tuple(row))
            gamma_rows.append(tuple(u_rows))
        table.append(tuple(gamma_rows))
    return tuple(table)


NUMERATOR = _make_numerators()
DDT_COUNTS = tuple(tuple(NUMERATOR[d][g][0][0] for g in range(N)) for d in range(N))
DDT_SUPPORT = tuple(tuple(g for g, count in enumerate(row) if count) for row in DDT_COUNTS)


def row_numerators(difference_in: int, difference_out: int, value_mask_out: int):
    return tuple(
        (value_mask_in, NUMERATOR[difference_in][difference_out][value_mask_in][value_mask_out])
        for value_mask_in in range(N)
        if NUMERATOR[difference_in][difference_out][value_mask_in][value_mask_out]
    )


def kernel(difference_in: int, difference_out: int, u: int, v: int) -> Fraction:
    return Fraction(NUMERATOR[difference_in][difference_out][u][v], N)


def validate_identities() -> None:
    for difference_in in range(N):
        for difference_out in range(N):
            assert NUMERATOR[difference_in][difference_out][0][0] == DDT_COUNTS[difference_in][difference_out]
        for u in range(N):
            for v in range(N):
                lat_numerator = sum(
                    -1 if parity(u & value) ^ parity(v & SBOX[value]) else 1
                    for value in range(N)
                )
                assert sum(
                    NUMERATOR[difference_in][difference_out][u][v]
                    for difference_out in range(N)
                ) == lat_numerator

