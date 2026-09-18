"""Ascon DL.8 word, mask, transform, and serialization helpers.

Column zero is the MSB of a word. A local S-box value orders x0,...,x4
from its MSB to LSB. Packed integers below concatenate literal words;
they deliberately do not use the legacy bit-reversed state representation.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROTATIONS = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))
RC = (0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
      0x78, 0x69, 0x5A, 0x4B)
MASK64 = (1 << 64) - 1
IV = 0x80400C0600000000
NAMES = ("IV", "K0", "K1", "N0", "N1")


def _ror(word, distance):
    return ((word >> distance) | (word << (64 - distance))) & MASK64


def adjoint_linear(words):
    return tuple(w ^ _ror(w, 64 - a) ^ _ror(w, 64 - b)
                 for w, (a, b) in zip(parse_words(words), ROTATIONS))


def sbox_boolean(x):
    a = [(x >> (4 - row)) & 1 for row in range(5)]
    a[0] ^= a[4]
    a[4] ^= a[3]
    a[2] ^= a[1]
    b = [a[row] ^ ((1 ^ a[(row + 1) % 5]) & a[(row + 2) % 5])
         for row in range(5)]
    b[1] ^= b[0]
    b[0] ^= b[4]
    b[3] ^= b[2]
    b[2] ^= 1
    return sum(b[row] << (4 - row) for row in range(5))


def span(basis):
    values = [0]
    for item in basis:
        values += [value ^ int(item) for value in values]
    return values


def fwht(values):
    result = list(values)
    width = 1
    while width < len(result):
        for start in range(0, len(result), 2 * width):
            for index in range(start, start + width):
                left, right = result[index], result[index + width]
                result[index], result[index + width] = left + right, left - right
        width *= 2
    return result


def power(value):
    return "0" if not value else ("-" if value < 0 else "") + f"2^{math.log2(abs(value)):.12f}"


def bits(value, width=6):
    return "".join(str((value >> index) & 1) for index in range(width))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_range(path, offset, size):
    digest = hashlib.sha256()
    remaining = int(size)
    with Path(path).open("rb") as stream:
        stream.seek(int(offset))
        while remaining:
            block = stream.read(min(1 << 20, remaining))
            if not block:
                raise EOFError("graph container ended early")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rref_basis(vectors):
    pivots = {}
    for original in sorted(set(vectors)):
        value = int(original)
        for pivot in sorted(pivots, reverse=True):
            if (value >> pivot) & 1:
                value ^= pivots[pivot]
        if value:
            pivot = value.bit_length() - 1
            for index in list(pivots):
                if (pivots[index] >> pivot) & 1:
                    pivots[index] ^= value
            pivots[pivot] = value
    return [pivots[pivot] for pivot in sorted(pivots)]


def fraction_record(value):
    value = Fraction(value)
    return {
        "value": float(value),
        "power": power(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def mask_expression(mask):
    words = int_to_words(mask)
    return " XOR ".join(
        f"{NAMES[row]}[{col}]"
        for row in range(5)
        for col in range(64)
        if (words[row] >> (63 - col)) & 1
    ) or "0"


def parse_words(values):
    words = tuple(int(x, 0) if isinstance(x, str) else int(x) for x in values)
    if len(words) != 5 or any(x < 0 or x > MASK64 for x in words):
        raise ValueError("expected five unsigned 64-bit words")
    return words


def words_hex(words):
    return [f"0x{x:016x}" for x in parse_words(words)]


def words_to_int(words):
    return sum(word << (64 * row) for row, word in enumerate(parse_words(words)))


def int_to_words(value):
    if int(value) < 0 or int(value) >> 320:
        raise ValueError("packed word mask must fit 320 bits")
    return tuple((int(value) >> (64 * row)) & MASK64 for row in range(5))


def column(words, col):
    if not 0 <= col < 64:
        raise ValueError("column must be in [0,64)")
    return sum(((word >> (63 - col)) & 1) << (4 - row)
               for row, word in enumerate(parse_words(words)))


def column_words(value, col):
    if not 0 <= value < 32 or not 0 <= col < 64:
        raise ValueError("invalid local mask or column")
    return tuple(((value >> (4 - row)) & 1) << (63 - col) for row in range(5))
