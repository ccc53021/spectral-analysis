"""Shared state and serialization helpers for Ascon joint-U DDT prefixes."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cipher_config as config
import round_based as rb


WORDS = 5
COLUMNS = 64
STATE_BITS = WORDS * 64
WORD_MASK = (1 << 64) - 1
ROOT_MAGIC = b"SLDRT001"


def words_to_int(words: Sequence[int] | Iterable[int]) -> int:
    words = tuple(words)
    if len(words) != WORDS:
        raise ValueError("an Ascon state must contain five words")
    return sum((int(word) & WORD_MASK) << (64 * index) for index, word in enumerate(words))


def int_to_words(state: int) -> tuple[int, ...]:
    state = int(state)
    if state < 0 or state >= 1 << STATE_BITS:
        raise ValueError("state does not fit 320 bits")
    return tuple((state >> (64 * index)) & WORD_MASK for index in range(WORDS))


def mask_to_hex(state: int) -> list[str]:
    return rb.mask_to_hex(int_to_words(state))


def mask_to_expression(state: int, domain: str = "ascon128") -> str:
    return rb.mask_to_expression(int_to_words(state), domain)


def local_from_state(state: int, column: int) -> int:
    words = int_to_words(state)
    return sum(((words[row] >> (63 - column)) & 1) << (4 - row) for row in range(WORDS))


def columns(state: int) -> tuple[int, ...]:
    return tuple(local_from_state(state, column) for column in range(COLUMNS))


def column_mask(column: int, local: int) -> int:
    physical = 1 << (63 - column)
    result = 0
    for row in range(WORDS):
        if (int(local) >> (4 - row)) & 1:
            result ^= physical << (64 * row)
    return result


def active_columns(state: int) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(columns(state)) if value)


def _rotr(value: int, amount: int) -> int:
    amount %= 64
    return ((value >> amount) | (value << (64 - amount))) & WORD_MASK


def _rotl(value: int, amount: int) -> int:
    amount %= 64
    return ((value << amount) | (value >> (64 - amount))) & WORD_MASK


def forward_linear(state: int) -> int:
    words = int_to_words(state)
    return words_to_int(
        word ^ _rotr(word, r0) ^ _rotr(word, r1)
        for word, (r0, r1) in zip(words, config.ROT)
    )


def adjoint_linear(state: int) -> int:
    words = int_to_words(state)
    return words_to_int(
        word ^ _rotl(word, r0) ^ _rotl(word, r1)
        for word, (r0, r1) in zip(words, config.ROT)
    )


def round_constant_state(round_index: int) -> int:
    return int(config.ROUND_CONSTANTS[round_index]) << (64 * 2)


def canonical_original_mask(mask: int, domain: str, iv: int) -> tuple[int, float]:
    """Restrict a physical mask to the original input domain.

    Summing the two Fourier coordinates that differ on a fixed IV bit gives
    the correctly normalized conditional transform; the discarded bit only
    contributes its known sign.
    """
    words = list(int_to_words(mask))
    phase = -1.0 if (words[0] & int(iv)).bit_count() & 1 else 1.0
    if domain == "permutation":
        return int(mask), 1.0
    if domain not in ("ascon128", "ascon128_equal"):
        raise ValueError(f"unsupported input domain {domain!r}")
    words[0] = 0
    if domain == "ascon128_equal":
        words[3] ^= words[4]
        words[4] = 0
    return words_to_int(words), phase


def read_root_dump(path: Path) -> dict[int, float]:
    path = Path(path)
    data = path.read_bytes()
    if len(data) < 16 or data[:8] != ROOT_MAGIC:
        raise ValueError(f"invalid Ascon root dump: {path}")
    count = struct.unpack_from("<Q", data, 8)[0]
    expected = 16 + 48 * count
    if len(data) != expected:
        raise ValueError(f"invalid Ascon root dump length: {path}")
    result: dict[int, float] = {}
    offset = 16
    for _ in range(count):
        coefficient, *words = struct.unpack_from("<d5Q", data, offset)
        offset += 48
        if coefficient:
            mask = words_to_int(words)
            value = result.get(mask, 0.0) + coefficient
            if value:
                result[mask] = value
            else:
                result.pop(mask, None)
    return result


def write_root_dump(path: Path, spectrum: Mapping[int, float]) -> None:
    terms = sorted((int(mask), float(value)) for mask, value in spectrum.items() if value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(ROOT_MAGIC)
        stream.write(struct.pack("<Q", len(terms)))
        for mask, coefficient in terms:
            stream.write(struct.pack("<d5Q", coefficient, *int_to_words(mask)))


def add_scaled(destination: dict[int, float], source: Mapping[int, float], scale: float) -> None:
    if not scale:
        return
    for mask, coefficient in source.items():
        value = destination.get(mask, 0.0) + scale * coefficient
        if value:
            destination[mask] = value
        else:
            destination.pop(mask, None)


def fwht(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result or len(result) & (len(result) - 1):
        raise ValueError("FWHT length must be a non-zero power of two")
    width = 1
    while width < len(result):
        for start in range(0, len(result), 2 * width):
            for index in range(start, start + width):
                left, right = result[index], result[index + width]
                result[index], result[index + width] = left + right, left - right
        width <<= 1
    return result


def span(basis: Iterable[int]) -> list[int]:
    result = [0]
    for vector in basis:
        result += [mask ^ int(vector) for mask in result]
    return result


def weight(value: float) -> float | None:
    return -math.log2(abs(value)) if value else None


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
