"""Shared state, Fourier-mask and serialization helpers for Xoodoo v4."""

from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .config import Distinguisher


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE.parent / "output"
WORDS64 = 6
LANES32 = 12
STATE_BITS = 384
ZERO_MASK = (0,) * WORDS64

# c[-11], ..., c[0], in Xoodoo[12] order.
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


def find_compiler() -> list[str]:
    for candidate in (shutil.which("gcc"), r"D:\MinGW\bin\gcc.exe"):
        if candidate and Path(candidate).is_file():
            return [str(candidate)]
    raise RuntimeError("No C compiler found (tried gcc and D:\\MinGW\\bin\\gcc.exe)")


def bit_index(x: int, y: int, z: int) -> int:
    if not (0 <= x < 4 and 0 <= y < 3 and 0 <= z < 32):
        raise ValueError("invalid Xoodoo coordinate")
    return z + 32 * (x + 4 * y)


def column_coordinates(column: int) -> tuple[int, int]:
    if not 0 <= column < 128:
        raise ValueError("column must be in [0,128)")
    return divmod(column, 32)


def column_mask(column: int, value: int) -> int:
    if not 0 <= value < 8:
        raise ValueError("local value must be in [0,8)")
    x, z = column_coordinates(column)
    return sum(((value >> y) & 1) << bit_index(x, y, z) for y in range(3))


def state_from_columns(entries: Iterable[tuple[int, int]]) -> int:
    result = 0
    for column, value in entries:
        result ^= column_mask(int(column), int(value))
    return result


def int_to_words(value: int) -> tuple[int, ...]:
    if value < 0 or value >= (1 << STATE_BITS):
        raise ValueError("state does not fit 384 bits")
    return tuple((value >> (64 * index)) & ((1 << 64) - 1) for index in range(WORDS64))


def words_to_int(words: Sequence[int]) -> int:
    words = validate_mask(words)
    return sum(word << (64 * index) for index, word in enumerate(words))


def validate_mask(mask: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(word) for word in mask)
    if len(result) != WORDS64:
        raise ValueError("a Xoodoo mask needs six uint64 words")
    if any(word < 0 or word >= (1 << 64) for word in result):
        raise ValueError("mask word outside uint64")
    return result


def input_difference(config: Distinguisher) -> tuple[int, ...]:
    return int_to_words(state_from_columns(config.input_columns))


def output_mask(config: Distinguisher) -> tuple[int, ...]:
    return int_to_words(column_mask(config.output_column, config.output_mask))


def mask_xor(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(a) ^ int(b) for a, b in zip(left, right))


def mask_to_hex(mask: Sequence[int]) -> list[str]:
    return [f"0x{word:016x}" for word in validate_mask(mask)]


def mask_to_expression(mask: Sequence[int]) -> str:
    value = words_to_int(mask)
    terms: list[str] = []
    for x in range(4):
        for y in range(3):
            for z in range(32):
                if (value >> bit_index(x, y, z)) & 1:
                    terms.append(f"A[{x},{y},{z}]")
    return " + ".join(terms) if terms else "0"


def gf2_rank(masks: Iterable[Sequence[int]]) -> int:
    pivots: dict[int, int] = {}
    for mask in masks:
        value = words_to_int(mask)
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def span_masks(basis: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    result = [ZERO_MASK]
    for vector in basis:
        vector = validate_mask(vector)
        result += [mask_xor(mask, vector) for mask in result]
    return result


def fwht(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if len(result) == 0 or len(result) & (len(result) - 1):
        raise ValueError("FWHT length must be a non-zero power of two")
    step = 1
    while step < len(result):
        for start in range(0, len(result), 2 * step):
            left = result[start : start + step].copy()
            right = result[start + step : start + 2 * step].copy()
            result[start : start + step] = left + right
            result[start + step : start + 2 * step] = left - right
        step *= 2
    return result


def weight(value: float) -> float | None:
    return -math.log2(abs(value)) if value else None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_dir(config: Distinguisher) -> Path:
    return OUTPUT_DIR / config.name


def read_root_dump(path: Path) -> dict[tuple[int, ...], float]:
    data = path.read_bytes()
    if data[:8] != b"XUDRT001":
        raise RuntimeError(f"bad root dump magic: {path}")
    count = struct.unpack_from("<Q", data, 8)[0]
    record_size = 8 + 8 * WORDS64
    if len(data) != 16 + count * record_size:
        raise RuntimeError(f"bad root dump length: {path}")
    result: dict[tuple[int, ...], float] = {}
    offset = 16
    for _ in range(count):
        fields = struct.unpack_from("<d6Q", data, offset)
        result[tuple(fields[1:])] = fields[0]
        offset += record_size
    return result


def config_metadata(
    config: Distinguisher,
    *,
    explicit_ddt_prefix_rounds: int = 0,
    prefix_mode: str | None = None,
) -> dict:
    return {
        "name": config.name,
        "boundary": "chi-input to chi-output",
        "rounds": config.rounds,
        "input_columns": [list(item) for item in config.input_columns],
        "input_difference_words": mask_to_hex(input_difference(config)),
        "output_column": config.output_column,
        "output_local_mask": config.output_mask,
        "output_mask_words": mask_to_hex(output_mask(config)),
        "connector_round_constants": [
            f"0x{value:08x}"
            for value in ROUND_CONSTANTS[12 - config.rounds + 1 :]
        ],
        "explicit_ddt_prefix_rounds": explicit_ddt_prefix_rounds,
        "prefix_mode": prefix_mode or ("identity" if explicit_ddt_prefix_rounds == 0 else None),
        "round_based_mode": (
            "first-order non-zero with joint global U labels and "
            "Fourier-resolved DDT prefix"
            if explicit_ddt_prefix_rounds
            else "first-order non-zero with joint global U labels"
        ),
    }
