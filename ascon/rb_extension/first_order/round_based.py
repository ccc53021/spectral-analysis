"""Python interface and shared helpers for the v4 non-zero engine."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import cipher_config as config


HERE = Path(__file__).resolve().parent
ENGINE_SOURCE = HERE / "round_based_engine.c"
ENGINE_BINARY = HERE / "round_based_engine.exe"
OUTPUT_DIR = HERE / "output"

ZERO_MASK = (0, 0, 0, 0, 0)
DOMAIN_IDS = {
    "permutation": 0,
    "ascon128": 1,
    "ascon128_equal": 2,
    # Ascon-128a has the same initialization-word freedom as Ascon-128;
    # the distinct names keep experiment metadata unambiguous.
    "ascon128a": 1,
    "ascon128a_equal": 2,
}


def _find_compiler() -> list[str]:
    gcc_candidates = [shutil.which("gcc"), r"D:\MinGW\bin\gcc.exe"]
    for path in gcc_candidates:
        if path and Path(path).is_file():
            return [str(path)]
    raise RuntimeError("No C compiler found (tried gcc and D:\\MinGW\\bin\\gcc.exe).")


def build_engine(force: bool = False) -> Path:
    if (
        not force
        and ENGINE_BINARY.exists()
        and ENGINE_BINARY.stat().st_mtime >= ENGINE_SOURCE.stat().st_mtime
    ):
        return ENGINE_BINARY

    compiler = _find_compiler()
    command = compiler + [
        "-std=c99",
        "-O3",
        str(ENGINE_SOURCE),
        "-o",
        str(ENGINE_BINARY),
        "-lm",
    ]
    result = subprocess.run(command, cwd=HERE, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Failed to build round_based_engine.exe:\n"
            + result.stdout
            + result.stderr
        )
    return ENGINE_BINARY


def validate_words(words: Sequence[int], label: str) -> tuple[int, ...]:
    if len(words) != 5:
        raise ValueError(f"{label} must contain exactly five 64-bit words")
    values = tuple(int(x) for x in words)
    if any(x < 0 or x >= (1 << 64) for x in values):
        raise ValueError(f"{label} contains a value outside uint64")
    return values


def allowed_input_bit_positions(domain: str | None = None) -> list[tuple[int, int]]:
    domain = config.INPUT_DOMAIN if domain is None else domain
    if domain == "permutation":
        words = range(5)
    elif domain in ("ascon128", "ascon128_equal", "ascon128a", "ascon128a_equal"):
        words = range(1, 5)
    else:
        raise ValueError(f"unsupported input domain: {domain}")

    if domain in ("ascon128_equal", "ascon128a_equal"):
        # x4 is tied to x3.  Use x3 as the canonical coordinate.
        words = range(1, 4)
    return [(word, bit) for word in words for bit in range(64)]


def validate_input_mask(mask: Sequence[int], domain: str | None = None) -> tuple[int, ...]:
    mask = validate_words(mask, "input value mask")
    domain = config.INPUT_DOMAIN if domain is None else domain
    if domain in ("ascon128", "ascon128_equal", "ascon128a", "ascon128a_equal") and mask[0] != 0:
        raise ValueError("x0 is fixed by the IV; its mask must use the canonical value 0")
    if domain in ("ascon128_equal", "ascon128a_equal") and mask[4] != 0:
        raise ValueError("x4 equals x3; use the canonical x3 mask instead of x4")
    return mask


def mask_xor(left: Sequence[int], right: Sequence[int]) -> tuple[int, ...]:
    return tuple(int(a) ^ int(b) for a, b in zip(left, right))


def unit_mask(word: int, bit: int) -> tuple[int, ...]:
    words = [0] * 5
    words[word] = 1 << bit
    return tuple(words)


def local_mask_value(mask: Sequence[int], column: int) -> int:
    """Return the five-bit S-box-column slice in the authors' bit order."""
    mask = validate_words(mask, "mask")
    return sum(
        ((mask[word] >> (63 - column)) & 1) << (4 - word)
        for word in range(5)
    )


def single_column_input_masks(
    domain: str | None = None,
) -> list[tuple[int, ...]]:
    """Return every legal non-zero input mask supported on one S-box column."""
    domain = config.INPUT_DOMAIN if domain is None else domain
    if domain == "permutation":
        free_words = tuple(range(5))
    elif domain in ("ascon128", "ascon128a"):
        free_words = (1, 2, 3, 4)
    elif domain in ("ascon128_equal", "ascon128a_equal"):
        free_words = (1, 2, 3)
    else:
        raise ValueError(f"unsupported input domain: {domain}")

    result: list[tuple[int, ...]] = []
    for column in range(64):
        physical_bit = 1 << (63 - column)
        for subset in range(1, 1 << len(free_words)):
            words = [0] * 5
            for index, word in enumerate(free_words):
                if (subset >> index) & 1:
                    words[word] = physical_bit
            result.append(tuple(words))
    return result


def mask_to_hex(mask: Sequence[int]) -> list[str]:
    return [f"0x{int(word):016x}" for word in mask]


def mask_to_expression(mask: Sequence[int], domain: str | None = None) -> str:
    domain = config.INPUT_DOMAIN if domain is None else domain
    if domain == "permutation":
        names = ("x0", "x1", "x2", "x3", "x4")
    else:
        names = ("IV", "K0", "K1", "N0", "N1")
    terms: list[str] = []
    for word, value in enumerate(mask):
        for bit in range(64):
            if (int(value) >> bit) & 1:
                # Model index 0 is Ascon S-box column 0, i.e. uint64 bit 63.
                terms.append(f"{names[word]}_{63 - bit}")
    return " + ".join(terms) if terms else "0"


def _evaluate_queries(
    queries: Iterable[tuple[Sequence[int], Sequence[int], Sequence[int]]],
    *,
    rounds: int | None = None,
    begin_round: int | None = None,
    domain: str | None = None,
    difference: Sequence[int] | None = None,
    output_mask: Sequence[int] | None = None,
    iv: int | None = None,
) -> list[float]:
    queries = list(queries)
    rounds = config.ROUNDS if rounds is None else int(rounds)
    begin_round = config.BEGIN_ROUND if begin_round is None else int(begin_round)
    domain = config.INPUT_DOMAIN if domain is None else domain
    difference = validate_words(
        config.INPUT_DIFFERENCE_WORDS if difference is None else difference,
        "input difference",
    )
    output_mask = validate_words(
        config.OUTPUT_MASK_WORDS if output_mask is None else output_mask,
        "output mask",
    )
    iv = config.IV if iv is None else int(iv)
    if domain not in DOMAIN_IDS:
        raise ValueError(f"unsupported input domain: {domain}")
    normalized_queries = []
    for fourier_mask, constraint_mask, constraint_value in queries:
        fourier_mask = validate_input_mask(fourier_mask, domain)
        constraint_mask = validate_input_mask(constraint_mask, domain)
        constraint_value = validate_input_mask(constraint_value, domain)
        if any(value & ~mask for value, mask in zip(constraint_value, constraint_mask)):
            raise ValueError("constraint value contains a bit outside its mask")
        normalized_queries.append((fourier_mask, constraint_mask, constraint_value))
    queries = normalized_queries
    if not queries:
        return []

    engine = build_engine()

    def run_chunk(
        chunk: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
    ) -> list[float]:
        header = (
            f"{rounds} {begin_round} {DOMAIN_IDS[domain]} {iv:x} {len(chunk)}"
        )
        request_lines = [
            header,
            " ".join(f"{word:x}" for word in difference),
            " ".join(f"{word:x}" for word in output_mask),
        ]
        for fourier_mask, constraint_mask, constraint_value in chunk:
            request_lines.append(" ".join(f"{word:x}" for word in fourier_mask))
            request_lines.append(" ".join(f"{word:x}" for word in constraint_mask))
            request_lines.append(" ".join(f"{word:x}" for word in constraint_value))
        result = subprocess.run(
            [str(engine)],
            input="\n".join(request_lines) + "\n",
            cwd=HERE,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout)
        values = [float(line) for line in result.stdout.splitlines() if line.strip()]
        if len(values) != len(chunk):
            raise RuntimeError(
                f"engine returned {len(values)} values for {len(chunk)} queries"
            )
        return values

    worker_count = max(1, min(int(config.ENGINE_WORKERS), len(queries)))
    if worker_count == 1:
        return run_chunk(queries)
    chunk_size = (len(queries) + worker_count - 1) // worker_count
    chunks = [
        queries[i : i + chunk_size]
        for i in range(0, len(queries), chunk_size)
    ]
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        results = list(pool.map(run_chunk, chunks))
    return [value for chunk in results for value in chunk]


def evaluate_masks(
    masks: Iterable[Sequence[int]],
    **kwargs,
) -> list[float]:
    """Evaluate the response to each fixed-U signed input separately.

    The authors' linear-layer approximation contains products and is nonlinear
    in the input correlation vector.  Consequently these separate responses
    are not, in general, the Fourier coefficients of the results obtained by
    running the approximation on conditioned input classes.
    """
    return _evaluate_queries(
        ((mask, ZERO_MASK, ZERO_MASK) for mask in masks), **kwargs
    )


def evaluate_conditionings(
    conditionings: Iterable[tuple[Sequence[int], Sequence[int]]],
    **kwargs,
) -> list[float]:
    """Evaluate unweighted input classes ``(fixed-bit mask, fixed value)``."""
    return _evaluate_queries(
        ((ZERO_MASK, mask, value) for mask, value in conditionings), **kwargs
    )


def gf2_rank(masks: Iterable[Sequence[int]]) -> int:
    pivots: dict[int, int] = {}
    for mask in masks:
        value = sum(int(word) << (64 * i) for i, word in enumerate(mask))
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def span_masks(basis: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    masks = [ZERO_MASK]
    for vector in basis:
        vector = tuple(int(x) for x in vector)
        masks += [mask_xor(mask, vector) for mask in masks]
    return masks


def assignment_conditionings(
    basis: Sequence[Sequence[int]],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return all factorable bit-fixing classes for a unit-mask basis."""
    fixed_mask = ZERO_MASK
    for vector in basis:
        if sum(int(word).bit_count() for word in vector) != 1:
            raise ValueError(
                "conditional initialization currently requires unit input masks"
            )
        fixed_mask = mask_xor(fixed_mask, vector)
    return [(fixed_mask, value) for value in span_masks(basis)]


def fwht(values: Sequence[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    n = len(result)
    if n == 0 or n & (n - 1):
        raise ValueError("FWHT input length must be a non-zero power of two")
    h = 1
    while h < n:
        for start in range(0, n, 2 * h):
            left = result[start : start + h].copy()
            right = result[start + h : start + 2 * h].copy()
            result[start : start + h] = left + right
            result[start + h : start + 2 * h] = left - right
        h <<= 1
    return result


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
