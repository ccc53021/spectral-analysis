"""Independent word-oriented four-message Ascon Monte Carlo.

All public masks are five physical uint64 words, including affine constraints.
Packed GF(2) integers inside the sampler store word r in bits 64*r..64*r+63;
they do not use v3's reversed column coordinates. No propagation engine is
imported: each sample evaluates the real four-message permutation.
"""

from __future__ import annotations

import math
import time

import numpy as np

MASK64 = (1 << 64) - 1
ROT = ((19, 28), (61, 39), (1, 6), (10, 17), (7, 41))
RC = (0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B)
ZERO_MASK = (0, 0, 0, 0, 0)


def _integer(value):
    return int(value, 0) if isinstance(value, str) else int(value)


def words(value):
    result = tuple(_integer(v) for v in value)
    if len(result) != 5 or any(v < 0 or v > MASK64 for v in result):
        raise ValueError("masks must contain five uint64 words")
    return result


def words_hex(value):
    return [f"0x{v:016x}" for v in words(value)]


def target_config():
    return {"rounds": 5, "begin_round": 0, "iv": 0x80400C0600000000,
            "delta1_words": (0, 0, 0, 0x8000000000000000, 0x8000000000000000),
            "delta2_words": (0, 0, 0, 0x1000000000000000, 0x1000000000000000),
            "output_words": (0x2000, 0, 0, 0, 0), "equal_columns": (0, 3),
            "mode": "active_equal"}


def _config(config):
    result = target_config()
    if config is not None:
        result.update(config)
    for key in ("delta1_words", "delta2_words", "output_words"):
        result[key] = words(result[key])
    return result


def sbox(rows):
    x0, x1, x2, x3, x4 = rows
    x0, x4, x2 = x0 ^ x4, x4 ^ x3, x2 ^ x1
    t0, t1, t2, t3, t4 = (~x0) & x1, (~x1) & x2, (~x2) & x3, (~x3) & x4, (~x4) & x0
    x0, x1, x2, x3, x4 = x0 ^ t1, x1 ^ t2, x2 ^ t3, x3 ^ t4, x4 ^ t0
    x1, x0, x3, x2 = x1 ^ x0, x0 ^ x4, x3 ^ x2, ~x2
    return tuple(x & MASK64 for x in (x0, x1, x2, x3, x4))


def linear(rows):
    return tuple(x ^ (((x >> a) | (x << (64 - a))) & MASK64)
                 ^ (((x >> b) | (x << (64 - b))) & MASK64)
                 for x, (a, b) in zip(rows, ROT))


def permutation(rows, rounds=5, begin_round=0):
    if not 1 <= rounds or not 0 <= begin_round or begin_round + rounds > 12:
        raise ValueError("round interval must be a nonempty subset of the 12 Ascon rounds")
    rows = tuple(int(x) if np.isscalar(x) else x for x in rows)
    for offset in range(rounds):
        rows = sbox((*rows[:2], rows[2] ^ RC[begin_round + offset], *rows[3:]))
        if offset + 1 < rounds:
            rows = linear(rows)
    return rows


def word_dot(rows, mask):
    folded = np.zeros_like(rows[0], dtype=np.uint64)
    for x, m in zip(rows, mask):
        if m:
            folded ^= x & np.uint64(m)
    for shift in (32, 16, 8, 4, 2, 1):
        folded ^= folded >> shift
    return folded & np.uint64(1)


def _pack(mask):
    return sum(value << (64 * row) for row, value in enumerate(words(mask)))


def _unpack(mask):
    return tuple((mask >> (64 * row)) & MASK64 for row in range(5))


def input_domain(config=None, constraints=()):
    config = _config(config)
    iv = _integer(config["iv"])
    if not 0 <= iv <= MASK64:
        raise ValueError("IV must fit 64 bits")
    mode = config["mode"]
    if mode in ("active_equal", "equal", "partial_equal"):
        columns = config["equal_columns"]
    elif mode in ("unrestricted", "independent"):
        columns = ()
    elif mode == "full_equal":
        columns = range(64)
    else:
        raise ValueError(f"unknown loading mode {mode!r}")
    equations = []
    for col in columns:
        if not 0 <= int(col) < 64:
            raise ValueError("equal_columns must lie in 0..63 (column 0 is the MSB)")
        equations.append(((1 << (192 + 63 - int(col))) | (1 << (256 + 63 - int(col))), 0))
    for mask, rhs in constraints:
        mask, rhs = words(mask), _integer(rhs)
        if rhs not in (0, 1):
            raise ValueError("constraint right-hand sides must be bits")
        equations.append((_pack(mask) & ~MASK64, rhs ^ ((mask[0] & iv).bit_count() & 1)))
    reduced = {}
    for mask, rhs in equations:
        while mask:
            pivot = (mask & -mask).bit_length() - 1
            if pivot not in reduced:
                reduced[pivot] = (mask, rhs)
                break
            other, value = reduced[pivot]
            mask, rhs = mask ^ other, rhs ^ value
        else:
            if rhs:
                raise ValueError("constraints are inconsistent with the loading domain")
    for pivot in sorted(reduced, reverse=True):
        source, value = reduced[pivot]
        for earlier in tuple(reduced):
            if earlier < pivot and (reduced[earlier][0] >> pivot) & 1:
                mask, rhs = reduced[earlier]
                reduced[earlier] = (mask ^ source, rhs ^ value)
    return {"iv": iv, "mode": mode, "rank": len(reduced), "dimension": 256 - len(reduced),
            "equations": [(pivot, _unpack(mask), rhs) for pivot, (mask, rhs) in sorted(reduced.items())]}


def sample_inputs(rng, count, domain):
    rows = [np.full(count, domain["iv"], dtype=np.uint64)]
    rows.extend(rng.bit_generator.random_raw((4, count)))
    for pivot, mask, rhs in domain["equations"]:
        row, bit = divmod(pivot, 64)
        rows[row] ^= (word_dot(rows, mask) ^ np.uint64(rhs)) << bit
    return tuple(rows)


def four_message_bits(rows, config=None):
    config = _config(config)
    result = np.zeros_like(rows[0], dtype=np.uint64)
    d1, d2 = config["delta1_words"], config["delta2_words"]
    for choice in range(4):
        offsets = tuple((d1[r] if choice & 1 else 0) ^ (d2[r] if choice & 2 else 0) for r in range(5))
        message = tuple(x ^ np.uint64(d) for x, d in zip(rows, offsets))
        result ^= word_dot(permutation(message, int(config["rounds"]), int(config["begin_round"])), config["output_words"])
    return result


def monte_carlo(config=None, masks=(ZERO_MASK,), sample_log2=16, seed=0x358, constraints=(), batch_log2=16):
    """Estimate E[(-1)^(four-message output parity + <mask,base input>)]."""
    config = _config(config)
    masks = [words(mask) for mask in masks]
    constraints = tuple(constraints)
    if not 0 <= int(sample_log2) <= 40 or not 0 <= int(batch_log2) <= 22:
        raise ValueError("sample_log2 must be 0..40 and batch_log2 0..22")
    started = time.perf_counter()
    domain = input_domain(config, constraints)
    rng = np.random.default_rng(seed)
    count, batch = 1 << int(sample_log2), 1 << int(batch_log2)
    odd = [0] * len(masks)
    baseline_odd = 0
    for offset in range(0, count, batch):
        rows = sample_inputs(rng, min(batch, count - offset), domain)
        event = four_message_bits(rows, config)
        baseline_odd += int(np.count_nonzero(event))
        for j, mask in enumerate(masks):
            odd[j] += int(np.count_nonzero(event ^ word_dot(rows, mask)))

    def estimate(odd_count):
        signedsum = count - 2 * odd_count
        corr = signedsum / count
        return {"corr": corr, "correlation": corr, "se": math.sqrt(max(0., 1. - corr * corr) / count),
                "signedsum": signedsum, "log2_abs": math.log2(abs(corr)) if corr else None}

    return {"N": count, "seed": int(seed), "batch_log2": int(batch_log2),
            "elapsed_seconds": time.perf_counter() - started, "config": config,
            "domain_rank": domain["rank"], "domain_dimension": domain["dimension"],
            "constraints": [{"mask_words": words_hex(mask), "rhs": int(rhs)} for mask, rhs in constraints],
            "baseline": estimate(baseline_odd),
            "coefficients": [{"mask_words": words_hex(mask), **estimate(n)} for mask, n in zip(masks, odd)],
            "semantics": "true four-message Ascon Fourier coefficients on the specified affine input domain; final linear layer omitted"}
