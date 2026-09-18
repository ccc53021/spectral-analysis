"""Certified input periods and a sufficient linear quotient for open head H4.

This does not infer an unrestricted distribution from a fixed key. All-key
first-Superbox truth functions are represented by their four exact 16-bit
key-acceptance masks; no random sampling is used for the period certificate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
from engine import core
from engine.connected import Transfer


def all_key_signatures(di: int, do: int) -> np.ndarray:
    """Signature at x exactly describes all k for which B_k gives di->do.

    Four independent 16-entry masks encode allowed key nibbles. The product
    is empty iff at least one mask is empty; every empty product is mapped
    to signature zero. All other product-set encodings are unique.
    """
    x = np.arange(1 << 16, dtype=np.uint16)
    first = core.ml(core.sl(x))
    delta = first ^ first[x ^ di]
    signature = np.zeros(len(x), dtype=np.uint64)
    nonempty = np.ones(len(x), dtype=bool)
    sbox = np.asarray(core.SBOX, dtype=np.uint16)
    for row in range(4):
        z = (first >> (4 * row)) & 15
        d = (delta >> (4 * row)) & 15
        out = (do >> (4 * row)) & 15
        mask = np.zeros(len(x), dtype=np.uint64)
        for k in range(16):
            good = (sbox[z ^ k] ^ sbox[z ^ d ^ k]) == out
            mask |= good.astype(np.uint64) << np.uint64(k)
        signature |= mask << np.uint64(16 * row)
        nonempty &= mask != 0
    signature[~nonempty] = 0
    return signature


def exact_periods(table: np.ndarray) -> list[int]:
    """All xor periods of a complete finite table, using one smallest fibre."""
    unique, first, counts = np.unique(table, return_index=True, return_counts=True)
    index = int(np.argmin(counts))
    anchor = int(first[index])
    candidates = np.flatnonzero(table == unique[index]) ^ anchor
    points = np.arange(len(table))
    return [int(d) for d in candidates if np.array_equal(table, table[points ^ d])]


def annihilator_basis(periods: list[int], width: int) -> list[int]:
    pivots = core.Space(periods).pivots
    result = []
    for free in range(width):
        if free in pivots:
            continue
        m = 1 << free
        for p, row in sorted(pivots.items()):
            m |= core.parity(m & row) << p
        assert all(core.parity(m & p) == 0 for p in periods)
        result.append(m)
    return result


def output_shift(di: int, delta: int, signatures: np.ndarray,
                 masks: list[int]) -> int | None:
    """Certify a constant projected-output shift on the complete J support.

    For each x, successful keys form a direct product of four nibble sets.
    Varying one key nibble must therefore keep that nibble's projected shift
    constant. Check every allowed nibble key, then all x. No key enumeration
    is dropped: Cartesian independence makes these checks exhaustive.
    """
    all_x = np.arange(65536, dtype=np.uint16)
    x = all_x[signatures != 0]
    sig = signatures[signatures != 0]
    v = core.ml(core.sl(x))
    shifted = core.ml(core.sl(x ^ delta))
    s = np.asarray(core.SBOX, dtype=np.uint16)
    total = np.zeros(len(x), dtype=np.uint32)
    for r in range(4):
        projected = np.asarray([sum(core.parity(m & (d << (4*r))) << i
                                   for i, m in enumerate(masks)) for d in range(16)], dtype=np.uint32)
        allowed = (sig >> np.uint64(16*r)) & np.uint64(65535)
        seen = np.zeros(len(x), dtype=bool)
        target = np.zeros(len(x), dtype=np.uint32)
        for k in range(16):
            good = ((allowed >> np.uint64(k)) & 1) != 0
            out_delta = s[((v >> (4*r)) & 15) ^ k] ^ s[((shifted >> (4*r)) & 15) ^ k]
            value = projected[out_delta]
            if np.any(good & seen & (target != value)):
                return None
            target[good & ~seen] = value[good & ~seen]
            seen |= good
        assert np.all(seen)
        total ^= target
    return int(total[0]) if np.all(total == total[0]) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "output/head_quotient.json")
    args = parser.parse_args()
    start = time.perf_counter()
    tr = Transfer()
    assert all((int(a) & 0x8888888888888888) == 0 for a in tr.a)
    records = []
    affine_period_rows = []
    for c in range(4):
        di, do = core.extract(core.U, c), core.extract(core.T, c)
        signatures = all_key_signatures(di, do)
        periods = exact_periods(signatures)
        # Complete input-only periodicity of the all-key first-box event.
        # Keep all 21 global g-interface coordinates, including dependencies
        # in one column, so constant shifts can cancel across columns later.
        output_masks = [core.extract(core.iperm(core.mix(core.iperm(m))), c)
                        for m in tr.g_basis]
        affine = []
        for d in periods:
            shift = output_shift(di, d, signatures, output_masks)
            if shift is not None:
                affine.append((d, shift))
                affine_period_rows.append(core.embed(d, c) | (shift << 64))
        print("column", c, "J input periods", len(periods),
              "constant output-shift periods", len(affine), flush=True)
        # Independent bit-mask representation checks use complete x tables.
        checks = 0
        for key in (0, 1, 0x1234, 0xabcd, 0xffff):
            from_signature = np.ones(1 << 16, dtype=bool)
            for r in range(4):
                allowed = (signatures >> np.uint64(16 * r)) & np.uint64(65535)
                from_signature &= ((allowed >> np.uint64((key >> (4 * r)) & 15)) & 1) != 0
            assert np.array_equal(from_signature, core.local_indicator(di, do, key).astype(bool))
            checks += 1 << 16
        records.append({"column": c, "input_difference": f"0x{di:04x}",
                        "output_difference": f"0x{do:04x}",
                        "all_input_periods": [f"0x{p:04x}" for p in periods],
                        "J_input_projection_rank": 16 - core.Space(periods).rank,
                        "constant_projected_output_shift_periods": [
                            {"input_period": hex(d), "shift_21bit": hex(t)} for d, t in affine],
                        "complete_point_key_checks": checks})
    # Eliminate projected-output shifts to obtain precisely the input-only
    # H4 periods (all four local x/key coordinates vary independently).
    pivots = {}
    input_periods = []
    for row in affine_period_rows:
        while row >> 64:
            p = row.bit_length() - 1
            if p not in pivots:
                pivots[p] = row
                break
            row ^= pivots[p]
        if not row >> 64:
            input_periods.append(row)
    input_periods = core.independent(input_periods)
    input_basis = annihilator_basis(input_periods, 64)
    input_rank = len(input_basis)
    assert input_rank <= 60
    # Coordinates are X|rk1<<64|rk2<<128|rk3<<192|rk4<<256|rk5<<320.
    h_basis = input_basis + [1 << (64 + b) for b in range(64)]
    h_basis += [int(mask) << 128 for mask in tr.beta_basis]
    assert core.Space(h_basis).rank == input_rank + 85
    result = {
        "scope": "Open head H4(X,rk1,rk2), both retained two-round endpoints; all keys, zero tweak",
        "input_projection_rank_exact": input_rank,
        "joint_sufficient_quotient_rank": input_rank + 85,
        "joint_sufficient_rank_is_claimed_minimal": False,
        "input_period_generators": [f"0x{p:016x}" for p in input_periods],
        "input_basis": [f"0x{m:016x}" for m in input_basis],
        "rk1_basis": [f"0x{1 << b:016x}" for b in range(64)],
        "rk2_basis": [f"0x{int(m):016x}" for m in tr.beta_basis],
        "internal_encoding": "X low 64; rk1,rk2,rk3,rk4,rk5 in subsequent 64-bit words",
        "joint_sufficient_basis": [hex(m) for m in h_basis],
        "columns": records,
        "proof": {
            "input_period_lower_bound": "Each independent whole-column input difference exchanges the successful pair, giving an 0/8-only output shift. P-M-P preserves this bit plane, which is annihilated by every nonzero g Fourier mask. H4 is invariant under all four shifts.",
            "input_period_upper_bound": "Averaging H4 over free rk2 gives positive mean(g) times J. Every input-only H4 period must preserve all-key J, whose complete product-key-set signatures give all candidate periods. On J support, each column's projected-output shift must be constant because its x/key variables vary independently. Every allowed key nibble and every x is checked. Finally constant shifts must sum to zero across columns; exact GF(2) elimination imposes that condition.",
            "sufficient_basis": "Input coordinates modulo the certified complete input-only period group, all 64 rk1 bits, and the exact 21 g-interface rk2 coordinates determine H4. This is a certified refinement, not a minimum joint Fourier support claim.",
            "whitening": "Under X=P xor W1, each input mask u becomes the equal pair (P-mask u,W1-mask u); W2 mask remains zero and rank is preserved at zero tweak."
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("basis")}, indent=2))


if __name__ == "__main__":
    main()
