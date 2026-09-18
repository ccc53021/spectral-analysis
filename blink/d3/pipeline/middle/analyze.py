"""Exact all-key Fourier representation of scheme A's middle six-round block.

Only files below this directory are written. No key is fixed when constructing
the support. A 16-bit joint truth table represents each endpoint factor.
Physical packed key masks use rk1 at bits 0..63, ..., rk5 at 256..319.
"""

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import time

import numpy as np


HERE = Path(__file__).resolve().parent
from engine import complete_middle as fixed

core = fixed.core


def rec(p):
    p = Fraction(p)
    return {"numerator": p.numerator, "denominator": p.denominator,
            "text": str(p)}


def joint_local():
    """f(x,h) for D3 -> C3; only x rows 0,1,3 and key row 2 matter."""
    di, do = core.extract(core.D, 3), core.extract(core.C, 3)
    rows = [r for r, n in enumerate(core.cells(di, 4)) if n]
    output_row, = [r for r, n in enumerate(core.cells(do, 4)) if n]
    assert len(rows) == 3 and rows == [0, 1, 3] and output_row == 2
    word = np.arange(65536, dtype=np.uint16)
    x = sum(((word >> (4*i)) & 15) << (4*r) for i, r in enumerate(rows))
    h = (word >> 12) << (4*output_row)
    y = core.sl(core.ml(core.sl(x)) ^ h)
    yp = core.sl(core.ml(core.sl(x ^ di)) ^ h)
    truth = (y ^ yp == do).astype(np.int64)
    wave = core.fwht(truth)
    modes = []
    for packed in np.flatnonzero(wave):
        packed = int(packed)
        n = sum(((packed >> (4*i)) & 15) << (4*r) for i, r in enumerate(rows))
        hmask = packed >> 12
        keymask = int(core.ml(hmask << (4*output_row)))
        q = core.mix(core.perm(core.embed(n, 3)))
        a = core.perm(q)
        modes.append({"packed": packed, "input_mask": n,
                      "local_physical_key_mask": keymask,
                      "key_mask64": core.embed(keymask, 3),
                      "connector_mask64": q, "center_mask64": a,
                      "coefficient": int(wave[packed])})
    return truth, wave, modes


def calculate(output, verification_cases=8):
    started = time.perf_counter()
    truth, wave, modes = joint_local()
    inputs = sorted({m["input_mask"] for m in modes})
    basis = core.independent(inputs)
    coords = core.Coordinates(basis)
    d = len(basis)
    center_basis = [core.perm(core.mix(core.perm(core.embed(n, 3)))) for n in basis]
    print(json.dumps({"local_joint_nonzero": len(modes), "input_modes": len(inputs),
                      "input_rank": d, "local_joint_rank": len(core.independent(m["packed"] for m in modes))}), flush=True)
    assert 2*d <= 24, "Refuse an unexpectedly large materialized center table."
    center = fixed._center_data()
    tables = []
    for c, data in enumerate(center):
        indices = np.zeros(len(data["good"]), dtype=np.uint64)
        for i, a in enumerate(center_basis):
            ac = np.uint64(core.extract(a, c))
            indices |= fixed.parity64(data["good"].astype(np.uint64) & ac) << np.uint64(i)
            indices |= fixed.parity64(data["output"].astype(np.uint64) & ac) << np.uint64(d+i)
        tables.append(core.fwht(np.bincount(indices.astype(np.intp), minlength=1 << (2*d))))
    central = {}
    for n in inputs:
        for m in inputs:
            index = coords.encode(n) | (coords.encode(m) << d)
            coefficient = 1
            for table in tables:
                coefficient *= int(table[index])
            if coefficient:
                central[n, m] = coefficient
    terms = {}
    for lm in modes:
        for rm in modes:
            coefficient = central.get((lm["input_mask"], rm["input_mask"]), 0)
            if not coefficient:
                continue
            coefficient *= lm["coefficient"] * rm["coefficient"]
            mask = (rm["connector_mask64"]
                    | (rm["key_mask64"] << 64)
                    | (lm["key_mask64"] << 192)
                    | (lm["connector_mask64"] << 256))
            constant = (core.parity(lm["key_mask64"] & core.RC[3])
                        ^ core.parity(lm["connector_mask64"] & core.RC[4])
                        ^ core.parity(rm["key_mask64"] & core.RCI[1])
                        ^ core.parity(rm["connector_mask64"] & core.RCI[0]))
            if constant:
                coefficient = -coefficient
            terms[mask] = terms.get(mask, 0) + coefficient
    terms = {m: n for m, n in terms.items() if n}
    ordered = sorted(terms.items(), key=lambda item: (-abs(item[1]), item[0]))
    physical_basis = core.independent(m for m, n in ordered)
    divisor = 1 << 96
    from math import gcd
    for n in terms.values():
        divisor = gcd(divisor, abs(n))
    denominator = (1 << 96) // divisor
    terms = {m: n // divisor for m, n in terms.items()}
    output.mkdir(parents=True, exist_ok=True)
    rows = [{"mask": hex(m), "coefficient": n} for m, n in sorted(terms.items())]
    (output / "complete_middle_fourier.json").write_text(json.dumps({
        "packing": "rk1 | rk2<<64 | rk3<<128 | rk4<<192 | rk5<<256; round constants already absorbed",
        "denominator": denominator, "terms": rows}, indent=2) + "\n")
    summary = {
        "scope": "Exact all-physical-round-key middle six-round block, uniform block input; no fixed key, no mask truncation.",
        "local_joint_table_length": 65536,
        "local_joint_support_size": len(modes),
        "local_joint_rank": len(core.independent(m["packed"] for m in modes)),
        "local_input_support_size": len(inputs), "local_input_support_rank": d,
        "local_input_basis_hex": [hex(m) for m in basis],
        "center_table_length": 1 << (2*d),
        "center_good_column_counts": [len(t["good"]) for t in center],
        "nonzero_central_input_mask_pairs": len(central),
        "candidate_joint_mode_pairs": len(modes)**2,
        "complete_middle_fourier_terms": len(terms),
        "complete_middle_fourier_support_rank": len(physical_basis),
        "physical_round_key_basis": [hex(m) for m in physical_basis],
        "basis_hex": [hex(m) for m in physical_basis],
        "fourier_denominator": denominator,
        "fourier_coefficient_histogram": dict(sorted(Counter(terms.values()).items())),
        "mean_over_physical_round_keys": rec(Fraction(terms.get(0, 0), denominator)),
        "verification": [],
    }
    quotient = core.Coordinates(physical_basis)
    coefficients = np.zeros(1 << len(physical_basis), dtype=np.int64)
    for m, n in terms.items():
        coefficients[quotient.encode(m)] = n
    distribution = core.fwht(coefficients)
    assert np.all(distribution >= 0)
    assert int(distribution.sum()) == (1 << len(physical_basis)) * terms.get(0, 0)
    np.save(output / "middle_18d_distribution_numerators.npy", distribution)
    values, multiplicities = np.unique(distribution, return_counts=True)
    summary["probability_histogram"] = [
        {"probability": rec(Fraction(int(v), denominator)), "coset_count": int(count)}
        for v, count in zip(values, multiplicities)]
    extrema = {}
    for name, numerator in (("maximum", int(distribution.max())),
                             ("minimum_nonzero", int(distribution[distribution > 0].min()))):
        syndromes = np.flatnonzero(distribution == numerator)
        syndrome = int(syndromes[0])
        packed = core.solve((m, (syndrome >> i) & 1) for i, m in enumerate(physical_basis))
        keys = [(packed >> (64*i)) & core.MASK64 for i in range(5)]
        reference = fixed.compute(keys, include_terms=False, verification=True)
        expected = Fraction(reference["exact_probability"]["numerator"], reference["exact_probability"]["denominator"])
        assert expected == Fraction(numerator, denominator)
        extrema[name] = {"probability": rec(expected), "coset_count": len(syndromes),
                         "all_syndromes": list(map(int, syndromes)),
                         "representative_physical_round_keys": [hex(k) for k in keys],
                         "representative_syndrome": syndrome,
                         "verified_with_existing_full_middle_contraction": True}
    summary["extrema"] = extrema
    summary["physical_round_key_coset_size"] = str(1 << (320-len(physical_basis)))
    rng = np.random.default_rng(2026091506)
    for i in range(verification_cases):
        if i == 0:
            keys = [int(v, 16) for v in ("2874cc1e532a2747", "1062f52829c95f66", "18c078facb7bfa85", "73ae068f503c26b9", "90479091156bcc2d")]
        else:
            keys = [int(v) for v in rng.integers(0, 1 << 64, size=5, dtype=np.uint64)]
        packed = sum(k << (64*j) for j, k in enumerate(keys))
        evaluated = Fraction(sum(-n if core.parity(m & packed) else n for m, n in terms.items()), denominator)
        reference = fixed.compute(keys, include_terms=False, verification=False)
        expected = Fraction(reference["exact_probability"]["numerator"], reference["exact_probability"]["denominator"])
        assert evaluated == expected, (keys, evaluated, expected)
        summary["verification"].append({"keys": [hex(k) for k in keys], "probability": rec(evaluated), "agrees_with_existing_fixed_key_exact_contraction": True})
        print("verified", i, evaluated, flush=True)
    # Independently verify the local reduction against complete 16-bit input
    # tables, including irrelevant-key and inactive-input coordinates.
    for _ in range(8):
        k = int(rng.integers(65536))
        full = core.local_indicator(core.extract(core.D, 3), core.extract(core.C, 3), int(core.ml(k)))
        x = np.arange(65536, dtype=np.uint16)
        reduced = (x & 255) | ((x >> 4) & 0xF00) | ((int(core.ml(k)) >> 8 & 15) << 12)
        assert np.array_equal(full, truth[reduced])
    summary["independent_complete_local_table_checks"] = 8
    summary["elapsed_seconds"] = time.perf_counter() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("physical_round_key_basis", "basis_hex", "fourier_coefficient_histogram", "verification", "extrema")}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--verification-cases", type=int, default=8)
    args = parser.parse_args()
    calculate(args.output, args.verification_cases)
