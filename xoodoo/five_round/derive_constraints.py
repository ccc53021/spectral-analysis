"""Pull the saved six core constraints through one full Xoodoo round.

This is an exact Boolean substitution, not a correlation computation.
The program only reads its source and prints a reproducible JSON record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import xoodoo as xo

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = (
    HERE.parent / "four_round" / "output" / "known_distinguisher_joint"
    / "r4_public_full_extension_core" / "p1_complete" / "distribution_analysis.json"
)


def bits(value: int):
    while value:
        bit = value & -value
        yield bit.bit_length() - 1
        value ^= bit


def rank(vectors):
    pivots = {}
    for value in vectors:
        while value:
            pivot = value.bit_length() - 1
            if pivot in pivots:
                value ^= pivots[pivot]
            else:
                pivots[pivot] = value
                break
    return len(pivots)


def coordinate(index):
    y, rest = divmod(index, 128)
    x, z = divmod(rest, 32)
    return x, y, z


def variable(index, symbol):
    x, y, z = coordinate(index)
    return f"{symbol}[{x},{y},{z}]"


def polynomial_expression(poly, symbol):
    return " + ".join(
        "*".join(variable(i, symbol) for i in bits(term)) if term else "1"
        for term in sorted(poly, key=lambda m: (m.bit_count(), m))
    ) or "0"


def multiply(left, right):
    result = set()
    for a in left:
        for b in right:
            term = a | b  # Boolean ring: x_i squared equals x_i.
            result.symmetric_difference_update((term,))
    return result


def substitute(poly, inputs):
    result = set()
    for term in poly:
        product = {0}
        for index in bits(term):
            product = multiply(product, inputs[index])
        result.symmetric_difference_update(product)
    return result


def evaluate(poly, state):
    return sum((state & term) == term for term in poly) & 1


def counts(poly):
    return {
        "degree": max((term.bit_count() for term in poly), default=0),
        "constant_terms": int(0 in poly),
        "linear_terms": sum(term.bit_count() == 1 for term in poly),
        "quadratic_terms": sum(term.bit_count() == 2 for term in poly),
    }


def build(source, constant, random_tests, include_original_anf=False):
    raw = source.read_bytes()
    saved = json.loads(raw)
    bases = [sum(int(word, 16) << (64 * j) for j, word in enumerate(item["mask_words"]))
             for item in saved["reported_input_basis"]]
    assert len(bases) == rank(bases) == 6
    assert saved["boundary"] == "complete-round input constraints"

    # T = rho_west(theta(X)) xor c. Build its affine coordinates independently
    # from the compact rho_east pullback used below.
    affine_rows = [0] * 384
    for j in range(384):
        for i in bits(xo.pre_chi_linear(1 << j)):
            affine_rows[i] |= 1 << j
    t_inputs = [{1 << j for j in bits(row)} for row in affine_rows]
    for i in bits(constant):
        t_inputs[i].symmetric_difference_update((0,))

    polynomials_t = []
    polynomials_x = []
    records = []
    for i, base in enumerate(bases):
        poly = set()
        q_indices = []
        for index in bits(base):
            x, y, z = coordinate(index)
            if y == 1:
                z = (z - 1) % 32
            elif y == 2:
                x, z = (x - 2) % 4, (z - 8) % 32
            q_indices.append(xo.bit_index(x, y, z))
            a = 1 << xo.bit_index(x, y, z)
            b = 1 << xo.bit_index(x, (y + 1) % 3, z)
            c = 1 << xo.bit_index(x, (y + 2) % 3, z)
            poly.symmetric_difference_update((a, c, b | c))
        expanded = substitute(poly, t_inputs)
        polynomials_t.append(poly)
        polynomials_x.append(expanded)
        record = {
            "index": i,
            "original_core_expression": saved["reported_input_basis"][i]["expression"].replace("X[", "Y["),
            "Q_expression": " + ".join(variable(j, "Q") for j in sorted(q_indices)),
            "T_anf_expression": polynomial_expression(poly, "T"),
            "T_counts": counts(poly),
            "X_counts": counts(expanded),
            "X_anf_sha256": hashlib.sha256(
                json.dumps(sorted(expanded)).encode("ascii")
            ).hexdigest(),
        }
        if include_original_anf:
            record["X_anf_expression"] = polynomial_expression(expanded, "X")
        records.append(record)

    rng = random.Random(2026091106)
    states = [0, xo.STATE_MASK] + [1 << i for i in range(384)]
    states += [rng.getrandbits(384) for _ in range(random_tests)]
    for state in states:
        t = xo.pre_chi_linear(state) ^ constant
        actual = xo.xoodoo_round(state, constant)
        for base, poly_t, poly_x in zip(bases, polynomials_t, polynomials_x):
            expected = (base & actual).bit_count() & 1
            assert evaluate(poly_t, t) == expected
            assert evaluate(poly_x, state) == expected

    return {
        "purpose": "exact substitution of the six core constraints through one preceding full round",
        "source": str(source),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_ddt_prefix_rounds": saved["ddt_prefix_rounds"],
        "source_case": saved["case"],
        "round_constant": f"0x{constant:08x}",
        "round_constant_origin": "first constant of the last-five-round convention" if constant == xo.reduced_round_constants(5)[0] else "explicit override",
        "definitions": {
            "Y": "Y = R(X) = rho_east(chi(rho_west(theta(X)) xor c))",
            "T": "T = rho_west(theta(X)) xor c",
            "Q": "Q[x,y,z] = T[x,y,z] + T[x,(y+2)%3,z] + T[x,(y+1)%3,z]*T[x,(y+2)%3,z]",
            "constraint": "F_i(X) = B_i(R(X)) = a_i, for i=0,...,5",
            "arithmetic": "addition is XOR; multiplication is AND; Boolean ring x*x=x",
        },
        "original_basis_rank": 6,
        "constraint_count_after_pullback": 6,
        "class_count": 64,
        "class_size_log2": 378,
        "class_size_reason": "a bijection preserves every preimage size of the rank-six linear map B",
        "prefix_difference_event_imposed": False,
        "correlations_recomputed": False,
        "constraints": records,
        "validation": {
            "reference": str(Path(xo.__file__).resolve()),
            "states_checked": len(states),
            "constraint_evaluations_per_representation": 6 * len(states),
            "seed": 2026091106,
            "T_substitution_matches_reference": True,
            "X_anf_matches_reference": True,
            "scope": "functional identities only, not statistical correlation verification",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--constant", type=lambda s: int(s, 0), default=xo.reduced_round_constants(5)[0])
    parser.add_argument("--random-tests", type=int, default=512)
    parser.add_argument("--include-original-anf", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.constant <= xo.LANE_MASK or args.random_tests < 0:
        parser.error("constant must be a 32-bit value and random-tests must be nonnegative")
    result = build(args.source, args.constant, args.random_tests, args.include_original_anf)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
