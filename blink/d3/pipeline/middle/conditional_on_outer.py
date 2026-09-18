"""Exact middle-block histograms conditional on its shared outer-factor forms."""

import json
from pathlib import Path
import time

import numpy as np

from .analyze import core, rec
from fractions import Fraction


HERE = Path(__file__).resolve().parent


def combine(rows, code):
    out = 0
    for i, row in enumerate(rows):
        if (code >> i) & 1:
            out ^= row
    return out


def intersection(rows, other):
    """Return independent physical vectors spanning span(rows) ∩ span(other)."""
    ambient = core.Space(other)
    residual_pivots = {}
    common = []
    for i, row in enumerate(rows):
        residual = row
        for p in sorted(ambient.pivots, reverse=True):
            if (residual >> p) & 1:
                residual ^= ambient.pivots[p]
        code = 1 << i
        while residual:
            pivot = residual.bit_length() - 1
            if pivot not in residual_pivots:
                residual_pivots[pivot] = residual, code
                break
            old, oldcode = residual_pivots[pivot]
            residual ^= old
            code ^= oldcode
        if residual == 0:
            common.append(combine(rows, code))
    assert len(core.independent(common)) == len(common)
    return common


def affine_conditions(points, bit_count, physical_basis):
    points = list(map(int, points))
    if not points:
        return {"empty": True}
    base = points[0]
    differences = [x ^ base for x in points]
    affine_rank = len(core.independent(differences))
    annihilator = core.independent(m for m in range(1, 1 << bit_count)
                                  if all(core.parity(m & d) == 0 for d in differences))
    assert len(annihilator) + affine_rank == bit_count
    return {
        "empty": False, "affine_hull_rank": affine_rank,
        "necessary_independent_affine_conditions": [
            {"syndrome_mask": hex(m), "rhs": core.parity(m & base),
             "physical_key_mask": hex(combine(physical_basis, m))} for m in annihilator],
        "points_fill_affine_hull": len(points) == 1 << affine_rank,
        "condition_scope": "Necessary for at least one positive middle completion; not sufficient for an arbitrary remaining-coordinate assignment.",
    }


def main():
    started = time.perf_counter()
    out = HERE / "output"
    audit = json.loads((HERE.parent / "output/factor_space_audit.json").read_text())
    middle = [int(m, 16) for m in audit["middle_basis_hex"]]
    outer = [int(m, 16) for name in ("head_basis_hex", "tail_basis_hex") for m in audit[name]]
    common = intersection(middle, outer)
    assert len(common) == 8
    basis = core.independent(common + middle)
    assert len(basis) == 18 and basis[:8] == common
    coordinates = core.Coordinates(basis)
    fourier = json.loads((out / "complete_middle_fourier.json").read_text())
    denominator = fourier["denominator"]
    wave = np.zeros(1 << len(basis), dtype=np.int64)
    for term in fourier["terms"]:
        wave[coordinates.encode(int(term["mask"], 16))] = term["coefficient"]
    distribution = core.fwht(wave)
    table = distribution.reshape(1 << 10, 1 << 8).T.copy()
    np.save(out / "middle_shared8_remaining10_numerators.npy", table)
    original = np.load(out / "middle_18d_distribution_numerators.npy")
    assert np.array_equal(np.sort(original), np.sort(distribution))
    rows = []
    active = []
    for syndrome, row in enumerate(table):
        values, counts = np.unique(row, return_counts=True)
        positive = row[row > 0]
        if len(positive):
            active.append(syndrome)
        rows.append({
            "shared_syndrome": syndrome,
            "zero_count_among_1024_remaining_classes": int(np.count_nonzero(row == 0)),
            "positive_count_among_1024_remaining_classes": len(positive),
            "histogram": [{"probability": rec(Fraction(int(n), denominator)), "remaining_coset_count": int(count)} for n, count in zip(values, counts)],
            "maximum": rec(Fraction(int(row.max()), denominator)),
            "maximum_remaining_syndromes": list(map(int, np.flatnonzero(row == row.max()))) if len(positive) else [],
            "minimum_nonzero": rec(Fraction(int(positive.min()), denominator)) if len(positive) else None,
        })
    result = {
        "scope": "Complete middle-block conditional distributions on the intersection with the two outer-factor spaces; no independent-marginal multiplication.",
        "physical_key_encoding": "rk1 low64; rk2 next64; rk3 next64; rk4 next64; rk5 high64",
        "shared_rank": 8, "remaining_middle_rank": 10,
        "shared_basis_hex": [hex(m) for m in common],
        "full_basis_shared_first_hex": [hex(m) for m in basis],
        "denominator": denominator,
        "numpy_table_shape": [256, 1024],
        "numpy_axes": ["shared syndrome (basis rows 0..7)", "remaining syndrome (basis rows 8..17)"],
        "positive_shared_syndromes": active,
        "positive_shared_syndrome_count": len(active),
        "all_zero_shared_syndrome_count": 256-len(active),
        "positive_shared_affine_analysis": affine_conditions(active, 8, common),
        "rows": rows,
        "verification": {"marginal_histogram_equals_original_18d_distribution": True,
                         "intersection_rank_equals_gf2_dimension_formula": True},
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "middle_conditional_on_outer_shared8.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "full_basis_shared_first_hex")}, indent=2))


if __name__ == "__main__":
    main()
