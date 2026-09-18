"""Exact conditional head average used only as a joint-histogram weight.

Marginalizing head-only key coordinates is valid here because pH is a
multiplicity weight, not part of the positive joint probability value.
"""

from fractions import Fraction
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from engine import core
from engine.connected import Transfer
from engine.local import counts


def remainder(row, space):
    while row:
        p = row.bit_length() - 1
        if p not in space.pivots:
            # Reduction cannot stop at a free high bit: remove it temporarily.
            high = 1 << p
            return high | remainder(row ^ high, space)
        row ^= space.pivots[p]
    return 0


def intersection(left, right):
    right_space = core.Space(right)
    pivots = {}
    common = []
    for original in left:
        row = remainder(original, right_space)
        source = original
        while row:
            p = row.bit_length() - 1
            if p not in pivots:
                pivots[p] = row, source
                break
            other, witness = pivots[p]
            row ^= other
            source ^= witness
        if not row:
            common.append(source)
    return core.independent(common)


class HeadCoefficient:
    def __init__(self):
        self.tr = Transfer()
        self.index = {int(beta): i for i, beta in enumerate(self.tr.beta)}
        path = ROOT / "data"
        raw = []
        for c in (0, 3):
            with np.load(path / f"local_joint_column_{c}.npz") as data:
                raw.append((data["terms"].copy(), int(data["denominator"])))

        def swap(value, r):
            cells = core.cells(value, 4)
            cells[2], cells[r] = cells[r], cells[2]
            return core.pack(cells)

        self.columns = []
        for c, r in enumerate((2, 0, 1, 0)):
            data, denominator = raw[0 if c < 3 else 1]
            rows = {}
            for a, q, n in data:
                a, q = int(a), int(q)
                if c < 3:
                    a, q = swap(a, r), swap(q, r)
                rows[(a,q)] = int(n)
            self.columns.append((rows, denominator))

    def __call__(self, physical_mask):
        first = physical_mask & core.MASK64
        connector = (physical_mask >> 64) & core.MASK64
        assert not physical_mask >> 128
        if connector not in self.index:
            return Fraction(0)
        i = self.index[connector]
        value = Fraction(int(self.tr.weights[i]), self.tr.denominator)
        normalized = core.mix(first)
        for c, (rows, denominator) in enumerate(self.columns):
            a, q = int(self.tr.bcols[c][i]), core.extract(normalized, c)
            value *= Fraction(rows.get((a,q), 0), denominator)
        if core.parity(first & core.RC[0]) ^ core.parity(connector & core.RC[1]):
            value = -value
        return value


def main():
    audit = json.loads((HERE.parent / "output/factor_space_audit.json").read_text(encoding="utf-8"))
    h = [int(m,0) for m in audit["head_basis_hex"]]
    m = [int(v,0) for v in audit["middle_basis_hex"]]
    t = [int(v,0) for v in audit["tail_basis_hex"]]
    retained = core.independent(m + t)
    common = intersection(h, retained)
    assert len(retained) == 68 and len(common) == 6
    coefficient = HeadCoefficient()
    terms = []
    direct_checks = 0
    direct_waves = [core.fwht(counts(core.extract(core.T,c), core.extract(core.U,c)))
                    for c in range(4)]
    for code in range(1 << len(common)):
        mask = 0
        for i,basis in enumerate(common):
            if (code >> i) & 1:
                mask ^= basis
        value = coefficient(mask)
        # All common masks happen to have zero connector component. Verify
        # them independently through exact key-count tables, not cached joint
        # spectra: input averaging of R is mean(g) times the key-only J mean.
        assert mask >> 64 == 0
        direct = Fraction(int(coefficient.tr.weights[0]), coefficient.tr.denominator)
        normalized = core.mix(mask)
        for c, wave in enumerate(direct_waves):
            direct *= Fraction(int(wave[core.extract(normalized,c)]), 1 << 32)
        if core.parity(mask & core.RC[0]):
            direct = -direct
        assert direct == value
        direct_checks += 1
        terms.append(value)
    assert terms[0] == Fraction(3, 1 << 41)
    denominator = max(term.denominator for term in terms)
    assert all(denominator % term.denominator == 0 for term in terms)
    integer = np.asarray([int(term * denominator) for term in terms], dtype=np.int64)
    truth = core.fwht(integer)
    assert np.all(truth >= 0)
    result = {
        "scope": "Conditional expectation of pH given the complete pM,pT key quotient. All shared forms and round constants retained.",
        "middle_tail_union_rank": len(retained),
        "head_intersection_rank": len(common),
        "intersection_basis_physical_RK1_to_RK5_hex": [hex(v) for v in common],
        "retained_middle_tail_basis_hex": [hex(v) for v in retained],
        "projected_head_fourier": [core.fraction_record(v) for v in terms],
        "projected_head_truth_table": [core.fraction_record(Fraction(int(v), denominator)) for v in truth],
        "all_projected_head_values_strictly_positive": bool(np.all(truth > 0)),
        "nonzero_fourier_terms": int(np.count_nonzero(integer)),
        "integer_numerators": [int(v) for v in truth],
        "common_denominator": denominator,
        "independent_exact_key_count_table_coefficient_checks": direct_checks,
        "counting_identity": "For v>0, N_joint(v)=2^138 sum_(u in 2^68) E[pH|u] [pM(u)*pT(u)=v] = 3*2^97*N_middle_tail(v); N_joint(0)=2^206-sum_(v>0)N_joint(v).",
        "why_2pow138": "2^206 total joint classes, and 2^68 retained key classes; integrate all input and unretained key coordinates.",
        "maximum_feasibility": "E[pH|u] is the strictly positive constant 3/2^41 for EVERY retained class u. Thus the positive joint maximum/minimum and their frequencies reduce exactly to the complete 68-dimensional middle-tail product distribution; no further head feasibility restriction remains.",
        "complete_joint_histogram_computed": False,
    }
    output = HERE / "output"
    output.mkdir(parents=True, exist_ok=True)
    (output / "projected_head_weight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "retained_middle_tail_basis_hex"}, indent=2))


if __name__ == "__main__":
    main()
