"""Exact scheme-A key Fourier queries and a noncancellation rank certificate.

No Cartesian expansion of the two billion-term R spectra is performed.
For each requested coordinate we sum *all* compatible middle terms.
"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import argparse
import json
import math
import random
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from engine.connected import Transfer
from engine.core import Space, MASK64, RC, RCI, embed, extract, mix, cells, pack, parity, fraction_record
from .space_audit import rref


class OuterFourier:
    def __init__(self):
        self.tr = Transfer()
        self.source = ROOT / 'data'
        raw = []
        for col in (0, 3):
            with np.load(self.source / f'local_joint_column_{col}.npz') as data:
                raw.append((data['terms'].copy(), int(data['denominator'])))
        self.tables, self.rows = [], []
        self.denominator = self.tr.denominator
        for col, row in enumerate((2, 0, 1, 0)):
            terms, den = raw[0 if col < 3 else 1]
            table, by_input = {}, defaultdict(list)
            for a, q, n in terms:
                a, q, n = int(a), int(q), int(n)
                if col < 3:
                    aa, qq = cells(a, 4), cells(q, 4)
                    aa[2], aa[row] = aa[row], aa[2]
                    qq[2], qq[row] = qq[row], qq[2]
                    a, q = pack(aa), pack(qq)
                table[(a, q)] = n
                by_input[a].append((q, n))
            self.tables.append(table)
            self.rows.append(by_input)
            self.denominator *= den
        self.index = {int(beta): i for i, beta in enumerate(self.tr.beta)}
        self.live = [i for i in range(len(self.tr.beta))
                     if all(int(self.tr.bcols[c][i]) in self.rows[c] for c in range(4))]

    def coefficient(self, c, physical_inner, side):
        i = self.index.get(c)
        if i is None:
            return 0
        q = mix(physical_inner)
        n = int(self.tr.weights[i])
        for col in range(4):
            n *= self.tables[col].get((int(self.tr.bcols[col][i]), extract(q, col)), 0)
            if not n:
                return 0
        constants = (RC[1], RC[0]) if side == 'head' else (RCI[3], RCI[4])
        if parity((c & constants[0]) ^ (physical_inner & constants[1])):
            n = -n
        return n

    def sample_mask(self, rng, side):
        i = rng.choice(self.live)
        q = sum(embed(rng.choice(self.rows[col][int(self.tr.bcols[col][i])])[0], col)
                for col in range(4))
        c, inner = int(self.tr.beta[i]), mix(q)
        if side == 'head':
            return inner | (c << 64)
        return (c << 192) | (inner << 256)

    def verify_basis(self):
        source = json.loads((self.source / 'four_zero_complete_endpoint_support.json').read_text())
        checked = 0
        for row in source['basis']:
            c = int(row['connector_key_mask'], 0)
            inner = mix(int(row['normalized_outer_key_mask'], 0))
            p = Fraction(row['coefficient']['numerator'], row['coefficient']['denominator'])
            for side, constants in [('head', (RC[1], RC[0])), ('tail', (RCI[3], RCI[4]))]:
                expected = p * (-1 if parity((c & constants[0]) ^ (inner & constants[1])) else 1)
                assert Fraction(self.coefficient(c, inner, side), self.denominator) == expected
                checked += 1
        return checked


class SchemeA:
    def __init__(self):
        self.outer = OuterFourier()
        self.audit = json.loads((HERE / 'output/factor_space_audit.json').read_text())
        self.rh = [int(m, 0) for m in self.audit['head_basis_hex']]
        self.rt = [int(m, 0) for m in self.audit['tail_basis_hex']]
        self.mbasis = [int(m, 0) for m in self.audit['middle_basis_hex']]
        self.outer_space = Space(self.rh + self.rt)
        m = json.loads((HERE / 'middle/output/complete_middle_fourier.json').read_text())
        self.middle_terms = [(int(t['mask'], 0), int(t['coefficient'])) for t in m['terms']]
        self.middle_denominator = m['denominator']
        self.denominator = self.outer.denominator ** 2 * self.middle_denominator
        self.groups = defaultdict(list)
        for mask, n in self.middle_terms:
            self.groups[self.remainder(mask)].append((mask, n))

    def remainder(self, mask):
        for p in sorted(self.outer_space.pivots, reverse=True):
            if (mask >> p) & 1:
                mask ^= self.outer_space.pivots[p]
        return mask

    def coefficient(self, w, exhaustive=False):
        if w < 0:
            raise ValueError('A Fourier mask must be nonnegative')
        if w >> 320 or ((w >> 128) & MASK64):
            return 0, 0
        total, count = 0, 0
        terms = self.middle_terms if exhaustive else self.groups.get(self.remainder(w), ())
        for mask, n in terms:
            candidate = mask ^ w
            h = self.outer.coefficient((candidate >> 64) & MASK64, candidate & MASK64, 'head')
            if not h:
                continue
            t = self.outer.coefficient((candidate >> 192) & MASK64, (candidate >> 256) & MASK64, 'tail')
            if t:
                total += n * h * t
                count += 1
        return total, count


def run(args):
    start = time.monotonic()
    output = HERE / 'output'
    output.mkdir(exist_ok=True)
    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    model = SchemeA()
    checks = model.outer.verify_basis()
    mean_n, mean_terms = model.coefficient(0)
    assert mean_n > 0
    candidates = model.rh + model.rt + model.mbasis
    seed = 464202609152
    rng = random.Random(seed)
    for _ in range(args.max_random):
        # These are sampled candidate coordinates, not an approximate coefficient:
        # each requested coefficient is then summed exactly over every middle term.
        candidates.append(model.outer.sample_mask(rng, 'head') ^
                          rng.choice(model.middle_terms)[0] ^
                          model.outer.sample_mask(rng, 'tail'))
    basis, span = [], Space()
    tested, nonzero, direct_checks = 0, 0, 0
    target = model.audit['factor_union_rank']
    for w in candidates:
        raw, count = model.coefficient(w)
        tested += 1
        if tested <= 12:
            assert (raw, count) == model.coefficient(w, exhaustive=True)
            direct_checks += 1
        if raw:
            nonzero += 1
            if span.add(w):
                basis.append({'mask': hex(w), 'coefficient': fraction_record(Fraction(raw, model.denominator)),
                              'nonzero_contracted_terms': count})
        if tested % 100 == 0 or span.rank == target:
            progress = {'state': 'running', 'candidate_queries': tested, 'certified_lower_rank': span.rank,
                        'sufficient_upper_rank': target, 'elapsed_seconds': time.monotonic() - start}
            save('global_coefficient_status.json', progress)
            print(json.dumps(progress), flush=True)
        if span.rank == target:
            break
    report = {
        'object': 'P_key(K)=p_H(K)*p_M(K)*p_T(K), scheme A, two outer value masks zero',
        'input_and_whitening_scope': 'Uniform block inputs and zero tweak; unused RK3, W1, W2 free',
        'mean_over_all_master_keys': fraction_record(Fraction(mean_n, model.denominator)),
        'nonzero_terms_in_mean_contraction': mean_terms,
        'factor_space_upper_rank': target, 'certified_nonzero_support_lower_rank': span.rank,
        'exact_product_support_rank': span.rank if span.rank == target else None,
        'rank_certificate': basis,
        'mean_query_is_complete': True,
        'coefficient_queries_are_exact': True,
        'candidate_order_seed': seed, 'candidate_queries': tested, 'nonzero_queries': nonzero,
        'outer_basis_coefficient_checks': checks, 'complete_vs_grouped_middle_sum_checks': direct_checks,
        'physical_key_mask_packing': 'RK1 low64, RK2 next64, RK3 next64, RK4 next64, RK5 high64',
        'full_histogram_complete': False, 'global_extrema_complete': False,
        'elapsed_seconds': time.monotonic() - start,
    }
    save('global_key_rank_certificate.json', report)
    save('global_coefficient_status.json', {'state': 'complete', 'rank_certificate_complete': span.rank == target,
                                          'full_distribution_complete': False, 'elapsed_seconds': time.monotonic() - start})
    print(json.dumps({k:v for k,v in report.items() if k != 'rank_certificate'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-random', type=int, default=3000)
    run(parser.parse_args())
