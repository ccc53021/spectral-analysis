"""Exact GF(2) dependency joins for scheme A, without independence assumptions."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from engine.core import Space, independent, mix, MASK64


def rank(rows):
    return Space(rows).rank


def rref(rows):
    pivots = dict(Space(rows).pivots)
    for p in sorted(pivots):
        for q in sorted(pivots):
            if q != p and (pivots[q] >> p) & 1:
                pivots[q] ^= pivots[p]
    return [pivots[p] for p in sorted(pivots)]


def parse_basis(obj):
    for name in ('basis_hex', 'physical_key_basis_hex', 'basis'):
        if name in obj and obj[name] and isinstance(obj[name][0], str):
            return [int(x, 0) for x in obj[name]]
    raise KeyError('No string basis field: ' + str(list(obj)))


def main():
    old = ROOT / 'data/four_zero_complete_endpoint_support.json'
    outer = json.loads(old.read_text(encoding='utf-8'))
    head, tail = [], []
    for row in outer['basis']:
        c = int(row['connector_key_mask'], 0)
        k = mix(int(row['normalized_outer_key_mask'], 0))
        head.append(k | (c << 64))
        tail.append((c << 192) | (k << 256))
    middle_file = HERE / 'middle/output/summary.json'
    middle = json.loads(middle_file.read_text(encoding='utf-8'))
    ms = parse_basis(middle)
    left_words = (1 << 128) - 1
    right_words = (MASK64 << 192) | (MASK64 << 256)
    ml = independent(m & left_words for m in ms)
    mr = independent(m & right_words for m in ms)
    both = head + tail
    report = {
        'model': 'scheme A 4+6+4 restricted; only two outer value masks zero',
        'encoding': 'five physical RK words; RK1 low64; RK3 unused',
        'head_complete_rank': rank(head), 'tail_complete_rank': rank(tail),
        'middle_complete_rank': rank(ms),
        'outer_sum_rank': rank(both),
        'factor_union_rank': rank(both + ms),
        'middle_outer_intersection_rank': rank(both) + rank(ms) - rank(both + ms),
        'middle_left_projection_rank': rank(ml), 'middle_right_projection_rank': rank(mr),
        'left_outer_middle_union_rank': rank(head + ml),
        'right_outer_middle_union_rank': rank(tail + mr),
        'left_intersection_rank': rank(head) + rank(ml) - rank(head + ml),
        'right_intersection_rank': rank(tail) + rank(mr) - rank(tail + mr),
        'key_only_sufficient_basis_hex': [hex(m) for m in rref(both + ms)],
        'head_basis_hex': [hex(m) for m in rref(head)],
        'tail_basis_hex': [hex(m) for m in rref(tail)],
        'middle_basis_hex': [hex(m) for m in rref(ms)],
        'middle_left_projection_basis_hex': [hex(m) for m in rref(ml)],
        'middle_right_projection_basis_hex': [hex(m) for m in rref(mr)],
        'rank_scope': 'union of exact factor dependency spaces is a proved sufficient quotient; minimal support rank of the product is NOT inferred without a noncancellation proof',
    }
    out = HERE / 'output'
    out.mkdir(exist_ok=True)
    (out / 'factor_space_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_hex')}, indent=2))


if __name__ == '__main__':
    main()
