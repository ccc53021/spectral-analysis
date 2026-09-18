"""Complete per-column difference aggregation, with exact integer FWHT."""

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import product
import math
import numpy as np
from .core import (G, SBOX, Coordinates, independent, pack, cells, ml, fwht,
                   local_indicator, fraction_record, parity)


@lru_cache(maxsize=None)
def internal_vectors(di, do):
    options = [G if v else (0,) for v in cells(di, 4)]
    result = []
    for ds in product(*options):
        d = pack(ds)
        e = int(ml(d))
        if all(v in G if w else v == 0 for v, w in zip(cells(e, 4), cells(do, 4))):
            result.append((d, e))
    return result


def product_indicator(allowed, transform=False):
    values = np.asarray([pack(v) for v in product(*allowed)], dtype=np.uint16)
    if transform:
        values = ml(values)
    out = np.zeros(65536, dtype=np.int64)
    out[values] = 1
    return out


@lru_cache(maxsize=None)
def counts(di, do):
    """N[k] = number of successful 16-bit inputs for every 16-bit key."""
    out = np.zeros(65536, dtype=np.int64)
    for d, e in internal_vectors(di, do):
        left = [[SBOX[x] for x in range(16) if SBOX[x] ^ SBOX[x ^ a] == b]
                for a, b in zip(cells(di, 4), cells(d, 4))]
        right = [[x for x in range(16) if SBOX[x] ^ SBOX[x ^ a] == b]
                 for a, b in zip(cells(e, 4), cells(do, 4))]
        n = fwht(fwht(product_indicator(left, True))*fwht(product_indicator(right)))
        assert np.all(n % 65536 == 0)
        out += n // 65536
    assert np.all(out >= 0)
    return out


def spectrum_quotient(terms, denominator):
    terms = [(int(m), int(n)) for m, n in terms if n]
    rows = independent(m for m, n in sorted(terms, key=lambda t: (-abs(t[1]), t[0])))
    coord = Coordinates(rows)
    q = np.zeros(1 << len(rows), dtype=np.int64)
    for m, n in terms:
        q[coord.encode(m)] = n
    truth = fwht(q)
    records = []
    for v in sorted(set(map(int, truth))):
        good = list(map(int, np.flatnonzero(truth == v)))
        records.append({'probability': fraction_record(Fraction(v, denominator)),
                        'class_count': len(good), 'syndromes': good})
    return {'rank': len(rows), 'basis': [hex(m) for m in rows],
            'fourier_denominator': denominator, 'nonzero_fourier': terms,
            'histogram': records, 'support_size': len(terms)}


def column_record(di, do):
    n = counts(di, do)
    f = fwht(n)
    r = spectrum_quotient([(int(m), int(f[m])) for m in np.flatnonzero(f)], 1 << 32)
    r.update(input_difference=hex(di), output_difference=hex(do),
             internal_difference_sequences=len(internal_vectors(di, do)),
             minimum_nonzero=fraction_record(Fraction(int(np.min(n[n > 0])), 65536)),
             maximum=fraction_record(Fraction(int(np.max(n)), 65536)),
             mean=fraction_record(Fraction(int(np.sum(n)), 1 << 32)))
    return r


def check_column(di, do, keys):
    n = counts(di, do)
    for k in keys:
        assert int(local_indicator(di, do, k).sum()) == int(n[k])
    return len(keys)*65536
