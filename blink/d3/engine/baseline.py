"""Seven exact two-round factors; all six value connectors projected to zero."""

from collections import Counter
from fractions import Fraction
import json
import math
import random
import time
import numpy as np
from .core import (U, T, A, B, C, D, E, RC, RCI, Space, MASK64, embed, extract,
                   mix, connector, middle_cache, key_factor, fraction_read,
                   fraction_record, parity, solve, local_indicator, head_two,
                   encrypt_zero_tweak, box, old_g, ml)
from .local import column_record, spectrum_quotient, check_column, counts


def factor_hist(f):
    return {fraction_read(h['probability']): Fraction(h['class_count'], 1 << f['rank'])
            for h in f['histogram']}


def multiply_histograms(factors, scale=Fraction(1)):
    out = {Fraction(scale): Fraction(1)}
    for factor in factors:
        nxt = Counter()
        for a, mass_a in out.items():
            for b, mass_b in factor.items():
                nxt[a*b] += mass_a*mass_b
        out = nxt
    return dict(sorted(out.items()))


def histogram_record(hist, dimension=None):
    assert sum(hist.values()) == 1
    out = []
    for p, mass in sorted(hist.items()):
        row = {'probability': fraction_record(p), 'mass': fraction_record(mass)}
        if dimension is not None:
            count = mass*(1 << dimension)
            assert count.denominator == 1
            row['class_count'] = int(count)
        out.append(row)
    return {'histogram': out, 'maximum': fraction_record(max(hist)),
            'minimum_nonzero': fraction_record(min(p for p in hist if p > 0)),
            'mean': fraction_record(sum(p*m for p, m in hist.items())),
            'dimension': dimension}


def map_factor(record, transform, offset, constant, name):
    r = dict(record)
    r['name'], r['master_key_offset'], r['applied_to'] = name, offset, f'rk{(offset-128)//64+1} xor 0x{constant:016x}'
    r['physical_basis'] = [f'0x{transform(int(m, 0)):016x}' for m in record['basis']]
    r['physical_terms'] = [(transform(m), n) for m, n in record['nonzero_fourier']]
    r['constant'] = constant
    return r


def factors():
    m = middle_cache()
    head, tail = [], []
    for c in range(4):
        f = column_record(extract(U, c), extract(T, c))
        head.append(map_factor(f, lambda q, c=c: embed(q, c), 128, RC[0], f'head_column_{c}'))
        g = column_record(extract(T, c), extract(U, c))
        tail.append(map_factor(g, lambda q, c=c: mix(embed(q, c)), 384, RCI[4], f'tail_column_{c}'))
    u4 = map_factor(spectrum_quotient(m['u4'], 1 << 32), lambda q: q, 320, RC[3], 'u4')
    u2 = map_factor(spectrum_quotient(m['u2'], 1 << 32), lambda q: q, 192, RCI[1], 'u2')
    return m, head, [u4, u2], tail


def physical_term(f, m, n):
    coef = Fraction(n, f['fourier_denominator'])
    if parity(m & f['constant']):
        coef = -coef
    return m << f['master_key_offset'], coef


def ordered_product_basis(factors, scale):
    constant = Fraction(scale)
    for f in factors:
        constant *= Fraction(dict(f['physical_terms'])[0], f['fourier_denominator'])
    candidates = []
    for f in factors:
        zero = Fraction(dict(f['physical_terms'])[0], f['fourier_denominator'])
        for m, n in f['physical_terms']:
            mask, coef = physical_term(f, m, n)
            if mask:
                candidates.append((mask, constant*coef/zero, f['name']))
    candidates.sort(key=lambda t: (-abs(t[1]), t[0]))
    basis, s = [], Space()
    for mask, coef, name in candidates:
        if s.add(mask):
            basis.append({'index': len(basis), 'master_key_mask': f'0x{mask:0112x}',
                          'coefficient': fraction_record(coef), 'factor': name,
                          'weight': math.log2(coef.denominator)-math.log2(abs(coef.numerator))})
    return {'rank': s.rank, 'basis': basis,
            'nonzero_fourier_terms': math.prod(len(f['physical_terms']) for f in factors),
            'coefficient_zero': fraction_record(constant),
            'selection': 'complete disjoint-factor support, exact coefficient order; one-factor deviations preserve every weight-threshold span',
            'materialized_full_support': False}


def factor_condition(f):
    positive = [h for h in f['histogram'] if fraction_read(h['probability']) > 0]
    return {'name': f['name'], 'applied_to': f['applied_to'],
            'physical_key_masks': f['physical_basis'],
            'maximum_syndromes': max(positive, key=lambda h: fraction_read(h['probability']))['syndromes'],
            'minimum_positive_syndromes': min(positive, key=lambda h: fraction_read(h['probability']))['syndromes'],
            'syndrome_order': 'basis[0] is least significant bit'}


def extremal_key(factors, maximum=True):
    eq = []
    for f in factors:
        cond = factor_condition(f)
        syndrome = cond['maximum_syndromes' if maximum else 'minimum_positive_syndromes'][0]
        for i, m in enumerate(map(lambda s: int(s, 0), f['physical_basis'])):
            eq.append((m << f['master_key_offset'], ((syndrome >> i) & 1) ^ parity(m & f['constant'])))
    return solve(eq)


def fixed_probability(x, key, m, tail, input_average=False, head=None):
    k2, k4 = (key >> 192) & MASK64, (key >> 320) & MASK64
    value = m['center']*m['g_mean']**2*key_factor(m['u4'], k4 ^ RC[3])*key_factor(m['u2'], k2 ^ RCI[1])
    if input_average:
        for c in range(4):
            k = extract(((key >> 128) & MASK64) ^ RC[0], c)
            value *= Fraction(int(counts(extract(U, c), extract(T, c))[k]), 65536)
    else:
        value *= head_two(x, ((key >> 128) & MASK64) ^ RC[0])
    tk = mix(((key >> 384) & MASK64) ^ RCI[4])
    for c in range(4):
        value *= Fraction(int(counts(extract(T, c), extract(U, c))[extract(tk, c)]), 65536)
    return value


def verify(m, head, middle, tail):
    rng = random.Random(142222222)
    for _ in range(128):
        p, k = rng.getrandbits(64), rng.getrandbits(448)
        assert encrypt_zero_tweak(p, k, True) == encrypt_zero_tweak(p, k, False)
    # Every fixed DIFFERENCE interface must match, even with value mask zero.
    states = [(U,T),(A,B),(C,D),(E,E),(D,C),(B,A),(T,U)]
    for j in range(6):
        assert connector(states[j][1], 0, reverse=(j >= 3)) == states[j+1][0]
    checked = 0
    for c in range(4):
        for di, do in ((extract(U,c),extract(T,c)), (extract(T,c),extract(U,c))):
            checked += check_column(di, do, (0,1,0x1234,0x5678,0xabcd,0xffff))
    # Direct inverse-key coordinate conversion and inverse block counts.
    for c in range(4):
        for k in (0,1,0x1234,0xffff):
            a = counts(extract(U,c),extract(T,c))[k]
            b = counts(extract(T,c),extract(U,c))[int(ml(k))]
            assert a == b
    assert sum(f['rank'] for f in head+middle+tail) == 32
    return {'passed': True, 'fourteen_layer_regrouping_checks': 128,
            'difference_interfaces_checked': 6, 'exhaustive_local_input_cases': checked,
            'inverse_outer_key_mapping_checks': 16, 'old_core_rank': 6,
            'old_core_support_size': 36, 'physical_inverse_key_mapping': 'M^T applied',
            'full_cipher_probability_experiment': False}


def analyze(output, save):
    start = time.monotonic()
    m, head, middle, tail = factors()
    check = verify(m, head, middle, tail)
    save(output/'six_zero_factors.json', {'head': head, 'middle': middle, 'tail': tail,
                                         'cache_sha256': m['sha256']})
    scale = m['center']*m['g_mean']**2
    key_hist = multiply_histograms([factor_hist(f) for f in head+middle+tail], scale)
    support = ordered_product_basis(head+middle+tail, scale)
    key = histogram_record(key_hist, support['rank'])
    head_mean = math.prod(fraction_read(f['mean']) for f in head)
    joint_hist = multiply_histograms([{Fraction(0):1-head_mean, Fraction(1):head_mean}]
                                    +[factor_hist(f) for f in middle+tail], scale)
    joint = histogram_record(joint_hist)
    joint['dimension_note'] = 'Exact masses on uniform 64-bit X and 448-bit master key; no unverified minimum joint Fourier rank claimed.'
    joint['first_block_success_density'] = fraction_record(head_mean)
    assert key['mean'] == joint['mean'] == support['coefficient_zero']
    conditions = {'input': 'X = plaintext xor w1; the old ten-layer entrance mask is zero',
                  'key_only': [factor_condition(f) for f in head+middle+tail],
                  'input_open': [factor_condition(f) for f in middle+tail],
                  'additional_input_open_predicate': 'B_(rk1 xor rc1)(X) xor B_(rk1 xor rc1)(X xor U) = T',
                  'logic': 'Maximum: all required factors at maximum and the input predicate true (if input-open). Minimum positive: all factors at minimum positive and predicate true. Otherwise values are the listed factor products, zero if any factor vanishes.',
                  'representatives': {}}
    for mode, fs in (('key_only',head+middle+tail), ('input_open',middle+tail)):
        for maximum in (True,False):
            name = 'maximum' if maximum else 'minimum_nonzero'
            k = extremal_key(fs,maximum)
            x = 0
            if mode == 'input_open':
                # Choose a valid first-block event at effective first key zero.
                k |= RC[0] << 128
                x = sum(embed(int(np.flatnonzero(local_indicator(extract(U,c),extract(T,c)))[0]),c) for c in range(4))
            value = fixed_probability(x,k,m,tail,mode=='key_only',head)
            target = key[name] if mode=='key_only' else joint[name]
            assert value == fraction_read(target)
            conditions['representatives'][mode+'_'+name] = {'X':f'0x{x:016x}',
                'master_key':f'0x{k:0112x}', 'probability':fraction_record(value)}
    save(output/'six_zero_support.json',support)
    save(output/'six_zero_key_distribution.json',key)
    save(output/'six_zero_joint_distribution.json',joint)
    save(output/'six_zero_conditions.json',conditions)
    save(output/'six_zero_verification.json',check)
    result = {'state':'complete','model':'2+2+2+2+2+2+2, all six connector masks zero',
              'key_only':{k:key[k] for k in ('dimension','maximum','minimum_nonzero','mean')},
              'input_open':{k:joint[k] for k in ('maximum','minimum_nonzero','mean')},
              'maximum_input_open_above_random_fixed_output':fraction_read(joint['maximum'])>Fraction(1,(1<<64)-1),
              'random_fixed_output_baseline':'1/(2^64-1)',
              'exactness_scope':'specified fixed-difference cluster with six zero value connectors, not full-cipher success probability',
              'elapsed_seconds':time.monotonic()-start}
    save(output/'six_zero_result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result
