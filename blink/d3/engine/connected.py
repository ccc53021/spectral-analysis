"""Exact four-layer endpoint contraction; original four boundaries stay zero.

R(c,k) = E_y J_k(y) g(P M P y xor P c),
J_k(y) = [B_k(y) xor B_k(y xor T) = U].

Both endpoint averages use R. The input-open head is the Boolean joint
event H4(X,k1,k2), not a product of averaged two-layer probabilities.
The complete unrestricted R-key histogram is NOT inferred from a key slice.
"""

from collections import Counter
from fractions import Fraction
from functools import lru_cache
from itertools import product
import json
import math
import random
import time
import numpy as np
from .core import (A, B, U, T, RC, RCI, PBOX, MASK64, Space, Coordinates,
                   independent, embed, extract, mix, iperm, perm, parity, solve,
                   local_indicator, fwht, box, connector, head_four, old_g,
                   head_two, middle_cache, fraction_read, fraction_record, key_factor)
from .local import counts, spectrum_quotient
from .baseline import (factor_hist, multiply_histograms, histogram_record,
                       factors, factor_condition)


def np_perm(v, inverse=False):
    out = np.zeros_like(v, dtype=np.uint64)
    for i, j in enumerate(PBOX):
        source, target = (i, j) if inverse else (j, i)
        out |= ((v >> np.uint64(4*source)) & np.uint64(15)) << np.uint64(4*target)
    return out


def np_mix(v):
    out = np.zeros_like(v, dtype=np.uint64)
    for i in range(16):
        out |= (((v >> np.uint64(4*(i^4))) ^ (v >> np.uint64(4*(i^8)))
                 ^ (v >> np.uint64(4*(i^12)))) & np.uint64(15)) << np.uint64(4*i)
    return out


def signs(masks, value):
    p = masks & np.uint64(value)
    for s in (32,16,8,4,2,1):
        p ^= p >> np.uint64(s)
    return 1-2*(p & np.uint64(1)).astype(np.int64)


class Transfer:
    def __init__(self):
        self.a = np.zeros(1, dtype=np.uint64)
        self.weights = np.ones(1, dtype=np.int64)
        self.denominator = 1
        self.g_basis = []
        self.g_quotients = []
        for c in range(4):
            wave = fwht(local_indicator(extract(A,c), extract(B,c)))
            idx = np.flatnonzero(wave)
            gcd = math.gcd(*(int(wave[m]) for m in idx))
            masks = np.asarray([embed(int(m),c) for m in idx], dtype=np.uint64)
            self.a = (self.a[:,None] ^ masks[None,:]).reshape(-1)
            self.weights = (self.weights[:,None]*(wave[idx]//gcd)[None,:]).reshape(-1)
            self.denominator *= 65536//gcd
            basis = independent(int(m) for m in idx)
            self.g_basis.extend(embed(m,c) for m in basis)
            self.g_quotients.append(spectrum_quotient([(int(m),int(wave[m])) for m in idx],65536))
        self.beta = np_perm(self.a, True)
        b = np_perm(np_mix(self.beta), True)
        self.bcols = [np.asarray(sum(((b >> np.uint64(4*(c+4*r))) & np.uint64(15))
                                     << np.uint64(4*r) for r in range(4)), dtype=np.uint16)
                      for c in range(4)]
        assert all(np.all((v & 0x8888) == 0) for v in self.bcols)
        self.beta_basis = [iperm(m) for m in self.g_basis]
        assert len(self.a) == 140608 and Space(self.beta_basis).rank == 21
        self.coords = Coordinates(self.beta_basis)
        self.indices = np.asarray([self.coords.encode(int(v)) for v in self.beta],dtype=np.int64)

    @lru_cache(maxsize=32)
    def column_wave(self,c,k):
        return fwht(local_indicator(extract(T,c),extract(U,c),k))

    def spectrum(self,k):
        n = self.weights.copy()
        den = self.denominator
        bound = int(np.abs(self.weights).max())
        for c in range(4):
            v = self.column_wave(c,extract(k,c))[self.bcols[c]]
            gcd = math.gcd(*(int(x) for x in np.unique(v)))
            if not gcd:
                return np.zeros_like(n),1
            v = v//gcd
            bound *= int(np.abs(v).max())
            assert bound*len(n) < (1 << 62), 'integer accumulator needs a wider type'
            n *= v
            den *= 65536//gcd
        return n,den

    def evaluate(self,c,k):
        n,den = self.spectrum(k)
        return Fraction(int(np.dot(n,signs(self.beta,c))),den)

    def dense(self,k):
        n,den = self.spectrum(k)
        active = np.flatnonzero(n)
        if not len(active):
            return np.zeros(1,dtype=np.int64),1,Coordinates([]),[]
        basis = independent(int(self.beta[i]) for i in active)
        coords = Coordinates(basis)
        values = np.zeros(1 << len(basis),dtype=np.int64)
        for i in active:
            values[coords.encode(int(self.beta[i]))] = n[i]
        truth = fwht(values)
        assert np.all(truth >= 0)
        hist = Counter(map(int,truth))
        rows = [{'probability':fraction_record(Fraction(v,den)), 'class_count':count}
                for v,count in sorted(hist.items())]
        return truth,den,coords,rows

    def conditional_direct(self,c,k,draws=256,seed=14004):
        """Importance check: sample J-successes EXACTLY uniformly by columns.

        This is a diagnostic estimate of E[g | J], never the exact result.
        Also checks the reverse endpoint's cipher event on every sample.
        """
        rng = random.Random(seed)
        good = [np.flatnonzero(local_indicator(extract(T,col),extract(U,col),extract(k,col)))
                for col in range(4)]
        if any(not len(v) for v in good):
            assert self.evaluate(c,k) == 0
            return {'samples':0,'successes':0,'outer_event_probability':fraction_record(0)}
        successes = 0
        for _ in range(draws):
            y = sum(embed(int(v[rng.randrange(len(v))]),col) for col,v in enumerate(good))
            assert box(y,k)^box(y^T,k) == U
            z = connector(y,c)
            a = old_g(z)
            zp = z^A
            # Tail in forward encryption: old B0^{-1}, reverse connector,
            # and the normalized outer inverse Superbox.
            old_in, old_inp = box(z), box(zp)
            direct = int(old_in^old_inp == B and
                         box(connector(z,c,True),k)^box(connector(zp,c,True),k) == U)
            assert connector(z,c,True) == y
            assert a == direct
            successes += a
        p = math.prod(Fraction(len(v),65536) for v in good)
        return {'samples':draws,'successes':successes,'outer_event_probability':fraction_record(p),
                'estimate_only':fraction_record(p*Fraction(successes,draws))}

    def importance_experiment(self,c,k,samples=1<<20):
        """Bounded statistical check of one endpoint, not a 14-round trial."""
        rng = np.random.default_rng(140042)
        good = [np.flatnonzero(local_indicator(extract(T,col),extract(U,col),extract(k,col)))
                for col in range(4)]
        g_tables = [local_indicator(extract(A,col),extract(B,col)).astype(bool) for col in range(4)]
        p_outer = math.prod(Fraction(len(v),65536) for v in good)
        hits,done = 0,0
        while done<samples:
            n = min(1<<16,samples-done)
            y = np.zeros(n,dtype=np.uint64)
            for col,v in enumerate(good):
                chosen = rng.choice(v,n).astype(np.uint64)
                for r in range(4):
                    y |= ((chosen >> np.uint64(4*r))&np.uint64(15)) << np.uint64(4*(col+4*r))
            z = np_perm(np_mix(np_perm(y))) ^ np.uint64(perm(c))
            accepted = np.ones(n,dtype=bool)
            for col,table in enumerate(g_tables):
                local = sum(((z >> np.uint64(4*(col+4*r)))&np.uint64(15)) << np.uint64(4*r) for r in range(4))
                accepted &= table[local]
            hits += int(accepted.sum())
            done += n
        exact = self.evaluate(c,k)
        conditional = float(exact/p_outer)
        expected = samples*conditional
        sd = math.sqrt(samples*conditional*(1-conditional))
        assert abs(hits-expected)<=8*sd
        return {'scope':'four-round endpoint only, uniform importance samples from its outer J event; NOT full-cipher pairs',
                'samples':samples,'hits':hits,'expected_hits_from_exact_R':expected,
                'standardized_deviation':(hits-expected)/sd,
                'exact_R':fraction_record(exact),
                'estimated_R':fraction_record(p_outer*Fraction(hits,samples)),
                'outer_event_probability':fraction_record(p_outer),
                'seed':140042,'passed_eight_sigma_sanity_check':True}


def local_inner_product_checks(tr):
    """Exact local Parseval contractions, with active masks on both sides."""
    rng = random.Random(414)
    checked = 0
    for c in range(4):
        for k in (0,1,0x1234,0x5678):
            truth = local_indicator(extract(T,c),extract(U,c),k)
            wave = tr.column_wave(c,k)
            # Fourier inverse at arbitrary points and masked weighted sums.
            for _ in range(4):
                point = rng.randrange(65536)
                total = int(np.dot(wave,signs(np.arange(65536,dtype=np.uint64),point)))
                assert total == 65536*int(truth[point])
                checked += 1
    return checked


def analyze(output,save):
    start = time.monotonic()
    tr = Transfer()
    m,head,mid,tail = factors()
    keys = [int(q) for q,n in m['u4'] if q]
    assert Space(tr.beta_basis+keys).rank == 24
    assert Space(tr.beta_basis+[int(q) for q,n in m['u2'] if q]).rank == 24
    from .support import analyze as analyze_support
    support = analyze_support(tr,output,save)
    tail_q_basis = independent(int(r['normalized_outer_key_mask'],0) for r in support['basis'])
    assert len(tail_q_basis) == 31
    checks = {'passed':True, 'local_exact_walsh_inversion_checks':local_inner_product_checks(tr),
              'R_connector_key_projection_rank':21,'R_and_shared_middle_key_rank':24,
              'shared_key_independence_justification':'full rank 21+3 of explicit physical master-key forms; no independent-round-key assumption'}
    rng = random.Random(42224)
    for _ in range(256):
        x,k1,k2 = [rng.getrandbits(64) for _ in range(3)]
        y,yp = box(x,k1),box(x^U,k1)
        z,zp = connector(y,k2),connector(yp,k2)
        direct = int(y^yp==T and z^zp==A and box(z)^box(zp)==B)
        assert direct == head_four(x,k1,k2)
    checks['head_four_cipher_event_checks'] = 256
    print('Exact 4-layer contraction built: streaming 140608 old g terms, no zeroing of the new connector.',flush=True)
    # The complete R support proves that only these 31 key forms matter.
    # Setting them to zero suffices; all other 33 rk5 bits remain free.
    # kappa=0 is a REPRESENTATIVE of that whole 2^-31 master-key family.
    kappa = 0
    values,den,coords,rows = tr.dense(kappa)
    dimension = len(coords.rows)
    tail_hist = {fraction_read(r['probability']):Fraction(r['class_count'],1 << dimension) for r in rows}
    save(output/'four_zero_tail_slice.json', histogram_record(tail_hist,dimension))
    print(f'Tail slice complete: rank={dimension}, distinct probabilities={len(rows)}',flush=True)
    wave_n,wave_den = tr.spectrum(kappa)
    mean = Fraction(int(wave_n[0]),wave_den)
    assert mean == sum(p*mass for p,mass in tail_hist.items())
    outer_probability = math.prod(Fraction(int(counts(extract(T,c),extract(U,c))[0]),65536) for c in range(4))
    assert mean == outer_probability*m['g_mean']
    p_head = math.prod(fraction_read(f['mean']) for f in head)*m['g_mean']
    joint_hist = multiply_histograms([{Fraction(0):1-p_head,Fraction(1):p_head}]
                                    +[factor_hist(f) for f in mid]+[tail_hist],m['center'])
    joint = histogram_record(joint_hist)
    # Input-averaged complete slice fixes BOTH effective outer keys to zero.
    # The product can have millions of DISTINCT rational probabilities.
    # Preserve its exact independent factors instead of a huge flattened map.
    kh = [tail_hist,tail_hist]+[factor_hist(f) for f in mid]
    key_dist = {'dimension':2*dimension+6,
                'representation':'complete independent product distribution, not flattened',
                'factors':[histogram_record(h) for h in kh],
                'scale':fraction_record(m['center']),
                'maximum':fraction_record(m['center']*math.prod(max(h) for h in kh)),
                'minimum_nonzero':fraction_record(m['center']*math.prod(min(p for p in h if p>0) for h in kh)),
                'mean':fraction_record(m['center']*math.prod(sum(p*mass for p,mass in h.items()) for h in kh))}
    conditions = {'slice_scope':'Input-open histogram: the listed 31 linear forms on rk5 xor rc5prime equal zero; all other key bits and X are uniform. Key-only double slice additionally imposes the analogous 31 forms on rk1 xor rc1. These are NOT unrestricted full-key-space distributions.',
                  'input':'X = plaintext xor w1',
                  'tail_key_equations':{'applied_to':f'rk5 xor 0x{RCI[4]:016x}',
                                        'masks':[f'0x{mix(q):016x}' for q in tail_q_basis],
                                        'right_hand_sides':[0]*31, 'rank':31,
                                        'fraction_of_master_keys':fraction_record(Fraction(1,1<<31))},
                  'additional_key_only_double_slice_equations':{'applied_to':f'rk1 xor 0x{RC[0]:016x}',
                                        'masks':[f'0x{mix(q):016x}' for q in tail_q_basis],
                                        'right_hand_sides':[0]*31, 'rank':31},
                  'head_joint_predicate':'[B_(rk1 xor rc1)(X) xor B_(rk1 xor rc1)(X xor U)=T] AND g(P M P B_(rk1 xor rc1)(X) xor P(rk2 xor rc2))=1',
                  'tail_basis_on_rk4_xor_rc4prime':[f'0x{b:016x}' for b in coords.rows],
                  'tail_constant':f'0x{RCI[3]:016x}',
                  'key_only_head_basis_on_rk2_xor_rc2':[f'0x{b:016x}' for b in coords.rows],
                  'key_only_head_constant':f'0x{RC[1]:016x}',
                  'middle_factors':[factor_condition(f) for f in mid],
                  'tail_syndrome_order':'basis[0] is least significant bit',
                  'maximum_syndromes_file':'four_zero_slice_extrema.npz: maximum',
                  'minimum_positive_syndromes_file':'four_zero_slice_extrema.npz: minimum_positive',
                  'all_tail_values_file':'four_zero_slice_values.npz: values / denominator',
                  'logic':{'input_open':'Within the 31-condition tail-key slice, maximum/minimum positive requires H4=1, both middle factors at their respective extrema, and the tail syndrome in its corresponding complete extremal set.',
                           'key_only_double_slice':'Within the additional 31-condition head-key slice, both the head and tail R syndromes must be in the corresponding extrema set, and both middle factors must be at their corresponding extrema. No input predicate is imposed after input averaging.'}}
    nonzero = values[values>0]
    max_syndromes = np.flatnonzero(values==np.max(values))
    min_syndromes = np.flatnonzero(values==np.min(nonzero))
    np.savez_compressed(output/'four_zero_slice_extrema.npz',maximum=max_syndromes,minimum_positive=min_syndromes)
    np.savez_compressed(output/'four_zero_slice_values.npz',values=values,denominator=str(den))
    # Nonzero full-key witnesses: choose a successful first new block then
    # solve the old g forms in rk2 simultaneously with the cached u2 forms.
    representatives = {}
    for maximum,syndromes in ((True,max_syndromes),(False,min_syndromes)):
        name = 'maximum' if maximum else 'minimum_nonzero'
        eq4 = [(mask,((int(syndromes[0]) >> i)&1)^parity(mask & RCI[3])) for i,mask in enumerate(coords.rows)]
        u4,u2 = [factor_condition(f) for f in mid]
        target4 = u4['maximum_syndromes' if maximum else 'minimum_positive_syndromes'][0]
        eq4 += [(int(mask,0),((target4 >> i)&1)^parity(int(mask,0)&RC[3])) for i,mask in enumerate(u4['physical_key_masks'])]
        rk4 = solve(eq4)
        x = sum(embed(int(np.flatnonzero(local_indicator(extract(U,c),extract(T,c)))[0]),c) for c in range(4))
        y = box(x)
        z0 = connector(y,RC[1])
        eq2 = []
        for c,record in enumerate(tr.g_quotients):
            good = [r for r in record['histogram'] if fraction_read(r['probability']) == 1][0]['syndromes'][0]
            eq2.extend((iperm(embed(int(mask,0),c)),((good >> i)&1)^parity(embed(int(mask,0),c)&z0))
                       for i,mask in enumerate(record['basis']))
        target2 = u2['maximum_syndromes' if maximum else 'minimum_positive_syndromes'][0]
        eq2 += [(int(mask,0),((target2 >> i)&1)^parity(int(mask,0)&RCI[1])) for i,mask in enumerate(u2['physical_key_masks'])]
        rk2 = solve(eq2)
        assert head_four(x,0,rk2^RC[1]) == 1
        actual = m['center']*key_factor(m['u4'],rk4^RC[3])
        actual *= key_factor(m['u2'],rk2^RCI[1])*tr.evaluate(rk4^RCI[3],0)
        assert actual == fraction_read(joint[name])
        key = (RC[0]<<128)|(rk2<<192)|(rk4<<320)|(RCI[4]<<384)
        representatives[name] = {'X':f'0x{x:016x}','master_key':f'0x{key:0112x}',
                                  'probability':fraction_record(actual),
                                  'tail_probability':fraction_record(tr.evaluate(rk4^RCI[3],0))}
        head_eq = [(mask,((int(syndromes[0]) >> i)&1)^parity(mask&RC[1])) for i,mask in enumerate(coords.rows)]
        head_eq += [(int(mask,0),((target2 >> i)&1)^parity(int(mask,0)&RCI[1])) for i,mask in enumerate(u2['physical_key_masks'])]
        averaged_rk2 = solve(head_eq)
        averaged_value = (m['center']*key_factor(m['u4'],rk4^RC[3])
                          *key_factor(m['u2'],averaged_rk2^RCI[1])
                          *tr.evaluate(averaged_rk2^RC[1],0)*tr.evaluate(rk4^RCI[3],0))
        assert averaged_value == fraction_read(key_dist[name])
        averaged_key = (RC[0]<<128)|(averaged_rk2<<192)|(rk4<<320)|(RCI[4]<<384)
        representatives['key_only_double_slice_'+name] = {'master_key':f'0x{averaged_key:0112x}',
                                                          'probability':fraction_record(averaged_value)}
    conditions['representatives'] = representatives
    # Independent random-key point evaluation against exact dense FWHT.
    for _ in range(128):
        c = rng.getrandbits(64)
        assert tr.evaluate(c,0) == Fraction(int(values[coords.evaluate(c)]),den)
    checks['point_contraction_equals_dense_fwht_checks'] = 128
    # Check that freeing all 33 unused normalized-key directions leaves the
    # ENTIRE endpoint connector spectrum unchanged, not just one probability.
    for _ in range(16):
        raw = rng.getrandbits(64)
        period = raw ^ solve([(q,parity(q&raw)) for q in tail_q_basis])
        n2,den2 = tr.spectrum(period)
        common = max(wave_den,den2)
        assert np.array_equal(wave_n*(common//wave_den),n2*(common//den2))
    checks['whole_endpoint_spectrum_key_period_checks'] = 16
    checks['tail_mean_equals_g_mean_times_outer_mean'] = fraction_record(mean)
    checks['conditional_cipher_event_diagnostic'] = tr.conditional_direct(0,0,1024)
    max_rk4 = (int(representatives['maximum']['master_key'],0) >> 320)&MASK64
    checks['endpoint_importance_experiment'] = tr.importance_experiment(max_rk4^RCI[3],0)
    result = {'state':'exact_model_and_complete_slice_verified',
              'model':'4+2+2+2+4; only original four connector masks zero',
              'endpoint':'R(c,k)=E_y J_k(y) g(P M P y xor P c)',
              'head_input_open':'H4(X,rk1 xor rc1,rk2 xor rc2)',
              'tail':'R(rk4 xor rc4prime,M(rk5 xor rc5prime))',
              'full_function':'H4 * u4(rk4 xor rc4) * (3/2^20) * u2(rk2 xor rc2prime) * R_tail',
              'tail_slice_rank':dimension,'tail_slice_support_size':int(np.count_nonzero(wave_n)),
              'unrestricted_endpoint_support_rank':support['rank'],
              'unrestricted_key_only_fourteen_layer_rank':support['full_key_only_fourteen_layer_rank'],
              'slice_key_constraint_rank':31,
              'slice_fraction_of_master_keys':fraction_record(Fraction(1,1<<31)),
              'unrestricted_full_key_mean':fraction_record(Fraction(27,1<<114)),
              'tail_slice_distribution':histogram_record(tail_hist,dimension),
              'input_open_slice':joint,'key_only_double_slice':key_dist,
              'unrestricted_full_key_distribution_complete':False,
              'unrestricted_extrema_claimed':False,
              'elapsed_seconds':time.monotonic()-start}
    save(output/'four_zero_conditions.json',conditions)
    save(output/'four_zero_verification.json',checks)
    save(output/'four_zero_result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('tail_slice_distribution','input_open_slice','key_only_double_slice')},indent=2),flush=True)
    print('Input-open slice maximum:',joint['maximum']['text'],'minimum positive:',joint['minimum_nonzero']['text'],flush=True)
    return result
