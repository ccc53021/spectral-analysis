"""Signed-period prefix connection and exact EVADD graph evaluation."""
from __future__ import annotations
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

import utils
from utils import adjoint_linear, sbox_boolean, span


def affine_solve(equations, dimension):
    """Return particular + nullspace for dot(v,mask)=rhs, or None."""
    pivots = {}
    for mask, rhs in equations:
        mask, rhs = int(mask), int(rhs)
        while mask:
            p = (mask & -mask).bit_length()-1
            if p not in pivots:
                pivots[p] = (mask, rhs)
                break
            old, bit = pivots[p]
            mask ^= old
            rhs ^= bit
        if not mask and rhs:
            return None
    def solve(free, homogeneous):
        value = free
        for p, (mask, rhs) in sorted(pivots.items(), reverse=True):
            if ((mask & value).bit_count() & 1) ^ (0 if homogeneous else rhs):
                value ^= 1 << p
        return value
    basis = [solve(1 << p, True) for p in range(dimension) if p not in pivots]
    return solve(0, False), basis


def signed_periods(counts):
    """Exact support constraints for the Walsh transform of signed counts."""
    counts = {int(k): int(v) for k,v in counts.items() if v}
    if not counts:
        return None
    anchor = next(iter(counts))
    base = counts[anchor]
    equations = []
    for target, value in counts.items():
        if abs(value) != abs(base):
            continue
        shift, sign = anchor ^ target, 1 if value == base else -1
        if all(counts.get(t ^ shift, 0) == sign*n for t,n in counts.items()):
            equations.append((shift, int(sign < 0)))
    return equations


class Prefix:
    def __init__(self, profile):
        self.profile = profile
        self.dimension = profile['second_label_dimension']
        self.key = profile['key_bits']
        parity_bits = profile.get('nonce_parity_bits','00')
        assert len(parity_bits)==2 and set(parity_bits)<={'0','1'}
        self.label_masks = [adjoint_linear(utils.column_words(m, item['column']))
                            for item in profile['local'] for m in item['input_value_masks']]
        # This instance has no second active column touched by e1. If reused for
        # another instance, incorporate that affine T offset instead of ignoring it.
        assert not any((utils.RC[1] >> (63-item['column'])) & 1 for item in profile['local'])
        self.local = []
        self.cache = {}
        for col in range(64):
            iv = ((utils.IV >> (63-col)) & 1) << 4
            if col in (0, 45):
                offset = 0 if col == 0 else 2
                key = (int(self.key[offset]) << 3) | (int(self.key[offset+1]) << 2)
                q = int(parity_bits[0 if col == 0 else 1])
                inputs = [iv | key | (n << 1) | (n ^ q) for n in range(2)]
            else:
                inputs = [iv | n for n in range(16)]
            rc = 4 if (utils.RC[0] >> (63-col)) & 1 else 0
            masks = [utils.column(m, col) for m in self.label_masks]
            labels = [sum(((sbox_boolean(x ^ rc) & m).bit_count() & 1) << j
                          for j,m in enumerate(masks)) for x in inputs]
            self.local.append((inputs, labels))
        self.denominator_exponent = sum((len(x).bit_length()-1) for x,_ in self.local)
        assert self.denominator_exponent == 250

    def local_signed(self, col, u):
        key = col, u
        if key not in self.cache:
            inputs, labels = self.local[col]
            counts = Counter()
            for x,t in zip(inputs,labels):
                counts[t] += 1-2*((x & u).bit_count() & 1)
            counts = {t:n for t,n in counts.items() if n}
            self.cache[key] = counts, signed_periods(counts)
        return self.cache[key]

    def plan(self, mask, max_free=16):
        words = utils.int_to_words(mask)
        local, equations = [], []
        for col in range(64):
            counts, periods = self.local_signed(col, utils.column(words,col))
            if periods is None:
                return [], dict(reason='zero local signed function', support_bound=0)
            local.append(counts)
            equations += periods
        solution = affine_solve(equations,self.dimension)
        if solution is None:
            return [], dict(reason='inconsistent signed periods', support_bound=0)
        origin,basis = solution
        if len(basis)>max_free:
            raise RuntimeError(f'{self.key}: exact connection needs 2^{len(basis)} V values; no truncation applied')
        terms = []
        for offset in span(basis):
            v = origin ^ offset
            integer = 1
            for counts in local:
                integer *= sum(n*(1-2*((t & v).bit_count() & 1)) for t,n in counts.items())
                if not integer:
                    break
            if integer:
                terms.append((v,Fraction(integer,1 << self.denominator_exponent)))
        return terms, dict(reason='complete signed-period support', free_dimension=len(basis),
                           support_bound=1 << len(basis), nonzero_kernel_terms=len(terms))


NODE_DTYPE = np.dtype([('level','<u4'),('lo','<u4'),('hi','<u4'),('ls','<f8'),('hs','<f8')])


class Graph:
    def __init__(self, file, offset=0, size=None):
        file=Path(file)
        with file.open('rb') as stream:
            stream.seek(offset)
            magic,self.dimension,n,root = map(int,np.fromfile(stream,dtype='<u4',count=4))
            assert magic == 0x243eed15
            self.scale = float(np.fromfile(stream,dtype='<f8',count=1)[0])
        expected_size = 24+n*NODE_DTYPE.itemsize
        assert size is None or size == expected_size
        # The saved construction graph includes discarded temporaries. Page in
        # only nodes reachable from the root instead of copying the whole file.
        nodes=np.memmap(file,dtype=NODE_DTYPE,mode='r',offset=offset+24,shape=(n,))
        active, stack = {0,1}, [root]
        while stack:
            u = stack.pop()
            if u in active:
                continue
            active.add(u)
            stack += [int(nodes[u]['lo']),int(nodes[u]['hi'])]
        selected = sorted(active)
        mapping = np.zeros(n,dtype=np.uint32)
        mapping[selected] = np.arange(len(selected),dtype=np.uint32)
        nodes = nodes[selected]
        self.root = int(mapping[root])
        self.levels = nodes['level'].copy()
        self.lo = mapping[nodes['lo']].copy()
        self.hi = mapping[nodes['hi']].copy()
        self.ls, self.hs = nodes['ls'].copy(),nodes['hs'].copy()
        self.gaps_lo = np.zeros(len(nodes),dtype=np.uint64)
        self.gaps_hi = np.zeros(len(nodes),dtype=np.uint64)
        for i in range(2,len(nodes)):
            level = int(self.levels[i])
            assert self.lo[i] < i and self.hi[i] < i
            assert level < self.levels[self.lo[i]] and level < self.levels[self.hi[i]]
            for child, gaps in ((self.lo[i],self.gaps_lo),(self.hi[i],self.gaps_hi)):
                gaps[i] = ((1 << int(self.levels[child]))-1) ^ ((1 << (level+1))-1)

    def exact_query(self, v):
        """Independent rational evaluation, exact relative to saved dyadic edges."""
        if v & ((1 << int(self.levels[self.root]))-1):
            return Fraction()
        values = [Fraction(),Fraction(1)]
        for i in range(2,len(self.levels)):
            low = Fraction() if v & int(self.gaps_lo[i]) else Fraction(float(self.ls[i]))*values[self.lo[i]]
            high = Fraction() if v & int(self.gaps_hi[i]) else Fraction(float(self.hs[i]))*values[self.hi[i]]
            values.append((low + (-high if (v >> int(self.levels[i])) & 1 else high))/2)
        return Fraction(self.scale)*values[self.root]

