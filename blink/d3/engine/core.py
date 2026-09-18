"""Cipher coordinates. All masks/differences use LSB-first nibble order."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

SBOX = (1, 0, 9, 3, 8, 5, 14, 7, 4, 2, 12, 11, 10, 15, 6, 13)
PBOX = (0, 5, 11, 10, 1, 6, 4, 13, 2, 12, 9, 15, 3, 7, 14, 8)
RC = (0x13198A2E03707344, 0x082EFA98EC4E6C89, 0xBE5466CF34E90C6C,
      0x3F84D5B5B5470917, 0xD1310BA698DFB5AC)
RCI = (0x0D95748F728EB658, 0x7B54A41DC25A59B5, 0xC5D1B023286085F0,
       0x8E79DCB0603A180E, 0xD71577C1BD314B27)
G = (2, 5, 8, 10)
MASK64 = (1 << 64) - 1
ROOT = Path(__file__).resolve().parents[1]


def pack(v):
    return sum(int(x) << (4*i) for i, x in enumerate(v))


def cells(x, n=16):
    return [(int(x) >> (4*i)) & 15 for i in range(n)]


def sub(x):
    return pack(SBOX[v] for v in cells(x))


def mix(x):
    v = cells(x)
    return pack(v[i ^ 4] ^ v[i ^ 8] ^ v[i ^ 12] for i in range(16))


def perm(x):
    v = cells(x)
    return pack(v[i] for i in PBOX)


def iperm(x):
    v, out = cells(x), [0]*16
    for i, j in enumerate(PBOX):
        out[j] = v[i]
    return pack(out)


def forward(x, k):
    return perm(mix(sub(x)) ^ k)


def inverse(x, k):
    return sub(mix(iperm(x) ^ k))


def box(x, k=0):
    return sub(mix(sub(x)) ^ k)


def connector(x, k, reverse=False):
    if reverse:
        return iperm(mix(iperm(x) ^ k))
    return perm(mix(perm(x)) ^ k)


def extract(x, c):
    return pack((int(x) >> (4*(c+4*r))) & 15 for r in range(4))


def embed(x, c):
    return sum(((int(x) >> (4*r)) & 15) << (4*(c+4*r)) for r in range(4))


def parity(x):
    return int(x).bit_count() & 1


def fwht(v):
    out = np.asarray(v, dtype=np.int64).copy()
    width = 1
    while width < len(out):
        a = out.reshape(-1, 2*width)
        l, r = a[:, :width].copy(), a[:, width:].copy()
        a[:, :width], a[:, width:] = l+r, l-r
        width *= 2
    return out


def sl(v):
    sv = np.asarray(SBOX, dtype=np.uint16)
    return sum(sv[(v >> (4*r)) & 15] << (4*r) for r in range(4))


def ml(v):
    n = [(v >> (4*r)) & 15 for r in range(4)]
    t = n[0] ^ n[1] ^ n[2] ^ n[3]
    return sum((t ^ n[r]) << (4*r) for r in range(4))


def local_indicator(di, do, k=0):
    x = np.arange(65536, dtype=np.uint16)
    y = sl(ml(sl(x)) ^ k)
    return ((y ^ y[x ^ di]) == do).astype(np.int64)


class Space:
    def __init__(self, rows=()):
        self.pivots = {}
        for r in rows:
            self.add(r)

    @property
    def rank(self):
        return len(self.pivots)

    def add(self, r):
        r = int(r)
        while r:
            p = r.bit_length()-1
            if p not in self.pivots:
                self.pivots[p] = r
                return True
            r ^= self.pivots[p]
        return False


class Coordinates:
    def __init__(self, rows):
        self.rows, self.pivots = list(rows), {}
        for i, r in enumerate(self.rows):
            coord = 1 << i
            while r:
                p = r.bit_length()-1
                if p not in self.pivots:
                    self.pivots[p] = (r, coord)
                    break
                other, c = self.pivots[p]
                r, coord = r ^ other, coord ^ c
            if not r:
                raise ValueError('dependent basis')

    def encode(self, r):
        result = 0
        while r:
            other, c = self.pivots[r.bit_length()-1]
            r, result = r ^ other, result ^ c
        return result

    def evaluate(self, x):
        return sum(parity(m & x) << i for i, m in enumerate(self.rows))


def independent(rows):
    s, out = Space(), []
    for r in rows:
        if s.add(r):
            out.append(int(r))
    return out


def solve(equations):
    pivots = {}
    for r, b in equations:
        while r:
            p = r.bit_length()-1
            if p not in pivots:
                pivots[p] = r, b
                break
            other, rhs = pivots[p]
            r, b = r ^ other, b ^ rhs
        if not r and b:
            raise ValueError('inconsistent conditions')
    x = 0
    for p, (r, b) in sorted(pivots.items()):
        x |= (b ^ parity(r & x)) << p
    return x


def fraction_record(p):
    p = Fraction(p)
    return {'numerator': p.numerator, 'denominator': p.denominator, 'text': str(p)}


def fraction_read(p):
    return Fraction(p['numerator'], p['denominator'])


def pattern(rows):
    return pack(8 if v != '.' else 0 for row in rows for v in row)


A = pattern(('.888', '.888', '.8.8', '..8.'))
B = pattern(('....', '....', '..8.', '.8.8'))
C = pattern(('....', '....', '...8', '....'))
D = pattern(('...8', '...8', '....', '...8'))
E = pattern(('88..', '8.8.', '..88', '88.8'))
T = iperm(mix(iperm(A)))
# A specified, symmetric fixed-8 extension, not a global trail search:
# three columns are 3->1, the fourth 2->4. Row symmetry makes any two
# source rows in the fourth column equivalent for the six-zero baseline.
U = sum(embed(int(ml(extract(T, c))), c) for c in range(3)) | embed(0x0088, 3)


def middle_cache():
    path = ROOT/'data'/'local_superbox_spectra.json'
    raw = path.read_bytes()
    records = {r['pattern']: r for r in json.loads(raw)}
    def swap(m):
        v = cells(m, 4)
        v[0], v[2] = v[2], v[0]
        return pack(v)
    u4 = [(embed(swap(m), 3), int(n)) for m, n in records['1-to-3']['nonzero_fourier']]
    u2 = [(mix(embed(swap(m), 3)), int(n)) for m, n in records['3-to-1']['nonzero_fourier']]
    return {'u4': u4, 'u2': u2, 'sha256': hashlib.sha256(raw).hexdigest(),
            'center': Fraction(3, 1 << 20), 'g_mean': Fraction(1, 1 << 12)}


def key_factor(terms, k):
    return Fraction(sum(n * (-1 if parity(m & k) else 1) for m, n in terms), 1 << 32)


def old_g(z):
    return int(box(z) ^ box(z ^ A) == B)


def head_two(x, k1):
    return int(box(x, k1) ^ box(x ^ U, k1) == T)


def head_four(x, k1, k2):
    y, yp = box(x, k1), box(x ^ U, k1)
    return int(y ^ yp == T and old_g(connector(y, k2)))


def encrypt_zero_tweak(p, master, grouped=True):
    keys = [(master >> (128+64*i)) & MASK64 for i in range(5)]
    k, kp = [a ^ b for a, b in zip(keys, RC)], [a ^ b for a, b in zip(keys, RCI)]
    x = p ^ (master & MASK64)
    if not grouped:
        for v in (k[0], k[1], 0, k[2], k[3], k[4]):
            x = forward(x, v)
        x = box(x)
        for v in (kp[0], kp[1], kp[2], 0, kp[3], kp[4]):
            x = inverse(x, v)
    else:
        x = box(x, k[0])
        x = box(connector(x, k[1]))
        x = box(connector(x, k[2]), k[3])
        x = box(connector(x, k[4]))
        x = box(connector(x, kp[0], True), mix(kp[1]))
        x = box(connector(x, kp[2], True))
        x = box(connector(x, kp[3], True), mix(kp[4]))
    return x ^ ((master >> 64) & MASK64)
