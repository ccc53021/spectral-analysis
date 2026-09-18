from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict

@dataclass
class ProbabilitySet:
    average_prob: float
    probability: dict[float] = field(default_factory=dict)

@dataclass
class Solution:
    objective_value: float
    sign: int
    state_mask_int_vars: int
    state_all_mask_int_vars: int
    keys_mask_mod2_int_vars: int
    state_mask_vars: dict[tuple, int] = field(default_factory=dict)
    keys_mask_vars: dict[tuple, int] = field(default_factory=dict)
    keys_mask_sum_vars: dict[tuple, int] = field(default_factory=dict)
    keys_mask_quotient_vars: dict[tuple, int] = field(default_factory=dict)
    keys_mask_mod2_vars:dict[tuple, int] = field(default_factory=dict)
    state_corr_vars: dict[tuple, float] = field(default_factory=dict)

def int2bin(s,dim):
    return [int(bin(s)[2:].zfill(dim)[i]) for i in range(dim)]

def flatten(arr):
    """Flatten a nested list (row-major, last axis fastest) into a 1D list."""
    result = []
    def _walk(node):
        if isinstance(node, list):
            for child in node:
                _walk(child)
        else:
            result.append(node)
    _walk(arr)
    return result

def flatten_dict(d):
    """Flatten a dict keyed by index tuples (e.g. a Gurobi tupledict) into
    a 1D list, in the same row-major order as flatten()."""
    return [d[k] for k in sorted(d.keys())]
    
def array_to_dict(arr):
    if isinstance(arr, dict):
        raise TypeError(
            "array_to_dict expects a nested list straight from "
            "convert_int_to_array, but got a dict — this looks like "
            "it was already converted. Don't call it twice."
        )

    result = {}

    def walk(node, idx):
        if isinstance(node, list):
            for pos, child in enumerate(node):
                walk(child, idx + (pos,))
        else:
            result[idx] = node

    walk(arr, ())
    return result

def dict_to_array(d, size=None):
    """
    Inverse of array_to_dict: rebuild a nested list from a dict keyed by
    index tuples, e.g. {(n,i,j,k,l): value, ...} -> nested list.

    size: the shape to rebuild, e.g. (nr, key_size, 4, 4, sbox_size).
          If omitted, it's inferred from the max index used along each
          axis (assumes the dict is fully/rectangularly populated).
    """
    if isinstance(d, list):
        raise TypeError(
            "dict_to_array expects a dict keyed by index tuples, but got "
            "a list — this looks like it was already converted. Don't "
            "call it twice."
        )

    if size is None:
        depth = len(next(iter(d)))
        size = tuple(max(k[axis] for k in d) + 1 for axis in range(depth))

    def build(dims, prefix):
        if len(dims) == 1:
            return [d[prefix + (i,)] for i in range(dims[0])]
        return [build(dims[1:], prefix + (i,)) for i in range(dims[0])]

    return build(size, ())

def convert_int_to_array(x, size=None):
    if size is None:
        if x == 0:
            return [0]
        return [int(b) for b in bin(x)[2:]]

    total_bits = 1
    for d in size:
        total_bits *= d

    bits = format(x, f'0{total_bits}b')
    if len(bits) > total_bits:
        raise ValueError(
            f"x needs {len(bits)} bits but size {size} only allows {total_bits}"
        )

    bit_iter = iter(bits)

    def build(dims):
        if len(dims) == 1:
            return [int(next(bit_iter)) for _ in range(dims[0])]
        return [build(dims[1:]) for _ in range(dims[0])]

    return build(size)


def rehash_dict(A,c):
    # we swap the keys of A for the ones in c
    A_new = defaultdict(int)
    for k,v in c.items():
        A_new[v] = A[k]
    return A_new

# def convert_to_flat_array(A):
#     dim = int(np.ceil(np.log2(len(A))))
#     return [A[i] if i in A.keys() else 0 for i in range(2**dim)]

def convert_to_flat_array(A, dim):
    return [A[i] if i in A.keys() else 0 for i in range(2**dim)]

def fwht(a):
    """In-place Fast Walsh-Hadamard Transform. len(a) must be a power of 2."""
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[j], a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2
    return a

# def print_coset(inv_c,index,key_size):
#     masks = inv_c[index]
#     state_mask = masks >> (key_size * 64)
#     key_mask = masks & ((1 << (key_size * 64)) - 1)
#     bin_state_mask = bin(state_mask)

def set_sbox_size(size):
    global SBOX_SIZE
    SBOX_SIZE = size

def set_tk(tk):
    global TK
    TK = tk

def set_nr(nr):
    global NR
    NR = nr

def set_key_size(key_size):
    global KEY_SIZE
    KEY_SIZE = key_size

def set_thread_count(threads):
    global THREADS
    THREADS = threads

def set_max_search_time(max_search_time):
    global MAX_SEARCH_TIME
    MAX_SEARCH_TIME = max_search_time

def set_max_trail(max_trails):
    global MAX_TRAILS
    MAX_TRAILS = max_trails

def set_max_correlation(max_correlation):
    global MAX_CORRELATION
    MAX_CORRELATION = max_correlation

VERBOSE = False
COMPUTE_CONFLICTING_CONSTRAINTS = True
LINEAR_TRAILS_ONLY = True

MAX_CORRELATION = None
MAX_TRAILS = 5
MAX_SEARCH_TIME = 3600 # 5 hours
THREADS = 1

