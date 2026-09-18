import numpy as np

def get_linear_combination(coefficients,state_masks,key_masks,state_all_masks):
    sum_state_mask = 0
    for coeff,mask in zip(coefficients,state_masks):
        if coeff == 1:
            sum_state_mask ^= mask
    sum_key_mask = 0
    for coeff,mask in zip(coefficients,key_masks):
        if coeff == 1:
            sum_key_mask ^= mask
    sum_state_all_mask = 0
    for coeff,mask in zip(coefficients,state_all_masks):
        if coeff == 1:
            sum_state_all_mask ^= mask
    return sum_state_mask, sum_key_mask, sum_state_all_mask

def gf2_nullspace_basis(vectors, n):
    """Basis of the orthogonal complement of span(vectors) in F_2^n."""
    if not vectors:
        return [tuple(1 if k == i else 0 for k in range(n)) for i in range(n)]

    A = (np.array(vectors, dtype=int) % 2).copy()
    rows, cols = A.shape
    pivot_cols, r = [], 0
    for c in range(cols):
        pivot_row = next((i for i in range(r, rows) if A[i, c] == 1), None)
        if pivot_row is None:
            continue
        A[[r, pivot_row]] = A[[pivot_row, r]]
        for i in range(rows):
            if i != r and A[i, c] == 1:
                A[i] = (A[i] + A[r]) % 2
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break

    free_cols = [c for c in range(cols) if c not in pivot_cols]
    basis = []
    for fc in free_cols:
        h = [0] * cols
        h[fc] = 1
        for row_idx, pc in enumerate(pivot_cols):
            if A[row_idx, fc] == 1:
                h[pc] = 1
        basis.append(tuple(h))
    return basis

def gaussian_elimination(A):
    """
    A: dict[int, float] mapping bitmask keys (plain Python ints) to
       correlation-like values, e.g. {0b0000: 1.02, 0b1010: 0.38, ...}

    Returns (c, basis):
      basis: list [b_0, b_1, ..., b_{d-1}] of the discovered basis vectors
             (each a plain int bitmask), found by Gaussian elimination
             over GF(2) on the keys of A.
      c:     dict[int, int] mapping each key w in A to its coordinate
             vector, ALSO encoded as an integer bitmask -- bit i of
             c[w] is set iff basis[i] is used in w's expansion, so
             w == xor of {basis[i] : bit i of c[w] is 1}.
    """
    masks = [w for w in A if w != 0]

    # Pass 1: build the basis using the standard "XOR/linear basis"
    # technique -- represent every vector as a plain integer and reduce
    # using its highest set bit as the pivot (no numpy needed, and this
    # scales fine to 192-bit masks since Python ints are arbitrary precision).
    pivots = {}  # pivot bit position -> basis vector with that pivot
    for w in masks:
        cur = w
        while cur != 0:
            p = cur.bit_length() - 1
            if p in pivots:
                cur ^= pivots[p]
            else:
                pivots[p] = cur
                break

    pivot_positions = sorted(pivots.keys(), reverse=True)
    basis = [pivots[p] for p in pivot_positions]

    # Pass 2: express every key of A (including 0) in terms of that basis.
    c = {}
    for w in A:
        cur = w
        coord = 0
        for i, b in enumerate(basis):
            p = pivot_positions[i]
            if (cur >> p) & 1:
                cur ^= b
                coord |= (1 << i)
        if cur != 0:
            raise ValueError(f"key {w} (0b{w:b}) is not in the span of the discovered basis")
        c[w] = coord
    return c, basis