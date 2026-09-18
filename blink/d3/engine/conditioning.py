"""Exact uniform samples from the old 4+2+2+2+4 maximum-condition family.

Samples are uniform in JOINT (X, rk1, ..., rk5) space conditional on
C(key) AND H4(X,rk1,rk2). They are NOT obtained by first choosing a uniform
key and then normalizing its successful input set separately. Returned
columns are X, rk1, rk2, rk3, rk4, rk5, with physical (unshifted) round keys.

Only a constant-probability 1/8 rejection is used for the three u2 forms;
neither the 3/2^29 B1 event nor the 2^-12 old-g event is rejection sampled.
No SAT solver or approximate sampler is used here.
"""

from pathlib import Path
import json
import math
import numpy as np

from . import core

HERE = Path(__file__).resolve().parent
S = np.asarray(core.SBOX, dtype=np.uint16)


def sub64(x):
    x = np.asarray(x, dtype=np.uint64)
    out = np.zeros_like(x)
    for i in range(16):
        out |= S[((x >> np.uint64(4 * i)) & np.uint64(15)).astype(np.intp)].astype(np.uint64) << np.uint64(4 * i)
    return out


def mix64(x):
    x = np.asarray(x, dtype=np.uint64)
    out = np.zeros_like(x)
    for i in range(16):
        nibble = ((x >> np.uint64(4 * (i ^ 4))) ^ (x >> np.uint64(4 * (i ^ 8))) ^ (x >> np.uint64(4 * (i ^ 12)))) & np.uint64(15)
        out |= nibble << np.uint64(4 * i)
    return out


def perm64(x, inverse=False):
    x = np.asarray(x, dtype=np.uint64)
    out = np.zeros_like(x)
    for i, j in enumerate(core.PBOX):
        source, target = (i, j) if inverse else (j, i)
        out |= ((x >> np.uint64(4 * source)) & np.uint64(15)) << np.uint64(4 * target)
    return out


def box64(x, key=0):
    return sub64(mix64(sub64(x)) ^ np.asarray(key, dtype=np.uint64))


def connector64(x, key, reverse=False):
    if reverse:
        return perm64(mix64(perm64(x, True) ^ key), True)
    return perm64(mix64(perm64(x)) ^ key)


def parity64(x):
    v = np.asarray(x, dtype=np.uint64).copy()
    for shift in (32, 16, 8, 4, 2, 1):
        v ^= v >> np.uint64(shift)
    return v & np.uint64(1)


def _affine(rows, rhs):
    """Particular solution and a complete independent kernel basis."""
    pivots = {}
    for mask, bit in zip(rows, rhs):
        r, b = int(mask), int(bit)
        while r:
            p = r.bit_length() - 1
            if p not in pivots:
                pivots[p] = r, b
                break
            m, t = pivots[p]
            r, b = r ^ m, b ^ t
        if not r and b:
            raise ValueError("inconsistent family conditions")
    def complete(x, homogeneous):
        for p, (r, b) in sorted(pivots.items()):
            v = core.parity(r & x) ^ (0 if homogeneous else b)
            x |= v << p
        return x
    particular = complete(0, False)
    kernel = [complete(1 << f, True) for f in range(64) if f not in pivots]
    assert all(core.parity(m & particular) == b for m, b in zip(rows, rhs))
    assert all(not core.parity(m & v) for m in rows for v in kernel)
    assert core.Space(kernel).rank == len(kernel)
    return particular, kernel


class ConditionalSampler:
    def __init__(self, seed=42224001, reference=None):
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        reference = Path(reference) if reference is not None else HERE.parent / "data"
        self.conditions = json.loads((reference / "four_zero_conditions.json").read_text(encoding="utf-8"))
        with np.load(reference / "four_zero_slice_extrema.npz") as data:
            self.max_syndromes = data["maximum"].astype(np.uint64).copy()
        assert len(self.max_syndromes) == 64 and len(set(map(int, self.max_syndromes))) == 64
        self.max_syndrome_set = set(map(int, self.max_syndromes))
        self.tail_masks = [int(x, 0) for x in self.conditions["tail_key_equations"]["masks"]]
        self.tail_basis = [int(x, 0) for x in self.conditions["tail_basis_on_rk4_xor_rc4prime"]]
        by_name = {f["name"]: f for f in self.conditions["middle_factors"]}
        self.u2_masks = [int(x, 0) for x in by_name["u2"]["physical_key_masks"]]
        self.u4_masks = [int(x, 0) for x in by_name["u4"]["physical_key_masks"]]
        assert by_name["u2"]["maximum_syndromes"] == [0]
        assert by_name["u4"]["maximum_syndromes"] == [0]
        self.k5_base, self.k5_kernel = _affine(self.tail_masks, [core.parity(m & core.RCI[4]) for m in self.tail_masks])
        self.k4_bases = []
        for syndrome in self.max_syndromes:
            rows = self.tail_basis + self.u4_masks
            rhs = [((int(syndrome) >> i) & 1) ^ core.parity(m & core.RCI[3]) for i, m in enumerate(self.tail_basis)]
            rhs += [core.parity(m & core.RC[3]) for m in self.u4_masks]
            base, kernel = _affine(rows, rhs)
            self.k4_bases.append(base)
            if hasattr(self, "k4_kernel"):
                assert kernel == self.k4_kernel
            self.k4_kernel = kernel
        self.k4_bases = np.asarray(self.k4_bases, dtype=np.uint64)
        assert len(self.k4_kernel) == 40 and len(self.k5_kernel) == 33
        # The 64 maximum syndromes happen to form ONE affine 6-space.
        # Membership can therefore be expressed by 15 equations, not an OR
        # of 64 assignments to all 21 syndrome bits. These are exact physical
        # round-key equations, useful for independent Boolean counting.
        origin = int(self.max_syndromes[0])
        directions = core.independent(int(v) ^ origin for v in self.max_syndromes)
        assert len(directions) == 6
        _, full_annihilator = _affine(directions, [0] * len(directions))
        annihilator = [m for m in full_annihilator if m < (1 << 21)]
        assert len(annihilator) == 15
        self.compact_key_equations = []
        for mask in self.tail_masks:
            self.compact_key_equations.append({"round_key": 5, "mask": f"0x{mask:016x}", "rhs": core.parity(mask & core.RCI[4]), "source": "tail_key_slice"})
        for mask in self.u2_masks:
            self.compact_key_equations.append({"round_key": 2, "mask": f"0x{mask:016x}", "rhs": core.parity(mask & core.RCI[1]), "source": "u2_maximum"})
        tail_max_physical = []
        for syndrome_mask in annihilator:
            mask = 0
            for i, basis_mask in enumerate(self.tail_basis):
                if (syndrome_mask >> i) & 1:
                    mask ^= basis_mask
            rhs = core.parity(syndrome_mask & origin) ^ core.parity(mask & core.RCI[3])
            self.compact_key_equations.append({"round_key": 4, "mask": f"0x{mask:016x}", "rhs": rhs, "source": "tail_maximum_affine_space"})
            tail_max_physical.append(mask)
            assert all(core.parity(syndrome_mask & int(v)) == core.parity(syndrome_mask & origin) for v in self.max_syndromes)
        for mask in self.u4_masks:
            self.compact_key_equations.append({"round_key": 4, "mask": f"0x{mask:016x}", "rhs": core.parity(mask & core.RC[3]), "source": "u4_maximum"})
        assert core.Space(tail_max_physical + self.u4_masks).rank == 18
        assert core.Space(int(e["mask"], 0) << (64 * (e["round_key"] - 1)) for e in self.compact_key_equations).rank == 52

        # DDT-compatible nibble inputs for the second S layer of B1.
        self.allowed = np.zeros((16, 16, 16), dtype=np.uint16)
        self.ddt = np.zeros((16, 16), dtype=np.uint64)
        for difference in range(16):
            for target in range(16):
                good = [x for x in range(16) if core.SBOX[x] ^ core.SBOX[x ^ difference] == target]
                self.ddt[difference, target] = len(good)
                self.allowed[difference, target, :len(good)] = good
        x = np.arange(65536, dtype=np.uint16)
        sx = core.sl(x)
        self.head_columns = []
        for column in range(4):
            di, do = core.extract(core.U, column), core.extract(core.T, column)
            delta = core.ml(sx ^ sx[x ^ di])
            weight = np.ones(65536, dtype=np.uint64)
            for row in range(4):
                weight *= self.ddt[(delta >> (4 * row)) & 15, (do >> (4 * row)) & 15]
            cumulative = np.cumsum(weight, dtype=np.uint64)
            self.head_columns.append((delta, cumulative, int(cumulative[-1]), do))
        self.g_good = [np.flatnonzero(core.local_indicator(core.extract(core.A, c), core.extract(core.B, c))).astype(np.uint16) for c in range(4)]
        g_basis = []
        for column in range(4):
            wave = core.fwht(core.local_indicator(core.extract(core.A, column), core.extract(core.B, column)))
            g_basis += [core.embed(m, column) for m in core.independent(map(int, np.flatnonzero(wave)))]
        # Rank 21+3 means all eight u2 syndromes occur equally often on g=1,
        # even after the affine phase imposed by the preceding B1 output.
        assert core.Space(g_basis).rank == 21
        assert core.Space(g_basis + [core.perm(m) for m in self.u2_masks]).rank == 24
        assert math.prod(len(g) for g in self.g_good) == 1 << 52
        totals = [t[2] for t in self.head_columns]
        assert math.prod(totals) == 3 * (1 << 99)
        self.metadata = {
            "measure": "uniform joint (X,rk1,rk2,rk3,rk4,rk5) conditional on C AND H4; not equal-weight key-then-input conditioning",
            "seed": self.seed,
            "sample_columns": ["X", "rk1", "rk2", "rk3", "rk4", "rk5"],
            "physical_round_keys_include_no_constants": True,
            "per_tail_syndrome_key_constraint_rank": 58,
            "number_disjoint_tail_maximum_syndromes": 64,
            "tail_maximum_syndrome_affine_dimension": 6,
            "equivalent_compact_key_affine_constraint_rank": 52,
            "key_family_mass": "1/2^52",
            "head_B1_mean_before_conditioning": "3/2^29",
            "old_g_probability_given_B1_and_u2_maximum": "1/2^12",
            "head_H4_probability_within_key_family": "3/2^41",
            "joint_condition_mass": "3/2^93",
            "conditioned_domain_size_384bit_X_and_RKs": "3*2^291",
            "conditioned_domain_size_512bit_X_and_full_master": "3*2^419",
            "B1_column_success_pair_counts": totals,
            "old_g_input_count": str(1 << 52),
            "old_g_input_count_per_u2_syndrome": str(1 << 49),
            "g_input_fourier_rank": 21,
            "g_plus_u2_input_forms_rank": 24,
            "rk4_free_dimension_per_syndrome": 40,
            "rk5_free_dimension": 33,
            "largest_weighted_sampling_table": 65536,
            "only_rejection_acceptance_probability": "1/8",
        }

    def _kernel_words(self, kernel, n):
        bits = self.rng.bit_generator.random_raw(n)
        out = np.zeros(n, dtype=np.uint64)
        for i, vector in enumerate(kernel):
            out ^= ((bits >> np.uint64(i)) & np.uint64(1)) * np.uint64(vector)
        return out

    def _first_box(self, n):
        """Uniform (X, effective rk1) among all successful B1 pairs."""
        x64 = np.zeros(n, dtype=np.uint64)
        key64 = np.zeros(n, dtype=np.uint64)
        for column, (delta, cumulative, total, do) in enumerate(self.head_columns):
            draws = self.rng.integers(0, total, size=n, dtype=np.uint64)
            x = np.searchsorted(cumulative, draws, side="right").astype(np.uint16)
            difference = delta[x]
            second_s_input = np.zeros(n, dtype=np.uint16)
            for row in range(4):
                d = (difference >> (4 * row)) & 15
                target = (do >> (4 * row)) & 15
                size = self.ddt[d, target]
                assert np.all(size > 0)
                choice = self.rng.integers(0, size, size=n, dtype=np.uint64)
                second_s_input |= self.allowed[d, target, choice] << (4 * row)
            local_key = second_s_input ^ core.ml(core.sl(x))
            for row in range(4):
                x64 |= ((x.astype(np.uint64) >> np.uint64(4 * row)) & np.uint64(15)) << np.uint64(4 * (column + 4 * row))
                key64 |= ((local_key.astype(np.uint64) >> np.uint64(4 * row)) & np.uint64(15)) << np.uint64(4 * (column + 4 * row))
        return x64, key64

    def sample(self, n):
        n = int(n)
        if n < 0:
            raise ValueError("sample size must be nonnegative")
        if n == 0:
            return np.zeros((0, 6), dtype=np.uint64)
        result = np.empty((n, 6), dtype=np.uint64)
        x, effective_k1 = self._first_box(n)
        result[:, 0] = x
        result[:, 1] = effective_k1 ^ np.uint64(core.RC[0])
        y = box64(x, effective_k1)
        shift = mix64(perm64(y)) ^ np.uint64(core.RC[1])
        pending = np.arange(n)
        while len(pending):
            z = np.zeros(len(pending), dtype=np.uint64)
            for column, good in enumerate(self.g_good):
                local = good[self.rng.integers(0, len(good), size=len(pending))].astype(np.uint64)
                for row in range(4):
                    z |= ((local >> np.uint64(4 * row)) & np.uint64(15)) << np.uint64(4 * (column + 4 * row))
            physical_k2 = perm64(z, True) ^ shift[pending]
            accept = np.ones(len(pending), dtype=bool)
            for mask in self.u2_masks:
                accept &= parity64((physical_k2 ^ np.uint64(core.RCI[1])) & np.uint64(mask)) == 0
            result[pending[accept], 2] = physical_k2[accept]
            pending = pending[~accept]
        result[:, 3] = self.rng.bit_generator.random_raw(n)
        selected = self.rng.integers(0, len(self.k4_bases), size=n)
        result[:, 4] = self.k4_bases[selected] ^ self._kernel_words(self.k4_kernel, n)
        result[:, 5] = np.uint64(self.k5_base) ^ self._kernel_words(self.k5_kernel, n)
        return result

    def verify(self, samples, scalar_checks=8):
        samples = np.asarray(samples, dtype=np.uint64)
        assert samples.ndim == 2 and samples.shape[1] == 6
        x, k1, k2, k3, k4, k5 = samples.T
        for mask in self.tail_masks:
            assert not np.any(parity64((k5 ^ np.uint64(core.RCI[4])) & np.uint64(mask)))
        for equation in self.compact_key_equations:
            assert np.all(parity64(samples[:, equation["round_key"]] & np.uint64(int(equation["mask"], 0))) == equation["rhs"])
        for key, constant, masks in ((k2, core.RCI[1], self.u2_masks), (k4, core.RC[3], self.u4_masks)):
            for mask in masks:
                assert not np.any(parity64((key ^ np.uint64(constant)) & np.uint64(mask)))
        syndromes = np.zeros(len(samples), dtype=np.uint64)
        for i, mask in enumerate(self.tail_basis):
            syndromes |= parity64((k4 ^ np.uint64(core.RCI[3])) & np.uint64(mask)) << np.uint64(i)
        assert all(int(v) in self.max_syndrome_set for v in syndromes)
        y = box64(x, k1 ^ np.uint64(core.RC[0]))
        yp = box64(x ^ np.uint64(core.U), k1 ^ np.uint64(core.RC[0]))
        assert np.all(y ^ yp == np.uint64(core.T))
        z = connector64(y, k2 ^ np.uint64(core.RC[1]))
        assert np.all(box64(z) ^ box64(z ^ np.uint64(core.A)) == np.uint64(core.B))
        for row in samples[:int(scalar_checks)]:
            xv, kv1, kv2 = map(int, row[:3])
            assert core.head_four(xv, kv1 ^ core.RC[0], kv2 ^ core.RC[1]) == 1
        return {"passed": True, "samples": len(samples), "scalar_head_checks": min(len(samples), int(scalar_checks)), "all_family_conditions_checked": True, "distinct_tail_syndromes_seen": len(set(map(int, syndromes)))}
