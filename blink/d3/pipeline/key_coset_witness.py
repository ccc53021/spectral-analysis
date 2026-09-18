"""Glue compatible H, M, T classes into a full 114-dimensional key coset.

Representatives only specify the desired factor syndromes. The returned
linear equations describe all 2^334 full master keys in the resulting
coset, not just the representative used for numerical checks.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import random

import numpy as np

HERE = Path(__file__).resolve().parent
from .outer.outer_model import OuterModel
from .distribution_join import read_json, syndrome, frac_record


def solve_affine(rows, rhs):
    pivots = {}
    for row, bit in zip(rows, rhs):
        row, bit = int(row), int(bit)
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = (row, bit)
                break
            row ^= pivots[pivot][0]
            bit ^= pivots[pivot][1]
        if not row and bit:
            raise ValueError("Incompatible factor cosets: shared physical-key conditions disagree")
    value = 0
    for p in sorted(pivots):
        row, bit = pivots[p]
        value |= (bit ^ ((row & value).bit_count() & 1)) << p
    assert all(((int(m) & value).bit_count() & 1) == int(b) for m, b in zip(rows, rhs))
    return value, len(pivots)


class KeyCosetBuilder:
    def __init__(self):
        audit = read_json(HERE / "output/factor_space_audit.json")
        self.head_basis = [int(m, 0) for m in audit["head_basis_hex"]]
        self.tail_basis = [int(m, 0) for m in audit["tail_basis_hex"]]
        self.key_basis = [int(m, 0) for m in audit["key_only_sufficient_basis_hex"]]
        mid = read_json(HERE / "middle/output/middle_conditional_on_outer_shared8.json")
        self.middle_basis = [int(m, 0) for m in mid["full_basis_shared_first_hex"]]
        self.middle_table = np.load(HERE / "middle/output/middle_shared8_remaining10_numerators.npy")
        self.middle_den = int(mid["denominator"])
        self.middle_fourier = read_json(HERE / "middle/output/complete_middle_fourier.json")
        self.middle_terms = [(int(row["mask"], 0), int(row["coefficient"]))
                             for row in self.middle_fourier["terms"]]
        self.outer = OuterModel()

    @staticmethod
    def pack_keys(keys):
        if len(keys) != 5 or any(k < 0 or k >= 1 << 64 for k in keys):
            raise ValueError("Expected five unsigned 64-bit physical round-key words")
        return sum(int(k) << (64 * i) for i, k in enumerate(keys))

    def probabilities(self, packed):
        keys = [(packed >> (64 * i)) & ((1 << 64) - 1) for i in range(5)]
        h = self.outer.physical_head(keys[0], keys[1])
        t = self.outer.physical_tail(keys[3], keys[4])
        m = Fraction(sum(-n if (mask & packed).bit_count() & 1 else n
                         for mask, n in self.middle_terms), self.middle_fourier["denominator"])
        ms = syndrome(self.middle_basis, packed)
        assert m == Fraction(int(self.middle_table[ms & 255, ms >> 8]), self.middle_den)
        return h, m, t

    def construct(self, head_keys, tail_keys, middle_syndrome, checks=8, seed=202609150464):
        head = self.pack_keys([*head_keys, 0, 0, 0])
        tail = self.pack_keys([0, 0, 0, *tail_keys])
        if not 0 <= middle_syndrome < 1 << 18:
            raise ValueError("Middle syndrome must contain 18 bits, shared8 first")
        rows = self.head_basis + self.tail_basis + self.middle_basis
        rhs = [(m & head).bit_count() & 1 for m in self.head_basis]
        rhs += [(m & tail).bit_count() & 1 for m in self.tail_basis]
        rhs += [(middle_syndrome >> i) & 1 for i in range(18)]
        packed, rank = solve_affine(rows, rhs)
        assert rank == len(self.key_basis) == 114
        h, m, t = self.probabilities(packed)
        assert h == self.outer.physical_head(*head_keys)
        assert t == self.outer.physical_tail(*tail_keys)
        key_rhs = syndrome(self.key_basis, packed)
        rng = random.Random(seed)
        for _ in range(checks):
            random_word = rng.getrandbits(320)
            delta_rhs = [(mask & random_word).bit_count() & 1 for mask in self.key_basis]
            correction, _ = solve_affine(self.key_basis, delta_rhs)
            member = packed ^ random_word ^ correction
            assert syndrome(self.key_basis, member) == key_rhs
            assert self.probabilities(member) == (h, m, t)
        p = h * m * t
        return {
            "scope": "One full scheme-A key coset; no claim that its value is a global extremum",
            "quotient_rank": 114, "rank_proved_minimal": True,
            "full_master_key_bits": 448, "keys_per_coset": str(1 << 334),
            "probability": frac_record(p.numerator, p.denominator),
            "factors": {label: frac_record(q.numerator, q.denominator)
                        for label, q in zip(("head", "middle", "tail"), (h, m, t))},
            "physical_RK_representative_hex": [f"0x{(packed >> (64*i)) & ((1<<64)-1):016x}" for i in range(5)],
            "physical_master_mask_packing": "W1 low64, W2 next64, RK1..RK5 following; zero tweak",
            "all_114_linear_conditions": [{"master_key_mask_hex": hex(mask << 128),
                                          "rhs": (key_rhs >> i) & 1}
                                         for i, mask in enumerate(self.key_basis)],
            "condition_definition": "parity(master_key_mask_hex & K) = rhs for every row",
            "factor_middle_syndrome_shared8_first": middle_syndrome,
            "different_coset_members_exactly_checked": checks,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-keys", nargs=2, type=lambda s: int(s, 0))
    parser.add_argument("--tail-keys", nargs=2, type=lambda s: int(s, 0))
    parser.add_argument("--middle-syndrome", type=lambda s: int(s, 0))
    parser.add_argument("--output", type=Path, default=HERE / "output/key_coset_witness.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    builder = KeyCosetBuilder()
    if args.self_test:
        keys = [0x2874cc1e532a2747, 0x1062f52829c95f66, 0x18c078facb7bfa85,
                0x73ae068f503c26b9, 0x90479091156bcc2d]
        mid = syndrome(builder.middle_basis, builder.pack_keys(keys))
        result = builder.construct(keys[:2], keys[3:], mid, checks=16)
        assert result["probability"] == frac_record(434975685, 1 << 114)
        try:
            builder.construct(keys[:2], keys[3:], mid ^ 1, checks=0)
        except ValueError:
            result["incompatible_shared_condition_rejected"] = True
        else:
            raise AssertionError("Incompatible shared coset was silently accepted")
        result["self_test"] = "complete_verified; fixed-class check, not a global extremum"
    else:
        if args.head_keys is None or args.tail_keys is None or args.middle_syndrome is None:
            parser.error("Provide both endpoint representatives and the 18-bit middle syndrome")
        result = builder.construct(args.head_keys, args.tail_keys, args.middle_syndrome)
    result["pid"] = os.getpid()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"probability": result["probability"], "rank": 114,
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
