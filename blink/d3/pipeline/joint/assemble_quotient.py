"""Explicit common linear cosets and exact key-to-joint counting identities.

The 206-dimensional common quotient is now certified minimal by nonzero
coefficients of the complete product. No assertion of witness extremality
or a completed global histogram is made here.
"""

from fractions import Fraction
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
from engine import core
from engine import model


def full_joint_mask(mask: int) -> int:
    """Internal X|RK1..5 to physical P|W1|W2|RK1..5 (512 bits)."""
    xmask = mask & core.MASK64
    kmask = mask >> 64
    return xmask | (xmask << 64) | (kmask << 192)


def main():
    output = HERE / "output"
    output.mkdir(parents=True, exist_ok=True)
    head = json.loads((output / "head_quotient.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE.parent / "output/factor_space_audit.json").read_text(encoding="utf-8"))
    key_basis = [int(m, 0) for m in audit["key_only_sufficient_basis_hex"]]
    common_key_basis = core.independent(key_basis + [1 << b for b in range(64)])
    input_basis = [int(m, 0) for m in head["input_basis"]]
    joint_basis = [m << 64 for m in common_key_basis] + input_basis
    dk, dr, dj = len(key_basis), len(common_key_basis), len(joint_basis)
    assert (dk, dr, dj) == (114, 146, 206)
    assert core.Space(joint_basis).rank == dj
    certificate = json.loads((ROOT / "data/joint_rank_certificate.json").read_text())
    support_basis = [int(row["mask"],0) for row in certificate["rank_certificate"]]
    assert certificate["exact_product_support_rank"] == dj
    assert core.Space(support_basis).rank == dj
    assert core.Space(support_basis + joint_basis).rank == dj
    # Verify that this common space contains every certified head dependency.
    assert core.Space(joint_basis + [int(m,0) for m in head["joint_sufficient_basis"]]).rank == dj
    physical = [full_joint_mask(m) for m in joint_basis]
    assert core.Space(physical).rank == dj
    witness = model.CASES["sat_witness"]
    round_keys = sum(k << (64 * i) for i, k in enumerate(witness["keys"]))
    z = witness["X"] | (round_keys << 64)
    beta = [core.parity(m & z) for m in joint_basis]
    rng = random.Random(202609150464)
    checks = 0
    distinct_keys = set()
    for _ in range(128):
        raw = rng.getrandbits(384)
        representative = core.solve([(m, core.parity(m & raw)) for m in joint_basis])
        sample = z ^ raw ^ representative
        assert all(core.parity(m & sample) == bit for m, bit in zip(joint_basis, beta))
        x = sample & core.MASK64
        keys = [(sample >> (64 * (i+1))) & core.MASK64 for i in range(5)]
        assert bool(model.evaluate(x, keys, block="head"))
        assert all(core.parity(m & (sample >> 64)) == core.parity(m & round_keys)
                   for m in key_basis)
        distinct_keys.add(tuple(keys))
        checks += 1
    p_h = Fraction(12451, 1 << 44)
    p_m = Fraction(5, 1 << 27)
    p_t = Fraction(6987, 1 << 43)
    positive = p_m * p_t
    witness_fibre_classes = (1 << 60) * p_h
    assert witness_fibre_classes.denominator == 1
    report = {
        "scope": "Scheme A 4+6+4, retained internal differences and only two outer value masks zero; zero tweak",
        "key_only_support_rank_exact": dk,
        "key_only_rank_certificate": "../output/global_key_rank_certificate.json (114 independent nonzero Fourier coefficients of the complete product)",
        "common_key_refinement_rank": dr,
        "joint_sufficient_rank": dj,
        "joint_support_rank_exact": dj,
        "joint_rank_certificate": str(ROOT / "data/joint_rank_certificate.json"),
        "input_projection_rank_of_H4_exact": len(input_basis),
        "input_projection_rank_of_full_joint_exact": len(input_basis),
        "key_projection_rank_of_full_joint_exact": dr,
        "key_only_rank_claimed_minimal": True,
        "joint_rank_claimed_minimal": True,
        "full_physical_master_key_bits": 448,
        "full_physical_plaintext_key_bits": 512,
        "key_only_coset_size": f"2^{448-dk}",
        "joint_coset_size": f"2^{512-dj}",
        "common_key_coset_size": f"2^{448-dr}",
        "key_only_coset_count": f"2^{dk}",
        "joint_coset_count": f"2^{dj}",
        "common_key_refinements_per_key_only_coset": f"2^{dr-dk}",
        "input_cosets_per_common_key_coset": "2^60",
        "key_only_basis_on_RK1_to_RK5_hex": [hex(m) for m in key_basis],
        "common_key_basis_on_RK1_to_RK5_hex": [hex(m) for m in common_key_basis],
        "joint_basis_internal_hex": [hex(m) for m in joint_basis],
        "joint_basis_physical_hex": [hex(m) for m in physical],
        "internal_order": "X|RK1|RK2|RK3|RK4|RK5, each 64 bits, first word least significant",
        "physical_order": "P|W1|W2|RK1|RK2|RK3|RK4|RK5, each 64 bits, first word least significant",
        "exact_counting_identity": {
            "variables": "q ranges over all 2^114 common factor-key classes; h(q)=pH, m(q)=pM, t(q)=pT",
            "key_histogram": "N_key(v)=sum_q [h(q)*m(q)*t(q)=v]",
            "joint_positive_histogram": "For v>0, N_joint(v)=2^92 * sum_q h(q)*[m(q)*t(q)=v]",
            "joint_zero_histogram": "N_joint(0)=2^206-sum_(v>0) N_joint(v)",
            "proof": "Each 114-dimensional factor-key class splits into 2^32 common-key classes. Each has 2^60 equal-sized input classes; exactly 2^60*h(q) have H4=1. Thus each q contributes 2^92*h(q) positive joint classes, provided m(q)*t(q)>0. This preserves all shared key dependencies and requires no input enumeration.",
            "integer_requirement": "2^60*h(q) is an integer since H4 has the certified four-dimensional input-period group for every key.",
            "max_condition": "Joint maximum requires h(q)>0 and maximizes m(q)*t(q), then selects H4=1 input cosets. Key-only maximum instead maximizes h(q)*m(q)*t(q).",
            "coset_conditions": "Every class is given explicitly by basis[i] dot physical(P,K) = beta[i]. Histograms can be counted by the identity, but global extremal beta lists still require the key search and representative input solving."
        },
        "nonzero_coset_certificate_not_global_maximum": {
            "source": "Previously certified K_sat block probabilities and explicit full-route witness",
            "internal_X": hex(witness["X"]),
            "round_keys": [hex(k) for k in witness["keys"]],
            "syndrome_bits_in_basis_order": beta,
            "syndrome_lsb_first_hex": hex(sum(bit << i for i,bit in enumerate(beta))),
            "exact_constant_model_value": core.fraction_record(positive),
            "class_size_in_full_P_key_space": f"2^{512-dj}",
            "H4_positive_input_cosets_per_common_key_class": int(witness_fibre_classes),
            "tested_members": checks,
            "distinct_round_key_tuples_in_tests": len(distinct_keys),
            "test_scope": "Exact coset equations, direct numerical head event, and equality of all factor-key syndromes. Fixed witness probabilities are inherited only because factor dependency spaces are complete."
        },
        "complete_key_histogram_computed": False,
        "complete_joint_histogram_computed": False,
        "global_extrema_claimed": False,
    }
    out = output / "joint_common_quotient.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("_hex")}, indent=2))


if __name__ == "__main__":
    main()
