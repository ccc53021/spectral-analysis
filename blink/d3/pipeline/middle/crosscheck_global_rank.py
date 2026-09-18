"""Read-only audit of the global key Fourier rank certificate.

Queries all 9,360 middle modes for every certificate coefficient, bypassing
the optimized quotient grouping. Writes only this middle directory's output.
"""

from fractions import Fraction
import json
from pathlib import Path
import random
import time

HERE = Path(__file__).resolve().parent
from ..global_coefficients import SchemeA
from engine.core import MASK64, RC, RCI, mix, extract, parity


def record(p):
    return {"numerator": p.numerator, "denominator": p.denominator, "text": str(p)}


def independent_rank(rows):
    """Separate elimination implementation, not the producer's Space class."""
    rows = sorted(set(map(int, rows)), reverse=True)
    rank = 0
    while rows:
        lead = rows.pop(0)
        if not lead:
            continue
        pivot = 1 << (lead.bit_length()-1)
        rows = sorted((r ^ lead if r & pivot else r for r in rows), reverse=True)
        rank += 1
    return rank


def run():
    started = time.perf_counter()
    certificate = json.loads((HERE.parent / "output/global_key_rank_certificate.json").read_text())
    model = SchemeA()
    records = []
    for index, item in enumerate(certificate["rank_certificate"]):
        mask = int(item["mask"], 16)
        assert not (mask >> 320) and not ((mask >> 128) & MASK64)
        raw, count = model.coefficient(mask, exhaustive=True)
        optimized_raw, optimized_count = model.coefficient(mask)
        expected = Fraction(item["coefficient"]["numerator"], item["coefficient"]["denominator"])
        actual = Fraction(raw, model.denominator)
        assert raw != 0 and actual == expected
        assert (raw, count) == (optimized_raw, optimized_count)
        assert count == item["nonzero_contracted_terms"]
        records.append({"index": index, "mask": item["mask"], "coefficient": record(actual),
                        "all_middle_modes_examined": len(model.middle_terms),
                        "nonzero_contracted_terms": count, "passed": True})
    masks = [int(r["mask"], 16) for r in records]
    rank = independent_rank(masks)
    assert rank == 114 == certificate["factor_space_upper_rank"]
    mean_raw, mean_count = model.coefficient(0, exhaustive=True)
    mean = Fraction(mean_raw, model.denominator)
    assert mean == Fraction(27, 1 << 114) and mean_count == 1
    # Reconstruct selected entire local-key sums, rather than merely testing
    # lookup-table entries at the old 52 selected basis vectors.
    # The reference performs a fresh complete 16-bit input truth-table FWHT
    # through Transfer.spectrum for each actual normalized physical key.
    rng = random.Random(20260915114)
    local_checks = []
    for side, constants in (("head", (RC[1], RC[0])), ("tail", (RCI[3], RCI[4]))):
        for trial in range(8):
            physical_key = rng.getrandbits(64)
            normalized_key = mix(physical_key ^ constants[1])
            direct, direct_denominator = model.outer.tr.spectrum(normalized_key)
            indices = [0] + rng.sample(model.outer.live, 7)
            for index in indices:
                connector = int(model.outer.tr.beta[index])
                numerator = int(model.outer.tr.weights[index])
                for col in range(4):
                    a = int(model.outer.tr.bcols[col][index])
                    key_column = extract(normalized_key, col)
                    local_sum = sum(-n if parity(q & key_column) else n
                                    for q, n in model.outer.rows[col][a])
                    numerator *= local_sum
                if parity(connector & constants[0]):
                    numerator = -numerator
                calculated = Fraction(numerator, model.outer.denominator)
                direct_numerator = int(direct[index])
                if parity(connector & constants[0]):
                    direct_numerator = -direct_numerator
                expected = Fraction(direct_numerator, direct_denominator)
                assert calculated == expected
                local_checks.append({"side": side, "trial": trial,
                                     "physical_inner_key": hex(physical_key),
                                     "connector_mask": hex(connector),
                                     "coefficient_as_function_of_connector": record(expected),
                                     "passed": True})
    result = {
        "scope": "Independent exhaustive-middle crosscheck of the scheme-A complete key-support rank certificate; does not enumerate its full probability distribution.",
        "certificate_entries_checked": len(records),
        "independent_elimination_rank": rank,
        "upper_rank_from_exact_factor_spaces": certificate["factor_space_upper_rank"],
        "minimum_complete_product_support_rank_certified": rank,
        "full_middle_modes_per_exhaustive_query": len(model.middle_terms),
        "all_114_exhaustive_coefficients_match_certificate_and_grouped_queries": True,
        "mean": record(mean), "nonzero_terms_in_mean": mean_count,
        "outer_complete_local_key_sum_vs_fresh_input_truth_fwht_checks": len(local_checks),
        "math_review": {
            "coefficient_formula": "P_hat(w)=sum_m M_hat(m) H_hat((w xor m)_rk1,rk2) T_hat((w xor m)_rk4,rk5)",
            "shared_keys": "Each residual uses the actual four physical words of w xor m, so reused RK1/RK2/RK4/RK5 constraints remain coupled.",
            "normalization": "All local coefficients are normalized Walsh coefficients. Outer denominator is the g denominator times all four joint local denominators. Global denominator is D_outer^2 D_middle; no extra 2^114 factor is used in Fourier convolution.",
            "physical_inner_mapping": "Normalized endpoint key kappa=M(physical_key xor constant); M is symmetric and involutory, so q=M(physical_inner_mask) in lookup and the affine sign uses physical_inner_mask dot the unmixed physical constant.",
            "connector_phases": "head (RC2,RC1), tail (RCI4,RCI5); parity of XOR of the two masked words equals the sum of their two parity exponents.",
            "completeness": "All g beta masks are enumerated and beta is invertible in a; for each beta the four local joint spectra give all possible normalized inner-key masks. Disjoint columns make their key-mask assembly unique. The stored local projection omits only input-mask bits known absent from every pulled-back g mask.",
            "quotient_optimization": "Compatible m must satisfy remainder(m)=remainder(w) modulo the outer support span. Exhaustive summation independently confirms this optimization on every rank-certificate entry.",
            "scope_note": "The certificate proves rank and mean, not extrema, class histogram, or full-cipher nonzero outer-connector corrections.",
        },
        "coefficient_records": records,
        "outer_lookup_reconstruction_records": local_checks,
        "all_passed": True,
        "elapsed_seconds": time.perf_counter()-started,
    }
    (HERE / "output/global_rank_crosscheck.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("coefficient_records", "outer_lookup_reconstruction_records", "math_review")}, indent=2), flush=True)


if __name__ == "__main__":
    run()
