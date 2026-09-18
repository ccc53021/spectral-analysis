"""Complete exact contraction of the middle 2+2+2 layers into one 6-layer block.

Both inner connectors keep all nonzero value masks and their actual keys.
The two OUTER rk3 connectors to the four-layer head/tail remain outside
this computation: this is NOT the true complete fourteen-layer probability.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import time
import numpy as np

from .conditioning import (ConditionalSampler, core, mix64, perm64, box64,
                           connector64, parity64)


HERE = Path(__file__).resolve().parent


def record(p):
    p = Fraction(p)
    return {"numerator": p.numerator, "denominator": p.denominator, "text": str(p)}


def _read_key(value):
    return int(value, 0) if isinstance(value, str) else int(value)


def _column64(values, column):
    values = np.asarray(values, dtype=np.uint64)
    out = np.zeros_like(values)
    for row in range(4):
        out |= ((values >> np.uint64(4 * (column + 4 * row))) & np.uint64(15)) << np.uint64(4 * row)
    return out


def _embed64(values, column):
    values = np.asarray(values, dtype=np.uint64)
    out = np.zeros_like(values)
    for row in range(4):
        out |= ((values >> np.uint64(4 * row)) & np.uint64(15)) << np.uint64(4 * (column + 4 * row))
    return out


def _local_factor(effective_key, left):
    # Left is the OUTPUT event of B_k(C->D), hence the inverse B_Mk(D->C).
    # Right is already the INPUT event of B_M(k2prime)(D->C).
    key = int(core.ml(effective_key))
    di, do = core.extract(core.D, 3), core.extract(core.C, 3)
    truth = core.local_indicator(di, do, key)
    if left:
        x = np.arange(65536, dtype=np.uint16)
        output = core.sl(core.ml(core.sl(x)) ^ int(effective_key))
        success = output ^ output[x ^ do] == di
        independently = np.zeros(65536, dtype=np.int64)
        independently[output] = success
        assert np.array_equal(independently, truth)
    wave = core.fwht(truth)
    support = np.flatnonzero(wave)
    basis = core.independent(map(int, support))
    coordinates = core.Coordinates(basis)
    modes = []
    for n in support:
        n = int(n)
        q = core.mix(core.perm(core.embed(n, 3)))
        modes.append({"mask": n, "coefficient": int(wave[n]), "coordinate": coordinates.encode(n), "q": q, "a": core.perm(q)})
    return {"truth": truth, "wave": wave, "basis": basis, "modes": modes}


def _center_data():
    x = np.arange(65536, dtype=np.uint16)
    output = core.sl(core.ml(core.sl(x)))
    data = []
    for column in range(4):
        difference = core.extract(core.E, column)
        good = np.flatnonzero(output ^ output[x ^ difference] == difference).astype(np.uint16)
        data.append({"good": good, "output": output[good], "all_output": output})
    return data


def _phase(mask, value):
    return -1 if core.parity(mask & value) else 1


def compute(keys, witness_X=None, include_terms=True, verification=True):
    """Return an exact JSON-serializable result for five physical round keys.

    `exact_probability` is the middle six-layer success probability with
    its 64-bit incoming value averaged uniformly. The supplied rk3 does not
    enter this marginal and must not be independently reused at the two
    still-open outer connectors.
    """
    started = time.perf_counter()
    keys = list(map(_read_key, keys))
    if len(keys) != 5:
        raise ValueError("keys must contain five physical 64-bit round keys")
    rk1, rk2, rk3, rk4, rk5 = keys
    left = _local_factor(core.extract(rk4 ^ core.RC[3], 3), True)
    right = _local_factor(core.extract(rk2 ^ core.RCI[1], 3), False)
    dl, dr = len(left["basis"]), len(right["basis"])
    if dl + dr > 24:
        raise ValueError("unexpectedly large local quotient; do not silently allocate an exponential table")
    left_basis = [core.perm(core.mix(core.perm(core.embed(b, 3)))) for b in left["basis"]]
    right_basis = [core.perm(core.mix(core.perm(core.embed(b, 3)))) for b in right["basis"]]
    center = _center_data()
    projected_waves = []
    for column, data in enumerate(center):
        code = np.zeros(len(data["good"]), dtype=np.uint64)
        for i, basis in enumerate(left_basis):
            code |= parity64(data["good"].astype(np.uint64) & np.uint64(core.extract(basis, column))) << np.uint64(i)
        for i, basis in enumerate(right_basis):
            code |= parity64(data["output"].astype(np.uint64) & np.uint64(core.extract(basis, column))) << np.uint64(dl + i)
        histogram = np.bincount(code.astype(np.intp), minlength=1 << (dl + dr))
        projected_waves.append(core.fwht(histogram))

    left_shift, right_shift = rk5 ^ core.RC[4], rk1 ^ core.RCI[0]
    numerator = 0
    nonzero_terms = []
    term_data = []
    for lm in left["modes"]:
        for rm in right["modes"]:
            coordinate = lm["coordinate"] | (rm["coordinate"] << dl)
            central_numerator = 1
            for wave in projected_waves:
                central_numerator *= int(wave[coordinate])
            phase = _phase(lm["q"], left_shift) * _phase(rm["q"], right_shift)
            raw = lm["coefficient"] * rm["coefficient"] * central_numerator * phase
            numerator += raw
            term_data.append((lm, rm, coordinate, phase))
            if raw:
                nonzero_terms.append({
                    "left_output_mask_local": f"0x{lm['mask']:04x}",
                    "right_input_mask_local": f"0x{rm['mask']:04x}",
                    "center_input_mask": f"0x{lm['a']:016x}",
                    "center_output_mask": f"0x{rm['a']:016x}",
                    "rk5_connector_mask": f"0x{lm['q']:016x}",
                    "rk1_connector_mask": f"0x{rm['q']:016x}",
                    "phase": phase,
                    "left_fourier_raw_numerator_over_2pow16": lm["coefficient"],
                    "right_fourier_raw_numerator_over_2pow16": rm["coefficient"],
                    "center_bilateral_raw_numerator_over_2pow64": central_numerator,
                    "total_raw_numerator_over_2pow96": raw,
                })
    exact = Fraction(numerator, 1 << 96)
    assert 0 <= exact <= 1
    center_probability = Fraction(1)
    for data in center:
        center_probability *= Fraction(len(data["good"]), 65536)
    independent = Fraction(int(left["wave"][0]), 65536) * center_probability * Fraction(int(right["wave"][0]), 65536)
    checks = {"enabled": bool(verification)}
    if verification:
        rng = np.random.default_rng(42224004)
        direct_coefficients = 0
        for column, data in enumerate(center):
            for _ in range(32):
                lm = left["modes"][int(rng.integers(len(left["modes"])))]
                rm = right["modes"][int(rng.integers(len(right["modes"])))]
                parity = parity64((data["good"].astype(np.uint64) & np.uint64(core.extract(lm["a"], column))))
                parity ^= parity64((data["output"].astype(np.uint64) & np.uint64(core.extract(rm["a"], column))))
                raw = int((1 - 2 * parity.astype(np.int64)).sum())
                coordinate = lm["coordinate"] | (rm["coordinate"] << dl)
                assert raw == int(projected_waves[column][coordinate])
                direct_coefficients += 1
        # Directly reconstruct the endpoint factors at 64 complete center
        # values through the original connectors (not their mask formulas).
        anchors = []
        for _ in range(64):
            anchors.append(sum(core.embed(int(data["good"][rng.integers(len(data["good"]))]), c) for c, data in enumerate(center)))
        if witness_X is not None:
            state = _read_key(witness_X)
            state = core.box(state, rk1 ^ core.RC[0])
            state = core.box(core.connector(state, rk2 ^ core.RC[1]))
            state = core.box(core.connector(state, rk3 ^ core.RC[2]), rk4 ^ core.RC[3])
            candidate = core.connector(state, rk5 ^ core.RC[4])
            if core.box(candidate) ^ core.box(candidate ^ core.E) == core.E:
                anchors.insert(0, candidate)
        for z in anchors:
            output = core.box(z)
            lpoint = core.extract(core.connector(z, left_shift, True), 3)
            rpoint = core.extract(core.connector(output, right_shift, True), 3)
            lsum = sum(m["coefficient"] * _phase(m["a"], z) * _phase(m["q"], left_shift) for m in left["modes"])
            rsum = sum(m["coefficient"] * _phase(m["a"], output) * _phase(m["q"], right_shift) for m in right["modes"])
            assert lsum == 65536 * int(left["truth"][lpoint])
            assert rsum == 65536 * int(right["truth"][rpoint])

        # Independent exact 16-bit quadrature: fix three center columns to
        # actual successful values, enumerate the fourth column in full,
        # and compare original connector predicates with the Fourier sum.
        quadratures = []
        x = np.arange(65536, dtype=np.uint64)
        for varying_column in range(4):
            anchor = anchors[0]
            column_mask = core.embed(65535, varying_column)
            values = np.uint64(anchor & (core.MASK64 ^ column_mask)) ^ _embed64(x, varying_column)
            output = box64(values)
            valid_center = box64(values ^ np.uint64(core.E)) ^ output == np.uint64(core.E)
            lpoints = _column64(connector64(values, np.uint64(left_shift), True), 3)
            rpoints = _column64(connector64(output, np.uint64(right_shift), True), 3)
            direct = int((valid_center & left["truth"][lpoints].astype(bool) & right["truth"][rpoints].astype(bool)).sum())
            anchor_output = core.box(anchor)
            fixed_mask = core.MASK64 ^ column_mask
            raw = 0
            for lm, rm, coordinate, phase in term_data:
                fixed_phase = _phase(lm["a"] & fixed_mask, anchor) * _phase(rm["a"] & fixed_mask, anchor_output)
                raw += lm["coefficient"] * rm["coefficient"] * phase * fixed_phase * int(projected_waves[varying_column][coordinate])
            assert Fraction(raw, 1 << 48) == Fraction(direct, 65536)
            quadratures.append({"varying_column": varying_column, "enumerated_inputs": 65536, "successful_inputs": direct, "direct_probability": record(Fraction(direct, 65536)), "fourier_equals_direct": True})
        checks = {
            "enabled": True,
            "left_output_indicator_equals_direct_forward_box_table": True,
            "direct_local_bilateral_coefficient_checks": direct_coefficients,
            "complete_connector_pointwise_fourier_inversions": 2 * len(anchors),
            "independent_single_column_quadratures": quadratures,
            "all_passed": True,
        }
    result = {
        "state": "complete_middle_six_layer_exact",
        "scope": "Both connectors inside B3 + B4(center) + B5 are exact, including all their Fourier terms and fixed physical rk5/rk1 values. Uniform 64-bit input to this six-layer block. Not the full fourteen-layer probability.",
        "physical_round_keys": [f"0x{k:016x}" for k in keys],
        "formula": "E_z fLeft(extract(Linv_(rk5 xor RC5)(z),3)) * g_center(z) * fRight(extract(Linv_(rk1 xor RCI1)(B0(z)),3))",
        "left_local_fourier_rank": dl,
        "right_local_fourier_rank": dr,
        "left_nonzero_fourier_modes": len(left["modes"]),
        "right_nonzero_fourier_modes": len(right["modes"]),
        "all_candidate_mode_pairs_evaluated": len(term_data),
        "local_projection_table_size": 1 << (dl + dr),
        "nonzero_contracted_mode_pairs": len(nonzero_terms),
        "exact_probability": record(exact),
        "old_three_independent_blocks_product": record(independent),
        "additive_connector_correction": record(exact - independent),
        "ratio_to_old_product": None if independent == 0 else record(exact / independent),
        "integer_sum_numerator_over_2pow96": numerator,
        "left_output_basis_local": [f"0x{b:04x}" for b in left["basis"]],
        "right_input_basis_local": [f"0x{b:04x}" for b in right["basis"]],
        "verification": checks,
        "resulting_outer_block_partition": "4+6+4",
        "outer_connector_masks_still_zero": ["head4 to middle6 (rk3)", "middle6 to tail4 (same rk3 with reverse-side constant)"],
        "full_fourteen_layer_probability_computed": False,
        "full_route_zero_certified_by_middle": exact == 0,
        "zero_certificate_reason": "A zero exact average of the Boolean middle-six-layer event means no middle input can succeed, so no full specified route can succeed for this fixed key." if exact == 0 else None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if include_terms:
        result["all_nonzero_contracted_terms"] = nonzero_terms
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sat-result", type=Path, default=HERE / "output/final_verified/sat/sat_result.json")
    args = parser.parse_args()
    sampler = ConditionalSampler()
    old = sampler.conditions["representatives"]["maximum"]
    master = int(old["master_key"], 0)
    cases = [("old_maximum_representative", [(master >> (128 + 64 * i)) & core.MASK64 for i in range(5)], old["X"])]
    if args.sat_result.exists():
        sat = json.loads(args.sat_result.read_text(encoding="utf-8"))
        for case in sat["cases"]:
            if case.get("mode") == "family" and case.get("result") == "SAT":
                witness = case["witness"]
                cases.append(("full_route_sat_family_witness", witness["round_keys"], witness["X"]))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "complete exact middle six-layer block; two outer connectors remain uncompleted", "cases": {}}
    for name, keys, witness in cases:
        result = compute(keys, witness)
        (args.output / (name + ".json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        summary["cases"][name] = {k: result[k] for k in ("exact_probability", "old_three_independent_blocks_product", "additive_connector_correction", "ratio_to_old_product", "all_candidate_mode_pairs_evaluated", "nonzero_contracted_mode_pairs", "elapsed_seconds")}
    (args.output / "complete_middle_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
