"""Real full-five-round Xoodoo experiments in the six exact pulled-back classes.

Only F_i(X)=B_i(R(X)) is imposed (by uniform sampling and classification).
The first-round differential event E is measured, never imposed. A paired
four-round fixed-core-difference control distinguishes input-class pullback
from an unjustified transfer of conditional correlations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
from numba import njit, prange, set_num_threads

HERE = Path(__file__).resolve().parent
import xoodoo as xo

SOURCE = (
    HERE.parent / "four_round" / "output" / "known_distinguisher_joint"
    / "r4_public_full_extension_core" / "p1_complete" / "distribution_analysis.json"
)
SOURCE_HASH = hashlib.sha256(SOURCE.read_bytes()).hexdigest() if SOURCE.is_file() else ""
DELTA = (0xA8B23B19, 0x98810919, 0x52674513, 0x95A876F3,
         0xA8B23B18, 0x98810919, 0x52674513, 0x95A876F3,
         0xA8B23B18, 0x98810919, 0x52676513, 0x95A876F3)
CORE_DELTA = (1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)


@njit(inline="always")
def rot(v, n):
    return np.uint32((v << n) | (v >> (32 - n)))


@njit(cache=True)
def rnd(a, c):
    p0, p1 = a[0] ^ a[4] ^ a[8], a[1] ^ a[5] ^ a[9]
    p2, p3 = a[2] ^ a[6] ^ a[10], a[3] ^ a[7] ^ a[11]
    e0, e1 = rot(p3, 5) ^ rot(p3, 14), rot(p0, 5) ^ rot(p0, 14)
    e2, e3 = rot(p1, 5) ^ rot(p1, 14), rot(p2, 5) ^ rot(p2, 14)
    b0, b1, b2, b3 = a[0] ^ e0 ^ c, a[1] ^ e1, a[2] ^ e2, a[3] ^ e3
    b4, b5, b6, b7 = a[7] ^ e3, a[4] ^ e0, a[5] ^ e1, a[6] ^ e2
    b8, b9 = rot(a[8] ^ e0, 11), rot(a[9] ^ e1, 11)
    b10, b11 = rot(a[10] ^ e2, 11), rot(a[11] ^ e3, 11)
    return (np.uint32(b0 ^ (~b4 & b8)), np.uint32(b1 ^ (~b5 & b9)),
            np.uint32(b2 ^ (~b6 & b10)), np.uint32(b3 ^ (~b7 & b11)),
            rot(b4 ^ (~b8 & b0), 1), rot(b5 ^ (~b9 & b1), 1),
            rot(b6 ^ (~b10 & b2), 1), rot(b7 ^ (~b11 & b3), 1),
            rot(b10 ^ (~b2 & b6), 8), rot(b11 ^ (~b3 & b7), 8),
            rot(b8 ^ (~b0 & b4), 8), rot(b9 ^ (~b1 & b5), 8))


@njit(cache=True)
def suffix(a):
    a = rnd(a, np.uint32(0x380))
    a = rnd(a, np.uint32(0xf0))
    a = rnd(a, np.uint32(0x1a0))
    return rnd(a, np.uint32(0x12))


@njit(inline="always")
def xor_delta(a, d):
    return (a[0] ^ d[0], a[1] ^ d[1], a[2] ^ d[2], a[3] ^ d[3],
            a[4] ^ d[4], a[5] ^ d[5], a[6] ^ d[6], a[7] ^ d[7],
            a[8] ^ d[8], a[9] ^ d[9], a[10] ^ d[10], a[11] ^ d[11])


@njit(inline="always")
def parity(v):
    v ^= v >> 16
    v ^= v >> 8
    v ^= v >> 4
    return (0x6996 >> (v & 15)) & 1


@njit(inline="always")
def label(a, masks):
    result = 0
    for i in range(6):
        v = np.uint32(0)
        for j in range(12):
            v ^= a[j] & masks[i, j]
        result |= parity(v) << i
    return result


@njit(cache=True)
def sample_result(x, masks, d, dc):
    y = rnd(x, np.uint32(0x2c))
    yp = rnd(xor_delta(x, d), np.uint32(0x2c))
    yc = xor_delta(y, dc)
    event = True
    for k in range(12):
        if yp[k] != yc[k]:
            event = False
    left = suffix(y)
    right = suffix(yp)
    control = suffix(yc)
    s5 = 1 - 2 * int((left[0] ^ right[0]) & 1)
    s4 = 1 - 2 * int((left[0] ^ control[0]) & 1)
    return label(y, masks), s5, s4, event


@njit(parallel=True, cache=True)
def batch_kernel(states, masks, d, dc):
    blocks = 64
    h = np.zeros((blocks, 64, 6), dtype=np.int64)
    n = len(states)
    for block in prange(blocks):
        start, stop = n * block // blocks, n * (block + 1) // blocks
        for i in range(start, stop):
            x = (states[i, 0], states[i, 1], states[i, 2], states[i, 3],
                 states[i, 4], states[i, 5], states[i, 6], states[i, 7],
                 states[i, 8], states[i, 9], states[i, 10], states[i, 11])
            a, s5, s4, event = sample_result(x, masks, d, dc)
            h[block, a, 0] += 1
            h[block, a, 1] += s5
            h[block, a, 2] += s4
            if event:
                h[block, a, 3] += 1
                h[block, a, 4] += s5
            h[block, a, 5] += s5 * s4
    return h


def signed_power(value):
    if value == 0:
        return "0"
    return f"{'+' if value > 0 else '-'}2^{math.log2(abs(value)):.9f}"


def metric(n, total):
    if n == 0:
        return None
    c = total / n
    se = math.sqrt(max(0., 1 - c * c) / n)
    z = NormalDist().inv_cdf(1 - .05 / (2 * 64))
    return dict(samples=int(n), signed_sum=int(total), correlation=c,
                signed_power=signed_power(c), standard_error=se,
                confidence_95=[max(-1., c - 1.959963984540054 * se),
                               min(1., c + 1.959963984540054 * se)],
                simultaneous_64_confidence_95=[max(-1., c - z * se), min(1., c + z * se)])


def source_data():
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASH:
        raise ValueError("Original six-constraint source changed; audit before running.")
    data = json.loads(raw)
    bases = [sum(int(w, 16) << (64 * j) for j, w in enumerate(b["mask_words"]))
             for b in data["reported_input_basis"]]
    masks = np.array([xo.lanes(b) for b in bases], dtype=np.uint32)
    return data, bases, masks


def validate(bases, masks):
    rng = np.random.default_rng(2026091107)
    states = rng.integers(0, 1 << 32, size=(64, 12), dtype=np.uint32)
    states = np.concatenate((np.zeros((1, 12), dtype=np.uint32),
                             np.full((1, 12), 0xffffffff, dtype=np.uint32),
                             np.array([xo.lanes(1 << j) for j in range(384)], dtype=np.uint32),
                             states))
    d, dc = tuple(np.uint32(v) for v in DELTA), tuple(np.uint32(v) for v in CORE_DELTA)
    delta_value = xo.from_lanes(DELTA)
    delta_core = xo.from_lanes(CORE_DELTA)
    assert xo.pre_chi_linear(delta_value) == (xo.column_mask(0, 1) ^ xo.column_mask(88, 4))
    compact = json.loads((HERE / "constraints.json").read_text(encoding="utf-8"))
    assert compact["source_sha256"] == SOURCE_HASH
    qlists = [re.findall(r"Q\[(\d+),(\d+),(\d+)\]", r["Q_expression"])
              for r in compact["constraints"]]
    hist = np.zeros((64, 6), dtype=np.int64)
    for row in states:
        value = xo.from_lanes(row)
        y = xo.xoodoo_round(value, 0x2c)
        yp = xo.xoodoo_round(value ^ delta_value, 0x2c)
        event_from_four_bits = all(((y >> bit) & 1) == rhs
                                  for bit, rhs in ((129, 0), (328, 1), (88, 0), (217, 1)))
        assert event_from_four_bits == (yp == (y ^ delta_core))
        reference = (sum(((base & y).bit_count() & 1) << i for i, base in enumerate(bases)),
                     1 - 2 * ((xo.permute(y, 4) ^ xo.permute(yp, 4)) & 1),
                     1 - 2 * ((xo.permute(y, 4) ^ xo.permute(y ^ delta_core, 4)) & 1),
                     yp == (y ^ delta_core))
        compiled = sample_result(tuple(row), masks, d, dc)
        assert tuple(compiled) == reference, (compiled, reference)
        assert xo.from_lanes(rnd(tuple(row), np.uint32(0x2c))) == y
        q = xo.chi(xo.pre_chi_linear(value) ^ 0x2c)
        for i, coords in enumerate(qlists):
            pulled = sum((q >> xo.bit_index(int(x), int(yc), int(z))) & 1
                         for x, yc, z in coords) & 1
            assert pulled == ((reference[0] >> i) & 1)
        a, s5, s4, event = reference
        assert not event or s5 == s4
        hist[a] += np.array((1, s5, s4, int(event), s5 if event else 0, s5*s4))
    compiled_hist = batch_kernel(states, masks, d, dc).sum(axis=0)
    assert np.array_equal(compiled_hist, hist)
    # Exact local enumeration: two active chi columns, all 8^2 values.
    # rho_east^-1 maps core e0+e256 to chi-output (col0,1)+(col88,4).
    output_diff = xo.column_mask(0, 1) ^ xo.column_mask(88, 4)
    assert xo.rho_east(output_diff) == delta_core
    successes = sum(xo.CHI[u] ^ xo.CHI[u ^ 1] == 1 for u in range(8)) * \
                sum(xo.CHI[v] ^ xo.CHI[v ^ 4] == 4 for v in range(8))
    assert successes == 4
    from derive_constraints import rank
    assert rank(bases) == 6
    assert rank(bases + [1 << bit for bit in (129, 328, 88, 217)]) == 10
    return {"states_checked": len(states), "six_Q_expressions_match_B_after_round": True,
            "compiled_round_and_real_5round_sign_match_reference": True,
            "paired_4round_control_matches_reference": True,
            "parallel_histogram_matches_scalar_reference": True,
            "exact_prefix_event_probability": successes / 64,
            "exact_prefix_event_probability_in_every_class": 1 / 16,
            "combined_rank_six_class_masks_and_four_event_masks": 10,
            "event_equivalent_Y_bit_values": [[129, 0], [328, 1], [88, 0], [217, 1]],
            "local_prefix_values_enumerated": 64}


def result_document(args, data, h, validation, elapsed, complete):
    theory = data["capacities"][-1]["correlation_distribution"]
    classes = []
    for a, row in enumerate(h):
        n, s5, s4, ne, se, cross = map(int, row)
        event_metric = metric(ne, se)
        five, control = metric(n, s5), metric(n, s4)
        if n == 0:
            continue
        ce = se / n
        co = (s5 - se) / n
        classes.append({
            "index": a, "assignment_a0_to_a5": [(a >> i) & 1 for i in range(6)],
            "display_a0_to_a5": "".join(str((a >> i) & 1) for i in range(6)),
            "core_4round_joint_U_1DDT_H64_model": theory[a],
            "core_4round_model_signed_power": signed_power(theory[a]),
            "naive_core_model_div16_hypothesis_NOT_5round_theory": theory[a] / 16,
            "five_round_experiment": five,
            "paired_four_round_core_experiment": control,
            "prefix_event_probability_estimate": ne / n,
            "on_prefix_event": event_metric,
            "off_prefix_event": metric(n - ne, s5 - se),
            "event_contribution_to_five_round_correlation": ce,
            "off_event_contribution_to_five_round_correlation": co,
            "paired_sign_product_sum": cross,
        })
    totals = h.sum(axis=0)
    n, s5, s4, ne, se, cross = map(int, totals)
    return {
        "version": 1, "complete": complete, "purpose": __doc__,
        "source": str(SOURCE), "source_sha256": SOURCE_HASH,
        "source_theory_ddt_prefix_rounds": 1,
        "experiment_ddt_prefix_rounds": None,
        "experiment_description": "real permutation, no DDT approximation or prefix-event filtering",
        "constraints": "F_i(X)=B_i(R_0x2c(X)), i=0..5",
        "class_encoding": "index=sum(a_i*2^i); display columns explicitly use a0..a5 left to right",
        "constraint_count": 6, "class_count": 64,
        "sampling": "uniform independent 384-bit X, classify by F(X); one endpoint constrained",
        "input_difference_lanes_y_major": [f"0x{v:08x}" for v in DELTA],
        "output_mask": "full-round output bit e0 (col0,mask1)",
        "core_input_difference": "(col0,difference5)=e0+e256",
        "round_constants": [f"0x{c:x}" for c in xo.reduced_round_constants(5)],
        "seed": args.seed, "rng": "NumPy default_rng PCG64, uint32 rows",
        "target_samples": 1 << args.samples_log2, "samples": n,
        "batch_size": 1 << args.batch_log2, "threads": args.threads,
        "wall_seconds_excluding_compilation_and_validation": elapsed,
        "validation": validation,
        "hypothesis_warning": "core C[a] and core C[a]/16 are comparison baselines, NOT derived five-round conditional theory",
        "overall_five_round": metric(n, s5),
        "overall_paired_four_round_core": metric(n, s4),
        "overall_prefix_event_probability": ne / n if n else None,
        "overall_on_prefix_event": metric(ne, se),
        "overall_off_prefix_event": metric(n-ne, s5-se),
        "histogram_columns": ["count", "sum_sign_5", "sum_sign_core4", "event_count", "event_sum_sign", "sum_sign5_times_sign4"],
        "raw_histogram": h.tolist(), "classes": classes,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples-log2", type=int, default=28)
    p.add_argument("--batch-log2", type=int, default=19)
    p.add_argument("--seed", type=int, default=2026091101)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not 10 <= args.samples_log2 <= 36 or not 10 <= args.batch_log2 <= 23:
        p.error("sample or batch exponent out of supported range")
    if args.output.exists():
        p.error("output exists; select a new path to preserve prior runs")
    set_num_threads(args.threads)
    data, bases, masks = source_data()
    print("Compiling and checking reference identities...", flush=True)
    validation = validate(bases, masks)
    print(json.dumps(validation), flush=True)
    d, dc = tuple(np.uint32(v) for v in DELTA), tuple(np.uint32(v) for v in CORE_DELTA)
    rng = np.random.default_rng(args.seed)
    h = np.zeros((64, 6), dtype=np.int64)
    target, processed = 1 << args.samples_log2, 0
    started, last_report = time.perf_counter(), 0.
    while processed < target:
        size = min(1 << args.batch_log2, target - processed)
        states = rng.integers(0, 1 << 32, size=(size, 12), dtype=np.uint32)
        h += batch_kernel(states, masks, d, dc).sum(axis=0)
        processed += size
        elapsed = time.perf_counter() - started
        if elapsed - last_report >= 20 or processed == target:
            doc = result_document(args, data, h, validation, elapsed, processed == target)
            write_json(args.output, doc)
            eta = elapsed / processed * (target - processed)
            print(f"{processed}/{target} ({processed/target:.1%}), "
                  f"C5={doc['overall_five_round']['signed_power']}, "
                  f"C4={doc['overall_paired_four_round_core']['signed_power']}, "
                  f"P(E)={doc['overall_prefix_event_probability']:.7f}, "
                  f"elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
            last_report = elapsed
    print(f"Saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
