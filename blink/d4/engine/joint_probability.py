"""Exact factorized FWHT for prefixes of the zero-connector ranked basis.

For A1=A2=A3=A4=0 the global Fourier coefficient factorizes over A0, q2,
and q4.  A product of three small FWHTs is exactly equal to a monolithic
2^d FWHT, while avoiding a 2^25 floating-point array and a huge JSON term
list for the dimension-25 experiment.
"""

from fractions import Fraction
import json
import math
from pathlib import Path
import time

import numpy

from .joint_basis import (
    RankAccumulator,
    exact_a0_support,
    exact_rest_support,
)
from .linear_layer import MASTER_KEY_BITS


STATE_BITS = 64


class LinearCoordinates:
    """Coordinates in an independent integer-vector basis over GF(2)."""

    def __init__(self, rows):
        self.dimension = len(rows)
        self.pivots = {}
        for index, row in enumerate(rows):
            value = int(row)
            coordinate = 1 << index
            while value:
                pivot = value.bit_length() - 1
                previous = self.pivots.get(pivot)
                if previous is None:
                    self.pivots[pivot] = (value, coordinate)
                    break
                value ^= previous[0]
                coordinate ^= previous[1]
            if not value:
                raise ValueError("component basis is linearly dependent")

    def coordinate(self, value):
        value = int(value)
        coordinate = 0
        while value:
            pivot = value.bit_length() - 1
            previous = self.pivots.get(pivot)
            if previous is None:
                return None
            value ^= previous[0]
            coordinate ^= previous[1]
        return coordinate


def load_basis(path, dimension):
    meta = None
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("type") == "meta":
            meta = record
        elif record.get("type") == "trail":
            rows.append(record)
    if not 1 <= dimension <= len(rows):
        raise ValueError(f"dimension must be in [1,{len(rows)}]")
    return meta, rows[:dimension]


def fwht_int(values):
    width = 1
    while width < len(values):
        blocks = values.reshape(-1, 2 * width)
        left = blocks[:, :width].copy()
        right = blocks[:, width:].copy()
        blocks[:, :width] = left + right
        blocks[:, width:] = left - right
        width *= 2
    return values


def component_fwht(support, basis):
    coordinates = LinearCoordinates(basis)
    accepted = []
    for mask, numerator in support:
        index = coordinates.coordinate(mask)
        if index is not None:
            accepted.append((index, int(numerator)))
    common_factor = 0
    for _, numerator in accepted:
        common_factor = math.gcd(common_factor, abs(numerator))
    if not common_factor:
        raise ValueError("component Fourier support is empty")
    absolute_bound = sum(abs(value // common_factor) for _, value in accepted)
    if absolute_bound >= 1 << 62:
        raise OverflowError("scaled component FWHT exceeds int64 safety bound")
    values = numpy.zeros(1 << len(basis), dtype=numpy.int64)
    for index, numerator in accepted:
        values[index] = numerator // common_factor
    transformed = fwht_int(values)
    unique, representatives, counts = numpy.unique(
        transformed, return_index=True, return_counts=True
    )
    histogram = [
        {
            "value": int(value),
            "representative": int(representative),
            "count": int(count),
        }
        for value, representative, count in zip(unique, representatives, counts)
    ]
    return {
        "dimension": len(basis),
        "support_size": len(accepted),
        "common_factor": common_factor,
        "histogram": histogram,
    }


def embed_coordinate(local_coordinate, global_indices):
    result = 0
    for local_index, global_index in enumerate(global_indices):
        if (local_coordinate >> local_index) & 1:
            result |= 1 << global_index
    return result


def indexed_expression(mask, prefix, width):
    return "+".join(
        f"{prefix}{bit}" for bit in range(width) if (mask >> bit) & 1
    ) or "0"


def joint_expression(mask):
    key_mask = mask & ((1 << MASTER_KEY_BITS) - 1)
    input_mask = mask >> MASTER_KEY_BITS
    terms = []
    terms.extend(
        f"K{bit}" for bit in range(MASTER_KEY_BITS) if (key_mask >> bit) & 1
    )
    terms.extend(f"X{bit}" for bit in range(STATE_BITS) if (input_mask >> bit) & 1)
    return "+".join(terms) or "0"


def condition_records(rows, coordinate):
    core_conditions = []
    full_conditions = []
    full_key_masks = []
    for index, row in enumerate(rows):
        mask = int(row["joint_mask"], 0)
        core_key_mask = mask & ((1 << MASTER_KEY_BITS) - 1)
        input_mask = mask >> MASTER_KEY_BITS
        full_key_mask = core_key_mask ^ input_mask
        full_key_masks.append(full_key_mask)
        bit = (coordinate >> index) & 1
        plaintext = indexed_expression(input_mask, "P", STATE_BITS)
        right = str(bit) + (("+" + plaintext) if plaintext != "0" else "")
        key_expression = indexed_expression(full_key_mask, "K", MASTER_KEY_BITS)
        core_conditions.append(f"{joint_expression(mask)}={bit}")
        full_conditions.append({
            "key_mask": f"0x{full_key_mask:0112x}",
            "key_expression": key_expression,
            "plaintext_mask": f"0x{input_mask:016x}",
            "plaintext_expression": plaintext,
            "right_hand_side": right,
            "equation": f"{key_expression}={right}",
            "zero_plaintext_equation": f"{key_expression}={bit}",
        })
    return core_conditions, full_conditions, full_key_masks


def run(args):
    started = time.time()
    input_meta, rows = load_basis(args.input, args.dimension)
    group_indices = {"A0": [], "q2": [], "q4": []}
    group_basis = {"A0": [], "q2": [], "q4": []}
    for index, row in enumerate(rows):
        boundaries = [int(value, 0) for value in row["boundary_inputs"]]
        q2 = int(row["q2"], 0)
        q4 = int(row["q4"], 0)
        if any(boundaries[1:]):
            raise ValueError("factorized FWHT requires A1=A2=A3=A4=0")
        active = sum(value != 0 for value in (boundaries[0], q2, q4))
        if active != 1:
            raise ValueError("each ranked basis row must belong to one factor")
        if boundaries[0]:
            name, value = "A0", boundaries[0]
        elif q2:
            name, value = "q2", q2
        else:
            name, value = "q4", q4
        group_indices[name].append(index)
        group_basis[name].append(value)

    a0_support, column_summary = exact_a0_support()
    sb1_zero = next(numerator for mask, numerator in a0_support if mask == 0)
    rest_support, q2_support_raw, q4_support_raw = exact_rest_support(sb1_zero)
    rest_map = {(q2, q4): value for q2, q4, value in rest_support}
    rest_zero = rest_map[(0, 0)]
    q2_support = [(q2, rest_map[(q2, 0)]) for q2, _ in q2_support_raw]
    q4_support = [(q4, rest_map[(0, q4)]) for q4, _ in q4_support_raw]

    components = {
        "A0": component_fwht(a0_support, group_basis["A0"]),
        "q2": component_fwht(q2_support, group_basis["q2"]),
        "q4": component_fwht(q4_support, group_basis["q4"]),
    }
    full_denominator = rest_zero * (1 << 448)
    if full_denominator < 0:
        full_denominator = -full_denominator
        components["A0"]["common_factor"] *= -1

    distribution = {}
    for a_entry in components["A0"]["histogram"]:
        for q2_entry in components["q2"]["histogram"]:
            for q4_entry in components["q4"]["histogram"]:
                numerator = (
                    a_entry["value"] * components["A0"]["common_factor"]
                    * q2_entry["value"] * components["q2"]["common_factor"]
                    * q4_entry["value"] * components["q4"]["common_factor"]
                )
                probability = Fraction(numerator, full_denominator)
                coordinate = (
                    embed_coordinate(a_entry["representative"], group_indices["A0"])
                    | embed_coordinate(q2_entry["representative"], group_indices["q2"])
                    | embed_coordinate(q4_entry["representative"], group_indices["q4"])
                )
                count = (
                    a_entry["count"] * q2_entry["count"] * q4_entry["count"]
                )
                entry = distribution.setdefault(
                    probability, {"count": 0, "representative": coordinate}
                )
                entry["count"] += count

    if sum(entry["count"] for entry in distribution.values()) != 1 << args.dimension:
        raise AssertionError("factorized probability class count mismatch")
    minimum = min(distribution)
    minimum_nonzero = min(probability for probability in distribution if probability > 0)
    maximum = max(distribution)
    minimum_coordinate = distribution[minimum]["representative"]
    minimum_nonzero_coordinate = distribution[minimum_nonzero]["representative"]
    maximum_coordinate = distribution[maximum]["representative"]
    maximum_conditions, full_maximum_conditions, full_key_masks = condition_records(
        rows, maximum_coordinate
    )
    minimum_conditions, full_minimum_conditions, _ = condition_records(
        rows, minimum_coordinate
    )
    minimum_nonzero_conditions, full_minimum_nonzero_conditions, _ = (
        condition_records(rows, minimum_nonzero_coordinate)
    )
    key_rank = RankAccumulator(full_key_masks).rank
    slice_names = ("w1", "w2", "rk1", "rk2", "rk3", "rk4", "rk5")
    key_slice_ranks = {
        name: RankAccumulator(
            (mask >> (64 * index)) & ((1 << 64) - 1)
            for mask in full_key_masks
        ).rank
        for index, name in enumerate(slice_names)
    }
    histogram = []
    for probability in sorted(distribution):
        histogram.append({
            "probability": float(probability),
            "probability_exact": {
                "numerator": probability.numerator,
                "denominator": probability.denominator,
            },
            "log2_probability": (
                math.log2(float(probability)) if probability > 0 else None
            ),
            "class_count": distribution[probability]["count"],
            "representative_coordinate": distribution[probability]["representative"],
        })

    record = {
        "cipher": "Blink-64",
        "scope": "Figure-3 ten-round cluster; A1=A2=A3=A4=0",
        "selection": "prefix of complete exact minimum-weight ranked basis",
        "basis_dimension": args.dimension,
        "basis_prefix": f"b0..b{args.dimension - 1}",
        "ranked_basis": str(args.input),
        "ranked_basis_meta": input_meta,
        "method": "exact factorized FWHT; equivalent to monolithic 2^d span FWHT",
        "spanned_trails": 1 << args.dimension,
        "nonzero_fourier_terms": math.prod(
            component["support_size"] for component in components.values()
        ),
        "component_dimensions": {
            name: component["dimension"] for name, component in components.items()
        },
        "component_support_sizes": {
            name: component["support_size"] for name, component in components.items()
        },
        "a0_column_support": column_summary,
        "p_max": float(maximum),
        "log2_p_max": math.log2(float(maximum)) if maximum > 0 else None,
        "p_max_exact": {
            "numerator": maximum.numerator,
            "denominator": maximum.denominator,
        },
        "p_max_class_count": distribution[maximum]["count"],
        "p_min": float(minimum),
        "p_min_exact": {
            "numerator": minimum.numerator,
            "denominator": minimum.denominator,
        },
        "p_min_class_count": distribution[minimum]["count"],
        "p_zero_class_count": distribution.get(Fraction(0), {"count": 0})["count"],
        "p_min_nonzero": float(minimum_nonzero),
        "log2_p_min_nonzero": math.log2(float(minimum_nonzero)),
        "p_min_nonzero_exact": {
            "numerator": minimum_nonzero.numerator,
            "denominator": minimum_nonzero.denominator,
        },
        "p_min_nonzero_class_count": distribution[minimum_nonzero]["count"],
        "probability_distinct_values": len(histogram),
        "probability_histogram_complete": True,
        "probability_histogram": histogram,
        "conditions": maximum_conditions,
        "minimum_conditions": minimum_conditions,
        "minimum_nonzero_conditions": minimum_nonzero_conditions,
        "full_cipher_relation": "X=P xor w1",
        "full_cipher_key_rank_for_fixed_plaintext": key_rank,
        "full_cipher_key_slice_ranks": key_slice_ranks,
        "full_cipher_conditions": full_maximum_conditions,
        "full_cipher_minimum_conditions": full_minimum_conditions,
        "full_cipher_minimum_nonzero_conditions": full_minimum_nonzero_conditions,
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"Exact factorized FWHT dimension={args.dimension} "
        f"components={record['component_dimensions']}",
        flush=True,
    )
    print(
        f"p_max={float(maximum):.17e}=2^{record['log2_p_max']:.12f} "
        f"classes={record['p_max_class_count']}",
        flush=True,
    )
    print(
        f"p_min={float(minimum):.17e} classes={record['p_min_class_count']}",
        flush=True,
    )
    print(
        f"p_min_nonzero={float(minimum_nonzero):.17e}="
        f"2^{record['log2_p_min_nonzero']:.12f} "
        f"classes={record['p_min_nonzero_class_count']}",
        flush=True,
    )
    print(f"Saved: {output}", flush=True)
    return record

