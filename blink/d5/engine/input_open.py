"""Factorized first-Superbox input analysis for CGZ+26 Blink-128.

The first Superbox has five effective columns.  Each has a rank-14 input
Fourier support, so materializing their Cartesian product would create a
``2^70`` table before the four key components are even included.  This module
keeps every column as an independent quotient table and combines only their
small probability histograms.
"""

from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path

from .column_spectrum import input_quotient, insert_column
from .figure7_trail import SUPERBOXES, superbox_column_pairs
from .gf2 import RankAccumulator
from .key_only import (
    MASTER_KEY_BITS,
    VARIABLE_COMPONENTS,
    component_histogram,
    condition_records,
    exact_record,
    fixed_superbox_probability,
    log2_fraction,
    published_condition_records,
    serialize_component,
    shifted_key_component,
)
from .linear_layer import STATE_BITS, STATE_COLUMNS


def input_component(column):
    source, target = superbox_column_pairs(0)[column]
    quotient = input_quotient(source, target, 0)
    basis = tuple(int(mask) for mask in quotient["basis"])
    state_basis = tuple(insert_column(mask, column) for mask in basis)
    joint_basis = tuple(mask << MASTER_KEY_BITS for mask in state_basis)
    return {
        "name": f"SB1_input_column{column}",
        "kind": "input",
        "superbox": 0,
        "column": column,
        "source_activity": source,
        "target_activity": target,
        "rank": len(basis),
        "support_size": int(quotient["support_size"]),
        "denominator_bits": int(quotient["denominator_bits"]),
        "local_basis": basis,
        "state_basis": state_basis,
        "joint_basis": joint_basis,
        "values": tuple(int(value) for value in quotient["values"]),
    }


def serialize_input_component(component):
    histogram = component_histogram(component)
    maximum = max(histogram)
    maximum_coordinates = [
        coordinate
        for coordinate, numerator in enumerate(component["values"])
        if Fraction(numerator, 1 << component["denominator_bits"]) == maximum
    ]
    return {
        "name": component["name"],
        "superbox": 1,
        "superbox_rounds": SUPERBOXES[0]["name"],
        "column": component["column"],
        "pattern": (
            f"{component['source_activity']:04b}-to-"
            f"{component['target_activity']:04b}"
        ),
        "rank": component["rank"],
        "support_size": component["support_size"],
        "local_basis": [f"0x{mask:04x}" for mask in component["local_basis"]],
        "input_state_basis": [
            f"0x{mask:032x}" for mask in component["state_basis"]
        ],
        "joint_basis": [
            f"0x{mask:0288x}" for mask in component["joint_basis"]
        ],
        "quotient_probability_numerators": list(component["values"]),
        "quotient_probability_denominator_bits": component["denominator_bits"],
        "maximum": exact_record(maximum),
        "maximum_coordinates": maximum_coordinates,
        "histogram": [
            {
                "probability_exact": exact_record(probability),
                "class_count": row["count"],
                "representative_coordinate": row["representative"],
            }
            for probability, row in sorted(histogram.items())
        ],
    }


def convolve_component_histograms(components, fixed_factor):
    """Return an exact global histogram without enumerating joint classes."""
    distribution = {
        Fraction(fixed_factor): {"count": 1, "representative": 0}
    }
    shift = 0
    for component in components:
        local_histogram = component_histogram(component)
        updated = defaultdict(lambda: {"count": 0, "representative": 0})
        for left_probability, left in distribution.items():
            for right_probability, right in local_histogram.items():
                probability = left_probability * right_probability
                row = updated[probability]
                if not row["count"]:
                    row["representative"] = (
                        left["representative"]
                        | (right["representative"] << shift)
                    )
                row["count"] += left["count"] * right["count"]
        distribution = dict(updated)
        shift += component["rank"]
    return distribution


def representative_input_conditions(input_components, coordinate):
    records = []
    shift = 0
    for component in input_components:
        local_coordinate = (coordinate >> shift) & ((1 << component["rank"]) - 1)
        for index, (local_mask, state_mask) in enumerate(
            zip(component["local_basis"], component["state_basis"])
        ):
            bit = local_coordinate >> index & 1
            expression = " xor ".join(
                f"X[{state_bit}]"
                for state_bit in range(STATE_BITS)
                if state_mask >> state_bit & 1
            ) or "0"
            records.append(
                {
                    "component": component["name"],
                    "column": component["column"],
                    "local_mask": f"0x{local_mask:04x}",
                    "input_mask": f"0x{state_mask:032x}",
                    "right_hand_side": bit,
                    "equation": f"{expression} = {bit}",
                }
            )
        shift += component["rank"]
    return records


def compute_input_open_distribution():
    all_input_components = tuple(
        input_component(column) for column in range(STATE_COLUMNS)
    )
    input_components = tuple(
        component for component in all_input_components if component["rank"]
    )
    key_components = tuple(
        shifted_key_component(specification)
        for specification in VARIABLE_COMPONENTS
    )
    components = input_components + key_components

    input_rank = sum(component["rank"] for component in input_components)
    key_rank = sum(component["rank"] for component in key_components)
    joint_rank = input_rank + key_rank
    joint_basis = [
        mask for component in input_components for mask in component["joint_basis"]
    ] + [
        mask for component in key_components for mask in component["master_basis"]
    ]
    if RankAccumulator(joint_basis).rank != joint_rank:
        raise AssertionError("input and master-key quotient components overlap")

    fixed_probabilities = {
        "SB3_rounds_6_7": fixed_superbox_probability(2),
        "SB5_rounds_10_11": fixed_superbox_probability(4),
    }
    fixed_factor = math.prod(fixed_probabilities.values(), start=Fraction(1))
    distribution = convolve_component_histograms(components, fixed_factor)
    if sum(row["count"] for row in distribution.values()) != 1 << joint_rank:
        raise AssertionError("factorized 84D class count mismatch")

    maximum = max(distribution)
    minimum = min(distribution)
    minimum_nonzero = min(value for value in distribution if value > 0)
    maximum_coordinate = distribution[maximum]["representative"]

    good_input_classes = math.prod(
        sum(
            1
            for numerator in component["values"]
            if Fraction(numerator, 1 << component["denominator_bits"]) == 1
        )
        for component in input_components
    )
    good_input_fraction = Fraction(good_input_classes, 1 << input_rank)
    if maximum * good_input_fraction != Fraction(9, 1 << 72):
        raise AssertionError("input averaging does not recover the key-only maximum")

    # The global coordinate concatenates input components first, then key
    # components.  Extract the key part before reusing the key condition helper.
    key_coordinate = maximum_coordinate >> input_rank
    input_fourier_support_size = math.prod(
        component["support_size"] for component in all_input_components
    )
    key_fourier_support_size = math.prod(
        component["support_size"] for component in key_components
    )
    return {
        "cipher": "Blink-128",
        "source": "CGZ+26 Figure 7",
        "scope": (
            "10-round trail, rounds 2--11; connector masks A1=A2=A3=A4=0; "
            "first Superbox input masks open"
        ),
        "tweak": 0,
        "representation": (
            "exact tensor product of per-column Fourier quotients; no 8-column "
            "input-candidate Cartesian product is materialized"
        ),
        "input_state_bits": STATE_BITS,
        "input_support_rank": input_rank,
        "master_key_support_rank": key_rank,
        "joint_support_rank": joint_rank,
        "spanned_joint_classes": 1 << joint_rank,
        "input_fourier_support_size": input_fourier_support_size,
        "key_fourier_support_size": key_fourier_support_size,
        "global_fourier_support_size": (
            input_fourier_support_size * key_fourier_support_size
        ),
        "effective_input_columns": [
            component["column"] for component in input_components
        ],
        "input_column_ranks": {
            str(component["column"]): component["rank"]
            for component in all_input_components
        },
        "fixed_superbox_probabilities": {
            name: exact_record(value) for name, value in fixed_probabilities.items()
        },
        "fixed_factor": exact_record(fixed_factor),
        "input_components": [
            serialize_input_component(component)
            for component in all_input_components
        ],
        "key_components": [
            serialize_component(component) for component in key_components
        ],
        "probability_histogram_complete": True,
        "probability_distinct_values": len(distribution),
        "probability_histogram": [
            {
                "probability_exact": exact_record(value),
                "log2_probability": log2_fraction(value),
                "class_count": row["count"],
                "representative_coordinate": row["representative"],
            }
            for value, row in sorted(distribution.items())
        ],
        "p_max_exact": exact_record(maximum),
        "log2_p_max": log2_fraction(maximum),
        "p_max_class_count": distribution[maximum]["count"],
        "p_max_representative_coordinate": maximum_coordinate,
        "p_max_representative_input_conditions": representative_input_conditions(
            input_components, maximum_coordinate
        ),
        "p_max_key_conditions": published_condition_records(
            key_components, key_coordinate
        ),
        "p_max_key_basis_conditions": condition_records(
            key_components, key_coordinate
        ),
        "p_min_exact": exact_record(minimum),
        "p_min_class_count": distribution[minimum]["count"],
        "p_zero_class_count": distribution.get(
            Fraction(0), {"count": 0}
        )["count"],
        "p_min_nonzero_exact": exact_record(minimum_nonzero),
        "log2_p_min_nonzero": log2_fraction(minimum_nonzero),
        "p_min_nonzero_class_count": distribution[minimum_nonzero]["count"],
        "good_input_quotient_classes": good_input_classes,
        "total_input_quotient_classes": 1 << input_rank,
        "good_input_fraction": exact_record(good_input_fraction),
        "input_kernel_dimension": STATE_BITS - input_rank,
        "good_raw_input_count": good_input_classes << (STATE_BITS - input_rank),
        "key_only_maximum_recovered_by_input_average": exact_record(
            maximum * good_input_fraction
        ),
        "conditional_maximum_gain_over_key_only": 1 << 20,
        "full_cipher_relation": "X = P xor w1",
        "fixed_plaintext_master_key_rank": joint_rank,
        "maximum_input_condition_form": (
            "For each effective column, evaluate the listed rank-14 syndrome "
            "and require membership in that component's maximum_coordinates; "
            "do not interpret the 70 representative equations as a necessary "
            "description of all maximum inputs."
        ),
    }


def write_input_open_distribution(path):
    record = compute_input_open_distribution()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
