"""Exact 14-dimensional master-key distribution for CGZ+26 Figure 7."""

from collections import defaultdict
from fractions import Fraction
import json
import math
from pathlib import Path

from .column_spectrum import (
    extract_column,
    insert_column,
    key_probability_numerators,
    key_quotient,
    mix_column_word,
)
from .figure7_trail import SUPERBOXES, superbox_column_pairs
from .gf2 import LinearCoordinates, RankAccumulator
from .linear_layer import (
    INVERSE_ROUND_CONSTANTS,
    ROUND_CONSTANTS,
    STATE_BITS,
    STATE_COLUMNS,
)


MASTER_KEY_BITS = 1024
KEY_SLICE_OFFSETS = {
    "w1": 0,
    "w2": 128,
    "rk1": 256,
    "rk2": 384,
    "rk3": 512,
    "rk4": 640,
    "rk5": 768,
    "rk6": 896,
}

# Figure 7: the only variable middle values when connector masks are zero.
VARIABLE_COMPONENTS = (
    {
        "name": "SB2_column4_2to2",
        "superbox": 1,
        "column": 4,
        "key_slice": "rk5",
        "round_constant": ROUND_CONSTANTS[4],
        "key_transform": "identity",
    },
    {
        "name": "SB2_column5_4to2",
        "superbox": 1,
        "column": 5,
        "key_slice": "rk5",
        "round_constant": ROUND_CONSTANTS[4],
        "key_transform": "identity",
    },
    {
        "name": "SB4_column4_2to2",
        "superbox": 3,
        "column": 4,
        "key_slice": "rk2",
        "round_constant": INVERSE_ROUND_CONSTANTS[1],
        "key_transform": "M",
    },
    {
        "name": "SB4_column5_2to4",
        "superbox": 3,
        "column": 5,
        "key_slice": "rk2",
        "round_constant": INVERSE_ROUND_CONSTANTS[1],
        "key_transform": "M",
    },
)


def parity(value):
    return int(value).bit_count() & 1


def exact_record(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator}


def log2_fraction(value):
    value = Fraction(value)
    if value <= 0:
        return None
    return math.log2(value.numerator) - math.log2(value.denominator)


def mask_expression(mask, prefix):
    return " xor ".join(
        f"{prefix}[{bit}]" for bit in range(STATE_BITS) if mask >> bit & 1
    ) or "0"


def fixed_superbox_probability(superbox_index):
    """Average cluster probability for a known-zero middle Superbox."""
    value = Fraction(1)
    for source, target in superbox_column_pairs(superbox_index):
        numerator = int(key_probability_numerators(source, target)[0])
        value *= Fraction(numerator, 1 << 16)
    return value


def shifted_key_component(specification):
    superbox = int(specification["superbox"])
    column = int(specification["column"])
    source, target = superbox_column_pairs(superbox)[column]
    quotient = key_quotient(source, target)
    basis = tuple(int(mask) for mask in quotient["basis"])
    if specification["key_transform"] == "identity":
        actual_basis = basis
    elif specification["key_transform"] == "M":
        # The inverse-side canonical Table-3 key is M(rk2 xor rc2').
        # Hence q.M(rk2 xor rc2') = M(q).(rk2 xor rc2') because M is
        # symmetric and involutory.
        actual_basis = tuple(mix_column_word(mask) for mask in basis)
    else:
        raise ValueError(f"unknown key transform {specification['key_transform']}")
    constant_word = extract_column(specification["round_constant"], column)
    constant_syndrome = sum(
        parity(mask & constant_word) << index
        for index, mask in enumerate(actual_basis)
    )
    raw_values = quotient["values"]
    # The public coordinate is the syndrome of rk, whereas the local table is
    # indexed by the syndrome of rk xor rc.
    values = tuple(
        int(raw_values[coordinate ^ constant_syndrome])
        for coordinate in range(1 << len(basis))
    )
    state_basis = tuple(insert_column(mask, column) for mask in actual_basis)
    offset = KEY_SLICE_OFFSETS[specification["key_slice"]]
    master_basis = tuple(mask << offset for mask in state_basis)
    return {
        **specification,
        "source_activity": source,
        "target_activity": target,
        "rank": len(basis),
        "support_size": int(quotient["support_size"]),
        "denominator_bits": int(quotient["denominator_bits"]),
        "constant_column": constant_word,
        "constant_syndrome": constant_syndrome,
        "local_basis": basis,
        "actual_key_local_basis": actual_basis,
        "state_basis": state_basis,
        "master_basis": master_basis,
        "values": values,
    }


def component_histogram(component):
    grouped = defaultdict(lambda: {"count": 0, "representative": 0})
    denominator = 1 << component["denominator_bits"]
    for coordinate, numerator in enumerate(component["values"]):
        probability = Fraction(numerator, denominator)
        row = grouped[probability]
        if not row["count"]:
            row["representative"] = coordinate
        row["count"] += 1
    return grouped


def serialize_component(component):
    histogram = component_histogram(component)
    maximum = max(histogram)
    maximum_coordinates = [
        coordinate
        for coordinate, numerator in enumerate(component["values"])
        if Fraction(numerator, 1 << component["denominator_bits"]) == maximum
    ]
    return {
        "name": component["name"],
        "superbox": component["superbox"] + 1,
        "superbox_rounds": SUPERBOXES[component["superbox"]]["name"],
        "column": component["column"],
        "pattern": (
            f"{component['source_activity']:04b}-to-"
            f"{component['target_activity']:04b}"
        ),
        "key_slice": component["key_slice"],
        "key_transform": component["key_transform"],
        "rank": component["rank"],
        "support_size": component["support_size"],
        "round_constant_column": f"0x{component['constant_column']:04x}",
        "round_constant_syndrome": component["constant_syndrome"],
        "local_basis": [f"0x{mask:04x}" for mask in component["local_basis"]],
        "actual_key_local_basis": [
            f"0x{mask:04x}" for mask in component["actual_key_local_basis"]
        ],
        "state_basis": [f"0x{mask:032x}" for mask in component["state_basis"]],
        "master_key_basis": [
            f"0x{mask:0256x}" for mask in component["master_basis"]
        ],
        "quotient_probability_numerators": list(component["values"]),
        "quotient_probability_denominator_bits": component["denominator_bits"],
        "maximum": exact_record(maximum),
        "maximum_coordinates": maximum_coordinates,
        "histogram": [
            {
                "probability_exact": exact_record(probability),
                "class_count": data["count"],
                "representative_coordinate": data["representative"],
            }
            for probability, data in sorted(histogram.items())
        ],
    }


def condition_records(components, coordinate):
    records = []
    shift = 0
    for component in components:
        local_coordinate = (coordinate >> shift) & ((1 << component["rank"]) - 1)
        for index, (local_mask, state_mask, master_mask) in enumerate(
            zip(
                component["local_basis"],
                component["state_basis"],
                component["master_basis"],
            )
        ):
            bit = local_coordinate >> index & 1
            expression = mask_expression(state_mask, component["key_slice"])
            records.append(
                {
                    "component": component["name"],
                    "key_slice": component["key_slice"],
                    "column": component["column"],
                    "local_mask": f"0x{local_mask:04x}",
                    "state_mask": f"0x{state_mask:032x}",
                    "master_key_mask": f"0x{master_mask:0256x}",
                    "right_hand_side": bit,
                    "equation": f"{expression} = {bit}",
                }
            )
        shift += component["rank"]
    return records


def published_condition_records(components, maximum_coordinate):
    """Return and verify CGZ+26's 14 raw rk5/rk2 conditions."""
    specifications = []
    common = (
        ((1 << 16) ^ (1 << 112), 1),
        ((1 << 17) ^ (1 << 113), 0),
        ((1 << 18) ^ (1 << 114), 0),
        (1 << 20, 1),
        (1 << 22, 1),
        (1 << 84, 1),
    )
    for key_slice, final_bit in (("rk5", 1), ("rk2", 0)):
        specifications.extend(
            (key_slice, state_mask, rhs) for state_mask, rhs in common
        )
        specifications.append((key_slice, 1 << 86, final_bit))

    master_basis = tuple(
        mask for component in components for mask in component["master_basis"]
    )
    coordinates = LinearCoordinates(master_basis)
    records = []
    published_masks = []
    for key_slice, state_mask, expected_rhs in specifications:
        master_mask = state_mask << KEY_SLICE_OFFSETS[key_slice]
        combination = coordinates.coordinate(master_mask)
        if combination is None:
            raise AssertionError("published condition is outside computed support")
        actual_rhs = parity(combination & maximum_coordinate)
        if actual_rhs != expected_rhs:
            raise AssertionError(
                f"published {key_slice} condition RHS mismatch: "
                f"{mask_expression(state_mask, key_slice)}={actual_rhs}, "
                f"expected {expected_rhs}"
            )
        published_masks.append(master_mask)
        records.append(
            {
                "key_slice": key_slice,
                "state_mask": f"0x{state_mask:032x}",
                "master_key_mask": f"0x{master_mask:0256x}",
                "right_hand_side": expected_rhs,
                "equation": (
                    f"{mask_expression(state_mask, key_slice)} = {expected_rhs}"
                ),
            }
        )
    if RankAccumulator(published_masks).rank != 14:
        raise AssertionError("published conditions are not rank 14")
    if RankAccumulator(master_basis + tuple(published_masks)).rank != 14:
        raise AssertionError("published and computed condition spaces differ")
    return records


def compute_key_only_distribution():
    fixed_probabilities = {
        "SB1_rounds_2_3": fixed_superbox_probability(0),
        "SB3_rounds_6_7": fixed_superbox_probability(2),
        "SB5_rounds_10_11": fixed_superbox_probability(4),
    }
    fixed_factor = math.prod(fixed_probabilities.values(), start=Fraction(1))
    components = tuple(shifted_key_component(spec) for spec in VARIABLE_COMPONENTS)
    dimension = sum(component["rank"] for component in components)
    master_basis = [
        mask for component in components for mask in component["master_basis"]
    ]
    if RankAccumulator(master_basis).rank != dimension:
        raise AssertionError("embedded master-key component bases overlap")

    probabilities = []
    histogram = defaultdict(lambda: {"count": 0, "representative": 0})
    for coordinate in range(1 << dimension):
        value = fixed_factor
        shift = 0
        for component in components:
            local = (coordinate >> shift) & ((1 << component["rank"]) - 1)
            value *= Fraction(
                component["values"][local],
                1 << component["denominator_bits"],
            )
            shift += component["rank"]
        probabilities.append(value)
        row = histogram[value]
        if not row["count"]:
            row["representative"] = coordinate
        row["count"] += 1

    if sum(row["count"] for row in histogram.values()) != 1 << dimension:
        raise AssertionError("14D probability class count mismatch")
    maximum = max(histogram)
    minimum = min(histogram)
    minimum_nonzero = min(value for value in histogram if value > 0)
    maximum_coordinate = histogram[maximum]["representative"]
    common_denominator = max(value.denominator for value in probabilities)
    if common_denominator & (common_denominator - 1):
        raise AssertionError("Blink quotient denominator is not a power of two")
    table_numerators = [
        value.numerator * (common_denominator // value.denominator)
        for value in probabilities
    ]

    basis_conditions = condition_records(components, maximum_coordinate)
    published_conditions = published_condition_records(
        components, maximum_coordinate
    )
    return {
        "cipher": "Blink-128",
        "source": "CGZ+26 Figure 7",
        "scope": "10-round trail, rounds 2--11; connector masks fixed to zero",
        "tweak": 0,
        "distribution_variables": "1024-bit master key quotient",
        "basis_dimension": dimension,
        "spanned_key_classes": 1 << dimension,
        "master_key_support_rank": RankAccumulator(master_basis).rank,
        "global_fourier_support_size": math.prod(
            component["support_size"] for component in components
        ),
        "variable_component_dimensions": {
            component["name"]: component["rank"] for component in components
        },
        "fixed_superbox_probabilities": {
            name: exact_record(value) for name, value in fixed_probabilities.items()
        },
        "fixed_factor": exact_record(fixed_factor),
        "components": [serialize_component(component) for component in components],
        "probability_table_coordinate_order": [
            {
                "component": component["name"],
                "rank": component["rank"],
                "least_significant_coordinate_first": True,
            }
            for component in components
        ],
        "probability_table_common_denominator_bits": common_denominator.bit_length() - 1,
        "probability_table_numerators": table_numerators,
        "probability_histogram_complete": True,
        "probability_distinct_values": len(histogram),
        "probability_histogram": [
            {
                "probability_exact": exact_record(value),
                "log2_probability": log2_fraction(value),
                "class_count": row["count"],
                "representative_coordinate": row["representative"],
            }
            for value, row in sorted(histogram.items())
        ],
        "p_max_exact": exact_record(maximum),
        "log2_p_max": log2_fraction(maximum),
        "p_max_class_count": histogram[maximum]["count"],
        "p_max_representative_coordinate": maximum_coordinate,
        "p_max_conditions": published_conditions,
        "p_max_basis_conditions": basis_conditions,
        "p_min_exact": exact_record(minimum),
        "p_min_class_count": histogram[minimum]["count"],
        "p_zero_class_count": histogram.get(
            Fraction(0), {"count": 0}
        )["count"],
        "p_min_nonzero_exact": exact_record(minimum_nonzero),
        "log2_p_min_nonzero": log2_fraction(minimum_nonzero),
        "p_min_nonzero_class_count": histogram[minimum_nonzero]["count"],
        "published_maximum": "9/2^72",
        "published_maximum_reproduced": maximum == Fraction(9, 1 << 72),
    }


def write_key_only_distribution(path):
    record = compute_key_only_distribution()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
