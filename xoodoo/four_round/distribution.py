"""Build capacity-checked input-value correlation distributions for R4 cases.

The ordinary joint-U sweep selects stable Fourier directions in the chi-input
coordinates.  A coset-aware rerun then preserves the complete selected span.
For complete-round boundaries this module pulls every selected direction back
through ``(rho_west o theta)^T`` and applies the first-round-constant phase
before the inverse Walsh transform.  Consequently the recorded classes are
constraints on the *complete permutation input*, not on the post-linear chi
input.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Sequence

from engine import common
from engine import linear_layer as xoodoo_linear
from engine.config import DISTINGUISHERS, get
from basis import run_coset_from_basis
from calibration import CASES


R4_CASES = (
    "r4_public_full_standard",
    "r4_public_full_extension_core",
    "r4_full_new_5_to_33_2",
    "r4_full_new_5_to_83_4",
    "r4_full_new_3_to_37_4",
    "r4_full_new_3_to_48_4",
    "r4_full_new_6_to_33_4",
    "r4_chi_zero_scan_best",
    "r4_chi_one_scan_best",
)


def _mask_int(words: Sequence[str]) -> int:
    return common.words_to_int(tuple(int(word, 16) for word in words))


def _mask_record(mask: int, *, full_input: bool) -> dict:
    expression = common.mask_to_expression(common.int_to_words(mask))
    if full_input:
        expression = expression.replace("A[", "X[")
    return {
        "mask_words": common.mask_to_hex(common.int_to_words(mask)),
        "expression": expression,
    }


def _subset_mask(basis: Sequence[int], index: int) -> int:
    result = 0
    for bit, value in enumerate(basis):
        if (index >> bit) & 1:
            result ^= value
    return result


def _rank_indices(values: Iterable[int]) -> int:
    rows: dict[int, int] = {}
    for raw in values:
        value = int(raw)
        while value:
            pivot = value.bit_length() - 1
            if pivot in rows:
                value ^= rows[pivot]
            else:
                rows[pivot] = value
                break
    return len(rows)


def _signed_power(value: float) -> str:
    if value == 0.0:
        return "0"
    sign = "+" if value > 0 else "-"
    return f"{sign}2^-{-math.log2(abs(value)):.9f}"


def _distribution_summary(
    coefficients: Sequence[float],
    *,
    meaningful_relative_threshold: float,
    zero_relative_tolerance: float,
) -> dict:
    coefficients = [float(value) for value in coefficients]
    distribution = common.fwht(coefficients).tolist()
    coefficient_scale = max((abs(value) for value in coefficients), default=0.0)
    class_scale = max((abs(value) for value in distribution), default=0.0)
    meaningful_threshold = coefficient_scale * meaningful_relative_threshold
    zero_tolerance = class_scale * zero_relative_tolerance
    meaningful_support = [
        index
        for index, value in enumerate(coefficients)
        if index and abs(value) > meaningful_threshold
    ]
    nonzero = [value for value in distribution if abs(value) > zero_tolerance]
    maximum_index = (
        max(range(len(distribution)), key=lambda index: abs(distribution[index]))
        if distribution
        else 0
    )
    minimum = min((abs(value) for value in nonzero), default=0.0)
    maximum = max((abs(value) for value in nonzero), default=0.0)
    dimension = (len(coefficients).bit_length() - 1) if coefficients else 0
    return {
        "fourier_coefficients": coefficients,
        "correlation_distribution": distribution,
        "meaningful_fourier_relative_threshold": meaningful_relative_threshold,
        "meaningful_fourier_absolute_threshold": meaningful_threshold,
        "meaningful_nonzero_fourier_coordinates": len(meaningful_support),
        "effective_constraint_rank": _rank_indices(meaningful_support),
        "class_zero_relative_tolerance": zero_relative_tolerance,
        "class_zero_absolute_tolerance": zero_tolerance,
        "zero_class_count": len(distribution) - len(nonzero),
        "nonzero_class_count": len(nonzero),
        "minimum_nonzero_absolute_correlation": minimum,
        "minimum_nonzero_signed_power_weight": (
            -math.log2(minimum) if minimum else None
        ),
        "maximum_nonzero_absolute_correlation": maximum,
        "maximum_nonzero_signed_power_weight": (
            -math.log2(maximum) if maximum else None
        ),
        "maximum_absolute_class_index": maximum_index,
        "maximum_absolute_assignment": [
            (maximum_index >> bit) & 1 for bit in range(dimension)
        ],
        "maximum_absolute_class_correlation": (
            distribution[maximum_index] if distribution else 0.0
        ),
        "maximum_absolute_class_signed_power": (
            _signed_power(distribution[maximum_index]) if distribution else "0"
        ),
    }


def _capacity_drift(left: Sequence[float], right: Sequence[float]) -> dict:
    scale = max(
        max((abs(value) for value in left), default=0.0),
        max((abs(value) for value in right), default=0.0),
    )
    maximum = max(
        (abs(float(a) - float(b)) for a, b in zip(left, right)),
        default=0.0,
    )
    return {
        "maximum_absolute_drift": maximum,
        "maximum_relative_drift": maximum / scale if scale else 0.0,
    }


def _class_stability(
    left: Sequence[float],
    right: Sequence[float],
    *,
    relative_strength_threshold: float,
    relative_drift_limit: float,
) -> dict:
    scale = max(
        max((abs(value) for value in left), default=0.0),
        max((abs(value) for value in right), default=0.0),
    )
    threshold = scale * relative_strength_threshold
    stable_nonzero = []
    zero = []
    unstable = []
    for index, (old, new) in enumerate(zip(left, right)):
        old = float(old)
        new = float(new)
        local_scale = max(abs(old), abs(new))
        if local_scale <= threshold:
            zero.append(index)
            continue
        drift = abs(new - old) / local_scale
        if old * new > 0.0 and drift <= relative_drift_limit:
            stable_nonzero.append(index)
        else:
            unstable.append(
                {
                    "index": index,
                    "lower_capacity_value": old,
                    "higher_capacity_value": new,
                    "relative_drift": drift,
                }
            )
    stable_values = [float(right[index]) for index in stable_nonzero]
    minimum = min((abs(value) for value in stable_values), default=0.0)
    maximum = max((abs(value) for value in stable_values), default=0.0)
    return {
        "relative_strength_threshold": relative_strength_threshold,
        "absolute_strength_threshold": threshold,
        "relative_drift_limit": relative_drift_limit,
        "stable_nonzero_class_count": len(stable_nonzero),
        "stable_nonzero_class_indices": stable_nonzero,
        "stable_zero_class_count": len(zero),
        "stable_zero_class_indices": zero,
        "unstable_class_count": len(unstable),
        "unstable_classes": unstable,
        "stable_nonzero_minimum_absolute_correlation": minimum,
        "stable_nonzero_minimum_weight": -math.log2(minimum) if minimum else None,
        "stable_nonzero_maximum_absolute_correlation": maximum,
        "stable_nonzero_maximum_weight": -math.log2(maximum) if maximum else None,
    }


def analyze(
    case_name: str,
    prefix_rounds: int,
    *,
    run_coset: bool,
    coset_capacities: tuple[int, int] = (32, 64),
    meaningful_relative_threshold: float = 1e-6,
    zero_relative_tolerance: float = 1e-12,
) -> dict:
    if case_name not in CASES or case_name not in R4_CASES:
        raise ValueError(f"unsupported R4 case {case_name!r}")
    if prefix_rounds not in (0, 1):
        raise ValueError("this distribution driver currently accepts p=0 or p=1")
    case = CASES[case_name]
    config = case.config
    mode = "identity" if prefix_rounds == 0 else "complete"
    core_directory = (
        common.OUTPUT_DIR
        / config.name
        / "joint_ddt_prefix"
        / f"p{prefix_rounds}_{mode}"
    )
    basis_path = core_directory / "stable_basis_diagonal.json"
    if not basis_path.is_file():
        raise FileNotFoundError(f"missing stable basis: {basis_path}")
    basis_record = common.read_json(basis_path)
    chi_basis = tuple(
        _mask_int(item["mask_words"]) for item in basis_record["selected_basis"]
    )
    full_input = any(view.pull_back_to_full_input for view in case.views)

    if chi_basis and run_coset:
        if prefix_rounds == 0:
            sweep = run_coset_from_basis(
                basis_path,
                config,
                suffix_capacities=(64, 128),
                prefix_capacities=(64, 128),
                aggregate_capacities=(64, 128),
            )
        else:
            low, high = coset_capacities
            sweep = run_coset_from_basis(
                basis_path,
                config,
                suffix_capacities=(low, high),
                prefix_capacities=(64, 64),
                aggregate_capacities=(64, 64),
            )
        sweep_path = Path(sweep["capacities"][-1]["root_dump_path"]).parent / "capacity_sweep.json"
    elif chi_basis:
        candidates = sorted(core_directory.glob("coset_*/capacity_sweep.json"))
        if not candidates:
            raise FileNotFoundError(f"no coset sweep below {core_directory}")
        sweep_path = candidates[-1]
        sweep = common.read_json(sweep_path)
    else:
        sweep_path = core_directory / "capacity_sweep.json"
        sweep = common.read_json(sweep_path)

    first_constant = common.ROUND_CONSTANTS[12 - config.rounds]
    constant_state = xoodoo_linear.constant_state(first_constant)
    reported_basis = (
        tuple(xoodoo_linear.pre_chi_adjoint(value) for value in chi_basis)
        if full_input
        else chi_basis
    )
    capacities = []
    for item in sweep["capacities"]:
        if chi_basis:
            span = item.get("coset_span")
            if span is None or int(span["basis_dimension"]) != len(chi_basis):
                continue
            coefficients = [float(value) for value in span["fourier_coefficients"]]
        else:
            coefficients = [float(item["joint_zero_label_coefficient"])]
        if full_input:
            coefficients = [
                (-value if (_subset_mask(chi_basis, index) & constant_state).bit_count() & 1 else value)
                for index, value in enumerate(coefficients)
            ]
        capacities.append(
            {
                "suffix_capacity": item["suffix_capacity"],
                "prefix_capacity": item["prefix_capacity"],
                "aggregate_capacity": item["aggregate_capacity"],
                "joint_U0_coefficient": coefficients[0],
                "joint_U0_signed_power": _signed_power(coefficients[0]),
                **_distribution_summary(
                    coefficients,
                    meaningful_relative_threshold=meaningful_relative_threshold,
                    zero_relative_tolerance=zero_relative_tolerance,
                ),
            }
        )
    if len(capacities) < 2:
        # Dimension-zero cases reuse the last two ordinary capacity points.
        capacities = []
        for item in sweep["capacities"][-2:]:
            coefficient = float(item["joint_zero_label_coefficient"])
            capacities.append(
                {
                    "suffix_capacity": item["suffix_capacity"],
                    "prefix_capacity": item["prefix_capacity"],
                    "aggregate_capacity": item["aggregate_capacity"],
                    "joint_U0_coefficient": coefficient,
                    "joint_U0_signed_power": _signed_power(coefficient),
                    **_distribution_summary(
                        [coefficient],
                        meaningful_relative_threshold=meaningful_relative_threshold,
                        zero_relative_tolerance=zero_relative_tolerance,
                    ),
                }
            )
    distribution_drift = _capacity_drift(
        capacities[-2]["correlation_distribution"],
        capacities[-1]["correlation_distribution"],
    )
    class_stability = _class_stability(
        capacities[-2]["correlation_distribution"],
        capacities[-1]["correlation_distribution"],
        relative_strength_threshold=meaningful_relative_threshold,
        relative_drift_limit=0.01,
    )
    zero_scale = max(
        abs(capacities[-2]["joint_U0_coefficient"]),
        abs(capacities[-1]["joint_U0_coefficient"]),
    )
    zero_drift = abs(
        capacities[-1]["joint_U0_coefficient"]
        - capacities[-2]["joint_U0_coefficient"]
    )
    result = {
        "version": 1,
        "case": case_name,
        "boundary": (
            "complete-round input constraints"
            if full_input
            else "first-chi-input constraints"
        ),
        "ddt_prefix_rounds": prefix_rounds,
        "prefix_mode": mode,
        "source_basis": str(basis_path),
        "source_coset_sweep": str(sweep_path),
        "selected_basis_dimension": len(reported_basis),
        "chi_input_basis": [
            _mask_record(value, full_input=False) for value in chi_basis
        ],
        "reported_input_basis": [
            _mask_record(value, full_input=full_input) for value in reported_basis
        ],
        "first_round_constant_phase_applied": full_input,
        "capacities": capacities,
        "highest_two_joint_U0_relative_drift": (
            zero_drift / zero_scale if zero_scale else 0.0
        ),
        "highest_two_distribution_drift": distribution_drift,
        "highest_two_class_stability": class_stability,
        "capacity_stable_at_one_percent": (
            (zero_drift / zero_scale if zero_scale else 0.0) <= 0.01
            and class_stability["unstable_class_count"] == 0
        ),
        "scope": (
            "correlation distribution on the selected stable input-expression "
            "span; not the full 384-dimensional value distribution"
        ),
    }
    output_directory = (
        common.OUTPUT_DIR
        / "known_distinguisher_joint"
        / case_name
        / f"p{prefix_rounds}_{mode}"
    )
    common.write_json(output_directory / "distribution_analysis.json", result)
    print(
        f"[{case_name}] p={prefix_rounds}: d={len(reported_basis)}, "
        f"stable={result['capacity_stable_at_one_percent']}, "
        f"stable-range=[{class_stability['stable_nonzero_minimum_absolute_correlation']:.12g},"
        f"{class_stability['stable_nonzero_maximum_absolute_correlation']:.12g}], "
        f"zero={class_stability['stable_zero_class_count']}, "
        f"unstable={class_stability['unstable_class_count']}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=(*R4_CASES, "all"))
    parser.add_argument("prefix_rounds", type=int, choices=(0, 1))
    parser.add_argument("--run-coset", action="store_true")
    parser.add_argument("--coset-capacities", default="32,64")
    args = parser.parse_args()
    coset_capacities = tuple(
        int(value) for value in args.coset_capacities.split(",") if value.strip()
    )
    if len(coset_capacities) != 2 or any(value <= 0 for value in coset_capacities):
        parser.error("coset capacities must contain two positive integers")
    selected = R4_CASES if args.case == "all" else (args.case,)
    for case_name in selected:
        analyze(
            case_name,
            args.prefix_rounds,
            run_coset=args.run_coset,
            coset_capacities=coset_capacities,
        )


if __name__ == "__main__":
    main()
