"""Reproduce the missing endpoint experiments for Xoodoo appendix tables.

The model-selected input classes are read from the existing p=0 and p=1
``distribution_analysis.json`` records.  Experiments are performed on the real
four-round Xoodoo permutation, including all round constants and both boundary
linear layers.  The output is a resumable, machine-readable JSON record.

This driver deliberately does not edit the paper's LaTeX sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

HERE = Path(__file__).resolve().parent

from engine import common
from engine import permutation as xoodoo_experiment
from engine.config import get


THEORY_ROOT = HERE / "output" / "known_distinguisher_joint"
DEFAULT_OUTPUT = HERE / "results" / "four_round_experiments.json"


@dataclass(frozen=True)
class TableCase:
    name: str
    target: str
    input_columns: tuple[tuple[int, int], ...]
    output_columns: tuple[tuple[int, int], ...]


TABLE_CASES = (
    TableCase("r4_public_full_standard", "(0,6)->(15,2)", ((0, 6),), ((15, 2),)),
    TableCase(
        "r4_public_full_extension_core",
        "(0,5)->(0,1)",
        ((0, 5),),
        ((0, 1),),
    ),
    TableCase("r4_full_new_5_to_33_2", "(0,5)->(33,2)", ((0, 5),), ((33, 2),)),
    TableCase("r4_full_new_5_to_83_4", "(0,5)->(83,4)", ((0, 5),), ((83, 4),)),
    TableCase("r4_full_new_3_to_37_4", "(0,3)->(37,4)", ((0, 3),), ((37, 4),)),
    TableCase("r4_full_new_3_to_48_4", "(0,3)->(48,4)", ((0, 3),), ((48, 4),)),
    TableCase("r4_full_new_6_to_33_4", "(0,6)->(33,4)", ((0, 6),), ((33, 4),)),
)
CASE_BY_NAME = {case.name: case for case in TABLE_CASES}
CASE_INDEX = {case.name: index for index, case in enumerate(TABLE_CASES)}


def _signed_power(value: float) -> str:
    if value == 0.0:
        return "0"
    sign = "+" if value > 0.0 else "-"
    return f"{sign}2^-{-math.log2(abs(value)):.9f}"


def _full_round_permutation(state: np.ndarray, rounds: int) -> np.ndarray:
    result = state
    for constant in common.ROUND_CONSTANTS[12 - rounds :]:
        result = xoodoo_experiment._theta(result)
        result = xoodoo_experiment._rho_west(result)
        result[:, 0, 0] ^= np.uint32(constant)
        result = xoodoo_experiment._chi(result)
        result = xoodoo_experiment._rho_east(result)
    return result


def _evaluate_full_assignment(
    boundary_record: Mapping[str, Any],
    system: xoodoo_experiment.ConstraintSystem,
    assignment: int,
    samples: int,
    seed: int,
    *,
    batch_size: int = 1 << 16,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    difference_value = common.state_from_columns(
        boundary_record["requested_input_columns"]
    )
    difference = xoodoo_experiment._state_to_lanes32(difference_value).reshape(
        1, 3, 4
    )
    output_value = common.state_from_columns(
        boundary_record["requested_output_columns"]
    )
    output_lanes = xoodoo_experiment._state_to_lanes32(output_value)
    rounds = int(boundary_record["rounds"])
    signed_sum = 0
    processed = 0
    while processed < samples:
        count = min(batch_size, samples - processed)
        left_input = xoodoo_experiment.sample_affine_class(
            rng, system, assignment, count
        )
        right_input = left_input ^ difference
        left = _full_round_permutation(left_input, rounds)
        right = _full_round_permutation(right_input, rounds)
        difference_lanes = (left ^ right).reshape(count, 12)
        parity = np.zeros(count, dtype=np.uint32)
        for lane, mask in enumerate(output_lanes):
            if mask:
                parity ^= xoodoo_experiment._parity32(
                    difference_lanes[:, lane] & mask
                )
        odd = int(np.count_nonzero(parity))
        signed_sum += count - 2 * odd
        processed += count
    correlation = signed_sum / samples
    standard_error = math.sqrt(
        max(0.0, 1.0 - correlation * correlation) / samples
    )
    return {
        "assignment_index": assignment,
        "assignment": [
            (assignment >> bit) & 1 for bit in range(len(system.rows))
        ],
        "samples": samples,
        "signed_sum": signed_sum,
        "correlation": correlation,
        "signed_power": _signed_power(correlation),
        "standard_error": standard_error,
        "confidence_95": [
            correlation - 1.959963984540054 * standard_error,
            correlation + 1.959963984540054 * standard_error,
        ],
        "seed": seed,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_prefixes(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value not in (0, 1) for value in values):
        raise argparse.ArgumentTypeError("prefixes must be a non-empty subset of 0,1")
    return values


def _parse_cases(text: str) -> tuple[str, ...]:
    if text.strip().lower() == "all":
        return tuple(case.name for case in TABLE_CASES)
    values = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = [value for value in values if value not in CASE_BY_NAME]
    if not values or unknown:
        raise argparse.ArgumentTypeError(f"unknown or empty case selection: {unknown}")
    return values


def _mask_words(item: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(int(word, 16) for word in item["mask_words"])


def _endpoint_selection(analysis: Mapping[str, Any]) -> dict[str, Any]:
    capacity = analysis["capacities"][-1]
    distribution = [float(value) for value in capacity["correlation_distribution"]]
    stable = [
        int(value)
        for value in analysis["highest_two_class_stability"][
            "stable_nonzero_class_indices"
        ]
    ]
    if stable:
        maximum = max(stable, key=lambda index: distribution[index])
        minimum = min(stable, key=lambda index: distribution[index])
        endpoints = {"maximum": maximum, "minimum": minimum}
        selection_rule = (
            "signed extrema among nonzero classes stable over the two highest "
            "recorded capacities"
        )
    elif int(analysis["selected_basis_dimension"]) == 0:
        endpoints = {"unconditioned_capacity_bounded": 0}
        selection_rule = (
            "no stable nonzero conditional class exists; retain the sole d=0 "
            "capacity-bounded value for diagnostic comparison only"
        )
    else:
        raise RuntimeError("no stable endpoint class is available")

    records = {}
    dimension = int(analysis["selected_basis_dimension"])
    for label, index in endpoints.items():
        value = distribution[index]
        records[label] = {
            "assignment_index": index,
            "assignment_lsb_first": [(index >> bit) & 1 for bit in range(dimension)],
            "assignment_display_msb_first": format(index, f"0{dimension}b") if dimension else "0",
            "theory_correlation": value,
            "theory_signed_power": _signed_power(value),
        }
    return {
        "selection_rule": selection_rule,
        "endpoints": records,
        "highest_capacity": {
            "suffix": int(capacity["suffix_capacity"]),
            "prefix": int(capacity["prefix_capacity"]),
            "aggregate": int(capacity["aggregate_capacity"]),
        },
        "joint_U0_coefficient": float(capacity["joint_U0_coefficient"]),
        "joint_U0_signed_power": capacity["joint_U0_signed_power"],
        "whole_distribution_stable_at_one_percent": bool(
            analysis["capacity_stable_at_one_percent"]
        ),
        "stable_nonzero_class_count": len(stable),
        "stable_zero_class_count": int(
            analysis["highest_two_class_stability"]["stable_zero_class_count"]
        ),
        "unstable_class_count": int(
            analysis["highest_two_class_stability"]["unstable_class_count"]
        ),
    }


def _new_document(samples_log2: int) -> dict[str, Any]:
    return {
        "version": 1,
        "purpose": "Xoodoo Appendix B Tables 12 and 13 model-selected endpoint experiments",
        "correlation_definition": "Pr[Z=0]-Pr[Z=1]",
        "function": (
            "real four-round Xoodoo permutation including theta, rho_west, iota, "
            "chi, rho_east in every round"
        ),
        "selection_policy": (
            "classes are selected only from theory before sampling; experiments "
            "do not select or tune endpoints"
        ),
        "samples_per_endpoint_log2": samples_log2,
        "batch_size_log2": None,
        "records": [],
    }


def _record_key(record: Mapping[str, Any]) -> tuple[str, int]:
    return str(record["case"]), int(record["ddt_prefix_rounds_of_theory"])


def _write_document(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(path, dict(document))


def run(
    case_names: Iterable[str],
    prefixes: Iterable[int],
    *,
    samples_log2: int,
    batch_log2: int,
    output_path: Path,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    if not 10 <= samples_log2 <= 34:
        raise ValueError("samples-log2 must lie in [10,34]")
    if not 10 <= batch_log2 <= min(samples_log2, 22):
        raise ValueError("batch-log2 must lie in [10,min(samples-log2,22)]")
    if output_path.is_file():
        document = json.loads(output_path.read_text(encoding="utf-8"))
        if int(document.get("samples_per_endpoint_log2", -1)) != samples_log2:
            if not force:
                raise ValueError(
                    "existing output uses a different sample count; pass --force "
                    "or choose a different output path"
                )
            document = _new_document(samples_log2)
    else:
        document = _new_document(samples_log2)
    previous_batch_log2 = document.get("batch_size_log2")
    if previous_batch_log2 not in (None, batch_log2) and not force:
        raise ValueError(
            "existing output uses a different batch size; pass --force or "
            "choose a different output path"
        )
    document["batch_size_log2"] = batch_log2
    completed = {_record_key(record): record for record in document["records"]}

    selected_names = set(case_names)
    ordered_names = [case.name for case in TABLE_CASES if case.name in selected_names]
    for case_name in ordered_names:
        case_index = CASE_INDEX[case_name]
        case = CASE_BY_NAME[case_name]
        config = get(case_name)
        boundary_record = {
            "requested_boundary": "full",
            "rounds": 4,
            "requested_input_columns": [list(item) for item in case.input_columns],
            "requested_output_columns": [list(item) for item in case.output_columns],
        }
        for prefix in prefixes:
            mode = "identity" if prefix == 0 else "complete"
            theory_path = THEORY_ROOT / case_name / f"p{prefix}_{mode}" / "distribution_analysis.json"
            if not theory_path.is_file():
                raise FileNotFoundError(theory_path)
            theory = json.loads(theory_path.read_text(encoding="utf-8"))
            selection = _endpoint_selection(theory)
            key = (case_name, prefix)
            theory_digest = _sha256(theory_path)
            cached = completed.get(key)
            if (
                not force
                and cached is not None
                and cached.get("theory_sha256") == theory_digest
                and int(cached.get("samples_per_endpoint_log2", -1)) == samples_log2
                and cached.get("status") == "complete"
                and set(cached.get("experiments", {}))
                == set(selection["endpoints"])
            ):
                print(f"[{case_name}] p={prefix}: cached", flush=True)
                continue

            basis_words = [_mask_words(item) for item in theory["reported_input_basis"]]
            system = xoodoo_experiment.make_constraint_system(basis_words)
            record: dict[str, Any] = {
                "case": case_name,
                "target": case.target,
                "ddt_prefix_rounds_of_theory": prefix,
                "theory_source": str(theory_path.resolve()),
                "theory_sha256": theory_digest,
                "basis_dimension": len(basis_words),
                "basis": theory["reported_input_basis"],
                "selection": selection,
                "samples_per_endpoint_log2": samples_log2,
                "samples_per_endpoint": 1 << samples_log2,
                "batch_size_log2": batch_log2,
                "experiments": {},
                "status": "dry-run" if dry_run else "running",
            }
            completed[key] = record
            document["records"] = [completed[item] for item in sorted(completed)]
            if not dry_run:
                _write_document(output_path, document)

            for endpoint_number, (label, endpoint) in enumerate(
                selection["endpoints"].items()
            ):
                assignment = int(endpoint["assignment_index"])
                seed = 2_026_091_000 + case_index * 10_000 + prefix * 1_000 + endpoint_number
                print(
                    f"[{case_name}] p={prefix} {label}: class={assignment}, "
                    f"N=2^{samples_log2}",
                    flush=True,
                )
                if dry_run:
                    continue
                observed = _evaluate_full_assignment(
                    boundary_record,
                    system,
                    assignment,
                    1 << samples_log2,
                    seed,
                    batch_size=1 << batch_log2,
                )
                theory_value = float(endpoint["theory_correlation"])
                observed_value = float(observed["correlation"])
                standard_error = float(observed["standard_error"])
                observed.update(
                    {
                        "signed_power": _signed_power(observed_value),
                        "theory_correlation": theory_value,
                        "theory_signed_power": endpoint["theory_signed_power"],
                        "experiment_minus_theory": observed_value - theory_value,
                        "standardized_residual": (
                            (observed_value - theory_value) / standard_error
                            if standard_error
                            else None
                        ),
                        "theory_inside_experiment_95_percent_interval": (
                            observed["confidence_95"][0]
                            <= theory_value
                            <= observed["confidence_95"][1]
                        ),
                    }
                )
                record["experiments"][label] = observed
                _write_document(output_path, document)
            record["status"] = "selected-only" if dry_run else "complete"
            if not dry_run:
                _write_document(output_path, document)

    document["records"] = [completed[item] for item in sorted(completed)]
    if not dry_run:
        _write_document(output_path, document)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=_parse_cases, default=tuple(case.name for case in TABLE_CASES))
    parser.add_argument("--prefixes", type=_parse_prefixes, default=(0, 1))
    parser.add_argument("--samples-log2", type=int, default=24)
    parser.add_argument("--batch-log2", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        args.cases,
        args.prefixes,
        samples_log2=args.samples_log2,
        batch_log2=args.batch_log2,
        output_path=args.output.resolve(),
        force=args.force,
        dry_run=args.dry_run,
    )
    print(f"records={len(result['records'])}; output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
