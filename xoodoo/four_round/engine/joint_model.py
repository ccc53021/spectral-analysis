"""Unified p=0/1/2 DDT-prefix plus joint-global-U pipeline.

The existing C engine evaluates a bounded joint-U spectrum for the remaining
round-based suffix.  This driver enumerates or samples fixed DDT-prefix
characteristics, pulls every suffix label V back with ``Q[delta,gamma]``, and
aggregates the resulting original-input labels U.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
import time
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from . import common
from . import scalar_model as scalar_compat
from . import c_engine as u_dag
from . import linear_layer as xoodoo_linear
from .config import DISTINGUISHERS, Distinguisher, get
from .prefix_characteristics import (
    CharacteristicBatch,
    PrefixCharacteristic,
    build_batch,
)
from .prefix_operator import CosetReducer, PrefixOperator, add_scaled, prune_polynomial


def _fraction(value: Fraction | None) -> dict | None:
    if value is None:
        return None
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def _state_words(value: int) -> list[str]:
    return common.mask_to_hex(common.int_to_words(value))


def _state_id(value: int) -> str:
    data = int(value).to_bytes(common.STATE_BITS // 8, "little")
    return hashlib.sha256(data).hexdigest()[:16]


def _term(mask: int, coefficient: float) -> dict:
    words = common.int_to_words(mask)
    return {
        "coefficient": coefficient,
        "absolute_coefficient": abs(coefficient),
        "weight": common.weight(coefficient),
        "mask_words": common.mask_to_hex(words),
        "expression": common.mask_to_expression(words),
    }


def _ordered_terms(spectrum: Mapping[int, float]) -> list[tuple[int, float]]:
    return sorted(
        ((int(mask), float(value)) for mask, value in spectrum.items() if value),
        key=lambda item: (
            -abs(item[1]),
            common.int_to_words(item[0]),
        ),
    )


def write_root_dump(path: Path, spectrum: Mapping[int, float]) -> None:
    """Write the same binary root format as the C suffix engine."""
    terms = sorted(
        ((int(mask), float(value)) for mask, value in spectrum.items() if value),
        key=lambda item: common.int_to_words(item[0]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"XUDRT001")
        stream.write(struct.pack("<Q", len(terms)))
        for mask, coefficient in terms:
            stream.write(
                struct.pack("<d6Q", coefficient, *common.int_to_words(mask))
            )


def _batch_record(batch: CharacteristicBatch) -> dict:
    weighted_probability = sum(
        (item.weighted_probability for item in batch.characteristics),
        Fraction(0, 1),
    )
    labels = {
        "identity": "complete 0DDT identity model",
        "complete": "complete deterministic DDT-prefix sum",
        "restricted_deterministic": "deterministic restricted partial DDT-prefix sum",
        "restricted_weighted_legacy": (
            "weighted replay of a legacy sampled restricted DDT-prefix set"
        ),
        "stratified_unbiased": "stratified unbiased statistical DDT-prefix estimator",
    }
    return {
        "prefix_rounds": batch.prefix_rounds,
        "mode": batch.mode,
        "total_complete_characteristics": batch.total_characteristics,
        "represented_characteristics": batch.represented_characteristics,
        "sampled_characteristic_records": batch.sampled_characteristics,
        "covered_probability": _fraction(batch.covered_probability),
        "weighted_probability_or_estimator_mass": _fraction(weighted_probability),
        "completeness_label": labels[batch.mode],
        "samples_per_first_family": batch.samples_per_first_family,
        "seed": batch.seed,
    }


def _configuration_record(
    config: Distinguisher, batch: CharacteristicBatch
) -> dict:
    return common.config_metadata(
        config,
        explicit_ddt_prefix_rounds=batch.prefix_rounds,
        prefix_mode=batch.mode,
    )


def _characteristic_record(characteristic: PrefixCharacteristic) -> dict:
    return {
        "characteristic_id": characteristic.characteristic_id,
        "first_family_id": characteristic.first_family_id,
        "prefix_rounds": characteristic.prefix_rounds,
        "delta_inputs": [_state_words(value) for value in characteristic.delta_inputs],
        "gamma_outputs": [_state_words(value) for value in characteristic.gamma_outputs],
        "boundary_difference": _state_words(characteristic.boundary_difference),
        "boundary_active_columns": xoodoo_linear.active_columns(
            characteristic.boundary_difference
        ),
        "probability": _fraction(characteristic.probability),
        "aggregation_weight": _fraction(characteristic.aggregation_weight),
        "weighted_probability": _fraction(characteristic.weighted_probability),
        "sampling_probability": _fraction(characteristic.sampling_probability),
    }


def _result_directory(
    config: Distinguisher,
    batch: CharacteristicBatch,
    coset_basis: Sequence[int] = (),
) -> Path:
    base = (
        common.OUTPUT_DIR
        / config.name
        / "joint_ddt_prefix"
        / f"p{batch.prefix_rounds}_{batch.mode}"
    )
    if batch.mode == "stratified_unbiased":
        base = base / (
            f"k{batch.samples_per_first_family}_seed{batch.seed}"
        )
    elif batch.mode in ("restricted_deterministic", "restricted_weighted_legacy"):
        digest = hashlib.sha256()
        for item in batch.characteristics:
            for value in item.gamma_outputs:
                digest.update(int(value).to_bytes(common.STATE_BITS // 8, "little"))
        base = base / f"set_{digest.hexdigest()[:16]}"
    if coset_basis:
        digest = hashlib.sha256()
        for value in coset_basis:
            digest.update(int(value).to_bytes(common.STATE_BITS // 8, "little"))
        base = base / f"coset_d{len(coset_basis)}_{digest.hexdigest()[:12]}"
    return base


def _dynamic_suffix_config(
    parent: Distinguisher,
    suffix_rounds: int,
    boundary_difference: int,
) -> Distinguisher:
    return Distinguisher(
        name=(
            f"{parent.name}_suffix{suffix_rounds}_"
            f"{_state_id(boundary_difference)}"
        ),
        rounds=suffix_rounds,
        input_columns=xoodoo_linear.column_entries(boundary_difference),
        output_column=parent.output_column,
        output_mask=parent.output_mask,
        capacities=parent.capacities,
        basis_min_dimension=parent.basis_min_dimension,
        basis_max_dimension=parent.basis_max_dimension,
        experiment_samples_per_class_log2=parent.experiment_samples_per_class_log2,
        experiment_best_samples_log2=parent.experiment_best_samples_log2,
    )


def _zero_suffix_spectrum(
    characteristic: PrefixCharacteristic,
    output_mask: int,
) -> dict[int, float]:
    if not characteristic.gamma_outputs:
        difference = characteristic.boundary_difference
    else:
        difference = characteristic.gamma_outputs[-1]
    sign = -1.0 if (difference & output_mask).bit_count() & 1 else 1.0
    return {0: sign}


def _evaluate_suffixes(
    config: Distinguisher,
    batch: CharacteristicBatch,
    suffix_capacity: int,
    directory: Path,
) -> tuple[dict[int, dict[int, float]], list[dict], float]:
    suffix_rounds = config.rounds - batch.prefix_rounds
    output_mask = common.words_to_int(common.output_mask(config))
    spectra: dict[int, dict[int, float]] = {}
    evaluations: list[dict] = []
    elapsed = 0.0
    suffix_cache_directory = (
        directory.parent if directory.name.startswith("coset_d") else directory
    )
    for boundary_difference in dict.fromkeys(
        item.boundary_difference for item in batch.characteristics
    ):
        if suffix_rounds == 0:
            # This case is characteristic-dependent through the last gamma,
            # so it is handled in the main aggregation loop.
            continue
        suffix = _dynamic_suffix_config(
            config, suffix_rounds, boundary_difference
        )
        suffix_id = _state_id(boundary_difference)
        dump_path = (
            suffix_cache_directory
            / "suffix_root_dumps"
            / f"hs{suffix_capacity}_{suffix_id}.bin"
        )
        evaluation_path = dump_path.with_suffix(".json")
        if dump_path.exists() and evaluation_path.exists():
            evaluation = common.read_json(evaluation_path)
            evaluation["reused_from_cache"] = True
        elif dump_path.exists():
            # Dumps written by an earlier implementation remain valid.  Their
            # coefficients can be reused even though detailed node counters
            # were not persisted at the time.
            evaluation = {
                "zero_label_coefficient": 0.0,
                "coefficient_status": "capacity-bounded-cached",
                "pruned_polynomial_count": 1,
                "maximum_unpruned_term_count": None,
                "seconds": 0.0,
                "reused_from_cache": True,
                "diagnostics_available": False,
            }
            common.write_json(evaluation_path, evaluation)
        else:
            evaluation = u_dag.evaluate(
                suffix,
                suffix_capacity,
                root_dump_path=dump_path,
            )
            evaluation["reused_from_cache"] = False
            evaluation["diagnostics_available"] = True
            common.write_json(evaluation_path, evaluation)
            elapsed += evaluation["seconds"]
        raw = common.read_root_dump(dump_path)
        if not evaluation.get("diagnostics_available", True):
            evaluation["zero_label_coefficient"] = raw.get(common.ZERO_MASK, 0.0)
            common.write_json(evaluation_path, evaluation)
        spectra[boundary_difference] = {
            common.words_to_int(mask): coefficient
            for mask, coefficient in raw.items()
        }
        evaluations.append(
            {
                "boundary_id": suffix_id,
                "boundary_difference": _state_words(boundary_difference),
                "boundary_active_columns": xoodoo_linear.active_columns(
                    boundary_difference
                ),
                "root_hash_terms": len(raw),
                "zero_label_coefficient": evaluation["zero_label_coefficient"],
                "coefficient_status": evaluation["coefficient_status"],
                "pruned_polynomial_count": evaluation[
                    "pruned_polynomial_count"
                ],
                "maximum_unpruned_term_count": evaluation[
                    "maximum_unpruned_term_count"
                ],
                "seconds": evaluation["seconds"],
                "reused_from_cache": evaluation["reused_from_cache"],
                "diagnostics_available": evaluation.get(
                    "diagnostics_available", True
                ),
                "root_dump_path": str(dump_path),
            }
        )
    return spectra, evaluations, elapsed


def _regression_record(
    expected: Mapping[int, float],
    observed: Mapping[int, float],
) -> dict:
    labels = set(expected) | set(observed)
    maximum_error = max(
        (abs(expected.get(mask, 0.0) - observed.get(mask, 0.0)) for mask in labels),
        default=0.0,
    )
    mismatched = sum(
        abs(expected.get(mask, 0.0) - observed.get(mask, 0.0)) > 1e-14
        for mask in labels
    )
    return {
        "expected_terms": len(expected),
        "observed_terms": len(observed),
        "union_terms": len(labels),
        "mismatched_terms_at_1e-14": mismatched,
        "maximum_absolute_error": maximum_error,
        "passed": mismatched == 0,
    }


def run_capacity(
    config: Distinguisher,
    batch: CharacteristicBatch,
    *,
    suffix_capacity: int,
    prefix_capacity: int,
    aggregate_capacity: int,
    directory: Path,
    coset_basis: Sequence[int] = (),
) -> dict:
    started = time.perf_counter()
    suffix_spectra, suffix_evaluations, suffix_seconds = _evaluate_suffixes(
        config, batch, suffix_capacity, directory
    )
    operator = PrefixOperator(
        config.rounds,
        prefix_capacity,
        coset_basis=coset_basis,
    )
    output_mask = common.words_to_int(common.output_mask(config))
    total: dict[int, float] = {}
    per_characteristic = []
    suffix_rounds = config.rounds - batch.prefix_rounds

    for characteristic in batch.characteristics:
        if suffix_rounds == 0:
            suffix_spectrum = _zero_suffix_spectrum(characteristic, output_mask)
        else:
            suffix_spectrum = suffix_spectra[characteristic.boundary_difference]
        contribution: dict[int, float] = {}
        for suffix_mask, suffix_coefficient in suffix_spectrum.items():
            prefix_polynomial = operator.pullback(characteristic, suffix_mask)
            add_scaled(
                contribution,
                prefix_polynomial,
                suffix_coefficient,
            )
        sampling_weight = float(characteristic.aggregation_weight)
        add_scaled(total, contribution, sampling_weight)
        per_characteristic.append(
            {
                "characteristic_id": characteristic.characteristic_id,
                "first_family_id": characteristic.first_family_id,
                "suffix_terms": len(suffix_spectrum),
                "composed_terms_before_root_aggregation": len(contribution),
                "zero_label_contribution_before_sampling_weight": contribution.get(
                    0, 0.0
                ),
                "sampling_aggregation_weight": sampling_weight,
            }
        )

    raw_root_path = directory / (
        f"joint_root_hs{suffix_capacity}_hp{prefix_capacity}_"
        f"ha{aggregate_capacity}.bin"
    )
    write_root_dump(raw_root_path, total)
    coset_reducer = CosetReducer(coset_basis)
    public = prune_polynomial(
        total,
        aggregate_capacity,
        coset_reducer=coset_reducer if coset_basis else None,
    )
    public_terms = [_term(mask, coefficient) for mask, coefficient in _ordered_terms(public)]
    nonzero_values = [abs(value) for mask, value in public.items() if mask and value]
    zero = total.get(0, 0.0)

    regression = None
    if batch.prefix_rounds == 0 and suffix_spectra:
        expected = next(iter(suffix_spectra.values()))
        regression = _regression_record(expected, total)

    coset_record = None
    if coset_basis:
        span = [0]
        for basis_mask in coset_basis:
            span += [mask ^ int(basis_mask) for mask in span]
        coefficients = [total.get(mask, 0.0) for mask in span]
        distribution = common.fwht(coefficients).tolist()
        nonzero_classes = [abs(value) for value in distribution if value != 0.0]
        maximum_index = max(
            range(len(distribution)), key=lambda index: abs(distribution[index])
        )
        coset_record = {
            "basis_dimension": len(coset_basis),
            "basis": [_state_words(value) for value in coset_basis],
            "span_size": len(span),
            "target_coverage_in_root": sum(mask in total for mask in span),
            "fourier_coefficients": coefficients,
            "target_terms": [
                {
                    **_term(mask, coefficient),
                    "present_in_root": mask in total,
                }
                for mask, coefficient in zip(span, coefficients)
            ],
            "correlation_distribution": distribution,
            "zero_class_count_exact_float": sum(value == 0.0 for value in distribution),
            "nonzero_class_count_exact_float": len(nonzero_classes),
            "minimum_nonzero_absolute_correlation": (
                min(nonzero_classes) if nonzero_classes else None
            ),
            "minimum_nonzero_weight": (
                common.weight(min(nonzero_classes)) if nonzero_classes else None
            ),
            "maximum_nonzero_absolute_correlation": (
                max(nonzero_classes) if nonzero_classes else None
            ),
            "maximum_nonzero_weight": (
                common.weight(max(nonzero_classes)) if nonzero_classes else None
            ),
            "maximum_absolute_class_index": maximum_index,
            "maximum_absolute_assignment": [
                (maximum_index >> bit) & 1 for bit in range(len(coset_basis))
            ],
            "maximum_absolute_class_correlation": distribution[maximum_index],
            "maximum_absolute_class_weight": common.weight(
                distribution[maximum_index]
            ),
        }

    return {
        "suffix_capacity": suffix_capacity,
        "prefix_capacity": prefix_capacity,
        "aggregate_capacity": aggregate_capacity,
        "coefficient_status": (
            "capacity-bounded" if any(
                item["pruned_polynomial_count"]
                for item in suffix_evaluations
            ) or operator.diagnostics.pruning_events else "exact"
        ),
        "joint_zero_label_coefficient": zero,
        "joint_zero_label_weight": common.weight(zero),
        "root_term_count_before_aggregate_pruning": len(total),
        "root_dump_path": str(raw_root_path),
        "public_term_count": len(public),
        "public_terms": public_terms,
        "public_nonzero_label_absolute_range": (
            {
                "minimum": min(nonzero_values),
                "minimum_weight": common.weight(min(nonzero_values)),
                "maximum": max(nonzero_values),
                "maximum_weight": common.weight(max(nonzero_values)),
            }
            if nonzero_values
            else None
        ),
        "root_squared_l2_norm": math.fsum(value * value for value in total.values()),
        "suffix_unique_boundary_count": len(suffix_spectra),
        "suffix_evaluations": suffix_evaluations,
        "prefix_operator_diagnostics": operator.diagnostics.as_dict(),
        "per_characteristic_composition": per_characteristic,
        "coset_span": coset_record,
        "p0_identity_regression": regression,
        "suffix_engine_seconds": suffix_seconds,
        "total_seconds": time.perf_counter() - started,
    }


def run_sweep(
    config: Distinguisher,
    prefix_rounds: int,
    *,
    mode: str | None = None,
    capacities: Iterable[int] = (4, 8, 16, 32, 64, 128),
    prefix_capacities: Sequence[int] | None = None,
    aggregate_capacities: Sequence[int] | None = None,
    max_characteristics: int = 1_000_000,
    samples_per_first_family: int = 1,
    seed: int = 20260815,
    restricted_gamma_pairs: Sequence[
        tuple[int, int] | tuple[int, int, Fraction]
    ] = (),
    coset_basis: Sequence[int] = (),
    force: bool = False,
) -> dict:
    capacities = tuple(int(value) for value in capacities)
    if not capacities or any(value <= 0 for value in capacities):
        raise ValueError("capacities must be positive")
    prefix_capacities = tuple(prefix_capacities or capacities)
    aggregate_capacities = tuple(aggregate_capacities or capacities)
    if not (
        len(capacities)
        == len(prefix_capacities)
        == len(aggregate_capacities)
    ):
        raise ValueError("H_S, H_P and H_A sweeps must have the same length")
    coset_basis = tuple(int(value) for value in coset_basis)
    reducer = CosetReducer(coset_basis)
    if reducer.dimension and any(
        value < (1 << reducer.dimension)
        for value in (*prefix_capacities, *aggregate_capacities)
    ):
        raise ValueError("H_P and H_A must cover the complete coset span")

    input_difference = common.words_to_int(common.input_difference(config))
    output_mask = common.words_to_int(common.output_mask(config))
    batch = build_batch(
        input_difference,
        config.rounds,
        prefix_rounds,
        mode=mode,
        max_characteristics=max_characteristics,
        samples_per_first_family=samples_per_first_family,
        seed=seed,
        restricted_gamma_pairs=restricted_gamma_pairs,
    )
    directory = _result_directory(config, batch, coset_basis)
    directory.mkdir(parents=True, exist_ok=True)
    common.write_json(
        directory / "characteristics.json",
        {
            "version": 1,
            "configuration": _configuration_record(config, batch),
            "batch": _batch_record(batch),
            "characteristics": [
                _characteristic_record(item) for item in batch.characteristics
            ],
        },
    )
    scalar = scalar_compat.evaluate_batch(batch, output_mask, config.rounds)
    output_path = directory / "capacity_sweep.json"
    previous = common.read_json(output_path) if output_path.exists() else None
    by_capacities = {
        (
            item["suffix_capacity"],
            item["prefix_capacity"],
            item["aggregate_capacity"],
        ): item
        for item in (previous["capacities"] if previous else [])
    }
    for suffix_capacity, prefix_capacity, aggregate_capacity in zip(
        capacities, prefix_capacities, aggregate_capacities
    ):
        capacity_key = (
            suffix_capacity,
            prefix_capacity,
            aggregate_capacity,
        )
        cached = by_capacities.get(capacity_key)
        if (
            not force
            and cached is not None
            and Path(cached.get("root_dump_path", "")).is_file()
        ):
            print(
                f"[{config.name}] p={prefix_rounds} {batch.mode}: "
                f"H_S={suffix_capacity}, H_P={prefix_capacity}, "
                f"H_A={aggregate_capacity} (cached)",
                flush=True,
            )
            continue
        print(
            f"[{config.name}] p={prefix_rounds} {batch.mode}: "
            f"H_S={suffix_capacity}, H_P={prefix_capacity}, "
            f"H_A={aggregate_capacity}",
            flush=True,
        )
        record = run_capacity(
            config,
            batch,
            suffix_capacity=suffix_capacity,
            prefix_capacity=prefix_capacity,
            aggregate_capacity=aggregate_capacity,
            directory=directory,
            coset_basis=coset_basis,
        )
        by_capacities[capacity_key] = record
        records = [by_capacities[key] for key in sorted(by_capacities)]
        common.write_json(
            output_path,
            {
                "version": 1,
                "method": (
                    "fixed DDT characteristics with local Fourier-resolved "
                    "Q[delta,gamma,u,v]; suffix joint-V DAG; affine constant "
                    "signs; global original-input U aggregation"
                ),
                "configuration": _configuration_record(config, batch),
                "batch": _batch_record(batch),
                "old_zero_value_mask_scalar": scalar_compat.fraction_record(scalar),
                "probability_accounting": (
                    "Q already contains each characteristic probability; "
                    "only aggregation_weight is applied outside Q"
                ),
                "coset_basis": [_state_words(value) for value in coset_basis],
                "capacities": records,
            },
        )
        print(
            "  U=0={:+.12g}, root={}, prefix-pruned={}, time={:.3f}s".format(
                record["joint_zero_label_coefficient"],
                record["root_term_count_before_aggregate_pruning"],
                record["prefix_operator_diagnostics"]["pruning_events"],
                record["total_seconds"],
            ),
            flush=True,
        )
    return common.read_json(output_path)


def _parse_capacities(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value.strip())


def _parse_state(value) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    if isinstance(value, list) and len(value) == common.WORDS64:
        return common.words_to_int(
            tuple(int(word, 0) if isinstance(word, str) else int(word) for word in value)
        )
    raise ValueError("a state must be an integer or six little-endian words")


def _parse_fraction(value) -> Fraction:
    if isinstance(value, dict):
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        return Fraction(value)
    raise ValueError("aggregation_weight must be an exact fraction")


def load_restricted_gamma_pairs(
    path: Path,
) -> tuple[tuple[int, int] | tuple[int, int, Fraction], ...]:
    data = common.read_json(Path(path))
    entries = (
        data.get("joint_characteristics", data.get("characteristics", []))
        if isinstance(data, dict)
        else data
    )
    result = []
    for item in entries:
        if "gamma_outputs" in item:
            values = tuple(_parse_state(value) for value in item["gamma_outputs"])
        else:
            values = (_parse_state(item["gamma0"]), _parse_state(item["gamma1"]))
        if len(values) != 2:
            raise ValueError("every restricted characteristic needs gamma0 and gamma1")
        if "aggregation_weight" in item:
            result.append(
                (
                    values[0],
                    values[1],
                    _parse_fraction(item["aggregation_weight"]),
                )
            )
        else:
            result.append((values[0], values[1]))
    if not result:
        raise ValueError("restricted characteristic file is empty")
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("distinguisher", choices=tuple(DISTINGUISHERS))
    parser.add_argument("prefix_rounds", type=int, choices=(0, 1, 2))
    parser.add_argument(
        "--mode",
        choices=("complete", "stratified_unbiased", "restricted_deterministic"),
    )
    parser.add_argument("--capacities", default="4,8,16,32,64,128")
    parser.add_argument("--prefix-capacities")
    parser.add_argument("--aggregate-capacities")
    parser.add_argument("--max-characteristics", type=int, default=1_000_000)
    parser.add_argument("--samples-per-first-family", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--restricted-characteristics", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    capacities = _parse_capacities(args.capacities)
    run_sweep(
        get(args.distinguisher),
        args.prefix_rounds,
        mode=args.mode,
        capacities=capacities,
        prefix_capacities=(
            _parse_capacities(args.prefix_capacities)
            if args.prefix_capacities
            else None
        ),
        aggregate_capacities=(
            _parse_capacities(args.aggregate_capacities)
            if args.aggregate_capacities
            else None
        ),
        max_characteristics=args.max_characteristics,
        samples_per_first_family=args.samples_per_first_family,
        seed=args.seed,
        restricted_gamma_pairs=(
            load_restricted_gamma_pairs(args.restricted_characteristics)
            if args.restricted_characteristics
            else ()
        ),
        force=args.force,
    )


if __name__ == "__main__":
    main()
