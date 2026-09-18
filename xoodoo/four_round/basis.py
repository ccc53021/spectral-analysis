"""Stable-U selection and original-input coset completion for joint DDT runs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

from engine import common
from engine.config import DISTINGUISHERS, Distinguisher, get
from engine.joint_model import load_restricted_gamma_pairs, run_sweep


def _mask_int(words: Sequence[str]) -> int:
    return common.words_to_int(tuple(int(word, 16) for word in words))


def _rank_int(values: Iterable[int]) -> int:
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


def _selected_records(sweep: dict, axis: str) -> list[dict]:
    records = list(sweep["capacities"])
    if axis == "diagonal":
        records = [
            item
            for item in records
            if item["suffix_capacity"]
            == item["prefix_capacity"]
            == item["aggregate_capacity"]
        ]
        key = lambda item: item["suffix_capacity"]
    elif axis == "suffix":
        maximum_prefix = max(item["prefix_capacity"] for item in records)
        maximum_aggregate = max(item["aggregate_capacity"] for item in records)
        records = [
            item
            for item in records
            if item["prefix_capacity"] == maximum_prefix
            and item["aggregate_capacity"] == maximum_aggregate
        ]
        key = lambda item: item["suffix_capacity"]
    elif axis == "prefix":
        maximum_suffix = max(item["suffix_capacity"] for item in records)
        maximum_aggregate = max(item["aggregate_capacity"] for item in records)
        records = [
            item
            for item in records
            if item["suffix_capacity"] == maximum_suffix
            and item["aggregate_capacity"] == maximum_aggregate
        ]
        key = lambda item: item["prefix_capacity"]
    elif axis == "all":
        key = lambda item: (
            item["suffix_capacity"],
            item["prefix_capacity"],
            item["aggregate_capacity"],
        )
    else:
        raise ValueError("axis must be diagonal, suffix, prefix, or all")
    records.sort(key=key)
    if len(records) < 2:
        raise RuntimeError(f"only {len(records)} capacity points on {axis} axis")
    return records


def _point_label(item: dict) -> str:
    return "S{}-P{}-A{}".format(
        item["suffix_capacity"],
        item["prefix_capacity"],
        item["aggregate_capacity"],
    )


def select_stable_basis(
    sweep_path: Path,
    config: Distinguisher,
    *,
    axis: str = "diagonal",
    minimum_presence: int = 4,
    relative_drift_limit: float = 0.25,
    minimum_relative_strength: float = 0.0,
    basis_min_dimension: int | None = None,
    basis_max_dimension: int | None = None,
    strength_first: bool = False,
) -> dict:
    sweep_path = Path(sweep_path).resolve()
    sweep = common.read_json(sweep_path)
    records = _selected_records(sweep, axis)
    spectra = []
    for record in records:
        raw = common.read_root_dump(Path(record["root_dump_path"]))
        spectra.append(
            (
                _point_label(record),
                {
                    common.words_to_int(mask): coefficient
                    for mask, coefficient in raw.items()
                },
            )
        )

    observations: dict[int, list[tuple[str, float]]] = {}
    for label, spectrum in spectra:
        for mask, coefficient in spectrum.items():
            if mask and coefficient:
                observations.setdefault(mask, []).append((label, coefficient))
    high_labels = {label for label, _ in spectra[-2:]}
    high_reference_scale = max(
        (abs(value) for _, spectrum in spectra[-2:] for value in spectrum.values()),
        default=0.0,
    )
    required_presence = min(minimum_presence, len(spectra))
    candidates = []
    for mask, values in observations.items():
        labels = {label for label, _ in values}
        if len(values) < required_presence or not high_labels.issubset(labels):
            continue
        coefficients = [coefficient for _, coefficient in values]
        if len({coefficient > 0 for coefficient in coefficients}) != 1:
            continue
        by_label = dict(values)
        high = [by_label[label] for label, _ in spectra[-2:]]
        scale = max(abs(value) for value in high)
        drift = abs(high[-1] - high[-2]) / scale if scale else 0.0
        if drift > relative_drift_limit:
            continue
        minimum_absolute = min(abs(v) for v in coefficients)
        if minimum_absolute < minimum_relative_strength * high_reference_scale:
            continue
        words = common.int_to_words(mask)
        candidates.append(
            {
                "mask_words": common.mask_to_hex(words),
                "expression": common.mask_to_expression(words),
                "presence_count": len(values),
                "same_sign": True,
                "minimum_absolute_coefficient": minimum_absolute,
                "maximum_absolute_coefficient": max(abs(v) for v in coefficients),
                "highest_two_relative_drift": drift,
                "observations": [
                    {
                        "capacity_point": label,
                        "coefficient": value,
                        "weight": common.weight(value),
                    }
                    for label, value in values
                ],
                "_mask": mask,
            }
        )
    if strength_first:
        candidates.sort(
            key=lambda item: (
                -item["minimum_absolute_coefficient"],
                item["highest_two_relative_drift"],
                -item["presence_count"],
                item["_mask"],
            )
        )
    else:
        candidates.sort(
            key=lambda item: (
                -item["presence_count"],
                -item["minimum_absolute_coefficient"],
                item["highest_two_relative_drift"],
                item["_mask"],
            )
        )

    requested_minimum = (
        config.basis_min_dimension
        if basis_min_dimension is None
        else int(basis_min_dimension)
    )
    requested_maximum = (
        config.basis_max_dimension
        if basis_max_dimension is None
        else int(basis_max_dimension)
    )
    if not 0 <= requested_minimum <= requested_maximum <= 12:
        raise ValueError("basis dimensions must satisfy 0 <= min <= max <= 12")

    basis = []
    basis_masks: list[int] = []
    for item in candidates:
        mask = item["_mask"]
        if _rank_int([*basis_masks, mask]) == len(basis_masks):
            continue
        selected = {key: value for key, value in item.items() if key != "_mask"}
        selected["index"] = len(basis)
        basis.append(selected)
        basis_masks.append(mask)
        if len(basis) == requested_maximum:
            break
    if len(basis) < requested_minimum:
        raise RuntimeError(
            f"stable candidate rank {len(basis)} is below requested minimum "
            f"{requested_minimum}"
        )
    result = {
        "version": 1,
        "configuration": sweep["configuration"],
        "batch": sweep["batch"],
        "source_capacity_sweep": str(sweep_path),
        "capacity_axis": axis,
        "capacity_points": [label for label, _ in spectra],
        "minimum_presence": required_presence,
        "relative_drift_limit": relative_drift_limit,
        "minimum_relative_strength": minimum_relative_strength,
        "high_capacity_reference_scale": high_reference_scale,
        "strength_first": strength_first,
        "requested_basis_min_dimension": requested_minimum,
        "requested_basis_max_dimension": requested_maximum,
        "stable_candidate_count": len(candidates),
        "stable_candidate_rank": _rank_int(item["_mask"] for item in candidates),
        "selected_basis_dimension": len(basis),
        "selected_basis": basis,
        "span_size": 1 << len(basis),
        "stable_candidates": [
            {key: value for key, value in item.items() if key != "_mask"}
            for item in candidates
        ],
    }
    output_path = sweep_path.parent / f"stable_basis_{axis}.json"
    common.write_json(output_path, result)
    return result


def run_coset_from_basis(
    basis_path: Path,
    config: Distinguisher,
    *,
    suffix_capacities: Sequence[int],
    prefix_capacities: Sequence[int],
    aggregate_capacities: Sequence[int],
) -> dict:
    basis_path = Path(basis_path).resolve()
    basis_record = common.read_json(basis_path)
    basis = tuple(_mask_int(item["mask_words"]) for item in basis_record["selected_basis"])
    sweep = common.read_json(Path(basis_record["source_capacity_sweep"]))
    batch = sweep["batch"]
    prefix_rounds = int(batch["prefix_rounds"])
    mode = batch["mode"]
    gamma_pairs = ()
    if mode in ("restricted_deterministic", "restricted_weighted_legacy"):
        gamma_pairs = load_restricted_gamma_pairs(
            Path(basis_record["source_capacity_sweep"]).parent
            / "characteristics.json"
        )
        # ``build_batch`` selects the weighted replay label from the explicit
        # per-characteristic weights; its public input mode remains the
        # restricted deterministic loader.
        mode = "restricted_deterministic"
    return run_sweep(
        config,
        prefix_rounds,
        mode=mode,
        capacities=suffix_capacities,
        prefix_capacities=prefix_capacities,
        aggregate_capacities=aggregate_capacities,
        samples_per_first_family=batch.get("samples_per_first_family") or 1,
        seed=batch.get("seed") or 20260815,
        restricted_gamma_pairs=gamma_pairs,
        coset_basis=basis,
    )


def _parse_capacities(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("distinguisher", choices=tuple(DISTINGUISHERS))
    parser.add_argument("stage", choices=("select", "coset"))
    parser.add_argument("record", type=Path)
    parser.add_argument("--axis", default="diagonal")
    parser.add_argument("--suffix-capacities", default="64,128")
    parser.add_argument("--prefix-capacities", default="64,128")
    parser.add_argument("--aggregate-capacities", default="64,128")
    parser.add_argument("--minimum-presence", type=int, default=4)
    parser.add_argument("--relative-drift-limit", type=float, default=0.25)
    parser.add_argument("--minimum-relative-strength", type=float, default=0.0)
    parser.add_argument("--basis-min-dimension", type=int)
    parser.add_argument("--basis-max-dimension", type=int)
    parser.add_argument("--strength-first", action="store_true")
    args = parser.parse_args()
    config = get(args.distinguisher)
    if args.stage == "select":
        result = select_stable_basis(
            args.record,
            config,
            axis=args.axis,
            minimum_presence=args.minimum_presence,
            relative_drift_limit=args.relative_drift_limit,
            minimum_relative_strength=args.minimum_relative_strength,
            basis_min_dimension=args.basis_min_dimension,
            basis_max_dimension=args.basis_max_dimension,
            strength_first=args.strength_first,
        )
        print(
            f"stable={result['stable_candidate_count']}, "
            f"rank={result['stable_candidate_rank']}, "
            f"basis={result['selected_basis_dimension']}"
        )
    else:
        result = run_coset_from_basis(
            args.record,
            config,
            suffix_capacities=_parse_capacities(args.suffix_capacities),
            prefix_capacities=_parse_capacities(args.prefix_capacities),
            aggregate_capacities=_parse_capacities(args.aggregate_capacities),
        )
        for item in result["capacities"]:
            if item["coset_span"] is None:
                continue
            span = item["coset_span"]
            print(
                "S{} P{} A{}: coverage={}/{}, max={:+.12g}".format(
                    item["suffix_capacity"],
                    item["prefix_capacity"],
                    item["aggregate_capacity"],
                    span["target_coverage_in_root"],
                    span["span_size"],
                    span["maximum_absolute_class_correlation"],
                )
            )


if __name__ == "__main__":
    main()
