"""Run joint-global-U analysis on published and already verified boundaries.

This is the calibration entry point.  It deliberately separates two claims:

1. ``scalar_reference_reproduced`` checks the old zero-value-mask recurrence
   on exactly the known boundary and DDT prefix.
2. ``joint_U`` is the new calculation on that same boundary.  Its U=0 term is
   not required to equal the old scalar because non-zero intermediate value
   masks may aggregate back to the global label U=0.

Full-round boundary cases share a chi-to-chi core with their equivalent chi
description.  Their root labels are pulled back through the first linear layer
and receive the phase of the first round constant.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from engine import common
from engine import joint_model as joint_ddt_pipeline
from engine import linear_layer as xoodoo_linear
from engine.config import (
    R4_CHI,
    R4_CHI_ZERO_SCAN_BEST,
    R4_FULL_NEW_3_TO_37_4,
    R4_FULL_NEW_3_TO_48_4,
    R4_FULL_NEW_5_TO_33_2,
    R4_FULL_NEW_5_TO_83_4,
    R4_FULL_NEW_6_TO_33_4,
    R4_PUBLIC_FULL_EXTENSION_CORE,
    R4_PUBLIC_FULL_STANDARD,
    R5_CHI,
    R5_NEW_TWO_COLUMN_83,
    R5_PUBLIC_TWO_COLUMN,
    Distinguisher,
)


@dataclass(frozen=True)
class BoundaryView:
    name: str
    boundary: str
    input_description: str
    output_description: str
    pull_back_to_full_input: bool = False


@dataclass(frozen=True)
class KnownCase:
    name: str
    config: Distinguisher
    status: str
    source: str
    views: tuple[BoundaryView, ...]
    old_scalar: tuple[
        Fraction | float | None,
        Fraction | float | None,
        Fraction | float | None,
    ]
    experiment: Mapping[str, object]
    full_input_difference: int | None = None
    full_output_mask: int | None = None
    restricted_two_ddt: Mapping[str, object] | None = None


PUBLIC_R5_FULL_DIFFERENCE = xoodoo_linear.from_lanes(
    (
        0xA8B23B19,
        0x98810919,
        0x52674513,
        0x95A876F3,
        0xA8B23B18,
        0x98810919,
        0x52674513,
        0x95A876F3,
        0xA8B23B18,
        0x98810919,
        0x52676513,
        0x95A876F3,
    )
)


CASES = {
    item.name: item
    for item in (
        KnownCase(
            name="r4_public_full_standard",
            config=R4_PUBLIC_FULL_STANDARD,
            status="published",
            source="Dunkelman-Weizman four-round ordinary Xoodoo SDL",
            views=(
                BoundaryView(
                    "full_round",
                    "full-round input to full-round output",
                    "(col 0,difference 6)",
                    "(col 15,mask 2)",
                    True,
                ),
            ),
            old_scalar=(Fraction(0, 1), Fraction(5, 2048), 2.0 ** -8.740551),
            experiment={
                "published_correlation": "approximately +2^-8.7",
                "local_correlation": 2.0 ** -8.600722,
                "local_signed_power": "+2^-8.600722",
            },
            full_input_difference=common.state_from_columns(((0, 6),)),
            full_output_mask=common.state_from_columns(((15, 2),)),
            restricted_two_ddt={
                "complete_characteristics": "2^43.087504",
                "old_result": "+2^-8.740551",
                "manifest_status": "not yet recovered",
            },
        ),
        KnownCase(
            name="r4_public_full_extension_core",
            config=R4_PUBLIC_FULL_EXTENSION_CORE,
            status="published-related core",
            source="four-round core used by the published five-round extension",
            views=(
                BoundaryView(
                    "full_round",
                    "full-round input to full-round output",
                    "(col 0,difference 5)",
                    "(col 0,mask 1)",
                    True,
                ),
            ),
            old_scalar=(Fraction(0, 1), Fraction(-25, 1024), -(2.0 ** -4.359275)),
            experiment={
                "local_correlation": -(2.0 ** -4.344602),
                "local_signed_power": "-2^-4.344602",
            },
            full_input_difference=common.state_from_columns(((0, 5),)),
            full_output_mask=common.state_from_columns(((0, 1),)),
            restricted_two_ddt={
                "complete_characteristics": "2^61.590239",
                "old_result": "-2^-4.359275",
                "manifest_status": "not yet recovered",
            },
        ),
        KnownCase(
            name="r4_full_new_5_to_33_2",
            config=R4_FULL_NEW_5_TO_33_2,
            status="new verified distinguisher",
            source="v3 complete-four-round single-column scan and experiment",
            views=(
                BoundaryView(
                    "full_round",
                    "full-round input to full-round output",
                    "(col 0,difference 5)",
                    "(col 33,mask 2)",
                    True,
                ),
            ),
            old_scalar=(Fraction(0, 1), Fraction(25, 1024), 2.0 ** -5.368880),
            experiment={
                "local_correlation": 2.0 ** -5.346994,
                "local_signed_power": "+2^-5.346994",
            },
            full_input_difference=common.state_from_columns(((0, 5),)),
            full_output_mask=common.state_from_columns(((33, 2),)),
            restricted_two_ddt={
                "complete_characteristics": "12672421534367744 ~= 2^53.492542",
                "old_result": "+2^-5.368880",
                "manifest_status": "v3 family sampler must be replayed",
            },
        ),
        KnownCase(
            name="r4_full_new_5_to_83_4",
            config=R4_FULL_NEW_5_TO_83_4,
            status="new verified distinguisher",
            source="v3 complete-four-round single-column scan and experiment",
            views=(
                BoundaryView(
                    "full_round",
                    "full-round input to full-round output",
                    "(col 0,difference 5)",
                    "(col 83,mask 4)",
                    True,
                ),
            ),
            old_scalar=(Fraction(0, 1), Fraction(25, 1024), 2.0 ** -4.359275),
            experiment={
                "local_correlation": 2.0 ** -4.367885,
                "local_signed_power": "+2^-4.367885",
            },
            full_input_difference=common.state_from_columns(((0, 5),)),
            full_output_mask=common.state_from_columns(((83, 4),)),
            restricted_two_ddt={
                "complete_characteristics": "3471436935354908672 ~= 2^61.590239",
                "old_result": "+2^-4.359275",
                "manifest_status": "v3 family sampler must be replayed",
            },
        ),
        KnownCase(
            name="r4_full_new_3_to_37_4",
            config=R4_FULL_NEW_3_TO_37_4,
            status="new verified distinguisher",
            source="v3 complete-four-round single-column scan and experiment",
            views=(
                BoundaryView(
                    "full_round",
                    "full-round input to full-round output",
                    "(col 0,difference 3)",
                    "(col 37,mask 4)",
                    True,
                ),
            ),
            old_scalar=(
                Fraction(12375, 33554432),
                Fraction(5, 1024),
                2.0 ** -7.418917,
            ),
            experiment={
                "local_correlation": 2.0 ** -7.437042,
                "local_signed_power": "+2^-7.437042",
            },
            full_input_difference=common.state_from_columns(((0, 3),)),
            full_output_mask=common.state_from_columns(((37, 4),)),
            restricted_two_ddt={
                "complete_characteristics": "22808819230834688 ~= 2^54.340441",
                "old_result": "+2^-7.418917",
                "manifest_status": "v3 family sampler must be replayed",
            },
        ),
        KnownCase(
            name="r4_full_new_3_to_48_4",
            config=R4_FULL_NEW_3_TO_48_4,
            status="new verified distinguisher",
            source="v3 complete-four-round single-column scan and experiment",
            views=(
                BoundaryView(
                    "full_round",
                    "full-round input to full-round output",
                    "(col 0,difference 3)",
                    "(col 48,mask 4)",
                    True,
                ),
            ),
            old_scalar=(Fraction(0, 1), Fraction(0, 1), 2.0 ** -7.246992),
            experiment={
                "local_correlation": 2.0 ** -7.191639,
                "local_signed_power": "+2^-7.191639",
            },
            full_input_difference=common.state_from_columns(((0, 3),)),
            full_output_mask=common.state_from_columns(((48, 4),)),
            restricted_two_ddt={
                "complete_characteristics": "9346117271552 ~= 2^43.087504",
                "old_result": "+2^-7.246992",
                "manifest_status": "v3 family sampler must be replayed",
            },
        ),
        KnownCase(
            name="r4_full_new_6_to_33_4",
            config=R4_FULL_NEW_6_TO_33_4,
            status="new verified distinguisher",
            source="v3 complete-four-round single-column scan and experiment",
            views=(
                BoundaryView(
                    "full_round",
                    "full-round input to full-round output",
                    "(col 0,difference 6)",
                    "(col 33,mask 4)",
                    True,
                ),
            ),
            old_scalar=(
                Fraction(0, 1),
                Fraction(387, 262144),
                2.0 ** -7.402879,
            ),
            experiment={
                "local_correlation": 2.0 ** -7.427152,
                "local_signed_power": "+2^-7.427152",
            },
            full_input_difference=common.state_from_columns(((0, 6),)),
            full_output_mask=common.state_from_columns(((33, 4),)),
            restricted_two_ddt={
                "complete_characteristics": "22808819230834688 ~= 2^54.340441",
                "old_result": "+2^-7.402879",
                "manifest_status": "v3 family sampler must be replayed",
            },
        ),
        KnownCase(
            name="r4_chi_zero_scan_best",
            config=R4_CHI_ZERO_SCAN_BEST,
            status="new verified chi-boundary distinguisher",
            source="v3 four-chi single-column 0DDT optimum and experiment",
            views=(
                BoundaryView(
                    "chi_boundary",
                    "first chi input to fourth chi output",
                    "(col 0,difference 2)",
                    "(col 92,mask 4)",
                ),
            ),
            old_scalar=(Fraction(1, 32), Fraction(1, 8), None),
            experiment={
                "local_correlation": 2.0 ** -2.812555,
                "local_signed_power": "+2^-2.812555",
            },
        ),
        KnownCase(
            name="r4_chi_one_scan_best",
            config=R4_CHI,
            status="new verified chi-boundary distinguisher",
            source="v3 four-chi single-column complete-1DDT optimum and experiment",
            views=(
                BoundaryView(
                    "chi_boundary",
                    "first chi input to fourth chi output",
                    "(col 0,difference 4)",
                    "(col 12,mask 4)",
                ),
            ),
            old_scalar=(Fraction(1, 128), Fraction(13, 64), None),
            experiment={
                "chi_boundary_local_correlation": 2.0 ** -2.300845,
                "chi_boundary_local_signed_power": "+2^-2.300845",
                "equivalent_full_local_correlation": 2.0 ** -2.300215,
                "equivalent_full_local_signed_power": "+2^-2.300215",
            },
        ),
        KnownCase(
            name="r5_public_sdl",
            config=R5_PUBLIC_TWO_COLUMN,
            status="published",
            source="published five-round Xoodoo SDL and its equivalent chi boundary",
            views=(
                BoundaryView(
                    "full_round",
                    "full-round input to full-round output",
                    (
                        "dense lanes A0=a8b23b19 98810919 52674513 95a876f3; "
                        "A1=a8b23b18 98810919 52674513 95a876f3; "
                        "A2=a8b23b18 98810919 52676513 95a876f3"
                    ),
                    "(col 0,mask 1)",
                    True,
                ),
                BoundaryView(
                    "equivalent_two_column_chi",
                    "first chi input to fifth chi output",
                    "(col 0,difference 1),(col 88,difference 4)",
                    "(col 0,mask 1)",
                ),
            ),
            old_scalar=(
                Fraction(0, 1),
                Fraction(0, 1),
                Fraction(-12795, 8388608),
            ),
            experiment={
                "published_theory": "approximately -2^-8.36",
                "full_round_local_correlation": -(2.0 ** -8.504457),
                "full_round_local_signed_power": "-2^-8.504457",
                "equivalent_chi_local_correlation": -(2.0 ** -8.347491),
                "equivalent_chi_local_signed_power": "-2^-8.347491",
            },
            full_input_difference=PUBLIC_R5_FULL_DIFFERENCE,
            full_output_mask=common.state_from_columns(((0, 1),)),
            restricted_two_ddt={
                "legacy_seed": 20260815,
                "samples_per_first_characteristic": 8192,
                "sampled_complete_characteristics": 131072,
                "old_nonzero_suffix_characteristics": 2061,
                "represented_complete_characteristics": "1153203005354410000",
                "old_result": "-2^-9.356707473",
                "manifest_status": "legacy NumPy stream is being recovered exactly",
            },
        ),
        KnownCase(
            name="r5_new_two_column_83",
            config=R5_NEW_TWO_COLUMN_83,
            status="new verified distinguisher",
            source="v3 restricted-2DDT scan followed by a real five-chi experiment",
            views=(
                BoundaryView(
                    "two_column_chi",
                    "first chi input to fifth chi output",
                    "(col 0,difference 7),(col 72,difference 7)",
                    "(col 83,mask 5)",
                ),
            ),
            old_scalar=(
                Fraction(0, 1),
                Fraction(0, 1),
                Fraction(6623, 4194304),
            ),
            experiment={
                "local_correlation": 2.0 ** -8.455337,
                "local_signed_power": "+2^-8.455337",
            },
            restricted_two_ddt={
                "legacy_seed": 20260815,
                "samples_per_first_characteristic": 8192,
                "sampled_complete_characteristics": 131072,
                "old_nonzero_suffix_characteristics": 2065,
                "represented_complete_characteristics": "1441151891495976976",
                "old_result": "+2^-9.306730857",
                "manifest_status": "legacy NumPy stream is being recovered exactly",
            },
        ),
        KnownCase(
            name="r5_new_two_column_72",
            config=R5_CHI,
            status="new verified distinguisher",
            source="same v3 restricted-2DDT family; strongest local experiment",
            views=(
                BoundaryView(
                    "two_column_chi",
                    "first chi input to fifth chi output",
                    "(col 0,difference 7),(col 72,difference 7)",
                    "(col 72,mask 5)",
                ),
            ),
            old_scalar=(
                Fraction(0, 1),
                Fraction(0, 1),
                Fraction(-6623, 4194304),
            ),
            experiment={
                "local_correlation": -(2.0 ** -8.219922),
                "local_signed_power": "-2^-8.219922",
            },
            restricted_two_ddt={
                "legacy_seed": 20260815,
                "samples_per_first_characteristic": 8192,
                "sampled_complete_characteristics": 131072,
                "old_nonzero_suffix_characteristics": 2065,
                "represented_complete_characteristics": "1441151891495976976",
                "old_result": "-2^-9.306730857",
                "manifest_status": "legacy NumPy stream is being recovered exactly",
            },
        ),
    )
}

LEGACY_P2_MANIFESTS = {
    name: (
        common.OUTPUT_DIR
        / "known_distinguisher_joint"
        / "legacy_v3_p2_manifests"
        / f"{name}.json"
    )
    for name in (
        "r5_public_sdl",
        "r5_new_two_column_72",
        "r5_new_two_column_83",
    )
}


def _signed_power(value: float) -> str:
    if value == 0.0:
        return "0"
    return ("+" if value > 0 else "-") + f"2^-{-math.log2(abs(value)):.9f}"


def _state_words(value: int) -> list[str]:
    return common.mask_to_hex(common.int_to_words(value))


def transform_chi_spectrum_to_full_input(
    spectrum: Mapping[Sequence[int], float],
    first_constant: int,
) -> dict[int, float]:
    constant = xoodoo_linear.constant_state(first_constant)
    result: dict[int, float] = {}
    for words, coefficient in spectrum.items():
        chi_mask = common.words_to_int(words)
        full_mask = xoodoo_linear.pre_chi_adjoint(chi_mask)
        sign = -1.0 if (chi_mask & constant).bit_count() & 1 else 1.0
        value = result.get(full_mask, 0.0) + sign * float(coefficient)
        if value:
            result[full_mask] = value
        else:
            result.pop(full_mask, None)
    return result


def _validate_boundary(case: KnownCase) -> dict:
    core_difference = common.words_to_int(common.input_difference(case.config))
    core_output = common.words_to_int(common.output_mask(case.config))
    if case.full_input_difference is None:
        return {
            "has_full_boundary_view": False,
            "difference_map_passed": None,
            "output_mask_map_passed": None,
        }
    if case.full_output_mask is None:
        raise ValueError(f"{case.name} has a full input but no full output mask")
    mapped_difference = xoodoo_linear.pre_chi_linear(case.full_input_difference)
    mapped_output = xoodoo_linear.rho_east_adjoint(case.full_output_mask)
    return {
        "has_full_boundary_view": True,
        "full_input_difference_words": _state_words(case.full_input_difference),
        "mapped_chi_input_difference_words": _state_words(mapped_difference),
        "expected_chi_input_difference_words": _state_words(core_difference),
        "difference_map_passed": mapped_difference == core_difference,
        "full_output_mask_words": _state_words(case.full_output_mask),
        "mapped_chi_output_mask_words": _state_words(mapped_output),
        "expected_chi_output_mask_words": _state_words(core_output),
        "output_mask_map_passed": mapped_output == core_output,
    }


def _top_terms(spectrum: Mapping[int, float], limit: int = 32) -> list[dict]:
    ordered = sorted(
        ((mask, value) for mask, value in spectrum.items() if value),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    return [
        {
            "coefficient": coefficient,
            "signed_power": _signed_power(coefficient),
            "weight": common.weight(coefficient),
            "mask_words": _state_words(mask),
            "expression": common.mask_to_expression(common.int_to_words(mask)),
        }
        for mask, coefficient in ordered[:limit]
    ]


def _root_summary(spectrum: Mapping[int, float], dump_path: Path) -> dict:
    zero = float(spectrum.get(0, 0.0))
    nonzero_labels = [abs(value) for mask, value in spectrum.items() if mask and value]
    return {
        "root_dump_path": str(dump_path),
        "root_term_count": len(spectrum),
        "joint_U0_coefficient": zero,
        "joint_U0_signed_power": _signed_power(zero),
        "nonzero_label_count": len(nonzero_labels),
        "nonzero_label_absolute_range": (
            {
                "minimum": min(nonzero_labels),
                "minimum_signed_power_weight": common.weight(min(nonzero_labels)),
                "maximum": max(nonzero_labels),
                "maximum_signed_power_weight": common.weight(max(nonzero_labels)),
            }
            if nonzero_labels
            else None
        ),
        "top_terms": _top_terms(spectrum),
    }


def _reference_record(case: KnownCase, prefix_rounds: int) -> dict:
    reference = case.old_scalar[prefix_rounds]
    value = float(reference) if reference is not None else None
    return {
        "coefficient": value,
        "signed_power": _signed_power(value) if value is not None else None,
        "source": "v3 old zero-value-mask round-based result",
    }


def write_global_summary() -> dict:
    base = common.OUTPUT_DIR / "known_distinguisher_joint"
    records = []
    for case in CASES.values():
        case_directory = base / case.name
        for calibration_path in sorted(case_directory.glob("p*/calibration.json")):
            document = common.read_json(calibration_path)
            if not document.get("capacities"):
                continue
            highest = max(
                document["capacities"], key=lambda item: item["suffix_capacity"]
            )
            records.append(
                {
                    "case": case.name,
                    "status": case.status,
                    "ddt_prefix_rounds": document["ddt_prefix_rounds"],
                    "prefix_mode": document["prefix_mode"],
                    "old_scalar": document["computed_old_scalar"],
                    "scalar_reference_reproduced": document[
                        "scalar_reference_reproduced"
                    ],
                    "highest_completed_capacity": highest["suffix_capacity"],
                    "coefficient_status": highest["coefficient_status"],
                    "views": [
                        {
                            "name": view["name"],
                            "boundary": view["boundary"],
                            "joint_U0_coefficient": view["joint_U0_coefficient"],
                            "joint_U0_signed_power": view["joint_U0_signed_power"],
                            "nonzero_label_count": view["nonzero_label_count"],
                            "nonzero_label_absolute_range": view[
                                "nonzero_label_absolute_range"
                            ],
                        }
                        for view in highest["views"]
                    ],
                    "experiment": document["experiment"],
                    "legacy_two_ddt_manifest": document.get(
                        "legacy_two_ddt_manifest"
                    ),
                    "calibration_path": str(calibration_path),
                }
            )
    records.sort(key=lambda item: (item["case"], item["ddt_prefix_rounds"]))
    result = {
        "version": 1,
        "ordering": (
            "published/verified boundaries first; exploratory sparse candidates second"
        ),
        "records": records,
    }
    common.write_json(base / "summary.json", result)
    return result


def run_case(
    case: KnownCase,
    prefix_rounds: int,
    capacities: Iterable[int],
) -> dict:
    if prefix_rounds not in (0, 1, 2):
        raise ValueError("prefix_rounds must be 0, 1, or 2")
    validation = _validate_boundary(case)
    if validation["has_full_boundary_view"] and not (
        validation["difference_map_passed"] and validation["output_mask_map_passed"]
    ):
        raise AssertionError(f"full-boundary map failed for {case.name}")

    manifest = None
    restricted_pairs = ()
    mode = None
    if prefix_rounds == 2:
        manifest_path = LEGACY_P2_MANIFESTS.get(case.name)
        if manifest_path is None or not manifest_path.is_file():
            raise ValueError(
                f"{case.name} has no exactly recovered legacy 2DDT manifest"
            )
        manifest = common.read_json(manifest_path)
        if not (
            manifest.get("correlation_reproduced")
            and manifest.get("feature_count_reproduced")
        ):
            raise ValueError("legacy 2DDT manifest has not passed both regressions")
        restricted_pairs = joint_ddt_pipeline.load_restricted_gamma_pairs(
            manifest_path
        )
        mode = "restricted_deterministic"
    result = joint_ddt_pipeline.run_sweep(
        case.config,
        prefix_rounds,
        mode=mode,
        capacities=capacities,
        restricted_gamma_pairs=restricted_pairs,
    )
    mode = result["batch"]["mode"]
    directory = (
        common.OUTPUT_DIR
        / "known_distinguisher_joint"
        / case.name
        / f"p{prefix_rounds}_{mode}"
    )
    directory.mkdir(parents=True, exist_ok=True)

    computed_scalar = float(result["old_zero_value_mask_scalar"]["coefficient"])
    stored_reference = case.old_scalar[prefix_rounds]
    stored_value = float(stored_reference) if stored_reference is not None else None
    scalar_passed = (
        stored_value is not None
        and math.isclose(computed_scalar, stored_value, rel_tol=0.0, abs_tol=1e-15)
    )
    capacity_records = []
    first_constant = common.ROUND_CONSTANTS[12 - case.config.rounds]
    for capacity in result["capacities"]:
        core_dump = Path(capacity["root_dump_path"])
        core_words = common.read_root_dump(core_dump)
        core_spectrum = {
            common.words_to_int(mask): coefficient
            for mask, coefficient in core_words.items()
        }
        views = []
        for view in case.views:
            if view.pull_back_to_full_input:
                spectrum = transform_chi_spectrum_to_full_input(
                    core_words, first_constant
                )
            else:
                spectrum = dict(core_spectrum)
            dump_path = directory / (
                f"{view.name}_root_hs{capacity['suffix_capacity']}_"
                f"hp{capacity['prefix_capacity']}_ha{capacity['aggregate_capacity']}.bin"
            )
            joint_ddt_pipeline.write_root_dump(dump_path, spectrum)
            views.append(
                {
                    "name": view.name,
                    "boundary": view.boundary,
                    "input": view.input_description,
                    "output": view.output_description,
                    "label_coordinates": (
                        "full permutation input; includes first-round-constant phase"
                        if view.pull_back_to_full_input
                        else "first chi input"
                    ),
                    **_root_summary(spectrum, dump_path),
                }
            )
        capacity_records.append(
            {
                "suffix_capacity": capacity["suffix_capacity"],
                "prefix_capacity": capacity["prefix_capacity"],
                "aggregate_capacity": capacity["aggregate_capacity"],
                "coefficient_status": capacity["coefficient_status"],
                "prefix_pruning_events": capacity["prefix_operator_diagnostics"][
                    "pruning_events"
                ],
                "source_core_root_dump_path": str(core_dump),
                "views": views,
            }
        )

    document = {
        "version": 1,
        "case": case.name,
        "status": case.status,
        "source": case.source,
        "rule": (
            "first reproduce the old scalar on the exact known boundary and "
            "DDT prefix; then compare the new joint-U spectrum on that same setup"
        ),
        "configuration": result["configuration"],
        "boundary_validation": validation,
        "ddt_prefix_rounds": prefix_rounds,
        "prefix_mode": mode,
        "old_scalar_reference": _reference_record(case, prefix_rounds),
        "computed_old_scalar": result["old_zero_value_mask_scalar"],
        "scalar_reference_reproduced": scalar_passed,
        "joint_U0_is_not_expected_to_equal_old_scalar": True,
        "experiment": dict(case.experiment),
        "restricted_two_ddt": dict(case.restricted_two_ddt or {}),
        "legacy_two_ddt_manifest": (
            {
                "path": str(LEGACY_P2_MANIFESTS[case.name]),
                "unique_dominant_complete_two_round_characteristics": manifest[
                    "unique_dominant_complete_two_round_characteristics"
                ],
                "sample_hit_count": manifest["sample_hit_count"],
                "sampled_complete_two_round_characteristics": manifest[
                    "sampled_complete_two_round_characteristics"
                ],
                "correlation_reproduced": manifest["correlation_reproduced"],
                "feature_count_reproduced": manifest["feature_count_reproduced"],
                "joint_U_limitation": manifest["joint_U_limitation"],
            }
            if manifest is not None
            else None
        ),
        "capacities": capacity_records,
    }
    output_path = directory / "calibration.json"
    common.write_json(output_path, document)
    write_global_summary()
    print(
        f"[{case.name}] p={prefix_rounds}: scalar reference "
        f"{'PASS' if scalar_passed else 'FAIL'}; saved {output_path}",
        flush=True,
    )
    return document


def _parse_capacities(text: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in text.split(",") if value.strip())
    if not result or any(value <= 0 for value in result):
        raise ValueError("capacities must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=(*CASES, "published", "all"))
    parser.add_argument("prefix_rounds", type=int, choices=(0, 1, 2))
    parser.add_argument("--capacities", default="4,8,16,32,64,128")
    args = parser.parse_args()
    if args.case == "published":
        selected = [item for item in CASES.values() if item.status.startswith("published")]
    elif args.case == "all":
        selected = list(CASES.values())
    else:
        selected = [CASES[args.case]]
    capacities = _parse_capacities(args.capacities)
    for case in selected:
        run_case(case, args.prefix_rounds, capacities)


if __name__ == "__main__":
    main()
