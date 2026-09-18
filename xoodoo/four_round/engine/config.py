"""Configurations for the Xoodoo non-zero input-value-mask analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Distinguisher:
    name: str
    rounds: int
    input_columns: tuple[tuple[int, int], ...]
    output_column: int
    output_mask: int
    capacities: tuple[int, ...] = (4, 8, 16, 32, 64, 128)
    basis_min_dimension: int = 4
    basis_max_dimension: int = 6
    experiment_samples_per_class_log2: int = 16
    experiment_best_samples_log2: int = 20


R4_CHI = Distinguisher(
    name="r4_chi_single",
    rounds=4,
    input_columns=((0, 4),),
    output_column=12,
    output_mask=4,
    experiment_samples_per_class_log2=16,
    experiment_best_samples_log2=20,
)

R4_CHI_ZERO_SCAN_BEST = Distinguisher(
    name="r4_chi_zero_scan_best",
    rounds=4,
    input_columns=((0, 2),),
    output_column=92,
    output_mask=4,
    basis_min_dimension=1,
    experiment_samples_per_class_log2=16,
    experiment_best_samples_log2=20,
)

R5_CHI = Distinguisher(
    name="r5_chi_double",
    rounds=5,
    input_columns=((0, 7), (72, 7)),
    output_column=72,
    output_mask=5,
    experiment_samples_per_class_log2=18,
    experiment_best_samples_log2=22,
)

# Published/previously verified calibration boundaries.  Full-round input and
# output layers are converted to their equivalent chi-to-chi boundaries here;
# ``calibration.py`` records the full-boundary view and pulls the
# resulting Fourier labels back when required.
R4_PUBLIC_FULL_STANDARD = Distinguisher(
    name="r4_public_full_standard",
    rounds=4,
    input_columns=((11, 4), (32, 2)),
    output_column=14,
    output_mask=2,
)

R4_PUBLIC_FULL_EXTENSION_CORE = Distinguisher(
    name="r4_public_full_extension_core",
    rounds=4,
    input_columns=((0, 1), (11, 4)),
    output_column=0,
    output_mask=1,
)

# Equivalent chi-to-chi boundaries of the additional complete-four-round
# single-column distinguishers retained in the v3 comparison table.  The
# original complete-round boundaries and the label pull-back through the first
# linear layer are recorded by ``calibration.py``.
R4_FULL_NEW_5_TO_33_2 = Distinguisher(
    name="r4_full_new_5_to_33_2",
    rounds=4,
    input_columns=((0, 1), (11, 4)),
    output_column=32,
    output_mask=2,
    basis_min_dimension=1,
)

R4_FULL_NEW_5_TO_83_4 = Distinguisher(
    name="r4_full_new_5_to_83_4",
    rounds=4,
    input_columns=((0, 1), (11, 4)),
    output_column=11,
    output_mask=4,
    basis_min_dimension=1,
)

R4_FULL_NEW_3_TO_37_4 = Distinguisher(
    name="r4_full_new_3_to_37_4",
    rounds=4,
    input_columns=((0, 1), (32, 2)),
    output_column=125,
    output_mask=4,
    basis_min_dimension=1,
)

R4_FULL_NEW_3_TO_48_4 = Distinguisher(
    name="r4_full_new_3_to_48_4",
    rounds=4,
    input_columns=((0, 1), (32, 2)),
    output_column=104,
    output_mask=4,
    basis_min_dimension=1,
)

R4_FULL_NEW_6_TO_33_4 = Distinguisher(
    name="r4_full_new_6_to_33_4",
    rounds=4,
    input_columns=((11, 4), (32, 2)),
    output_column=121,
    output_mask=4,
    basis_min_dimension=1,
)

R5_PUBLIC_TWO_COLUMN = Distinguisher(
    name="r5_public_two_column",
    rounds=5,
    input_columns=((0, 1), (88, 4)),
    output_column=0,
    output_mask=1,
)

R5_NEW_TWO_COLUMN_83 = Distinguisher(
    name="r5_new_two_column_83",
    rounds=5,
    input_columns=((0, 7), (72, 7)),
    output_column=83,
    output_mask=5,
    experiment_samples_per_class_log2=18,
    experiment_best_samples_log2=22,
)

DISTINGUISHERS = {
    R4_CHI.name: R4_CHI,
    R4_CHI_ZERO_SCAN_BEST.name: R4_CHI_ZERO_SCAN_BEST,
    R5_CHI.name: R5_CHI,
    R4_PUBLIC_FULL_STANDARD.name: R4_PUBLIC_FULL_STANDARD,
    R4_PUBLIC_FULL_EXTENSION_CORE.name: R4_PUBLIC_FULL_EXTENSION_CORE,
    R4_FULL_NEW_5_TO_33_2.name: R4_FULL_NEW_5_TO_33_2,
    R4_FULL_NEW_5_TO_83_4.name: R4_FULL_NEW_5_TO_83_4,
    R4_FULL_NEW_3_TO_37_4.name: R4_FULL_NEW_3_TO_37_4,
    R4_FULL_NEW_3_TO_48_4.name: R4_FULL_NEW_3_TO_48_4,
    R4_FULL_NEW_6_TO_33_4.name: R4_FULL_NEW_6_TO_33_4,
    R5_PUBLIC_TWO_COLUMN.name: R5_PUBLIC_TWO_COLUMN,
    R5_NEW_TWO_COLUMN_83.name: R5_NEW_TWO_COLUMN_83,
}


def get(name: str) -> Distinguisher:
    try:
        return DISTINGUISHERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown distinguisher {name!r}") from exc
