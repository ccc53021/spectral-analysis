"""Unified identity, complete and sampled DDT-prefix characteristics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from math import prod
from typing import Iterable, Iterator, Sequence

from . import common, qd_kernel
from . import linear_layer as xoodoo_linear


class CharacteristicLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class PrefixCharacteristic:
    characteristic_id: int
    prefix_rounds: int
    delta_inputs: tuple[int, ...]
    gamma_outputs: tuple[int, ...]
    boundary_difference: int
    probability: Fraction
    aggregation_weight: Fraction = Fraction(1, 1)
    first_family_id: int = 0
    sampling_probability: Fraction | None = None

    @property
    def weighted_probability(self) -> Fraction:
        return self.probability * self.aggregation_weight


@dataclass(frozen=True)
class CharacteristicBatch:
    prefix_rounds: int
    mode: str
    characteristics: tuple[PrefixCharacteristic, ...]
    total_characteristics: int
    represented_characteristics: int
    sampled_characteristics: int
    covered_probability: Fraction | None
    samples_per_first_family: int | None = None
    seed: int | None = None


def _parallel_active_columns(state: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (column, value)
        for column, value in enumerate(xoodoo_linear.columns(state))
        if value
    )


def raw_branch_count(state: int) -> int:
    return prod(
        len(qd_kernel.DDT_SUPPORT[value])
        for _, value in _parallel_active_columns(state)
    )


def expand_chi_layer(input_difference: int) -> Iterator[tuple[int, Fraction]]:
    active = _parallel_active_columns(input_difference)
    if not active:
        yield 0, Fraction(1, 1)
        return

    def visit(position: int, output: int, probability: Fraction):
        if position == len(active):
            yield output, probability
            return
        column, difference = active[position]
        for gamma in qd_kernel.DDT_SUPPORT[difference]:
            count = qd_kernel.DDT_COUNTS[difference][gamma]
            yield from visit(
                position + 1,
                output ^ common.column_mask(column, gamma),
                probability * Fraction(count, 8),
            )

    yield from visit(0, 0, Fraction(1, 1))


def complete_characteristic_count(input_difference: int, prefix_rounds: int) -> int:
    if prefix_rounds == 0:
        return 1
    if prefix_rounds == 1:
        return raw_branch_count(input_difference)
    if prefix_rounds == 2:
        return sum(
            raw_branch_count(xoodoo_linear.inter_chi_linear(first_output))
            for first_output, _ in expand_chi_layer(input_difference)
        )
    raise ValueError("prefix_rounds must be 0, 1, or 2")


def _boundary_difference(gamma: int, total_rounds: int, prefix_rounds: int) -> int:
    if prefix_rounds < total_rounds:
        return xoodoo_linear.inter_chi_linear(gamma)
    return gamma


def _complete_batch(
    input_difference: int,
    total_rounds: int,
    prefix_rounds: int,
    *,
    max_characteristics: int,
) -> CharacteristicBatch:
    total = complete_characteristic_count(input_difference, prefix_rounds)
    if total > max_characteristics:
        raise CharacteristicLimitExceeded(
            f"complete {prefix_rounds}DDT needs {total} characteristics, "
            f"above max_characteristics={max_characteristics}"
        )
    if prefix_rounds == 0:
        characteristic = PrefixCharacteristic(
            characteristic_id=0,
            prefix_rounds=0,
            delta_inputs=(),
            gamma_outputs=(),
            boundary_difference=input_difference,
            probability=Fraction(1, 1),
        )
        return CharacteristicBatch(0, "identity", (characteristic,), 1, 1, 1, Fraction(1, 1))

    records: list[PrefixCharacteristic] = []
    if prefix_rounds == 1:
        for characteristic_id, (gamma, probability) in enumerate(
            expand_chi_layer(input_difference)
        ):
            records.append(
                PrefixCharacteristic(
                    characteristic_id=characteristic_id,
                    prefix_rounds=1,
                    delta_inputs=(input_difference,),
                    gamma_outputs=(gamma,),
                    boundary_difference=_boundary_difference(gamma, total_rounds, 1),
                    probability=probability,
                    first_family_id=characteristic_id,
                )
            )
    else:
        characteristic_id = 0
        for family_id, (gamma0, probability0) in enumerate(
            expand_chi_layer(input_difference)
        ):
            delta1 = xoodoo_linear.inter_chi_linear(gamma0)
            for gamma1, probability1 in expand_chi_layer(delta1):
                records.append(
                    PrefixCharacteristic(
                        characteristic_id=characteristic_id,
                        prefix_rounds=2,
                        delta_inputs=(input_difference, delta1),
                        gamma_outputs=(gamma0, gamma1),
                        boundary_difference=_boundary_difference(gamma1, total_rounds, 2),
                        probability=probability0 * probability1,
                        first_family_id=family_id,
                    )
                )
                characteristic_id += 1
    covered = sum((item.probability for item in records), Fraction(0, 1))
    return CharacteristicBatch(
        prefix_rounds,
        "complete",
        tuple(records),
        total,
        total,
        len(records),
        covered,
    )


def _sample_second_output(
    rng: random.Random, input_difference: int
) -> tuple[int, Fraction]:
    output = 0
    probability = Fraction(1, 1)
    for column, difference in _parallel_active_columns(input_difference):
        support = qd_kernel.DDT_SUPPORT[difference]
        gamma = support[rng.randrange(len(support))]
        count = qd_kernel.DDT_COUNTS[difference][gamma]
        output ^= common.column_mask(column, gamma)
        probability *= Fraction(count, 8)
    return output, probability


def _stratified_two_round_batch(
    input_difference: int,
    total_rounds: int,
    *,
    samples_per_first_family: int,
    seed: int,
) -> CharacteristicBatch:
    if samples_per_first_family <= 0:
        raise ValueError("samples_per_first_family must be positive")
    first = list(expand_chi_layer(input_difference))
    total = complete_characteristic_count(input_difference, 2)
    rng = random.Random(seed)
    records: list[PrefixCharacteristic] = []
    characteristic_id = 0
    for family_id, (gamma0, probability0) in enumerate(first):
        delta1 = xoodoo_linear.inter_chi_linear(gamma0)
        family_size = raw_branch_count(delta1)
        aggregation_weight = Fraction(family_size, samples_per_first_family)
        sampling_probability = Fraction(1, family_size)
        for _ in range(samples_per_first_family):
            gamma1, probability1 = _sample_second_output(rng, delta1)
            records.append(
                PrefixCharacteristic(
                    characteristic_id=characteristic_id,
                    prefix_rounds=2,
                    delta_inputs=(input_difference, delta1),
                    gamma_outputs=(gamma0, gamma1),
                    boundary_difference=_boundary_difference(gamma1, total_rounds, 2),
                    probability=probability0 * probability1,
                    aggregation_weight=aggregation_weight,
                    first_family_id=family_id,
                    sampling_probability=sampling_probability,
                )
            )
            characteristic_id += 1
    return CharacteristicBatch(
        2,
        "stratified_unbiased",
        tuple(records),
        total,
        total,
        len(records),
        None,
        samples_per_first_family=samples_per_first_family,
        seed=seed,
    )


def _restricted_two_round_batch(
    input_difference: int,
    total_rounds: int,
    gamma_pairs: Sequence[tuple[int, int] | tuple[int, int, Fraction]],
) -> CharacteristicBatch:
    if not gamma_pairs:
        raise ValueError("restricted_deterministic mode needs at least one gamma pair")
    first_entries = list(expand_chi_layer(input_difference))
    first_probabilities = {
        gamma: (family_id, probability)
        for family_id, (gamma, probability) in enumerate(first_entries)
    }
    records: list[PrefixCharacteristic] = []
    covered = Fraction(0, 1)
    weighted = False
    for characteristic_id, raw_entry in enumerate(gamma_pairs):
        if len(raw_entry) == 2:
            gamma0, gamma1 = raw_entry
            aggregation_weight = Fraction(1, 1)
        elif len(raw_entry) == 3:
            gamma0, gamma1, aggregation_weight = raw_entry
            aggregation_weight = Fraction(aggregation_weight)
            weighted = weighted or aggregation_weight != 1
        else:
            raise ValueError(
                "restricted characteristics need gamma0, gamma1 and an optional weight"
            )
        if gamma0 not in first_probabilities:
            raise ValueError("restricted gamma0 is not a valid first-DDT output")
        family_id, probability0 = first_probabilities[gamma0]
        delta1 = xoodoo_linear.inter_chi_linear(gamma0)
        probability1 = Fraction(1, 1)
        input_columns = xoodoo_linear.columns(delta1)
        output_columns = xoodoo_linear.columns(gamma1)
        for difference, gamma in zip(input_columns, output_columns):
            count = qd_kernel.DDT_COUNTS[difference][gamma]
            if not count:
                raise ValueError("restricted gamma1 is not valid for gamma0")
            probability1 *= Fraction(count, 8)
        probability = probability0 * probability1
        covered += probability
        records.append(
            PrefixCharacteristic(
                characteristic_id=characteristic_id,
                prefix_rounds=2,
                delta_inputs=(input_difference, delta1),
                gamma_outputs=(gamma0, gamma1),
                boundary_difference=_boundary_difference(gamma1, total_rounds, 2),
                probability=probability,
                aggregation_weight=aggregation_weight,
                first_family_id=family_id,
            )
        )
    return CharacteristicBatch(
        2,
        "restricted_weighted_legacy" if weighted else "restricted_deterministic",
        tuple(records),
        complete_characteristic_count(input_difference, 2),
        len(records),
        len(records),
        None if weighted else covered,
    )


def build_batch(
    input_difference: int,
    total_rounds: int,
    prefix_rounds: int,
    *,
    mode: str | None = None,
    max_characteristics: int = 1_000_000,
    samples_per_first_family: int = 1,
    seed: int = 20260815,
    restricted_gamma_pairs: Sequence[
        tuple[int, int] | tuple[int, int, Fraction]
    ] = (),
) -> CharacteristicBatch:
    if prefix_rounds not in (0, 1, 2):
        raise ValueError("prefix_rounds must be 0, 1, or 2")
    if prefix_rounds > total_rounds:
        raise ValueError("DDT prefix cannot exceed total rounds")
    if prefix_rounds < 2:
        if mode not in (None, "identity", "complete"):
            raise ValueError("0/1DDT only support identity/complete mode")
        return _complete_batch(
            input_difference,
            total_rounds,
            prefix_rounds,
            max_characteristics=max_characteristics,
        )
    selected_mode = mode or "stratified_unbiased"
    if selected_mode == "complete":
        return _complete_batch(
            input_difference,
            total_rounds,
            prefix_rounds,
            max_characteristics=max_characteristics,
        )
    if selected_mode == "stratified_unbiased":
        return _stratified_two_round_batch(
            input_difference,
            total_rounds,
            samples_per_first_family=samples_per_first_family,
            seed=seed,
        )
    if selected_mode == "restricted_deterministic":
        return _restricted_two_round_batch(
            input_difference, total_rounds, restricted_gamma_pairs
        )
    raise ValueError(f"unknown prefix mode {selected_mode!r}")
