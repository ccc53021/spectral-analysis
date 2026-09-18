"""Complete 0/1DDT and complete or stratified 2DDT prefix characteristics."""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from math import prod

import ascon_ddt_common as common
import ascon_qd_kernel as qd


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


@dataclass(frozen=True)
class CharacteristicBatch:
    prefix_rounds: int
    mode: str
    characteristics: tuple[PrefixCharacteristic, ...]
    total_characteristics: int
    covered_probability: Fraction | None
    samples_per_first_family: int | None = None
    seed: int | None = None


class CharacteristicLimitExceeded(RuntimeError):
    pass


def raw_branch_count(state: int) -> int:
    return prod(len(qd.DDT_SUPPORT[value]) for value in common.columns(state) if value)


def expand_sbox_layer(input_difference: int):
    active = tuple((column, value) for column, value in enumerate(common.columns(input_difference)) if value)
    if not active:
        yield 0, Fraction(1, 1)
        return

    def visit(position: int, output: int, probability: Fraction):
        if position == len(active):
            yield output, probability
            return
        column, difference = active[position]
        for gamma in qd.DDT_SUPPORT[difference]:
            yield from visit(
                position + 1,
                output ^ common.column_mask(column, gamma),
                probability * Fraction(qd.DDT_COUNTS[difference][gamma], 32),
            )

    yield from visit(0, 0, Fraction(1, 1))


def complete_characteristic_count(input_difference: int, prefix_rounds: int) -> int:
    if prefix_rounds == 0:
        return 1
    if prefix_rounds == 1:
        return raw_branch_count(input_difference)
    if prefix_rounds == 2:
        return sum(raw_branch_count(common.forward_linear(gamma)) for gamma, _ in expand_sbox_layer(input_difference))
    raise ValueError("prefix_rounds must be 0, 1, or 2")


def _boundary(gamma: int, total_rounds: int, prefix_rounds: int) -> int:
    return common.forward_linear(gamma) if prefix_rounds < total_rounds else gamma


def _sample_layer(rng: random.Random, difference: int) -> tuple[int, Fraction]:
    gamma_state = 0
    probability = Fraction(1, 1)
    for column, local in enumerate(common.columns(difference)):
        if not local:
            continue
        support = qd.DDT_SUPPORT[local]
        gamma = support[rng.randrange(len(support))]
        gamma_state ^= common.column_mask(column, gamma)
        probability *= Fraction(qd.DDT_COUNTS[local][gamma], 32)
    return gamma_state, probability


def build_batch(
    input_difference: int,
    total_rounds: int,
    prefix_rounds: int,
    *,
    mode: str | None = None,
    max_characteristics: int = 2_000_000,
    samples_per_first_family: int = 1,
    seed: int = 20260816,
) -> CharacteristicBatch:
    if prefix_rounds not in (0, 1, 2) or prefix_rounds > total_rounds:
        raise ValueError("invalid DDT prefix length")
    total = complete_characteristic_count(input_difference, prefix_rounds)
    if prefix_rounds == 0:
        item = PrefixCharacteristic(0, 0, (), (), input_difference, Fraction(1, 1))
        return CharacteristicBatch(0, "identity", (item,), 1, Fraction(1, 1))

    first = tuple(expand_sbox_layer(input_difference))
    if prefix_rounds == 1:
        records = tuple(
            PrefixCharacteristic(index, 1, (input_difference,), (gamma,), _boundary(gamma, total_rounds, 1), probability, first_family_id=index)
            for index, (gamma, probability) in enumerate(first)
        )
        return CharacteristicBatch(1, "complete", records, total, sum((x.probability for x in records), Fraction()))

    selected_mode = mode or "stratified_unbiased"
    if selected_mode == "complete":
        if total > max_characteristics:
            raise CharacteristicLimitExceeded(f"complete 2DDT needs {total} characteristics")
        records = []
        characteristic_id = 0
        for family_id, (gamma0, probability0) in enumerate(first):
            delta1 = common.forward_linear(gamma0)
            for gamma1, probability1 in expand_sbox_layer(delta1):
                records.append(PrefixCharacteristic(
                    characteristic_id, 2, (input_difference, delta1), (gamma0, gamma1),
                    _boundary(gamma1, total_rounds, 2), probability0 * probability1,
                    first_family_id=family_id,
                ))
                characteristic_id += 1
        values = tuple(records)
        return CharacteristicBatch(2, "complete", values, total, sum((x.probability for x in values), Fraction()))

    if selected_mode != "stratified_unbiased" or samples_per_first_family <= 0:
        raise ValueError("2DDT mode must be complete or stratified_unbiased")
    rng = random.Random(seed)
    records = []
    characteristic_id = 0
    for family_id, (gamma0, probability0) in enumerate(first):
        delta1 = common.forward_linear(gamma0)
        family_size = raw_branch_count(delta1)
        for _ in range(samples_per_first_family):
            gamma1, probability1 = _sample_layer(rng, delta1)
            records.append(PrefixCharacteristic(
                characteristic_id, 2, (input_difference, delta1), (gamma0, gamma1),
                _boundary(gamma1, total_rounds, 2), probability0 * probability1,
                Fraction(family_size, samples_per_first_family), family_id,
            ))
            characteristic_id += 1
    return CharacteristicBatch(
        2, "stratified_unbiased", tuple(records), total, None,
        samples_per_first_family=samples_per_first_family, seed=seed,
    )
