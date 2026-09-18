"""Sparse pullback of suffix value masks through fixed Ascon DDT prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import ascon_ddt_common as common
import ascon_qd_kernel as qd
from ascon_prefix_characteristics import PrefixCharacteristic


Polynomial = dict[int, float]


class CosetReducer:
    def __init__(self, basis: Iterable[int] = ()) -> None:
        rows: dict[int, int] = {}
        normalized = []
        for raw in basis:
            original = value = int(raw)
            while value:
                pivot = value.bit_length() - 1
                if pivot in rows:
                    value ^= rows[pivot]
                else:
                    rows[pivot] = value
                    normalized.append(original)
                    break
            if not value:
                raise ValueError("coset basis must be GF(2)-independent")
        self.basis = tuple(normalized)
        self.rows = tuple(sorted(rows.items(), reverse=True))

    def quotient(self, mask: int) -> int:
        value = int(mask)
        for pivot, row in self.rows:
            if (value >> pivot) & 1:
                value ^= row
        return value


@dataclass
class PullbackDiagnostics:
    local_kernel_calls: int = 0
    local_kernel_cache_hits: int = 0
    connector_pullbacks: int = 0
    polynomial_products: int = 0
    polynomial_sums: int = 0
    pruning_events: int = 0
    dropped_terms: int = 0
    maximum_unpruned_terms: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(vars(self))


def prune_polynomial(
    polynomial: Mapping[int, float],
    capacity: int,
    *,
    diagnostics: PullbackDiagnostics | None = None,
    preserve_masks: Iterable[int] = (0,),
    coset_reducer: CosetReducer | None = None,
) -> Polynomial:
    cleaned = {int(mask): float(value) for mask, value in polynomial.items() if value}
    size = len(cleaned)
    if diagnostics is not None:
        diagnostics.maximum_unpruned_terms = max(diagnostics.maximum_unpruned_terms, size)
    if size <= capacity:
        return cleaned

    if coset_reducer is None:
        retained = sorted(cleaned, key=lambda mask: (-abs(cleaned[mask]), mask))[:capacity]
    else:
        groups: dict[int, list[int]] = {}
        for mask in cleaned:
            groups.setdefault(coset_reducer.quotient(mask), []).append(mask)
        ordered_groups = sorted(
            groups.values(),
            key=lambda group: (-sum(abs(cleaned[mask]) for mask in group), min(group)),
        )
        retained = []
        for group in ordered_groups:
            for mask in sorted(group, key=lambda value: (-abs(cleaned[value]), value)):
                if len(retained) == capacity:
                    break
                retained.append(mask)
            if len(retained) == capacity:
                break

    retained_set = set(retained)
    for mask in preserve_masks:
        mask = int(mask)
        if mask not in cleaned or mask in retained_set:
            continue
        replaceable = [value for value in retained if value not in set(preserve_masks)]
        if not replaceable:
            continue
        victim = min(replaceable, key=lambda value: (abs(cleaned[value]), -value))
        retained.remove(victim)
        retained_set.remove(victim)
        retained.append(mask)
        retained_set.add(mask)
    if diagnostics is not None:
        diagnostics.pruning_events += 1
        diagnostics.dropped_terms += size - len(retained)
    return {mask: cleaned[mask] for mask in sorted(retained)}


def multiply_xor(
    left: Mapping[int, float],
    right: Mapping[int, float],
    capacity: int,
    *,
    diagnostics: PullbackDiagnostics,
    coset_reducer: CosetReducer | None = None,
) -> Polynomial:
    diagnostics.polynomial_products += 1
    result: Polynomial = {}
    for left_mask, left_value in left.items():
        for right_mask, right_value in right.items():
            mask = left_mask ^ right_mask
            value = result.get(mask, 0.0) + left_value * right_value
            if value:
                result[mask] = value
            else:
                result.pop(mask, None)
    return prune_polynomial(result, capacity, diagnostics=diagnostics, coset_reducer=coset_reducer)


class PrefixOperator:
    """Bounded sparse ``K_tau(U,V)`` for an Ascon DDT characteristic."""

    def __init__(
        self,
        total_rounds: int,
        capacity: int,
        *,
        begin_round: int = 0,
        domain: str = "ascon128",
        iv: int = 0x80400C0600000000,
        coset_basis: Iterable[int] = (),
    ) -> None:
        self.total_rounds = int(total_rounds)
        self.capacity = int(capacity)
        self.begin_round = int(begin_round)
        self.domain = domain
        self.iv = int(iv)
        self.coset_reducer = CosetReducer(coset_basis)
        self.diagnostics = PullbackDiagnostics()
        self._local_cache: dict[tuple[int, int, int, int], tuple[tuple[int, float], ...]] = {}
        self._characteristic_cache: dict[tuple[PrefixCharacteristic, int], tuple[tuple[int, float], ...]] = {}

    def _conditioned_sbox_pullback(
        self,
        difference_input: int,
        difference_output: int,
        output_value_mask: int,
        layer: int,
    ) -> Polynomial:
        key = (difference_input, difference_output, output_value_mask, layer)
        cached = self._local_cache.get(key)
        if cached is not None:
            self.diagnostics.local_kernel_cache_hits += 1
            return dict(cached)
        self.diagnostics.local_kernel_calls += 1
        polynomial: Polynomial = {0: 1.0}
        constant = common.round_constant_state(self.begin_round + layer)
        original_layer = layer == 0
        for column, (difference, gamma, value_mask) in enumerate(zip(
            common.columns(difference_input),
            common.columns(difference_output),
            common.columns(output_value_mask),
        )):
            if difference == gamma == value_mask == 0:
                continue
            local: Polynomial = {}
            for input_mask, numerator in qd.row_numerators(difference, gamma, value_mask):
                physical = common.column_mask(column, input_mask)
                phase = -1.0 if (physical & constant).bit_count() & 1 else 1.0
                if original_layer:
                    physical, domain_phase = common.canonical_original_mask(physical, self.domain, self.iv)
                    phase *= domain_phase
                value = local.get(physical, 0.0) + phase * numerator / 32.0
                if value:
                    local[physical] = value
                else:
                    local.pop(physical, None)
            if not local:
                polynomial = {}
                break
            polynomial = multiply_xor(
                polynomial, local, self.capacity,
                diagnostics=self.diagnostics,
                coset_reducer=self.coset_reducer if original_layer and self.coset_reducer.basis else None,
            )
        self._local_cache[key] = tuple(sorted(polynomial.items()))
        return polynomial

    def pullback(self, characteristic: PrefixCharacteristic, suffix_value_mask: int) -> Polynomial:
        cache_key = (characteristic, int(suffix_value_mask))
        cached = self._characteristic_cache.get(cache_key)
        if cached is not None:
            return dict(cached)
        if characteristic.prefix_rounds == 0:
            result = {int(suffix_value_mask): 1.0}
            self._characteristic_cache[cache_key] = tuple(result.items())
            return result

        current: Polynomial = {int(suffix_value_mask): 1.0}
        for layer in range(characteristic.prefix_rounds - 1, -1, -1):
            accumulator: Polynomial = {}
            for mask, coefficient in current.items():
                chi_output_mask = mask
                if layer + 1 < self.total_rounds:
                    chi_output_mask = common.adjoint_linear(mask)
                    self.diagnostics.connector_pullbacks += 1
                local = self._conditioned_sbox_pullback(
                    characteristic.delta_inputs[layer],
                    characteristic.gamma_outputs[layer],
                    chi_output_mask,
                    layer,
                )
                common.add_scaled(accumulator, local, coefficient)
            self.diagnostics.polynomial_sums += 1
            current = prune_polynomial(
                accumulator, self.capacity, diagnostics=self.diagnostics,
                coset_reducer=self.coset_reducer if layer == 0 and self.coset_reducer.basis else None,
            )
            if not current:
                break
        self._characteristic_cache[cache_key] = tuple(sorted(current.items()))
        return current

