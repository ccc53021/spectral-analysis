"""Sparse value-mask pullback through a fixed DDT prefix.

For a fixed local differential transition ``delta -> gamma`` the transform

    Q[delta,gamma,u,v] = 2^-3 sum_x 1[D_delta chi(x)=gamma]
                         (-1)^(<u,x> + <v,chi(x)>)

maps a Fourier mask ``v`` at the chi output to Fourier masks ``u`` at its
input.  Products over Xoodoo's 128 columns build the parallel-chi operator;
the affine connectors are handled by their adjoint and their constant sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from . import common, qd_kernel
from . import linear_layer as xoodoo_linear
from .prefix_characteristics import PrefixCharacteristic


Polynomial = dict[int, float]


class CosetReducer:
    """Canonical GF(2) quotient map for an original-input mask span."""

    def __init__(self, basis: Iterable[int]) -> None:
        rows: dict[int, int] = {}
        normalized = []
        for raw in basis:
            value = int(raw)
            if value < 0 or value >= 1 << common.STATE_BITS:
                raise ValueError("coset basis mask does not fit the Xoodoo state")
            original = value
            while value:
                pivot = value.bit_length() - 1
                if pivot in rows:
                    value ^= rows[pivot]
                else:
                    rows[pivot] = value
                    normalized.append(original)
                    break
            if value == 0:
                raise ValueError("coset basis must be GF(2)-independent")
        self.basis = tuple(normalized)
        self.rows = tuple(sorted(rows.items(), reverse=True))

    @property
    def dimension(self) -> int:
        return len(self.basis)

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
        return {
            "local_kernel_calls": self.local_kernel_calls,
            "local_kernel_cache_hits": self.local_kernel_cache_hits,
            "connector_pullbacks": self.connector_pullbacks,
            "polynomial_products": self.polynomial_products,
            "polynomial_sums": self.polynomial_sums,
            "pruning_events": self.pruning_events,
            "dropped_terms": self.dropped_terms,
            "maximum_unpruned_terms": self.maximum_unpruned_terms,
        }


def _clean(polynomial: Mapping[int, float]) -> Polynomial:
    return {int(mask): float(value) for mask, value in polynomial.items() if value}


def prune_polynomial(
    polynomial: Mapping[int, float],
    capacity: int,
    *,
    preserve_masks: Iterable[int] = (0,),
    diagnostics: PullbackDiagnostics | None = None,
    coset_reducer: CosetReducer | None = None,
) -> Polynomial:
    """Retain the largest coefficients while preserving requested labels.

    This intentionally mirrors the ordinary v4 DAG policy: absolute
    coefficient first and deterministic mask order, with the zero label kept
    whenever it was present before pruning.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    cleaned = _clean(polynomial)
    size = len(cleaned)
    if diagnostics is not None:
        diagnostics.maximum_unpruned_terms = max(
            diagnostics.maximum_unpruned_terms, size
        )
    if size <= capacity:
        return cleaned

    if coset_reducer is None:
        retained = sorted(
            cleaned, key=lambda mask: (-abs(cleaned[mask]), mask)
        )[:capacity]
    else:
        groups: dict[int, list[int]] = {}
        for mask in cleaned:
            groups.setdefault(coset_reducer.quotient(mask), []).append(mask)
        ordered_groups = []
        for quotient, masks in groups.items():
            masks.sort(key=lambda mask: (-abs(cleaned[mask]), mask))
            squared_norm = sum(cleaned[mask] ** 2 for mask in masks)
            ordered_groups.append((quotient, squared_norm, masks))
        ordered_groups.sort(
            key=lambda item: (item[0] != 0, -item[1], item[0])
        )
        retained = []
        for _, _, masks in ordered_groups:
            take = min(len(masks), capacity - len(retained))
            retained.extend(masks[:take])
            if len(retained) == capacity:
                break
    retained_set = set(retained)
    protected = [
        int(mask)
        for mask in preserve_masks
        if int(mask) in cleaned and int(mask) not in retained_set
    ]
    # Replace the weakest non-protected retained labels.  The final sort makes
    # the outcome independent of set/dictionary iteration order.
    protected_set = {int(mask) for mask in preserve_masks}
    for mask in protected:
        replaceable = [item for item in retained if item not in protected_set]
        if not replaceable:
            break
        victim = min(replaceable, key=lambda item: (abs(cleaned[item]), -item))
        retained.remove(victim)
        retained_set.remove(victim)
        retained.append(mask)
        retained_set.add(mask)

    if diagnostics is not None:
        diagnostics.pruning_events += 1
        diagnostics.dropped_terms += size - len(retained)
    return {mask: cleaned[mask] for mask in sorted(retained)}


def add_scaled(
    destination: Polynomial,
    source: Mapping[int, float],
    scale: float,
) -> None:
    if not scale:
        return
    for mask, coefficient in source.items():
        value = destination.get(mask, 0.0) + scale * coefficient
        if value:
            destination[mask] = value
        else:
            destination.pop(mask, None)


def multiply_xor(
    left: Mapping[int, float],
    right: Mapping[int, float],
    capacity: int,
    *,
    diagnostics: PullbackDiagnostics | None = None,
    coset_reducer: CosetReducer | None = None,
) -> Polynomial:
    if diagnostics is not None:
        diagnostics.polynomial_products += 1
    result: Polynomial = {}
    for left_mask, left_coefficient in left.items():
        for right_mask, right_coefficient in right.items():
            mask = left_mask ^ right_mask
            value = result.get(mask, 0.0) + left_coefficient * right_coefficient
            if value:
                result[mask] = value
            else:
                result.pop(mask, None)
    return prune_polynomial(
        result,
        capacity,
        diagnostics=diagnostics,
        coset_reducer=coset_reducer,
    )


class PrefixOperator:
    """A bounded sparse ``K_tau(U,V)`` operator for one characteristic."""

    def __init__(
        self,
        total_rounds: int,
        capacity: int,
        *,
        connector_constants: tuple[int, ...] | None = None,
        coset_basis: Iterable[int] = (),
    ) -> None:
        if not 0 <= total_rounds <= 12:
            raise ValueError("total_rounds must be in [0,12]")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if connector_constants is None:
            connector_constants = tuple(
                common.ROUND_CONSTANTS[12 - total_rounds + 1 :]
            )
        if len(connector_constants) != max(0, total_rounds - 1):
            raise ValueError("one constant is required for every chi connector")
        self.total_rounds = total_rounds
        self.capacity = capacity
        self.connector_constants = tuple(int(value) for value in connector_constants)
        self.coset_reducer = CosetReducer(coset_basis)
        if self.coset_reducer.dimension and capacity < (1 << self.coset_reducer.dimension):
            raise ValueError("prefix capacity must cover the complete requested span")
        self.diagnostics = PullbackDiagnostics()
        self._local_cache: dict[
            tuple[int, int, int, bool], tuple[tuple[int, float], ...]
        ] = {}
        self._characteristic_cache: dict[
            tuple[PrefixCharacteristic, int], tuple[tuple[int, float], ...]
        ] = {}

    def _conditioned_chi_pullback(
        self,
        difference_input: int,
        difference_output: int,
        output_value_mask: int,
        *,
        original_input_layer: bool,
    ) -> Polynomial:
        cache_key = (
            difference_input,
            difference_output,
            output_value_mask,
            original_input_layer,
        )
        cached = self._local_cache.get(cache_key)
        if cached is not None:
            self.diagnostics.local_kernel_cache_hits += 1
            return dict(cached)

        self.diagnostics.local_kernel_calls += 1
        input_columns = xoodoo_linear.columns(difference_input)
        output_columns = xoodoo_linear.columns(difference_output)
        value_columns = xoodoo_linear.columns(output_value_mask)
        polynomial: Polynomial = {0: 1.0}
        for column, (difference, gamma, value_mask) in enumerate(
            zip(input_columns, output_columns, value_columns)
        ):
            if difference == 0 and gamma == 0 and value_mask == 0:
                continue
            local = {
                common.column_mask(column, input_mask): numerator / 8.0
                for input_mask, numerator in qd_kernel.row_numerators(
                    difference, gamma, value_mask
                )
            }
            if not local:
                polynomial = {}
                break
            polynomial = multiply_xor(
                polynomial,
                local,
                self.capacity,
                diagnostics=self.diagnostics,
                coset_reducer=(
                    self.coset_reducer
                    if original_input_layer and self.coset_reducer.dimension
                    else None
                ),
            )
            if not polynomial:
                break

        cached_value = tuple(sorted(polynomial.items()))
        self._local_cache[cache_key] = cached_value
        return dict(cached_value)

    def _cross_connector(self, mask: int, connector_index: int) -> tuple[int, float]:
        self.diagnostics.connector_pullbacks += 1
        constant = xoodoo_linear.constant_state(
            self.connector_constants[connector_index]
        )
        sign = -1.0 if (int(mask) & constant).bit_count() & 1 else 1.0
        return xoodoo_linear.inter_chi_adjoint(mask), sign

    def pullback(
        self,
        characteristic: PrefixCharacteristic,
        suffix_value_mask: int,
    ) -> Polynomial:
        """Return the bounded input-``U`` polynomial for one suffix mask.

        The returned coefficients already contain every local DDT branch
        probability.  A caller must therefore *not* multiply by
        ``characteristic.probability`` again.  Only an explicit sampling
        Horvitz--Thompson weight belongs outside this operator.
        """
        if characteristic.prefix_rounds > self.total_rounds:
            raise ValueError("characteristic prefix exceeds total rounds")
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
                sign = 1.0
                # The current mask is at the following chi input whenever a
                # connector exists after this prefix layer.
                if layer + 1 < self.total_rounds:
                    chi_output_mask, sign = self._cross_connector(mask, layer)
                local = self._conditioned_chi_pullback(
                    characteristic.delta_inputs[layer],
                    characteristic.gamma_outputs[layer],
                    chi_output_mask,
                    original_input_layer=(layer == 0),
                )
                add_scaled(accumulator, local, coefficient * sign)
            self.diagnostics.polynomial_sums += 1
            current = prune_polynomial(
                accumulator,
                self.capacity,
                diagnostics=self.diagnostics,
                coset_reducer=(
                    self.coset_reducer
                    if layer == 0 and self.coset_reducer.dimension
                    else None
                ),
            )
            if not current:
                break

        cached_value = tuple(sorted(current.items()))
        self._characteristic_cache[cache_key] = cached_value
        return dict(cached_value)


__all__ = [
    "Polynomial",
    "CosetReducer",
    "PrefixOperator",
    "PullbackDiagnostics",
    "add_scaled",
    "multiply_xor",
    "prune_polynomial",
]
