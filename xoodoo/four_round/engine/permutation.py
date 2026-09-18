"""Balanced affine-class experiments on the real Xoodoo chi-to-chi map."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np

from . import common
from .config import Distinguisher, get


def _parse_mask(words: list[str]) -> tuple[int, ...]:
    return tuple(int(word, 16) for word in words)


def _rotl32(values: np.ndarray, amount: int) -> np.ndarray:
    amount %= 32
    if amount == 0:
        return values.copy()
    return (values << np.uint32(amount)) | (values >> np.uint32(32 - amount))


def _shift_plane(plane: np.ndarray, x_shift: int, z_shift: int) -> np.ndarray:
    return np.roll(_rotl32(plane, z_shift), x_shift, axis=2)


def _theta(state: np.ndarray) -> np.ndarray:
    parity_plane = state[:, 0:1, :] ^ state[:, 1:2, :] ^ state[:, 2:3, :]
    effect = _shift_plane(parity_plane, 1, 5) ^ _shift_plane(parity_plane, 1, 14)
    return state ^ effect


def _rho_west(state: np.ndarray) -> np.ndarray:
    result = state.copy()
    result[:, 1:2, :] = _shift_plane(state[:, 1:2, :], 1, 0)
    result[:, 2:3, :] = _shift_plane(state[:, 2:3, :], 0, 11)
    return result


def _rho_east(state: np.ndarray) -> np.ndarray:
    result = state.copy()
    result[:, 1:2, :] = _shift_plane(state[:, 1:2, :], 0, 1)
    result[:, 2:3, :] = _shift_plane(state[:, 2:3, :], 2, 8)
    return result


def _chi(state: np.ndarray) -> np.ndarray:
    old = state
    return np.stack(
        [
            old[:, y, :] ^ ((~old[:, (y + 1) % 3, :]) & old[:, (y + 2) % 3, :])
            for y in range(3)
        ],
        axis=1,
    ).astype(np.uint32, copy=False)


def chi_to_chi(state: np.ndarray, rounds: int) -> np.ndarray:
    constants = common.ROUND_CONSTANTS[12 - rounds + 1 :]
    result = state
    for round_index in range(rounds):
        result = _chi(result)
        if round_index + 1 < rounds:
            result = _rho_west(_theta(_rho_east(result)))
            result[:, 0, 0] ^= np.uint32(constants[round_index])
    return result


def _state_to_lanes32(value: int) -> np.ndarray:
    return np.asarray(
        [(value >> (32 * lane)) & 0xFFFFFFFF for lane in range(12)],
        dtype=np.uint32,
    )


def _parity32(values: np.ndarray) -> np.ndarray:
    values = values.copy()
    values ^= values >> np.uint32(16)
    values ^= values >> np.uint32(8)
    values ^= values >> np.uint32(4)
    return (np.uint32(0x6996) >> (values & np.uint32(0xF))) & np.uint32(1)


@dataclass(frozen=True)
class ConstraintSystem:
    rows: tuple[int, ...]
    transforms: tuple[int, ...]
    pivots: tuple[int, ...]


def make_constraint_system(basis: list[tuple[int, ...]]) -> ConstraintSystem:
    rows = [common.words_to_int(mask) for mask in basis]
    transforms = [1 << index for index in range(len(rows))]
    position = 0
    pivots: list[int] = []
    for column in range(common.STATE_BITS - 1, -1, -1):
        selected = next(
            (row for row in range(position, len(rows)) if (rows[row] >> column) & 1),
            None,
        )
        if selected is None:
            continue
        rows[position], rows[selected] = rows[selected], rows[position]
        transforms[position], transforms[selected] = transforms[selected], transforms[position]
        for row in range(len(rows)):
            if row != position and ((rows[row] >> column) & 1):
                rows[row] ^= rows[position]
                transforms[row] ^= transforms[position]
        pivots.append(column)
        position += 1
        if position == len(rows):
            break
    if position != len(rows):
        raise ValueError("basis is not independent")
    return ConstraintSystem(tuple(rows), tuple(transforms), tuple(pivots))


def sample_affine_class(
    rng: np.random.Generator,
    system: ConstraintSystem,
    assignment: int,
    samples: int,
) -> np.ndarray:
    lanes = rng.integers(0, 1 << 32, size=(samples, 12), dtype=np.uint32)
    for row, transform, pivot in zip(system.rows, system.transforms, system.pivots):
        rhs = (transform & assignment).bit_count() & 1
        accumulator = np.zeros(samples, dtype=np.uint32)
        for lane in range(12):
            word = (row >> (32 * lane)) & 0xFFFFFFFF
            if word:
                accumulator ^= lanes[:, lane] & np.uint32(word)
        current = _parity32(accumulator)
        toggle = current ^ np.uint32(rhs)
        lanes[:, pivot // 32] ^= toggle << np.uint32(pivot % 32)
    return lanes.reshape(samples, 3, 4)


def evaluate_assignment(
    config: Distinguisher,
    system: ConstraintSystem,
    assignment: int,
    samples: int,
    seed: int,
    batch_size: int = 1 << 16,
) -> dict:
    rng = np.random.default_rng(seed)
    difference_value = common.state_from_columns(config.input_columns)
    difference = _state_to_lanes32(difference_value).reshape(1, 3, 4)
    output_x, output_z = divmod(config.output_column, 32)
    signed_sum = 0
    processed = 0
    while processed < samples:
        count = min(batch_size, samples - processed)
        left_input = sample_affine_class(rng, system, assignment, count)
        right_input = left_input ^ difference
        left = chi_to_chi(left_input, config.rounds)
        right = chi_to_chi(right_input, config.rounds)
        parity = np.zeros(count, dtype=np.uint32)
        for y in range(3):
            if (config.output_mask >> y) & 1:
                parity ^= ((left[:, y, output_x] ^ right[:, y, output_x]) >> np.uint32(output_z)) & np.uint32(1)
        odd = int(np.count_nonzero(parity))
        signed_sum += count - 2 * odd
        processed += count
    correlation = signed_sum / samples
    standard_error = math.sqrt(max(0.0, 1.0 - correlation * correlation) / samples)
    return {
        "assignment_index": assignment,
        "assignment": [(assignment >> bit) & 1 for bit in range(len(system.rows))],
        "samples": samples,
        "signed_sum": signed_sum,
        "correlation": correlation,
        "weight": common.weight(correlation),
        "standard_error": standard_error,
        "confidence_95": [
            correlation - 1.959963984540054 * standard_error,
            correlation + 1.959963984540054 * standard_error,
        ],
        "seed": seed,
    }


def run_experiment(config: Distinguisher) -> dict:
    directory = common.result_dir(config)
    basis_record = common.read_json(directory / "stable_basis.json")
    basis = [_parse_mask(item["mask_words"]) for item in basis_record["selected_basis"]]
    system = make_constraint_system(basis)
    coset = common.read_json(directory / "coset_span.json")
    model = coset["configurations"][-1]
    samples = 1 << config.experiment_samples_per_class_log2
    print(
        f"[{config.name}] real experiment: {len(basis)} dimensions, "
        f"{samples} samples/class",
        flush=True,
    )
    classes = []
    for assignment in range(1 << len(basis)):
        item = evaluate_assignment(
            config,
            system,
            assignment,
            samples,
            seed=2026081500 + 1000 * config.rounds + assignment,
        )
        classes.append(item)
        print(
            f"  class {assignment:0{len(basis)}b}: {item['correlation']:+.9f}",
            flush=True,
        )
    distribution = np.asarray([item["correlation"] for item in classes])
    experimental_spectrum = (common.fwht(distribution) / len(distribution)).tolist()
    experimental_best = max(range(len(classes)), key=lambda i: abs(classes[i]["correlation"]))
    model_best = int(model["maximum_absolute_index"])
    high_samples = 1 << config.experiment_best_samples_log2
    high_precision = {}
    for label, assignment in (("model_best", model_best), ("experimental_best", experimental_best)):
        if assignment in [item["assignment_index"] for item in high_precision.values()]:
            continue
        print(
            f"[{config.name}] high precision {label} class "
            f"{assignment:0{len(basis)}b}, N=2^{config.experiment_best_samples_log2}",
            flush=True,
        )
        high_precision[label] = evaluate_assignment(
            config,
            system,
            assignment,
            high_samples,
            seed=2026081599 + 1000 * config.rounds + assignment,
        )

    model_spectrum = np.asarray(model["fourier_coefficients"])
    model_distribution = np.asarray(model["correlation_distribution"])
    result = {
        "version": 1,
        "configuration": common.config_metadata(config),
        "method": "balanced affine classes on the real Xoodoo chi-to-chi function",
        "basis_dimension": len(basis),
        "basis": basis_record["selected_basis"],
        "samples_per_class": samples,
        "classes": classes,
        "experimental_correlation_distribution": distribution.tolist(),
        "experimental_fourier_spectrum": experimental_spectrum,
        "experimental_zero_frequency": float(distribution.mean()),
        "experimental_best_index": experimental_best,
        "experimental_best_assignment": classes[experimental_best]["assignment"],
        "experimental_best_correlation": classes[experimental_best]["correlation"],
        "model_capacity": model["capacity"],
        "model_best_index": model_best,
        "model_best_assignment": model["maximum_absolute_assignment"],
        "model_best_correlation": model["maximum_absolute_correlation"],
        "model_experiment_distribution_rmse": float(
            np.sqrt(np.mean(np.square(distribution - model_distribution)))
        ),
        "model_experiment_spectrum_rmse": float(
            np.sqrt(np.mean(np.square(np.asarray(experimental_spectrum) - model_spectrum)))
        ),
        "high_precision": high_precision,
    }
    common.write_json(directory / "real_distribution_experiment.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("distinguisher", choices=("r4_chi_single", "r5_chi_double"))
    args = parser.parse_args()
    result = run_experiment(get(args.distinguisher))
    print(
        "saved: exp-zero={:+.12g}, exp-best={:+.12g}, model-best={:+.12g}".format(
            result["experimental_zero_frequency"],
            result["experimental_best_correlation"],
            result["model_best_correlation"],
        )
    )


if __name__ == "__main__":
    main()
