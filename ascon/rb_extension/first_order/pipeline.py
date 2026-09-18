"""Single-entry Ascon v4 joint-U and 0-2DDT pipeline.

The orchestration reuses the validated Q kernel, DDT characteristics,
prefix pullback, and C DAG engine from the parent directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import distinguisher_config as user


HERE = Path(__file__).resolve().parent
V4_ROOT = HERE
OUTPUT_ROOT = HERE / "output"
BUILD_ROOT = HERE / "build"
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Settings:
    name: str
    description: str
    rounds: int
    begin_round: int
    input_domain: str
    iv: int
    input_difference_words: tuple[int, ...]
    output_mask_words: tuple[int, ...]
    prefix_rounds: tuple[int, ...]
    suffix_capacity: int
    prefix_capacity: int
    aggregate_capacity: int
    p2_mode: str
    p2_samples_per_first_family: int
    p2_random_seed: int
    max_complete_characteristics: int
    basis_mode: str
    auto_basis_dimension: int
    basis_masks: tuple[tuple[int, ...], ...]
    run_actual_experiments: bool
    actual_sample_log2: int
    actual_random_seed: int
    round_constants: tuple[int, ...]
    rot: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Backend:
    config: object
    rb: object
    common: object
    qd: object
    characteristics: object
    prefix_uv: object
    suffix_dag: object


def settings_from_module() -> Settings:
    cfg = Settings(
        name=str(user.NAME),
        description=str(user.DESCRIPTION),
        rounds=int(user.ROUNDS),
        begin_round=int(user.BEGIN_ROUND),
        input_domain=str(user.INPUT_DOMAIN),
        iv=int(user.IV),
        input_difference_words=tuple(map(int, user.INPUT_DIFFERENCE_WORDS)),
        output_mask_words=tuple(map(int, user.OUTPUT_MASK_WORDS)),
        prefix_rounds=tuple(map(int, user.PREFIX_ROUNDS)),
        suffix_capacity=int(user.SUFFIX_CAPACITY),
        prefix_capacity=int(user.PREFIX_CAPACITY),
        aggregate_capacity=int(user.AGGREGATE_CAPACITY),
        p2_mode=str(user.P2_MODE),
        p2_samples_per_first_family=int(user.P2_SAMPLES_PER_FIRST_FAMILY),
        p2_random_seed=int(user.P2_RANDOM_SEED),
        max_complete_characteristics=int(user.MAX_COMPLETE_CHARACTERISTICS),
        basis_mode=str(user.BASIS_MODE),
        auto_basis_dimension=int(user.AUTO_BASIS_DIMENSION),
        basis_masks=tuple(tuple(map(int, mask)) for mask in user.BASIS_MASKS),
        run_actual_experiments=bool(user.RUN_ACTUAL_EXPERIMENTS),
        actual_sample_log2=int(user.ACTUAL_SAMPLE_LOG2),
        actual_random_seed=int(user.ACTUAL_RANDOM_SEED),
        round_constants=tuple(map(int, user.ROUND_CONSTANTS)),
        rot=tuple(tuple(map(int, pair)) for pair in user.ROT),
    )
    validate_settings(cfg)
    return cfg


def validate_settings(cfg: Settings) -> None:
    if cfg.rounds <= 0:
        raise ValueError("ROUNDS must be positive")
    if cfg.begin_round < 0 or cfg.begin_round + cfg.rounds > len(cfg.round_constants):
        raise ValueError("BEGIN_ROUND/ROUNDS exceed ROUND_CONSTANTS")
    if cfg.input_domain not in {"permutation", "ascon128", "ascon128_equal"}:
        raise ValueError("unsupported INPUT_DOMAIN")
    for label, words in (
        ("INPUT_DIFFERENCE_WORDS", cfg.input_difference_words),
        ("OUTPUT_MASK_WORDS", cfg.output_mask_words),
    ):
        if len(words) != 5 or any(value < 0 or value > MASK64 for value in words):
            raise ValueError(f"{label} must contain five uint64 values")
    if not cfg.prefix_rounds or any(p not in (0, 1, 2) or p > cfg.rounds for p in cfg.prefix_rounds):
        raise ValueError("PREFIX_ROUNDS must be a nonempty subset of (0,1,2)")
    if len(set(cfg.prefix_rounds)) != len(cfg.prefix_rounds):
        raise ValueError("PREFIX_ROUNDS contains duplicates")
    if min(cfg.suffix_capacity, cfg.prefix_capacity, cfg.aggregate_capacity) <= 0:
        raise ValueError("all capacities must be positive")
    if cfg.p2_mode not in {"collapsed_complete", "complete", "stratified_unbiased"}:
        raise ValueError("invalid P2_MODE")
    if cfg.basis_mode not in {"configured", "auto"}:
        raise ValueError("BASIS_MODE must be configured or auto")
    if cfg.basis_mode == "configured" and len(cfg.basis_masks) > 20:
        raise ValueError("configured basis dimension must not exceed 20")
    if not 0 <= cfg.auto_basis_dimension <= 20:
        raise ValueError("AUTO_BASIS_DIMENSION must be in [0,20]")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ASCII-only JSON avoids Windows PowerShell 5.1 guessing UTF-8 incorrectly.
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonable_settings(cfg: Settings) -> dict:
    value = asdict(cfg)
    for key in ("iv",):
        value[key] = f"0x{value[key]:016x}"
    for key in ("input_difference_words", "output_mask_words"):
        value[key] = [f"0x{word:016x}" for word in getattr(cfg, key)]
    value["basis_masks"] = [
        [f"0x{word:016x}" for word in mask] for mask in cfg.basis_masks
    ]
    return value


def fingerprint(cfg: Settings) -> str:
    canonical = json.dumps(_jsonable_settings(cfg), sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest()[:16]


def slug(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.")
    return value or "ascon_run"


def run_directory(cfg: Settings) -> Path:
    return OUTPUT_ROOT / slug(cfg.name) / fingerprint(cfg)


def load_backend(cfg: Settings) -> Backend:
    if str(V4_ROOT) not in sys.path:
        sys.path.insert(0, str(V4_ROOT))
    import cipher_config as config_module

    config_module.ROUNDS = cfg.rounds
    config_module.BEGIN_ROUND = cfg.begin_round
    config_module.INPUT_DOMAIN = cfg.input_domain
    config_module.IV = cfg.iv
    config_module.INPUT_DIFFERENCE_WORDS = cfg.input_difference_words
    config_module.OUTPUT_MASK_WORDS = cfg.output_mask_words
    config_module.ROUND_CONSTANTS = cfg.round_constants
    config_module.ROT = cfg.rot

    import round_based as rb
    import ascon_ddt_common as common
    import ascon_qd_kernel as qd
    import ascon_prefix_characteristics as characteristics
    import ascon_prefix_uv as prefix_uv
    import suffix_label_dag as suffix_dag

    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    suffix_dag.SOURCE = V4_ROOT / "suffix_label_dag_engine.c"
    suffix_dag.BINARY = BUILD_ROOT / "suffix_label_dag_engine.exe"
    return Backend(
        config_module, rb, common, qd, characteristics, prefix_uv, suffix_dag
    )


def _is_dl1_target(cfg: Settings) -> bool:
    return (
        cfg.rounds == 4
        and cfg.begin_round == 0
        and cfg.input_domain == "ascon128"
        and cfg.iv == 0x80400C0600000000
        and cfg.input_difference_words
        == (0, 0, 0, 0x8000000000000000, 0x8000000000000000)
        and cfg.output_mask_words == (0x200, 0, 0, 0, 0)
        and cfg.basis_mode == "configured"
        and cfg.basis_masks
        == (
            (0, 0x8000000000000000, 0, 0, 0),
            (0, 0, 0, 0, 0x0120000080000000),
            (0, 0, 0x0000000080000000, 0x0000000080000000, 0x0120000080000000),
            (0, 0, 0x0020000000000000, 0x0020000000000000, 0x0120000080000000),
            (0, 0, 0x0100000000000000, 0x0100000000000000, 0x0120000080000000),
        )
    )


def _fraction(value: Fraction | None):
    if value is None:
        return None
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def _batch_record(batch) -> dict:
    probability = sum((item.probability for item in batch.characteristics), Fraction())
    mass = sum(
        (item.probability * item.aggregation_weight for item in batch.characteristics),
        Fraction(),
    )
    return {
        "prefix_rounds": batch.prefix_rounds,
        "mode": batch.mode,
        "total_complete_characteristics": batch.total_characteristics,
        "represented_records": len(batch.characteristics),
        "covered_probability": _fraction(batch.covered_probability),
        "represented_probability_sum": _fraction(probability),
        "weighted_probability_or_estimator_mass": _fraction(mass),
        "samples_per_first_family": batch.samples_per_first_family,
        "seed": batch.seed,
    }


def _characteristic_record(item, backend: Backend) -> dict:
    return {
        "characteristic_id": item.characteristic_id,
        "first_family_id": item.first_family_id,
        "delta_inputs": [backend.common.mask_to_hex(value) for value in item.delta_inputs],
        "gamma_outputs": [backend.common.mask_to_hex(value) for value in item.gamma_outputs],
        "boundary_difference": backend.common.mask_to_hex(item.boundary_difference),
        "probability": _fraction(item.probability),
        "aggregation_weight": _fraction(item.aggregation_weight),
    }


def _term(mask: int, coefficient: float, cfg: Settings, backend: Backend) -> dict:
    return {
        "mask_words": backend.common.mask_to_hex(mask),
        "expression": backend.common.mask_to_expression(mask, cfg.input_domain),
        "coefficient": coefficient,
        "absolute_coefficient": abs(coefficient),
        "weight": backend.common.weight(coefficient),
    }


def _top_terms(
    spectrum: Mapping[int, float], capacity: int, cfg: Settings, backend: Backend
) -> list[dict]:
    ordered = sorted(spectrum.items(), key=lambda item: (-abs(item[1]), item[0]))
    return [_term(mask, value, cfg, backend) for mask, value in ordered[:capacity]]


def _suffix_id(state: int) -> str:
    return hashlib.sha256(int(state).to_bytes(40, "little")).hexdigest()[:16]


def _evaluate_suffix(
    difference: int,
    *,
    rounds: int,
    begin_round: int,
    domain: str,
    iv: int,
    capacity: int,
    cfg: Settings,
    backend: Backend,
    directory: Path,
    force: bool,
) -> tuple[dict[int, float], dict]:
    key = _suffix_id(difference)
    dump_path = directory / "suffix_root_dumps" / f"hs{capacity}_{key}.bin"
    metadata_path = dump_path.with_suffix(".json")
    if dump_path.is_file() and metadata_path.is_file() and not force:
        metadata = read_json(metadata_path)
        reused = True
    else:
        metadata = backend.suffix_dag.evaluate(
            capacity,
            rounds=rounds,
            begin_round=begin_round,
            domain=domain,
            iv=iv,
            difference=backend.common.int_to_words(difference),
            output_mask=cfg.output_mask_words,
            root_dump_path=dump_path,
        )
        metadata.pop("_masks", None)
        write_json(metadata_path, metadata)
        reused = False
    spectrum = backend.common.read_root_dump(dump_path)
    return spectrum, {
        "boundary_id": key,
        "difference_words": backend.common.mask_to_hex(difference),
        "root_hash_terms": len(spectrum),
        "zero_label_coefficient": spectrum.get(0, 0.0),
        "coefficient_status": metadata["coefficient_status"],
        "pruned_polynomial_count": metadata["pruned_polynomial_count"],
        "maximum_unpruned_term_count": metadata["maximum_unpruned_term_count"],
        "seconds": metadata["seconds"],
        "reused_from_integrated_cache": reused,
        "root_dump_path": str(dump_path),
    }


def _write_prefix_result(
    prefix_rounds: int,
    spectrum: Mapping[int, float],
    record: dict,
    cfg: Settings,
    backend: Backend,
    run_dir: Path,
) -> dict:
    directory = run_dir / f"p{prefix_rounds}"
    root_path = directory / "joint_root.bin"
    backend.common.write_root_dump(root_path, spectrum)
    output = {
        "version": 1,
        "distinguisher": cfg.description,
        "prefix_rounds": prefix_rounds,
        "input_domain": cfg.input_domain,
        "input_difference_words": backend.rb.mask_to_hex(cfg.input_difference_words),
        "output_mask_words": backend.rb.mask_to_hex(cfg.output_mask_words),
        "joint_zero_label_coefficient": spectrum.get(0, 0.0),
        "joint_zero_label_weight": backend.common.weight(spectrum.get(0, 0.0)),
        "root_term_count": len(spectrum),
        "root_dump_path": str(root_path),
        "public_terms": _top_terms(spectrum, cfg.aggregate_capacity, cfg, backend),
        **record,
    }
    write_json(directory / "result.json", output)
    return output


def run_p0(
    cfg: Settings,
    backend: Backend,
    run_dir: Path,
    *,
    force: bool,
) -> tuple[dict[int, float], dict]:
    directory = run_dir / "p0"
    started = time.perf_counter()
    difference = backend.common.words_to_int(cfg.input_difference_words)
    spectrum, suffix_record = _evaluate_suffix(
        difference,
        rounds=cfg.rounds,
        begin_round=cfg.begin_round,
        domain=cfg.input_domain,
        iv=cfg.iv,
        capacity=cfg.suffix_capacity,
        cfg=cfg,
        backend=backend,
        directory=directory,
        force=force,
    )
    source = {
        "kind": "fresh_or_integrated_cache",
        "capacity": cfg.suffix_capacity,
        "path": suffix_record["root_dump_path"],
        "fresh_engine_execution": not suffix_record["reused_from_integrated_cache"],
    }
    suffix = [suffix_record]
    record = {
        "mode": "identity / no DDT prefix",
        "source": source,
        "suffix_evaluations": suffix,
        "prefix_operator_diagnostics": None,
        "seconds": time.perf_counter() - started,
    }
    return spectrum, _write_prefix_result(0, spectrum, record, cfg, backend, run_dir)


def run_explicit_prefix(
    prefix_rounds: int,
    cfg: Settings,
    backend: Backend,
    run_dir: Path,
    *,
    force: bool,
) -> tuple[dict[int, float], dict]:
    if prefix_rounds not in (1, 2):
        raise ValueError("explicit prefix must contain one or two rounds")
    mode = "complete" if prefix_rounds == 1 else cfg.p2_mode
    batch_mode = None if prefix_rounds == 1 else mode
    started = time.perf_counter()
    difference = backend.common.words_to_int(cfg.input_difference_words)
    batch = backend.characteristics.build_batch(
        difference,
        cfg.rounds,
        prefix_rounds,
        mode=batch_mode,
        max_characteristics=cfg.max_complete_characteristics,
        samples_per_first_family=cfg.p2_samples_per_first_family,
        seed=cfg.p2_random_seed,
    )
    directory = run_dir / f"p{prefix_rounds}"
    write_json(
        directory / "characteristics.json",
        {
            "batch": _batch_record(batch),
            "characteristics": [
                _characteristic_record(item, backend) for item in batch.characteristics
            ],
        },
    )
    suffix_rounds = cfg.rounds - prefix_rounds
    spectra: dict[int, dict[int, float]] = {}
    suffix_records = []
    for boundary in dict.fromkeys(item.boundary_difference for item in batch.characteristics):
        spectrum, suffix_record = _evaluate_suffix(
            boundary,
            rounds=suffix_rounds,
            begin_round=cfg.begin_round + prefix_rounds,
            domain="permutation",
            iv=0,
            capacity=cfg.suffix_capacity,
            cfg=cfg,
            backend=backend,
            directory=directory,
            force=force,
        )
        spectra[boundary] = spectrum
        suffix_records.append(suffix_record)

    configured_basis = (
        [backend.common.words_to_int(mask) for mask in cfg.basis_masks]
        if cfg.basis_mode == "configured"
        else []
    )
    operator = backend.prefix_uv.PrefixOperator(
        cfg.rounds,
        cfg.prefix_capacity,
        begin_round=cfg.begin_round,
        domain=cfg.input_domain,
        iv=cfg.iv,
        coset_basis=configured_basis,
    )
    total: dict[int, float] = {}
    for characteristic in batch.characteristics:
        contribution: dict[int, float] = {}
        for suffix_mask, suffix_coefficient in spectra[
            characteristic.boundary_difference
        ].items():
            backend.common.add_scaled(
                contribution,
                operator.pullback(characteristic, suffix_mask),
                suffix_coefficient,
            )
        # Q already includes the DDT probability. Only an estimator's
        # aggregation weight belongs outside Q.
        backend.common.add_scaled(
            total, contribution, float(characteristic.aggregation_weight)
        )
    record = {
        "mode": batch.mode,
        "batch": _batch_record(batch),
        "suffix_capacity": cfg.suffix_capacity,
        "prefix_capacity": cfg.prefix_capacity,
        "suffix_evaluations": suffix_records,
        "prefix_operator_diagnostics": operator.diagnostics.as_dict(),
        "probability_accounting": (
            "Q includes DDT probability; only aggregation_weight is external"
        ),
        "seconds": time.perf_counter() - started,
    }
    return total, _write_prefix_result(
        prefix_rounds, total, record, cfg, backend, run_dir
    )


def run_p2_collapsed(
    p0_spectrum: Mapping[int, float],
    cfg: Settings,
    backend: Backend,
    run_dir: Path,
) -> tuple[dict[int, float], dict]:
    difference = backend.common.words_to_int(cfg.input_difference_words)
    count = backend.characteristics.complete_characteristic_count(difference, 2)
    spectrum = dict(p0_spectrum)
    record = {
        "mode": "collapsed_complete",
        "total_complete_characteristics": count,
        "identity": "sum_gamma Q[delta,gamma,u,v] = LAT[v,u]",
        "interpretation": (
            "all two-DDT branches are summed algebraically before bounded "
            "suffix evaluation; this is not a sampled characteristic set"
        ),
        "source_prefix_rounds": 0,
        "seconds": 0.0,
    }
    return spectrum, _write_prefix_result(2, spectrum, record, cfg, backend, run_dir)


def _rank_ints(values: Iterable[int]) -> int:
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


def choose_basis(
    cfg: Settings, backend: Backend, spectrum: Mapping[int, float]
) -> tuple[list[int], dict]:
    if cfg.basis_mode == "configured":
        words = [
            backend.rb.validate_input_mask(mask, cfg.input_domain)
            for mask in cfg.basis_masks
        ]
        basis = [backend.common.words_to_int(mask) for mask in words]
        if _rank_ints(basis) != len(basis):
            raise ValueError("BASIS_MASKS are not GF(2)-independent")
        return basis, {
            "mode": "configured",
            "validated_for_default_dl1": _is_dl1_target(cfg),
            "warning": None,
        }

    basis: list[int] = []
    ordered = sorted(
        ((mask, value) for mask, value in spectrum.items() if mask and value),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    for mask, _ in ordered:
        if _rank_ints([*basis, mask]) > len(basis):
            basis.append(mask)
            if len(basis) == cfg.auto_basis_dimension:
                break
    return basis, {
        "mode": "auto",
        "validated_for_default_dl1": False,
        "warning": (
            "auto basis is a bounded-spectrum candidate only; check multiple "
            "capacities and real experiments before treating it as a result"
        ),
    }


def analyze_spectrum(
    spectrum: Mapping[int, float],
    basis: Sequence[int],
    cfg: Settings,
    backend: Backend,
) -> dict:
    masks = backend.common.span(basis)
    coefficients = [spectrum.get(mask, 0.0) for mask in masks]
    distribution = backend.common.fwht(coefficients)
    dimension = len(basis)
    maximum = max(range(len(distribution)), key=lambda index: distribution[index])
    minimum = min(range(len(distribution)), key=lambda index: distribution[index])
    maximum_absolute = max(
        range(len(distribution)), key=lambda index: abs(distribution[index])
    )

    def extremum(index: int) -> dict:
        value = distribution[index]
        return {
            "index": index,
            "assignment": [(index >> bit) & 1 for bit in range(dimension)],
            "correlation": value,
            "weight": backend.common.weight(value),
        }

    present = sum(mask in spectrum for mask in masks)
    return {
        "basis_dimension": dimension,
        "basis": [_term(mask, spectrum.get(mask, 0.0), cfg, backend) for mask in basis],
        "span_size": len(masks),
        "span_coordinates_present": present,
        "span_coordinates_absent_from_sparse_root": len(masks) - present,
        "fourier_coefficients": coefficients,
        "correlation_distribution": distribution,
        "maximum": extremum(maximum),
        "minimum": extremum(minimum),
        "maximum_absolute": extremum(maximum_absolute),
        "validity": (
            "bounded sparse spectrum; absent coordinates are zero in the stored "
            "model but capacity stability must be checked for a new distinguisher"
        ),
    }


def _rotr64(value: int, amount: int) -> int:
    return ((value >> amount) | (value << (64 - amount))) & MASK64


def _sbox_layer(state: list[int]) -> None:
    x0, x1, x2, x3, x4 = state
    x0 ^= x4
    x4 ^= x3
    x2 ^= x1
    t0 = x0 ^ ((~x1) & x2)
    t1 = x1 ^ ((~x2) & x3)
    t2 = x2 ^ ((~x3) & x4)
    t3 = x3 ^ ((~x4) & x0)
    t4 = x4 ^ ((~x0) & x1)
    state[:] = [
        (t0 ^ t4) & MASK64,
        (t1 ^ t0) & MASK64,
        (~t2) & MASK64,
        (t3 ^ t2) & MASK64,
        t4 & MASK64,
    ]


def ascon_rounds(state: list[int], cfg: Settings) -> None:
    for offset in range(cfg.rounds):
        state[2] ^= cfg.round_constants[cfg.begin_round + offset]
        _sbox_layer(state)
        # The round-based distinguisher observes immediately after the last S-box.
        if offset + 1 < cfg.rounds:
            for word, (first, second) in enumerate(cfg.rot):
                value = state[word]
                state[word] = value ^ _rotr64(value, first) ^ _rotr64(value, second)


def actual_sign(input_words: Sequence[int], cfg: Settings) -> int:
    words = list(map(int, input_words))
    if cfg.input_domain == "permutation":
        left = words
    else:
        left = [cfg.iv, *words[1:]]
        if cfg.input_domain == "ascon128_equal":
            left[4] = left[3]
    right = [
        value ^ difference
        for value, difference in zip(left, cfg.input_difference_words)
    ]
    ascon_rounds(left, cfg)
    ascon_rounds(right, cfg)
    parity = sum(
        ((a ^ b) & mask).bit_count()
        for a, b, mask in zip(left, right, cfg.output_mask_words)
    ) & 1
    return -1 if parity else 1


def _rref_equations(basis: Sequence[int], allowed: Sequence[int]):
    rows = [[int(mask), 1 << index] for index, mask in enumerate(basis)]
    rank = 0
    for column in sorted(allowed, reverse=True):
        selected = next(
            (
                index
                for index in range(rank, len(rows))
                if (rows[index][0] >> column) & 1
            ),
            None,
        )
        if selected is None:
            continue
        rows[rank], rows[selected] = rows[selected], rows[rank]
        for index in range(len(rows)):
            if index != rank and ((rows[index][0] >> column) & 1):
                rows[index][0] ^= rows[rank][0]
                rows[index][1] ^= rows[rank][1]
        rows[rank].append(column)
        rank += 1
        if rank == len(rows):
            break
    if rank != len(basis):
        raise ValueError("basis is dependent or uses fixed-domain bits")
    return [(int(row[0]), int(row[1]), int(row[2])) for row in rows]


def _allowed_positions(cfg: Settings) -> list[int]:
    if cfg.input_domain == "permutation":
        words = range(5)
    elif cfg.input_domain == "ascon128":
        words = range(1, 5)
    else:
        words = range(1, 4)
    return [64 * word + bit for word in words for bit in range(64)]


def _random_domain_value(rng: random.Random, cfg: Settings) -> int:
    if cfg.input_domain == "permutation":
        words = range(5)
    elif cfg.input_domain == "ascon128":
        words = range(1, 5)
    else:
        words = range(1, 4)
    return sum(rng.getrandbits(64) << (64 * word) for word in words)


def _int_to_words(value: int, cfg: Settings) -> tuple[int, ...]:
    words = tuple((value >> (64 * word)) & MASK64 for word in range(5))
    if cfg.input_domain == "ascon128_equal":
        words = (*words[:4], words[3])
    return words


def _conditioned_sampler(basis: Sequence[int], assignment: int, cfg: Settings):
    allowed = _allowed_positions(cfg)
    equations = _rref_equations(basis, allowed)
    pivots = [pivot for _, _, pivot in equations]

    def sample(rng: random.Random) -> tuple[int, ...]:
        value = _random_domain_value(rng, cfg)
        for pivot in pivots:
            value &= ~(1 << pivot)
        for row, rhs_transform, pivot in equations:
            rhs = (rhs_transform & assignment).bit_count() & 1
            other = ((row ^ (1 << pivot)) & value).bit_count() & 1
            if rhs ^ other:
                value |= 1 << pivot
        observed = sum(
            (((row & value).bit_count() & 1) << index)
            for index, row in enumerate(basis)
        )
        if observed != assignment:
            raise AssertionError("conditioned sampler violated a parity equation")
        return _int_to_words(value, cfg)

    return sample


def _experiment_stats(signed_sum: int, sample_count: int, seconds: float) -> dict:
    correlation = signed_sum / sample_count
    se = math.sqrt(max(0.0, 1.0 - correlation * correlation) / sample_count)
    return {
        "sample_count": sample_count,
        "correlation": correlation,
        "weight": -math.log2(abs(correlation)) if correlation else None,
        "standard_error": se,
        "correlation_95_percent_interval": [
            correlation - 1.96 * se,
            correlation + 1.96 * se,
        ],
        "seconds": seconds,
    }


def _run_actual_class(
    basis: Sequence[int], assignment: Sequence[int] | None, cfg: Settings, seed: int
) -> dict:
    sample_count = 1 << cfg.actual_sample_log2
    rng = random.Random(seed)
    if assignment is None:
        sampler = lambda source: _int_to_words(_random_domain_value(source, cfg), cfg)
    else:
        assignment_int = sum(bit << index for index, bit in enumerate(assignment))
        sampler = _conditioned_sampler(basis, assignment_int, cfg)
    signed_sum = 0
    started = time.perf_counter()
    for _ in range(sample_count):
        signed_sum += actual_sign(sampler(rng), cfg)
    result = _experiment_stats(signed_sum, sample_count, time.perf_counter() - started)
    result["assignment"] = list(assignment) if assignment is not None else None
    result["seed"] = seed
    return result


def run_actual_experiments(
    basis: Sequence[int], theory: dict, cfg: Settings
) -> dict:
    maximum = theory["maximum"]["assignment"]
    minimum = theory["minimum"]["assignment"]
    return {
        "source": "fresh_actual_Ascon",
        "sample_log2": cfg.actual_sample_log2,
        "unconstrained": _run_actual_class(
            basis, None, cfg, cfg.actual_random_seed
        ),
        "maximum": _run_actual_class(
            basis, maximum, cfg, cfg.actual_random_seed + 1
        ),
        "minimum": _run_actual_class(
            basis, minimum, cfg, cfg.actual_random_seed + 2
        ),
    }


def verify_reproduction(
    cfg: Settings,
    backend: Backend,
    spectra: Mapping[int, Mapping[int, float]],
    analyses: Mapping[int, dict],
    run_dir: Path,
    *,
    force: bool,
) -> dict:
    checks: dict[str, dict] = {}
    backend.qd.validate_identities()
    checks["Q_DDT_LAT_identities"] = {"passed": True}

    rng = random.Random(358)
    linear_ok = True
    for _ in range(32):
        state = rng.getrandbits(backend.common.STATE_BITS)
        mask = rng.getrandbits(backend.common.STATE_BITS)
        left = (backend.common.forward_linear(state) & mask).bit_count() & 1
        right = (state & backend.common.adjoint_linear(mask)).bit_count() & 1
        linear_ok &= left == right
    checks["linear_adjoint"] = {"passed": linear_ok, "trials": 32}

    if _is_dl1_target(cfg):
        difference = backend.common.words_to_int(cfg.input_difference_words)
        counts = [
            backend.characteristics.complete_characteristic_count(difference, p)
            for p in (0, 1, 2)
        ]
        checks["DL1_characteristic_counts"] = {
            "passed": counts == [1, 8, 1_272_384],
            "observed": counts,
            "expected": [1, 8, 1_272_384],
        }

        if 0 in analyses and spectra[0].get(0, 0.0) == 0.5:
            observed = (
                analyses[0]["correlation_distribution"][0],
                analyses[0]["maximum"]["correlation"],
                analyses[0]["minimum"]["correlation"],
            )
            checks["DL1_five_dimensional_theory"] = {
                "passed": observed == (0.375, 0.75, 0.375),
                "observed_first_max_min": observed,
                "expected_first_max_min": [0.375, 0.75, 0.375],
                "U0": spectra[0].get(0, 0.0),
                "U0_passed": spectra[0].get(0, 0.0) == 0.5,
            }
        elif 0 in analyses:
            checks["DL1_five_dimensional_theory"] = {
                "passed": True,
                "applicable": False,
                "reason": (
                    "fresh bounded p0 capacity has not reached the validated cap512 root"
                ),
                "observed_U0": spectra[0].get(0, 0.0),
            }

        if 0 in spectra and 2 in spectra and cfg.p2_mode == "collapsed_complete":
            checks["p2_collapsed_equals_p0"] = {
                "passed": dict(spectra[2]) == dict(spectra[0])
            }

    passed = all(item.get("passed", False) for item in checks.values())
    output = {"passed": passed, "checks": checks}
    write_json(run_dir / "verification.json", output)
    return output


def _power(value: float | None) -> str:
    if value is None:
        return "—"
    if value == 0:
        return "0"
    sign = "-" if value < 0 else "+"
    return f"{sign}{abs(value):.12g} (|C|=2^-{-math.log2(abs(value)):.6f})"


def _summary_markdown(summary: dict) -> str:
    rows = []
    for p in sorted(summary["prefix_results"], key=int):
        record = summary["prefix_results"][p]
        theory = record["analysis"]
        rows.append(
            "| {p} | {mode} | {u0} | {minimum} | {maximum} | {coverage}/{size} |".format(
                p=p,
                mode=record["mode"],
                u0=_power(record["joint_zero_label_coefficient"]),
                minimum=_power(theory["minimum"]["correlation"]),
                maximum=_power(theory["maximum"]["correlation"]),
                coverage=theory["span_coordinates_present"],
                size=theory["span_size"],
            )
        )
    experiment = summary.get("actual_experiments")
    experiment_text = "Not run."
    if experiment:
        experiment_text = (
            f"Source: `{experiment['source']}`. Unconstrained: "
            f"{_power(experiment['unconstrained']['correlation'])}; "
            f"theoretical maximum class: {_power(experiment['maximum']['correlation'])}; "
            f"theoretical minimum class: {_power(experiment['minimum']['correlation'])}."
        )
    basis_lines = "\n".join(
        f"- B{index} = `{item['expression']}`"
        for index, item in enumerate(summary["basis"]["terms"])
    ) or "- Empty basis"
    return f"""# {summary['configuration']['description']}: integrated results

Configuration fingerprint: `{summary['configuration_fingerprint']}`

## Theoretical results

| DDT prefix rounds | Mode | U=0 | Minimum | Maximum | Span coverage |
|---:|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Input-mask basis

Mode: `{summary['basis']['selection']['mode']}`.

{basis_lines}

## Actual Ascon experiments

{experiment_text}

## Verification

Overall status: `{summary['verification']['passed']}`. See `verification.json` for details.

## Scope

- The root spectrum and FWHT use a capacity-bounded model.
- The `auto` basis mode selects candidates; it does not validate constraints.
- Actual experiments observe the state after the final S-box without the final linear layer.
"""


def run_all(
    *,
    force: bool = False,
    run_experiments: bool | None = None,
    run_verification: bool = True,
) -> dict:
    cfg = settings_from_module()
    backend = load_backend(cfg)
    run_dir = run_directory(cfg)
    run_dir.mkdir(parents=True, exist_ok=True)
    experiment_enabled = (
        cfg.run_actual_experiments
        if run_experiments is None
        else bool(run_experiments)
    )
    started = time.perf_counter()
    spectra: dict[int, dict[int, float]] = {}
    results: dict[int, dict] = {}

    need_p0 = 0 in cfg.prefix_rounds or (
        2 in cfg.prefix_rounds and cfg.p2_mode == "collapsed_complete"
    )
    if need_p0:
        spectra[0], results[0] = run_p0(
            cfg,
            backend,
            run_dir,
            force=force,
        )
    if 1 in cfg.prefix_rounds:
        spectra[1], results[1] = run_explicit_prefix(
            1, cfg, backend, run_dir, force=force
        )
    if 2 in cfg.prefix_rounds:
        if cfg.p2_mode == "collapsed_complete":
            spectra[2], results[2] = run_p2_collapsed(
                spectra[0], cfg, backend, run_dir
            )
        else:
            spectra[2], results[2] = run_explicit_prefix(
                2, cfg, backend, run_dir, force=force
            )

    requested = {p: spectra[p] for p in cfg.prefix_rounds}
    basis_source = spectra[0] if 0 in spectra else next(iter(requested.values()))
    basis, basis_selection = choose_basis(cfg, backend, basis_source)
    analyses = {
        p: analyze_spectrum(spectrum, basis, cfg, backend)
        for p, spectrum in requested.items()
    }
    for p in requested:
        results[p]["analysis"] = analyses[p]
        write_json(run_dir / f"p{p}" / "result.json", results[p])

    experiments = None
    if experiment_enabled:
        theory_key = 0 if 0 in analyses else next(iter(analyses))
        experiments = run_actual_experiments(
            basis,
            analyses[theory_key],
            cfg,
        )
        write_json(run_dir / "actual_experiments.json", experiments)

    verification = (
        verify_reproduction(
            cfg,
            backend,
            spectra,
            analyses,
            run_dir,
            force=force,
        )
        if run_verification
        else {"passed": None, "checks": {}}
    )
    summary = {
        "version": 1,
        "configuration": _jsonable_settings(cfg),
        "configuration_fingerprint": fingerprint(cfg),
        "run_directory": str(run_dir),
        "basis": {
            "selection": basis_selection,
            "dimension": len(basis),
            "terms": [_term(mask, basis_source.get(mask, 0.0), cfg, backend) for mask in basis],
        },
        "prefix_results": {str(p): results[p] for p in cfg.prefix_rounds},
        "actual_experiments": experiments,
        "verification": verification,
        "total_seconds": time.perf_counter() - started,
        "warnings": [
            basis_selection["warning"],
            (
                "Changing only the distinguisher runs the pipeline, but a new "
                "basis and bounded capacities are not automatically validated."
            ),
        ],
    }
    summary["warnings"] = [item for item in summary["warnings"] if item]
    overview = {
        "version": 1,
        "distinguisher": cfg.description,
        "configuration_fingerprint": fingerprint(cfg),
        "basis": {
            "mode": basis_selection["mode"],
            "dimension": len(basis),
            "expressions": [
                _term(mask, basis_source.get(mask, 0.0), cfg, backend)["expression"]
                for mask in basis
            ],
        },
        "theoretical_results": [
            {
                "DDT_prefix_rounds": p,
                "mode": results[p]["mode"],
                "U0_correlation": results[p]["joint_zero_label_coefficient"],
                "U0_weight": results[p]["joint_zero_label_weight"],
                "minimum": analyses[p]["minimum"],
                "maximum": analyses[p]["maximum"],
                "maximum_absolute": analyses[p]["maximum_absolute"],
                "span_present": analyses[p]["span_coordinates_present"],
                "span_size": analyses[p]["span_size"],
                "root_terms": results[p]["root_term_count"],
            }
            for p in cfg.prefix_rounds
        ],
        "actual_experiments": (
            {
                "source": experiments["source"],
                "unconstrained": experiments["unconstrained"]["correlation"],
                "theoretical_maximum_class": experiments["maximum"]["correlation"],
                "theoretical_minimum_class": experiments["minimum"]["correlation"],
            }
            if experiments
            else None
        ),
        "verification_passed": verification["passed"],
        "files": {
            "full_summary": str(run_dir / "summary.json"),
            "readable_report": str(run_dir / "summary.md"),
            "verification": str(run_dir / "verification.json"),
            "actual_experiments": (
                str(run_dir / "actual_experiments.json") if experiments else None
            ),
        },
    }
    summary["overview_path"] = str(run_dir / "overview.json")
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "overview.json", overview)
    (run_dir / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    write_json(
        OUTPUT_ROOT / "latest.json",
        {
            "configuration_fingerprint": fingerprint(cfg),
            "run_directory": str(run_dir),
            "summary": str(run_dir / "summary.json"),
            "overview": str(run_dir / "overview.json"),
            "verification_passed": verification["passed"],
        },
    )
    return summary
