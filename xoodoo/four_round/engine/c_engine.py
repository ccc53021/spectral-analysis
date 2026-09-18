"""Python interface to the bounded global-U Xoodoo DAG engine."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Mapping, Sequence

from . import common
from .config import Distinguisher


SOURCE = common.HERE / "u_dag_engine.c"
BINARY = common.HERE / "u_dag_engine.exe"
MAX_CAPACITY = 4096


def build(force: bool = False) -> Path:
    if (
        not force
        and BINARY.exists()
        and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime
    ):
        return BINARY
    command = common.find_compiler() + [
        "-std=c99",
        "-O3",
        "-Wall",
        "-Wextra",
        str(SOURCE),
        "-o",
        str(BINARY),
        "-lm",
    ]
    result = subprocess.run(command, cwd=common.HERE, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return BINARY


def evaluate(
    config: Distinguisher,
    capacity: int,
    *,
    target_masks: Sequence[Sequence[int]] = (),
    coset_basis: Sequence[Sequence[int]] = (),
    root_dump_path: Path | None = None,
) -> dict:
    return evaluate_masks(
        config.rounds,
        common.input_difference(config),
        common.output_mask(config),
        capacity,
        configuration=common.config_metadata(config),
        target_masks=target_masks,
        coset_basis=coset_basis,
        root_dump_path=root_dump_path,
    )


def evaluate_masks(
    rounds: int,
    difference: Sequence[int],
    output_mask: Sequence[int],
    capacity: int,
    *,
    configuration: Mapping[str, object] | None = None,
    target_masks: Sequence[Sequence[int]] = (),
    coset_basis: Sequence[Sequence[int]] = (),
    root_dump_path: Path | None = None,
) -> dict:
    """Evaluate an arbitrary chi-boundary difference/output-mask pair.

    Unlike :func:`evaluate`, this entry point is not restricted to a single
    active output column.  It is used by the full-round boundary adapter after
    pulling a sparse full-output mask through the final ``rho_east`` layer.
    """
    rounds = int(rounds)
    if not 1 <= rounds <= 12:
        raise ValueError("rounds must be in [1,12]")
    if not 1 <= capacity <= MAX_CAPACITY:
        raise ValueError(f"capacity must be in [1,{MAX_CAPACITY}]")
    difference = common.validate_mask(difference)
    output_mask = common.validate_mask(output_mask)
    target_masks = [common.validate_mask(mask) for mask in target_masks]
    coset_basis = [common.validate_mask(mask) for mask in coset_basis]
    if len(target_masks) > 4096:
        raise ValueError("at most 4096 target masks are supported")
    if len(coset_basis) > 12:
        raise ValueError("at most 12 coset basis masks are supported")
    if common.gf2_rank(coset_basis) != len(coset_basis):
        raise ValueError("coset basis must be independent")

    request = [
        f"{rounds} {capacity}",
        " ".join(f"{word:x}" for word in difference),
        " ".join(f"{word:x}" for word in output_mask),
        str(len(target_masks)),
        *(" ".join(f"{word:x}" for word in mask) for mask in target_masks),
        str(len(coset_basis)),
        *(" ".join(f"{word:x}" for word in mask) for mask in coset_basis),
    ]
    environment = os.environ.copy()
    if root_dump_path is not None:
        root_dump_path = Path(root_dump_path).resolve()
        root_dump_path.parent.mkdir(parents=True, exist_ok=True)
        environment["XOODOO_U_ROOT_DUMP"] = str(root_dump_path)
    else:
        environment.pop("XOODOO_U_ROOT_DUMP", None)

    started = time.perf_counter()
    process = subprocess.run(
        [str(build())],
        input="\n".join(request) + "\n",
        cwd=common.HERE,
        text=True,
        capture_output=True,
        env=environment,
    )
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(process.stderr or process.stdout)
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("terms "):
        raise RuntimeError(f"bad engine output: {process.stdout!r}")

    term_count = int(lines[0].split()[1])
    terms = []
    for line in lines[1 : 1 + term_count]:
        fields = line.split()
        coefficient = float(fields[0])
        mask = tuple(int(word, 16) for word in fields[1:])
        terms.append(
            {
                "coefficient": coefficient,
                "absolute_coefficient": abs(coefficient),
                "weight": common.weight(coefficient),
                "mask_words": common.mask_to_hex(mask),
                "expression": common.mask_to_expression(mask),
                "_mask": mask,
            }
        )

    node_fields = lines[1 + term_count].split()
    if node_fields[0] != "nodes":
        raise RuntimeError("missing node diagnostics")
    node_values = [int(item) for item in node_fields[1:]]
    sbox_nodes = node_values[:rounds]
    linear_nodes = node_values[rounds : 2 * rounds - 1]
    xor_products, pruned_count, max_unpruned = node_values[-3:]

    target_header = lines[2 + term_count].split()
    if target_header != ["targets", str(len(target_masks))]:
        raise RuntimeError("missing target diagnostics")
    targets = []
    for index, line in enumerate(
        lines[3 + term_count : 3 + term_count + len(target_masks)]
    ):
        fields = line.split()
        mask = tuple(int(word, 16) for word in fields[2:])
        if mask != target_masks[index]:
            raise RuntimeError("target order mismatch")
        coefficient = float(fields[1])
        targets.append(
            {
                "index": index,
                "present_in_root_hash": bool(int(fields[0])),
                "coefficient": coefficient,
                "weight": common.weight(coefficient),
                "mask_words": common.mask_to_hex(mask),
                "expression": common.mask_to_expression(mask),
            }
        )
    root_line = lines[3 + term_count + len(target_masks)].split()
    if root_line[0] != "root_hash_terms":
        raise RuntimeError("missing root hash size")

    zero = next(
        (term["coefficient"] for term in terms if term["_mask"] == common.ZERO_MASK),
        0.0,
    )
    public_terms = [
        {key: value for key, value in term.items() if key != "_mask"}
        for term in terms
    ]
    return {
        "configuration": dict(configuration or {
            "boundary": "chi-input to chi-output",
            "rounds": rounds,
            "input_difference_words": common.mask_to_hex(difference),
            "output_mask_words": common.mask_to_hex(output_mask),
        }),
        "capacity": capacity,
        "coefficient_status": "exact" if pruned_count == 0 else "capacity-bounded",
        "term_count": len(terms),
        "zero_label_coefficient": zero,
        "zero_label_weight": common.weight(zero),
        "sbox_nodes_by_round": sbox_nodes,
        "linear_nodes_by_round": linear_nodes,
        "xor_product_operations": xor_products,
        "pruned_polynomial_count": pruned_count,
        "maximum_unpruned_term_count": max_unpruned,
        "root_hash_term_count_before_root_pruning": int(root_line[1]),
        "root_dump_path": str(root_dump_path) if root_dump_path else None,
        "coset_basis_dimension": len(coset_basis),
        "coset_basis": [common.mask_to_hex(mask) for mask in coset_basis],
        "target_terms": targets,
        "seconds": elapsed,
        "terms": public_terms,
    }
