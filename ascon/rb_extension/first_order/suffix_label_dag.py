"""Python interface for complete-V suffix DAG / bounded-U propagation."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import cipher_config as config
import round_based as rb


SOURCE = rb.HERE / "suffix_label_dag_engine.c"
BINARY = Path(
    os.environ.get("SUFFIX_LABEL_BINARY", rb.HERE / "suffix_label_dag_engine.exe")
)
OUTPUT_PATH = rb.OUTPUT_DIR / "suffix_label_dag_sweep.json"
MAX_TERM_CAP = 1024


def build(force: bool = False) -> Path:
    if (
        not force
        and BINARY.exists()
        and BINARY.stat().st_mtime >= SOURCE.stat().st_mtime
    ):
        return BINARY
    command = rb._find_compiler() + [
        "-std=c99",
        "-O3",
        "-Wall",
        "-Wextra",
        str(SOURCE),
        "-o",
        str(BINARY),
        "-lm",
    ]
    result = subprocess.run(command, cwd=rb.HERE, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return BINARY


def evaluate(
    term_cap: int,
    rounds: int = config.ROUNDS,
    target_masks: list[tuple[int, ...]] | None = None,
    root_dump_path: Path | None = None,
    coset_basis: list[tuple[int, ...]] | None = None,
    *,
    begin_round: int = config.BEGIN_ROUND,
    domain: str = config.INPUT_DOMAIN,
    iv: int = config.IV,
    difference: tuple[int, ...] = config.INPUT_DIFFERENCE_WORDS,
    output_mask: tuple[int, ...] = config.OUTPUT_MASK_WORDS,
) -> dict:
    if not 1 <= term_cap <= MAX_TERM_CAP:
        raise ValueError(f"term_cap must be in [1,{MAX_TERM_CAP}]")
    difference = rb.validate_words(difference, "input difference")
    output_mask = rb.validate_words(output_mask, "output mask")
    if domain not in rb.DOMAIN_IDS:
        raise ValueError(f"unsupported input domain: {domain}")
    target_masks = [] if target_masks is None else [
        rb.validate_input_mask(mask, domain) for mask in target_masks
    ]
    if len(target_masks) > 4096:
        raise ValueError("the engine accepts at most 4096 target masks")
    coset_basis = [] if coset_basis is None else [
        rb.validate_input_mask(mask, domain) for mask in coset_basis
    ]
    if len(coset_basis) > 20:
        raise ValueError("the engine accepts at most 20 coset basis masks")
    if rb.gf2_rank(coset_basis) != len(coset_basis):
        raise ValueError("coset basis must be GF(2)-independent")
    request = [
        f"{rounds} {begin_round} "
        f"{rb.DOMAIN_IDS[domain]} {iv:x} {term_cap}",
        " ".join(f"{word:x}" for word in difference),
        " ".join(f"{word:x}" for word in output_mask),
        str(len(target_masks)),
        *(" ".join(f"{word:x}" for word in mask) for mask in target_masks),
        str(len(coset_basis)),
        *(" ".join(f"{word:x}" for word in mask) for mask in coset_basis),
    ]
    started = time.perf_counter()
    environment = os.environ.copy()
    if root_dump_path is not None:
        root_dump_path = Path(root_dump_path).resolve()
        root_dump_path.parent.mkdir(parents=True, exist_ok=True)
        environment["SUFFIX_LABEL_ROOT_DUMP"] = str(root_dump_path)
    else:
        environment.pop("SUFFIX_LABEL_ROOT_DUMP", None)
    process = subprocess.run(
        [str(build())],
        input="\n".join(request) + "\n",
        cwd=rb.HERE,
        text=True,
        capture_output=True,
        env=environment,
    )
    if process.returncode:
        raise RuntimeError(process.stderr or process.stdout)
    elapsed = time.perf_counter() - started
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if not lines or not lines[0].startswith("terms "):
        raise RuntimeError(f"malformed engine output: {process.stdout!r}")
    count = int(lines[0].split()[1])
    terms = []
    for line in lines[1 : 1 + count]:
        fields = line.split()
        mask = tuple(int(value, 16) for value in fields[1:])
        terms.append(
            {
                "coefficient": float(fields[0]),
                "absolute_coefficient": abs(float(fields[0])),
                "mask": mask,
                "mask_words": rb.mask_to_hex(mask),
                "expression": rb.mask_to_expression(mask),
            }
        )
    node_fields = lines[1 + count].split()
    if node_fields[0] != "nodes":
        raise RuntimeError("missing node diagnostics")
    node_values = list(map(int, node_fields[1:]))
    sbox_counts = node_values[:rounds]
    linear_counts = node_values[rounds : rounds + max(0, rounds - 1)]
    xor_products = node_values[-3]
    pruned_polynomials = node_values[-2]
    maximum_unpruned_terms = node_values[-1]
    target_header = lines[2 + count].split()
    if target_header != ["targets", str(len(target_masks))]:
        raise RuntimeError("missing target diagnostics")
    target_terms = []
    for index, line in enumerate(
        lines[3 + count : 3 + count + len(target_masks)]
    ):
        fields = line.split()
        mask = tuple(int(value, 16) for value in fields[2:])
        if mask != target_masks[index]:
            raise RuntimeError("target mask order mismatch")
        target_terms.append(
            {
                "index": index,
                "present_in_root_hash": bool(int(fields[0])),
                "coefficient": float(fields[1]),
                "mask_words": rb.mask_to_hex(mask),
                "expression": rb.mask_to_expression(mask),
            }
        )
    root_line_index = 3 + count + len(target_masks)
    root_line = lines[root_line_index].split()
    if len(root_line) != 2 or root_line[0] != "root_hash_terms":
        raise RuntimeError("missing root hash diagnostics")
    root_hash_term_count = int(root_line[1])
    zero = next(
        (item["coefficient"] for item in terms if item["mask"] == rb.ZERO_MASK),
        0.0,
    )
    return {
        "rounds": rounds,
        "begin_round": begin_round,
        "domain": domain,
        "iv": f"0x{iv:016x}",
        "difference_words": rb.mask_to_hex(difference),
        "output_mask_words": rb.mask_to_hex(output_mask),
        "term_cap": term_cap,
        "coefficient_status": "exact" if pruned_polynomials == 0 else "bounded",
        "term_count": len(terms),
        "zero_label_coefficient": zero,
        "sbox_nodes_by_round": sbox_counts,
        "linear_nodes_by_round": linear_counts,
        "xor_product_operations": xor_products,
        "pruned_polynomial_count": pruned_polynomials,
        "maximum_unpruned_term_count": maximum_unpruned_terms,
        "root_hash_term_count_before_root_pruning": root_hash_term_count,
        "root_dump_path": str(root_dump_path) if root_dump_path else None,
        "coset_basis_dimension": len(coset_basis),
        "coset_basis": [rb.mask_to_hex(mask) for mask in coset_basis],
        "target_terms": target_terms,
        "seconds": elapsed,
        "terms": [
            {key: value for key, value in item.items() if key != "mask"}
            for item in terms
        ],
        "_masks": {item["mask"] for item in terms},
    }


def run_sweep(caps: list[int], rounds: int, append: bool = False) -> dict:
    results = []
    for cap in caps:
        print(f"run rounds={rounds}, U-term-cap={cap}", flush=True)
        item = evaluate(cap, rounds)
        results.append(item)
        print(
            f"  terms={item['term_count']}, "
            f"zero={item['zero_label_coefficient']:+.12g}, "
            f"time={item['seconds']:.3f}s"
        )
        for term in item["terms"][:8]:
            print(
                f"    {term['expression']}: {term['coefficient']:+.12g}"
            )
    for item in results:
        del item["_masks"]
    if append and OUTPUT_PATH.exists():
        previous = rb.read_json(OUTPUT_PATH)
        by_cap = {item["term_cap"]: item for item in previous["configurations"]}
        by_cap.update({item["term_cap"]: item for item in results})
        results = [by_cap[cap] for cap in sorted(by_cap)]

    result_masks = [
        {
            parse_mask
            for term in item["terms"]
            for parse_mask in [tuple(int(word, 16) for word in term["mask_words"])]
        }
        for item in results
    ]
    stability = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            left, right = result_masks[i], result_masks[j]
            union = left | right
            stability.append(
                {
                    "left_cap": results[i]["term_cap"],
                    "right_cap": results[j]["term_cap"],
                    "intersection": len(left & right),
                    "union": len(union),
                    "jaccard": len(left & right) / len(union) if union else 1.0,
                }
            )

    output = {
        "version": 1,
        "method": (
            "all reachable V-DAG nodes; exact local first-S-box U labels; "
            "XOR convolution at products; bounded U maps at intermediate nodes"
        ),
        "no_V_pruning": True,
        "configurations": results,
        "candidate_set_stability": stability,
    }
    rb.write_json(OUTPUT_PATH, output)
    print(f"saved {OUTPUT_PATH}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps", default="4,8,16,32,64")
    parser.add_argument("--rounds", type=int, default=config.ROUNDS)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="merge new capacities into the existing sweep instead of replacing it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(force=args.force_build)
    run_sweep(
        [int(value) for value in args.caps.split(",")],
        args.rounds,
        append=args.append,
    )


if __name__ == "__main__":
    main()
