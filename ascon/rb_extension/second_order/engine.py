"""Bounded C engine for four-message, joint-input-label DAG propagation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

try:
    from .common import HERE, default_config, target_config, validate_config, serializable_config, column
except ImportError:
    from common import HERE, default_config, target_config, validate_config, serializable_config, column

SOURCE = HERE / "engine.c"


def build_engine(force: bool = False) -> Path:
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    binary = HERE / ("engine_" + source_hash[:16] + (".exe" if os.name == "nt" else ""))
    if not force and binary.exists():
        return binary
    candidates = (shutil.which("gcc"), "D:/MinGW/bin/gcc.exe", shutil.which("clang"))
    compiler = next((x for x in candidates if x and Path(x).is_file()), None)
    if compiler is None:
        raise RuntimeError("No existing C99 compiler was found; no toolchain was installed")
    command = [str(compiler), "-std=c99", "-O3", "-Wall", "-Wextra",
               str(SOURCE), "-o", str(binary), "-lm"]
    result = subprocess.run(command, capture_output=True, text=True, cwd=HERE)
    if result.returncode:
        raise RuntimeError("Four-message engine compilation failed:\n" + result.stdout + result.stderr)
    return binary


def evaluate(config=None, *, cap=16, max_nodes=100000, time_limit=30.0,
             max_terms=1048576, rounds=None, force_build=False, klein_cache=False,
             merge_linear_alias_edges=False):
    """Compute joint U labels; never return a partial root on budget failure.

    ``cap=0`` disables label-cap pruning (resource budgets still apply).
    A completed result includes the FULL root accumulator before root-cap
    pruning; intermediate nodes retain at most cap labels and protect U=0.
    ``no_label_pruning`` describes this run, not an unqualified promise of
    practical 20-dimensional spectral completeness at a finite cap.
    """
    cfg = dict(default_config() if config is None else config)
    if isinstance(cfg.get("iv"), str):
        cfg["iv"] = int(cfg["iv"], 0)
    if rounds is not None:
        cfg["rounds"] = int(rounds)
    cfg = validate_config(cfg)
    cap, max_nodes, max_terms = int(cap), int(max_nodes), int(max_terms)
    time_limit = float(time_limit)
    if cap < 0 or max_nodes <= 0 or max_terms <= 0 or time_limit <= 0:
        raise ValueError("cap must be nonnegative and all resource budgets positive")
    active = [col for col in range(64) if column(cfg["output_words"], col)]
    if len(active) > 1:
        raise ValueError("this Table 5 engine supports at most one active output column")
    mode = cfg.get("mode", "active_equal")
    equal_cols = range(64) if mode == "full_equal" else (
        cfg.get("equal_columns", (0, 3)) if mode == "active_equal" else ())
    equal_bits = sum(1 << (63-int(col)) for col in set(equal_cols))
    request = [f"{cfg['rounds']} {cfg.get('begin_round', 0)} {cap} {max_nodes} {time_limit:.17g} {max_terms} {int(bool(klein_cache))} {int(bool(merge_linear_alias_edges))}",
               f"{cfg['iv']:x} {equal_bits:x}"]
    request.extend(" ".join(f"{int(word):x}" for word in cfg[key])
                   for key in ("delta1_words", "delta2_words", "output_words"))
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    binary = build_engine(force_build)
    if not binary.stem.endswith(source_hash[:16]):
        raise RuntimeError("Engine source changed during build selection; retry against a fixed source snapshot")
    started = time.perf_counter()
    try:
        process = subprocess.run([str(binary)], input="\n".join(request)+"\n",
                                 capture_output=True, text=True, cwd=HERE,
                                 timeout=time_limit+15)
    except subprocess.TimeoutExpired:
        result = {"status": "process_timeout", "complete": False, "terms": [],
                  "stats": {"elapsed_seconds": time.perf_counter()-started},
                  "failure_note": "Host timeout killed the process; no root spectrum is available"}
    else:
        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Engine failed without a valid diagnostic (exit {process.returncode}): "
                               + process.stderr + process.stdout) from exc
        if process.returncode not in (0, 2):
            raise RuntimeError(f"Engine returned unexpected exit {process.returncode}: {process.stderr}")
        if not result["complete"] and result["terms"]:
            raise RuntimeError("An incomplete engine result must not contain a partial root spectrum")
    result.update(config=serializable_config(cfg),
                  source_sha256=source_hash,
                  engine_binary=str(binary),
                  budgets={"cap": cap, "max_nodes": max_nodes,
                           "time_limit": time_limit, "max_terms": max_terms},
                  optimizations={"klein_cache": bool(klein_cache),
                                 "merge_linear_alias_edges": bool(merge_linear_alias_edges)},
                  implementation="four physical-message masks; sparse reachable DAG; joint U XOR convolution",
                  spectrum_semantics="complete root of the capacity-limited run; missing labels are not certified zeros when intermediate pruning occurred")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("5r", "6r"), default="5r")
    parser.add_argument("--rounds", type=int, help="low-round override of the selected target, not a target selector")
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=100000)
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--max-terms", type=int, default=1048576)
    parser.add_argument("--klein-cache", action="store_true", help="phase-aware translation cache with shared label arrays")
    parser.add_argument("--merge-linear-alias-edges", action="store_true", help="sum equal linear-child dyadic LAT weights before recursion")
    args = parser.parse_args()
    result = evaluate(target_config(args.target), rounds=args.rounds, cap=args.cap, max_nodes=args.max_nodes,
                      time_limit=args.time_limit, max_terms=args.max_terms, klein_cache=args.klein_cache,
                      merge_linear_alias_edges=args.merge_linear_alias_edges)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["complete"] else 2)


if __name__ == "__main__":
    main()
