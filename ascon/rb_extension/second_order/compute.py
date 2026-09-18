"""Five-round unrestricted full-U propagation.

This is the original joint-input engine, not the six-round route-C split.
Missing coordinates in a pruned spectrum are never treated as known zeros.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from common import HERE, ZERO, serializable_config, target_config
from engine import evaluate
from spectrum import coefficient_map, independent_basis


def pruning_count(result):
    stats = result.get("stats", result.get("diagnostics", {}))
    for key in ("pruned_polynomials", "pruned_polynomial_count", "pruned_maps", "pruned"):
        if key in stats:
            return int(stats[key])
    return None


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def unrestricted_config():
    cfg = target_config("5r")
    cfg.update(mode="unrestricted", equal_columns=())
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--caps", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--time-limit", type=float, default=180)
    parser.add_argument("--max-nodes", type=int, default=2000000)
    parser.add_argument("--max-terms", type=int, default=1048576)
    parser.add_argument("--without-optimizations", action="store_true")
    args = parser.parse_args()
    if not args.caps or min(args.caps) < 1 or len(set(args.caps)) != len(args.caps):
        parser.error("give distinct positive capacities")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = unrestricted_config()
    request = dict(config=serializable_config(cfg),
                   arguments={**vars(args), "output": str(args.output.resolve())},
                   method="five complete rounds, physical four-message masks, joint original-input U; no scalar suffix",
                   zero_policy="absent coefficients after pruning are unknown, not certified zeros")
    save_new(args.output / "request.json", request)
    for cap in args.caps:
        path = args.output / f"cap_{cap}.json"
        if path.exists():
            print(f"PRESERVED {path}; use a new output directory to rerun", flush=True)
            continue
        print(f"START H{cap} {datetime.now(timezone.utc).isoformat()}", flush=True)
        start = time.perf_counter()
        result = evaluate(cfg, cap=cap, max_nodes=args.max_nodes, time_limit=args.time_limit,
                          max_terms=args.max_terms, klein_cache=not args.without_optimizations,
                          merge_linear_alias_edges=not args.without_optimizations)
        result["host_wall_seconds"] = time.perf_counter() - start
        result["request_file"] = "request.json"
        save_new(path, result)
        summary = dict(cap=cap, complete=result["complete"], status=result["status"],
                       host_wall_seconds=result["host_wall_seconds"], stats=result.get("stats"),
                       pruned_polynomials=pruning_count(result))
        if result["complete"]:
            coefficients = coefficient_map(result)
            summary.update(root_terms=len(coefficients), zero_frequency=coefficients.get(ZERO),
                           root_support_rank=len(independent_basis(coefficients)))
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
