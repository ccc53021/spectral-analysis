"""Bounded exact histogram probes; no claim of complete all-key enumeration."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import permutations
import json
from pathlib import Path
import random
import time
import traceback

from .outer_model import OuterModel

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    state = {"state": "running", "started_unix": time.time(), "completed": 0, "global_enumeration_complete": False}
    status_file = args.output / "histogram_probe_status.json"
    status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    try:
        model = OuterModel(args.output)
        cases = [(index, 0, 0, 0) for index in range(66)]
        cases += [(0, 0, 0, index) for index in range(1, 4)]
        rng = random.Random(202609150462)
        for _ in range(8):
            triple = tuple(rng.sample(range(66), 3))
            cases += [tuple(value) + (0,) for value in permutations(triple)]
        cases = list(dict.fromkeys(cases))
        rows, signatures = [], defaultdict(list)
        for index, classes in enumerate(cases):
            began = time.perf_counter()
            hist = model.template_histogram(classes)
            signature = hashlib.sha256(json.dumps(hist["histogram"], separators=(",", ":")).encode()).hexdigest()
            signatures[signature].append(classes)
            row = {"classes": classes, "histogram_sha256": signature, "distinct_values": len(hist["histogram"]), "maximum": hist["histogram"][-1], "minimum": hist["histogram"][0], "denominator": hist["denominator"], "elapsed_seconds": time.perf_counter() - began}
            rows.append(row)
            print(json.dumps(row), flush=True)
            state.update(completed=index + 1, total=len(cases), unique_histograms=len(signatures), elapsed_seconds=time.time() - state["started_unix"])
            status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        result = {"scope": "Bounded complete-connector histogram probes only, not all templates.", "cases": rows, "groups": [list(group) for group in signatures.values()], "unique_histograms": len(signatures), "case_count": len(cases), "elapsed_seconds": time.time() - state["started_unix"], "global_enumeration_complete": False}
        (args.output / "histogram_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        state["state"] = "complete"
        status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    except BaseException as error:
        state.update(state="failed", error=repr(error), traceback=traceback.format_exc())
        status_file.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
