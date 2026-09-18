"""Export exact compact arrays and 117 complete Python reference histograms."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import struct
import time

import numpy as np

from .outer_model import OuterModel

HERE = Path(__file__).resolve().parent
OUT = HERE / "output/cpp"


def template_id(classes):
    a, b, c, d = classes
    return ((a * 66 + b) * 66 + c) * 4 + d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status_path = OUT / "prepare_status.json"
    status = {"state": "running", "references_complete": 0}
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    model = OuterModel()
    transfer = model.transfer
    proof_bound = np.abs(transfer.weights).copy()
    payload = bytearray(struct.pack("<6I", len(transfer.weights), 1 << 21, 66, 66, 66, 4))
    payload += transfer.indices.astype("<u4").tobytes()
    assert int(np.abs(transfer.weights).max()) < 128
    payload += transfer.weights.astype("i1").tobytes()
    for col in range(4):
        expanded = model.templates[col][:, model.template_locations[col]]
        assert int(np.abs(expanded).max()) < 128
        proof_bound *= np.abs(expanded).max(axis=0)
        payload += expanded.astype("i1").tobytes()
    bound = int(proof_bound.sum())
    assert bound == 46093824 and bound < (1 << 31)
    signature = hashlib.sha256(payload).hexdigest()
    (OUT / "outer_data.bin").write_bytes(b"BLKOUTR1" + signature.encode("ascii") + payload)
    probes = json.loads((HERE / "output/histogram_probe.json").read_text())
    cases = [template_id(row["classes"]) for row in probes["cases"]]
    rng = random.Random(202609150463)
    present = set(cases)
    while len(cases) < 1024:
        candidate = rng.randrange(66 ** 3 * 4)
        if candidate not in present:
            present.add(candidate)
            cases.append(candidate)
    (OUT / "benchmark_cases.bin").write_bytes(b"OUTCASE1" + struct.pack("<I", len(cases)) + np.asarray(cases, dtype="<u4").tobytes())
    with (OUT / "python_reference.bin").open("wb") as handle:
        handle.write(b"OUTREF01" + struct.pack("<I", len(probes["cases"])))
        for index, row in enumerate(probes["cases"]):
            hist = model.template_histogram(tuple(row["classes"]))
            signature_now = hashlib.sha256(json.dumps(hist["histogram"], separators=(",", ":")).encode()).hexdigest()
            assert signature_now == row["histogram_sha256"]
            handle.write(struct.pack("<II", template_id(row["classes"]), len(hist["histogram"])))
            pairs = np.asarray([[v["numerator"], v["count"]] for v in hist["histogram"]], dtype="<u4")
            handle.write(pairs.tobytes())
            status.update(references_complete=index + 1, elapsed_seconds=time.time() - started)
            status_path.write_text(json.dumps(status, indent=2) + "\n")
    metadata = {"data_sha256": signature, "templates": 66 ** 3 * 4, "terms": len(transfer.weights), "connector_dimension": 21, "universal_l1_bound": bound, "integer_safety": "Every intermediate FWHT value is bounded by the L1 sum; the sum of per-term maxima over every template is below 2^26, so signed int32 is safe for all templates.", "probability_normalization": "all live templates have scalar 2^25 and canonical denominator 2^79, so output numerator n represents R=n/2^54", "python_reference_histograms": len(probes["cases"]), "benchmark_cases": len(cases), "random_seed": 202609150463, "elapsed_seconds": time.time() - started}
    (OUT / "data_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    status.update(state="complete", elapsed_seconds=time.time() - started)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
