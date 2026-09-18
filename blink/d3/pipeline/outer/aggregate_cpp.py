"""Exact shared-six aggregation of checkpointed complete connector histograms.

The raw numerator n always means R=n/2^54. This program determines a
common power-of-two factor from every supplied histogram before choosing
the compact accumulator axis; the divisor is never guessed. Partial
benchmarks are explicitly marked incomplete and are refused by the final
distribution join.
"""

from __future__ import annotations

import argparse
import json
import mmap
from pathlib import Path
import struct
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
TOTAL_TEMPLATES = 66 ** 3 * 4
ROW_COSETS = 1 << 46
KNOWN_ZERO_J_COSETS = 2021 * (1 << 35)
STAT_DTYPE = np.dtype([("id", "<u4"), ("lower", "<i8"), ("minimum", "<u4"), ("minimum_nonzero", "<u4"), ("maximum", "<u4"), ("zero_connector_count", "<u4"), ("common_power2", "<u4"), ("elapsed_ns", "<u8")])


def records(paths, headers_only=False):
    for path in paths:
        with path.open("rb") as handle:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
            magic = mapped[:8]
            assert magic in (b"OUTCHK01", b"OUTCHK02")
            count, = struct.unpack_from("<I", mapped, 8)
            offset = 12
            for _ in range(count):
                template, lower, minimum, maximum, shift, size = struct.unpack_from("<IqIIII", mapped, offset)
                offset += 28
                elapsed_ns = struct.unpack_from("<Q", mapped, offset)[0] if magic == b"OUTCHK02" else 0
                offset += 8 if magic == b"OUTCHK02" else 0
                histogram = None if headers_only else np.frombuffer(mapped, dtype="<u4", count=2 * size, offset=offset).reshape(-1, 2).copy()
                offset += 8 * size
                yield (template, lower, minimum, maximum, shift, elapsed_ns, histogram)
            assert offset == len(mapped), (str(path), offset, len(mapped))
            mapped.close()


def column_weights():
    data = json.loads((HERE / "output/outer_conditional_factors.json").read_text())
    result = []
    for col, dimension in enumerate((66, 66, 66, 4)):
        array = np.zeros((dimension, 8 if col < 2 else 1), dtype=np.uint64)
        for label, group in enumerate(data["local_factors"][col]["template_key_counts_by_shared_label"]):
            for template, count in group["nonzero_class_key_counts"].items():
                array[int(template), label] = count
        result.append(array)
    return result


def weights_for(template, weights):
    d = template % 4
    c = (template // 4) % 66
    b = (template // (4 * 66)) % 66
    a = template // (4 * 66 * 66)
    values = (weights[0][a, :, None] * weights[1][b, None, :] * weights[2][c, 0] * weights[3][d, 0]).ravel(order="F")
    labels = np.flatnonzero(values)
    return labels, values[labels]


def write_status(path, **values):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def aggregate(run_dir, output, allow_partial=False):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "aggregation_status.json"
    run_status = json.loads((run_dir / "status.json").read_text())
    assert run_status["state"] == "complete", "Kernel must finish before authoritative aggregation"
    paths = sorted(run_dir.glob("chunk_*.bin"), key=lambda p: int(p.stem.split("_")[1]))
    write_status(status_path, state="running", phase="scan", complete=False, elapsed_seconds=0)
    ids = []
    shift = 31
    maximum = 0
    for template, lower, minimum, largest, power, elapsed, _ in records(paths, headers_only=True):
        ids.append(template)
        shift = min(shift, power)
        maximum = max(maximum, largest)
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated template in input chunks")
    complete = len(ids) == TOTAL_TEMPLATES and min(ids) == 0 and max(ids) == TOTAL_TEMPLATES - 1
    if not complete and not allow_partial:
        raise ValueError("Incomplete template coverage; no global statistics emitted")
    axis_length = (maximum >> shift) + 1
    required_bytes = 64 * axis_length * 8
    if required_bytes > (1 << 30):
        raise MemoryError("Exact compact accumulator exceeds the explicit 1 GiB guard")
    counts = np.zeros((64, axis_length), dtype=np.uint64)
    counts[:, 0] = KNOWN_ZERO_J_COSETS
    live_key_counts = np.zeros(64, dtype=np.uint64)
    weights = column_weights()
    template_stats = np.empty(len(ids), dtype=STAT_DTYPE)
    extrema = [{"maximum_raw": -1, "minimum_nonzero_raw": None, "maximum_template_ids": [], "minimum_nonzero_template_ids": []} for _ in range(64)]
    begin_accumulate = time.perf_counter()
    for index, record in enumerate(records(paths)):
        template, lower, minimum, largest, power, elapsed, histogram = record
        numerators = histogram[:, 0].astype(np.uint64)
        frequencies = histogram[:, 1].astype(np.uint64)
        assert int(frequencies.sum()) == 1 << 21
        assert np.all(numerators[1:] > numerators[:-1])
        assert int(numerators[0]) == minimum and int(numerators[-1]) == largest
        assert np.all((numerators & ((1 << shift) - 1)) == 0)
        locations = (numerators >> shift).astype(np.intp)
        labels, multiplicities = weights_for(template, weights)
        counts[np.ix_(labels, locations)] += multiplicities[:, None] * frequencies[None, :]
        live_key_counts[labels] += multiplicities
        positive = numerators[numerators > 0]
        min_positive = int(positive[0]) if len(positive) else 0
        zero_count = int(frequencies[0]) if minimum == 0 else 0
        template_stats[index] = (template, lower, minimum, min_positive, largest, zero_count, power, elapsed)
        for label in labels:
            summary = extrema[int(label)]
            if largest > summary["maximum_raw"]:
                summary["maximum_raw"] = largest
                summary["maximum_template_ids"] = [template]
            elif largest == summary["maximum_raw"]:
                summary["maximum_template_ids"].append(template)
            previous = summary["minimum_nonzero_raw"]
            if min_positive and (previous is None or min_positive < previous):
                summary["minimum_nonzero_raw"] = min_positive
                summary["minimum_nonzero_template_ids"] = [template]
            elif min_positive and min_positive == previous:
                summary["minimum_nonzero_template_ids"].append(template)
        if (index + 1) % 4096 == 0:
            elapsed_total = time.perf_counter() - started
            write_status(status_path, state="running", phase="accumulate", complete=False, templates_processed=index + 1, templates_total=len(ids), elapsed_seconds=elapsed_total, eta_seconds=elapsed_total * (len(ids) - index - 1) / (index + 1), accumulator_bytes=required_bytes)
    accumulation_seconds = time.perf_counter() - begin_accumulate
    expected_totals = np.uint64(KNOWN_ZERO_J_COSETS) + (live_key_counts << np.uint64(21))
    assert np.array_equal(counts.sum(axis=1, dtype=np.uint64), expected_totals)
    if complete:
        assert np.all(expected_totals == ROW_COSETS)
    selected = np.flatnonzero(np.any(counts, axis=0))
    numerators = (selected.astype(np.uint64) << np.uint64(shift)).astype(np.uint32)
    counts = counts[:, selected]
    rows = []
    for syndrome, row in enumerate(counts):
        nonzero_positions = np.flatnonzero((numerators > 0) & (row > 0))
        first_moment = sum(int(n) * int(c) for n, c in zip(numerators, row))
        if complete:
            assert first_moment == 3 * (1 << 59)
        record = {"shared_syndrome": syndrome, "total_cosets": int(row.sum()), "zero_cosets": int(row[0]), "first_raw_moment": first_moment, **extrema[syndrome]}
        if len(nonzero_positions):
            lo, hi = nonzero_positions[0], nonzero_positions[-1]
            assert int(numerators[lo]) == record["minimum_nonzero_raw"]
            assert int(numerators[hi]) == record["maximum_raw"]
            record["minimum_nonzero_cosets"] = int(row[lo])
            record["maximum_cosets"] = int(row[hi])
        rows.append(record)
    np.savez_compressed(output / "conditional_histogram.npz", numerators=numerators, counts_by_shared6=counts)
    np.savez_compressed(output / "template_statistics.npz", records=template_stats)
    summary = {"complete": complete, "model": "Complete R(c,k), all 52-dimensional endpoint cosets, conditioned on normalized shared6", "input_run_dir": str(run_dir.resolve()), "templates_processed": len(ids), "templates_total": TOTAL_TEMPLATES, "denominator": 1 << 54, "probability_denominator": 1 << 54, "shared_dimension": 6, "endpoint_dimension": 52, "row_cosets_if_complete": ROW_COSETS, "known_zero_J_cosets_per_row": KNOWN_ZERO_J_COSETS, "zero_counts_obtained_from": "Exact local zero-key multiplicities plus every actual connector histogram zero bin; no L1 positivity shortcut", "distinct_probability_numerators": len(numerators), "common_power2": shift, "accumulator_bytes": required_bytes, "counts_dtype": "uint64", "numerators_dtype": "uint32", "shared_syndrome_order": "col0 row3 low3 bits in syndrome bits0..2; col1 row3 low3 in bits3..5", "conditional_first_moment_verified": complete, "rows": rows, "python_reference_histograms_checked_in_kernel": run_status["python_histograms_checked"], "accumulation_seconds": accumulation_seconds, "elapsed_seconds": time.perf_counter() - started}
    if complete:
        assert run_status["python_histograms_checked"] == 117
        summary["all_live_R_positive"] = bool(np.all(template_stats["zero_connector_count"] == 0))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_status(status_path, state="complete", phase="done", complete=complete, templates_processed=len(ids), templates_total=TOTAL_TEMPLATES, elapsed_seconds=time.perf_counter() - started)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        aggregate(args.run_dir, args.output, args.allow_partial)
    except BaseException as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_status(args.output / "aggregation_status.json", state="failed", complete=False, error=repr(error), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
