"""Select and independently validate a complete small span of the five-round U run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import time

import numpy as np

from common import ZERO, canonical_mask, column_words, parse_words, serializable_config, words_hex
from experiment import monte_carlo
from compute import save_new, unrestricted_config
from spectrum import analyze_span, coefficient_map, independent_basis, span_masks


def compare_classes(model, actual, threshold=None, familywise_alpha=0.01):
    predicted = model["distribution"]
    if not predicted or len(actual["classes"]) != len(predicted):
        raise ValueError("model and complete class experiment must have equal nonzero lengths")
    comparisons = 2 * len(predicted)
    automatic_threshold = threshold is None
    if threshold is None:
        if not 0 < familywise_alpha < 1:
            raise ValueError("familywise alpha must lie in (0,1)")
        threshold = NormalDist().inv_cdf(1 - familywise_alpha / (2 * comparisons))
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    measured = [row["baseline"]["correlation"] for row in actual["classes"]]
    errors = [row["baseline"]["se"] for row in actual["classes"]]
    rows = []
    for index, (theory, observed, se) in enumerate(zip(predicted, measured, errors)):
        difference = abs(theory - observed)
        rows.append(dict(class_index=index, predicted=theory, actual=observed, se=se,
                         absolute_error=difference,
                         standard_errors=difference / se if se else None,
                         compatible=bool(difference <= threshold * se + 1e-12)))
    spectrum = fwht(measured) / len(measured)
    frequency_se = float(np.sqrt(sum(error * error for error in errors)) / len(errors))
    model_spectrum = fwht(predicted) / len(predicted)
    frequency_errors = np.abs(model_spectrum - spectrum)
    zero_variance_mismatches = [row["class_index"] for row in rows
                                if row["se"] == 0 and row["absolute_error"] > 1e-12]
    resolved_nonzero = [index for index in range(1, len(spectrum))
                        if abs(float(spectrum[index])) > threshold * frequency_se + 1e-12]
    return dict(classes=rows, all_classes_compatible=all(row["compatible"] for row in rows),
                estimated_fourier=spectrum.tolist(), fourier_se=frequency_se,
                model_fourier=model_spectrum.tolist(),
                maximum_fourier_error=float(frequency_errors.max()),
                all_fourier_compatible=bool(np.max(frequency_errors) <= threshold * frequency_se + 1e-12),
                resolved_nonzero_fourier_indices=resolved_nonzero,
                input_dependence_resolved=bool(resolved_nonzero), threshold_sigma=threshold,
                per_capacity_familywise_alpha=familywise_alpha if automatic_threshold else None,
                comparison_count=comparisons, zero_variance_mismatches=zero_variance_mismatches,
                maximum_class_standard_errors=(None if zero_variance_mismatches else
                                               max((row["standard_errors"] or 0) for row in rows)),
                maximum_fourier_standard_errors=(float(frequency_errors.max()) / frequency_se
                                                  if frequency_se else None))


def expression(mask):
    return " xor ".join(f"{('IV','K0','K1','N0','N1')[row]}[{col}]"
                        for row, word in enumerate(mask) for col in range(64)
                        if (word >> (63-col)) & 1) or "0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_config(cfg):
    return json.loads(json.dumps(serializable_config(cfg)))


def validated_selection(directory):
    """Check the frozen domain, basis, source, and statistical protocol."""
    path = directory / "selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    cfg = unrestricted_config()
    if selection["config"] != json_config(cfg):
        raise ValueError("selection does not use the unrestricted target")
    basis = [parse_words(m) for m in selection["basis_words"]]
    if (not basis or len(independent_basis(basis)) != len(basis)
            or any(canonical_mask(mask, cfg) != (mask, 1) for mask in basis)):
        raise ValueError("selection basis must contain independent canonical free-input masks")
    if selection["dimension"] != len(basis) or selection["class_count"] != 1 << len(basis):
        raise ValueError("selection dimension or class count does not match its basis")
    if selection["basis_expressions"] != [expression(mask) for mask in basis]:
        raise ValueError("selection expressions do not match its basis")
    source = directory / "cap_32.json"
    if Path(selection["source"]).name != source.name or selection["source_sha256"] != sha(source):
        raise ValueError("selection source does not match the recorded H32 spectrum")
    source_result = json.loads(source.read_text(encoding="utf-8"))
    if source_result["config"] != json_config(cfg):
        raise ValueError("selection source does not use the unrestricted target")
    source_coefficients = coefficient_map(source_result)
    if any(mask not in source_coefficients for mask in span_masks(basis)):
        raise ValueError("selection source does not explicitly contain every span coordinate")
    protocol = selection["validation"]
    # The report implements this pre-registered, per-capacity comparison family.
    # Reject changed metadata rather than silently report a different protocol.
    comparisons = 2 * selection["class_count"]
    threshold = NormalDist().inv_cdf(1 - 0.01 / (2 * comparisons))
    if (protocol["samples_log2_per_class"] != 22
            or protocol["per_capacity_familywise_alpha"] != 0.01
            or protocol["comparison_count"] != comparisons
            or not math.isclose(protocol["threshold_sigma"], threshold, rel_tol=0., abs_tol=1e-14)):
        raise ValueError("selection validation metadata differs from the implemented frozen protocol")
    return selection, basis, sha(path)


def validate_class_record(record, index, selection, basis):
    """Reject a stale or mixed-domain class record before reuse/comparison."""
    protocol = selection["validation"]
    constraints = [{"mask_words": words_hex(mask), "rhs": (index >> j) & 1}
                   for j, mask in enumerate(basis)]
    if (record["class_index"] != index
            or record["N"] != 1 << protocol["samples_log2_per_class"]
            or record["seed"] != protocol["seed_base"] + index
            or json_config(record["config"]) != selection["config"]
            or record["constraints"] != constraints
            or record["domain_rank"] != len(basis)
            or record["domain_dimension"] != 256 - len(basis)):
        raise ValueError(f"class {index} does not match the frozen unrestricted protocol")
    baseline = record["baseline"]
    count, signed = record["N"], baseline["signedsum"]
    if (not isinstance(signed, int) or abs(signed) > count or (count - signed) % 2
            or baseline["correlation"] != signed / count
            or not math.isclose(baseline["se"], math.sqrt((1 - (signed / count) ** 2) / count),
                                rel_tol=1e-12, abs_tol=1e-15)):
        raise ValueError(f"class {index} has inconsistent signed counts or sampling statistics")


def validate_actual(actual, selection, basis, selection_sha256):
    """A complete aggregate must match this exact frozen selection and order."""
    if (actual.get("selection_sha256") != selection_sha256
            or [parse_words(mask) for mask in actual["basis_words"]] != basis):
        raise ValueError("actual-class aggregate uses a different or stale selection")
    records = actual["classes"]
    if (len(records) != 1 << len(basis)
            or [record["class_index"] for record in records] != list(range(1 << len(basis)))):
        raise ValueError("actual-class aggregate must contain every class exactly once in index order")
    for index, record in enumerate(records):
        validate_class_record(record, index, selection, basis)


def prepare(directory):
    source = directory / "cap_32.json"
    result = json.loads(source.read_text())
    cfg = unrestricted_config()
    if result["config"] != json_config(cfg):
        raise ValueError("wrong target or input domain")
    coefficients = coefficient_map(result)
    selected, history = [], []
    for mask, value in sorted(coefficients.items(), key=lambda pair: (-abs(pair[1]), pair[0])):
        if mask == ZERO or len(independent_basis(selected + [mask])) == len(selected):
            continue
        proposed = span_masks(selected + [mask])
        missing = sum(m not in coefficients for m in proposed)
        if missing:
            continue
        selected.append(mask)
        history.append(dict(mask_words=words_hex(mask), expression=expression(mask), coefficient=value))
        if len(selected) == 8:
            break
    # This readable basis must span exactly the data-selected space, not enlarge it.
    readable = [(0, 1 << 63, 1 << 63, 0, 0),
                (0, 1 << 60, 0, 0, 0), (0, 0, 1 << 60, 0, 0),
                column_words(3, 0), column_words(3, 3)]
    if set(span_masks(selected)) != set(span_masks(readable)):
        raise ValueError("observed selected space differs from the intended readable change of basis")
    selection = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                     config=serializable_config(cfg), source=str(source), source_sha256=sha(source),
                     dimension=len(readable), class_count=1 << len(readable),
                     basis_words=[words_hex(m) for m in readable],
                     basis_expressions=[expression(m) for m in readable],
                     selection_history=history,
                     selection_rule="descending absolute H32 root coefficients; greedily add independent masks only when every span coordinate is explicitly present; ceiling 8, resulting rank 5",
                     readable_basis_rule="invertible change of basis, exact equality of spans checked",
                     primary_capacity=128, secondary_capacity=64,
                     validation=dict(samples_log2_per_class=22, seed_base=914510000,
                                     independent_baseline_samples_log2=24, baseline_seed=914500001,
                                     per_capacity_familywise_alpha=0.01,
                                     comparison_count=64,
                                     threshold_sigma=NormalDist().inv_cdf(1-0.01/128)),
                     zero_policy="count exact rational cancellations of the stored float64 span coefficients; no epsilon threshold, no missing-coordinate imputation, no proof of true zero",
                     statistical_zero_policy="a confidence interval containing zero is unresolved at this sample size, not proof of zero")
    save_new(directory / "selection.json", selection)
    print(json.dumps(selection, indent=2), flush=True)


def sample(directory):
    selection, basis, selection_digest = validated_selection(directory)
    cfg = unrestricted_config()
    protocol = selection["validation"]
    output = directory / "actual_classes_n22.json"
    if output.exists():
        validate_actual(json.loads(output.read_text(encoding="utf-8")), selection, basis, selection_digest)
    records = []
    start = time.perf_counter()
    for index in range(1 << len(basis)):
        path = directory / "class_mc" / f"class_{index:02d}.json"
        constraints = [(mask, (index >> j) & 1) for j, mask in enumerate(basis)]
        if path.exists():
            record = json.loads(path.read_text())
            validate_class_record(record, index, selection, basis)
        else:
            result = monte_carlo(cfg, masks=(ZERO,), sample_log2=protocol["samples_log2_per_class"],
                                 seed=protocol["seed_base"] + index, constraints=constraints)
            record = dict(class_index=index, **result)
            save_new(path, record)
        records.append(record)
        print(f"CLASS {index:02d}: C={record['baseline']['correlation']:.12g}; N={record['N']}; seconds={record['elapsed_seconds']:.3f}", flush=True)
    if not output.exists():
        save_new(output, dict(basis_words=selection["basis_words"], classes=records,
                              wall_seconds_this_invocation=time.perf_counter()-start,
                              selection_sha256=sha(directory / "selection.json")))


def exact_values(coefficients, basis):
    targets = span_masks(basis)
    if any(m not in coefficients for m in targets):
        return None
    values = [Fraction.from_float(coefficients[m]) for m in targets]
    step = 1
    while step < len(values):
        for block in range(0, len(values), 2*step):
            for j in range(step):
                a, b = values[block+j], values[block+j+step]
                values[block+j], values[block+j+step] = a+b, a-b
        step *= 2
    return values


def report(directory):
    selection, basis, selection_digest = validated_selection(directory)
    cfg = unrestricted_config()
    actual_file = directory / "actual_classes_n22.json"
    actual = json.loads(actual_file.read_text()) if actual_file.exists() else None
    if actual is not None:
        validate_actual(actual, selection, basis, selection_digest)
    models = []
    for path in sorted(directory.glob("cap_*.json"), key=lambda p:int(p.stem.split("_")[-1])):
        result = json.loads(path.read_text())
        row = dict(cap=result["budgets"]["cap"], source=path.name, sha256=sha(path),
                   complete=result["complete"], status=result["status"], seconds=result["host_wall_seconds"],
                   stats=result.get("stats", {}))
        if result["config"] != json_config(cfg):
            raise ValueError("wrong target or domain")
        if result["complete"]:
            c = coefficient_map(result)
            row["root_terms"] = len(c)
            row["support_rank"] = len(independent_basis(c))
            row["zero_frequency"] = c.get(ZERO)
            row["model"] = model = analyze_span(c, basis, config=cfg)
            exact = exact_values(c, basis)
            if exact is not None:
                row["exact_stored_coefficient_fwht"] = [str(v) for v in exact]
                row["model_zero_classes_exact"] = [i for i,v in enumerate(exact) if v == 0]
                row["model_zero_classes_float"] = [i for i,v in enumerate(model["distribution"]) if v == 0.0]
                row["maximum_float_fwht_rounding_error"] = max(abs(Fraction.from_float(v)-q) for v,q in zip(model["distribution"],exact)).__float__()
                if actual is not None:
                    row["comparison"] = compare_classes(model, actual)
        models.append(row)
    primary = next((r for r in models if r["cap"] == selection["primary_capacity"]), None)
    summary = dict(config=selection["config"], selection_sha256=sha(directory / "selection.json"),
                   dimension=len(basis), class_count=1 << len(basis), models=models,
                   primary_capacity=selection["primary_capacity"],
                   status="pending_primary_or_validation", true_zero_certified_classes=[],
                   zero_policy=selection["zero_policy"])
    if actual is not None:
        zcrit = NormalDist().inv_cdf(1-0.01/(2*len(actual["classes"])))
        summary["experimental_zero_diagnostics"] = dict(
            per_class_samples=actual["classes"][0]["N"], total_samples=sum(r["N"] for r in actual["classes"]),
            point_estimate_zero_classes=[r["class_index"] for r in actual["classes"] if r["baseline"]["signedsum"] == 0],
            ordinary_95pct_interval_contains_zero=[r["class_index"] for r in actual["classes"] if abs(r["baseline"]["correlation"]) <= 1.959963984540054*r["baseline"]["se"]],
            simultaneous_99pct_interval_contains_zero=[r["class_index"] for r in actual["classes"] if abs(r["baseline"]["correlation"]) <= zcrit*r["baseline"]["se"]],
            simultaneous_threshold_sigma=zcrit,
            note="normal-approximation intervals; unresolved is not proven zero")
    if primary and primary.get("comparison"):
        test = primary["comparison"]
        summary["status"] = ("primary_model_not_rejected_at_sample_resolution" if test["all_classes_compatible"] and test["all_fourier_compatible"] else "primary_model_rejected_at_sample_resolution")
        summary["primary_zero_class_count"] = len(primary["model_zero_classes_exact"])
        summary["primary_zero_classes"] = primary["model_zero_classes_exact"]
    save_new(directory / "analysis.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "models"}, indent=2), flush=True)
    for row in models:
        print(json.dumps({"cap":row["cap"], "seconds":row["seconds"], "zero":row.get("zero_frequency"),
                          "span":row.get("model",{}).get("status"), "min":row.get("model",{}).get("minimum"),
                          "max":row.get("model",{}).get("maximum"), "zeros":row.get("model_zero_classes_exact"),
                          "max_class_SE":row.get("comparison",{}).get("maximum_class_standard_errors"),
                          "classes_pass":row.get("comparison",{}).get("all_classes_compatible"),
                          "frequencies_pass":row.get("comparison",{}).get("all_fourier_compatible")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "sample", "report"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    {"prepare":prepare, "sample":sample, "report":report}[args.stage](args.output)
