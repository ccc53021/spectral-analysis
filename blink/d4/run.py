#!/usr/bin/env python3
"""Run the exact Blink-64 D.4 key-only and joint analyses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import time
from types import SimpleNamespace

from engine.key_only import write_key_only_distribution
from engine.joint_basis import run as run_joint_basis_search
from engine.joint_probability import run as run_joint_probability


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output"

EXPECTED_KEY_MAXIMUM = Fraction(3, 1 << 52)
EXPECTED_KEY_MINIMUM_NONZERO = Fraction(3, 1 << 54)
EXPECTED_JOINT_MAXIMUM = Fraction(3, 1 << 40)
EXPECTED_JOINT_MINIMUM_NONZERO = Fraction(3, 1 << 42)
EXPECTED_JOINT_ZERO_CLASSES = 134_213_120


def exact_value(record, field):
    payload = record[field]
    return Fraction(int(payload["numerator"]), int(payload["denominator"]))


def verify_key_only(record):
    if record["fourier_support_size"] != 36:
        raise AssertionError("master-key Fourier support must contain 36 terms")
    if record["basis_dimension"] != 6:
        raise AssertionError("master-key support rank must be 6")
    if exact_value(record, "p_max_exact") != EXPECTED_KEY_MAXIMUM:
        raise AssertionError("master-key maximum mismatch")
    if exact_value(record, "p_min_nonzero_exact") != EXPECTED_KEY_MINIMUM_NONZERO:
        raise AssertionError("master-key nonzero minimum mismatch")
    histogram = record["probability_histogram"]
    if sum(row["class_count"] for row in histogram) != 1 << 6:
        raise AssertionError("master-key histogram is incomplete")
    zero = next(
        row for row in histogram if row["probability_exact"]["numerator"] == 0
    )
    if zero["class_count"] != 55:
        raise AssertionError("unexpected master-key zero class count")


def verify_basis(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 28:
        raise AssertionError("27D basis must contain one meta row and 27 vectors")
    metadata = json.loads(lines[0])
    if metadata.get("full_support_rank") != 27:
        raise AssertionError("joint support rank must be 27")
    if metadata.get("selected_dimension") != 27:
        raise AssertionError("selected joint basis dimension must be 27")
    if metadata.get("global_support_size") != 5_061_888:
        raise AssertionError("unexpected joint Fourier support size")
    return metadata


def verify_joint(record):
    if record.get("basis_dimension") != 27:
        raise AssertionError("joint probability calculation did not use dimension 27")
    if exact_value(record, "p_max_exact") != EXPECTED_JOINT_MAXIMUM:
        raise AssertionError("joint maximum mismatch")
    if exact_value(record, "p_min_nonzero_exact") != EXPECTED_JOINT_MINIMUM_NONZERO:
        raise AssertionError("joint nonzero minimum mismatch")
    if record.get("p_zero_class_count") != EXPECTED_JOINT_ZERO_CLASSES:
        raise AssertionError("joint zero-probability class count mismatch")
    if record.get("component_dimensions") != {"A0": 21, "q2": 3, "q4": 3}:
        raise AssertionError("unexpected joint component dimensions")


def execute(mode, output_directory, reuse_basis=False):
    started = time.time()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {}

    if mode in ("key", "all"):
        path = output_directory / "key_only_6d.json"
        print("Computing the complete 6D master-key distribution...", flush=True)
        record = write_key_only_distribution(path)
        verify_key_only(record)
        outputs["key"] = path.name
        print("[verified] key rank=6 p_max=3/2^52 p_min_nonzero=3/2^54")

    if mode in ("joint", "all"):
        basis_path = output_directory / "joint_basis_27d.jsonl"
        probability_path = output_directory / "joint_probability_27d.json"
        if not (reuse_basis and basis_path.is_file()):
            run_joint_basis_search(
                SimpleNamespace(
                    dimension=27,
                    ranked_dimension=27,
                    ranked_output=str(basis_path),
                    output=str(basis_path),
                )
            )
        metadata = verify_basis(basis_path)
        probability = run_joint_probability(
            SimpleNamespace(
                input=str(basis_path),
                output=str(probability_path),
                dimension=27,
            )
        )
        verify_joint(probability)
        outputs["joint_basis"] = basis_path.name
        outputs["joint_probability"] = probability_path.name
        print("[verified] joint rank=27 p_max=3/2^40 p_min_nonzero=3/2^42")
    else:
        metadata = None

    report = {
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "verified": True,
        "cipher": "Blink-64",
        "source": "D.4 ten-round cluster",
        "mode": mode,
        "outputs": outputs,
        "elapsed_seconds": time.time() - started,
    }
    if metadata is not None:
        report["joint_support_size"] = metadata["global_support_size"]
        report["joint_support_rank"] = metadata["full_support_rank"]
    report_path = output_directory / "last_run.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Exact Blink-64 D.4 analysis")
    parser.add_argument("--mode", choices=("key", "joint", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-basis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    execute(args.mode, args.output_dir, args.reuse_basis)


if __name__ == "__main__":
    main()
