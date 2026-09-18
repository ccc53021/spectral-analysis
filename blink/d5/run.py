#!/usr/bin/env python3
"""Run the exact CGZ+26 Figure-7 Blink-128 analyses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import time

from engine.input_open import write_input_open_distribution
from engine.key_only import write_key_only_distribution


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output"


def verify_key_only(record):
    if record["basis_dimension"] != 14:
        raise AssertionError("Figure-7 master-key support rank must be 14")
    if record["master_key_support_rank"] != 14:
        raise AssertionError("embedded master-key rank must be 14")
    maximum = Fraction(**record["p_max_exact"])
    minimum_nonzero = Fraction(**record["p_min_nonzero_exact"])
    if maximum != Fraction(9, 1 << 72):
        raise AssertionError(f"key-only maximum mismatch: {maximum}")
    if minimum_nonzero != Fraction(1, 1 << 74):
        raise AssertionError(f"key-only nonzero minimum mismatch: {minimum_nonzero}")
    histogram = record["probability_histogram"]
    if sum(row["class_count"] for row in histogram) != 1 << 14:
        raise AssertionError("key-only histogram is not complete")
    zero = next(
        row for row in histogram if row["probability_exact"]["numerator"] == 0
    )
    if zero["class_count"] != 16_240:
        raise AssertionError("unexpected key-only zero class count")
    if record["p_max_class_count"] != 1:
        raise AssertionError("published maximum must be one 14D quotient class")


def verify_input_open(record):
    ranks = (
        record["input_support_rank"],
        record["master_key_support_rank"],
        record["joint_support_rank"],
    )
    if ranks != (70, 14, 84):
        raise AssertionError(f"unexpected input/key/joint ranks: {ranks}")
    maximum = Fraction(**record["p_max_exact"])
    minimum_nonzero = Fraction(**record["p_min_nonzero_exact"])
    if maximum != Fraction(9, 1 << 52):
        raise AssertionError(f"input-open maximum mismatch: {maximum}")
    if minimum_nonzero != Fraction(1, 1 << 54):
        raise AssertionError(f"input-open nonzero minimum mismatch: {minimum_nonzero}")
    if record["p_max_class_count"] != 1 << 50:
        raise AssertionError("unexpected input-open maximum class count")
    histogram = record["probability_histogram"]
    if sum(row["class_count"] for row in histogram) != 1 << 84:
        raise AssertionError("factorized input-open histogram is not complete")
    if Fraction(**record["good_input_fraction"]) != Fraction(1, 1 << 20):
        raise AssertionError("maximum-input density must be 2^-20")
    recovered = Fraction(**record["key_only_maximum_recovered_by_input_average"])
    if recovered != Fraction(9, 1 << 72):
        raise AssertionError("input average did not recover the key-only maximum")


def execute(profile, output_directory):
    started = time.time()
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    results = {}

    if profile in ("key-only", "all"):
        path = output_directory / "key_only_14d.json"
        print("[1/2] Computing the complete 14D master-key distribution...", flush=True)
        record = write_key_only_distribution(path)
        verify_key_only(record)
        results["key_only"] = path.name
        print(
            "[verified] key-only rank=14, p_max=9/2^72, "
            "p_min_nonzero=1/2^74",
            flush=True,
        )

    if profile in ("input-open", "all"):
        path = output_directory / "input_open_factorized_84d.json"
        print(
            "[2/2] Computing per-column input quotients and the factorized "
            "joint distribution...",
            flush=True,
        )
        record = write_input_open_distribution(path)
        verify_input_open(record)
        results["input_open"] = path.name
        print(
            "[verified] input/key/joint ranks=70/14/84, p_max=9/2^52, "
            "p_min_nonzero=1/2^54",
            flush=True,
        )

    report = {
        "completed_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "verified": True,
        "cipher": "Blink-128",
        "source": "CGZ+26 Figure 7",
        "profile": profile,
        "outputs": results,
        "elapsed_seconds": time.time() - started,
    }
    report_path = output_directory / "last_run.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}", flush=True)
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact column-factorized CGZ+26 Blink-128 analysis"
    )
    parser.add_argument(
        "--profile",
        choices=("key-only", "input-open", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    execute(args.profile, args.output_dir)


if __name__ == "__main__":
    main()

