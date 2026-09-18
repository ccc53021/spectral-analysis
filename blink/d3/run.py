#!/usr/bin/env python3
"""Run the Blink-64 D.3 Python/OpenMP C++ pipeline."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "pipeline"
OUTER = PIPELINE / "outer"
FULL_TEMPLATE_COUNT = 1_149_984


def python(module: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, "-B", "-m", module, *arguments],
        cwd=HERE,
        check=True,
    )


def command(arguments: list[str], cwd: Path) -> None:
    subprocess.run(arguments, cwd=cwd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact Blink-64 D.3 Python/OpenMP C++ pipeline"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("BLINK_D3_THREADS", "8")),
    )
    parser.add_argument(
        "--template-count",
        type=int,
        default=int(
            os.environ.get("BLINK_D3_TEMPLATE_COUNT", str(FULL_TEMPLATE_COUNT))
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threads = args.threads
    template_count = args.template_count
    if threads < 1:
        raise ValueError("BLINK_D3_THREADS must be positive")
    if not 1 <= template_count <= FULL_TEMPLATE_COUNT:
        raise ValueError(
            f"BLINK_D3_TEMPLATE_COUNT must be in [1, {FULL_TEMPLATE_COUNT}]"
        )
    compiler = os.environ.get("CXX", "g++")
    if shutil.which(compiler) is None:
        raise RuntimeError(f"C++ compiler not found: {compiler}")

    python("pipeline.outer.local_orbits")
    python("pipeline.outer.outer_model", "--checks", "64")
    python("pipeline.outer.conditional_factors")
    python("pipeline.outer.histogram_probe")
    python("pipeline.outer.prepare_cpp")

    python("pipeline.middle.analyze")
    python("pipeline.space_audit")
    python("pipeline.middle.conditional_on_outer")
    python("pipeline.global_coefficients", "--max-random", "3000")
    python("pipeline.joint.analyze_head")
    python("pipeline.joint.assemble_quotient")
    python("pipeline.joint.project_head_weight")

    cpp_output = OUTER / "output" / "cpp"
    executable = cpp_output / ("outer_kernel_full.exe" if os.name == "nt" else "outer_kernel_full")
    command(
        [
            compiler,
            "-O3",
            "-march=native",
            "-fopenmp",
            "-std=c++17",
            "-DFWHT_TILE_LOG2=16",
            str(OUTER / "outer_kernel.cpp"),
            "-o",
            str(executable),
        ],
        OUTER,
    )

    common = [
        str(executable),
        "--data",
        str(cpp_output / "outer_data.bin"),
        "--reference",
        str(cpp_output / "python_reference.bin"),
    ]
    command(
        common
        + [
            "--cases",
            str(cpp_output / "benchmark_cases.bin"),
            "--count",
            "117",
            "--threads",
            str(threads),
            "--chunk",
            "32",
            "--output",
            str(cpp_output / "full_binary_verification"),
            "--resume",
        ],
        OUTER,
    )
    full_run = cpp_output / "full_hist"
    command(
        common
        + [
            "--count",
            str(template_count),
            "--threads",
            str(threads),
            "--chunk",
            "256",
            "--output",
            str(full_run),
            "--resume",
        ],
        OUTER,
    )

    aggregate_args = [
        "--run-dir",
        str(full_run),
        "--output",
        str(OUTER / "output" / "full_R"),
    ]
    if template_count != FULL_TEMPLATE_COUNT:
        aggregate_args.append("--allow-partial")
    python("pipeline.outer.aggregate_cpp", *aggregate_args)
    if template_count != FULL_TEMPLATE_COUNT:
        print("Partial template run completed; final coset join requires the full count.")
        return

    python(
        "pipeline.distribution_join",
        "--outer-npz",
        str(OUTER / "output" / "full_R" / "conditional_histogram.npz"),
        "--outer-summary",
        str(OUTER / "output" / "full_R" / "summary.json"),
    )
    python("pipeline.extend_zero_extrema")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
