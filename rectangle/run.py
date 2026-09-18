#!/usr/bin/env python3
"""Run the original RECTANGLE Python/C++ pipeline for D.1 and D.2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent
CASES = {"d1": 0, "d2": 1}
MODES = {"key": False, "joint": True}


def run_case(
    case: str,
    mode: str,
    basis_number: int,
    dim_truncated: int,
    weight_range: int,
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "RECTANGLE_CHARACTERISTIC": str(CASES[case]),
            "RECTANGLE_INPUT_X": str(MODES[mode]),
            "RECTANGLE_BASIS_NUMBER": str(basis_number),
            "RECTANGLE_DIM_TRUNCATED": str(dim_truncated),
            "RECTANGLE_WEIGHT_RANGE": str(weight_range),
        }
    )
    print(f"\n######## {case.upper()} / {mode} ########", flush=True)
    subprocess.run(
        [sys.executable, "-B", str(BASE_DIR / "run_all.py")],
        cwd=BASE_DIR,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--mode", choices=(*MODES, "all"), default="all")
    parser.add_argument("--basis-number", type=int, default=30)
    parser.add_argument("--dim-truncated", type=int, default=30)
    parser.add_argument("--weight-range", type=int, default=100)
    args = parser.parse_args()

    if not 1 <= args.dim_truncated <= args.basis_number <= 30:
        parser.error("require 1 <= dim-truncated <= basis-number <= 30")
    if args.weight_range < 1:
        parser.error("weight-range must be positive")

    cases = CASES if args.case == "all" else (args.case,)
    modes = MODES if args.mode == "all" else (args.mode,)
    for case in cases:
        for mode in modes:
            run_case(
                case,
                mode,
                args.basis_number,
                args.dim_truncated,
                args.weight_range,
            )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
