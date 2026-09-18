#!/usr/bin/env python3
"""Run the original GIFT-64 Python/C++ pipeline for D.6--D.17."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


BASE_DIR = Path(__file__).resolve().parent
CASES = tuple(f"d{index}" for index in range(6, 18))
MODES = {"key": False, "joint": True}


def run_case(case: str, mode: str) -> None:
    default_dimension = 20 if case == "d17" else 30
    env = os.environ.copy()
    env.update(
        {
            "GIFT_TARGET": case,
            "GIFT_INPUT_X": str(MODES[mode]),
            "GIFT_BASIS_NUMBER": env.get("GIFT_BASIS_NUMBER", str(default_dimension)),
            "GIFT_DIM_TRUNCATED": env.get("GIFT_DIM_TRUNCATED", str(default_dimension)),
            "GIFT_WEIGHT_RANGE": env.get("GIFT_WEIGHT_RANGE", "100"),
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
    args = parser.parse_args()

    cases = CASES if args.case == "all" else (args.case,)
    modes = MODES if args.mode == "all" else (args.mode,)
    for case in cases:
        for mode in modes:
            run_case(case, mode)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
