#!/usr/bin/env python3
"""Run the paper configurations of the Blink pipelines."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def call(directory: str, *arguments: str) -> None:
    target = HERE / directory
    subprocess.run(
        [sys.executable, "-B", str(target / "run.py"), *arguments],
        cwd=target,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("d3", "d4", "d5", "all"), default="all")
    parser.add_argument("--mode", choices=("key", "joint", "all"), default="all")
    args = parser.parse_args()

    cases = ("d3", "d4", "d5") if args.case == "all" else (args.case,)
    for case in cases:
        if case == "d3":
            call(case)
        elif case == "d4":
            call(case, "--mode", args.mode)
        else:
            profile = {"key": "key-only", "joint": "input-open", "all": "all"}[args.mode]
            call(case, "--profile", profile)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
