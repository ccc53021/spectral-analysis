import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def command(directory, arguments):
    print("+", " ".join(map(str, arguments)), flush=True)
    subprocess.run(arguments, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description="Xoodoo paper-result runner")
    parser.add_argument(
        "--case",
        choices=tuple(f"d{index}" for index in range(9, 17)) + ("all",),
        default="all",
    )
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases = (
        tuple(f"d{index}" for index in range(9, 17))
        if args.case == "all"
        else (args.case,)
    )
    for case in cases:
        print(f"\n=== {case.upper()} ===", flush=True)
        directory = ROOT / ("five_round" if case == "d16" else "four_round")
        arguments = [sys.executable, "run.py"]
        if case != "d16":
            arguments.extend(("--case", case))
        if args.smoke:
            arguments.append("--smoke")
        command(directory, arguments)


if __name__ == "__main__":
    main()
