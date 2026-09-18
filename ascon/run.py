import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def command(directory, args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description="Ascon paper-result runner")
    parser.add_argument("--case", choices=("d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "all"), default="all")
    args = parser.parse_args()
    cases = tuple(f"d{i}" for i in range(1, 9)) if args.case == "all" else (args.case,)
    for case in cases:
        print(f"\n=== {case.upper()} ===", flush=True)
        if case in ("d1", "d2"):
            command(ROOT / "rb_extension", [sys.executable, "run.py", "--case", case])
        elif case == "d8":
            command(ROOT / "prefix_2rd", [sys.executable, "run.py"])
        else:
            command(ROOT / "prefix_1st", [sys.executable, "run.py", "--case", case])


if __name__ == "__main__":
    main()
