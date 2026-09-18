import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def command(directory, args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=directory, check=True)


def main():
    parser = argparse.ArgumentParser(description="Ascon round-based extensions")
    parser.add_argument("--case", choices=("d1", "d2", "all"), default="all")
    args = parser.parse_args()
    if args.case in ("d1", "all"):
        command(ROOT / "first_order", [sys.executable, "run.py"])
    if args.case in ("d2", "all"):
        command(ROOT / "second_order", [sys.executable, "run.py"])


if __name__ == "__main__":
    main()
