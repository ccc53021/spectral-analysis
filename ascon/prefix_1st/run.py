import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def command(args, env):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def compiler():
    for name in ("g++", "clang++"):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("A C++ compiler (g++ or clang++) is required.")


def sage_python():
    sage = shutil.which("sage")
    if not sage:
        raise SystemExit("SageMath is required for step3.py.")
    return sage


def run_target(target):
    env = os.environ.copy()
    env["ASCON_TARGET"] = target
    command([sys.executable, "step1.py"], env)
    command([sys.executable, "prep_step2_data_for_c++.py"], env)
    binary = ROOT / ("step2.exe" if os.name == "nt" else "step2")
    command([compiler(), "-O3", "-std=c++17", "-pthread", "step2.cpp", "-o", str(binary)], env)
    command([str(binary)], env)
    command([sage_python(), "-python", "step3.py"], env)


def main():
    parser = argparse.ArgumentParser(description="Ascon DL.3-DL.7 native pipeline")
    parser.add_argument("--case", choices=("d3", "d4", "d5", "d6", "d7", "all"), default="all")
    args = parser.parse_args()
    targets = ("d3", "d4", "d5", "d6", "d7") if args.case == "all" else (args.case,)
    for target in targets:
        print(f"\n=== {target.upper()} ===", flush=True)
        run_target(target)


if __name__ == "__main__":
    main()
