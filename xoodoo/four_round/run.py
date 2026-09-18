import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CASES = {
    "d9": "r4_public_full_standard",
    "d10": "r4_public_full_extension_core",
    "d11": "r4_full_new_5_to_33_2",
    "d12": "r4_full_new_5_to_83_4",
    "d13": "r4_full_new_3_to_37_4",
    "d14": "r4_full_new_6_to_33_4",
    "d15": "r4_full_new_3_to_48_4",
}


def command(arguments):
    print("+", " ".join(map(str, arguments)), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def run_case(case, smoke=False):
    name = CASES[case]
    capacities = "4,8" if smoke else "4,8,16,32,64,128"
    samples_log2 = "10" if smoke else "24"
    batch_log2 = "10" if smoke else "16"
    command([
        sys.executable,
        "calibration.py",
        name,
        "1",
        "--capacities",
        capacities,
    ])
    sweep = (
        ROOT
        / "output"
        / name
        / "joint_ddt_prefix"
        / "p1_complete"
        / "capacity_sweep.json"
    )
    command([
        sys.executable,
        "basis.py",
        name,
        "select",
        sweep,
    ])
    distribution_command = [
        sys.executable,
        "distribution.py",
        name,
        "1",
        "--run-coset",
    ]
    if smoke:
        distribution_command.extend(("--coset-capacities", "4,8"))
    command(distribution_command)
    command([
        sys.executable,
        "experiment.py",
        "--cases",
        name,
        "--prefixes",
        "1",
        "--samples-log2",
        samples_log2,
        "--batch-log2",
        batch_log2,
        "--output",
        ROOT / "results" / "four_round_experiments.json",
    ])


def main():
    parser = argparse.ArgumentParser(description="Xoodoo DL.9-DL.15 pipelines")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected = tuple(CASES) if args.case == "all" else (args.case,)
    for case in selected:
        run_case(case, args.smoke)


if __name__ == "__main__":
    main()
