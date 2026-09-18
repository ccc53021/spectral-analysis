import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
FOUR_ROUND = ROOT.parent / "four_round"
DL10_NAME = "r4_public_full_extension_core"


def command(arguments, directory=ROOT, capture=False):
    print("+", " ".join(map(str, arguments)), flush=True)
    return subprocess.run(
        arguments,
        cwd=directory,
        check=True,
        text=True,
        capture_output=capture,
    )


def ensure_dl10_theory(smoke=False):
    source = (
        FOUR_ROUND
        / "output"
        / "known_distinguisher_joint"
        / DL10_NAME
        / "p1_complete"
        / "distribution_analysis.json"
    )
    if not source.is_file():
        capacities = "4,8" if smoke else "4,8,16,32,64,128"
        command([
            sys.executable,
            "calibration.py",
            DL10_NAME,
            "1",
            "--capacities",
            capacities,
        ], FOUR_ROUND)
        sweep = (
            FOUR_ROUND
            / "output"
            / DL10_NAME
            / "joint_ddt_prefix"
            / "p1_complete"
            / "capacity_sweep.json"
        )
        command([
            sys.executable,
            "basis.py",
            DL10_NAME,
            "select",
            sweep,
        ], FOUR_ROUND)
        distribution_command = [
            sys.executable,
            "distribution.py",
            DL10_NAME,
            "1",
            "--run-coset",
        ]
        if smoke:
            distribution_command.extend(("--coset-capacities", "4,8"))
        command([
            *distribution_command,
        ], FOUR_ROUND)
    return source


def main():
    parser = argparse.ArgumentParser(description="Xoodoo DL.16 five-round pipeline")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = ensure_dl10_theory(args.smoke)

    pullback = command([
        sys.executable,
        "derive_constraints.py",
        "--source",
        source,
        "--random-tests",
        "8" if args.smoke else "512",
    ], capture=True)
    (ROOT / "constraints.json").write_text(pullback.stdout, encoding="utf-8")

    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)
    run_a = output / "run_a.json"
    run_b = output / "run_b.json"
    if run_a.exists() or run_b.exists():
        raise SystemExit(f"Preserving existing experiment files in {output}")
    samples_log2 = "10" if args.smoke else "30"
    batch_log2 = "10" if args.smoke else "19"
    threads = "1" if args.smoke else "6"
    base = [
        sys.executable,
        "experiment.py",
        "--samples-log2",
        samples_log2,
        "--batch-log2",
        batch_log2,
        "--threads",
        threads,
    ]
    command([*base, "--seed", "2026091101", "--output", run_a])
    command([*base, "--seed", "2026091102", "--output", run_b])
    command([
        sys.executable,
        "summarize.py",
        run_a,
        run_b,
        "--output-json",
        output / "combined.json",
        "--output-md",
        output / "combined.md",
    ])


if __name__ == "__main__":
    main()
