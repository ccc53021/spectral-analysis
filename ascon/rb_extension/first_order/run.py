"""Single entry point.

Typical use:
    python run.py
    python run.py --no-experiment
    python run.py --show-config
"""

from __future__ import annotations

import argparse
import json
import math

import pipeline


def power(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:+.12g}=2^-{-math.log2(abs(value)):.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ascon v4 unified joint-U / 0-2DDT runner"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute integrated suffix caches and overwrite their files",
    )
    parser.add_argument(
        "--no-experiment",
        action="store_true",
        help="skip real-Ascon experiments",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip internal verification checks",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="validate and print the effective configuration, then exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = pipeline.settings_from_module()
    if args.show_config:
        print(json.dumps(pipeline._jsonable_settings(cfg), indent=2))
        print(f"fingerprint={pipeline.fingerprint(cfg)}")
        return

    summary = pipeline.run_all(
        force=args.force,
        run_experiments=False if args.no_experiment else None,
        run_verification=not args.no_verify,
    )
    print("\nDDT  U=0                         minimum                     maximum")
    print("---  --------------------------  --------------------------  --------------------------")
    for prefix in sorted(summary["prefix_results"], key=int):
        record = summary["prefix_results"][prefix]
        analysis = record["analysis"]
        print(
            f"{int(prefix):>3}  "
            f"{power(record['joint_zero_label_coefficient']):<26}  "
            f"{power(analysis['minimum']['correlation']):<26}  "
            f"{power(analysis['maximum']['correlation']):<26}"
        )
    print(f"\nverification={summary['verification']['passed']}")
    print(f"summary={summary['run_directory']}\\summary.json")
    print(f"report={summary['run_directory']}\\summary.md")


if __name__ == "__main__":
    main()
