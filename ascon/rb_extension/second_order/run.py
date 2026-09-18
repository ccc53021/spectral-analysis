from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def command(args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    output = ROOT / "results" / "dl2"
    if output.exists():
        raise SystemExit(f"Preserving existing output directory: {output}")
    command([
        sys.executable,
        "compute.py",
        "--output", output,
        "--caps", "1", "2", "4", "8", "16", "32", "64", "128",
        "--time-limit", "1800",
        "--max-nodes", "2000000",
        "--max-terms", "1048576",
    ])
    command([sys.executable, "analyze.py", "prepare", "--output", output])
    command([sys.executable, "analyze.py", "sample", "--output", output])
    command([sys.executable, "analyze.py", "report", "--output", output])


if __name__ == "__main__":
    main()
