#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

if __name__ == "__main__":
    sys.excepthook = sys.__excepthook__

import cipher_config as config


BASE_DIR = Path(__file__).resolve().parent
CPP_PATH = BASE_DIR / "step2.cpp"
HEADER_PATH = BASE_DIR / "step2_generated_config.h"
EXE_PATH = BASE_DIR / "step2.exe"


def validate_config():
    errors = []
    expected_route_len = (
        config.characteristic_model_number * config.total_rounds + 1
    )

    if config.state_bits != 64:
        errors.append(f"state_bits must be 64, got {config.state_bits}")
    if config.key_model != "expanded":
        errors.append("RECTANGLE currently supports independent expanded keys only")
    if config.round_key_bits != config.state_bits:
        errors.append("RECTANGLE adds a full-state (64-bit) round key")
    if config.key_bits != config.total_rounds * config.round_key_bits:
        errors.append("key_bits must equal total_rounds * round_key_bits")
    if config.round_constants != [0] * config.total_rounds:
        errors.append("RECTANGLE has no data-path constants in the expanded-key model")
    if sorted(config.sbox) != list(range(16)):
        errors.append("sbox must be a permutation of 0..15")
    if sorted(config.permutation_bits_table_64) != list(range(64)):
        errors.append("permutation_bits_table_64 must be a permutation of 0..63")
    for name in ("total_rounds", "basis_number", "dim_truncated"):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            errors.append(f"{name} must be a positive integer, got {value!r}")
    if type(config.begin_round) is not int or config.begin_round < 0:
        errors.append("begin_round must be a nonnegative integer")
    if type(config.weight_range) is not int or config.weight_range <= 0:
        errors.append("weight_range must be a positive integer (the search upper bound is exclusive)")
    if type(config.input_x) is not bool:
        errors.append("input_x must be the Boolean True or False")
    if config.basis_number > 30 or config.dim_truncated > 30:
        errors.append("this pipeline supports at most 30 search/span dimensions")
    if config.state_words != 16 or config.sbox_bits != 4:
        errors.append("RECTANGLE requires 16 four-bit S-boxes")
    if config.trail_model_number != 2:
        errors.append(
            f"trail_model_number must be 2, got {config.trail_model_number}"
        )
    if config.characteristic_model_number not in (2, 3):
        errors.append(
            "characteristic_model_number must be 2 or 3, got "
            f"{config.characteristic_model_number}"
        )
    if config.dim_truncated > config.basis_number:
        errors.append(
            f"dim_truncated ({config.dim_truncated}) must not exceed "
            f"basis_number ({config.basis_number})"
        )
    if not config.diffs:
        errors.append("diffs must contain at least one characteristic")
    if len({tuple(route) for route in config.diffs}) != len(config.diffs):
        errors.append("diffs contains duplicate characteristics (would double-count)")
    if len({(route[0], route[-1]) for route in config.diffs if route}) > 1:
        errors.append("all characteristics in a differential must share endpoints")

    for idx, characteristic in enumerate(config.diffs):
        if len(characteristic) != expected_route_len:
            errors.append(
                f"diffs[{idx}] has {len(characteristic)} entries; "
                f"expected {expected_route_len}"
            )
            continue
        if any(not 0 <= word < (1 << config.state_bits) for word in characteristic):
            errors.append(f"diffs[{idx}] contains a word outside the 64-bit state")
            continue
        for r in range(config.total_rounds):
            offset = r * config.characteristic_model_number
            s_output = characteristic[offset + 1]
            s_input = characteristic[offset]
            for bit in range(0, config.state_bits, 4):
                a, b = (s_input >> bit) & 15, (s_output >> bit) & 15
                if not any(config.sbox[x] ^ config.sbox[x ^ a] == b for x in range(16)):
                    errors.append(f"diffs[{idx}] round {r} nibble {bit // 4}: impossible S-box differential")
            p_output = sum(
                ((s_output >> source) & 1) << target
                for target, source in enumerate(config.permutation_bits_table_64)
            )
            next_input = characteristic[offset + config.characteristic_model_number]
            if p_output != next_input:
                errors.append(f"diffs[{idx}] round {r}: ShiftRow/next input mismatch")
            if config.characteristic_model_number == 3 and characteristic[offset + 2] != p_output:
                errors.append(f"diffs[{idx}] round {r}: stored P output mismatch")

    if errors:
        raise ValueError("Invalid cipher_config.py:\n  - " + "\n  - ".join(errors))


def cpp_string(value):
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def format_characteristics():
    characteristic_blocks = []
    for characteristic in config.diffs:
        values = [f"0x{value:016x}ULL" for value in characteristic]
        rows = [
            "        " + ",".join(values[i:i + 4])
            for i in range(0, len(values), 4)
        ]
        characteristic_blocks.append("    {\n" + ",\n".join(rows) + "\n    }")
    return ",\n".join(characteristic_blocks)


def generated_header_text():
    sbox_cpp = ",".join(str(value) for value in config.sbox)
    permutation_cpp = ",".join(str(value) for value in config.permutation_bits_table_64)
    constants_cpp = ",".join(f"0x{value:016x}ULL" for value in config.round_constants)
    return f"""#pragma once

// Generated from cipher_config.py by run_all.py.
// Do not edit this file manually.
const std::string CIPHER_NAME={cpp_string(config.cipher_name)};
const int SBOX[16]={{{sbox_cpp}}};
const int PERMUTATION_TABLE[64]={{{permutation_cpp}}};
const int ROUND_KEY_BITS={config.round_key_bits};
const int KEY_BITS={config.key_bits};
const uint64_t ROUND_CONSTANTS[{config.total_rounds}]={{{constants_cpp}}};
const int TOTAL_ROUNDS={config.total_rounds};
const int BEGIN_ROUND={config.begin_round};
const int BASIS_NUMBER={config.basis_number};
const int WEIGHT_RANGE={config.weight_range};
const std::string D_STR={cpp_string(config.d_str)};
const int TRAIL_NUMBER={config.trail_model_number};
const int ROUTE_NUMBER={config.characteristic_model_number};

const std::vector<std::vector<uint64_t>> diffs={{
{format_characteristics()}
}};
"""


def update_generated_header():
    content = generated_header_text()
    old_content = None
    if HEADER_PATH.exists():
        old_content = HEADER_PATH.read_text(encoding="utf-8")
    if old_content == content:
        return False
    with HEADER_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return True


def find_compiler():
    configured = os.environ.get("CXX")
    if configured:
        compiler = shutil.which(configured)
        if compiler is None:
            raise FileNotFoundError(
                f"CXX={configured!r} was set, but the compiler was not found"
            )
        return compiler

    for candidate in ("g++", "clang++", "x86_64-w64-mingw32-g++"):
        compiler = shutil.which(candidate)
        if compiler is not None:
            return compiler

    raise FileNotFoundError(
        "No C++ compiler found. Install g++/clang++, or set the CXX environment variable."
    )


def executable_is_stale(header_changed):
    if header_changed or not EXE_PATH.exists():
        return True
    exe_mtime = EXE_PATH.stat().st_mtime
    return CPP_PATH.stat().st_mtime > exe_mtime or HEADER_PATH.stat().st_mtime > exe_mtime


def build_step2(header_changed):
    if not executable_is_stale(header_changed):
        print(f"[build] {EXE_PATH.name} is up to date", flush=True)
        return

    compiler = find_compiler()
    command = [
        compiler,
        "-O3",
        "-std=c++17",
        str(CPP_PATH),
        "-o",
        str(EXE_PATH),
    ]
    print("[build] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def run_command(label, command):
    print(f"\n===== {label} =====", flush=True)
    print(" ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def _stop_pipeline_child(child):
    """Reap only a child started here; never signal unrelated experiments."""
    if child is None:
        return
    if child.poll() is None:
        child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def run_step2_step3_pipeline(step2_command, step3_command):
    """Bounded OS pipe: no Step-2 file, JSON mask list or captured byte array."""
    print("\n===== Step 2 -> Step 3 (direct Fourier stream; no intermediate file) =====", flush=True)
    print(" ".join(map(str, step2_command)), flush=True)
    print(" -> " + " ".join(map(str, step3_command)), flush=True)
    producer = consumer = None
    try:
        producer = subprocess.Popen(
            step2_command, cwd=BASE_DIR, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
        )
        try:
            consumer = subprocess.Popen(
                step3_command, cwd=BASE_DIR, stdin=producer.stdout,
            )
        finally:
            # This reference must close so consumer failure reaches the producer.
            producer.stdout.close()
        while True:
            producer_code, consumer_code = producer.poll(), consumer.poll()
            if producer_code not in (None, 0) or consumer_code not in (None, 0):
                raise RuntimeError(
                    "Direct pipeline failed: Step 2 exit=%s, Step 3 exit=%s; "
                    "see the preceding error (the other child will be stopped)"
                    % (producer_code, consumer_code)
                )
            if producer_code is not None and consumer_code is not None:
                break
            time.sleep(0.05)
    finally:
        _stop_pipeline_child(consumer)
        _stop_pipeline_child(producer)


def print_config():
    print(f"{config.cipher_name} quasidifferential pipeline (independent expanded keys)")
    print(f"  key_bits        = {config.key_bits} ({config.round_key_bits} per round)")
    print(f"  rounds          = {config.total_rounds}")
    print(f"  begin_round     = {config.begin_round}")
    print(f"  characteristics = {len(config.diffs)}")
    print(f"  characteristic_model_number = {config.characteristic_model_number}")
    print(
        "  characteristic_length       = "
        f"{config.characteristic_model_number * config.total_rounds + 1}"
    )
    print(f"  trail_model_number          = {config.trail_model_number}")
    print(
        "  trail_length                = "
        f"{config.trail_model_number * config.total_rounds + 1}"
    )
    print(f"  basis_number    = {config.basis_number}")
    print(f"  dim_truncated   = {config.dim_truncated}")
    print(f"  weight_range    = {config.weight_range}")
    print(f"  input_x         = {config.input_x}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RECTANGLE quasidifferential steps using cipher_config.py"
    )
    parser.add_argument("--skip-step1", action="store_true")
    parser.add_argument("--only-step3", action="store_true",
                        help="legacy small-file regression only; normal runs use the direct pipe")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_config()
    print_config()

    header_changed = update_generated_header()

    if args.only_step3:
        command = [sys.executable, "-u", "-B", str(BASE_DIR / "step3.py")]
        run_command("Step 3", command)
        return

    build_step2(header_changed)

    if not args.skip_step1:
        command = [sys.executable, "-u", "-B", str(BASE_DIR / "step1.py")]
        run_command("Step 1", command)

    step2_command = [
        str(EXE_PATH),
        str(config.dim_truncated),
        str(config.input_x),
    ]
    step3_command = [sys.executable, "-u", "-B", str(BASE_DIR / "step3.py")]
    step2_command.append("--stream")
    step3_command.append("--from-step2-stream")
    run_step2_step3_pipeline(step2_command, step3_command)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
