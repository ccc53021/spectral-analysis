#!/usr/bin/env python3
"""Build the binary manifest consumed by the ASCON v3 C++ step2."""

import os
import struct
import sys

from cipher_config import (
    basis_number,
    begin_round,
    characteristic_model_number,
    d_str,
    diffs,
    dim_truncated,
    IV,
    m,
    mode,
    n,
    total_rounds,
    trail_model_number,
    weight_range,
)
import utils


OUTPUT_DIR = "output/record_trails_and_coefficients"
DEFAULT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "step2_input_data.bin")
MAGIC = 0x32564341  # "ACV2" in little-endian
VERSION = 2


def write_i32(f_out, value):
    f_out.write(struct.pack("<i", int(value)))


def write_string(f_out, value):
    encoded = value.encode("utf-8")
    write_i32(f_out, len(encoded))
    f_out.write(encoded)


def write_u64(f_out, value):
    f_out.write(struct.pack("<Q", int(value)))


def write_word5(f_out, value):
    for row in range(5):
        word = (value >> (64 * row)) & 0xFFFFFFFFFFFFFFFF
        f_out.write(struct.pack("<Q", word))


def validate_config():
    characteristic_len = characteristic_model_number * total_rounds + 1
    trail_len = trail_model_number * total_rounds + 1

    if dim_truncated > basis_number:
        raise ValueError(
            f"dim_truncated={dim_truncated} exceeds basis_number={basis_number}"
        )
    if characteristic_len <= 0 or trail_len <= 0:
        raise ValueError("invalid characteristic/trail length")

    for idx, route in enumerate(diffs):
        flat = utils.trail_dict_to_flat_320(route)
        if len(flat) != characteristic_len:
            raise ValueError(
                f"diffs[{idx}] has {len(flat)} characteristic states; "
                f"expected {characteristic_len}"
            )

    return characteristic_len, trail_len


def main():
    if len(sys.argv) > 2:
        raise SystemExit(
            f"Usage: {sys.argv[0]} [output_manifest.bin]"
        )

    output_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_OUTPUT_PATH
    output_dir = os.path.dirname(output_path) or "."
    characteristic_len, trail_len = validate_config()
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f_out:
        f_out.write(struct.pack("<I", MAGIC))
        write_i32(f_out, VERSION)
        write_i32(f_out, total_rounds)
        write_i32(f_out, begin_round)
        write_i32(f_out, characteristic_model_number)
        write_i32(f_out, trail_model_number)
        write_i32(f_out, basis_number)
        write_i32(f_out, weight_range)
        write_i32(f_out, dim_truncated)
        write_string(f_out, d_str)
        write_string(f_out, mode)
        write_u64(f_out, IV)
        write_i32(f_out, len(diffs))

        for idx, route in enumerate(diffs):
            characteristic = utils.trail_dict_to_flat_320(route)
            characteristic_cor = float(route["cor_rb"])
            average_w = utils.get_average_w(characteristic, n, m, mode)
            min_w = average_w
            max_w = min_w + weight_range
            step1_path = (
                "output/record_quasidifferentials/"
                f"step1_quasidc_search_r_{total_rounds}_{d_str}"
                f"_mode_{mode}_c_{idx}"
                f"_basis_number_{basis_number}_w_{min_w}_to_{max_w}.jsonl"
            )

            f_out.write(struct.pack("<d", characteristic_cor))
            write_i32(f_out, len(characteristic))
            for state in characteristic:
                write_word5(f_out, state)
            write_string(f_out, step1_path)

    print(f"Written: {output_path}")
    print(f"  characteristics             = {len(diffs)}")
    print(f"  total_rounds                = {total_rounds}")
    print(f"  characteristic_model_number = {characteristic_model_number}")
    print(f"  trail_model_number          = {trail_model_number}")
    print(f"  characteristic_len          = {characteristic_len}")
    print(f"  trail_len                   = {trail_len}")
    print(f"  basis_number                = {basis_number}")
    print(f"  dim_truncated               = {dim_truncated}")
    print(f"  mode                        = {mode}")
    print(f"  IV                          = 0x{IV:016x}")


if __name__ == "__main__":
    main()
