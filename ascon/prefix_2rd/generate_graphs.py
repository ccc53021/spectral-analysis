"""Rebuild the 64 EVADD graphs used by the Ascon DL.8 computation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numba
import numpy as np
from numba import njit, prange

import utils


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BUILD = ROOT / "build"
PROFILES = DATA / "profiles.json"
MANIFEST = DATA / "graphs.json"
GRAPH_FILE = DATA / "graphs.bin"
SOURCE = ROOT / "route_c_scalar_evadd.c"
SIZE = 32 ** 3
CAPS = {
    "011001": 8_000_000,
    "011011": 16_000_000,
    "101001": 16_000_000,
    "101010": 8_000_000,
    "101011": 28_000_000,
}


def transition_map():
    ids = np.arange(SIZE, dtype=np.int32)
    d1, d2, d3 = ids >> 10, (ids >> 5) & 31, ids & 31
    sbox = np.asarray([utils.sbox_boolean(x) for x in range(32)], dtype=np.int32)
    result = np.empty((SIZE, 32), dtype=np.uint16)
    for x in range(32):
        base = sbox[x]
        result[:, x] = ((base ^ sbox[x ^ d1]) << 10) | ((base ^ sbox[x ^ d2]) << 5) | (base ^ sbox[x ^ d3])
    return result


def projection():
    sources = np.full((64, 11), -1, dtype=np.int32)
    masks = np.zeros((64, 11), dtype=np.int32)
    counts = np.zeros(64, dtype=np.int32)
    for target in range(64):
        groups = {}
        for row, rotations in enumerate(utils.ROTATIONS):
            for distance in (0,) + rotations:
                source = (target - distance) % 64
                groups[source] = groups.get(source, 0) ^ (1 << (4 - row))
        for index, (source, mask) in enumerate(sorted(groups.items())):
            sources[target, index] = source
            masks[target, index] = (mask << 10) | (mask << 5) | mask
        counts[target] = len(groups)
    return sources, masks, counts


@njit(cache=True)
def initial_support(states, transitions, parities):
    support = np.zeros(SIZE, dtype=np.bool_)
    for state in states:
        base = transitions[state, 0]
        directions = np.zeros(5, dtype=np.uint16)
        for index in range(5):
            directions[index] = base ^ transitions[state, 1 << index]
        for mask in range(SIZE):
            if support[mask]:
                continue
            valid = True
            for index in range(5):
                if parities[mask & directions[index]]:
                    valid = False
                    break
            if valid:
                support[mask] = True
        if np.all(support):
            break
    return support


def lat_tables():
    lat = np.array([
        utils.fwht([1 - 2 * ((utils.sbox_boolean(x) & mask).bit_count() & 1) for x in range(32)])
        for mask in range(32)
    ], dtype=np.int64)
    rows = np.asarray([
        sum(1 << value for value in range(32) if lat[mask, value])
        for mask in range(32)
    ], dtype=np.uint32)
    pairs = np.zeros((32, 32, 32), dtype=np.uint32)
    for first in range(32):
        for second in range(32):
            for delta in range(32):
                pairs[first, second, delta] = sum(
                    1 << value for value in range(32)
                    if lat[second, value] and lat[first, value ^ delta]
                )
    return rows, pairs


@njit(cache=True, parallel=True)
def support_sbox(tensor, rows, pairs):
    result = np.zeros_like(tensor)
    for col in prange(tensor.shape[0]):
        packed = np.zeros((32, 32), dtype=np.uint32)
        for first in range(32):
            for second in range(32):
                for third in range(32):
                    if tensor[col, (first << 10) | (second << 5) | third]:
                        packed[second, third] |= np.uint32(1) << np.uint32(first)
        intermediate = np.zeros((32, 32, 32), dtype=np.uint32)
        combined = np.zeros((32, 32, 32), dtype=np.uint32)
        for total in range(32):
            for first in range(32):
                for third in range(32):
                    value = np.uint32(0)
                    for second in range(32):
                        if packed[second, third] & pairs[total, first, second ^ third]:
                            value |= np.uint32(1) << np.uint32(second)
                    intermediate[total, first, third] = value
                for second in range(32):
                    value = np.uint32(0)
                    for third in range(32):
                        if intermediate[total, first, third] & rows[second]:
                            value |= np.uint32(1) << np.uint32(third)
                    combined[total, first, second] = value
        for first in range(32):
            for second in range(32):
                for third in range(32):
                    result[col, (first << 10) | (second << 5) | third] = (
                        combined[first ^ second ^ third, first, second] & rows[third]
                    ) != 0
    return result


@njit(cache=True, parallel=True)
def support_linear(tensor, sources, masks, counts):
    result = np.ones_like(tensor)
    for col in prange(64):
        for mask in range(SIZE):
            for index in range(counts[col]):
                if not tensor[sources[col, index], mask & masks[col, index]]:
                    result[col, mask] = False
                    break
    return result


def support_bounds(profile, transitions, parities, rows, pairs, sources, masks, counts):
    tensor = np.array([
        initial_support(np.asarray(column["states"], dtype=np.uint16), transitions, parities)
        for column in profile["endpoint_column_affine_sets"]
    ])
    stages = [tensor]
    for round_index in range(3, 6):
        if round_index > 3:
            tensor = support_sbox(tensor, rows, pairs)
            stages.append(tensor)
        tensor = support_linear(tensor, sources, masks, counts)
        stages.append(tensor)
    result = np.stack(stages)
    if result.shape != (6, 64, SIZE):
        raise RuntimeError(f"unexpected support shape: {result.shape}")
    return result


def xor_words(left, right):
    return tuple(a ^ b for a, b in zip(left, right))


def forward_linear(words):
    return tuple(
        word ^ utils._ror(word, first) ^ utils._ror(word, second)
        for word, (first, second) in zip(utils.parse_words(words), utils.ROTATIONS)
    )


def unpack_local(value):
    return value >> 10, (value >> 5) & 31, value & 31


def endpoint(profile, label):
    words = [(0, 0, 0, 0, 0) for _ in range(3)]
    for item in profile["local"]:
        local_index = (label >> item["label_offset"]) & ((1 << item["label_dimension"]) - 1)
        outputs = unpack_local(item["second_outputs_packed"][local_index])
        for index, value in enumerate(outputs):
            words[index] = xor_words(words[index], utils.column_words(value, item["column"]))
    return tuple(forward_linear(value) for value in words)


def final_terms(transitions):
    kernel = np.array([
        sum(
            1 - 2 * ((((int(value) >> 10) ^ ((int(value) >> 5) & 31) ^ int(value)) & 16) != 0)
            for value in transitions[index]
        ) / 32
        for index in range(SIZE)
    ])
    transformed = np.asarray(utils.fwht(np.rint(32 * kernel).astype(np.int64).tolist()), dtype=np.int64)
    final = np.flatnonzero(transformed).astype(np.uint32)
    if len(final) != 64:
        raise RuntimeError("unexpected final kernel support")
    return final, transformed[final].astype(np.float64) / (32 * SIZE)


def write_input(path, profile, bounds, transitions, final, final_weights):
    dimension = profile["second_label_dimension"]
    base = endpoint(profile, 0)
    basis = [endpoint(profile, 1 << index) for index in range(dimension)]
    constants = np.zeros((64, 5), dtype=np.uint16)
    directions = np.zeros((64, dimension, 5), dtype=np.uint16)
    phase_base = np.zeros(64, dtype=np.uint16)
    phase_directions = np.zeros((64, dimension), dtype=np.uint16)
    phase_quadratic = np.zeros((64, dimension * (dimension - 1) // 2), dtype=np.uint16)

    def packed(words, col):
        first, second, third = (utils.column(value, col) for value in words)
        return (first << 10) | (second << 5) | third

    def increments(words, col):
        state = packed(words, col)
        return np.asarray([transitions[state, 0] ^ transitions[state, 1 << index] for index in range(5)], dtype=np.uint16)

    for col in range(64):
        constants[col] = increments(base, col)
        base_state = packed(base, col)
        points = [packed(value, col) for value in basis]
        for index in range(dimension):
            directions[col, index] = increments(basis[index], col) ^ constants[col]
            phase_directions[col, index] = transitions[points[index], 0] ^ transitions[base_state, 0]
        phase_base[col] = transitions[base_state, 0]
        phase_quadratic[col] = np.asarray([
            transitions[points[i] ^ points[j] ^ base_state, 0]
            ^ transitions[points[i], 0] ^ transitions[points[j], 0] ^ transitions[base_state, 0]
            for i in range(dimension) for j in range(i + 1, dimension)
        ], dtype=np.uint16)

    with path.open("wb") as stream:
        np.asarray([0x243ADD15, dimension, 64, SIZE, 6, 64], dtype=np.uint32).tofile(stream)
        bounds.astype(np.uint8).tofile(stream)
        constants.tofile(stream)
        directions.tofile(stream)
        phase_base.tofile(stream)
        phase_directions.tofile(stream)
        phase_quadratic.tofile(stream)
        final.tofile(stream)
        final_weights.tofile(stream)


def compiler():
    requested = os.environ.get("CC")
    candidate = shutil.which(requested) if requested else shutil.which("gcc")
    if not candidate:
        raise RuntimeError("GCC was not found; install GCC or set the CC environment variable")
    return candidate


def compile_backend(cap):
    BUILD.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""
    target = BUILD / f"route_c_scalar_evadd_{cap}_{utils.sha(SOURCE)[:12]}{suffix}"
    if target.exists():
        return target
    unique_power = (2 * cap - 1).bit_length()
    command = [
        compiler(), "-O3", "-std=c11", "-DROUTE_C_LABEL_BITS=63",
        f"-DNODE_CAP={cap}", f"-DUNIQUE_SIZE=(1u<<{unique_power})",
        str(SOURCE), "-lm", "-o", str(target),
    ]
    run = subprocess.run(command, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(f"C compilation failed:\n{run.stderr}")
    return target


def run_branch(profile, expected, context, output):
    name = profile["branch_bits"]
    started = time.perf_counter()
    bounds = support_bounds(profile, *context[:7])
    input_file = BUILD / f"input_{name}.bin"
    write_input(input_file, profile, bounds, context[0], context[-2], context[-1])
    cap = CAPS.get(name, 4_000_000)
    executable = compile_backend(cap)
    command = [str(executable), str(input_file), "60", "65536", str(output)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=90)
    if run.returncode:
        raise RuntimeError(f"branch {name} failed:\n{run.stdout}\n{run.stderr}")
    result = json.loads(run.stdout)
    if not result.get("complete"):
        raise RuntimeError(f"branch {name} was incomplete: {result.get('reason', 'unknown reason')}")
    actual_size = output.stat().st_size
    actual_hash = utils.sha(output)
    if actual_size != expected["size"] or actual_hash != expected["sha256"]:
        raise RuntimeError(
            f"branch {name} differs from the archived graph: "
            f"size {actual_size}/{expected['size']}, sha256 {actual_hash}/{expected['sha256']}"
        )
    input_file.unlink(missing_ok=True)
    print(f"{name}: verified in {time.perf_counter() - started:.1f}s", flush=True)


def prepare_context():
    numba.set_num_threads(min(4, numba.config.NUMBA_NUM_THREADS))
    transitions = transition_map()
    parities = np.asarray([value.bit_count() & 1 for value in range(SIZE)], dtype=np.uint8)
    rows, pairs = lat_tables()
    sources, masks, counts = projection()
    final, final_weights = final_terms(transitions)
    return transitions, parities, rows, pairs, sources, masks, counts, final, final_weights


def verify_existing(manifest):
    if not GRAPH_FILE.exists():
        return False
    expected_size = sum(item["size"] for item in manifest.values())
    if GRAPH_FILE.stat().st_size != expected_size:
        return False
    return all(
        utils.sha_range(GRAPH_FILE, item["offset"], item["size"]) == item["sha256"]
        for item in manifest.values()
    )


def build_all(profiles, manifest, context):
    if verify_existing(manifest):
        print(f"{GRAPH_FILE} is already complete")
        return
    BUILD.mkdir(parents=True, exist_ok=True)
    partial = BUILD / "graphs.bin.partial"
    state_file = BUILD / "graphs.state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {"completed": []}
    completed = state["completed"]
    names = list(manifest)
    if completed != names[:len(completed)]:
        raise RuntimeError("partial graph state does not contain a valid branch prefix")
    expected_prefix = sum(manifest[name]["size"] for name in completed)
    if not partial.exists() or partial.stat().st_size != expected_prefix:
        if completed or partial.exists():
            raise RuntimeError("partial graph state is inconsistent; remove build/graphs.bin.partial and build/graphs.state.json")
        partial.touch()

    by_name = {profile["branch_bits"]: profile for profile in profiles}
    for name in manifest:
        if name in completed:
            continue
        part = BUILD / f"graph_{name}.bin"
        run_branch(by_name[name], manifest[name], context, part)
        with partial.open("ab") as destination, part.open("rb") as source:
            shutil.copyfileobj(source, destination, length=1 << 20)
        part.unlink()
        completed.append(name)
        utils.dump(state_file, {"completed": completed})
    if not verify_container(partial, manifest):
        raise RuntimeError("assembled graph container failed verification")
    os.replace(partial, GRAPH_FILE)
    state_file.unlink(missing_ok=True)
    print(f"created {GRAPH_FILE} ({GRAPH_FILE.stat().st_size} bytes)")


def verify_container(path, manifest):
    return path.stat().st_size == sum(item["size"] for item in manifest.values()) and all(
        utils.sha_range(path, item["offset"], item["size"]) == item["sha256"]
        for item in manifest.values()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", help=argparse.SUPPRESS)
    args = parser.parse_args()
    profiles = json.loads(PROFILES.read_text(encoding="utf-8"))["rows"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if [profile["branch_bits"] for profile in profiles] != list(manifest):
        raise RuntimeError("profile and graph manifests use different branch orders")
    offset = 0
    for metadata in manifest.values():
        if metadata["offset"] != offset:
            raise RuntimeError("graph manifest contains a noncontiguous offset")
        offset += metadata["size"]
    context = prepare_context()
    if args.branch:
        matches = [profile for profile in profiles if profile["branch_bits"] == args.branch]
        if not matches:
            raise ValueError(f"unknown branch: {args.branch}")
        BUILD.mkdir(parents=True, exist_ok=True)
        output = BUILD / f"graph_{args.branch}.check.bin"
        run_branch(matches[0], manifest[args.branch], context, output)
        output.unlink()
        return
    build_all(profiles, manifest, context)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
