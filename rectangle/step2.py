"""Small-dimension Python reference; production runs use C++ streaming Step 2."""

import sys

if __name__ == "__main__":
    sys.excepthook = sys.__excepthook__

import json, math, utils, numpy
from collections import defaultdict
from cipher_config import state_bits, sbox_bits, total_rounds, begin_round, key_bits, d_str, basis_number, weight_range, diffs, dim_truncated, input_x
from cipher_config import characteristic_model_number as route_number
from cipher_config import cipher_name, key_model, round_key_bits
from cipher_config import state_words, pro_bit, permutation_bits_table_64
from cipher_config import trail_model_number as trail_number
from datetime import datetime

def fixed_hex_to_int(x):
    if isinstance(x, int): return x
    if x.startswith("0x") or x.startswith("0X"): return int(x, 16)
    return int(x)


def print_correlation_distribution(counts, title):
    print()
    print(title)
    print(f"  distinct correlations: {len(counts)}")

    nonzero_records = [
        (cor, count)
        for cor, count in counts.items()
        if cor != 0
    ]
    zero_count = counts.get(0.0, 0)

    nonzero_records.sort(
        key=lambda item: (
            round(-math.log2(abs(item[0]))),
            item[0]
        )
    )

    for cor, count in nonzero_records:
        weight = round(-math.log2(abs(cor)))
        print(
            f"  cor={cor:+.17e}"
            f"  count={count}"
            f"  -log2(abs(cor))={weight}"
        )

    if zero_count:
        print(
            f"  cor={0.0:+.17e}"
            f"  count={zero_count}"
            f"  -log2(abs(cor))=infinity"
        )

def _validate_reference_parameters(dim):
    if type(dim) is not int or not 1 <= dim <= 20:
        raise ValueError("Python Step 2 is a reference only: dim must be in 1..20; use C++ streaming for larger runs")
    if type(basis_number) is not int or basis_number < dim:
        raise ValueError("basis_number must be a positive integer at least as large as dim")
    if type(weight_range) is not int or weight_range <= 0:
        raise ValueError("weight_range must be a positive integer")


def _validate_characteristic(characteristic):
    expected_length = route_number * total_rounds + 1
    if len(characteristic) != expected_length:
        raise ValueError(f"Characteristic has {len(characteristic)} entries; expected {expected_length}")
    if any(type(word) is not int or not 0 <= word < (1 << state_bits) for word in characteristic):
        raise ValueError("Characteristic words must be unsigned state-width integers")
    for r in range(total_rounds):
        offset = route_number * r
        permuted = sum(
            ((characteristic[offset + 1] >> source) & 1) << target
            for target, source in enumerate(permutation_bits_table_64)
        )
        if permuted != characteristic[route_number * (r + 1)]:
            raise ValueError(f"Characteristic permutation mismatch at round {r}")
        if route_number == 3 and permuted != characteristic[offset + 2]:
            raise ValueError(f"Characteristic stored permutation output mismatch at round {r}")
    return utils.get_average_w(characteristic, sbox_bits, sbox_bits)


def load_quasidiff_jsonl_group_by_dim(file_name, dim, characteristic=None):
    """Validate the entire Step 1 record, then retain the requested prefix."""
    _validate_reference_parameters(dim)
    meta = None
    trail_groups = {}
    count = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Step 1 line {line_number} must be an object")
            if obj.get("type") == "meta":
                if meta is not None or count:
                    raise ValueError("Step 1 must contain exactly one initial metadata record")
                meta = obj
                recorded_route = [fixed_hex_to_int(value) for value in meta.get("differential_route", [])]
                if characteristic is None:
                    if recorded_route not in diffs:
                        raise ValueError("Step 1 differential route does not match configured characteristics")
                    characteristic = recorded_route
                if recorded_route != characteristic:
                    raise ValueError("Step 1 differential route does not match the requested characteristic")
                average_w = _validate_characteristic(characteristic)
                expected = {
                    "version": 1, "cipher": cipher_name, "key_model": key_model,
                    "round_key_bits": round_key_bits, "key_bits": key_bits,
                    "total_rounds": total_rounds, "begin_round": begin_round,
                    "state_bits": state_bits, "state_words": state_words,
                    "sbox_bits": sbox_bits, "pro_bit": pro_bit,
                    "input_x": input_x, "basis_number": basis_number,
                    "average_weight": average_w, "min_weight": average_w,
                    "max_weight": average_w + weight_range,
                }
                for name, value in expected.items():
                    if type(meta.get(name)) is not type(value) or meta[name] != value:
                        raise ValueError(f"Step 1 metadata mismatch for {name}: expected {value!r}, got {meta.get(name)!r}")
            elif obj.get("type") == "trail":
                if meta is None:
                    raise ValueError("Step 1 metadata must precede all trails")
                trail_id = obj.get("trail_id")
                if type(trail_id) is not int or trail_id != count or count >= basis_number:
                    raise ValueError(f"Step 1 trail IDs must be unique and consecutive: expected {count}, got {trail_id!r}")
                trail = [fixed_hex_to_int(value) for value in obj.get("trail", [])]
                expected_length = trail_number * total_rounds + 1
                if obj.get("trail_len") != expected_length or len(trail) != expected_length:
                    raise ValueError(f"Step 1 trail {trail_id} has an invalid length")
                if any(type(word) is not int or not 0 <= word < (1 << state_bits) for word in trail):
                    raise ValueError(f"Step 1 trail {trail_id} contains an invalid state word")
                weight, sign = obj.get("weight"), obj.get("sign")
                if type(weight) is not int or not meta["min_weight"] <= weight < meta["max_weight"]:
                    raise ValueError(f"Step 1 trail {trail_id} has an invalid weight")
                if type(sign) is not int or sign not in (0, 1):
                    raise ValueError(f"Step 1 trail {trail_id} has an invalid sign")
                if trail_id < dim:
                    trail_groups[trail_id] = {
                        "trail_id": trail_id,
                        "weight": weight,
                        "sign": sign,
                        "trail": trail,
                    }
                count += 1
            else:
                raise ValueError(f"Unknown Step 1 record type at line {line_number}")
    if meta is None:
        raise ValueError("Step 1 metadata is missing")
    if count != basis_number:
        raise ValueError(f"Incomplete Step 1 basis: expected {basis_number} trails, found {count}")
    return {"meta": meta, "trails": trail_groups}


def span_trails(basis_trails):
    """
    Generate all 2^d XOR-combined trails in Gray-code order.
    basis_trails contains [m_a0, m_b0, m_a1, m_b1, ..., m_aR],
    with each element represented as a 64-bit integer.
    Yield (combo_idx, trail) for every combination. Combination zero is
    the all-zero trail, and combination 1<<j is basis_trails[j].
    """
    d = len(basis_trails)
    N = 1 << d
    trail_len = len(basis_trails[0])

    # Start with the all-zero combined trail.
    cur = [0] * trail_len
    yield 0, cur[:]

    # Cache trails for reuse by the Gray-code traversal.
    saved: list = [None] * N
    saved[0] = cur[:]

    for j in range(d):
        step = 1 << j
        bj = basis_trails[j]
        for i in range(step):
            prev = saved[i]
            new_trail = [prev[k] ^ bj[k] for k in range(trail_len)]
            idx = step + i
            saved[idx] = new_trail
            yield idx, new_trail


def span_trails_low_mem(basis_trails):
    d = len(basis_trails)
    N = 1 << d
    trail_len = len(basis_trails[0])
    for combo in range(N):
        trail = [0] * trail_len
        for j in range(d):
            if (combo >> j) & 1:
                for k in range(trail_len):
                    trail[k] ^= basis_trails[j][k]
        yield combo, trail


def step_2_combinate_quasidifferentials():
    _validate_reference_parameters(dim_truncated)
    if not diffs:
        raise ValueError("At least one characteristic is required")
    for characteristic in diffs:
        _validate_characteristic(characteristic)
    str_split = "-" * 100
    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Step 2. Span & coefficients, input x = {input_x} basis = {basis_number}, dim = {dim_truncated}")

    k_trails = []
    k_coefficients = []
    all_correlation_counts = defaultdict(int)
    # k_indexes = []

    for idx in range(len(diffs)):
        characteristic = diffs[idx]
        dc_correlation_counts = defaultdict(int)

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}]  DC[{idx}]")

        average_w = utils.get_average_w(characteristic, sbox_bits, sbox_bits)
        min_w = average_w
        max_w = min_w + weight_range
        file_name = f"output/record_quasidifferentials/step1_quasidc_search_r_{total_rounds}_{d_str}_c_{idx}_x_{input_x}_basis_number_{basis_number}_w_{min_w}_to_{max_w}.jsonl"

        data = load_quasidiff_jsonl_group_by_dim(file_name, dim_truncated, characteristic)
        meta = data["meta"]
        trail_dict = data["trails"]
        bases = [trail_dict[tid] for tid in sorted(trail_dict.keys())]
        d = len(bases)
        print(f"DC[{idx}] : Loaded {d} trails, span size = 2^{d} = {1<<d}")

        # span
        basis_trails = [b["trail"] for b in bases]
        count_trail = 0
        count_valid = 0
        for combo_idx, trail in span_trails_low_mem(basis_trails):
            one_q_cor, expanded_k_trail = utils.get_cor_and_key_by_one_quasidifferential(characteristic, trail)
            dc_correlation_counts[one_q_cor] += 1
            all_correlation_counts[one_q_cor] += 1
            if one_q_cor == 0:
                count_trail += 1
                continue
            if expanded_k_trail not in k_trails:
                k_trails.append(expanded_k_trail)
                k_coefficients.append(one_q_cor)
                # k_indexes.append([count_trail])
            else:
                pp = k_trails.index(expanded_k_trail)
                k_coefficients[pp] += one_q_cor
                # k_indexes[pp].append(count_trail)

            if one_q_cor != 0:
                count_valid += 1
            count_trail += 1

        print(f"DC[{idx}] : Spanned {count_trail} trails, {count_valid} valid trails")
        print(str_split)
        print()

        print(f"After DC[{idx}]: {len(k_coefficients)} k_coefficients")
        print(str_split)
        print()

        print_correlation_distribution(
            dc_correlation_counts,
            f"DC[{idx}] quasidifferential correlation distribution:"
        )

    print_correlation_distribution(
        all_correlation_counts,
        "All quasidifferential correlation distribution:"
    )

    print(str_split)
    print(str_split)
    zero_mask = [0] * (key_bits + state_bits)
    try:
        print(f"Averaged-key differential = C_[uk=0, ux=0] = 2^{numpy.log2(k_coefficients[k_trails.index(zero_mask)])}")
    except ValueError:
        print(f"WARNING: C_[uk=0, ux=0] not in trail set (combo=0 should produce it)")
    print(str_split)
    print(str_split)
    print()

    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Save bases and coefficients")
    print(str_split)

    import os
    os.makedirs("output/record_trails_and_coefficients", exist_ok=True)
    record_path = f"output/record_trails_and_coefficients/step2_save_trails_and_coefficients_r_{total_rounds}_{d_str}_x_{input_x}_basis_number_{basis_number}_weight_{weight_range}_dim_{dim_truncated}.jsonl"
    record_cp = {
        "cipher": cipher_name,
        "key_model": key_model,
        "round_key_bits": round_key_bits,
        "key_bits": key_bits,
        "mask_bits": key_bits + state_bits,
        "total_rounds": total_rounds,
        "begin_round": begin_round,
        "input_x": input_x,
        "basis_number": basis_number,
        "weight": weight_range,
        "dim": dim_truncated,
        "k_trails": k_trails,
        "k_coefficients": k_coefficients,
        # "k_indexes": k_indexes
    }
    with open(record_path, "w") as f:
        f.write(json.dumps(record_cp) + "\n")
    print(str_split)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}]")
    print(f"Bases and coefficients saved to: {record_path}")

if __name__ == "__main__":
    step_2_combinate_quasidifferentials()
