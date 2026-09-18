import json, math, utils, numpy
from collections import defaultdict
from cipher_config import state_bits, sbox_bits, total_rounds, begin_round, key_bits, d_str, basis_number, weight_range, diffs, dim_truncated, input_x
from cipher_config import characteristic_model_number as route_number
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

def load_quasidiff_jsonl_group_by_dim(file_name, dim):
    meta = None
    trail_groups = {}
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            if obj["type"] == "meta": meta = obj
            elif obj["type"] == "trail":
                trail_id = obj["trail_id"]
                if trail_id >= dim: break
                trail_groups[trail_id] = {
                    "trail_id": trail_id,
                    "weight": obj["weight"],
                    "sign": obj["sign"],
                    "trail": [fixed_hex_to_int(x) for x in obj["trail"]],
                }
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
        if (len(characteristic) // route_number) != total_rounds:
            print("error"); exit()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}]  DC[{idx}]")

        average_w = utils.get_average_w(characteristic, sbox_bits, sbox_bits)
        min_w = average_w
        max_w = min_w + weight_range
        file_name = f"output/record_quasidifferentials/step1_quasidc_search_r_{total_rounds}_{d_str}_c_{idx}_x_{input_x}_basis_number_{basis_number}_w_{min_w}_to_{max_w}.jsonl"

        data = load_quasidiff_jsonl_group_by_dim(file_name, dim_truncated)
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
            one_q_cor, master_k_trail = utils.get_cor_and_key_by_one_quasidifferential(characteristic, trail)
            dc_correlation_counts[one_q_cor] += 1
            all_correlation_counts[one_q_cor] += 1
            if one_q_cor == 0:
                count_trail += 1
                continue
            if master_k_trail not in k_trails:
                k_trails.append(master_k_trail)
                k_coefficients.append(one_q_cor)
                # k_indexes.append([count_trail])
            else:
                pp = k_trails.index(master_k_trail)
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

step_2_combinate_quasidifferentials()
