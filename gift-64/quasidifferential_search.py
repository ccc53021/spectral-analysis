# !/user/bin/env python3

# version 1, basic model
# version 2, use basis

import pyboolector
import json
from cipher_config import cipher_name, sbox, n, m, state_bits, state_words, sbox_bits, pro_bit, total_rounds, begin_round, round_constants, permutation_bits_table_64
from cipher_config import trail_model_number as trail_number
from cipher_config import characteristic_model_number as route_number
import utils

def int_to_fixed_hex(x, bits):
    width = (bits + 3) // 4
    return "0x{:0{}x}".format(x, width)

def list_to_fixed_hex(values, bits):
    return [int_to_fixed_hex(v, bits) for v in values]

def write_quasidiff_meta_jsonl(file_name, differential_route, average_w, min_w, max_w, basis_number, input_x):
    meta = {
        "type": "meta",
        "version": 1,
        "cipher": cipher_name,
        "total_rounds": total_rounds,
        "begin_round": begin_round,
        "state_bits": state_bits,
        "state_words": state_words,
        "sbox_bits": sbox_bits,
        "pro_bit": pro_bit,
        "input_x": input_x,
        "basis_number": basis_number,
        "average_weight": average_w,
        "min_weight": min_w,
        "max_weight": max_w,
        "differential_route": list_to_fixed_hex(differential_route, state_bits)
    }
    with open(file_name, "w") as f:
        f.write(json.dumps(meta) + "\n")

def append_quasidiff_record_jsonl(f_out, trail_id, weight, sign, trail):
    record = {
        "type": "trail",
        "trail_id": trail_id,
        "weight": weight,
        "sign": sign,
        "trail_len": len(trail),
        "trail": list_to_fixed_hex(trail, state_bits)
    }
    f_out.write(json.dumps(record) + "\n")
    f_out.flush()

def quasi_differential_search(differential_route, average_w, min_w, max_w, basis_number, input_x, file_name):
    btor = pyboolector.Boolector()
    btor.Set_opt(pyboolector.BtorOption.BTOR_OPT_MODEL_GEN, 1)
    btor.Set_opt(pyboolector.BtorOption.BTOR_OPT_INCREMENTAL, 1)

    # input
    m_a = [btor.Var(btor.BitVecSort(state_bits), "m_a_{}".format(i)) for i in range(total_rounds + 1)]
    # after s and c
    m_b = [btor.Var(btor.BitVecSort(state_bits), "m_b_{}".format(i)) for i in range(total_rounds)]

    # sign
    s_0 = [btor.Var(btor.BitVecSort(state_words), "s_0_{}".format(i)) for i in range(total_rounds)]
    s0_all = btor.Var(btor.BitVecSort(1), "s0_all")

    w_0 = [btor.Var(btor.BitVecSort(state_words * pro_bit), "w_0_{}".format(i)) for i in range(total_rounds)]
    w0_all = btor.Var(btor.BitVecSort(pro_bit), "w0_all")

    def permute_bits(x, y, table, bits):
        for i in range(bits):
            btor.Assert(y[i] == x[table[i]])

    def is_in_table(u, v, table, n, m):
        condition = btor.Const(0, 1)
        for ii in table:
            uu = ii[0]
            vv = ii[1]
            match = (u == btor.Const(uu, n)) & (v == btor.Const(vv, m))
            condition |= match

        return condition

    def set_sbox(x, y, s, w, w_positive, solutions_positive, w_negative, solutions_negative):
        if w_positive or w_negative:
            constraint_w = btor.Const(0, 1)
            for idx, ww in enumerate(w_positive):
                wp_solutions = solutions_positive[idx]
                in_positive_table = is_in_table(x, y, wp_solutions, n, m)
                constraint_w |= (in_positive_table & (s == 0b0) & (w == int(ww)))
            for idx, ww in enumerate(w_negative):
                wn_solutions = solutions_negative[idx]
                in_negative_table = is_in_table(x, y, wn_solutions, n, m)
                constraint_w |= (in_negative_table & (s == 0b1) & (w == int(ww)))

            btor.Assert(constraint_w)

    def set_fixed_difference(state, difference):
        for i in range(state_bits):
            btor.Assert(state[i] == ((difference >> i) & 0x1))

    # def add_basis_constraints(current_trails, current_w, count_time):
    #
    #     # u_trails = []
    #
    #     import basis_generate as bg
    #
    #     print()
    #     print(f"time = {count_time}, current weight = {current_w}, {len(current_trails)} trails")
    #     u_vectors = bg.get_vectors_by_current_trails(current_trails, total_rounds, state_bits, trail_number)
    #     print("number of u_trails : {}".format(len(u_vectors)))
    #     u_basis, b_basis = bg.generate_b_basis(u_vectors, total_rounds, state_bits)
    #     dim_u = len(u_basis)
    #     dim_b = len(b_basis)
    #     # dim_all = state_bits * (total_rounds - 1)
    #     dim_all = state_bits * (total_rounds + 1)
    #     print("dim u = {}, dim b = {}, dim all = {}".format(dim_u, dim_b, dim_all))
    #     if dim_all != dim_u + dim_b:
    #         print("dim error ! exit")
    #         exit()
    #     print("number of u_basis : {}".format(len(u_basis)))
    #     # for i in range(len(u_basis)):
    #     #     print(i, u_basis[i])
    #     print("number of b_basis : {}".format(len(b_basis)))
    #     # for i in range(len(b_basis)):
    #     #     print(i, b_basis[i])
    #
    #
    #     coefficients_b_0 = [btor.Var(btor.BitVecSort(1), "c_b_0_time_{}_{}".format(count_time, j)) for j in range(dim_b)]
    #
    #     bb_0 = [btor.Const(0, 1) for i in range(dim_all)]
    #     for i in range(dim_all):
    #         for j in range(dim_b):
    #             bb_0[i] ^= btor.Cond(coefficients_b_0[j] == btor.Const(1, 1), btor.Const(b_basis[j][i], 1), btor.Const(0, 1))
    #
    #     # # u0_all = btor.Concat(mask_a[total_rounds - 1], mask_a[total_rounds - 2])
    #     # u0_all = btor.Concat(mask_a[total_rounds], mask_a[total_rounds - 1])
    #     # # for i in range(total_rounds - 3, 0, -1):
    #     # for i in range(total_rounds - 2, -1, -1):
    #     #     u0_all = btor.Concat(u0_all, mask_a[i])
    #     # for i in range(len(bb_0)):
    #     #     btor.Assert(u0_all[i] == bb_0[i])
    #
    #     return bb_0

    def constrain_candidate_outside_current_span(current_trails, current_w, count_time):
        print(f"time = {count_time}, current weight = {current_w}, {len(current_trails)} trails")

        from sage.all import GF, Matrix

        dim_all = state_bits * (total_rounds + 1)

        if not current_trails:
            return u0_all != btor.Const(0, dim_all)

        u_vectors = []

        for trail in current_trails:
            vec = []

            for r in range(total_rounds + 1):
                m_a_value = trail[trail_number * r]

                for bit in range(state_bits):
                    vec.append((m_a_value >> bit) & 1)

            u_vectors.append(vec)

        R = Matrix(GF(2), u_vectors).rref()


        basis_rows = []
        pivot_cols = []

        for row_idx in range(R.nrows()):
            row = R[row_idx]

            if row.is_zero():
                continue

            pivot = next(
                col for col in range(R.ncols())
                if row[col] == 1
            )

            basis_rows.append(row)
            pivot_cols.append(pivot)

        pivot_set = set(pivot_cols)
        outside_span = btor.Const(0, 1)

        for col in range(dim_all):
            if col in pivot_set:
                continue

            expected = btor.Const(0, 1)

            for row_idx, pivot in enumerate(pivot_cols):
                if basis_rows[row_idx][col] == 1:
                    expected ^= u0_all[pivot]

            outside_span |= u0_all[col] ^ expected

        rank = len(basis_rows)

        if rank != len(current_trails):
            raise RuntimeError(
                f"Basis rank mismatch: "
                f"{len(current_trails)} stored trails, rank={rank}"
            )

        return outside_span == btor.Const(1, 1)

    list_ab = []
    list_qab = []

    for i in range(total_rounds):
        rc = round_constants[i]
        l_rc = [(rc >> 0) & 0b1, (rc >> 1) & 0b1, (rc >> 2) & 0b1, (rc >> 3) & 0b1, (rc >> 4) & 0b1,
                (rc >> 5) & 0b1, 1]
        round_c = 0
        if i != 0:
            round_c = utils.get_constant(l_rc)
        for j in range(0, state_bits, sbox_bits):
            d_aa = (differential_route[route_number * i] >> j) & 0xf
            d_bb = (differential_route[route_number * i + 1] >> j) & 0xf
            cc = (round_c >> j) & 0xf
            m_aa = m_a[i][j + sbox_bits - 1:j]
            m_bb = m_b[i][j + sbox_bits - 1:j]
            ss0 = s_0[i][j // sbox_bits]
            ww0 = w_0[i][(j // sbox_bits + 1) * pro_bit - 1:(j // sbox_bits) * pro_bit]

            if [d_aa, d_bb, cc] not in list_ab:
                list_ab.append([d_aa, d_bb, cc])
                q_ab = utils.get_quasidifferential_matrix_by_a_and_b_and_c(sbox, n, m, d_aa, d_bb, cc)
                list_qab.append(q_ab)
                w_positive, solutions_positive, w_negative, solutions_negative = q_ab
                set_sbox(m_aa, m_bb, ss0, ww0, w_positive, solutions_positive, w_negative, solutions_negative)
            elif [d_aa, d_bb, cc] in list_ab:
                pp = list_ab.index([d_aa, d_bb, cc])
                q_ab = list_qab[pp]
                w_positive, solutions_positive, w_negative, solutions_negative = q_ab
                set_sbox(m_aa, m_bb, ss0, ww0, w_positive, solutions_positive, w_negative, solutions_negative)

        permute_bits(m_b[i], m_a[i + 1], permutation_bits_table_64, state_bits)

    if not input_x:
        btor.Assert(m_a[0] == 0)
    btor.Assert(m_a[total_rounds] == 0)

    temp_s0 = btor.Const(0)
    temp_w0_all = btor.Const(0, pro_bit)
    for r in range(total_rounds):
        for i in range(state_words):
            temp_s0 ^= s_0[r][i]
            temp_w0_all += w_0[r][(i + 1) * pro_bit - 1: i * pro_bit]
    btor.Assert(s0_all == temp_s0)
    btor.Assert(w0_all == temp_w0_all)

    write_quasidiff_meta_jsonl(file_name, differential_route, average_w, min_w, max_w, basis_number, input_x)
    f_out = open(file_name, "a")

    # quasidifferentials = []
    w_list = []
    positive_list = []
    negative_list = []

    quasidifferential_bases = []

    u0_all = btor.Concat(m_a[total_rounds], m_a[total_rounds - 1])
    for i in range(total_rounds - 2, -1, -1):
        u0_all = btor.Concat(u0_all, m_a[i])

    # Exclude the all-zero mask because it belongs to every subspace and adds no rank.
    # non_zero = btor.Const(0, 1)
    # for i in range(state_bits * (total_rounds + 1)):
    #     non_zero |= u0_all[i]
    btor.Assert(u0_all != 0)

    count_time = 0

    for current_w in range(min_w, max_w):

        while True:
            btor.Assume(w0_all == current_w)

            # bb_0 = add_basis_constraints(quasidifferential_bases, current_w, count_time)
            # for i in range(len(bb_0)):
            #     btor.Assume(u0_all[i] == bb_0[i])
            independent_constraint = constrain_candidate_outside_current_span(quasidifferential_bases, current_w, count_time)
            btor.Assume(independent_constraint)

            r = btor.Sat()
            if r == btor.SAT:
                w = int(w0_all.assignment, base=2)
                s = int(s0_all.assignment, base=2)

                if w not in w_list:
                    w_list.append(w)
                    positive_list.append(0)
                    negative_list.append(0)
                pp = w_list.index(w)
                if s == 1:
                    negative_list[pp] += 1
                elif s == 0:
                    positive_list[pp] += 1

                trail = []
                ww = 0

                for i in range(total_rounds + 1):
                    m_aa = int(m_a[i].assignment, base=2)
                    trail.append(m_aa)
                    if i != total_rounds:
                        m_bb = int(m_b[i].assignment, base=2)
                        trail.append(m_bb)
                        ww_i = 0
                        for ii in range(state_words):
                            ww_i += int(w_0[i][(ii + 1) * pro_bit - 1:ii * pro_bit].assignment, base=2)
                        ww += ww_i

                quasidifferential_bases.append(trail)

                # f_out = open(file_name, "a")
                append_quasidiff_record_jsonl(
                    f_out,
                    trail_id=len(quasidifferential_bases) - 1,
                    weight=w,
                    sign=s,
                    trail=trail
                )

                count_time += 1

                if len(quasidifferential_bases) == basis_number:
                    break

            else:
                count_time += 1
                break

        if len(quasidifferential_bases) == basis_number:
            break

    f_out.close()

    return quasidifferential_bases, w_list, positive_list, negative_list


# =====================================================================================
# Enumerate the characteristic trails contained in the 18-round differential D^(0).
#
# This is deliberately kept in the existing PyBoolector module.  It is independent of
# the quasidifferential-basis entry above: the latter assumes a fixed characteristic,
# whereas this entry enumerates the characteristics themselves.
# =====================================================================================

R18_D0_MASTER_KEY_DIFFERENCE = 0x00000000000000000000000000280000
R18_D0_INPUT_DIFFERENCE = 0x0060000000060000
R18_D0_OUTPUT_DIFFERENCE = 0x1400000041000000
R18_D0_CORE_FIRST_ROUND = 5
R18_D0_CORE_ROUNDS = 8


def _gift64_ddt():
    ddt = [[0 for _ in range(16)] for _ in range(16)]
    for input_difference in range(16):
        for x in range(16):
            output_difference = sbox[x] ^ sbox[x ^ input_difference]
            ddt[input_difference][output_difference] += 1
    return ddt


def _gift64_permute_int(value):
    """Match the model convention: output[i] = input[P[i]]."""
    result = 0
    for output_bit, input_bit in enumerate(permutation_bits_table_64):
        result |= ((value >> input_bit) & 1) << output_bit
    return result


def _gift64_injected_round_key_differences(master_key_difference, rounds=18):
    """Generate the 64-bit ARK differences from the paper's 128-bit key format."""
    words = [
        (master_key_difference >> (16 * (7 - index))) & 0xffff
        for index in range(8)
    ]

    def rotate_right_16(value, amount):
        return ((value >> amount) | (value << (16 - amount))) & 0xffff

    differences = []
    for _ in range(rounds):
        u_word = words[6]
        v_word = words[7]
        injected = 0
        for nibble in range(16):
            injected |= ((u_word >> nibble) & 1) << (4 * nibble + 1)
            injected |= ((v_word >> nibble) & 1) << (4 * nibble + 2)
        differences.append(injected)
        words = [
            rotate_right_16(words[6], 2),
            rotate_right_16(words[7], 12),
        ] + words[:6]
    return differences


def _unwrap_existing_r18_d0():
    from differentials import r_18_differential_0

    # The existing object is a one-element tuple containing the characteristic list.
    if (
        isinstance(r_18_differential_0, tuple)
        and len(r_18_differential_0) == 1
    ):
        trails = r_18_differential_0[0]
    else:
        trails = r_18_differential_0

    if not trails or any(len(trail) != 55 for trail in trails):
        raise ValueError("r_18_differential_0 must contain 55-entry trails")
    return trails


def _r18_d0_search_context():
    existing_trails = _unwrap_existing_r18_d0()
    template = existing_trails[0]
    round_key_differences = _gift64_injected_round_key_differences(
        R18_D0_MASTER_KEY_DIFFERENCE
    )

    for index, trail in enumerate(existing_trails):
        if trail[0] != R18_D0_INPUT_DIFFERENCE:
            raise ValueError(f"existing trail {index} has a different input endpoint")
        if trail[-1] != R18_D0_OUTPUT_DIFFERENCE:
            raise ValueError(f"existing trail {index} has a different output endpoint")

        observed_round_keys = [
            trail[3 * round_index + 2] ^ trail[3 * round_index + 3]
            for round_index in range(18)
        ]
        if observed_round_keys != round_key_differences:
            raise ValueError(
                f"existing trail {index} does not match master-key difference "
                f"0x{R18_D0_MASTER_KEY_DIFFERENCE:032x}"
            )

        # The outer inactive portions must be common before they can be reused.
        if trail[:3 * R18_D0_CORE_FIRST_ROUND] != template[:3 * R18_D0_CORE_FIRST_ROUND]:
            raise ValueError(f"existing trail {index} has a different prefix")
        core_end_index = 3 * (R18_D0_CORE_FIRST_ROUND + R18_D0_CORE_ROUNDS)
        if trail[core_end_index + 1:] != template[core_end_index + 1:]:
            raise ValueError(f"existing trail {index} has a different suffix")

    core_start_index = 3 * R18_D0_CORE_FIRST_ROUND
    core_end_index = 3 * (R18_D0_CORE_FIRST_ROUND + R18_D0_CORE_ROUNDS)
    core_start = template[core_start_index]
    core_end = template[core_end_index]
    core_round_keys = round_key_differences[
        R18_D0_CORE_FIRST_ROUND:
        R18_D0_CORE_FIRST_ROUND + R18_D0_CORE_ROUNDS
    ]

    return {
        "existing_trails": existing_trails,
        "template": template,
        "round_key_differences": round_key_differences,
        "core_start": core_start,
        "core_end": core_end,
        "core_round_keys": core_round_keys,
    }


def _characteristic_weight(trail):
    ddt = _gift64_ddt()
    weight = 0
    decimal_transitions = 0
    for round_index in range(18):
        input_difference = trail[3 * round_index]
        output_difference = trail[3 * round_index + 1]
        for nibble in range(16):
            a = (input_difference >> (4 * nibble)) & 0xf
            b = (output_difference >> (4 * nibble)) & 0xf
            count = ddt[a][b]
            if count == 0:
                raise ValueError(
                    f"invalid DDT transition in round {round_index}, nibble {nibble}: "
                    f"{a:x}->{b:x}"
                )
            if count == 16:
                transition_weight = 0
            elif count == 6:
                transition_weight = 1
                decimal_transitions += 1
            elif count == 4:
                transition_weight = 2
            elif count == 2:
                transition_weight = 3
            else:
                raise ValueError(f"unexpected GIFT-64 DDT count: {count}")
            weight += transition_weight
    return weight, decimal_transitions


def enumerate_r18_differential_0_characteristics(max_core_weight=64):
    """Enumerate D^(0) core trails using only PyBoolector.

    The integer weight encoding is the one used by the paper's SAT model:
    DDT counts 16, 6, 4 and 2 have integer weights 0, 1, 2 and 3;
    count-6 transitions are tracked separately because their exact weight is
    non-integral.  The 117 reported trails contain no count-6 transition.
    """
    context = _r18_d0_search_context()
    btor = pyboolector.Boolector()
    btor.Set_opt(pyboolector.BtorOption.BTOR_OPT_MODEL_GEN, 1)
    btor.Set_opt(pyboolector.BtorOption.BTOR_OPT_INCREMENTAL, 1)

    core_inputs = [
        btor.Var(btor.BitVecSort(64), f"r18_d0_core_input_{round_index}")
        for round_index in range(R18_D0_CORE_ROUNDS + 1)
    ]
    core_sbox_outputs = [
        btor.Var(btor.BitVecSort(64), f"r18_d0_core_sbox_output_{round_index}")
        for round_index in range(R18_D0_CORE_ROUNDS)
    ]
    core_permuted = [
        btor.Var(btor.BitVecSort(64), f"r18_d0_core_permuted_{round_index}")
        for round_index in range(R18_D0_CORE_ROUNDS)
    ]
    weights = [
        [
            btor.Var(btor.BitVecSort(2), f"r18_d0_weight_{round_index}_{nibble}")
            for nibble in range(16)
        ]
        for round_index in range(R18_D0_CORE_ROUNDS)
    ]

    btor.Assert(core_inputs[0] == btor.Const(context["core_start"], 64))
    btor.Assert(core_inputs[-1] == btor.Const(context["core_end"], 64))

    ddt = _gift64_ddt()
    allowed_transitions = []
    for a in range(16):
        for b in range(16):
            count = ddt[a][b]
            if count == 16:
                allowed_transitions.append((a, b, 0))
            elif count == 6:
                allowed_transitions.append((a, b, 1))
            elif count == 4:
                allowed_transitions.append((a, b, 2))
            elif count == 2:
                allowed_transitions.append((a, b, 3))

    for round_index in range(R18_D0_CORE_ROUNDS):
        for nibble in range(16):
            low = 4 * nibble
            input_nibble = core_inputs[round_index][low + 3:low]
            output_nibble = core_sbox_outputs[round_index][low + 3:low]
            transition_allowed = btor.Const(0, 1)
            for a, b, weight in allowed_transitions:
                transition_allowed |= (
                    (input_nibble == btor.Const(a, 4))
                    & (output_nibble == btor.Const(b, 4))
                    & (weights[round_index][nibble] == btor.Const(weight, 2))
                )
            btor.Assert(transition_allowed)

        for output_bit, input_bit in enumerate(permutation_bits_table_64):
            btor.Assert(
                core_permuted[round_index][output_bit]
                == core_sbox_outputs[round_index][input_bit]
            )
        btor.Assert(
            core_inputs[round_index + 1]
            == (
                core_permuted[round_index]
                ^ btor.Const(context["core_round_keys"][round_index], 64)
            )
        )

    total_weight = btor.Const(0, 10)
    for round_weights in weights:
        for transition_weight in round_weights:
            total_weight += btor.Uext(transition_weight, 8)
    btor.Assert(btor.Ulte(total_weight, btor.Const(max_core_weight, 10)))

    # Endpoints are fixed.  The seven internal S-box inputs uniquely determine the
    # S-box outputs because P is bijective and all round-key differences are fixed.
    signature = core_inputs[R18_D0_CORE_ROUNDS - 1]
    for round_index in range(R18_D0_CORE_ROUNDS - 2, 0, -1):
        signature = btor.Concat(signature, core_inputs[round_index])

    core_trails = []
    while btor.Sat() == btor.SAT:
        core_trail = []
        for round_index in range(R18_D0_CORE_ROUNDS):
            core_trail.extend([
                int(core_inputs[round_index].assignment, 2),
                int(core_sbox_outputs[round_index].assignment, 2),
                int(core_permuted[round_index].assignment, 2),
            ])
        core_trail.append(int(core_inputs[-1].assignment, 2))
        core_trails.append(core_trail)

        signature_value = int(signature.assignment, 2)
        btor.Assert(signature != btor.Const(signature_value, 64 * 7))

    prefix = context["template"][:3 * R18_D0_CORE_FIRST_ROUND]
    core_end_index = 3 * (R18_D0_CORE_FIRST_ROUND + R18_D0_CORE_ROUNDS)
    suffix = context["template"][core_end_index + 1:]
    full_trails = [prefix + core_trail + suffix for core_trail in core_trails]
    full_trails.sort(key=lambda trail: (_characteristic_weight(trail)[0], tuple(trail)))
    return full_trails


def validate_r18_differential_0_characteristics(trails, expected_count=None):
    from collections import Counter
    from differentials import r_18_characteristic_0

    context = _r18_d0_search_context()
    if expected_count is not None and len(trails) != expected_count:
        raise ValueError(f"expected {expected_count} trails, found {len(trails)}")
    if len({tuple(trail) for trail in trails}) != len(trails):
        raise ValueError("duplicate trails were found")

    for index, trail in enumerate(trails):
        if len(trail) != 55:
            raise ValueError(f"trail {index} has length {len(trail)}, expected 55")
        if trail[0] != R18_D0_INPUT_DIFFERENCE or trail[-1] != R18_D0_OUTPUT_DIFFERENCE:
            raise ValueError(f"trail {index} has a wrong endpoint")
        observed_round_keys = [
            trail[3 * round_index + 2] ^ trail[3 * round_index + 3]
            for round_index in range(18)
        ]
        if observed_round_keys != context["round_key_differences"]:
            raise ValueError(f"trail {index} has a wrong round-key difference")
        for round_index in range(18):
            if _gift64_permute_int(trail[3 * round_index + 1]) != trail[3 * round_index + 2]:
                raise ValueError(f"trail {index}, round {round_index}: P-layer mismatch")

    trail_set = {tuple(trail) for trail in trails}
    missing_existing = [
        index for index, trail in enumerate(context["existing_trails"])
        if tuple(trail) not in trail_set
    ]
    if missing_existing:
        raise ValueError(f"existing r_18_differential_0 trails missing: {missing_existing}")
    if tuple(r_18_characteristic_0[0]) not in trail_set:
        raise ValueError("r_18_characteristic_0 is missing")

    distribution = Counter()
    decimal_transition_counts = Counter()
    for trail in trails:
        weight, decimal_transitions = _characteristic_weight(trail)
        distribution[weight] += 1
        decimal_transition_counts[decimal_transitions] += 1
    return dict(sorted(distribution.items())), dict(sorted(decimal_transition_counts.items()))


def write_r18_differential_0_characteristics(trails, output_path):
    from pathlib import Path

    output_path = Path(output_path)
    lines = [
        '"""117 characteristic trails of the 18-round GIFT-64 differential D^(0).',
        "",
        "Generated by the PyBoolector entry in quasidifferential_search.py.",
        "The flat list can be imported directly as cipher_config.diffs.",
        '"""',
        "",
        f"MASTER_KEY_DIFFERENCE = 0x{R18_D0_MASTER_KEY_DIFFERENCE:032x}",
        f"INPUT_DIFFERENCE = 0x{R18_D0_INPUT_DIFFERENCE:016x}",
        f"OUTPUT_DIFFERENCE = 0x{R18_D0_OUTPUT_DIFFERENCE:016x}",
        "",
        "r_18_differential_0_117 = [",
    ]

    for solution_index, trail in enumerate(trails, start=1):
        total_weight, decimal_transitions = _characteristic_weight(trail)
        lines.append(
            f"    # w = {total_weight}, decimal transitions = {decimal_transitions}, "
            f"Solution: #{solution_index}"
        )
        lines.append("    [")
        lines.append(f"        0x{trail[0]:016x},")
        for round_index in range(18):
            lines.append("        # after sb")
            lines.append(f"        0x{trail[3 * round_index + 1]:016x},")
            lines.append("        # after p")
            lines.append(f"        0x{trail[3 * round_index + 2]:016x},")
            lines.append("        # after k")
            lines.append(f"        0x{trail[3 * round_index + 3]:016x},")
        lines.append("    ],")
    lines.extend([
        "]",
        "",
        "# Compatibility wrapper matching the existing r_18_differential_0 object.",
        "r_18_differential_0 = (r_18_differential_0_117,)",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _run_r18_d0_enumeration_cli():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Enumerate the GIFT-64 D^(0) characteristic trails with PyBoolector"
    )
    parser.add_argument("--enumerate-r18-d0", action="store_true")
    parser.add_argument("--max-core-weight", type=int, default=64)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.enumerate_r18_d0:
        parser.error("specify --enumerate-r18-d0")

    trails = enumerate_r18_differential_0_characteristics(args.max_core_weight)
    distribution, decimal_counts = validate_r18_differential_0_characteristics(
        trails,
        expected_count=args.expected_count,
    )
    print(f"trails = {len(trails)}")
    print(f"full 18-round weight distribution = {distribution}")
    print(f"count-6 transition distribution = {decimal_counts}")
    if args.output is not None:
        write_r18_differential_0_characteristics(trails, args.output)
        print(f"written = {args.output}")


if __name__ == "__main__":
    _run_r18_d0_enumeration_cli()


