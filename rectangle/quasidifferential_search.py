"""Search independent RECTANGLE quasidifferential masks with PyBoolector."""

import pyboolector
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from cipher_config import cipher_name, sbox, n, m, state_bits, state_words, sbox_bits, pro_bit, total_rounds, begin_round, round_constants, permutation_bits_table_64
from cipher_config import trail_model_number as trail_number
from cipher_config import characteristic_model_number as route_number
from cipher_config import key_model, round_key_bits, key_bits
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
        "key_model": key_model,
        "round_key_bits": round_key_bits,
        "key_bits": key_bits,
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
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

@contextmanager
def _atomic_quasidiff_output(file_name, differential_route, average_w, min_w,
                            max_w, basis_number, input_x):
    """Publish only a complete search; retain failed attempts as diagnostics."""
    target = Path(file_name)
    with tempfile.NamedTemporaryFile(
        prefix="." + target.name + ".", suffix=".partial",
        dir=str(target.parent), delete=False,
    ) as staging:
        partial_name = staging.name
    try:
        write_quasidiff_meta_jsonl(
            partial_name, differential_route, average_w, min_w, max_w,
            basis_number, input_x,
        )
        with open(partial_name, "a", encoding="utf-8") as output:
            yield output
        os.replace(partial_name, str(target))
    except BaseException:
        print(f"Incomplete search record retained at: {partial_name}", file=sys.stderr)
        raise


def _require_decided_sat_result(btor, result):
    if result not in (btor.SAT, btor.UNSAT):
        raise RuntimeError(
            f"Boolector returned an undecided result ({result}); "
            "the search is incomplete, not UNSAT"
        )


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
    if basis_number <= 0:
        raise ValueError("basis_number must be positive")
    if min_w >= max_w:
        raise ValueError("The search weight range must not be empty")
    expected_length = route_number * total_rounds + 1
    if len(differential_route) != expected_length:
        raise ValueError(
            f"Differential route has {len(differential_route)} entries; "
            f"expected {expected_length}"
        )
    # An impossible differential has an empty local table. Reject it before
    # building the model instead of accidentally leaving that S-box unconstrained.
    utils.get_average_w(differential_route, n, m)
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
        # RECTANGLE constants are part of key expansion, not this S-box layer.
        round_c = round_constants[i]
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

    w_list = []
    positive_list = []
    negative_list = []

    quasidifferential_bases = []

    u0_all = btor.Concat(m_a[total_rounds], m_a[total_rounds - 1])
    for i in range(total_rounds - 2, -1, -1):
        u0_all = btor.Concat(u0_all, m_a[i])

    # Zero belongs to every span and cannot increase the basis rank.
    btor.Assert(u0_all != 0)

    count_time = 0

    with _atomic_quasidiff_output(
        file_name, differential_route, average_w, min_w, max_w, basis_number, input_x
    ) as f_out:
        for current_w in range(min_w, max_w):

            while True:
                btor.Assume(w0_all == current_w)

                independent_constraint = constrain_candidate_outside_current_span(quasidifferential_bases, current_w, count_time)
                btor.Assume(independent_constraint)

                r = btor.Sat()
                _require_decided_sat_result(btor, r)
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

                    for i in range(total_rounds + 1):
                        m_aa = int(m_a[i].assignment, base=2)
                        trail.append(m_aa)
                        if i != total_rounds:
                            m_bb = int(m_b[i].assignment, base=2)
                            trail.append(m_bb)

                    quasidifferential_bases.append(trail)

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

        if len(quasidifferential_bases) != basis_number:
            raise RuntimeError(
                f"Incomplete basis search: found {len(quasidifferential_bases)} "
                f"of {basis_number} requested independent trails in weights "
                f"[{min_w}, {max_w}); partial record: {f_out.name}"
            )

    return quasidifferential_bases, w_list, positive_list, negative_list



