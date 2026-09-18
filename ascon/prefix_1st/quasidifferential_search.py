# !/user/bin/env python3

# version 1, basic model
# version 2, use basis

import pyboolector
import json
from cipher_config import cipher_name, sbox, n, m, state_bits, state_words, sbox_bits, pro_bit, total_rounds, begin_round, round_constants, linear_mat_transpose, IV
from cipher_config import trail_model_number as trail_number
from cipher_config import characteristic_model_number as route_number
import utils

def int_to_fixed_hex(x, bits):
    width = (bits + 3) // 4
    return "0x{:0{}x}".format(x, width)

def list_to_fixed_hex(values, bits):
    return [int_to_fixed_hex(v, bits) for v in values]

def write_quasidiff_meta_jsonl(file_name, differential_route, average_w, min_w, max_w, basis_number, mode):
    meta = {
        "type": "meta",
        "version": 1,
        "cipher": cipher_name,
        "mode": mode,
        "total_rounds": total_rounds,
        "begin_round": begin_round,
        "state_bits": state_bits,
        "state_words": state_words,
        "sbox_bits": sbox_bits,
        "pro_bit": pro_bit,
        "average_weight": average_w,
        "min_weight": min_w,
        "max_weight": max_w,
        "basis_number": basis_number,
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

def quasi_differential_search(differential_route, average_w, min_w, max_w, basis_number, mode, file_name):
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
            i_bits = [j for j, b in enumerate(list(table[i])) if b == 1]
            temp = btor.Const(0)
            for ii in i_bits:
                temp ^= y[ii]
            btor.Assert(x[i] == temp)

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

    # def add_u_basis_constraints(mask_a, average_w_trails):
    #
    #     # u_trails = []
    #
    #     import basis_generate as bg
    #
    #     print()
    #     u_vectors = bg.get_vectors_by_u_trails(average_w_trails, total_rounds, state_bits, trail_number)
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
    #     coefficients_b_0 = [btor.Var(btor.BitVecSort(1), "c_b_0_{}".format(j)) for j in range(dim_b)]
    #
    #     bb_0 = [btor.Const(0, 1) for i in range(dim_all)]
    #     for i in range(dim_all):
    #         for j in range(dim_b):
    #             bb_0[i] ^= btor.Cond(coefficients_b_0[j] == btor.Const(1, 1), btor.Const(b_basis[j][i], 1), btor.Const(0, 1))
    #
    #     # u0_all = btor.Concat(mask_a[total_rounds - 1], mask_a[total_rounds - 2])
    #     u0_all = btor.Concat(mask_a[total_rounds], mask_a[total_rounds - 1])
    #     # for i in range(total_rounds - 3, 0, -1):
    #     for i in range(total_rounds - 2, -1, -1):
    #         u0_all = btor.Concat(u0_all, mask_a[i])
    #     for i in range(len(bb_0)):
    #         btor.Assert(u0_all[i] == bb_0[i])

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
        for j in range(0, state_bits, sbox_bits):
            p1 = 0 * state_words + (j // sbox_bits)
            p2 = 1 * state_words + (j // sbox_bits)
            p3 = 2 * state_words + (j // sbox_bits)
            p4 = 3 * state_words + (j // sbox_bits)
            p5 = 4 * state_words + (j // sbox_bits)
            d_aa = ((((differential_route[route_number * i] >> p1) & 0b1) << 4) ^
                   (((differential_route[route_number * i] >> p2) & 0b1) << 3) ^
                   (((differential_route[route_number * i] >> p3) & 0b1) << 2) ^
                   (((differential_route[route_number * i] >> p4) & 0b1) << 1) ^
                   (((differential_route[route_number * i] >> p5) & 0b1) << 0))
            d_bb = ((((differential_route[route_number * i + 1] >> p1) & 0b1) << 4) ^
                   (((differential_route[route_number * i + 1] >> p2) & 0b1) << 3) ^
                   (((differential_route[route_number * i + 1] >> p3) & 0b1) << 2) ^
                   (((differential_route[route_number * i + 1] >> p4) & 0b1) << 1) ^
                   (((differential_route[route_number * i + 1] >> p5) & 0b1) << 0))
            m_aa = btor.Concat(btor.Concat(btor.Concat(btor.Concat(m_a[i][p1], m_a[i][p2]), m_a[i][p3]), m_a[i][p4]), m_a[i][p5])
            m_bb = btor.Concat(btor.Concat(btor.Concat(btor.Concat(m_b[i][p1], m_b[i][p2]), m_b[i][p3]), m_b[i][p4]), m_b[i][p5])
            ss0 = s_0[i][j // sbox_bits]
            ww0 = w_0[i][(j // sbox_bits + 1) * pro_bit - 1:(j // sbox_bits) * pro_bit]

            rcs = round_constants[i + begin_round]
            cc = 0
            column = j // sbox_bits
            # Stored route/model indices are translated to the verifier by
            # bit i -> bit 63-i.  Its RC << 56 therefore occupies model
            # columns 0..7, in MSB-first byte order.
            if column in range(0, 8):
                cc = ((rcs >> (7 - column)) & 0b1) << 2

            x1 = (IV >> (63 - j // sbox_bits)) & 0b1
            r0 = 0
            if i > 0:
                r0 = 1

            if [d_aa, d_bb, cc, r0, x1] not in list_ab:
                list_ab.append([d_aa, d_bb, cc, r0, x1])
                q_ab = None
                if mode == "permutation" or i > 0:
                    q_ab = utils.get_quasidifferential_matrix_by_a_and_b_and_c_permutation(sbox, n, m, d_aa, d_bb, cc)
                elif mode == "equal" and r0 == 0:
                    q_ab = utils.get_quasidifferential_matrix_by_a_and_b_and_c_aead_equal(sbox, n, m, d_aa, d_bb, cc, x1)
                elif mode == "unrestricted" and r0 == 0:
                    q_ab = utils.get_quasidifferential_matrix_by_a_and_b_and_c_aead_unrestricted(sbox, n, m, d_aa, d_bb, cc, x1)
                else:
                    print("Error ! Invalid mode !")
                    exit()
                list_qab.append(q_ab)
                w_positive, solutions_positive, w_negative, solutions_negative = q_ab
                set_sbox(m_aa, m_bb, ss0, ww0, w_positive, solutions_positive, w_negative, solutions_negative)
            elif [d_aa, d_bb, cc, r0, x1] in list_ab:
                pp = list_ab.index([d_aa, d_bb, cc, r0, x1])
                q_ab = list_qab[pp]
                w_positive, solutions_positive, w_negative, solutions_negative = q_ab
                set_sbox(m_aa, m_bb, ss0, ww0, w_positive, solutions_positive, w_negative, solutions_negative)

        permute_bits(m_b[i], m_a[i + 1], linear_mat_transpose, state_bits)

    # btor.Assert(m_a[0] == 0)
    btor.Assert(m_a[total_rounds] == 0)
    if mode == "equal":
        for i in range(0, 64):
            btor.Assert(m_a[0][i] == 0)
        for i in range(256, 320):
            btor.Assert(m_a[0][i] == 0)
    if mode == "unrestricted":
        for i in range(0, 64):
            btor.Assert(m_a[0][i] == 0)

    temp_s0 = btor.Const(0)
    temp_w0_all = btor.Const(0, pro_bit)
    for r in range(total_rounds):
        for i in range(state_words):
            temp_s0 ^= s_0[r][i]
            temp_w0_all += w_0[r][(i + 1) * pro_bit - 1: i * pro_bit]
    btor.Assert(s0_all == temp_s0)
    btor.Assert(w0_all == temp_w0_all)

    write_quasidiff_meta_jsonl(file_name, differential_route, average_w, min_w, max_w, basis_number, mode)
    f_out = open(file_name, "a")

    # quasidifferentials = []
    w_list = []
    positive_list = []
    negative_list = []

    quasidifferential_bases = []

    u0_all = btor.Concat(m_a[total_rounds], m_a[total_rounds - 1])
    for i in range(total_rounds - 2, -1, -1):
        u0_all = btor.Concat(u0_all, m_a[i])

    # Exclude the zero mask because it contributes no rank.
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


