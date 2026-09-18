import numpy
from cipher_config import sbox, state_bits, key_bits, sbox_bits, total_rounds, begin_round, round_constants, master_round_key_table
from cipher_config import trail_model_number as trail_number
from cipher_config import characteristic_model_number as route_number

def get_ddt(sbox, n, m):
    table = numpy.zeros((2 ** m, 2 ** n))
    for a in range(2 ** n):
        for b in range(2 ** m):
            for x in range(2 ** n):
                x1 = x ^ a
                y = sbox[x]
                y1 = sbox[x1]
                if y ^ y1 == b:
                    table[b, a] += 1
    for a in range(2 ** n):
        for b in range(2 ** m):
            if table[b, a] != 0:
                table[b, a] = table[b, a] / (2 ** n)

    return table

def get_ddt_solutions(sbox, n, m):
    table = numpy.zeros((2 ** m, 2 ** n))
    for a in range(2 ** n):
        for b in range(2 ** m):
            for x in range(2 ** n):
                x1 = x ^ a
                y = sbox[x]
                y1 = sbox[x1]
                if y ^ y1 == b:
                    table[b, a] += 1

    w_list = []
    solutions = []

    for a in range(2 ** n):
        for b in range(2 ** m):
            p = table[b, a]
            if p != 0:
                p /= (2 ** n)
                w = int(-numpy.log2(p))
                if w not in w_list:
                    w_list.append(w)
                    solutions.append([[a, b]])
                else:
                    solutions[w_list.index(w)].append([a, b])

    return w_list, solutions

def get_quasidifferential_matrix(sbox, n, m):
    table = numpy.zeros((2 ** (2 * m), 2 ** (2 * n)))
    for a in range(2 ** n):
        for b in range(2 ** m):
            for u in range(2 ** n):
                for v in range(2 ** m):
                    input_idx = (b << m) + v
                    output_idx = (a << n) + u
                    for x in range(2 ** n):
                        x1 = x ^ a
                        y = sbox[x]
                        y1 = sbox[x1]
                        if y ^ y1 == b:
                            if vector_inner_product(u, x, n) ^ vector_inner_product(v, y, m) == 0:
                                table[input_idx, output_idx] += 1
                            else:
                                table[input_idx, output_idx] -= 1
    for a in range(2 ** n):
        for b in range(2 ** m):
            for u in range(2 ** n):
                for v in range(2 ** m):
                    input_idx = (b << m) + v
                    output_idx = (a << n) + u
                    if table[input_idx, output_idx] != 0:
                        table[input_idx, output_idx] /= (2 ** n)
    return table


def get_quasidifferential_matrix_by_a_and_b_and_c(sbox, n, m, a, b, c):
    table = numpy.zeros((2 ** m, 2 ** n))
    for u in range(2 ** n):
        for v in range(2 ** m):
            input_idx = v
            output_idx = u
            for x in range(2 ** n):
                x1 = x ^ a
                y = sbox[x ^ c]
                y1 = sbox[x1 ^ c]
                if y ^ y1 == b:
                    if vector_inner_product(u, x, n) ^ vector_inner_product(v, y, m) == 0:
                        table[input_idx, output_idx] += 1
                    else:
                        table[input_idx, output_idx] -= 1

    w_positive = []
    solutions_positive = []
    w_negative = []
    solutions_negative = []

    for u in range(2 ** n):
        for v in range(2 ** m):
            cor = table[v, u]
            if cor != 0:
                cor /= (2 ** n)
                w = int(-numpy.log2(abs(cor)))
                if cor < 0:
                    if w not in w_negative:
                        w_negative.append(w)
                        solutions_negative.append([[u, v]])
                    else:
                        solutions_negative[w_negative.index(w)].append([u, v])
                else:
                    if w not in w_positive:
                        w_positive.append(w)
                        solutions_positive.append([[u, v]])
                    else:
                        solutions_positive[w_positive.index(w)].append([u, v])


    return [w_positive, solutions_positive, w_negative, solutions_negative]

def vector_inner_product(u, x, n):
    left = 0
    for i in range(n):
        left += ((u >> i) & 0x1) * ((x >> i) & 0x1)
    left = left % 2
    return left

def get_correlation_by_abuv_and_c(a, b, u, v, c):
    cor = 0
    for x in range(2 ** sbox_bits):
        x1 = x ^ a
        y = sbox[x ^ c]
        y1 = sbox[x1 ^ c]
        if y ^ y1 == b:
            if vector_inner_product(u, x, sbox_bits) ^ vector_inner_product(v, y, sbox_bits) == 0:
                cor += 1
            else:
                cor -= 1

    return cor / (2 ** sbox_bits)

def get_constant(l_rc):
    l_ii = [3, 7, 11, 15, 19, 23, 63]
    result = 0
    for pos, bit_val in zip(l_ii, l_rc):
        if bit_val:
            result |= 1 << pos

    return result

def get_correlation_of_quasidifferential(characteristic, trail):

    cor = 1

    for i in range(total_rounds):
        rc = round_constants[i]
        l_rc = [(rc >> 0) & 0b1, (rc >> 1) & 0b1, (rc >> 2) & 0b1, (rc >> 3) & 0b1, (rc >> 4) & 0b1,
                (rc >> 5) & 0b1, 1]
        round_c = 0
        if i != 0:
            round_c = get_constant(l_rc)
        for j in range(0, state_bits, sbox_bits):
            d_aa = (characteristic[route_number * i] >> j) & 0xf
            d_bb = (characteristic[route_number * i + 1] >> j) & 0xf
            m_aa = (trail[trail_number * i] >> j) & 0xf
            m_bb = (trail[trail_number * i + 1] >> j) & 0xf
            cc = (round_c >> j) & 0xf
            cor *= get_correlation_by_abuv_and_c(d_aa, d_bb, m_aa, m_bb, cc)
            if cor == 0:
                return cor

    return cor

def get_average_w(differential_route, n, m):
    average_w = 0
    ddt = get_ddt(sbox, n, m)
    for i in range(total_rounds):
        for j in range(0, state_bits, sbox_bits):
            d_aa = (differential_route[route_number * i] >> j) & 0xf
            d_bb = (differential_route[route_number * i + 1] >> j) & 0xf
            pab = ddt[d_bb, d_aa]
            if pab == 0:
                print(f"Invalid differential transition: a={d_aa}, b={d_bb}")
                exit()
            wab = int(-numpy.log2(pab))
            average_w += wab

    return average_w


def get_master_key_mask(k_trail):

    k_expression = []

    for r in range(len(k_trail)):
        current_round_k_mask = k_trail[r]
        for i in range(state_bits // sbox_bits):
            if ((current_round_k_mask >> (sbox_bits * i)) & 0b1) == 0b1:
                if (master_round_key_table[r + begin_round][i]) not in k_expression:
                    k_expression.append(master_round_key_table[r + begin_round][i])
                else:
                    k_expression.remove(master_round_key_table[r + begin_round][i])
            if ((current_round_k_mask >> (sbox_bits * i + 1)) & 0b1) == 0b1:
                if (master_round_key_table[r + begin_round][i + 16]) not in k_expression:
                    k_expression.append(master_round_key_table[r + begin_round][i + 16])
                else:
                    k_expression.remove(master_round_key_table[r + begin_round][i + 16])

    k_expression.sort()
    k_mask = [0] * key_bits
    for kk in k_expression:
        k_mask[kk] = 1

    return k_mask

def get_a_key_trail(q_trail):
    k_trail = []
    for i in range(total_rounds):
        # k_trail.append(q_trail[2 * i + 1])
        k_trail.append(q_trail[trail_number * i + 2])

    return k_trail

def get_cor_and_key_by_one_quasidifferential(characteristic, one_q_trail):
    one_q_cor = get_correlation_of_quasidifferential(characteristic, one_q_trail)
    one_k_trail = get_a_key_trail(one_q_trail)
    master_k_trail = get_master_key_mask(one_k_trail)
    input_x = one_q_trail[0]
    for i in range(state_bits):
        master_k_trail.append((input_x >> i) & 0b1)

    return one_q_cor, master_k_trail

#===========================================================================================########


