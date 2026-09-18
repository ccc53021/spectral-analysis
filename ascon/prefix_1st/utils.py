import numpy
from cipher_config import sbox, state_bits, state_words, sbox_bits, total_rounds, begin_round, round_constants, IV
from cipher_config import trail_model_number as trail_number
from cipher_config import characteristic_model_number as route_number

def trail_dict_to_flat_320(trail_dict):
    """Convert a two-round dictionary route to flat 320-bit integers."""
    h0, h1, h2, h3, h4 = (trail_dict["r0_in"][i] for i in range(5))
    r0_in  = h0 | (h1 << (1 * state_words)) | (h2 << (2 * state_words)) | (h3 << (3 * state_words)) | (h4 << (4 * state_words))
    h0, h1, h2, h3, h4 = (trail_dict["r0_out"][i] for i in range(5))
    r0_out  = h0 | (h1 << (1 * state_words)) | (h2 << (2 * state_words)) | (h3 << (3 * state_words)) | (h4 << (4 * state_words))
    h0, h1, h2, h3, h4 = (trail_dict["r0_lin"][i] for i in range(5))
    r0_lin  = h0 | (h1 << (1 * state_words)) | (h2 << (2 * state_words)) | (h3 << (3 * state_words)) | (h4 << (4 * state_words))
    h0, h1, h2, h3, h4 = (trail_dict["r1_sbox"][i] for i in range(5))
    r1_sbox  = h0 | (h1 << (1 * state_words)) | (h2 << (2 * state_words)) | (h3 << (3 * state_words)) | (h4 << (4 * state_words))
    h0, h1, h2, h3, h4 = (trail_dict["r1_lin"][i] for i in range(5))
    r1_lin  = h0 | (h1 << (1 * state_words)) | (h2 << (2 * state_words)) | (h3 << (3 * state_words)) | (h4 << (4 * state_words))

    return [r0_in, r0_out, r0_lin, r1_sbox, r1_lin]  # Two rounds have five entries.

def get_ddt_permutation(sbox, n, m):
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

def get_ddt_aead_equal(sbox, n, m, x1):
    table = numpy.zeros((2 ** m, 2 ** n))
    x_list = []
    for xxx in range(2 ** 3):
        x4 = xxx & 0b1
        x = (x1 << 4) | (xxx << 1) | x4
        x_list.append(x)

    for a in range(2 ** n):
        for b in range(2 ** m):
            for x in x_list:
                x1 = x ^ a
                y = sbox[x]
                y1 = sbox[x1]
                if y ^ y1 == b:
                    table[b, a] += 1
    for a in range(2 ** n):
        for b in range(2 ** m):
            if table[b, a] != 0:
                # table[b, a] = table[b, a] / (2 ** n)
                table[b, a] = table[b, a] / len(x_list)

    return table

def get_ddt_aead_norestricted(sbox, n, m, x1):
    table = numpy.zeros((2 ** m, 2 ** n))
    x_list = []
    for xxxx in range(2 ** 4):
        x = (x1 << 4) | xxxx
        x_list.append(x)

    for a in range(2 ** n):
        for b in range(2 ** m):
            for x in x_list:
                x1 = x ^ a
                y = sbox[x]
                y1 = sbox[x1]
                if y ^ y1 == b:
                    table[b, a] += 1
    for a in range(2 ** n):
        for b in range(2 ** m):
            if table[b, a] != 0:
                # table[b, a] = table[b, a] / (2 ** n)
                table[b, a] = table[b, a] / len(x_list)

    return table

def get_ddt_solutions_permutation(sbox, n, m):
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

def get_ddt_solutions_aead_equal(sbox, n, m, x1):
    table = numpy.zeros((2 ** m, 2 ** n))

    x_list = []
    for xxx in range(2 ** 3):
        x4 = xxx & 0b1
        x = (x1 << 4) | (xxx << 1) | x4
        x_list.append(x)

    for a in range(2 ** n):
        for b in range(2 ** m):
            for x in x_list:
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
                p /= len(x_list)
                w = int(-numpy.log2(p))
                if w not in w_list:
                    w_list.append(w)
                    solutions.append([[a, b]])
                else:
                    solutions[w_list.index(w)].append([a, b])

    return w_list, solutions

def get_ddt_solutions_aead_unrestricted(sbox, n, m, x1):
    table = numpy.zeros((2 ** m, 2 ** n))

    x_list = []
    for xxxx in range(2 ** 4):
        x = (x1 << 4) | xxxx
        x_list.append(x)

    for a in range(2 ** n):
        for b in range(2 ** m):
            for x in x_list:
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
                p /= len(x_list)
                w = int(-numpy.log2(p))
                if w not in w_list:
                    w_list.append(w)
                    solutions.append([[a, b]])
                else:
                    solutions[w_list.index(w)].append([a, b])

    return w_list, solutions

# def get_quasidifferential_matrix(sbox, n, m):
#     table = numpy.zeros((2 ** (2 * m), 2 ** (2 * n)))
#     for a in range(2 ** n):
#         for b in range(2 ** m):
#             for u in range(2 ** n):
#                 for v in range(2 ** m):
#                     input_idx = (b << m) + v
#                     output_idx = (a << n) + u
#                     for x in range(2 ** n):
#                         x1 = x ^ a
#                         y = sbox[x]
#                         y1 = sbox[x1]
#                         if y ^ y1 == b:
#                             if vector_inner_product(u, x, n) ^ vector_inner_product(v, y, m) == 0:
#                                 table[input_idx, output_idx] += 1
#                             else:
#                                 table[input_idx, output_idx] -= 1
#     for a in range(2 ** n):
#         for b in range(2 ** m):
#             for u in range(2 ** n):
#                 for v in range(2 ** m):
#                     input_idx = (b << m) + v
#                     output_idx = (a << n) + u
#                     if table[input_idx, output_idx] != 0:
#                         table[input_idx, output_idx] /= (2 ** n)
#     return table


def get_quasidifferential_matrix_by_a_and_b_and_c_permutation(sbox, n, m, a, b, c):
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

def get_quasidifferential_matrix_by_a_and_b_and_c_aead_equal(sbox, n, m, a, b, c, x1):
    table = numpy.zeros((2 ** m, 2 ** n))

    x_list = []
    for xxx in range(2 ** 3):
        x4 = xxx & 0b1
        x = (x1 << 4) | (xxx << 1) | x4
        x_list.append(x)

    for u in range(2 ** n):
        for v in range(2 ** m):
            input_idx = v
            output_idx = u
            for x in x_list:
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
                cor /= len(x_list)
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


def get_quasidifferential_matrix_by_a_and_b_and_c_aead_unrestricted(sbox, n, m, a, b, c, x1):
    table = numpy.zeros((2 ** m, 2 ** n))

    x_list = []
    for xxxx in range(2 ** 4):
        x = (x1 << 4) | xxxx
        x_list.append(x)

    for u in range(2 ** n):
        for v in range(2 ** m):
            input_idx = v
            output_idx = u
            for x in x_list:
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
                cor /= len(x_list)
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

def get_correlation_by_abuv_and_c_permutation(a, b, u, v, c):
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

def get_correlation_by_abuv_and_c_aead_equal(a, b, u, v, c, x1):
    cor = 0

    x_list = []
    for xxx in range(2 ** 3):
        x4 = xxx & 0b1
        x = (x1 << 4) | (xxx << 1) | x4
        x_list.append(x)

    for x in x_list:
        x1 = x ^ a
        y = sbox[x ^ c]
        y1 = sbox[x1 ^ c]
        if y ^ y1 == b:
            if vector_inner_product(u, x, sbox_bits) ^ vector_inner_product(v, y, sbox_bits) == 0:
                cor += 1
            else:
                cor -= 1

    return cor / len(x_list)

def get_correlation_by_abuv_and_c_aead_unrestricted(a, b, u, v, c, x1):
    cor = 0

    x_list = []
    for xxxx in range(2 ** 4):
        x = (x1 << 4) | xxxx
        x_list.append(x)

    for x in x_list:
        x1 = x ^ a
        y = sbox[x ^ c]
        y1 = sbox[x1 ^ c]
        if y ^ y1 == b:
            if vector_inner_product(u, x, sbox_bits) ^ vector_inner_product(v, y, sbox_bits) == 0:
                cor += 1
            else:
                cor -= 1

    return cor / len(x_list)

def get_trail_input_list(trail_input):
    ix_list = []
    for i in range(state_bits):
        ix_list.append((trail_input >> i) & 0b1)

    return ix_list

def get_cor_and_x_by_one_quasidifferential(characteristic, trail, mode):

    cor = 1

    for i in range(total_rounds):
        for j in range(0, state_bits, sbox_bits):
            p1 = 0 * state_words + (j // sbox_bits)
            p2 = 1 * state_words + (j // sbox_bits)
            p3 = 2 * state_words + (j // sbox_bits)
            p4 = 3 * state_words + (j // sbox_bits)
            p5 = 4 * state_words + (j // sbox_bits)
            d_aa = ((((characteristic[route_number * i] >> p1) & 0b1) << 4) |
                   (((characteristic[route_number * i] >> p2) & 0b1) << 3) |
                   (((characteristic[route_number * i] >> p3) & 0b1) << 2) |
                   (((characteristic[route_number * i] >> p4) & 0b1) << 1) |
                   (((characteristic[route_number * i] >> p5) & 0b1) << 0))
            d_bb = ((((characteristic[route_number * i + 1] >> p1) & 0b1) << 4) |
                   (((characteristic[route_number * i + 1] >> p2) & 0b1) << 3) |
                   (((characteristic[route_number * i + 1] >> p3) & 0b1) << 2) |
                   (((characteristic[route_number * i + 1] >> p4) & 0b1) << 1) |
                   (((characteristic[route_number * i + 1] >> p5) & 0b1) << 0))

            m_aa = ((((trail[trail_number * i] >> p1) & 0b1) << 4) ^
                   (((trail[trail_number * i] >> p2) & 0b1) << 3) ^
                   (((trail[trail_number * i] >> p3) & 0b1) << 2) ^
                   (((trail[trail_number * i] >> p4) & 0b1) << 1) ^
                   (((trail[trail_number * i] >> p5) & 0b1) << 0))
            m_bb = ((((trail[trail_number * i + 1] >> p1) & 0b1) << 4) ^
                   (((trail[trail_number * i + 1] >> p2) & 0b1) << 3) ^
                   (((trail[trail_number * i + 1] >> p3) & 0b1) << 2) ^
                   (((trail[trail_number * i + 1] >> p4) & 0b1) << 1) ^
                   (((trail[trail_number * i + 1] >> p5) & 0b1) << 0))

            rcs = round_constants[begin_round + i]
            cc = 0
            column = j // sbox_bits
            # Model bit i is verifier bit 63-i, so RC << 56 maps to model
            # columns 0..7 with the byte read MSB-first.
            if column in range(0, 8):
                cc = ((rcs >> (7 - column)) & 0b1) << 2

            x1 = (IV >> (63 - j // sbox_bits)) & 0b1

            if mode == "permutation" or i > 0:
                cor *= get_correlation_by_abuv_and_c_permutation(d_aa, d_bb, m_aa, m_bb, cc)
            elif mode == "equal" and i == 0:
                cor *= get_correlation_by_abuv_and_c_aead_equal(d_aa, d_bb, m_aa, m_bb, cc, x1)
            elif mode == "unrestricted" and i == 0:
                cor *= get_correlation_by_abuv_and_c_aead_unrestricted(d_aa, d_bb, m_aa, m_bb, cc, x1)
            else:
                print("Error ! Invalid mode !")
                exit()

    return cor, get_trail_input_list(trail[0])


def get_average_w(characteristic, n, m, mode):
    average_w = 0
    for i in range(total_rounds):
        for j in range(0, state_bits, sbox_bits):
            p1 = 0 * state_words + (j // sbox_bits)
            p2 = 1 * state_words + (j // sbox_bits)
            p3 = 2 * state_words + (j // sbox_bits)
            p4 = 3 * state_words + (j // sbox_bits)
            p5 = 4 * state_words + (j // sbox_bits)
            d_aa = ((((characteristic[route_number * i] >> p1) & 0b1) << 4) |
                    (((characteristic[route_number * i] >> p2) & 0b1) << 3) |
                    (((characteristic[route_number * i] >> p3) & 0b1) << 2) |
                    (((characteristic[route_number * i] >> p4) & 0b1) << 1) |
                    (((characteristic[route_number * i] >> p5) & 0b1) << 0))
            d_bb = ((((characteristic[route_number * i + 1] >> p1) & 0b1) << 4) |
                    (((characteristic[route_number * i + 1] >> p2) & 0b1) << 3) |
                    (((characteristic[route_number * i + 1] >> p3) & 0b1) << 2) |
                    (((characteristic[route_number * i + 1] >> p4) & 0b1) << 1) |
                    (((characteristic[route_number * i + 1] >> p5) & 0b1) << 0))

            x1 = (IV >> (63 - j // sbox_bits)) & 0b1

            ddt = None
            if mode == "permutation" or i > 0:
                ddt = get_ddt_permutation(sbox, n, m)
            elif mode == "equal" and i == 0:
                ddt = get_ddt_aead_equal(sbox, n, m, x1)
            elif mode == "unrestricted" and i == 0:
                ddt = get_ddt_aead_norestricted(sbox, n, m, x1)
            else:
                print("Error ! Invalid mode !")
                exit()
            pab = ddt[d_bb, d_aa]
            if pab == 0:
                print(f"Invalid differential transition: a={d_aa}, b={d_bb}")
                exit()
            wab = int(-numpy.log2(pab))
            average_w += wab

    return average_w
