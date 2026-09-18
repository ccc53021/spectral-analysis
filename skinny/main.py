from helper.skinny import skinny 
import helper.trails as trails
import helper.linear_algebra as linear_algebra
import helper.printing as printing
import helper.inequalities as ineq
import helper.milp as milp
import helper.utils as utils
import helper.inequalities as inequalities
import helper.skinny_cpp as skinny_cpp

import sys
import numpy as np
from gurobipy import GRB, quicksum
import sys
from collections import defaultdict
from itertools import product


def search_quasidifferential_trails(diff_trail):
    nr = diff_trail.nr
    tk = diff_trail.tk

    instance = milp.skinny_milp(nr, tk, diff_trail)
    instance.init()
    for n in range(nr): instance.substitution(n)
    for n in range(nr): instance.add_round_key(n)
    for n in range(nr - 1): instance.linear(n)
    instance.sum_mask()
    instance.mod2_mask()
    instance.set_objective_function()

    solution_set = []
    while True:
        solution_set, continue_flag = instance.run(1)
        if not continue_flag:
            break
        print(f'trail #{len(solution_set)} correlation: {solution_set[-1].objective_value}', flush=True)

        # restrict the next solve to the nullspace of everything found so far
        sol = solution_set[-1]
        found_vector = (utils.flatten(utils.dict_to_array(sol.keys_mask_vars)) +
                         utils.flatten(utils.dict_to_array(sol.state_mask_vars)))
        instance.found_solutions.append(found_vector)
        instance.exclude_span_cut(instance.x_vars, instance.found_solutions)

        if len(solution_set) >= utils.MAX_TRAILS:
            break
    return solution_set, instance

def convert_into_subsets(solution_set, instance):    
    state_width = 4 * 4 * instance.sbox_size  # bits in state_int_mask
    subsets = []
    masks = []

    for sol in solution_set:
        combined = (sol.keys_mask_mod2_int_vars << state_width) + sol.state_mask_int_vars

        overlapping = [index for index, mask in enumerate(masks) if mask and (combined & mask) > 0]
        if not overlapping:
            subsets.append([sol])
            masks.append(combined)
        else:
            primary = overlapping[0]
            merged_mask = combined
            merged_solutions = [sol]
            for index in overlapping:
                merged_mask |= masks[index]
                merged_solutions.extend(subsets[index])
                if index != primary:
                    subsets[index] = []
                    masks[index] = 0
            subsets[primary] = merged_solutions
            masks[primary] = merged_mask
    solution_subsets = {i: group for i, group in enumerate(g for g in subsets if g)}
    return solution_subsets
        
def search_trails_in_space(solution_set, diff_trail):
    sk = skinny(utils.sbox_size)
    dim = len(solution_set)
    A = defaultdict(int)
    for c in range(2**dim):
        bin_c = utils.int2bin(c, dim)
        plaintext_int_mask, key_int_mask, state_int_mask = linear_algebra.get_linear_combination(
            bin_c,
            [s.state_mask_int_vars for s in solution_set],
            [s.keys_mask_mod2_int_vars for s in solution_set],
            [s.state_all_mask_int_vars for s in solution_set]
        )
        state_mask = np.array(
            utils.convert_int_to_array(state_int_mask, (utils.nr, 3, 4, 4, utils.sbox_size)),
            dtype=int
        )
        sign, corr = sk.get_correlation(diff_trail, state_mask)
        if corr == 0: 
            continue
        A[(plaintext_int_mask << (16 * utils.sbox_size * utils.key_size)) + key_int_mask] += sign * 2**(-corr)
    return A

def search_trails_in_space(solution_set, diff_trail):
    nr = utils.NR
    sbox_size = utils.SBOX_SIZE
    key_size = utils.KEY_SIZE
    plane_bytes = 2 * sbox_size  # bytes for one (4,4,sbox_size) plane
 
    state_bytes = [s.state_mask_int_vars.to_bytes(plane_bytes, 'big') for s in solution_set]
    key_bytes = [s.keys_mask_mod2_int_vars.to_bytes(plane_bytes * key_size, 'big') for s in solution_set]
    state_all_bytes = [s.state_all_mask_int_vars.to_bytes(plane_bytes * 3 * nr, 'big') for s in solution_set]
 
    before_sb = [diff_trail.rounds[n].beforeSB for n in range(nr)]
    after_sb = [diff_trail.rounds[n].afterSB for n in range(nr)]
 
    raw = skinny_cpp.search_trails_in_space_cpp(
        state_bytes, key_bytes, state_all_bytes,
        before_sb, after_sb,
        nr, sbox_size, key_size,
        skinny.qdtm,
    )
 
    A = defaultdict(int)
    for key_bytes_out, value in raw:
        A[int.from_bytes(key_bytes_out, 'big')] += value
    return A


def get_distribution(prob_sets):
    adjusted = [prob_sets[0].probability]
    # adjust according to average
    for s in prob_sets[1:]:
        adjusted.append({key - s.average_prob: value for key, value in s.probability.items()})

    # cartesian product, but we add the key, multiply the index
    combined = defaultdict(float)
    for combo in product(*(d.items() for d in adjusted)):
        keys, values = zip(*combo)
        combined_key = sum(keys)
        combined_value = 1
        for v in values:
            combined_value *= v
        combined[combined_key] += combined_value
    return dict(combined)


def main():
    filename = str(sys.argv[1])

    # trail parameters
    utils.set_sbox_size(int(sys.argv[2]))
    utils.set_tk(int(sys.argv[3]))
    utils.set_nr(int(sys.argv[4]))
    utils.set_key_size(int(sys.argv[5]))

    # search parameters
    utils.set_max_correlation(int(sys.argv[6]))
    utils.set_max_trail(int(sys.argv[7]))
    utils.set_max_search_time(int(sys.argv[8]))
    utils.set_thread_count(int(sys.argv[9]))
    skinny.initialize()

    if utils.SBOX_SIZE == 4:
        inequalities.compute_LFSR2S()
        inequalities.compute_LFSR3S()
    elif utils.SBOX_SIZE == 8:
        inequalities.compute_LFSR2L()
        # inequalities.compute_LFSR3L()

    print(f'Processing {filename}',flush=True)
    trail = trails.read_trail(filename)
    solution_set,instance = search_quasidifferential_trails(trail)
    print(f'Found {len(solution_set)} trails',flush=True)
    # optimizing to find potential subsets that don't interact?
    solution_subsets = convert_into_subsets(solution_set, instance)
    print(f'Number of solution subsets: {len(solution_subsets)}',flush=True)
    print(f'Max dimension of the subset: {max([len(v) for _,v in solution_subsets.items()])}')
    # we get one base probability
    probability_sets = []
    for subset_index, solution_subset in solution_subsets.items():
        A = search_trails_in_space(solution_subset,trail)
        c,basis = linear_algebra.gaussian_elimination(A)
        inv_c = {v: k for k,v in c.items()}
        A_new = utils.rehash_dict(A,c)
        A_new_flat = utils.convert_to_flat_array(A_new, len(basis))
        print(f'dimension of subset {subset_index}: {np.log2(len(A_new))}')
        print(f'number of trails in subset {subset_index}: {len(A_new)}')
        prob = utils.fwht(A_new_flat)
        if A[0] == 0:
            average_prob = 0
        else:
            average_prob = -np.log2(A[0])
        probs = defaultdict(int)
        for index,p in enumerate(prob):
            if p > 0:
                probs[-np.log2(p)] += 1
            elif p < 0:
                raise Exception('Negative probability encountered')
        prob_set = utils.ProbabilitySet(average_prob,probs)
        probability_sets.append(prob_set)
        print(prob_set)

    distribution = get_distribution(probability_sets)
    print('The probability distribution is:')
    for key in sorted(distribution):
        print(key, distribution[key])


if __name__ == '__main__':
    # filename = './data/trail_SK_R3.txt'
    # filename = './data/trail_TK2_R5.txt'
    # filename = './data/trail_SK_R7.txt'
    # filename = './data/trail_TK1_R10.txt'
    # filename = './data/trail_TK2_R13.txt'
    # filename = './data/trail_TK3_R15.txt'
    # filename = './data/trail128_SK_R13.txt'
    # filename = './data/trail128_TK1_R14.txt'
    # filename = './data/trail128_TK2_R16.txt'
    main()
    

