from helper.skinny import skinny
import numpy as np
# import helper.utils as utils
### This file includes skinny's differential characteristic
class round_char:
    def __init__(self,beforeSB=None,afterSB=None,afterARK=None,afterSR=None,afterMC=None,keyDiff=None,prob=None):
        self.beforeSB = beforeSB
        self.afterSB = afterSB
        self.afterARK = afterARK
        self.afterSR = afterSR
        self.afterMC = afterMC
        self.keyDiff = keyDiff
        self.prob = prob

class diff_char:
    def __init__(self,nr=None,rounds=None,tk=0, masterKeyDiff=None):
        if tk == 0: self.key_size = 1
        else: self.key_size = tk
        self.tk = tk

        self.nr = nr
        if masterKeyDiff == None:
            self.masterKeyDiff = [[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] for _ in range(self.key_size)]
        if rounds != None:
            self.rounds = [r for r in rounds]
        else:
            self.rounds = []
            for _ in range(nr):
                r = round_char()
                self.rounds.append(r)

def read_trail(filename):
    with open(filename) as f: raw_lines = f.readlines()
    lines = [s for s in (line.strip() for line in raw_lines) if s and not s.startswith('#')] # remove comments
    if len(lines) % 4 != 0:
        raise Exception(f'{filename}: {len(lines)} data lines is not a multiple of 4 (4 lines per round)')
    nr = len(lines) // 4
    # tk is inferred once, from the very first data line, then every line
    # is checked against it (a fresh key_rows is built per round below,
    # since tk itself doesn't change round to round).
    first_token_count = len(lines[0].split())
    tk = first_token_count // 4 - 2
    if first_token_count % 4 != 0 or not (0 <= tk <= 3):
        raise Exception(f'{filename}: first line implies tk={tk} ({first_token_count} numbers), expected 0-3')

    all_before, all_after, all_keydiff = [], [], []

    for n in range(nr):
        before_rows, after_rows = [], []
        key_rows = [[] for _ in range(tk)]

        for r in range(4):
            tokens = [int(x) for x in lines[n * 4 + r].split()]
            if len(tokens) < 8 or len(tokens) % 4 != 0:
                raise ValueError(f'{filename}: round {n} row {r} has {len(tokens)} numbers, '
                                  f'expected a multiple of 4 (>= 8)')

            this_tk = len(tokens) // 4 - 2
            if this_tk != tk:
                raise ValueError(f'{filename}: round {n} row {r} implies tk={this_tk} '
                                  f'but the file starts with tk={tk}')

            before_rows.append(tokens[0:4])
            after_rows.append(tokens[4:8])
            for t in range(tk):
                key_rows[t].append(tokens[(2 + t) * 4:(2 + t) * 4 + 4])

        all_before.append([v for row in before_rows for v in row])
        all_after.append([v for row in after_rows for v in row])
        all_keydiff.append([[v for row in key_rows[t] for v in row] for t in range(tk)])

    list_of_rounds = []
    for n in range(nr):
        r = round_char()
        r.beforeSB = all_before[n]
        r.afterSB = all_after[n]
        r.keyDiff = all_keydiff[n]
        r.afterARK = skinny.computeARK(r.afterSB, r.keyDiff)
        r.afterSR = skinny.computeSR(r.afterARK)
        r.afterMC = skinny.computeMC(r.afterSR)
        r.prob = skinny.compute_round_prob(r.beforeSB, r.afterSB)
        list_of_rounds.append(r)

    masterKeyDiff = [all_keydiff[0][t].copy() for t in range(tk)]
    return diff_char(nr, list_of_rounds, tk, masterKeyDiff)